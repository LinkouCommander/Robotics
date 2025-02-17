#!/usr/bin/env python3

# Developed for USC 545 Intro To Robotics.

import rospy
import rosbag
import tf
import time
import math

import geometry_msgs.msg
import nav_msgs.msg
import sensor_msgs.msg
import std_msgs.msg

import argparse
import matplotlib.pyplot as plt
import numba
import numpy as np
import skimage.draw
import sys
import copy

dtype = np.float64

LINEAR_MODEL_VAR_X = 0.5
LINEAR_MODEL_VAR_Y = 0.5
ANGULAR_MODEL_VAR = 0.3
SENSOR_MODEL_VAR = 15
NUM_PARTICLES = 2000


@numba.jit(nopython=True)
def _CastRay(p0, p1, grid_data):
    '''
    Cast a ray from p0 in the direction toward p1 and return the first cell the
    sensor will detect.
    '''
    delta = np.abs(np.array([p1[0] - p0[0], p1[1] - p0[1]]))
    p = p0.copy()
    n = 1 + np.sum(delta)
    inc = np.array([1 if p1[0] > p0[0] else -1, 1 if p1[1] > p0[1] else -1])
    error = delta[0] - delta[1]
    delta *= 2

    last_p = p.copy()
    for i in range(n):
        if not (0 <= p[0] < grid_data.shape[1]
                and 0 <= p[1] < grid_data.shape[0]):
            return last_p
        last_p = p.copy()

        if grid_data[p[1], p[0]] != 0:  # coord flip for handedness
            return p

        if error > 0:
            p[0] += inc[0]
            error -= delta[1]
        else:
            p[1] += inc[1]
            error += delta[0]

    return p1


@numba.jit(nopython=True)
def _ComputeSimulatedRanges(scan_angles, scan_range_max, world_t_map,
                            map_t_particle, map_R_particle, grid_data,
                            grid_resolution):
    '''
    Cast rays in all directions and calculate distances to obstacles in all
    directions from a given particle.
    '''
    src_cell = ((map_t_particle - world_t_map) / grid_resolution).astype(
        np.int32)
    # if ((map_t_particle - world_t_map) == 0).all():
    #     print("sub is zero")

    # Compute max possible cell indices
    angles = scan_angles + map_R_particle
    max_points = map_t_particle + scan_range_max * np.stack(
        (np.cos(angles), np.sin(angles)), axis=-1)
    max_cells = ((max_points - world_t_map) / grid_resolution).astype(np.int32)

    simulated_cell_ranges_sq = np.zeros_like(scan_angles)
    for i in range(len(max_cells)):
        hit_cell = _CastRay(src_cell, max_cells[i], grid_data)
        simulated_cell_ranges_sq[i] = np.sum(np.square(hit_cell - src_cell))

    return grid_resolution * np.sqrt(simulated_cell_ranges_sq)


@numba.jit(nopython=True)
def RotateBy(x, angle):
    c, s = np.cos(angle), np.sin(angle)
    m = np.zeros((2, 2), dtype=dtype)
    m[0, 0] = c
    m[0, 1] = -s
    m[1, 0] = s
    m[1, 1] = c
    return np.dot(m, x)


class Pose(object):
    '''
    2D pose representation of robot w.r.t. world frame.
    Given by the translation from origin to the robot, and then rotation to math
    robot orientation.
    '''
    def __init__(self, rotation=0, translation=[0, 0]):
        self.rotation = rotation
        self.translation = np.array(translation, dtype=dtype)

    @staticmethod
    def FromGeometryMsg(pose_msg):
        q = pose_msg.orientation
        tx = 1 - 2 * (q.y * q.y + q.z * q.z)
        ty = 2 * (q.x * q.y + q.z * q.w)
        yaw = np.arctan2(ty, tx)
        x, y = pose_msg.position.x, pose_msg.position.y
        return Pose(yaw, [x, y])

    def ToGeometryMsg(self):
        msg = geometry_msgs.msg.Pose()
        msg.position.x = self.translation[0]
        msg.position.y = self.translation[1]
        msg.orientation.z = np.sin(self.rotation / 2)
        msg.orientation.w = np.cos(self.rotation / 2)
        return msg


class Grid(object):
    '''
    Occupancy grid representation, with coordinate helpers.
    World is global coordinate frame and grid is index-based array coordinates.
    '''
    def __init__(self, grid_msg):
        self.cols = grid_msg.info.width
        self.rows = grid_msg.info.height
        self.resolution = grid_msg.info.resolution
        self.map_t_extents = [
            self.cols * self.resolution,
            self.rows * self.resolution,
        ]
        self.world_T_map = Pose.FromGeometryMsg(grid_msg.info.origin)
        self.data = np.asarray(grid_msg.data,
                               dtype=np.int8).reshape(self.rows, self.cols)

    def GridToWorld(self, cell_index):
        return self.resolution * cell_index + self.world_T_map.translation

    def WorldToGrid(self, point):
        return ((point - self.world_T_map.translation) /
                self.resolution).astype(np.int32)

    def GetWorldCoords(self, point):
        x, y = self.WorldToGrid(point)
        return self.data[y, x]

    def ToNavMsg(self):
        msg = nav_msgs.msg.OccupancyGrid()
        msg.info.resolution = self.resolution
        msg.info.width = self.cols
        msg.info.height = self.rows
        msg.info.origin = self.world_T_map.ToGeometryMsg()
        msg.data = self.data.flatten().tolist()
        return msg


class Scan(object):
    '''
    Scan representation / cache.
    Scan consists of angles and distances (ranges).
    '''
    def __init__(self, scan_msg):
        angle_min = scan_msg.angle_min
        angle_max = scan_msg.angle_max
        self.range_max = scan_msg.range_max

        self.angles = np.linspace(angle_min,
                                  angle_max,
                                  num=len(scan_msg.ranges))

        self.ranges = np.array(scan_msg.ranges, dtype=dtype)


class Particle(object):
    '''Individual particle representation.'''
    def __init__(self, grid, map_T_particle=None, weight=1):
        if map_T_particle is None:
            ll = grid.world_T_map.translation
            ul = ll + grid.map_t_extents

            found = False
            while not found:
                translation = np.random.uniform(ll, ul)
                if grid.GetWorldCoords(translation) == 0:
                    break

            map_T_particle = Pose(rotation=np.random.uniform(-np.pi, np.pi),
                                  translation=translation)

        self.grid = grid
        self.map_T_particle = map_T_particle
        self.last_odom_timestamp = None
        self.weight = weight

    def UpdateOdom(self, odom_msg):
        '''Propagate this particle according to the sensor data and motion
        model.'''
        if self.last_odom_timestamp is not None:
            dt = (odom_msg.header.stamp - self.last_odom_timestamp).to_sec()
        else:
            dt = 0
        self.last_odom_timestamp = odom_msg.header.stamp

        ##########
        #
        #  YOUR CODE HERE (Odometry Section)
        #
        #  1. Extract the particle's velocity in its local frame
        #     (odom_msg.twist.twist.linear.{x, y})
        velx = odom_msg.twist.twist.linear.x
        vely = odom_msg.twist.twist.linear.y
        #   print(str(velx) + "," + str(vely))
        #  2. Add noise to the velocity, with component variance
        #     LINEAR_MODEL_VAR_X and LINEAR_MODEL_VAR_Y
        velx = np.random.normal(velx, LINEAR_MODEL_VAR_X, 1)[0]
        vely = np.random.normal(vely, LINEAR_MODEL_VAR_Y, 1)[0]
        vel_local = np.array([velx,vely])
        #  3. Transform the linear velocity into map frame (use the provided
        #     RotateBy() function). The current pose of the particle, in map
        #     frame, is stored in self.map_T_particle
        vel_map = RotateBy(vel_local,self.map_T_particle.rotation)
        #  4. Integrate the linear velocity to the particle pose, stored in
        #     self.map_T_particle.translation
        self.map_T_particle.translation = self.map_T_particle.translation + vel_map*dt
        #  5. Extract the particle's rotational velocity
        #     (odom_msg.twist.twist.angular.z)
        velrot = odom_msg.twist.twist.angular.z
        #  6. Add noise to the rotational velocity, with variance
        #     ANGULAR_MODEL_VAR
        velrot = np.random.normal(velrot, ANGULAR_MODEL_VAR, 1)[0]
        #  7. Integrate the rotational velocity into the particle pose, stored
        #     in self.map_T_particle.rotation
        self.map_T_particle.rotation = self.map_T_particle.rotation + velrot*dt
        #
        ##########

    """
       ### EXTRA CREDIT 1 Attempt ###
       # odom previous(t-1) values: x_hat, y_hat, theta_hat
       # odom current(t) values: x_hat_prime, y_hat_prime, theta_hat_prime

       # there should be these lines in intialization:
       # self.previous_odom_x = None
       # self.previous_odom_y = None
       # self.previous_odom_theta = None
        linear_velocity_x = odom_msg.twist.twist.linear.x
        linear_velocity_y = odom_msg.twist.twist.linear.y
        angular_velocity_theta = odom_msg.twist.twist.angular.z


       # odom previous(t-1) values: x_hat, y_hat, theta_hat
       # odom current(t) values: x_hat_prime, y_hat_prime, theta_hat_prime
       # given dt:
       # y_hat_prime - y_hat = linear_velocity_y * dt
       # x_hat_prime - x_hat = linear_velocity_x * dt
       # theta_hat_prime - theta_hat = angular_velocity_theta * dt
       # so theta_hat = theta_hat_prime - angular_velocity_theta * dt
       # or theta_hat is the previous state angular
        theta_hat = self.map_T_particle.rotation


       # now calculate delta_rot1, delta_rot2, delta_trans
       # - Algorithm lines 2 to 4
        delta_rot1 = math.atan2(linear_velocity_y * dt, linear_velocity_x * dt) - theta_hat
        delta_rot2 = angular_velocity_theta * dt - delta_rot1
        delta_trans = np.sqrt((linear_velocity_x * dt)**2 + (linear_velocity_y * dt)**2)

        alpha1 = ANGULAR_MODEL_VAR
        alpha2= ANGULAR_MODEL_VAR
        alpha3= LINEAR_MODEL_VAR_X + LINEAR_MODEL_VAR_Y
        alpha4 = 0.1 * alpha3

       # calculate delta_rot1_hat, delta_rot2_hat, delta_trans_hat
       # (already initialized 4 params alpha1, alpha2, alpha3, alpha4)
       # - Algorithm lines 5 to 7
        delta_rot1_hat = delta_rot1 - np.random.normal(0, alpha1 * delta_rot1**2 + alpha2 * delta_trans**2)
        delta_rot2_hat = delta_rot2 - np.random.normal(0, alpha1 * delta_rot2**2 + alpha2 * delta_trans**2)
        delta_trans_hat = delta_trans - np.random.normal(0, alpha3 * delta_trans**2 + alpha4 * (delta_rot1**2 + delta_rot2**2))


       # get previous state X(t-1): x, y, theta
        x = self.map_T_particle.translation[0]
        y = self.map_T_particle.translation[1]
        theta = self.map_T_particle.rotation


       # calculate current state X(t): x_prime, y_prime, theta_prime
       # - Algorithm lines 8 to 10    
       # RotateBy(delta_trans_hat * math.cos(theta + delta_rot1_hat), self.map_T_particle.rotation)
        update_x = delta_trans_hat * math.cos(theta + delta_rot1_hat)
       # RotateBy(delta_trans_hat * math.sin(theta + delta_rot1_hat), self.map_T_particle.rotation)
        update_y = delta_trans_hat * math.sin(theta + delta_rot1_hat)
        # update_x_and_y = RotateBy(np.array([update_x, update_y]), self.map_T_particle.rotation)
        # x_prime = x + update_x_and_y[0]
        # y_prime = y + update_x_and_y[1]
        x_prime = x + update_x
        y_prime = y + update_y
        theta_prime = theta + delta_rot1_hat + delta_rot2_hat


       # - Algorithm lines 11
        self.map_T_particle.translation[0] = x_prime
        self.map_T_particle.translation[1] = y_prime
        self.map_T_particle.rotation = theta_prime
    """

    def _ComputeSimulatedRanges(self, scan):
        translation = self.map_T_particle.translation
        rotation = self.map_T_particle.rotation
        return _ComputeSimulatedRanges(scan.angles, scan.range_max,
                                       self.grid.world_T_map.translation,
                                       translation, rotation, self.grid.data,
                                       self.grid.resolution)

    def UpdateScan(self, scan):
        start = time.time()
        '''
        Calculate weights of particles according to scan / map matching.
        '''
        sim_ranges = self._ComputeSimulatedRanges(scan)
       
        ##########
        #
        #  YOUR CODE HERE (LIDAR Section)
        #
        #  1. Compute the likelihood of each beam, compared to sim_ranges. The
        #     true measurement vector is in scan.ranges. Use the Gaussian PDF
        #     formulation.
        #  2. Assign the weight of this particle as the product of these
        #     probabilities. Store this value in self.weight.
        #
        
        ####### EXTRA CREDIT PART 2 ############
        # weights = 1
        # for i in range(len(scan.ranges)):
        #     norm = (1 / (np.sqrt(2 * np.pi* SENSOR_MODEL_VAR) )) * np.exp(-0.5 * ((scan.ranges[i] - sim_ranges[i])** 2 / SENSOR_MODEL_VAR) )
        #     if scan.ranges[i] >= scan.range_max:
        #         maxi = 1
        #         rand = 0
        #     else:
        #         maxi = 0
        #         rand = 1 / scan.range_max

        #     p = norm * 0.9 + maxi * 0.05 + rand * 0.05
        #     weights = weights *  p


        ######DEFAULT - NOT EXTRA CREDIT######
        weights = 1
        for i in range(len(scan.ranges)):
            p = (1 / (np.sqrt(2 * np.pi) * SENSOR_MODEL_VAR)) * np.exp(-0.5 * ((scan.ranges[i] - sim_ranges[i]) / SENSOR_MODEL_VAR) ** 2)
            weights = weights *  p

        ##Modify weights to prevent being 0, which crashes code later
        if weights == 0:
            weights = 10**(-300)
        else:
            weights = weights * 10**300
        
        
        self.weight = weights
        


class ParticleFilter(object):
    def __init__(self, grid, num_particles):
        self.grid = grid
        self.last_timestamp = None
        self.particles = [Particle(grid) for _ in range(num_particles)]
        self.pose_publisher = rospy.Publisher("pose_hypotheses",
                                              geometry_msgs.msg.PoseArray,
                                              queue_size=10)
        self.scan_publisher = rospy.Publisher("lidar",
                                              sensor_msgs.msg.LaserScan,
                                              queue_size=10)
        self.tf_broadcaster = tf.TransformBroadcaster()

    def GetMeanPose(self):
        sum_translation = np.zeros(2, dtype=dtype)
        sum_rotation = 0
        for particle in self.particles:
            sum_translation += particle.map_T_particle.translation
            sum_rotation += particle.map_T_particle.rotation

        avg_translation = sum_translation / len(self.particles)
        avg_rotation = sum_rotation / len(self.particles)

        return Pose(avg_rotation, avg_translation)

    def GetPoseArray(self):
        msg = geometry_msgs.msg.PoseArray()
        msg.poses = [
            particle.map_T_particle.ToGeometryMsg()
            for particle in self.particles
        ]
        msg.header.stamp = self.last_timestamp
        msg.header.frame_id = "map"

        return msg

    def UpdateOdom(self, odom_msg):
        if self.last_timestamp is None:
            self.last_timestamp = odom_msg.header.stamp

        for particle in self.particles:
            particle.UpdateOdom(odom_msg)

    def UpdateScan(self, scan_msg):
        if self.last_timestamp is None:
            self.last_timestamp = scan_msg.header.stamp

        scan = Scan(scan_msg)

        # Update weights for particles.
        for particle in self.particles:
            particle.UpdateScan(scan)

        ##########
        #
        #  YOUR CODE HERE (Importance Sampling Section)
        #
        #  1. Each particle has a weight, i.e. self.particles[i].weight
        prob = []
        for i in range(NUM_PARTICLES):
            #  2. Use these weights to build a normalized discrete probability
            #     distribution for sampling each particle
            prob.append(self.particles[i].weight)
        
        prob = prob / np.sum(prob)
        
        #  3. Use np.random.choice to choose the indices of the particles that
        #     we want to propagate to the next step (use the p= argument to pass
        #     your probability distribution)
        
        new_indices = np.random.choice(NUM_PARTICLES, NUM_PARTICLES, p=prob)
        
        #  4. Use copy.deepcopy to copy the proper particles into a new population
        #     of particles.
        new_particles = []
        for i in range(NUM_PARTICLES):
            new_particles.append(copy.deepcopy(self.particles[new_indices[i]]))
        #  5. Set self.particles to the new population of particles.
        for i in range(NUM_PARTICLES):
            self.particles[i] = new_particles[i]
        ##########
        

        # Publish cloud for visualization.
        self.pose_publisher.publish(self.GetPoseArray())
        avg_pose = self.GetMeanPose()
        avg_trans = (avg_pose.translation[0], avg_pose.translation[1], 0)
        avg_quat = tf.transformations.quaternion_from_euler(
            0, 0, avg_pose.rotation)
        self.tf_broadcaster.sendTransform(avg_trans, avg_quat,
                                          scan_msg.header.stamp, "robot",
                                          "map")
        self.scan_publisher.publish(scan_msg)


def main():
    # rospy initialization.
    rospy.init_node('usc545mcl')
    args = rospy.myargv(argv=sys.argv)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num_particles', type=int, default=NUM_PARTICLES)
    argparser.add_argument('bag_filename', type=str)
    args = argparser.parse_args(args[1:])

    # Wait for map server to start up.
    print("Waiting for map_server...")
    grid_msg = rospy.wait_for_message("/map", nav_msgs.msg.OccupancyGrid)
    print("Map received.")

    # Construct particle filter for the received map.
    grid = Grid(grid_msg)
    mcl = ParticleFilter(grid, args.num_particles)

    # "Subscribe" to message channels from bag.
    # You can see the relevant channels by running
    # $ rosbag info /path/to/bag
    bag = rosbag.Bag(args.bag_filename)
    subscribers = {"lidar": mcl.UpdateScan, "odom": mcl.UpdateOdom}
    x_err = []
    y_err = []
    yaw_err = []
    # for topic, msg, t in bag.read_messages(topics=["gt_odom"] +
    #                                        list(subscribers.keys())):
    for topic, msg, t in bag.read_messages():
        if topic == "gt_odom":
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            yaw = 2 * np.arcsin(msg.pose.pose.orientation.z)

            mcl_pose = mcl.GetMeanPose()
            x_err.append((x - mcl_pose.translation[0])**2)
            y_err.append((y - mcl_pose.translation[1])**2)
            yaw_err.append((yaw - mcl_pose.rotation)**2)

        else:
            subscribers[topic](msg)

    plt.plot(x_err, label="x_mse")
    plt.plot(y_err, label="y_mse")
    plt.plot(yaw_err, label="yaw_mse")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()