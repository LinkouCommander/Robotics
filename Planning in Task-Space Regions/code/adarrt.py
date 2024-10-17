#!/usr/bin/env python
# COPIED FROM LAB 3
import argparse
import time

import adapy
import numpy as np
import rospy
import copy


class AdaRRT():
    """
    Rapidly-Exploring Random Trees (RRT) for the ADA controller.
    """
    joint_lower_limits = np.array([-3.14, 1.57, 0.33, -3.14, 0, 0])
    joint_upper_limits = np.array([3.14, 5.00, 5.00, 3.14, 3.14, 3.14])

    class Node():
        """
        A node for a doubly-linked tree structure.
        """
        def __init__(self, state, parent):
            """
            :param state: np.array of a state in the search space.
            :param parent: parent Node object.
            """
            self.state = np.asarray(state)
            self.parent = parent
            self.children = []

        def __iter__(self):
            """
            Breadth-first iterator.
            """
            nodelist = [self]
            while nodelist:
                node = nodelist.pop(0)
                nodelist.extend(node.children)
                yield node

        def __repr__(self):
            return 'Node({})'.format(', '.join(map(str, self.state)))

        def add_child(self, state):
            """
            Adds a new child at the given state.

            :param state: np.array of new child node's state
            :returns: child Node object.
            """
            child = AdaRRT.Node(state=state, parent=self)
            self.children.append(child)
            return child

    def __init__(self,
                 start_state,
                 goal_state,
                 ada,
                 joint_lower_limits=None,
                 joint_upper_limits=None,
                 ada_collision_constraint=None,
                 step_size=0.25,
                 goal_precision=0.2,
                 max_iter=10000):
        """
        :param start_state: Array representing the starting state.
        :param goal_state: Array representing the goal state.
        :param ada: libADA instance.
        :param joint_lower_limits: List of lower bounds of each joint.
        :param joint_upper_limits: List of upper bounds of each joint.
        :param ada_collision_constraint: Collision constraint object.
        :param step_size: Distance between nodes in the RRT.
        :param goal_precision: Maximum distance between RRT and goal before
            declaring completion.
        :param sample_near_goal_prob:
        :param sample_near_goal_range:
        :param max_iter: Maximum number of iterations to run the RRT before
            failure.
        """
        self.start = AdaRRT.Node(start_state, None)
        self.goal = AdaRRT.Node(goal_state, None)
        self.ada = ada
        self.joint_lower_limits = joint_lower_limits or AdaRRT.joint_lower_limits
        self.joint_upper_limits = joint_upper_limits or AdaRRT.joint_upper_limits
        self.ada_collision_constraint = ada_collision_constraint
        self.step_size = step_size
        self.goal_precision = goal_precision
        self.max_iter = max_iter

    def build(self):
        """
        Build an RRT.

        In each step of the RRT:
            1. Sample a random point.
            2. Find its nearest neighbor.
            3. Attempt to create a new node in the direction of sample from its
                nearest neighbor.
            4. If we have created a new node, check for completion.

        Once the RRT is complete, add the goal node to the RRT and build a path
        from start to goal.

        :returns: A list of states that create a path from start to
            goal on success. On failure, returns None.
        """
        for k in range(self.max_iter):
            # FILL in your code here
            # sample = self._get_random_sample()
            number = np.random.rand()
            if number <= 0.2:
                sample = self._get_random_sample_near_goal()
            else:
                sample = self._get_random_sample()
            neighbor = self._get_nearest_neighbor(sample)
            new_node = self._extend_sample(sample, neighbor)
            # print("Sample: " + str(sample))
            # print("NearestNeighbor: " + str(neighbor))
            # print("NewNode: " + str(new_node))

            if new_node and self._check_for_completion(new_node):
                # FILL in your code here
                self.goal = new_node.add_child(self.goal.state)#add goal node to the RRT
                path = self._trace_path_from_start()
                print("Path: " + str(path))
                return path

        print("Failed to find path from {0} to {1} after {2} iterations!".format(
            self.start.state, self.goal.state, self.max_iter))
        return None

    def _get_random_sample(self):
        """
        Uniformly samples the search space.

        :returns: A vector representing a randomly sampled point in the search
            space.
        """
        # FILL in your code here
        sample = copy.copy(self.joint_lower_limits)

        for i in range(len(self.joint_lower_limits)):
            limitation = np.arange(self.joint_lower_limits[i], self.joint_upper_limits[i], 0.01)
            sample[i] = np.random.choice(limitation)
        dist = np.linalg.norm(sample - self.start.state)
        # print("Norm From Start to Sample: " + str(dist))

        return sample

    def _get_random_sample_near_goal(self):
        """
        Uniformly samples the search space.

        :returns: A vector representing a randomly sampled point in the search
            space.
        """
        # FILL in your code here
        sample = copy.copy(self.joint_lower_limits)

        for i in range(len(self.joint_lower_limits)):
            limitation = np.arange(self.goal.state[i] - 0.05, self.goal.state[i] + 0.05, 0.01)
            sample[i] = np.random.choice(limitation)

        return sample

    def _get_nearest_neighbor(self, sample):
        """
        Finds the closest node to the given sample in the search space,
        excluding the goal node.

        :param sample: The target point to find the closest neighbor to.
        :returns: A Node object for the closest neighbor.
        """
        # FILL in your code here
        # all nodes in the tree
        nodes = list(self.start)
        
        # # distance between each node and sample
        # dist = [np.linalg.norm(nodes[i].state - sample.state) for i in range(len(node))]

        # # index of nearest neighbour
        # nearest_neighbour_indx = np.argsort(dist)[0]

        # # nearest neighbour node in tree
        # nearest_neighbour = nodes[nearest_neighbour_indx]

        nearest_neighbour = None
        min_dist = float('inf')

        for node in nodes:
            dist = np.linalg.norm(node.state - sample)
            if dist < min_dist:
                nearest_neighbour = node
                min_dist = dist
                # print(min_dist)
        
        return nearest_neighbour

    def _extend_sample(self, sample, neighbor):
        """
        Adds a new node to the RRT between neighbor and sample, at a distance
        step_size away from neighbor. The new node is only created if it will
        not collide with any of the collision objects (see
        RRT._check_for_collision)

        :param sample: target point
        :param neighbor: closest existing node to sample
        :returns: The new Node object. On failure (collision), returns None.
        """
        # FILL in your code here
        direction = sample - neighbor.state
        distance = np.linalg.norm(direction)


        if distance <= self.step_size:
            new_state = sample
        else:
            direction_unit = direction / distance
            new_state = neighbor.state + direction_unit * self.step_size

        print("Distance: " + str(distance))
        if not self._check_for_collision(new_state):
            new_node = neighbor.add_child(new_state)
            return new_node
        else:
            return None

    def _check_for_completion(self, node):
        """
        Check whether node is within self.goal_precision distance of the goal.

        :param node: The target Node
        :returns: Boolean indicating node is close enough for completion.
        """
        # FILL in your code here
        length = np.linalg.norm(self.goal.state - node.state)
        print("Length: " + str(length))
        if length <= self.goal_precision: 
            return True
        return False

    def _trace_path_from_start(self, node=None):
        """
        Traces a path from start to node, if provided, or the goal otherwise.

        :param node: The target Node at the end of the path. Defaults to
            self.goal
        :returns: A list of states (not Nodes!) beginning at the start state and
            ending at the goal state.
        """
        # FILL in your code here
        path = []
        if node == None:
            node = self.goal
        i = 0
        while node != None:
            path.append(node.state)
            node = node.parent
            i = i + 1
            print("Path before return: "  + str(path) + "  index:  "+ str(i))

        return path[::-1]

    def _check_for_collision(self, sample):
        """
        Checks if a sample point is in collision with any collision object.

        :returns: A boolean value indicating that sample is in collision.
        """
        if self.ada_collision_constraint is None:
            return False
        return self.ada_collision_constraint.is_satisfied(
            self.ada.get_arm_state_space(),
            self.ada.get_arm_skeleton(), sample)


def main(is_sim):
    
    if not is_sim:
        from moveit_ros_planning_interface._moveit_roscpp_initializer import roscpp_init
        roscpp_init('adarrt', [])

    # instantiate an ada
    ada = adapy.Ada(is_sim)

    armHome = [-1.5, 3.22, 1.23, -2.19, 1.8, 1.2]
    goalConfig = [-1.72, 4.44, 2.02, -2.04, 2.66, 1.39]
    delta = 0.25
    eps = 1.0

    if is_sim:
        ada.set_positions(goalConfig)
    else:
        raw_input("Please move arm to home position with the joystick. " +
            "Press ENTER to continue...")


    # launch viewer
    viewer = ada.start_viewer("dart_markers/simple_trajectories", "map")

    # add objects to world
    canURDFUri = "package://pr_assets/data/objects/can.urdf"
    sodaCanPose = [0.25, -0.35, 0.0, 0, 0, 0, 1]
    tableURDFUri = "package://pr_assets/data/furniture/uw_demo_table.urdf"
    tablePose = [0.3, 0.0, -0.7, 0.707107, 0, 0, 0.707107]
    world = ada.get_world()
    can = world.add_body_from_urdf(canURDFUri, sodaCanPose)
    table = world.add_body_from_urdf(tableURDFUri, tablePose)

    # add collision constraints
    collision_free_constraint = ada.set_up_collision_detection(
            ada.get_arm_state_space(),
            ada.get_arm_skeleton(),
            [can, table])
    full_collision_constraint = ada.get_full_collision_constraint(
            ada.get_arm_state_space(),
            ada.get_arm_skeleton(),
            collision_free_constraint)

    # easy goal
    adaRRT = AdaRRT(
        start_state=np.array(armHome),
        goal_state=np.array(goalConfig),
        ada=ada,
        ada_collision_constraint=full_collision_constraint,
        step_size=delta,
        goal_precision=eps)

    rospy.sleep(1.0)

    if not is_sim:
        ada.start_trajectory_executor()

    path = adaRRT.build()
    if path is not None:
        print("Path waypoints:")
        print(np.asarray(path))
        waypoints = []
        for i, waypoint in enumerate(path):
            waypoints.append((0.0 + i, waypoint))

        t0 = time.clock()
        # traj = ada.compute_joint_space_path(
        #     ada.get_arm_state_space(), waypoints)
        traj = ada.compute_smooth_joint_space_path(
            ada.get_arm_state_space(), waypoints)
        t = time.clock() - t0
        print(str(t) + "seconds elapsed")
        raw_input('Press ENTER to execute trajectory and exit')
        ada.execute_trajectory(traj)
        rospy.sleep(1.0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', dest='is_sim', action='store_true')
    parser.add_argument('--real', dest='is_sim', action='store_false')
    parser.set_defaults(is_sim=True)
    args = parser.parse_args()
    main(args.is_sim)
