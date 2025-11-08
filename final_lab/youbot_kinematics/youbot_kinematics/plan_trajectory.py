import rclpy
from rclpy.node import Node
from scipy.linalg import expm
from scipy.linalg import logm
from itertools import permutations
import time
import threading
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker
from rclpy.duration import Duration
from scipy.spatial.transform import Rotation as R, Slerp


import numpy as np
from youbot_kinematics.youbotKineStudent import YoubotKinematicStudent
from youbot_kinematics.target_data import TARGET_JOINT_POSITIONS


class YoubotTrajectoryPlanning(Node):
    def __init__(self):
        # Initialize node
        super().__init__("youbot_trajectory_planner")

        # Save question number for check in main run method
        self.kdl_youbot = YoubotKinematicStudent()

        # Create trajectory publisher and a checkpoint publisher to visualize checkpoints
        # 5 means depth of cached messages about trajectory
        self.traj_pub = self.create_publisher(
            JointTrajectory, "/youbot_arm_controller/joint_trajectory", 5
        )
        self.checkpoint_pub = self.create_publisher(Marker, "checkpoint_positions", 100)

    def run(self):
        """This function is the main run function of the class. When called, it runs question 6 by calling the q6()
        function to get the trajectory. Then, the message is filled out and published to the /command topic.
        """
        print("run q6a")
        self.get_logger().info("Waiting 5 seconds for everything to load up.")
        time.sleep(2.0)
        traj = self.q6()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = [
            "arm_joint_1",
            "arm_joint_2",
            "arm_joint_3",
            "arm_joint_4",
            "arm_joint_5",
        ]
        self.traj_pub.publish(traj)

    def q6(self):
        """This is the main q6 function. Here, other methods are called to create the shortest path required for this
        question. Below, a general step-by-step is given as to how to solve the problem.
        Returns:
            traj (JointTrajectory): A list of JointTrajectory points giving the robot joint positions to achieve in a
            given time period.
        """
        # TODO: implement this
        # Steps to solving Q6.
        # 1. Load in targets from the bagfile (checkpoint data and target joint positions).
        # 2. Compute the shortest path achievable visiting each checkpoint Cartesian position.
        # 3. Determine intermediate checkpoints to achieve a linear path between each checkpoint and have a full list of
        #    checkpoints the robot must achieve. You can publish them to see if they look correct. Look at slides 39 in lecture 7
        # 4. Convert all the checkpoints into joint values using an inverse kinematics solver.
        # 5. Create a JointTrajectory message.

        # Your code starts here ------------------------------
        # Step 1: Load in targets from the bagfile
        target_checkpoint_tfs, target_joint_positions = self.load_targets()
        # Step 2: Compute the shortest path achievable visiting each checkpoint Cartesian position.
        sorted_checkpoint_idx, min_dist = self.get_shortest_path(target_checkpoint_tfs)
        self.get_logger().info(f"Sorted checkpoint indices: {sorted_checkpoint_idx}")
        self.get_logger().info(f"Minimum distance to travel: {min_dist}")
        # Step 3: Determine intermediate checkpoints to achieve a linear path between each checkpoint
        num_intermediate_points = 10  # Number of intermediate points between each checkpoint
        full_checkpoint_tfs = self.intermediate_tfs(
            sorted_checkpoint_idx, target_checkpoint_tfs, num_intermediate_points
        )
        self.get_logger().info(f"full_checkpoint_tfs.shape: {full_checkpoint_tfs.shape}")
        self.publish_traj_tfs(full_checkpoint_tfs)
        # Step 4: Convert all the checkpoints into joint values using an inverse kinematics solver
        init_joint_position = target_joint_positions[:, 0]
        q_checkpoints = self.full_checkpoints_to_joints(
            full_checkpoint_tfs, init_joint_position
        )
        # Step 5: Create a JointTrajectory message
        traj = JointTrajectory()
        time_from_start = 0.0
        time_step = 0.1  # Time step between each trajectory point

        # q_checkpoints is shape (5, num_points) where each column is a joint vector
        num_points = q_checkpoints.shape[1]
        for i in range(num_points):
            point = JointTrajectoryPoint()
            point.positions = q_checkpoints[:, i].tolist()
            # Use rclpy Duration helper to build a proper Duration message
            point.time_from_start = Duration(seconds=time_from_start).to_msg()
            traj.points.append(point)
            time_from_start += time_step

        # Your code ends here ------------------------------

        assert isinstance(traj, JointTrajectory)
        return traj

    def load_targets(self):
        """This function loads the checkpoint data from the TARGET_JOINT_POSITIONS variable. In this variable you will find each
        row has target joint positions. You need to use forward kinematics to get the goal end-effector position.
        Returns:
            target_cart_tf (4x4x5 np.ndarray): The target 4x4 homogenous transformations of the checkpoints found in the
            bag file. There are a total of 5 transforms (4 checkpoints + 1 initial starting cartesian position).
            target_joint_positions (5x5 np.ndarray): The target joint values for the 4 checkpoints + 1 initial starting
            position.
        """
        num_target_positions = len(TARGET_JOINT_POSITIONS)
        self.get_logger().info(f"{num_target_positions} target positions")
        # Initialize arrays for checkpoint transformations and joint positions
        target_joint_positions = np.zeros((5, num_target_positions + 1))
        # Create a 4x4 transformation matrix, then stack 6 of these matrices together for each checkpoint
        target_cart_tf = np.repeat(
            np.identity(4), num_target_positions + 1, axis=1
        ).reshape((4, 4, num_target_positions + 1))

        # Get the current starting position of the robot
        target_joint_positions[:, 0] = self.kdl_youbot.current_joint_position
        # Initialize the first checkpoint as the current end effector position
        target_cart_tf[:, :, 0] = self.kdl_youbot.forward_kinematics(
            target_joint_positions[:, 0].tolist()
        )

        # TODO: populate the transforms in the target_cart_tf object
        # populate the joint positions in the target_joint_positions object
        # Your code starts here ------------------------------
        for i in range(num_target_positions):
            joint_pos = TARGET_JOINT_POSITIONS[i]
            target_joint_positions[:, i + 1] = joint_pos
            target_cart_tf[:, :, i + 1] = self.kdl_youbot.forward_kinematics(joint_pos.tolist())

        # Your code ends here ------------------------------

        self.get_logger().info(f"{target_cart_tf.shape} target poses")
        assert isinstance(target_cart_tf, np.ndarray)
        assert target_cart_tf.shape == (4, 4, num_target_positions + 1)
        assert isinstance(target_joint_positions, np.ndarray)
        assert target_joint_positions.shape == (5, num_target_positions + 1)

        return target_cart_tf, target_joint_positions

    def get_shortest_path(self, checkpoints_tf):
        """This function takes the checkpoint transformations and computes the order of checkpoints that results
        in the shortest overall path.
        the objective is to find the order visiting all checkpoints from start point that results in the minimum travel distance.
        Args:
            checkpoints_tf (np.ndarray): The target checkpoints transformations as a 4x4x5 numpy ndarray. just candidates
        Returns:
            sorted_order (np.array): An array of size 5 indicating the order of checkpoint
            min_dist:  (float): The associated distance to the sorted order giving the total estimate for travel
            distance.
        """
        num_checkpoints = checkpoints_tf.shape[2]
        # TODO: implement this method. Make it flexible to accomodate different numbers of targets.
        # the purpose is to find the order visiting all checkpoints from start point that results in the minimum travel distance.
        # Your code starts here ------------------------------
        checkpoint_indices = list(range(1, num_checkpoints))  # Exclude the starting point (index 0)
        min_dist = float('inf')
        sorted_order = None # To store the best order found so far
        # Generate all permutations of the checkpoint indices
        # emuneration method
        for perm in permutations(checkpoint_indices):
            current_order = [0] + list(perm)  # Start from the initial position (index 0)
            total_dist = 0.0
            for i in range(len(current_order) - 1):
                idx_a = current_order[i]
                idx_b = current_order[i + 1]
                pos_a = checkpoints_tf[0:3, 3, idx_a]
                pos_b = checkpoints_tf[0:3, 3, idx_b]
                dist = np.linalg.norm(pos_b - pos_a)
                total_dist += dist
            if total_dist < min_dist:
                min_dist = total_dist
                sorted_order = np.array(current_order)
       
        # Your code ends here ------------------------------

        assert isinstance(sorted_order, np.ndarray)
        assert sorted_order.shape == (num_checkpoints,)
        assert isinstance(min_dist, float)

        return sorted_order, min_dist

    def publish_traj_tfs(self, tfs):
        """This function gets a np.ndarray of transforms and publishes them in a color coded fashion to show how the
        Cartesian path of the robot end-effector.
        Args:
            tfs (np.ndarray): A array of 4x4xn homogenous transformations specifying the end-effector trajectory.
        """
        id = 0
        for i in range(0, tfs.shape[2]):
            marker = Marker()
            marker.id = id
            id += 1
            marker.header.frame_id = "base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0 #transparency
            marker.color.r = 0.0
            marker.color.g = 0.0 + id * 0.05
            marker.color.b = 1.0 - id * 0.05
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = tfs[0, -1, i]
            marker.pose.position.y = tfs[1, -1, i]
            marker.pose.position.z = tfs[2, -1, i]
            self.get_logger().info(str(tfs[0:3, -1, i]))
            self.checkpoint_pub.publish(marker)

    def intermediate_tfs(
        self, sorted_checkpoint_idx, target_checkpoint_tfs, num_points
    ):
        """This function takes the target checkpoint transforms and the desired order based on the shortest path sorting,
        and calls the decoupled_rot_and_trans() function.
        Args:
            sorted_checkpoint_idx (list): List describing order of checkpoints to follow. describing indexes in target_checkpoint_tfs by visiting order.
            target_checkpoint_tfs (np.ndarray): the state of the robot joints. In a youbot those are revolute
            num_points (int): Number of intermediate points between checkpoints.
        Returns:
            full_checkpoint_tfs: 4x4x(4xnum_points + 5) homogeneous transformations matrices describing the full desired
            poses of the end-effector position.
        """
        # TODO: implement this
        # Your code starts here ------------------------------
        full_checkpoint_tfs = []
        # add the first checkpoint
        full_checkpoint_tfs.append(target_checkpoint_tfs[:, :, sorted_checkpoint_idx[0]])
        #self.get_logger().info(f"full_checkpoint_tfs.shape: {full_checkpoint_tfs.shape}")
        for i in range(len(sorted_checkpoint_idx) - 1):
            idx_a = sorted_checkpoint_idx[i]
            idx_b = sorted_checkpoint_idx[i + 1]
            checkpoint_a_tf = target_checkpoint_tfs[:, :, idx_a]
            checkpoint_b_tf = target_checkpoint_tfs[:, :, idx_b]
            # produce intermediate points between two checkpoints
            intermediate_tfs = self.decoupled_rot_and_trans(checkpoint_a_tf, checkpoint_b_tf, num_points)
            self.get_logger().info(f"intermediate_tfs.shape: {intermediate_tfs.shape}")
            # store intermediate points
            for k in range(intermediate_tfs.shape[2]):
                full_checkpoint_tfs.append(intermediate_tfs[:, :, k])
            # store the end checkpoint
            full_checkpoint_tfs.append(checkpoint_b_tf)
            #self.get_logger().info(f"full_checkpoint_tfs.shape: {full_checkpoint_tfs.shape}")
        full_checkpoint_tfs = np.stack(full_checkpoint_tfs, axis=2) # 4x4x(4xnum_points + (initial_point(1) + num_targets(example:4)))
        self.get_logger().info(f"full_checkpoint_tfs.shape: {full_checkpoint_tfs.shape}")
        # Your code ends here ------------------------------

        return full_checkpoint_tfs

    def decoupled_rot_and_trans(self, checkpoint_a_tf, checkpoint_b_tf, num_points):
        """This function takes two checkpoint transforms and computes the intermediate transformations
        that follow a straight line path by decoupling rotation and translation.
        Args:
            checkpoint_a_tf (np.ndarray): 4x4 transformation describing pose of checkpoint a.
            checkpoint_b_tf (np.ndarray): 4x4 transformation describing pose of checkpoint b.
            num_points (int): Number of intermediate points between checkpoint a and checkpoint b.
        Returns:
            tfs: 4x4x(num_points) homogeneous transformations matrices describing the full desired
            poses of the end-effector position from checkpoint a to checkpoint b following a linear path.
        """
        self.get_logger().info("checkpoint a")
        self.get_logger().info(str(checkpoint_a_tf))
        self.get_logger().info("checkpoint b")
        self.get_logger().info(str(checkpoint_b_tf))
        # TODO: implement this
        # Your code starts here ------------------------------
        tfs = np.repeat(np.identity(4), num_points, axis=1).reshape((4, 4, num_points))
        # # slerp = Slerp([0.0, 1.0], R.from_matrix(np.stack([checkpoint_a_tf[0:3,0:3], checkpoint_b_tf[0:3,0:3]])))
        #Straight Line Paths– Cartesian space
        for i in range(num_points):
            # produce intermediate points excluding start and end points
            t = (i + 1) / (num_points + 1)
            # translation interpolation
            tfs[0:3, 3, i] = (1 - t) * checkpoint_a_tf[0:3, 3] + t * checkpoint_b_tf[0:3, 3]
            # rotation interpolation
            Rs = checkpoint_a_tf[0:3, 0:3]
            Rf = checkpoint_b_tf[0:3, 0:3]
            R_delta = Rs.T @ Rf
            log_R_delta = logm(R_delta)
            R_interp = Rs @ expm(t * log_R_delta)
            tfs[0:3, 0:3, i] = np.real(R_interp)
            # R_interp = slerp([t]).as_matrix()[0]  # 3x3
            # tfs[0:3,0:3,i] = R_interp

        # Your code ends here ------------------------------
        return tfs

    def full_checkpoints_to_joints(self, full_checkpoint_tfs, init_joint_position):
        """This function takes the full set of checkpoint transformations, including intermediate checkpoints,
        and computes the associated joint positions by calling the ik_position_only() function.
        Args:
            full_checkpoint_tfs (np.ndarray, 4x4xn): 4x4xn transformations describing all the desired poses of the end-effector
            to follow the desired path.
            init_joint_position (np.ndarray):A 5x1 array for the initial joint position of the robot.
        Returns:
            q_checkpoints (np.ndarray, 5xn): For each pose, the solution of the position IK to get the joint position
            for that pose.
        """
        # TODO: Implement this
        # Your code starts here ------------------------------
        # cartesian to joint space
        num_checkpoints = full_checkpoint_tfs.shape[2]
        q_checkpoints = np.zeros((5, num_checkpoints))
        q_checkpoints[:, 0] = init_joint_position

        for i in range(1, num_checkpoints):
            q_checkpoints[:, i], _ = self.ik_position_only(full_checkpoint_tfs[:, :, i], q_checkpoints[:, i-1])
        # Your code ends here ------------------------------

        return q_checkpoints

    def ik_position_only(self, pose, q0, lam=0.25, num=500):
        """This function implements position only inverse kinematics.
        Args:
            pose (np.ndarray, 4x4): 4x4 transformations describing the pose of the end-effector position.
            q0 (np.ndarray, 5x1):A 5x1 array for the initial starting point of the algorithm.
        Returns:
            q (np.ndarray, 5x1): The IK solution for the given pose.
            error (float): The Cartesian error of the solution.
        """

        # TODO: Implement this
        # Some useful notes:
        # We are only interested in position control - take only the position part of the pose as well as elements of the
        # Jacobian that will affect the position of the error.

        # Your code starts here ------------------------------
        q = q0.copy()
        error = 0.0
        #Iterative Approach
        for k in range(num):
            # Compute the current end-effector pose
            current_pose = self.kdl_youbot.forward_kinematics(q.tolist())

            # Compute the position error
            position_error = pose[0:3, 3] - current_pose[0:3, 3]
            error = np.linalg.norm(position_error)

            if error < 1e-6:
                break

            # Compute the Jacobian
            J = self.kdl_youbot.get_jacobian(q.tolist())
            Jv = J[0:3, :]
            # iterative optimization equations -Pseudoinverse
            J_inv = np.linalg.pinv(Jv)

            # Update the joint angles
            q += J_inv @ position_error * lam

        # Your code ends here ------------------------------
        return q, error


def main(args=None):
    rclpy.init(args=args)

    youbot_planner = YoubotTrajectoryPlanning()

    youbot_planner.run()

    rclpy.spin(youbot_planner)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    youbot_planner.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
