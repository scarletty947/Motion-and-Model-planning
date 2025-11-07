import rclpy
from rclpy.node import Node
import numpy as np
from typing import Optional, Tuple
from numpy.typing import NDArray
import os

from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.duration import Duration
import matplotlib.pyplot as plt


from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import PyKDL
from youbot_kinematics.urdf import treeFromUrdfModel
from youbot_kinematics.urdf_parser import URDF

from ament_index_python.packages import get_package_share_directory

import yaml


class TrajDemo(Node):
    def __init__(self):
        super().__init__("traj_demo")

        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 5
        )
        self.joint_state_sub  # prevent unused variable warning
        self.traj_publisher = self.create_publisher(
            JointTrajectory, "/franka_arm_controller/joint_trajectory", 5
        )
        self.traj_publisher

        # load from urdf file
        self.declare_parameter("urdf_package", "franka_description")
        self.urdf_package = (
            self.get_parameter("urdf_package").get_parameter_value().string_value
        )
        self.urdf_package_path = get_package_share_directory(self.urdf_package)
        self.declare_parameter("urdf_path_in_package", "urdfs/fr3.urdf")
        self.urdf_path_in_package = (
            self.get_parameter("urdf_path_in_package")
            .get_parameter_value()
            .string_value
        )
        self.urdf_name_path = os.path.join(
            self.urdf_package_path, self.urdf_path_in_package
        )

        self.get_logger().info(
            f"loading robot into KDL from urdf: {self.urdf_name_path}"
        )

        robot = URDF.from_xml_file(self.urdf_name_path)

        # 将一个解析好的 URDF 模型转换为 KDL 的运动学树结构
        (ok, self.kine_tree) = treeFromUrdfModel(robot)

        if not ok:
            raise RuntimeError("couldn't load URDF into KDL tree succesfully")

        self.declare_parameter("base_link", "base")
        self.declare_parameter("ee_link", "fr3_link8")
        self.base_link = (
            self.get_parameter("base_link").get_parameter_value().string_value
        )
        self.ee_link = self.get_parameter("ee_link").get_parameter_value().string_value

        # 从 KDL 的运动学树 self.kine_tree 中提取出从 base_link 到 ee_link（末端执行器）的关节链（KDL::Chain）
        # 这条链包含了所有中间关节和连杆的信息
        self.kine_chain = self.kine_tree.getChain(self.base_link, self.ee_link)
        # 获取关节数量
        self.NJoints = self.kine_chain.getNrOfJoints()
        self.current_joint_position = PyKDL.JntArray(self.NJoints)
        self.current_joint_velocity = PyKDL.JntArray(self.NJoints)
        # KDL solvers
        # 初始化一个逆运动学求解器
        # 输入目标末端位姿 → 输出对应的关节角度解
        # 用于轨迹规划中将笛卡尔空间目标转换为关节空间动作
        self.ik_solver = PyKDL.ChainIkSolverPos_LMA(self.kine_chain)

        # 实时监听坐标变换（tf2）
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # loading trajectory structure

        self.declare_parameter("traj_cfg_pkg", "trajectory_demo")
        self.traj_cfg_pkg = (
            self.get_parameter("traj_cfg_pkg").get_parameter_value().string_value
        )
        self.traj_cfg_pkg_path = get_package_share_directory(self.traj_cfg_pkg)
        self.declare_parameter("traj_cfg_path_within_pkg", "cfg/traj_waypoints.yaml")
        self.traj_cfg_path_within_pkg = (
            self.get_parameter("traj_cfg_path_within_pkg")
            .get_parameter_value()
            .string_value
        )

        self.traj_cfg_path = os.path.join(
            self.traj_cfg_pkg_path, self.traj_cfg_path_within_pkg
        )

        with open(self.traj_cfg_path) as stream:
            self.traj_cfg = yaml.safe_load(stream)

        assert type(self.traj_cfg["joint_names"]) == list, type(
            self.traj_cfg["joint_names"]
        )
        assert len(self.traj_cfg["joint_names"]) == self.NJoints

        self.created_traj = False

        self.get_logger().info(f"got traj cfg:\n{self.traj_cfg}")

        # TODO: modify to create a datastructure to store joint positions and cartesian positions along the trajectory
        self.joint_pos_history = []
        self.joint_vel_history = []
        self.end_effector_pos_history = []
        # self.get_logger().info("Waiting for TF to become available...")
        # rclpy.spin_once(self, timeout_sec=1.0)
        # self.tf_buffer.can_transform('world', 'fr3_link8', rclpy.time.Time(), timeout=Duration(seconds=2.0))
        # self.get_logger().info("TF listener ready ✅")

    def joint_state_callback(self, msg: JointState):
        """Callback for the joint states of the robot arm. It will get joint positions and velocities, and eventually the cartesian position of the end-effector and save this. It allows initializes the trajectory once the first joint state message comes through.

        Args:
            msg (JointState): ROS Joint State Message.
        """
        for i in range(len(msg.name)):
            n = msg.name[i]
            pos = msg.position[i]
            vel = msg.velocity[i]

            self.current_joint_position[i] = pos
            self.current_joint_velocity[i] = vel

        joint_pos, joint_vel = self.kdl_to_np(
            self.current_joint_position
        ), self.kdl_to_np(self.current_joint_velocity)
        # TODO: modify to save position and velocities into some array to plot later
        # 每个关节的当前位置（角度或位移）
        self.joint_pos_history.append(joint_pos)
        self.joint_vel_history.append(joint_vel)
        # TODO: modify to save the position of end effector using get_ee_pos_ros
        #self.get_logger().warn(f"End-effector position: {self.get_ee_pos_ros()}")
        self.end_effector_pos_history.append(self.get_ee_pos_ros())
        # we do this after we have our first callback to have current joint positions
        if not self.created_traj:
            joint_traj = self.create_traj()
            self.traj_publisher.publish(joint_traj)
            # # log what we published to help debug controllers not executing trajectories
            # try:
            #     names = joint_traj.joint_names
            # except Exception:
            #     names = []
            # self.get_logger().info(f"Published trajectory with joints: {names}, points: {len(joint_traj.points)}")
            self.created_traj = True

    def kdl_to_np(self, data: PyKDL.JntArray) -> NDArray:
        """Helper Function to go from KDL arrays to numpy arrays

        Args:
            data (PyKDL.JntArray): desired KDL array to convert

        Returns:
            NDArray: converted NP Array
        """
        is_1d = data.columns() == 1
        np_shape = (data.rows(), data.columns()) if not is_1d else (data.rows(),)
        mat = np.zeros(np_shape)
        for i in range(data.rows()):
            if not is_1d:
                for j in range(data.columns()):
                    mat[i, j] = data[i, j]
            else:
                mat[i] = data[i]
        return mat

    def get_ee_pos_ros(self) -> Tuple[float]:
        """FK function that uses TF2 ROS to do forward kinematics and return world coordinates of the end effector.

        Returns:
            Tuple[float]: position, 3 dimensional.
        """
        # TODO: Modify this to get current position of the robot's end effector using the TF2 Library
        pos = [0.0, 0.0, 0.0]

        try:
            if self.tf_buffer.can_transform('world', self.ee_link, rclpy.time.Time(seconds=0),timeout=rclpy.duration.Duration(seconds=2.0)):
            # 类型注解语法:“变量 transform 的类型是 TransformStamped。”
                transform: TransformStamped = self.tf_buffer.lookup_transform(
                    'world', self.ee_link, rclpy.time.Time(seconds=0)
                )
                pos = transform.transform.translation
                pos = [pos.x, pos.y, pos.z]
                return pos
            else:
                self.get_logger().warn(f'Transform world -> {self.ee_link} not available yet.')
                return (0.0, 0.0, 0.0)
        except Exception as e:
            self.get_logger().warn(f'Failed to get end-effector position: {e}')
            return (0.0, 0.0, 0.0)

        

    def get_joint_pos(
        self,
        current_angles: Tuple[float],
        target_position: Tuple[float],
        target_orientation: Optional[Tuple[float]] = None,
    ) -> Tuple[float]:
        """Helper function to solve inverse kinematics using the KDL library

        Args:
            current_angles (Tuple[float]): current joint angles of the robot arm
            target_position (Tuple[float]): target cartesian position, 3 dimensional, of the end effector in world coordinates
            target_orientation (Optional[Tuple[float]], optional): target orientation in RPY format, optional. Defaults to None.

        Raises:
            RuntimeError: raises error if IK fails to solve. Check the workspace of the robot if this happens.

        Returns:
            Tuple[float]: joint angles that were found to satisfy targets
        """

        assert len(target_position) == 3
        assert target_orientation is None or len(target_orientation) == 3
        assert len(current_angles) == self.NJoints

        pos = PyKDL.Vector(target_position[0], target_position[1], target_position[2])
        if target_orientation is not None:
            # Constructs a rotation by first applying a rotation of r around the x-axis, then a rotation of p around the original y-axis, and finally a rotation of y around the original z-axis
            rot = PyKDL.Rotation.RPY(
                target_orientation[0], target_orientation[1], target_orientation[2]
            )

        seed_array = PyKDL.JntArray(self.NJoints)
        for i in range(self.NJoints):
            seed_array[i] = current_angles[i]

        if target_orientation is not None:
            goal_pose = PyKDL.Frame(rot, pos)
        else:
            goal_pose = PyKDL.Frame(pos)
        result_angles = PyKDL.JntArray(self.NJoints)

        if self.ik_solver.CartToJnt(seed_array, goal_pose, result_angles) >= 0:
            result = list(result_angles)
            return result
        else:
            raise RuntimeError(
                f"Did not solve for goal_pose: {goal_pose} with initial seed {seed_array}"
            )
    def create_traj(self) -> JointTrajectory:
        """Helper function to generate the trajectory to send to the arm.  Loops over cartesian waypoints from the config file, processing them into joint positions using the IK Solver

        Returns:
            JointTrajectory: Ros JointTrajectory to publish
        """
        cartesian_waypoints = self.traj_cfg["waypoints"]["cartesion"]

        cur_joint_pos = self.kdl_to_np(self.current_joint_position)

        goal_positions = []#目标关节pos
        goal_times = []
        # 插入当前关节位置作为起点
        goal_positions.append(cur_joint_pos.tolist())

        time_since_start = 0.0
        goal_times.append(time_since_start)  # 起点时间为 0

        for waypoint in cartesian_waypoints:
            pos = waypoint["position"]
            time_since_start += waypoint["time"]
            goal_times.append(time_since_start)
            cur_joint_pos = self.get_joint_pos(cur_joint_pos, pos)
            goal_positions.append(cur_joint_pos)

        # TODO: instantiate a JointTrajectory message, populate it, and publish it via the publisher class member
        # note, populate the header with the relevant frame id and timestamp, in addition to the trajectory
        joint_traj = JointTrajectory()
        joint_traj.header.stamp = self.get_clock().now().to_msg()
        # use the configured base link (matches URDF / TF frames)
        joint_traj.header.frame_id = self.base_link
        # ensure joint names are set so controllers/action servers know which joints these positions map to
        joint_traj.joint_names = self.traj_cfg["joint_names"]

        for i in range(len(goal_positions)):
            point = JointTrajectoryPoint()
            point.positions = goal_positions[i]
            
            point.time_from_start = Duration(seconds=goal_times[i]).to_msg()
            joint_traj.points.append(point)

        return joint_traj

    def plot_joint_traj(self):
        """Helper function to plot the desired results from following the trajectory. Will be called once the process is interupted via a Keyboard Interupt."""

        # TODO: populate this method to use the datastructure you created for joint positions
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        if len(self.end_effector_pos_history) == 0:
            # nothing to plot
            self.get_logger().warn("No end-effector history to plot")
            return

        x = [p[0] for p in self.end_effector_pos_history]
        y = [p[1] for p in self.end_effector_pos_history]
        z = [p[2] for p in self.end_effector_pos_history]

        # plot trajectory line
        ax.plot(x, y, z, label="End Effector Trajectory", color="tab:blue")

        # mark start point
        ax.scatter([x[0]], [y[0]], [z[0]], color="green", s=60, marker="o", label="Start")
        # mark end/goal point
        ax.scatter([x[-1]], [y[-1]], [z[-1]], color="red", s=80, marker="X", label="Goal")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # annotate start and goal with small text offsets so markers are clear
        try:
            sx, sy, sz = x[0], y[0], z[0]
            gx, gy, gz = x[-1], y[-1], z[-1]
            ax.text(sx, sy, sz, f" Start\n({sx:.3f},{sy:.3f},{sz:.3f})", color="green", fontsize=8)
            ax.text(gx, gy, gz, f" Goal\n({gx:.3f},{gy:.3f},{gz:.3f})", color="red", fontsize=8)
        except Exception:
            # if annotation fails, continue without crashing
            pass

        # place legend outside the axes to avoid covering plotted points
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)

        # make room for the legend on the right
        fig.tight_layout()
        fig.subplots_adjust(right=0.75)

        plt.show()


def main(args=None):
    rclpy.init(args=args)
    traj_demo = TrajDemo()

    try:
        rclpy.spin(traj_demo)
    except KeyboardInterrupt:
        pass

    traj_demo.plot_joint_traj()
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    traj_demo.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":

    main()
