import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

from youbot_kinematics.youbotKineStudent import YoubotKinematicStudent
from transform_helpers.utils import rotmat2q


class YoubotTrajectoryFollower(Node):
    """A simple node that listens to joint states, computes forward kinematics and
    publishes the end-effector transform to the TF tree.

    This mirrors the behaviour requested in the lab: subscribe to JointState messages
    and publish the end-effector frame so RViz will show it following the trajectory.
    """

    def __init__(self):
        super().__init__("youbot_trajectory_follower")

        # Kinematics helper (provides forward_kinematics(list_of_5_angles))
        self.kdl = YoubotKinematicStudent()

        # Transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribe to joint states published by the controller
        self.joint_sub = self.create_subscription(
            JointState, 
            "/joint_states",
            self.joint_state_callback,
            10,
        )
        self.joint_sub

    def joint_state_callback(self, msg: JointState):
        # Convert to python list and only use first 5 joints (youbot arm)
        try:
            joint_positions = list(msg.position)
        except Exception:
            # Defensive: if message has no position field or invalid type
            self.get_logger().warning("Received JointState with no positions")
            return

        if len(joint_positions) < 5:
            self.get_logger().warning(
                f"Expected at least 5 joint positions, got {len(joint_positions)}"
            )
            return

        joint_positions = joint_positions[:5]

        # Compute forward kinematics (returns a 4x4 numpy array)
        try:
            pose = self.kdl.forward_kinematics(joint_positions)
        except Exception as e:
            self.get_logger().error(f"FK computation failed: {e}")
            return

        # Build TransformStamped message
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id = "arm_end_effector_follower"

        # Translation
        t.transform.translation.x = float(pose[0, 3])
        t.transform.translation.y = float(pose[1, 3])
        t.transform.translation.z = float(pose[2, 3])

        # Rotation (convert rotation matrix to geometry_msgs/Quaternion)
        try:
            t.transform.rotation = rotmat2q(pose[:3, :3])
        except Exception as e:
            self.get_logger().error(f"rotmat2q failed: {e}")
            return

        # Send the transform
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    follower = YoubotTrajectoryFollower()
    rclpy.spin(follower)
    follower.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
