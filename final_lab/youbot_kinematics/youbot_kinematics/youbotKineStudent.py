import rclpy
import time
import threading

import numpy as np
from youbot_kinematics.youbotKineBase import YoubotKinematicBase
from youbot_kinematics.target_data import TARGET_JOINT_POSITIONS


class YoubotKinematicStudent(YoubotKinematicBase):
    def __init__(self):
        super(YoubotKinematicStudent, self).__init__(tf_suffix='student')

        # Set the offset for theta --> This was updated on 22/11/2024. Fill it in with your calculated joint offsets in cw1 if you need testing.
        # the standard joint offsets will be updated soon.
        # this is encoder offsets in radians meaning that when the encoder reads 0, the joint angle is at the offset
        youbot_joint_offsets = [170.0 * np.pi / 180.0,
                                -65.0 * np.pi / 180.0,
                                146 * np.pi / 180,
                                -102.5 * np.pi / 180,
                                -167.5 * np.pi / 180]

        # Apply joint offsets to dh parameters
        self.dh_params['theta'] = [theta + offset for theta, offset in
                                   zip(self.dh_params['theta'], youbot_joint_offsets)]

        # Joint reading polarity signs
        # +1 means joint reading increases with positive rotation, -1 means it decreases
        self.youbot_joint_readings_polarity = [-1, 1, 1, 1, 1]

    def forward_kinematics(self, joints_readings, up_to_joint=5):
        """This function solve forward kinematics by multiplying frame transformation up until a specified
        frame number. The frame transformation used in the computation are derived from dh parameters and
        joint_readings.
        Args:
            joints_readings (list): the state of the robot joints. In a youbot those are revolute
            up_to_joint (int, optional): Specify up to what frame you want to compute forward kinematics.
                Defaults to 5.
        Returns:
            np.ndarray: A 4x4 homogeneous transformation matrix describing the pose of frame_{up_to_joint}
                w.r.t the base of the robot.
        """
        assert isinstance(self.dh_params, dict)
        assert isinstance(joints_readings, list), "joint readings of type " + str(type(joints_readings))
        assert isinstance(up_to_joint, int)
        assert up_to_joint >= 0
        assert up_to_joint <= len(self.dh_params['a'])

        T = np.identity(4)

        # Apply offset and polarity to joint readings (found in URDF file)
        joints_readings = [sign * angle for sign, angle in zip(self.youbot_joint_readings_polarity, joints_readings)]

        for i in range(up_to_joint):
            A = self.standard_dh(self.dh_params['a'][i],
                                 self.dh_params['alpha'][i],
                                 self.dh_params['d'][i],
                                 self.dh_params['theta'][i] + joints_readings[i])
            T = T.dot(A)
            
        assert isinstance(T, np.ndarray), "Output wasn't of type ndarray"
        assert T.shape == (4, 4), "Output had wrong dimensions"
        return T

    def get_jacobian(self, joint):
        """Given the joint values of the robot, compute the Jacobian matrix.

        Args:
            joint (list): the state of the robot joints. In a youbot those are revolute

        Returns:
            Jacobian (numpy.ndarray): NumPy matrix of size 6x5 which is the Jacobian matrix.
        """
        assert isinstance(joint, list)
        assert len(joint) == 5

        # Your code starts here ----------------------------
        z0 = np.array([0, 0, -1])
        o0 = np.array([0, 0, 0])
        o = [o0]
        z = [z0]    
        for i in range(5):
            forward_kinematics_i = self.forward_kinematics(joint, up_to_joint=i+1)
            z.append(forward_kinematics_i[0:3, 2])
            o.append(forward_kinematics_i[0:3, 3])
        p_e = o[-1] # end-effector position
        jacobian = np.zeros((6, 5))
        for i in range(5):
            jacobian[:3, i] = np.cross(z[i], (p_e - o[i]))
            jacobian[3:, i] = z[i]
        # Your code ends here ------------------------------    

        # For your solution to match the KDL Jacobian, z0 needs to be set [0, 0, -1] instead of [0, 0, 1], since that is how its defined in the URDF.
        # Both are correct.
        # Your code starts here ----------------------------
       
        # Your code ends here ------------------------------
        assert jacobian.shape == (6, 5)
        return jacobian

    def check_singularity(self, joint):
        """Check for singularity condition given robot joints. Coursework 2 Question 4c.
        Reference Lecture 5 slide 30.

        Args:
            joint (list): the state of the robot joints. In a youbot those are revolute

        Returns:
            singularity (bool): True if in singularity and False if not in singularity.

        """
        assert isinstance(joint, list)
        assert len(joint) == 5
        # Your code starts here ----------------------------
        J = self.get_jacobian(joint)
        rank_J = np.linalg.matrix_rank(J)
        singularity = (rank_J < 5)
        
        # det_J = np.linalg.det(J.T @ J)
        # singularity = (det_J < 1e-6)

        # Your code ends here ------------------------------
        assert isinstance(singularity, bool)
        return singularity


def main(args=None):
    rclpy.init(args=args)

    kinematic_student = YoubotKinematicStudent()

    for i in range(TARGET_JOINT_POSITIONS.shape[0]):
        target_joint_angles = TARGET_JOINT_POSITIONS[i]
        target_joint_angles = target_joint_angles.tolist()
        pose = kinematic_student.forward_kinematics(target_joint_angles)
        # we would probably compute the jacobian at our current joint angles, not the target
        # but this is just to check your work
        jacobian = kinematic_student.get_jacobian(target_joint_angles)
        print("target joint angles")
        print(target_joint_angles)
        print("pose")
        print(pose)
        print("jacobian")
        print(jacobian)

    rclpy.spin(kinematic_student)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    kinematic_student.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()