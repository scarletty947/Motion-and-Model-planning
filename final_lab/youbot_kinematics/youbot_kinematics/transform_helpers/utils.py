from geometry_msgs.msg import Quaternion

import numpy as np
from numpy.typing import NDArray

def rotmat2q(T: NDArray) -> Quaternion:
    # Function that transforms a 3x3 rotation matrix to a ros quaternion representation   
    if T.shape != (3, 3):
        raise ValueError

    # TODO: implement this
    r11,r12,r13=T[0]
    r21,r22,r23=T[1]
    r31,r32,r33=T[2]
    trace = r11+r22+r33
    if(trace>=0):
        #trace >0 1+r11+r22+r33>1 => w^2>1
        s=np.sqrt(1+trace)
        w=(1/2)*s
        x=(1/2)*(r32-r23)/s
        y=(1/2)*(r13-r31)/s
        z=(1/2)*(r21-r12)/s
    else:
        if(r11>r22 and r11>r33):
            s=np.sqrt(1+r11-r22-r33)
            x=(1/2)*s
            w=(1/2)*(r32-r23)/s
            y=(1/2)*(r12+r21)/s
            z=(1/2)*(r31+r13)/s
        elif (r22>r11 and r22>r33):
            s=np.sqrt(1+r22-r11-r33)
            y=(1/2)*s
            w=(1/2)*(r13-r31)/s
            x=(1/2)*(r12+r21)/s
            z=(1/2)*(r23+r32)/s
        else:
            s=np.sqrt(1+r33-r11-r22)
            z=(1/2)*s
            w=(1/2)*(r21-r12)/s
            x=(1/2)*(r13+r31)/s
            y=(1/2)*(r23+r32)/s
    #raise NotImplementedError
    q = Quaternion()
    q.w = w
    q.x = x
    q.y = y
    q.z = z
    return q
# TODO: This can also be implemented by scipy.spatial.transform.Rotation
# from scipy.spatial.transform import Rotation as R
#  def rotmat2q(T: NDArray) -> Quaternion:
#     """Convert a 3×3 rotation matrix to a ROS Quaternion."""
#     if T.shape != (3, 3):
#         raise ValueError("Input must be a 3×3 matrix")

#     rot = R.from_matrix(T)
#     q = rot.as_quat()  # [x, y, z, w] order in SciPy

#     return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])