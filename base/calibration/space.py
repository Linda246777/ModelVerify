"""
弃用该文件

"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ..datatype import Pose, PosesData


class HandEyeAlg:
    Tsai = cv2.CALIB_HAND_EYE_TSAI
    Andreff = cv2.CALIB_HAND_EYE_ANDREFF
    Horaud = cv2.CALIB_HAND_EYE_HORAUD
    Park = cv2.CALIB_HAND_EYE_PARK
    Daniilidis = cv2.CALIB_HAND_EYE_DANIILIDIS


def global21(
    cs1: PosesData,
    cs2: PosesData,
    alg=HandEyeAlg.Tsai,
    rot_only=False,
) -> Pose:
    """calibrateHandEye return R_gc, t_gc
    隐含信息：base 和 target 为刚体，gripper 和 camera 为刚体。
    """
    assert len(cs1) > 10, f"Not enough data points {len(cs1)}"
    assert len(cs1) == len(cs2), f"{len(cs1)} != {len(cs2)}"
    # 标定
    rots1 = cs1.rots.inv().as_matrix()
    rots2 = cs2.rots.as_matrix()
    ps1 = cs1.ps
    ps2 = cs2.ps
    if rot_only:
        ps1 = np.zeros_like(ps1)
        ps2 = np.zeros_like(ps2)

    rot_gc, t_gc = cv2.calibrateHandEye(rots1, ps1, rots2, ps2, method=alg)  # type:ignore
    rvec = cv2.Rodrigues(rot_gc)[0]
    ang = np.linalg.norm(rvec)
    if ang != 0:
        rvec = rvec / ang
    print("旋转向量: ", rvec.flatten(), ang * 180 / np.pi)
    print("位移: ", t_gc.flatten())

    rot = Rotation.from_matrix(rot_gc)
    p = t_gc.flatten()
    return Pose(rot, p)
