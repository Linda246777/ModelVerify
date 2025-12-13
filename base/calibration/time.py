import numpy as np
from numpy._typing import NDArray
from scipy.spatial.transform import Rotation

from base.datatype import Pose, PosesData
from base.interpolate import get_time_series


def _get_angvels(t_us: NDArray, rots: Rotation, step: int = 1):
    """获取角速度列表"""
    n = len(rots)
    step = max(int(step), 1)
    assert n >= 2, "At least two rotations are required"

    As: list = []
    Ts = []
    for i in range(0, n - step, step):
        drot = rots[i].inv() * rots[i + step]
        angle = float(np.linalg.norm(drot.as_rotvec()))
        dt_s = (t_us[i + step] - t_us[i]) * 1e-6
        assert dt_s > 0, "Time difference must be positive"
        ang_vel = angle / dt_s
        As.append(ang_vel)
        Ts.append(t_us[i])
    return As, Ts


def match21(
    cs1: PosesData,
    cs2: PosesData,
    *,
    time_range=(0, 50),
    resolution=100,
):
    # 分辨率不能大于时间序列的采样率，否则没有插值的意义
    rate = min(cs1.rate, cs2.rate)
    resolution = min(resolution, rate)
    print(f"Rate1:{cs1.rate}, Rate2: {cs2.rate}, reso: {resolution}")

    t_new_us = get_time_series([cs1.t_us, cs2.t_us], *time_range, rate=rate)
    cs1 = cs1.interpolate(t_new_us)
    cs2 = cs2.interpolate(t_new_us)
    print(f"使用时间范围：{(cs1.t_us[-1] - cs1.t_us[0]) / 1e6} 秒, 数量 {len(cs1)}")

    seq1, t1 = _get_angvels(cs1.t_us, cs1.rots, step=int(rate / resolution))
    seq2, t2 = _get_angvels(cs2.t_us, cs2.rots, step=int(rate / resolution))
    t_new_us = t1

    corr = np.correlate(seq1, seq2, mode="full")
    lag_arr = np.arange(-len(seq2) + 1, len(seq1))
    lag = lag_arr[np.argmax(corr)]
    t21_us = lag * (t_new_us[1] - t_new_us[0])
    print("Ground time gap: ", t21_us / 1e6)

    time_range = (0, 20)
    cs1 = cs1.get_time_range(time_range)
    cs2 = cs2.get_time_range(time_range)
    cs2.t_us += t21_us

    def get_angle(v1, v2):
        # v1 = v1 / np.linalg.norm(v1)
        # v2 = v2 / np.linalg.norm(v2)
        # 计算夹角
        dot = np.dot(v1, v2)
        det = np.cross(v1, v2)
        return np.arctan2(det, dot)

    angles = []
    for i in range(len(cs1) - lag):
        v1 = (cs1.ps[i + lag] - cs1.ps[lag])[:2]
        v2 = (cs2.ps[i] - cs2.ps[0])[:2]

        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)
        if l1 > 2 and np.abs(l1 - l2) < 1:
            t1 = cs1.t_us[i + lag] / 1e6
            t2 = cs2.t_us[i] / 1e6
            ang = get_angle(v1, v2)
            print(i, t1, t2, l1, l2, ang * 180 / np.pi)
            angles.append(ang)

    # 构造绕z轴的旋转
    rad = np.mean(np.array(angles))
    rot21 = Rotation.from_rotvec([0, 0, rad])
    print(rot21.as_rotvec(degrees=True))
    return t21_us, Pose(rot21, np.zeros(3))
