from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from base.datatype import ImuData, PosesData


class TLIOData:
    imu_data: ImuData
    gt_data: PosesData
    velocities: NDArray

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.load_npy_file(self.base_dir / "imu0_resampled.npy")

    def load_npy_file(self, file_path):
        data = np.load(file_path)
        t_us = data[:, 0]
        gyro = data[:, 1:4]
        acce = data[:, 4:7]
        rots = Rotation.from_quat(data[:, 7:11])
        ps = data[:, 11:14]
        vs = data[:, 14:17]

        self.imu_data = ImuData(t_us, gyro, acce, rots, "global")
        self.gt_data = PosesData(t_us, rots, ps)
        self.vs = vs
