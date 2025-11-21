import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from torch.utils.checkpoint import TypeAlias

from base.interpolate import get_time_series, interpolate_vector3, slerp_rotation


@dataclass
class Pose:
    rot: Rotation
    trans: NDArray

    @classmethod
    def identity(cls):
        return cls(Rotation.identity(), np.zeros(3))

    @classmethod
    def from_rotation(cls, rot: Rotation):
        return cls(rot, np.zeros(3))

    @classmethod
    def from_trans(cls, trans: NDArray):
        return cls(Rotation.identity(), trans)

    def compose(self, other: Self):
        return Pose(self.rot * other.rot, self.rot.apply(other.trans) + self.trans)

    def compose_self(self, other: Self):
        self.trans = self.rot.apply(other.trans) + self.trans
        self.rot = self.rot * other.rot

    def compose_trans_self(self, trans: NDArray):
        self.trans += self.rot.apply(trans)

    def inverse(self):
        return Pose(self.rot.inv(), -self.rot.apply(self.trans))


Frame: TypeAlias = Literal["local", "global"]


@dataclass
class ImuData:
    t_us: NDArray
    acce: NDArray
    gyro: NDArray
    ahrs: Rotation

    frame: Frame = "local"

    def __len__(self):
        return self.t_us.__len__()

    @classmethod
    def from_raw(cls, raw: NDArray):
        assert raw.shape[1] == 12, f"Invalid raw data shape: {raw.shape}"
        gyro = raw[:, 1:4]
        acce = raw[:, 4:7]
        ahrs = Rotation.from_quat(raw[:, 7:11], scalar_first=True)
        t_us = raw[:, 0] + raw[:, 11][0] + -raw[:, 0][0]
        return cls(t_us, acce, gyro, ahrs)

    @classmethod
    def from_csv(cls, path: Path):
        raw = pd.read_csv(path).to_numpy()
        return cls.from_raw(raw)

    def interpolate(self, t_new_us: NDArray):
        acce = interpolate_vector3(self.acce, self.t_us, t_new_us)
        gyro = interpolate_vector3(self.gyro, self.t_us, t_new_us)
        ahrs = slerp_rotation(self.ahrs, self.t_us, t_new_us)
        return ImuData(t_new_us, acce, gyro, ahrs)

    def transform(self, rots: Rotation | None = None):
        if rots is None:
            rots = self.ahrs
        matrix_rots = rots.as_matrix()
        acce = np.einsum("ijk,ik->ij", matrix_rots, self.acce)
        gyro = np.einsum("ijk,ik->ij", matrix_rots, self.gyro)
        return ImuData(self.t_us, acce, gyro, rots, frame="global")


@dataclass
class PosesData:
    t_us: NDArray
    rots: Rotation
    trans: NDArray

    def interpolate(self, t_new_us: NDArray):
        rots = slerp_rotation(self.rots, self.t_us, t_new_us)
        trans = interpolate_vector3(self.trans, self.t_us, t_new_us)
        return PosesData(t_new_us, rots, trans)

    def transform_local(self, tf: Pose):
        # R = R * R_loc
        # t = t + R * t_loc
        self.trans = self.trans + self.rots.apply(tf.trans)
        self.rots = self.rots * tf.rot

    def transform_global(self, tf: Pose):
        # R = R_glo * R
        # t = t_glo + R_glo * t
        self.rots = tf.rot * self.rots
        self.trans = tf.trans + tf.rot.apply(self.trans)


class GroundTruthData(PosesData):
    @classmethod
    def from_raw(cls, raw: NDArray):
        t_us = raw[:, 0]
        trans = raw[:, 1:4]
        quats = raw[:, 4:8]
        # qwxyz
        rots = Rotation.from_quat(quats, scalar_first=True)
        return cls(t_us, rots, trans)

    @classmethod
    def from_csv(cls, path: Path):
        raw = pd.read_csv(path).to_numpy()
        return cls.from_raw(raw)


def get_ang_vec(rot: Rotation):
    rot_vec = rot.as_rotvec()
    angle = np.linalg.norm(rot_vec)
    vec = rot_vec / angle
    return vec.tolist(), float(angle) * 180 / np.pi


@dataclass
class CalibrationData:
    tf_sg_local: Pose
    tf_sg_global: Pose

    @classmethod
    def from_json(cls, path: Path):
        with open(path, "r") as f:
            data = json.load(f)
            if not isinstance(data, list) or len(data) != 1:
                raise ValueError("Invalid JSON format")
            data = data[0]

            rot_local = np.array(data["rot_sensor_gt"])
            trans_local = np.array(data["trans_sensor_gt"]).flatten()
            tf_sg_local = Pose(Rotation.from_matrix(rot_local), trans_local)

            rot_global = np.array(data["rot_ref_sensor_gt"])
            trans_global = np.array(data["trans_ref_sensor_gt"]).flatten()
            tf_sg_global = Pose(Rotation.from_matrix(rot_global), trans_global)

            print(
                "Transforms local: ",
                get_ang_vec(tf_sg_local.rot),
                f"\n{tf_sg_local.trans}",
            )
            print(
                "Transforms global: ",
                get_ang_vec(tf_sg_global.rot),
                f"\n{tf_sg_global.trans}",
            )

            return cls(tf_sg_local, tf_sg_global)


@dataclass
class DataCheck:
    t_gi_us: int

    @classmethod
    def from_json(cls, path: Path):
        with open(path, "r") as f:
            data = json.load(f)
            assert "check_time_diff" in data
            check_time_diff = data["check_time_diff"]
            assert "time_diff_21_us" in check_time_diff
            t_gi_us = check_time_diff["time_diff_21_us"]
            return cls(t_gi_us)


class UnitData:
    imu_data: ImuData
    gt_data: PosesData
    calib_data: CalibrationData
    check_data: DataCheck

    def __init__(
        self,
        base_dir: Path | str,
        step: int = 10,
        block_size: int = 200,
        remove_gravity: bool = False,
    ):
        base_dir = Path(base_dir)
        self.step = step
        self.bs = block_size
        self.rm_g = remove_gravity

        self._imu_path = base_dir / "imu.csv"
        self._gt_path = base_dir / "rtab.csv"
        # 读取标定数据
        self._calib_file = base_dir / "Calibration.json"
        self._check_file = base_dir / "DataCheck.json"

        self.calib_data = CalibrationData.from_json(self._calib_file)
        self.check_data = DataCheck.from_json(self._check_file)

        self.load_data()

    def load_data(self):
        imu_data = ImuData.from_csv(self._imu_path)
        gt_data = GroundTruthData.from_csv(self._gt_path)

        # 时间修正
        gt_data.t_us += self.check_data.t_gi_us

        # 空间变换
        gt_data.transform_local(self.calib_data.tf_sg_local.inverse())
        # gt_data.transform_global(self.calib_data.tf_sg_global)

        # 数据对齐
        t_new_us = get_time_series([imu_data.t_us, gt_data.t_us])
        self.imu_data = imu_data.interpolate(t_new_us)
        self.gt_data = gt_data.interpolate(t_new_us)
