#!/usr/bin/env python3
"""
TLIO数据可视化

用于可视化TLIO（Tracker Lite Inertial Odometry）数据。

用法:
    # 可视化单个数据单元
    uv run python visualizers/TLIOView.py -u <unit_path> -m model_name

    # 指定模型文件夹
    uv run python visualizers/TLIOView.py -u <unit_path> -m model_name --models_path /path/to/models

参数:
    -u, --unit: 指定单个数据单元路径
    -m, --models: 指定模型文件名（支持多个）
    --models_path: 指定模型文件夹路径（默认为"models"）
"""
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

import base.rerun_ext as rre
from base.args_parser import DatasetArgsParser
from base.datatype import ImuData, PosesData
from base.model import InertialNetwork, InertialNetworkData, ModelLoader


class TLIOData:
    imu_data: ImuData
    gt_data: PosesData
    velocities: NDArray

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.name = self.base_dir.name
        self.load_npy_file(self.base_dir / "imu0_resampled.npy")

    def load_npy_file(self, file_path):
        data = np.load(file_path)
        t_us = data[:, 0]
        gyro = data[:, 1:4]
        acce = data[:, 4:7]
        rots = Rotation.from_quat(data[:, 7:11])
        ps = data[:, 11:14]
        vs = data[:, 14:17]

        self.imu_data = ImuData(
            t_us, gyro, acce, rots, np.zeros_like(acce), frame="global"
        )
        self.gt_data = PosesData(t_us, rots, ps)
        self.velocities = vs


def tlio_view(path: str | Path, net: InertialNetwork):
    td = TLIOData(path)
    rre.RerunView().add_imu_view(tags=["Groundtruth"]).add_spatial_view().send(td.name)
    rre.send_pose_data(td.gt_data, "Groundtruth")
    rre.send_imu_data(td.imu_data, tag="Groundtruth")

    in_data = InertialNetworkData(td.imu_data)
    in_data.predict_usings([net], td.gt_data)


if __name__ == "__main__":
    args = DatasetArgsParser().parse()
    models_path = "Models"
    loader = ModelLoader(models_path)

    net = loader.get_by_name("0122"Î)
    if args.unit is not None:
        tlio_view(args.unit, net)
    elif args.dataset is not None:
        for path in Path(args.dataset).iterdir():
            if path.is_dir():
                tlio_view(path, net)
