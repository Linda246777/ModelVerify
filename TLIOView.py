from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

import base.rerun_ext as rre
from base.args_parser import DatasetArgsParser
from base.datatype import ImuData, PosesData
from base.model import InerialNetwork, InerialNetworkData, ModelLoader


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

        self.imu_data = ImuData(t_us, gyro, acce, rots, "global")
        self.gt_data = PosesData(t_us, rots, ps)
        self.vs = vs


def tlio_view(path: str | Path, net: InerialNetwork):
    td = TLIOData(path)
    rre.rerun_init(td.name, imu_view_tags=["GT"])
    rre.send_pose_data(td.gt_data)
    rre.send_imu_data(td.imu_data, tag="GT")

    in_data = InerialNetworkData(td.imu_data)
    in_data.predict_using(net)


if __name__ == "__main__":
    args = DatasetArgsParser().parse()
    models_path = "/Users/qi/Resources/Models"
    loader = ModelLoader(models_path)

    net = loader.get_by_name("ZLX_03")
    if args.unit is not None:
        tlio_view(args.unit, net)
    elif args.dataset is not None:
        for path in Path(args.dataset).iterdir():
            if path.is_dir():
                tlio_view(path, net)
