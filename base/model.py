from pathlib import Path
from typing import Annotated, Literal, TypeAlias

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

import base.rerun_ext as rre

from .datatype import ImuData, Pose, UnitData
from .device import DefaultDevice

NetworkOutput: TypeAlias = tuple[NDArray, NDArray]


class InerialNetwork:
    model: torch.nn.Module

    def __init__(
        self,
        model_path: Path | str,
        input_shape: tuple[int, ...] | None = None,
    ):
        self.model_path = Path(model_path)
        self.name = self.model_path.name.split(".")[0]
        self.device = DefaultDevice
        self.model = torch.jit.load(self.model_path, map_location=self.device)
        self.input_shape = input_shape
        self.model.eval()

        print(f"Model: {self.model_path.name} load success.")

    def predict(self, block: NDArray) -> NetworkOutput:
        if self.input_shape:
            assert block.shape == self.input_shape, (
                f"Input shape mismatch: {block.shape} != {self.input_shape}"
            )
        inputs = torch.as_tensor(block, dtype=torch.float32, device=self.device)
        output = self.model(inputs)
        meas: NDArray = output[0].cpu().detach().numpy().flatten()
        meas_cov: NDArray = output[1].cpu().detach().numpy().flatten()
        return meas, meas_cov

    def reset(self):
        self.model.eval()


class ModelLoader:
    _suffix = ".pt"
    models: list = []

    def __init__(self, base_dir: Path | str):
        path = Path(base_dir)
        self.models = [it for it in path.iterdir() if it.name.endswith(self._suffix)]

    def __iter__(self):
        for model_path in self.models:
            yield InerialNetwork(model_path)

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return [InerialNetwork(model_path) for model_path in self.models[index]]
        return InerialNetwork(self.models[index])

    def __len__(self):
        return self.models.__len__()

    def get_by_names(self, names: list[str]):
        return [self.get_by_name(name) for name in names]

    def get_by_name(self, name: str):
        for model_path in self.models:
            if model_path.name.startswith(name):
                return InerialNetwork(model_path)
        raise ValueError(f"Model {name} not found.")

    def get_networks(self):
        return [InerialNetwork(model_path) for model_path in self.models]


class NetworkResult:
    step: int
    rate: int
    t_start_us: int

    meas_list: list[NDArray]
    meas_cov_list: list[NDArray]

    t_us: list[int]
    pose: Pose
    path: list[NDArray]
    pose_list: list[Pose]

    def __init__(
        self,
        suffix: str = "",
        *,
        step: int,
        rate: int,
        t_start_us: int,
        using_rerun: bool = True,
    ):
        self.suffix = suffix
        self.step = step
        self.rate = rate
        self.using_rerun = using_rerun
        self.interval_us = int(1e6 * step / rate)
        self.t_us = [t_start_us]

        self.meas_list = []
        self.meas_cov_list = []
        self.pose = Pose.identity()
        self.pose_list = []
        self.path = []

        if self.using_rerun:
            rre.log_coordinate(
                f"/world/network{self.suffix}",
                length=1,
                labels=[f"Network{self.suffix}"],
                show_labels=False,
            )

    def __len__(self):
        return len(self.meas_list)

    def __getitem__(self, index: int):
        return self.meas_list[index], self.meas_cov_list[index]

    def add(self, output: NetworkOutput):
        self.meas_list.append(output[0])
        self.meas_cov_list.append(output[1])
        self.pose.compose_trans_self(output[0] * self.interval_us / 1e6)
        self.pose_list.append(self.pose.copy())
        self.path.append(self.pose_list[-1].p)
        self.t_us.append(self.interval_us + self.t_us[-1])

        if self.using_rerun:
            rre.log_network_pose(
                self.t_us[-1], self.pose, self.path, suffix=self.suffix
            )

        return self.pose

    def to_csv(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        headers = [
            "#timestamp [us]",
            "p_RS_R_x [m]",
            "p_RS_R_y [m]",
            "p_RS_R_z [m]",
            "q_RS_w []",
            "q_RS_x []",
            "q_RS_y []",
            "q_RS_z []",
            "cov_x []",
            "cov_y []",
            "cov_z []",
        ]

        t_us = np.array(self.t_us[1:]).reshape(-1, 1)
        cov_arr = np.array(self.meas_cov_list)
        ps = np.array([pose.p for pose in self.pose_list])
        qs = np.array([pose.rot.as_quat(scalar_first=True) for pose in self.pose_list])
        assert len(t_us) == len(ps) == len(qs) == len(cov_arr), (
            f"Length mismatch: {t_us.shape}, {ps.shape}, {qs.shape}, {cov_arr.shape}"
        )
        data = np.hstack([t_us, ps, qs, cov_arr])
        pd.DataFrame(data, columns=headers).to_csv(  # type: ignore
            path, index=False, float_format="%.8f"
        )


class InerialNetworkData:
    step = 80
    rate = 200
    start_idx: int = 0
    rm_g: bool = False
    imu_block: NDArray[np.float32]
    BlockInput: Annotated

    def __init__(
        self,
        world_imu_data: ImuData,
        *,
        remove_gravity: bool | None = None,
    ) -> None:
        assert world_imu_data.frame == "global", "Imu not in global frame."
        if remove_gravity is not None:
            self.rm_g = remove_gravity
        if self.rm_g:
            world_imu_data.acce -= np.array([0, 0, 9.81])
        self.imu_block = np.hstack([world_imu_data.gyro, world_imu_data.acce])[
            self.start_idx :
        ]
        self.world_imu_data = world_imu_data[self.start_idx :]

        self.shape = (1, 6, self.rate)
        self.BlockInput = Annotated[NDArray[np.float32], Literal[1, 6, self.rate]]

    @classmethod
    def set_step(cls, step: int):
        cls.step = step
        return cls

    @classmethod
    def set_rate(cls, rate: int):
        cls.rate = rate
        return cls

    @classmethod
    def remove_gravity(cls):
        cls.rm_g = True
        return cls

    @classmethod
    def set_start_time(cls, t_s: float):
        cls.start_idx = int(t_s * cls.rate)
        return cls

    def get_block(self):
        self.bc = 0
        while self.bc + self.rate < len(self.imu_block):
            yield self.imu_block[self.bc : self.bc + self.rate].T.reshape(self.shape)
            self.bc += self.step

    def predict_using(self, net: InerialNetwork):
        t_start_us = self.world_imu_data.t_us[0]
        results = NetworkResult(
            net.name, step=self.step, rate=self.rate, t_start_us=t_start_us
        )
        for block in self.get_block():
            _pose = results.add(net.predict(block))
            print(f"{net.name} {self.bc:06d}: {_pose.p}")
        return results

    def predict_usings(self, networks: list[InerialNetwork]):
        t_start_us = self.world_imu_data.t_us[0]

        results = [
            NetworkResult(
                model.name, step=self.step, rate=self.rate, t_start_us=t_start_us
            )
            for model in networks
        ]
        for block in self.get_block():
            for i, net in enumerate(networks):
                _pose = results[i].add(net.predict(block))
                print(f"{net.name}-{self.bc}: {_pose.p}")

        return results


class DataRunner:
    def __init__(
        self,
        data: UnitData,
        Data: type[InerialNetworkData] = InerialNetworkData,
        rerun_init: bool = True,
    ):
        self.data = data
        world_imu_gt = data.imu_data.transform(data.gt_data.rots)
        self.in_data = Data(world_imu_gt)

        rre.rerun_init(data.name, imu_view_tags=["GT", "Raw"])
        rre.send_pose_data(data.gt_data)
        rre.send_imu_data(data.imu_data, tag="Raw")
        rre.send_imu_data(world_imu_gt, tag="GT")

    def predict(self, net: InerialNetwork):
        print("> Using Model:", net.name)
        results = self.in_data.predict_using(net)
        results.to_csv(f"results/{self.data.name}/{net.name}.csv")

        print(f"> Model {net.name} prediction completed.")

    def predict_batch(self, networks: list[InerialNetwork]):
        self.in_data.predict_usings(networks)
