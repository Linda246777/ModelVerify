from pathlib import Path
from typing import Annotated, Literal, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray

import base.rerun_ext as rre

from .datatype import ImuData, Pose
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
        # print(output)
        meas: NDArray = output[0].cpu().detach().numpy().flatten()
        meas_cov: NDArray = output[1].cpu().detach().numpy().flatten()
        return meas, meas_cov


class ModelLoader:
    _suffix = ".pt"
    models: list = []

    def __init__(self, base_dir: Path | str):
        path = Path(base_dir)
        self.models = [it for it in path.iterdir() if it.name.endswith(self._suffix)]
        self.__len__ = self.models.__len__

    def __iter__(self):
        for model_path in self.models:
            yield InerialNetwork(model_path)

    def __getitem__(self, index: int):
        return InerialNetwork(self.models[index])


class NetworkResult:
    step: int
    rate: int
    t_start_us: int

    meas_list: list[NDArray] = []
    meas_cov_list: list[NDArray] = []

    t_now_s: float
    pose: Pose = Pose.identity()
    path: list[NDArray] = []

    def __init__(
        self, *, step: int, rate: int, t_start_us: int, using_rerun: bool = True
    ):
        self.step = step
        self.rate = rate
        self.using_rerun = using_rerun
        self.interval = step / rate
        self.t_now_s = t_start_us / 1e6

        if self.using_rerun:
            rre.log_coordinate(
                "/world/network", length=1, labels=["Network"], show_labels=True
            )

    def __len__(self):
        return len(self.meas_list)

    def __getitem__(self, index: int):
        return self.meas_list[index], self.meas_cov_list[index]

    def add(self, output: NetworkOutput):
        self.meas_list.append(output[0])
        self.meas_cov_list.append(output[1])
        self.pose.compose_trans_self(output[0] * self.interval)
        self.path.append(self.pose.trans.copy())
        self.t_now_s += self.interval

        if self.using_rerun:
            rre.log_network_pose(self.t_now_s, self.pose, self.path)

        return self.pose

    def to_poses(self):
        meas_arr = np.array(self.meas_list)
        velocity = meas_arr * self.interval
        trans = np.cumsum(velocity, axis=0)
        return trans


class InerialNetworkData:
    step = 10
    rate = 200
    imu_block: NDArray[np.float32]
    BlockInput: Annotated

    def __init__(self, world_imu_data: ImuData) -> None:
        assert world_imu_data.frame == "global", "Imu not in global frame."
        self.imu_block = np.hstack([world_imu_data.gyro, world_imu_data.acce])
        self.world_imu_data = world_imu_data

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

    def get_block(self):
        self.bc = 0
        while self.bc + self.rate < len(self.imu_block):
            yield self.imu_block[self.bc : self.bc + self.rate].T.reshape(self.shape)
            self.bc += self.step

    def predict_using(
        self,
        net: InerialNetwork,
    ):
        results = NetworkResult(
            step=self.step, rate=self.rate, t_start_us=self.world_imu_data.t_us[0]
        )
        for block in self.get_block():
            pose = results.add(net.predict(block))
            print(f"{self.bc}: {pose.trans}")
        return results
