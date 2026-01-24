"""
使用如下结构存储数据

data.h5
├── metadata
│   ├── version               # 数据格式版本
│   ├── num_sequences         # 序列总数
│   ├── total_duration_hours  # 总时长
│   ├── readme                # H5文件结构说明（从本文件头部提取）
│   └── code                  # 本文件(H5Type.py)的完整代码
├── sequences
│   ├── sequence_001
│   │   ├── attributes
│   │   │   ├── num_rows: 12000
│   │   │   ├── t_start_us: 1609459200000000
│   │   │   ├── t_end_us: 1609459212000000
│   │   │   ├── frequency_hz: 200.0
│   │   │   └── label: "walking"  # 序列标签
│   │   └── resampled       # 重采样数据（时间轴对齐）
│   │       ├── t_us: [N]            # 统一时间戳（微秒）
│   │       ├── imu: [M, N, 6]       # M传感器, N时间步, 6特征(gyr(3), acc(3))
│   │       ├── mag: [M, N, 3]       # M传感器, N时间步, 3特征(mag(3))
│   │       ├── barom: [M, N, 2]     # M传感器, N时间步, 2特征(pressure, temp)
│   │       └── ground_truth: [N, 10] # pos(3), qwxyz(4), vel(3)
│   ├── sequence_002
│   └── ...
├── index_maps              # 预建索引（可选）
│   ├── train_index: [N, 2]  # [seq_idx, row_idx]
│   ├── val_index: [M, 2]
│   └── test_index: [K, 2]
└── sequence_lists          # 序列列表
    ├── train: ["sequence_001", "sequence_002", ...]
    ├── val: ["sequence_100", ...]
    └── test: ["sequence_200", ...]

"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np


@dataclass
class Metadata:
    """H5数据集元数据"""

    version: str
    num_sequences: int
    total_duration_hours: float
    readme: str = ""  # H5文件结构说明
    code: str = ""  # H5Type.py文件内容

    def create_group(self, group: h5py.Group):
        """
        将元数据写入 HDF5 Group

        Args:
            group: HDF5 Group 对象
        """
        group.create_dataset("version", data=self.version)
        group.create_dataset("num_sequences", data=self.num_sequences)
        group.create_dataset("total_duration_hours", data=self.total_duration_hours)
        group.create_dataset("readme", data=self.readme)
        group.create_dataset("code", data=self.code)

    @staticmethod
    def from_group(group: h5py.Group) -> "Metadata":
        """
        从 HDF5 Group 创建 Metadata 实例

        Args:
            group: HDF5 Group 对象，包含 metadata 数据集

        Returns:
            Metadata 实例
        """
        # 获取readme，如果不存在则使用默认值
        readme = ""
        if "readme" in group:
            readme_val = group["readme"][()]  # pyright: ignore[reportIndexIssue]
            if isinstance(readme_val, bytes):
                readme = readme_val.decode("utf-8")
            else:
                readme = str(readme_val)

        # 获取code，如果不存在则使用默认值
        code = ""
        if "code" in group:
            code_val = group["code"][()]  # pyright: ignore[reportIndexIssue]
            if isinstance(code_val, bytes):
                code = code_val.decode("utf-8")
            else:
                code = str(code_val)

        return Metadata(
            version=str(group["version"][()]),  # pyright: ignore[reportIndexIssue]
            num_sequences=int(group["num_sequences"][()]),  # pyright: ignore[reportIndexIssue, reportArgumentType]
            total_duration_hours=float(group["total_duration_hours"][()]),  # pyright: ignore[reportIndexIssue, reportArgumentType]
            readme=readme,
            code=code,
        )


@dataclass
class SequenceAttributes:
    """序列属性"""

    num_rows: int
    t_start_us: int
    t_end_us: int
    frequency_hz: float
    label: str = "unknown"  # 序列标签

    def create_group(self, group: h5py.Group):
        """
        将属性写入 HDF5 Group

        Args:
            group: HDF5 Group 对象
        """
        group.create_dataset("num_rows", data=self.num_rows)
        group.create_dataset("t_start_us", data=self.t_start_us)
        group.create_dataset("t_end_us", data=self.t_end_us)
        group.create_dataset("frequency_hz", data=self.frequency_hz)
        group.create_dataset("label", data=self.label)

    @staticmethod
    def from_group(group: h5py.Group) -> "SequenceAttributes":
        """
        从 HDF5 Group 创建 SequenceAttributes 实例

        Args:
            group: HDF5 Group 对象，包含 attributes 数据集

        Returns:
            SequenceAttributes 实例
        """
        # 获取label，如果不存在则使用默认值
        label = "unknown"
        if "label" in group:
            label_val = group["label"][()]  # pyright: ignore[reportIndexIssue]
            if isinstance(label_val, bytes):
                label = label_val.decode("utf-8")
            else:
                label = str(label_val)

        return SequenceAttributes(
            num_rows=int(group["num_rows"][()]),  # pyright: ignore[reportIndexIssue, reportArgumentType]
            t_start_us=int(group["t_start_us"][()]),  # pyright: ignore[reportIndexIssue, reportArgumentType]
            t_end_us=int(group["t_end_us"][()]),  # pyright: ignore[reportIndexIssue, reportArgumentType]
            frequency_hz=float(group["frequency_hz"][()]),  # pyright: ignore[reportIndexIssue, reportArgumentType]
            label=label,
        )


@dataclass
class ResampledData:
    """重采样数据（时间轴对齐）"""

    t_us: Optional[np.ndarray] = None  # [N] 统一时间戳（微秒）
    imu: Optional[np.ndarray] = (
        None  # [M, N, 6] M传感器, N时间步, 6特征(gyr(3), acc(3))
    )
    mag: Optional[np.ndarray] = None  # [M, N, 3] M传感器, N时间步, 3特征(mag(3))
    barom: Optional[np.ndarray] = (
        None  # [M, N, 2] M传感器, N时间步, 2特征(pressure, temp)
    )
    ground_truth: Optional[np.ndarray] = None  # [N, 10] pos(3), qwxyz(4), vel(3)

    def create_group(self, group: h5py.Group):
        """
        将数据写入 HDF5 Group

        Args:
            group: HDF5 Group 对象
        """
        # 时间戳
        if self.t_us is not None:
            group.create_dataset("t_us", data=self.t_us)

        # IMU数据
        if self.imu is not None:
            group.create_dataset("imu", data=self.imu)

        # 磁力计数据
        if self.mag is not None:
            group.create_dataset("mag", data=self.mag)

        # 气压计数据
        if self.barom is not None:
            group.create_dataset("barom", data=self.barom)

        # 真值数据
        if self.ground_truth is not None:
            group.create_dataset("ground_truth", data=self.ground_truth)

    @staticmethod
    def from_group(group: h5py.Group) -> "ResampledData":
        """
        从 HDF5 Group 创建 ResampledData 实例

        Args:
            group: HDF5 Group 对象，包含 resampled 数据集

        Returns:
            ResampledData 实例
        """
        t_us = None
        imu = None
        mag = None
        barom = None
        ground_truth = None

        # 加载数据
        for key in group.keys():
            data = group[key][()]  # pyright: ignore[reportIndexIssue]
            if not isinstance(data, np.ndarray):
                data = np.array(data)

            if key == "t_us":
                t_us = data
            elif key == "imu":
                # 提取时间戳并删除第一列
                if data.shape[-1] == 7:  # [M, N, 7] -> [M, N, 6]
                    imu = data[:, :, 1:]  # 删除 ts_us 列
                    # 从第一个传感器提取时间戳
                    if t_us is None and data.shape[0] > 0:
                        t_us = data[0, :, 0]
                else:
                    imu = data
            elif key == "mag":
                # 提取时间戳并删除第一列
                if data.shape[-1] == 4:  # [M, N, 4] -> [M, N, 3]
                    mag = data[:, :, 1:]  # 删除 ts_us 列
                    # 从第一个传感器提取时间戳
                    if t_us is None and data.shape[0] > 0:
                        t_us = data[0, :, 0]
                else:
                    mag = data
            elif key == "barom":
                # 提取时间戳并删除第一列
                if data.shape[-1] == 3:  # [M, N, 3] -> [M, N, 2]
                    barom = data[:, :, 1:]  # 删除 ts_us 列
                    # 从第一个传感器提取时间戳
                    if t_us is None and data.shape[0] > 0:
                        t_us = data[0, :, 0]
                else:
                    barom = data
            elif key == "ground_truth":
                # 提取时间戳并删除第一列
                if data.shape[-1] == 11:  # [N, 11] -> [N, 10]
                    ground_truth = data[:, 1:]  # 删除 ts_us 列
                    # 提取时间戳
                    if t_us is None:
                        t_us = data[:, 0]
                else:
                    ground_truth = data

        return ResampledData(
            t_us=t_us,
            imu=imu,
            mag=mag,
            barom=barom,
            ground_truth=ground_truth,
        )


@dataclass
class AlignedData:
    """旋转对齐数据（t_us, gyr(3), acc(3), pos(3), qwxyz(4), vel(3)）"""

    data: Optional[np.ndarray] = (
        None  # [N, 17] t_us, gyr(3), acc(3), pos(3), qwxyz(4), vel(3)
    )

    def create_group(self, group: h5py.Group):
        """
        将数据写入 HDF5 Group

        Args:
            group: HDF5 Group 对象
        """
        if self.data is not None:
            group.create_dataset("aligned", data=self.data)

    @staticmethod
    def from_group(group: h5py.Group) -> "AlignedData":
        """
        从 HDF5 Group 创建 AlignedData 实例

        Args:
            group: HDF5 Group 对象，包含 aligned 数据集

        Returns:
            AlignedData 实例
        """
        data = None
        if "aligned" in group:
            data = group["aligned"][()]  # pyright: ignore[reportIndexIssue]
            if not isinstance(data, np.ndarray):
                data = np.array(data)

        return AlignedData(data=data)


@dataclass
class Sequence:
    """单个序列数据"""

    name: str
    attributes: SequenceAttributes
    resampled: Optional[ResampledData] = None
    aligned: Optional[AlignedData] = None

    def get_window(
        self,
        start_idx: int,
        end_idx: int,
        keys: Optional[list[str]] = None,
        device_id: Optional[int] = 0,
    ) -> dict[str, np.ndarray]:
        """
        获取指定时间窗口的数据

        Args:
            start_idx: 窗口起始索引
            end_idx: 窗口结束索引
            keys: 要获取的数据键列表，默认为 ['imu', 'mag', 'barom', 'ground_truth']
            device_id: 指定传感器设备ID（M维度中的索引）
                      - 默认为0，获取第一个设备的数据
                      - 为None时，获取所有设备的数据

        Returns:
            包含窗口数据的字典，结构如下：
            - 'imu': 传感器数据，如果device_id不为None形状为 [6, window_size]，
                     否则为 [M, 6, window_size]（如果可用）
            - 'mag': 磁力计数据，如果device_id不为None形状为 [3, window_size]，
                     否则为 [M, 3, window_size]（如果可用）
            - 'barom': 气压计数据，如果device_id不为None形状为 [2, window_size]，
                       否则为 [M, 2, window_size]（如果可用）
            - 'ground_truth': 真值数据，形状为 [window_size, 10]（如果可用）
            - 'targ': 位移目标，从真值数据计算得到，形状为 [3]

            其中：
            - M: 传感器设备数量
            - window_size: 窗口大小（end_idx - start_idx）
            - imu features: 6维（陀螺仪3维 + 加速度计3维）
            - mag features: 3维（磁力计3维）
            - barom features: 2维（气压1维 + 温度1维）
            - ground_truth: 10维（位置3维 + 四元数4维 + 速度3维）
            - targ: 3维（位置位移 x, y, z）

        Raises:
            RuntimeError: 如果 resampled 数据不可用
        """
        if self.resampled is None:
            raise RuntimeError("Resampled data is not available")

        keys = keys or ["imu", "mag", "barom", "ground_truth"]
        result = {}

        for key in keys:
            data = getattr(self.resampled, key, None)
            if data is not None:
                # Slice the data window
                # For sensor data [M, N, features], slice along time dimension
                if key in ["imu", "mag", "barom"]:
                    # Shape: [M, N, features]
                    if device_id is not None:
                        # Select specific device: [N, features] -> [window_size, features]
                        windowed_data = data[device_id, start_idx:end_idx, :]
                        # Transpose to [features, window_size]
                        windowed_data = windowed_data.transpose(1, 0)
                    else:
                        # Select all devices: [M, N, features] -> [M, window_size, features]
                        windowed_data = data[:, start_idx:end_idx, :]
                        # Transpose to [M, features, window_size]
                        windowed_data = windowed_data.transpose(0, 2, 1)
                else:
                    # Shape: [N, features] -> [window_size, features]
                    windowed_data = data[start_idx:end_idx, :]

                result[key] = windowed_data

        # Calculate displacement (targ) from ground_truth
        # ground_truth format: [N, 10] -> pos(3), qwxyz(4), vel(3)
        assert self.resampled.ground_truth is not None
        gt = self.resampled.ground_truth
        # Get position at start and end
        pos_start = gt[start_idx, 0:3]  # [3]
        pos_end = gt[end_idx - 1, 0:3]  # [3]
        # Displacement = pos_end - pos_start
        targ = pos_end - pos_start
        result["targ"] = targ

        return result

    def create_group(self, parent_group: h5py.Group):
        """
        将序列数据写入 HDF5 Group

        Args:
            parent_group: 父 HDF5 Group 对象
        """
        seq_group = parent_group.create_group(self.name)

        # 创建属性
        attr_group = seq_group.create_group("attributes")
        self.attributes.create_group(attr_group)

        # 创建重采样数据
        if self.resampled is not None:
            resampled_group = seq_group.create_group("resampled")
            self.resampled.create_group(resampled_group)

        # 创建旋转对齐数据
        if self.aligned is not None:
            self.aligned.create_group(seq_group)

    @staticmethod
    def from_group(name: str, group: h5py.Group) -> "Sequence":
        """
        从 HDF5 Group 创建 Sequence 实例

        Args:
            name: 序列名称
            group: HDF5 Group 对象，包含序列数据

        Returns:
            Sequence 实例
        """
        # 加载属性
        attributes = SequenceAttributes.from_group(group["attributes"])  # pyright: ignore[reportArgumentType]

        # 加载重采样数据
        resampled = None
        if "resampled" in group:
            resampled = ResampledData.from_group(group["resampled"])  # pyright: ignore[reportArgumentType]

        # 加载旋转对齐数据
        aligned = None
        if "aligned" in group:
            aligned = AlignedData.from_group(group)  # pyright: ignore[reportArgumentType]

        return Sequence(
            name=name,
            attributes=attributes,
            resampled=resampled,
            aligned=aligned,
        )


@dataclass
class IndexMaps:
    """预建索引映射"""

    train_index: np.ndarray  # [N, 2] [seq_idx, row_idx]
    val_index: np.ndarray  # [M, 2]
    test_index: np.ndarray  # [K, 2]

    @staticmethod
    def from_group(group: h5py.Group) -> "IndexMaps":
        """
        从 HDF5 Group 创建 IndexMaps 实例

        Args:
            group: HDF5 Group 对象，包含 index_maps 数据集

        Returns:
            IndexMaps 实例
        """
        train_index = np.array(group["train_index"][()])  # pyright: ignore[reportIndexIssue]
        val_index = np.array(group["val_index"][()])  # pyright: ignore[reportIndexIssue]
        test_index = np.array(group["test_index"][()])  # pyright: ignore[reportIndexIssue]

        return IndexMaps(
            train_index=train_index, val_index=val_index, test_index=test_index
        )


@dataclass
class SequenceLists:
    """序列列表"""

    train: List[str]
    val: List[str]
    test: List[str]

    @staticmethod
    def from_group(group: h5py.Group) -> "SequenceLists":
        """
        从 HDF5 Group 创建 SequenceLists 实例

        Args:
            group: HDF5 Group 对象，包含 sequence_lists 数据集

        Returns:
            SequenceLists 实例
        """

        def decode_list(data: np.ndarray) -> List[str]:
            """解码字符串数组"""
            result = []
            for item in data:
                if isinstance(item, bytes):
                    result.append(item.decode("utf-8"))
                else:
                    result.append(str(item))
            return result

        train = decode_list(group["train"][()])  # pyright: ignore[reportIndexIssue, reportArgumentType]
        val = decode_list(group["val"][()])  # pyright: ignore[reportIndexIssue, reportArgumentType]
        test = decode_list(group["test"][()])  # pyright: ignore[reportIndexIssue, reportArgumentType]

        return SequenceLists(train=train, val=val, test=test)


class H5Dataset:
    """H5数据集读写类"""

    def __init__(self, file_path: Path | str, mode: str = "r"):
        """
        初始化H5数据集

        Args:
            file_path: H5文件路径
            mode: 打开模式 ('r', 'w', 'a', 'r+')
        """
        assert mode in ["r", "w", "a", "r+"], f"Invalid mode: {mode}"
        self.file_path = file_path
        self.mode = mode
        self._file: Optional[h5py.File] = None
        self.metadata: Optional[Metadata] = None
        self._sequences: Dict[str, Sequence] = {}
        self._index_maps: Optional[IndexMaps] = None
        self._sequence_lists: Optional[SequenceLists] = None

    def __enter__(self):
        """上下文管理器入口"""
        self._file = h5py.File(self.file_path, self.mode)
        if self.mode == "r" or self.mode == "r+":
            self._load_metadata()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        if self._file is not None:
            self._file.close()
            self._file = None

    def _load_metadata(self):
        """从H5文件加载元数据"""
        if self._file is None:
            raise RuntimeError("H5 file not opened")

        meta_group = self._file.get("metadata")
        if meta_group is not None:
            assert isinstance(meta_group, h5py.Group)
            self.metadata = Metadata.from_group(meta_group)

    def create_metadata(self, metadata: Metadata):
        """创建元数据"""
        if self._file is None:
            raise RuntimeError("H5 file not opened")

        meta_group = self._file.create_group("metadata")
        metadata.create_group(meta_group)
        self.metadata = metadata

    def create_sequence(self, sequence: Sequence):
        """创建序列数据"""
        if self._file is None:
            raise RuntimeError("H5 file not opened")

        # 获取或创建 sequences 组
        if "sequences" not in self._file:
            sequences_group = self._file.create_group("sequences")
        else:
            sequences_group = self._file["sequences"]

        # 使用 Sequence 类的 create_group 方法
        sequence.create_group(sequences_group)  # pyright: ignore[reportArgumentType]

        self._sequences[sequence.name] = sequence

    def load_sequence(self, sequence_name: str) -> Sequence:
        """从H5文件加载序列"""
        if self._file is None:
            raise RuntimeError("H5 file not opened")

        seq_group = self._file[f"sequences/{sequence_name}"]
        assert isinstance(seq_group, h5py.Group)
        sequence = Sequence.from_group(sequence_name, seq_group)
        self._sequences[sequence_name] = sequence
        return sequence

    def get_sequence(self, sequence_name: str) -> Optional[Sequence]:
        """获取序列(优先从缓存)"""
        if sequence_name in self._sequences:
            return self._sequences[sequence_name]
        return self.load_sequence(sequence_name)

    def create_index_maps(self):
        """
        根据已加载的sequence_lists和sequences自动创建索引映射

        Returns:
            IndexMaps对象

        Raises:
            RuntimeError: 如果文件未打开或sequence_lists未设置
        """
        if self._file is None:
            raise RuntimeError("H5 file not opened")

        if self._sequence_lists is None:
            raise RuntimeError(
                "sequence_lists not set. Call create_sequence_lists() first."
            )

        def create_indices(sequence_names: List[str]) -> np.ndarray:
            """为给定的序列名称列表创建索引"""
            indices = []
            for seq_name in sequence_names:
                if seq_name not in self._sequences:
                    continue
                seq = self._sequences[seq_name]
                num_rows = seq.attributes.num_rows
                # 获取序列在文件中的索引
                all_seq_names = self.list_sequences()
                seq_idx = all_seq_names.index(seq_name)

                for row_idx in range(num_rows):
                    indices.append([seq_idx, row_idx])

            return np.array(indices, dtype=np.int32)

        train_index = create_indices(self._sequence_lists.train)
        val_index = create_indices(self._sequence_lists.val)
        test_index = create_indices(self._sequence_lists.test)

        index_maps = IndexMaps(
            train_index=train_index, val_index=val_index, test_index=test_index
        )

        # 保存到文件
        index_group = self._file.create_group("index_maps")
        index_group.create_dataset("train_index", data=index_maps.train_index)
        index_group.create_dataset("val_index", data=index_maps.val_index)
        index_group.create_dataset("test_index", data=index_maps.test_index)

        self._index_maps = index_maps
        return index_maps

    def load_index_maps(self) -> IndexMaps:
        """加载索引映射"""
        if self._file is None:
            raise RuntimeError("H5 file not opened")

        index_group = self._file["index_maps"]
        assert isinstance(index_group, h5py.Group)

        self._index_maps = IndexMaps.from_group(index_group)
        return self._index_maps

    def create_sequence_lists(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle: bool = True,
        random_seed: int = 42,
    ) -> SequenceLists:
        """
        根据已加载的sequences自动创建序列列表分割

        Args:
            train_ratio: 训练集比例 (默认: 0.7)
            val_ratio: 验证集比例 (默认: 0.15)
            test_ratio: 测试集比例 (默认: 0.15)
            shuffle: 是否打乱顺序 (默认: True)
            random_seed: 随机种子 (默认: 42)

        Returns:
            SequenceLists对象

        Raises:
            RuntimeError: 如果文件未打开或没有已加载的序列
        """
        if self._file is None:
            raise RuntimeError("H5 file not opened")

        if len(self._sequences) == 0:
            raise RuntimeError("No sequences loaded. Call create_sequence() first.")

        # 获取所有序列名称
        sequence_names = list(self._sequences.keys())

        # 验证比例
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=1e-6):
            raise ValueError(
                f"Ratios must sum to 1.0: train={train_ratio}, val={val_ratio}, test={test_ratio}"
            )

        if shuffle:
            # 设置随机种子并打乱
            rng = np.random.default_rng(random_seed)
            shuffled_names = rng.permutation(sequence_names).tolist()
        else:
            shuffled_names = sequence_names.copy()

        # 计算分割点
        total = len(shuffled_names)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        # 分割列表
        train = shuffled_names[:train_end]
        val = shuffled_names[train_end:val_end]
        test = shuffled_names[val_end:]

        sequence_lists = SequenceLists(train=train, val=val, test=test)

        # 保存到文件
        dt = h5py.string_dtype(encoding="utf-8")
        list_group = self._file.create_group("sequence_lists")

        list_group.create_dataset(
            "train", data=np.array(sequence_lists.train, dtype=dt)
        )
        list_group.create_dataset("val", data=np.array(sequence_lists.val, dtype=dt))
        list_group.create_dataset("test", data=np.array(sequence_lists.test, dtype=dt))

        self._sequence_lists = sequence_lists
        return sequence_lists

    def load_sequence_lists(self) -> SequenceLists:
        """加载序列列表"""
        if self._file is None:
            raise RuntimeError("H5 file not opened")

        list_group = self._file["sequence_lists"]
        assert isinstance(list_group, h5py.Group)
        self._sequence_lists = SequenceLists.from_group(list_group)
        return self._sequence_lists

    def list_sequences(self) -> List[str]:
        """列出所有序列名称"""
        if self._file is None:
            raise RuntimeError("H5 file not opened")

        if "sequences" not in self._file:
            return []

        return list(self._file["sequences"].keys())  # pyright: ignore[reportAttributeAccessIssue]

    @property
    def sequences(self) -> Dict[str, Sequence]:
        """获取所有已加载的序列"""
        return self._sequences

    @property
    def index_maps(self) -> Optional[IndexMaps]:
        """获取索引映射"""
        return self._index_maps

    @property
    def sequence_lists(self) -> Optional[SequenceLists]:
        """获取序列列表"""
        return self._sequence_lists
