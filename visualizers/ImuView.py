#!/usr/bin/env python3
"""
基于惯性网络的运动估计系统

该脚本用于加载和运行预训练的惯性网络模型，对惯性测量单元(IMU)数据进行运动轨迹估计。

功能:
- 支持加载多个预训练模型进行批量预测
- 支持单设备数据(UnitData)和数据集(DeviceDataset)两种输入模式
- 可指定时间范围进行部分数据处理
- 可选择使用AHRS(姿态航向参考系统)或地面真值(ground truth)作为参考

参数说明:
- -m/--models: 指定要使用的模型名称列表
- --models_path: 指定模型文件夹路径
- --using_ahrs: 使用AHRS数据而非地面真值

作者: qi-xmu
版本: 1.0
"""

from base.args_parser import DatasetArgsParser
from base.datatype import DeviceDataset, ImuData, UnitData
from base.model import DataRunner, InertialNetworkData, ModelLoader
from base.rerun_ext import RerunView, send_imu_data


def main():
    dap = DatasetArgsParser()
    dap.parser.add_argument("-m", "--models", nargs="+", help="模型")
    dap.parser.add_argument("--models_path", type=str, help="模型文件夹")
    dap.parser.add_argument("--using_ahrs", action="store_true", default=False)
    dap.parse()

    if dap.unit:
        # 数据
        ud = UnitData(dap.unit)
        ud.load_data()

        RerunView().add_imu_view(visible=True, tags=["body", "global"]).send(ud.name)

        raw_imu_data = ud.imu_data

        global_imu_data = raw_imu_data.transform(ud.gt_data.rots)
        send_imu_data(raw_imu_data, "body")
        send_imu_data(global_imu_data, "global")


if __name__ == "__main__":
    main()
