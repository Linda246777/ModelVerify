#!/usr/bin/env python3
"""
数据集成分分析

分析数据集的组成成分，评估模型在数据集上的表现。

用法:
    # 分析单个数据单元
    uv run python validators/DatasetAnalysis.py -u <unit_path> -m model_name

    # 分析整个数据集
    uv run python validators/DatasetAnalysis.py -d <dataset_path> -m model_name

    # 指定模型文件夹
    uv run python validators/DatasetAnalysis.py -d <dataset_path> -m model_name --models_path /path/to/models

参数:
    -u, --unit: 指定单个数据单元路径
    -d, --dataset: 指定数据集路径
    -m, --models: 指定模型文件名（支持多个）
    --models_path: 指定模型文件夹路径（默认为"models"）

输出:
    - results/<model_name>/<unit_name>/: 单个单元的结果目录
      - CDF.png: 误差累积分布函数图
      - Eval.json: 评估指标
    - results/<model_name>_<device_name>/: 数据集结果目录
      - 综合分析结果
"""

from pathlib import Path

import base.rerun_ext as bre
from base.analysis import DatasetAnalysis, UnitAnalysis
from base.args_parser import DatasetArgsParser
from base.datatype import DeviceDataset, UnitData
from base.draw.CDF import plot_one_cdf
from base.evaluate import Evaluation, get_cdf_from_err
from base.model import DataRunner, InertialNetworkData, ModelLoader
from base.obj import Obj

# 默认结果输出路径
EvalDir = Path("/Users/qi/Resources/results")


def main():
    dap = DatasetArgsParser()
    dap.parser.add_argument("-m", "--models", nargs="+", help="模型")
    dap.parser.add_argument("--models_path", type=str, help="模型文件夹")
    dap.parse()

    # regen_fusion = dap.regen
    model_names = dap.args.models
    if model_names is None or len(model_names) == 0:
        model_names = ["model_resnet_0111_96"]

    models_path = "models"
    if dap.args.models_path is not None:
        models_path = dap.args.models_path

    loader = ModelLoader(models_path)
    Data = InertialNetworkData.set_step(20)
    nets = loader.get_by_names(model_names)

    def action(ud: UnitData, res_dir: Path):
        print(f"> Eval {ud.name}")
        # 数据保存路径
        unit_out_dir = res_dir / ud.name
        unit_out_dir.mkdir(parents=True, exist_ok=True)

        obj_path = res_dir / "temp" / f"action_{ud.name}.pkl"
        print(f"> 存储路径：{obj_path}")
        obj_path.parent.mkdir(parents=True, exist_ok=True)
        # 如果已经计算过
        if obj_path.exists():
            print(f"> 已存在结果：{obj_path}")
            netres, evaluator = Obj.load(obj_path)
            assert isinstance(netres, list)
            assert isinstance(evaluator, Evaluation)
        else:
            # 加载数据
            ud.load_data(using_opt=True)
            # 可视化
            bre.RerunView().add_spatial_view().send(ud.name)
            bre.send_pose_data(ud.gt_data, "Groundtruth", color=[192, 72, 72])

            # 模型推理
            netres = DataRunner(ud, Data, has_init_rerun=True).predict_batch(nets)

            # 计算 ATE
            evaluator = Evaluation(ud.gt_data, name=ud.name, rel_duration=1)
            evaluator.get_eval(netres[0].poses, f"{nets[0].name}_{ud.name}")
            evaluator.print()

            # 保存结果
            Obj.save((netres, evaluator), obj_path)

        # 绘制 CDF
        model_cdf = get_cdf_from_err(netres[0].err_list, nets[0].name)
        plot_one_cdf(model_cdf, unit_out_dir / "CDF.png", show=False)

        evaluator.save(unit_out_dir / "Eval.json")

        # 分析数据集
        ua = UnitAnalysis(ud.name, netres[0].gt_list)
        ua.analyze(res_dir / ud.name)

        return ua

    if dap.unit:
        unit_path = Path(dap.unit)
        # 使用 网络名称
        res_dir = EvalDir / f"{nets[0].name}"
        res_dir.mkdir(parents=True, exist_ok=True)

        ud = UnitData(unit_path)
        action(ud, res_dir)

    elif dap.dataset:
        dataset_path = Path(dap.dataset)
        datas = DeviceDataset(dataset_path)
        # 使用 网络名称 + 设备名称
        res_dir = EvalDir / f"{nets[0].name}_{datas.device_name}"
        res_dir.mkdir(parents=True, exist_ok=True)
        # 存储结果
        da_path = res_dir / f"{DatasetAnalysis._obj_name}.pkl"

        if da_path.exists():
            da = DatasetAnalysis.load(da_path)
        else:
            da = DatasetAnalysis()
            for ud in datas:
                ua = action(ud, res_dir)
                da.add(ua)
            da.save(res_dir / f"{da._obj_name}.pkl")

        da.analyze(res_dir, datas.device_name)


if __name__ == "__main__":
    main()
