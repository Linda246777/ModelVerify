#!/usr/bin/env python3
"""
验证模型效果

验证单个模型在数据集上的效果，生成CDF图和ATE/APE/RPE等评估指标。

用法:
    # 验证单个数据单元
    python VaildModel.py -u <unit_path> -m model_name

    # 验证整个数据集
    python VaildModel.py -d <dataset_path> -m model_name

    # 指定模型文件夹
    python VaildModel.py -u <unit_path> -m model_name --models_path /path/to/models

参数:
    -u, --unit: 指定单个数据单元路径
    -d, --dataset: 指定数据集路径
    -m, --models: 指定模型文件名（支持多个）
    --models_path: 指定模型文件夹路径（默认为"models"）

输出:
    - results/<model_name>/<unit_name>/: 单个单元的结果目录
      - CDF.png: 误差累积分布函数图
      - Eval.json: 评估指标（ATE/APE/RPE）
    - results/<model_name>_<device_name>/: 数据集结果目录
      - CDF.png: 整体误差CDF图
    - results/<model_name>/temp/: 临时结果缓存
"""

import pickle
from pathlib import Path

import base.rerun_ext as bre
from base.args_parser import DatasetArgsParser
from base.datatype import DeviceDataset, UnitData
from base.draw.CDF import plot_one_cdf
from base.evaluate import Evaluation
from base.model import DataRunner, InertialNetworkData, ModelLoader, NetworkResult

# 默认结果输出路径
EvalDir = Path("results")


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
        obj_path.parent.mkdir(parents=True, exist_ok=True)
        # 如果已经计算过
        if obj_path.exists():
            print(f"> 已存在结果：{obj_path}")
            with open(obj_path, "rb") as f:
                nr_list, evaluator = pickle.load(f)
                assert isinstance(nr_list, list)
                assert isinstance(evaluator, Evaluation)
        else:
            # 加载数据
            ud.load_data(using_opt=True)
            # 可视化
            bre.RerunView().add_spatial_view().send(ud.name)
            bre.send_pose_data(ud.gt_data, "Groundtruth", color=[192, 72, 72])

            # 模型推理
            dr = DataRunner(ud, Data, has_init_rerun=True)
            dr.test_scale = True
            nr_list = dr.predict_batch(nets)

            # 计算 ATE
            evaluator = Evaluation(ud.gt_data, name=ud.name, rel_duration=1)
            evaluator.get_eval(nr_list[0].poses, f"{nets[0].name}_{ud.name}")
            evaluator.print()

            # 保存结果
            with open(obj_path, "wb") as f:
                pickle.dump((nr_list, evaluator), f)

        # 绘制 CDF
        model_cdf = Evaluation.get_cdf(nr_list[0].err_list, nets[0].name)
        plot_one_cdf(model_cdf, unit_out_dir / "CDF.png", show=False)

        evaluator.save(unit_out_dir / "Eval.json")
        return nr_list, evaluator

    if dap.unit:
        unit_path = Path(dap.unit)
        # 使用 网络名称
        res_dir = EvalDir / f"{nets[0].name}"
        res_dir.mkdir(parents=True, exist_ok=True)

        ud = UnitData(unit_path)
        netres, evaluator = action(ud, res_dir)

    elif dap.dataset:
        dataset_path = Path(dap.dataset)
        datas = DeviceDataset(dataset_path)
        # 使用 网络名称 + 设备名称
        res_dir = EvalDir / f"{nets[0].name}_{datas.device_name}_scale"
        res_dir.mkdir(parents=True, exist_ok=True)
        # 存储结果
        netres_list: list[NetworkResult] = []
        for ud in datas:
            netres, evaluator = action(ud, res_dir)
            netres_list.extend(netres)

        # 合并所有netres的误差项
        all_errors = []
        for netres in netres_list:
            assert isinstance(netres, NetworkResult)
            all_errors.extend(netres.err_list)

        # 绘制总体的结果
        model_cdf = Evaluation.get_cdf(all_errors, nets[0].name)
        plot_one_cdf(model_cdf, res_dir / "CDF.png", show=False)


if __name__ == "__main__":
    main()
