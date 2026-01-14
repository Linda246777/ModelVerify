#!/usr/bin/env python3
"""
验证模型效果

验证单个模型在数据集上的效果。
"""

import pickle
from pathlib import Path

import base.rerun_ext as bre
from base.args_parser import DatasetArgsParser
from base.datatype import DeviceDataset, UnitData
from base.draw.CDF import plot_one_cdf
from base.evaluate import Evaluation
from base.model import DataRunner, InertialNetworkData, ModelLoader, NetworkResult

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

    res_dir = EvalDir / f"{nets[0].name}"
    res_dir.mkdir(parents=True, exist_ok=True)

    def action(ud: UnitData):
        print(f"> Eval {ud.name}")
        # 如果已经计算过
        obj_path = res_dir / "temp" / f"action_{ud.name}.pkl"
        obj_path.parent.mkdir(parents=True, exist_ok=True)
        if obj_path.exists():
            print(f"> 已存在结果：{obj_path}")
            with open(obj_path, "rb") as f:
                netres, evaluator = pickle.load(f)
                assert isinstance(netres, list)
                assert isinstance(evaluator, Evaluation)
                return netres, evaluator

        # 数据保存路径
        unit_dir = res_dir / ud.name
        unit_dir.mkdir(parents=True, exist_ok=True)

        # 加载数据
        ud.load_data(using_opt=True)
        # 可视化
        bre.RerunView().add_spatial_view().send(ud.name)
        bre.send_pose_data(ud.gt_data, "Groundtruth", color=[192, 72, 72])

        # 模型推理
        netres = DataRunner(ud, Data, has_init_rerun=True).predict_batch(nets)

        # 绘制 CDF
        model_cdf = Evaluation.get_cdf(netres[0].err_list, nets[0].name)
        plot_one_cdf(model_cdf, unit_dir / "CDF.png", show=False)

        # 计算 ATE
        evaluator = Evaluation(ud.gt_data, name=ud.name, rel_duration=1)
        evaluator.get_eval(netres[0].poses, f"{nets[0].name}_{ud.name}")
        evaluator.print()
        evaluator.save(unit_dir / "Eval.json")

        # 保存结果
        with open(obj_path, "wb") as f:
            pickle.dump((netres, evaluator), f)

        return netres, evaluator

    if dap.unit:
        ud = UnitData(dap.unit)
        netres, evaluator = action(ud)

    elif dap.dataset:
        dataset_path = dap.dataset
        datas = DeviceDataset(dataset_path)
        netres_list: list[NetworkResult] = []
        evaluator_list: list[Evaluation] = []
        for ud in datas:
            netres, evaluator = action(ud)
            netres_list.extend(netres)
            evaluator_list.append(evaluator)

        # 合并所有netres的误差项
        all_errors = []
        for netres in netres_list:
            assert isinstance(netres, NetworkResult)
            all_errors.extend(netres.err_list)

        model_cdf = Evaluation.get_cdf(all_errors, model_names[0])
        plot_one_cdf(model_cdf, res_dir / "CDF.png")


if __name__ == "__main__":
    main()
