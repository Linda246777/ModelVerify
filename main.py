from base.args_parser import DatasetArgsParser
from base.datatype import DeviceDataset, UnitData
from base.model import DataRunner, InerialNetworkData, ModelLoader


def main():
    args = DatasetArgsParser()
    args.parser.add_argument(
        "-m",
        "--models",
        dest="models",
        nargs="+",
        required=True,
        help="使用的模型",
    )
    args.parser.add_argument("--using_ahrs", action="store_true", default=False)
    args.parse()

    models = args.args.models
    assert len(models) != 0, "model name is empty"

    models_path = "/Users/qi/Resources/Models"
    loader = ModelLoader(models_path)

    if args.unit:
        # 数据
        data = UnitData(args.unit)
        Data = InerialNetworkData.set_step(20)
        runner = DataRunner(data, Data, using_gt=not args.args.using_ahrs)
        runner.predict_batch(loader.get_by_names(models))

    if args.dataset:
        dataset_path = args.dataset
        datas = DeviceDataset(dataset_path)
        for data in datas:
            runner = DataRunner(data, InerialNetworkData.set_step(10))
            runner.predict_batch(loader.get_by_names(models))


if __name__ == "__main__":
    main()
