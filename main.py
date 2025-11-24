import base.rerun_ext as rre
from base.args_parser import DatasetArgsParser
from base.datatype import DeviceDataset, UnitData
from base.model import DataRunner, InerialNetwork, InerialNetworkData, ModelLoader
from TLIOView import TLIOData


def tlio_view(path: str, net: InerialNetwork):
    path = (
        "/Users/qi/Codespace/Python/SpaceAlignment/dataset/1/20251031_101025_SM-G9900"
    )
    td = TLIOData(path)
    rre.rerun_init("IMU_GT", imu_view_tags=["GT"])
    rre.send_pose_data(td.gt_data)
    rre.send_imu_data(td.imu_data, tag="GT")
    in_data = InerialNetworkData(td.imu_data)
    in_data.predict_using(net)


def main():
    args = DatasetArgsParser()
    args.parse()

    models_path = "/Users/qi/Resources/Models"
    loader = ModelLoader(models_path)

    if args.unit:
        # 数据
        data = UnitData(args.unit)
        Data = InerialNetworkData.set_step(10)
        runner = DataRunner(data, Data)
        # runner.predict(loader.get_by_name("StarIO"))
        # runner.predict_batch(loader)
        # runner.predict(loader.get_by_name("model_huawei"))
        runner.predict_batch(loader.get_by_names(["model_huawei", "ZZH"]))

    if args.dataset:
        dataset_path = args.dataset
        datas = DeviceDataset(dataset_path)
        for data in datas:
            runner = DataRunner(data, InerialNetworkData.set_step(10))
            runner.predict(loader.get_by_name("model_huawei"))


if __name__ == "__main__":
    main()
