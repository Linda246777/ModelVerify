# from base.device import DefaultDevice
import base.rerun_ext as rre
from base.datatype import UnitData
from base.model import InerialNetworkData, ModelLoader


def main():
    path = "/Users/qi/Resources/Models"
    data_path = "/Users/qi/Resources/Dataset/Compress001/001/2025_ABR-AL60_in/20251105_144535_ABR-AL60"
    # data_path = "/Users/qi/Codespace/Python/SpaceAlignment/dataset/001/20251031_01_in/Calibration/20251031_095725_SM-G9900"

    loader = ModelLoader(path)
    net0 = loader[0]

    data = UnitData(data_path)
    world_imu_gt = data.imu_data.transform(data.gt_data.rots)
    world_imu_ahrs = data.imu_data.transform()

    # Rerun View
    rre.rerun_init("IMU_GT", imu_view_tags=["AHRS", "GT", "Raw"])
    rre.set_world_tf(data.calib_data.tf_sg_global)
    rre.send_pose_data(data.gt_data)
    rre.send_imu_data(world_imu_ahrs, tag="AHRS")
    rre.send_imu_data(world_imu_gt, tag="GT")
    rre.send_imu_data(data.imu_data, tag="Raw")

    in_data_gt = InerialNetworkData(world_imu_gt)
    res0 = in_data_gt.predict_using(net0)


    res0.to_poses()

    exit()

    # Model Predict
    loader.models


if __name__ == "__main__":
    main()
