import numpy as np

from calibrate_camera_system.core import CameraTransformer

from device_capture_system.deviceIO import load_all_devices_from_config


if __name__ == '__main__':

    cameras = load_all_devices_from_config("video", config_file="./devices.json")
    ct = CameraTransformer(cameras)
    
    test_points = {
        "cam0": np.array([[550, 400], [550, 500], [550, 600], [550, 700]]),
        "cam1": np.array([[600, 350], [600, 450], [600, 550], [600, 650]]),
        "cam2": np.array([[650, 300], [650, 400], [650, 500], [650, 600]]),
    }
    
    print(ct.DLT(test_points, main_cam_uuid="cam0"))
