import numpy as np

from core import CameraTransformer
from camera_capture_system.core import load_all_cameras_from_config


if __name__ == '__main__':

    cameras = load_all_cameras_from_config("./cameras_configs.json")
    ct = CameraTransformer(cameras)
    
    test_points = {
        "cam0": np.array([[550, 400], [550, 500], [550, 600], [550, 700]]),
        "cam1": np.array([[600, 350], [600, 450], [600, 550], [600, 650]]),
        "cam2": np.array([[650, 300], [650, 400], [650, 500], [650, 600]]),
    }
    
    print(ct.DLT(test_points, main_cam_uuid="cam0"))
