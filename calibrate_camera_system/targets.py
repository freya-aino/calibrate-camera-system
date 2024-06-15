import os
import json
import cv2
import numpy as np

from typing import List
from logging import getLogger
from device_capture_system.datamodel import CameraDevice
from multiprocessing import Pool

from .datamodel import CalibrationTarget, TargetData
from .core import NumpyJsonCodec

# def remove_images_without_targets(self):
#     valid_targets = {}
#     for cam in self.cameras:
#         self.logger.info(f"removing images without targets for {cam.name} ...")
#         with open(os.path.join(self.calibration_save_path, f"{cam.name}.json"), "r") as f:
#             all_targets = json.load(f, object_hook=decode_dict)
#             for k in all_targets:
#                 if all_targets[k] is None:
#                     logger.info(f"{cam_uuid} :: {k} has no target, removing ...")
#                     os.remove(k)
#                 else:
#                     valid_targets[k] = all_targets[k]
#         # save updated list of targets
#         with open(os.path.join("calibration_targets", f"{cam_uuid}.json"), "w") as f:
#             json.dump(valid_targets, f, cls=NumpyEncoder)

class TargetManager:
    
    def __init__(self, cameras: list[CameraDevice], calibration_data_path: str = "./calibration_data"):
        self.logger = getLogger(self.__class__.__name__)
        
        self.cameras = cameras
        self.calibration_data_path = calibration_data_path
        self.calibration_target = CalibrationTarget.load_from_file(os.path.join(calibration_data_path, "calibration_targets", "target-checkerboard.json"))
    
    def load_extracted_targets(self) -> dict[str, List[TargetData]]:
        file_path = os.path.join(self.calibration_data_path, "extracted_targets.json")
        assert os.path.exists(file_path), f"no target data found at {file_path}"
        with open(file_path, "r") as f:
            raw = json.load(f, object_hook=NumpyJsonCodec.decode_dict)
        return {cam_name: [TargetData(**data) for data in data_list] for cam_name, data_list in raw.items()}
    
    def save_extracted_targets(self, targets: dict[str, List[TargetData]]):
        file_path = os.path.join(self.calibration_data_path, "extracted_targets.json")
        with open(file_path, "w") as f:
            targets = {cam_name: data.model_demp() for cam_name, data in targets.items()}
            json.dump(targets, f, cls=NumpyJsonCodec)
    
    @staticmethod
    def get_target_corners_from_image(images_path: str, cam_name: str, image_name: str, target: CalibrationTarget, subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)):
        image_uri = os.path.join(images_path, cam_name, image_name)
        frame = cv2.imread(image_uri)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, target.num_target_corners_wh, None)
        
        if not ret_corners:
            return None
        target_points = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria).squeeze()
        return TargetData(image_name=image_name, camera_name=cam_name, target_points=target_points)
    
    def extract_target_data_from_images(self, num_workers: int = 8):
        process_pool = Pool(num_workers)
        process_results = []
        
        try:
            
            for cam in self.cameras:
                images_path = os.path.join(self.calibration_data_path, "images")
                
                for image_name in os.listdir(os.path.join(images_path, cam.name)):
                    result = process_pool.apply_async(TargetManager.get_target_corners_from_image, args=(images_path, cam.name, image_name, self.calibration_target))
                    process_results.append(result)
                
            results = [result.get() for result in process_results]
            
            cam_wise_results = {cam.name: [] for cam in self.cameras}
            for res in results:
                cam_wise_results[res.camera_name].append(res)
            
        except:
            raise
        finally:
            process_pool.close()
            for result in process_results:
                result.wait()
            process_pool.join()
        
        return cam_wise_results