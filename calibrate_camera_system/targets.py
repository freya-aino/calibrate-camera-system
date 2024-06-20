import os
import json
import cv2
import numpy as np

from tqdm import tqdm
from typing import List, Union, Dict
from logging import getLogger
from device_capture_system.datamodel import CameraDevice
from multiprocessing import Pool

from .datamodel import TargetData, TargetDataset, TargetParameters

class TargetManager:
    
    def __init__(self, target_parameters: TargetParameters, targets_save_path: str, images_save_path: str):
        
        self.logger = getLogger(self.__class__.__name__)
        self.targets_save_path = targets_save_path
        self.images_save_path = images_save_path
        self.target_parameters = target_parameters
    
    @staticmethod
    def get_target_corners_from_image(image_dir_path: str, image_name: str, target_params: TargetParameters, subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)):
        frame = cv2.imread(os.path.join(image_dir_path, image_name))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, target_params.num_target_corners_wh, None, flags = cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH)
        
        if not ret_corners:
            return None
        
        target_points = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria).squeeze()
        
        
        # check if the corners are flipped, remvoe if true
        top_left_corner = target_points[0]
        top_right_corner = target_points[target_params.num_target_corners_wh[0] - 1]
        bottom_left_corner = target_points[(target_params.num_target_corners_wh[1] - 1) * target_params.num_target_corners_wh[0]] 
        
        left_right_vec = top_right_corner - top_left_corner
        top_bottom_vec = bottom_left_corner - top_left_corner
        
        # if left_right_vec[0] < 0 or top_bottom_vec[1] > 0:
        #     return None
        
        if np.cross(left_right_vec, top_bottom_vec) < 0:
            return None
        
        return TargetData(image_name=image_name, target_points=target_points)
    
    def extract_target_data_from_images(self, cameras: List[CameraDevice], multiprocessing_workser: int = 16):
        process_pool = Pool(multiprocessing_workser)
        
        process_results = {camera.name: [] for camera in cameras}
        
        self.logger.info("extracting target data from images ...")
        try:
            
            for camera in cameras:
                
                image_path = os.path.join(self.images_save_path, camera.name)
                assert os.path.exists(image_path), f"images path {image_path} does not exist"
                
                for image_name in os.listdir(image_path):
                    result = process_pool.apply_async(TargetManager.get_target_corners_from_image, args=(image_path, image_name, self.target_parameters))
                    process_results[camera.name].append(result)
                
            
            results = {}
            for cam_name in process_results:
                results[cam_name] = [result.get() for result in tqdm(process_results[cam_name], desc=f"extracting targets for {cam_name}")] 
                results[cam_name] = [res for res in results[cam_name] if res is not None and res.target_points.shape[0] == np.prod(self.target_parameters.num_target_corners_wh)]
            
        except:
            raise
        finally:
            self.logger.info("closing process pool ...")
            process_pool.close()
            for cam_name in process_results:
                for result in process_results[cam_name]:
                    result.wait()
            process_pool.join()
        
        for cam_name in results:
            self.logger.info(f"camera {cam_name} extracted {len(results[cam_name])} targets")
        
        return [
            TargetDataset(camera=cameras[i], target_data=res)
            for i, res in enumerate(results.values())
        ]