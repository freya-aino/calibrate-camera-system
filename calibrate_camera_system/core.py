import os
import numpy as np
import json

from datetime import datetime
from collections import OrderedDict
from logging import getLogger
from time import sleep
from typing import List, Dict

from device_capture_system.datamodel import CameraDevice, FramePreprocessing
from device_capture_system.core import MultiInputStreamSender
from device_capture_system.fileIO import ImageSaver

from .datamodel import ModelExtrinsics, ModelIntrinsics

# --------------------- IMAGE IO ---------------------

class ImageManager:
    
    def __init__(
        self, 
        cameras: List[CameraDevice],
        calibration_data_path: str = "./calibration_data",
        frame_preprocessings: List[FramePreprocessing] = [],
        proxy_pub_port: int = 10000,
        proxy_sub_port: int = 10001):
        
        self.logger = getLogger(self.__class__.__name__)
        
        self.cameras = cameras
        image_save_path = os.path.join(calibration_data_path, "images")
        
        self.multi_stream_sender = MultiInputStreamSender(
            devices=cameras,
            proxy_sub_port=proxy_sub_port,
            proxy_pub_port=proxy_pub_port,
            zmq_proxy_queue_size=1,
            zmq_sender_queue_size=1,
            frame_preprocessings=frame_preprocessings)
        
        self.image_saver = ImageSaver(
            cameras=cameras,
            proxy_pub_port=proxy_pub_port,
            output_path=image_save_path,
            jpg_quality=100
        )
        
    def collect_images(self, num_images_to_collect: int, frame_collection_interval: float = 0.1):
        
        try:
            self.image_saver.start()
            self.multi_stream_sender.start_processes()
            
            collected_frames = 0
            error_frames = 0
            while collected_frames < num_images_to_collect:
                self.image_saver.save_image(datetime.now().timestamp())
                sleep(frame_collection_interval)
            
        except Exception as e:
            raise e
        finally:
            self.multi_stream_sender.stop_processes()
            self.image_saver.stop()

# --------------------- CAMERA TRANSFORMER ---------------------

class CameraTransformer:
    def __init__(
        self, 
        cameras: List[CameraDevice],
        camera_intrinsics: Dict[str, ModelIntrinsics],
        camera_extrinsics: List[ModelExtrinsics]):
        
        self.intrinsics = camera_intrinsics
        
        self.extrinsics = {}
        for extrinsic in camera_extrinsics:
            if extrinsic.from_camera not in self.extrinsics:
                self.extrinsics[extrinsic.from_camera] = {}
            self.extrinsics[extrinsic.from_camera][extrinsic.to_camera] = extrinsic
        
        self.projection_mats = {}
        for from_cam in cameras:
            self.projection_mats[from_cam.name] = {}
            for to_cam in cameras:
                extrinsic_transform = np.concatenate([self.extrinsics[from_cam.name][to_cam.name].rotation_mat, self.extrinsics[from_cam.name][to_cam.name].translation_vec], axis=1)
                self.projection_mats[from_cam.name][to_cam.name] = self.intrinsics[from_cam.name] @ extrinsic_transform
                
        
    def triangulate_point(self, points_2d: dict[str, np.ndarray], from_cam_name: str):
        
        # Create an empty list to store the equations
        equations = []
        
        # For each 2D point
        for to_cam_name, point in points_2d.items():
            
            # Get the corresponding projection matrix
            projection_mat = self.projection_mats[from_cam_name][to_cam_name]
            
            # Create the equations
            equations.append(point[0] * projection_mat[2, :] - projection_mat[0, :])
            equations.append(point[1] * projection_mat[2, :] - projection_mat[1, :])
        
        # Solve the system of equations using SVD
        # The solution is the last column of V
        # Convert the solution to homogeneous coordinates
        _, _, V = np.linalg.svd(np.vstack(equations))
        point3d = V[-1, :]
        point3d = point3d / point3d[3]
        return point3d[:3]
    
    def DLT(self, points_2d: dict[str, np.ndarray], main_cam_uuid: str):
        # direct linear transform
        
        assert len(points_2d) >= 2, "at least points from 2 cameras are required"
        
        # Get the number of points
        num_points = points_2d.values()[0][1].shape[0]
        
        # triangulate each point
        points_3d = []
        for i in range(num_points):
            point_2d = {cam_name: points[i] for cam_name, points in points_2d.items()}
            
            # Triangulate the 3D point and append to list
            point3d = self.triangulate_point(point_2d, main_cam_uuid)
            points_3d.append(point3d)
            
        return np.vstack(points_3d)