import os
import numpy as np
import cv2

from copy import deepcopy
from datetime import datetime
from logging import getLogger
from time import sleep
from typing import List

from device_capture_system.datamodel import CameraDevice, FramePreprocessing
from device_capture_system.core import MultiInputStreamSender
from device_capture_system.fileIO import ImageSaver

from .datamodel import CameraModelInformation, Intrinsics, TargetDataset

# --------------------- IMAGE IO ---------------------

class ImageManager:
    
    def __init__(
        self, 
        cameras: List[CameraDevice],
        image_save_path: str,
        frame_preprocessings: List[FramePreprocessing] = [],
        proxy_pub_port: int = 10000,
        proxy_sub_port: int = 10001):
        
        self.logger = getLogger(self.__class__.__name__)
        
        self.image_save_path = image_save_path
        
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
        
    def remove_images_without_targets(self, target_datasets: List[TargetDataset]):
        for target_dataset in target_datasets:
            valid_image_names = {target_data.image_name: None for target_data in target_dataset.target_data}
            target_ds_folder_path = os.path.join(self.image_save_path, target_dataset.camera.name)
            for image_name in os.listdir(target_ds_folder_path):
                if image_name in valid_image_names:
                    continue
                os.remove(os.path.join(target_ds_folder_path, image_name))
                self.logger.info(f"removed image without target: {target_dataset.camera.name}/{image_name}")
        
    def collect_images(self, num_images_to_collect: int, frame_collection_interval: float = 0.1, max_error_frames: int = 25):
        
        try:
            self.image_saver.start()
            self.multi_stream_sender.start_processes()
            
            collected_frames = 0
            error_frames = 0
            while collected_frames < num_images_to_collect:
                
                correct = self.image_saver.save_image(datetime.now().timestamp())
                sleep(frame_collection_interval)
                
                if not correct:
                    error_frames += 1
                    self.logger.warning(f"issus saving frame exeting in {max_error_frames - error_frames} frames ...")
                    if error_frames > max_error_frames:
                        self.logger.error("too many errors, stopping image collection")
                        break
                    continue
                collected_frames += 1
                error_frames = 0
            
        except Exception as e:
            raise e
        finally:
            self.multi_stream_sender.stop_processes()
            self.image_saver.stop()

# --------------------- CAMERA TRANSFORMER ---------------------


class CameraModel:
    def __init__(self, camera_model_information: CameraModelInformation):
        self.logger = getLogger(self.__class__.__name__)
        
        # put intrinsics and extrinsics in a dictionary for easy access
        self.intrinsics = {I.camera.name: I for I in camera_model_information.intrinsics}
        self.extrinsics = {}
        for extrinsic in camera_model_information.extrinsics:
            cam_from_name = extrinsic.camera_from.name
            cam_to_name = extrinsic.camera_to.name
            if cam_from_name not in self.extrinsics:
                self.extrinsics[cam_from_name] = {}
            self.extrinsics[cam_from_name][cam_to_name] = extrinsic
        
        # # calculate projection matrices
        # self.projection_matrices = {}
        # for cam_from_name in self.ext:
        #     self.projection_matrices[cam_from_name] = {}
        #     for cam_to_name in self.ext[cam_from_name]:
        #         rot_mat = self.ext[cam_from_name][cam_to_name].rotation_matrix
        #         trans_vec = self.ext[cam_from_name][cam_to_name].translation_vector.reshape(3, 1)
        #         self.projection_matrices[cam_from_name][cam_to_name] = self.intrinsics[cam_from_name].calibration_matrix @ np.concatenate([rot_mat, trans_vec], axis=1)
        
    # def undistort_points(self, points: np.ndarray, cam_name: str):
    #     intrinsics = self.intrinsics[cam_name]
        
    #     normed_points = cv2.undistortPoints(src = points, cameraMatrix = intrinsics.calibration_matrix, distCoeffs = intrinsics.distortion_coefficients)
    #     normed_points = normed_points.squeeze()
        
    #     f_x, f_y = intrinsics.calibration_matrix[0, 0], intrinsics.calibration_matrix[1, 1]
    #     c_x, c_y = intrinsics.calibration_matrix[0, 2], intrinsics.calibration_matrix[1, 2]
        
    #     pixel_points = np.zeros_like(normed_points)
    #     pixel_points[:, 0] = normed_points[:, 0] * f_x + c_x
    #     pixel_points[:, 1] = normed_points[:, 1] * f_y + c_y
        
    #     return pixel_points
    
    # def triangulate_points(self, points_from: np.ndarray, points_to: np.ndarray, cam_from_name: str, cam_to_name: str):
    #     points_4D = cv2.triangulatePoints(
    #         projMatr1 = self.projection_matrices[cam_from_name][cam_to_name],
    #         projMatr2 = self.projection_matrices[cam_to_name][cam_from_name],
    #         projPoints1 = points_from,
    #         projPoints2 = points_to
    #     )
        
    #     # to homogeneous coordinates and return
    #     return points_4D[:3] / points_4D[3]
    
    # def project_points(self, points: dict[str, np.ndarray], main_cam_name: str):
    #     points = deepcopy(points)
        
    #     # undistort all points
    #     for cam_name, points_ in points.items():
    #         points_ = points_.reshape(-1, 1, 2)
    #         points[cam_name] = self.undistort_points(points_, cam_name)
        
    #     # triangulate all points relative to main camera
    #     output_3D_points = {}
    #     for cam_name, points_ in points.items():
    #         if cam_name == main_cam_name:
    #             continue
            
    #         points_from = points_.T
    #         points_to = points[main_cam_name].T
            
    #         output_3D_points[cam_name] = self.triangulate_points(
    #                 points_from = points_from, 
    #                 points_to = points_to,
    #                 cam_from_name = cam_name,
    #                 cam_to_name = main_cam_name)
    #         output_3D_points[cam_name] = output_3D_points[cam_name].T
            
    #     return output_3D_points
    
    # def reproject_points(self, points_3D: np.ndarray, cam_from_name: str, cam_to_name: str):
    #     # # Project 3D points onto the 2D plane of the specified camera
    #     # points_3D = points_3D.T
        
    #     # homogeneous_points = np.vstack((points_3D, np.ones((1, points_3D.shape[1]))))
        
    #     # points_2D = self.projection_matrices[cam_to_name][cam_from_name] @ homogeneous_points
        
    #     # # Convert to non-homogeneous coordinates
    #     # points_2D = points_2D[:2] / points_2D[2]
        
    #     points_3D = np.expand_dims(points_3D, axis=1)
        
    #     ext = self.ext[cam_to_name][cam_from_name]
        
    #     rvec, _ = cv2.Rodrigues(ext.rotation_matrix)
    #     tvec = ext.translation_vector.reshape(3, 1)
    #     camera_mat = self.intrinsics[cam_to_name].calibration_matrix
    #     dist_coeffs = self.intrinsics[cam_to_name].distortion_coefficients
        
    #     points_2D, _ = cv2.projectPoints(
    #         objectPoints = points_3D,
    #         rvec = rvec,
    #         tvec = tvec,
    #         cameraMatrix = camera_mat,
    #         distCoeffs = dist_coeffs
    #     )
        
    #     return points_2D.squeeze()
    
    def project_to_world(self, cam_from_points, cam_from_name, cam_to_name):
        # Convert 2D points to 3D in camera coordinate system
        camera_matrix = self.intrinsics[cam_from_name].calibration_matrix
        dist_coeffs = self.intrinsics[cam_from_name].distortion_coefficients
        
        cam_from_points = cam_from_points.reshape(-1, 1, 2)
        points_3D = cv2.undistortPoints(cam_from_points, camera_matrix, dist_coeffs)
        points_3D = points_3D.squeeze()
        # Assuming a fixed Z with value 1 for simplicity
        points_3D = np.concatenate([points_3D, np.ones((points_3D.shape[0], 1))], axis=-1)
        
        # Convert 3D points in camera coordinate system to world coordinate system
        rmat = self.extrinsics[cam_from_name][cam_to_name].rotation_matrix
        tvec = self.extrinsics[cam_from_name][cam_to_name].translation_vector
        
        points_world = (rmat @ points_3D.T).T + tvec
        return points_world
    
    def project_to_camera(self, cam_to_world_points, cam_from_name, cam_to_name):
        # # Convert 3D points from world to target camera coordinate system
        
        rmat = self.extrinsics[cam_from_name][cam_to_name].rotation_matrix
        tvec = self.extrinsics[cam_from_name][cam_to_name].translation_vector
        inv_rmat = rmat.T
        inv_tvec = -inv_rmat @ tvec
        points_cam = (inv_rmat @ cam_to_world_points.T).T + inv_tvec
        
        # rmat = self.extrinsics[cam_to_name][cam_from_name].rotation_matrix
        # tvec = self.extrinsics[cam_to_name][cam_from_name].translation_vector
        # points_cam = (rmat @ (cam_to_world_points - tvec).T).T
        
        # Project 3D points to 2D in the target camera image coordinate space
        camera_matrix = self.intrinsics[cam_to_name].calibration_matrix
        dist_coeffs = self.intrinsics[cam_to_name].distortion_coefficients
        points_2D, _ = cv2.projectPoints(points_cam, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
        
        return points_2D.squeeze()