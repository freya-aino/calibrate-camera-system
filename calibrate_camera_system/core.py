import os
import numpy as np
import cv2

from datetime import datetime
from logging import getLogger
from time import sleep
from typing import List

from device_capture_system.datamodel import CameraDevice, FramePreprocessing
from device_capture_system.core import MultiInputStreamSender
from device_capture_system.fileIO import ImageSaver

from .datamodel import Extrinsics, Intrinsics



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
                collected_frames += 1
                error_frames = 0
            
        except Exception as e:
            raise e
        finally:
            self.multi_stream_sender.stop_processes()
            self.image_saver.stop()

# --------------------- CAMERA TRANSFORMER ---------------------




class CameraTransformer:
    def __init__(self, camera_intrinsics: List[Intrinsics], camera_extrinsics: List[Extrinsics]):
        self.logger = getLogger(self.__class__.__name__)
        
        self.intrinsics = camera_intrinsics
        
        ext = {}
        for extrinsic in camera_extrinsics:
            if extrinsic.camera_from not in ext:
                ext[extrinsic.camera_from] = {}
            ext[extrinsic.camera_from][extrinsic.camera_to] = extrinsic
        
        self.projection_matricies = {}
        for cam_from in ext:
            for cam_to in ext[cam_from]:
                rot_mat = ext[cam_from.name][cam_to.name].rotation_mat
                trans_vec = ext[cam_from.name][cam_to.name].translation_vec
                self.projection_matricies[cam_from.name][cam_to.name] = self.intrinsics[cam_from.name].calibration_matrix @ np.concatenate([rot_mat, trans_vec], axis=1)
                
        print("projection matricies", self.projection_matricies)
        
        
    def undistort_points(self, points: np.ndarray, intrinsics: Intrinsics):
        return cv2.undistortPoints(src = points, cameraMatrix = intrinsics.calibration_matrix, distCoeffs = intrinsics.distortion_coefficients)
    
    def triangulate_points(self, points_from: np.ndarray, points_to: np.ndarray, cam_from_name: str, cam_to_name: str):
        points_4D = cv2.triangulatePoints(
            projMatr1 = self.projection_matricies[cam_from_name][cam_to_name],
            projMatr2 = self.projection_matricies[cam_to_name][cam_from_name],
            projPoints1 = points_from,
            projPoints2 = points_to
        )
        # to homogeneous coordinates and return
        return points_4D[:3] / points_4D[3]
    
    def process_points(self, points: dict[str, np.ndarray], main_cam_name: str):
        
        # undistort all points
        for cam_name, point in points.items():
            points[cam_name] = self.undistort_points(point, self.intrinsics[cam_name])
        
        # triangulate all points relative to main camera
        output_3D_points = []
        for cam_name, point in points.items():
            if cam_name == main_cam_name:
                continue
            output_3D_points.append(
                self.triangulate_points(
                    points_from = points[cam_name], 
                    points_to = points[main_cam_name],
                    cam_from_name = cam_name,
                    cam_to_name = main_cam_name
                )
            )
        