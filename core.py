import os
import time
import cv2
import numpy as np
import json
import traceback

from collections import OrderedDict
from logging import getLogger
from time import sleep
from multiprocessing import Pool
from typing import List, Tuple
from camera_capture_system.datamodel import Camera, ImageParameters
from camera_capture_system.fileIO import CaptureImageSaver
from camera_capture_system.core import MultiCapturePublisher, load_all_cameras_from_config

# --- logging ---

logger = getLogger(__name__)

# --- logging ---


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def decode_dict(dct):
    for key, value in dct.items():
        if isinstance(value, list):
            dct[key] = np.array(value)
    return dct




def collect_images(cameras_cofig_file: str, num_frames_to_collect, frame_collection_interval=1):
    
    logger.info("initialize image collection")
    
    # load cameras
    cameras = load_all_cameras_from_config(cameras_cofig_file)
    
    # initialize camera capture publisher
    multi_capture_publisher = MultiCapturePublisher(cameras=cameras)
    
    # initialize capture save processor
    image_params = ImageParameters(
        save_path="./images",
        jpg_quality=100,
        png_compression=0,
        output_format="jpg"
    )
    capture_image_saver = CaptureImageSaver(cameras=cameras, image_params=image_params)
    
    try:
        
        logger.info("start image publisher and saver")
        
        # start both processes
        multi_capture_publisher.start()
        capture_image_saver.start()
        
        collected_frames = 0
        while collected_frames < num_frames_to_collect:
            successfull = capture_image_saver.save_image(visualize=True)
            sleep(frame_collection_interval)
            
            if successfull:
                collected_frames += 1
                
            logger.info(f"collected {collected_frames} frames")
    except:
        raise
    finally:
        capture_image_saver.stop()
        multi_capture_publisher.stop()


def check_and_get_target_corners(image_uri: str, num_target_corners: tuple[int, int], subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)) -> Tuple[str, np.ndarray]:
    
    frame = cv2.imread(image_uri)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_corners, corners = cv2.findChessboardCorners(gray, num_target_corners, None)
    
    if not ret_corners:
        return image_uri, None
    return image_uri, cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)

def find_all_calibration_targets(image_dir: str, num_target_corners: Tuple[int, int], num_workers: int = 8) -> dict[str, Tuple[str, np.ndarray]]:
    
    image_uris = [os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir)]
    process_pool = Pool(num_workers)
    process_results = []
    
    try:
        for image_uri in image_uris:
            result = process_pool.apply_async(check_and_get_target_corners, args=(image_uri, num_target_corners))
            process_results.append(result)
            
        results = OrderedDict([result.get() for result in process_results])
        
        with open(os.path.join("calibration_target_results", f"{image_dir.split('/')[-1]}.json"), "w") as f:
            json.dump(results, f, cls=NumpyEncoder)
        
    except:
        raise
    finally:
        process_pool.close()
        for result in process_results:
            result.wait()
        process_pool.join()
    



def Z_norm(array: np.ndarray, axis: int = 0) -> np.ndarray:
    return (array - array.mean(axis=axis)) / array.std(axis=axis)

def filter_images_for_intrinsics(cam_uuid: str, pattern_top_std_exclusion_percentle: float = 10):
    
    valid_images = OrderedDict()
    invalid_images = OrderedDict()
    
    logger.info(f"{cam_uuid} :: extracting target corners")
    image_data = get_images_with_calibration_target(image_dir=f"./images/{cam_uuid}")
    
    # TODO: check if images are in correct order
    
    # filter out images without target corners
    for k in image_data:
        if image_data[k] is None:
            invalid_images[k] = None
        else:
            valid_images[k] = image_data[k]
    
    # filter out images where the normaized pattern has too high std
    normalised_patterns = np.stack([Z_norm(valid_target.squeeze()) for valid_target in valid_images.values], axis=0)
    pattern_std = normalised_patterns.std(axis=0)
    top_std_percentile_threshold = np.percentile(pattern_std, (100 - pattern_top_std_exclusion_percentle), axis=0)
    valid_variance_mask = normalised_patterns > np.expand_dims(top_std_percentile_threshold, axis=0)
    
    print("valid_variance_mask", valid_variance_mask.shape)
    
    print("number of imaes before filtering", len(valid_images))
    
    for i, (k, v) in enumerate(valid_images.items()):
        if valid_variance_mask[i].any():
            invalid_images[k] = v
            valid_images.pop(k)
    
    print("number of imaes after filtering", len(valid_images))
    
    
    # filter out images so that they the means of the targets are evenly distributed
    target_means = np.stack([valid_target.mean(axis=0) for valid_target in valid_images.values], axis=0)
    normalised_pattern_means = Z_norm(target_means)




def prepare_checkerboard_model_points(num_target_corners: tuple[int, int], targets_size: tuple[int, int]):
    """
    Prepare model points (object points) for checkerboard calibration.
    
    params:
        capture_size = (width, height) size of capture
        num_target_corners = (width, height) number of corners on checkerboard
        target_size = (width, height) size of each corner on checkerboard in meters (m)
    """
    
    model_points = np.mgrid[:num_target_corners[0], :num_target_corners[1]].T.reshape(-1,2)
    return np.concatenate(model_points * targets_size, np.zeros((model_points.shape[0], 1), np.float32), axis=1)


def calibration():

    # PARAMETERS
    capture_size = (1920, 1080) # (width, height) of the capture camera
    num_target_corners = (3, 4) # (width, height) number of targets on the checkerboard
    target_size = (0.05, 0.055) # (width, height) of each target on the checkerboard in meters (m

    model_points = prepare_checkerboard_model_points(num_target_corners=num_target_corners, targets_size=target_size)

    effective_focal_length = capture_size # (width, height) focal length in pixels, because pixel density might be different in height and width and are also unknown
    principle_point = (capture_size[0] / 2, capture_size[1] / 2)    
    
    calibration_matrix = np.array(
        [
            [effective_focal_length[0], 1, principle_point[0]],
            [0, effective_focal_length[1], principle_point[1]],
            [0, 0, 1]
        ]
    )
    dist_coeffs = np.zeros((5, 1), np.float32)


# rmse, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, capture_size, None, None)
# camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, capture_size, 1, capture_size)


# rotation_matrix = np.eye(3)
# translation_vector = np.array([[0, 0, 1]]).T


# def get_projection_matrix(calibration_matrix, rotation_matrix, translation_vector):
#     intrinsic_matrix = np.concatenate([calibration_matrix, np.zeros((3, 1))], axis=1)
#     extrinsic_matrix = np.concatenate([np.concatenate([rotation_matrix, translation_vector], axis=1), np.array([[0, 0, 0, 1]])], axis=0)
#     return intrinsic_matrix @ extrinsic_matrix

# def get_componetes_from_projection_matrix(projection_matrix):
#     rotation_matrix_, calibration_matrix_ = np.linalg.qr(projection_matrix[:3, :3])
#     translation_vector_ = np.linalg.inv(calibration_matrix_) @ projection_matrix[:3, -1]
#     return calibration_matrix_, rotation_matrix_, translation_vector_


# projection_matrix = get_projection_matrix(calibration_matrix, rotation_matrix, translation_vector)


# world_space_point = np.array([0.5, 0.55, 0, 1])
# print(projection_matrix @ world_space_point)
# print(np.linalg.inv(projection_matrix) @ projection_matrix @ world_space_point)


# # TODO:
# cv2.initCameraMatrix2D()

# # there is also matMulDeriv for manual calculation
# cv2.matMulDeriv()

# # dose the same as above get_components_from_projection_matrix
# cv2.decomposeProjectionMatrix()