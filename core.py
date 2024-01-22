import os
import time
import cv2
import numpy as np
import json
import traceback
from copy import deepcopy

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
    return OrderedDict(sorted(list(dct.items()), key=lambda x: x[0]))


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
    return image_uri, cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria).squeeze()

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


def remove_images_without_targets(cam_uuid: str):
    
    valid_targets = {}
    
    logger.info(f"{cam_uuid} :: loading extracted target corners and removing images without targets ...")
    with open(os.path.join("calibration_target_results", f"{cam_uuid}.json"), "r") as f:
        all_targets = json.load(f, object_hook=decode_dict)
        for k in all_targets:
            if all_targets[k] is None:
                logger.info(f"{cam_uuid} :: {k} has no target, removing ...")
                os.remove(k)
            else:
                valid_targets[k] = all_targets[k]
    
    # save updated list of targets
    with open(os.path.join("calibration_target_results", f"{cam_uuid}.json"), "w") as f:
        json.dump(valid_targets, f, cls=NumpyEncoder)


def filter_images_for_intrinsics(cam_uuid: str, point_top_std_exclusion_percentle: float = 10, target_top_inverse_distance_exclusion_percentile: float = 20):
    
    with open(os.path.join("calibration_target_results", f"{cam_uuid}.json"), "r") as f:
        valid_targets = json.load(f, object_hook=decode_dict)
    
    # filter out images where the normaized pattern has too high std
    normalised_patterns = np.stack([Z_norm(valid_target) for valid_target in valid_targets.values()], axis=0)
    per_point_std = np.sqrt((normalised_patterns - normalised_patterns.mean(axis=0))**2)
    top_std_percentile_threshold = np.percentile(per_point_std, (100 - point_top_std_exclusion_percentle), axis=0)
    valid_std_mask = per_point_std <= np.expand_dims(top_std_percentile_threshold, axis=0)
    
    # print("normalised_patterns", normalised_patterns)
    # print("pattern_std", per_point_std)
    # print("top_std_percentile_threshold", top_std_percentile_threshold)
    # print("valid_variance_mask", valid_std_mask)
    
    for i, (k, v) in enumerate(valid_targets.items()):
        if not valid_std_mask[i].all():
            valid_targets[k] = None
    valid_targets = OrderedDict([(k, v) for (k, v) in valid_targets.items() if v is not None])
    
    if len(valid_targets) == 0:
        logger.warning(f"{cam_uuid} :: no valid targets after removing high per point std targets")
        return
    logger.info(f"{cam_uuid} :: {len(valid_targets)} valid targets after removing high per point std targets")
    
    
    # filter out images with targets means too close to each other
    target_means = np.stack([valid_target.mean(axis=0) for valid_target in valid_targets.values()], axis=0)
    normed_target_means = Z_norm(target_means)
    avg_total_distances_inv = np.stack([np.exp( - ((normed_target_means[i] - normed_target_means)**2).mean(axis=0)).mean(axis=0) for i in range(normed_target_means.shape[0])], axis=0)
    valid_distance_mask = avg_total_distances_inv < np.percentile(avg_total_distances_inv, 100 - target_top_inverse_distance_exclusion_percentile)
    
    # print("normed_target_means", normed_target_means)
    # print("valid_distance_mask", valid_distance_mask)
    # print("valid_targets", target_means[valid_distance_mask])
    # print("avg_total_distance", avg_total_distances_inv)
    
    for i, (k, v) in enumerate(valid_targets.items()):
        if not valid_distance_mask[i]:
            valid_targets[k] = None
    valid_targets = OrderedDict([(k, v) for (k, v) in valid_targets.items() if v is not None])
    
    if len(valid_targets) == 0:
        logger.warning(f"{cam_uuid} :: no valid targets after removing high per target mean std targets")
        return
    logger.info(f"{cam_uuid} :: {len(valid_targets)} valid targets after removing high per target mean std targets")
    
    return valid_targets


def calibration_intrinsics(
    targets: OrderedDict[str, np.ndarray], 
    num_target_corners: Tuple[int, int] = (3, 4), # (width, height) number of targets on the checkerboard
    target_size: Tuple[float, float] = (0.05, 0.055), # (width, height) of each target on the checkerboard in meters (m)
    capture_size: Tuple[int, int] = (1920, 1080)): # (width, height) of the capture camera
    
    target_points = np.array([*targets.values()], dtype=np.float32)
    
    # prepare model points
    model_points = np.mgrid[:num_target_corners[0], :num_target_corners[1]].T.reshape(-1,2)
    model_points = model_points * target_size
    model_points = np.concatenate([model_points, np.zeros([model_points.shape[0], 1])], axis=1).astype(np.float32)
    model_points = np.array(len(target_points) * [model_points], dtype=np.float32)
    
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
    
    
    batch_size = 20
    optim_iterations = 10
    
    logger.info(f"calibrating camera with {batch_size} samples per iteration for {optim_iterations} iterations ...")
    avg_rmse = 0
    for e in range(optim_iterations):
        
        # select batch
        rand_indexes = np.random.randint(model_points.shape[0], size=batch_size)
        model_points_ = model_points[rand_indexes, :, :]
        target_points_ = target_points[rand_indexes, :, :]
        
        rmse, calibration_matrix, dist_coeffs, r_vecs, t_vecs = cv2.calibrateCamera(
            objectPoints=model_points_, 
            imagePoints=target_points_, 
            imageSize=capture_size, 
            cameraMatrix=calibration_matrix, 
            distCoeffs=dist_coeffs)
        
        avg_rmse += rmse
        
    print("avg_rmse", avg_rmse / optim_iterations)
    # print("calibration_matrix", calibration_matrix)
    # print("dist_coeffs_", dist_coeffs)
    # print("r_vecs", r_vecs)
    # print("t_vecs", t_vecs)


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