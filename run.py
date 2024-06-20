import cv2
import os
import random
import numpy as np

from logging import basicConfig, getLogger
from argparse import ArgumentParser
from device_capture_system.deviceIO import load_all_devices_from_config
from device_capture_system.datamodel import FramePreprocessing

from calibrate_camera_system.datamodel import Intrinsics, TargetParameters, TargetDataset, CameraModelInformation
from calibrate_camera_system.targets import TargetManager
from calibrate_camera_system.calibration import IntrinsicsCalibrationManager, ExtrinsicsCalibrationManager
from calibrate_camera_system.core import ImageManager, CameraModel

# ------------ cmd args -------------

arg_parser = ArgumentParser()
arg_parser.add_argument("--logging-level", type=str, default="info", choices=["debug", "info", "warning"], help="debug, info, warning, error, critical")
arg_parser.add_argument("--calibration-data-path", type=str, default="./calibration_data", help="path to find all calibration & target data")

arg_parser.add_argument("--collect-images", action="store_true")
arg_parser.add_argument("--num-frames-to-collect", type=int, default=100)
arg_parser.add_argument("--frame-collection-interval", type=float, default=0.1)

arg_parser.add_argument("--find-targets", action="store_true")

arg_parser.add_argument("--calibrate-intrinsics", action="store_true")
arg_parser.add_argument("--calibrate-extrinsics", action="store_true")
arg_parser.add_argument("--calibration-iterations", type=int, default=100, help="number of iterations for calibration optimization, has to be > 1")
arg_parser.add_argument("--calibration-batch-size", type=int, default=30)
arg_parser.add_argument("--point-top-std-exclusion-percentle", type=float, default=10)
arg_parser.add_argument("--target-top-inverse-distance-exclusion-percentile", type=float, default=20)
arg_parser.add_argument("--high-rmse-cutoff-percentile", type=float, default=20, help="cutoff percentile for high rmse error in calibration")

arg_parser.add_argument("--remove-images-without-targets", action="store_true", help="remove images without targets after calibration")

arg_parser.add_argument("--test-calibration", action="store_true", help="test calibration after calibration")

ARGS = arg_parser.parse_args()

# if no option is providied list all options
if not any([
    ARGS.collect_images,
    ARGS.find_targets,
    ARGS.calibrate_intrinsics,
    ARGS.calibrate_extrinsics,
    ARGS.test_calibration,
    ARGS.remove_images_without_targets
]):
    arg_parser.print_help()
    exit()

# ------------ PARAMETERS ------------

TARGET_PARAMETER_PATH = os.path.join(ARGS.calibration_data_path, "target_parameters")
IMAGE_SAVE_PATH = os.path.join(ARGS.calibration_data_path, "images")
EXTRACTED_TARGET_SAVE_PATH = os.path.join(ARGS.calibration_data_path, "extracted_targets")
INTRINSICS_SAVE_PATH = os.path.join(ARGS.calibration_data_path, "intrinsics")
EXTRINSICS_SAVE_PATH = os.path.join(ARGS.calibration_data_path, "extrinsics")

# !currently only checkerboard is supported
TARGET_TYPE = "checkerboard"

FRAME_PREPROCESSINGS = [FramePreprocessing.ROTATE_90_CLOCKWISE, FramePreprocessing.ROTATE_90_CLOCKWISE, FramePreprocessing.ROTATE_90_COUNTERCLOCKWISE]
CAPTURE_SIZE = (1080, 1920) # ! depends on the frame preprocessing

# ----------- logging ------------

basicConfig(level=ARGS.logging_level.upper())
logger = getLogger(__name__)

# ----------- main ------------

if __name__ == "__main__":
    
    assert os.path.exists(TARGET_PARAMETER_PATH), f"target parameter path {TARGET_PARAMETER_PATH} is required but does not exist"
    
    # create missing directories
    for path in [IMAGE_SAVE_PATH, EXTRACTED_TARGET_SAVE_PATH, INTRINSICS_SAVE_PATH, EXTRINSICS_SAVE_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"created missing directory {path}")
    
    # load camera devices and calibration target parameters
    cameras = load_all_devices_from_config("video", config_file="./devices.json")
    target_parameters = TargetParameters.load(TARGET_PARAMETER_PATH, target_type=TARGET_TYPE)
    # turn target size from m to mm
    target_parameters.target_size_wh_m = (target_parameters.target_size_wh_m[0] * 1000, target_parameters.target_size_wh_m[1] * 1000)
    
    
    # instantiate managers
    image_manager = ImageManager(cameras=cameras, image_save_path=IMAGE_SAVE_PATH, frame_preprocessings=FRAME_PREPROCESSINGS)
    target_manager = TargetManager(target_parameters=target_parameters, targets_save_path=EXTRACTED_TARGET_SAVE_PATH, images_save_path=IMAGE_SAVE_PATH)
    intrinsics_calibration_manager = IntrinsicsCalibrationManager(target_parameters=target_parameters, capture_size=CAPTURE_SIZE)
    extrinsics_calibration_manager = ExtrinsicsCalibrationManager(target_parameters=target_parameters, capture_size=CAPTURE_SIZE)
    
    # COLLECT IMAGES ---------------------------------------------------------------------------------------------------------------------------
    
    # collect images and store in images folder under calibration_data
    if ARGS.collect_images:
        logger.info(" --- Collecting images ...")
        image_manager.collect_images(
            num_images_to_collect=ARGS.num_frames_to_collect,
            frame_collection_interval=ARGS.frame_collection_interval
        )
    
    # FIND TARGETS ---------------------------------------------------------------------------------------------------------------------------
    
    # find targets in all collected image and store in extracted_targets folder
    if ARGS.find_targets:
        logger.info(" --- Finding calibration targets ...")
        target_datasets = target_manager.extract_target_data_from_images(cameras=cameras)
        for dataset in target_datasets:
            dataset.save(EXTRACTED_TARGET_SAVE_PATH)
    
    # REMOVE IMAGES WITHOUT TARGETS ---------------------------------------------------------------------------------------------------------------------------
    
    if ARGS.remove_images_without_targets:
        logger.info(" --- Removing images without targets ...")
        target_datasets = [
            TargetDataset.load(file_path=EXTRACTED_TARGET_SAVE_PATH, camera=cam)
            for cam in cameras
        ]
        image_manager.remove_images_without_targets(target_datasets)
    
    # INTRINSICS ---------------------------------------------------------------------------------------------------------------------------
    
    # load and filter targets 
    # calibrate intrinsics
    # save intrinsics
    if ARGS.calibrate_intrinsics:
        logger.info(" --- Calibrating intrinsics ...")
        target_datasets = [
            intrinsics_calibration_manager.filter_targets(
                target_dataset=TargetDataset.load(file_path=EXTRACTED_TARGET_SAVE_PATH, camera=cam),
                point_top_std_exclusion_percentle=ARGS.point_top_std_exclusion_percentle,
                target_top_inverse_distance_exclusion_percentile=ARGS.target_top_inverse_distance_exclusion_percentile
            )
            for cam in cameras
        ]
        calibration_intrinsics = intrinsics_calibration_manager.calibrate(
            target_datasets=target_datasets,
            batch_size = ARGS.calibration_batch_size,
            optim_iterations = ARGS.calibration_iterations,
            high_rmse_cutoff_percentile=ARGS.high_rmse_cutoff_percentile
        )
        for intrinsics in calibration_intrinsics:
            intrinsics.save(INTRINSICS_SAVE_PATH)
    
    # EXTRINSICS ---------------------------------------------------------------------------------------------------------------------------
    
    # load and filter targets
    # load intrinsics
    # calibrate extrinsics
    # save extrinsics
    if ARGS.calibrate_extrinsics:
        logger.info(" --- Calibrating extrinsics ...")
        target_datasets = [
            intrinsics_calibration_manager.filter_targets(
                target_dataset=TargetDataset.load(file_path=EXTRACTED_TARGET_SAVE_PATH, camera=cam),
                point_top_std_exclusion_percentle=ARGS.point_top_std_exclusion_percentle,
                target_top_inverse_distance_exclusion_percentile=ARGS.target_top_inverse_distance_exclusion_percentile
            )
            for cam in cameras
        ]
        calibration_intrinsics = [
            Intrinsics.load(file_path=INTRINSICS_SAVE_PATH, camera=cam)
            for cam in cameras
        ]
        extrinsics = extrinsics_calibration_manager.calibrate(
            target_datasets=target_datasets,
            camera_intrinsics=calibration_intrinsics,
            batch_size=ARGS.calibration_batch_size,
            optim_iterations=ARGS.calibration_iterations,
            high_rmse_cutoff_percentile=ARGS.high_rmse_cutoff_percentile
        )
        for extrinsic in extrinsics:
            extrinsic.save(EXTRINSICS_SAVE_PATH)
    
    
    # TEST ---------------------------------------------------------------------------------------------------------------------------
    
    # load intrinsics and extrinsics
    # load test images and targets
    # project points 
    # reproject points and visualize
    if ARGS.test_calibration:
        logger.info(" --- Testing calibration ...")
        
        cam_model = CameraModel(
            CameraModelInformation.load(
                intrinsics_save_path=INTRINSICS_SAVE_PATH,
                extrinsics_save_path=EXTRINSICS_SAVE_PATH,
                cameras=cameras
            )
        )
        
        target_datasets = [
            TargetDataset.load(file_path=EXTRACTED_TARGET_SAVE_PATH, camera=cam)
            for cam in cameras
        ]
        
        # get target wise points for all cameras and filter for targets with points in all cameras
        target_wise_points = {}
        for target_ds in target_datasets:
            for target in target_ds.target_data:
                if target.image_name not in target_wise_points:
                    target_wise_points[target.image_name] = {}
                target_wise_points[target.image_name][target_ds.camera.name] = target.target_points
        filtered_target_wise_points = []
        for twp in target_wise_points:
            if len(target_wise_points[twp]) == len(cameras):
                filtered_target_wise_points.append({
                    "image_name": twp,
                    "target_points": target_wise_points[twp]
                })
        
        
        
        
        def rmse(a, b):
            return np.sqrt(((a - b)**2).mean(axis=0))
        
        total_rmse = {cam.name: {cam2.name: 0 for cam2 in cameras} for cam in cameras}
        for target in filtered_target_wise_points:
            
            image_name, points_2D = target["image_name"], target["target_points"]
            
            for main_cam in cameras:
                
                points_3D = cam_model.unified_3D_model(points = points_2D, main_cam_name = main_cam.name)
                
                for cam_name in points_3D:
                    repro = cam_model.project_points_to_view(points_3D[cam_name], cam_name, main_cam.name)
                    total_rmse[main_cam.name][cam_name] += rmse(points_2D[main_cam.name], repro).mean()
        
        for main_cam_name in total_rmse:
            for cam_name in total_rmse[main_cam_name]:
                total_rmse[main_cam_name][cam_name] /= len(filtered_target_wise_points)
                print(f"RMSE: {main_cam_name} -> {cam_name}: {total_rmse[main_cam_name][cam_name]}")
        
        
        # visualize image and points
        for target in random.sample(filtered_target_wise_points, 10):
            image_name, points_2D = target["image_name"], target["target_points"]
            
            points_3D = cam_model.unified_3D_model(points = points_2D, main_cam_name = main_cam.name)
            reprojections = {cam_name: cam_model.project_points_to_view(points_3D[cam_name], cam_name, main_cam.name) for cam_name in points_3D}
            combined_points = np.mean(np.stack([points_3D[cam_name] for cam_name in points_3D]), axis=0)
            
            image = cv2.imread(os.path.join(IMAGE_SAVE_PATH, f"{main_cam.name}", f"{image_name}"))
            
            for point in points_2D[main_cam.name]:
                cv2.circle(image, tuple(map(int, point)), 5, (255, 0, 0), -1)
            
            for cam_name in reprojections:
                for point in reprojections[cam_name]:
                    cv2.circle(image, tuple(map(int, point)), 5, (0, 255, 0), -1)
            
            
            cv2.imshow("main_cam_image", image)
            cv2.waitKey(0)
            