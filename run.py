import os
from logging import basicConfig, getLogger
from argparse import ArgumentParser
from device_capture_system.deviceIO import load_all_devices_from_config
from device_capture_system.datamodel import FramePreprocessing

from calibrate_camera_system.datamodel import Intrinsics, TargetParameters, TargetDataset
from calibrate_camera_system.targets import TargetManager
from calibrate_camera_system.calibration import IntrinsicsCalibrationManager, ExtrinsicsCalibrationManager
from calibrate_camera_system.core import ImageManager

# ------------ cmd args -------------

arg_parser = ArgumentParser()
arg_parser.add_argument("--logging-level", type=str, default="info", choices=["debug", "info", "warning"], help="debug, info, warning, error, critical")
arg_parser.add_argument("--calibration-data-path", type=str, default="./calibration_data", help="path to find all calibration & target data")

arg_parser.add_argument("--collect-images", action="store_true")
arg_parser.add_argument("--num-frames-to-collect", type=int, default=100)
arg_parser.add_argument("--frame-collection-interval", type=float, default=0.1)

arg_parser.add_argument("--find-targets", action="store_true")
# arg_parser.add_argument("--remove-images-without-targets", action="store_true")

arg_parser.add_argument("--calibrate-intrinsics", action="store_true")
arg_parser.add_argument("--calibrate-extrinsics", action="store_true")
arg_parser.add_argument("--calibration-iterations", type=int, default=3)
arg_parser.add_argument("--calibration-batch-size", type=int, default=40)
arg_parser.add_argument("--point-top-std-exclusion-percentle", type=float, default=5)
arg_parser.add_argument("--target_top_inverse_distance_exclusion_percentile", type=float, default=5)

ARGS = arg_parser.parse_args()

# if no option is providied list all options
if not any([
    ARGS.collect_images,
    ARGS.find_targets,
    ARGS.calibrate_intrinsics,
    ARGS.calibrate_extrinsics
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
    
    # instantiate managers
    image_manager = ImageManager(cameras=cameras, frame_preprocessings=FRAME_PREPROCESSINGS)
    target_manager = TargetManager(target_parameters=target_parameters, targets_save_path=EXTRACTED_TARGET_SAVE_PATH, images_save_path=IMAGE_SAVE_PATH)
    intrinsics_calibration_manager = IntrinsicsCalibrationManager(target_parameters=target_parameters, capture_size=CAPTURE_SIZE)
    extrinsics_calibration_manager = ExtrinsicsCalibrationManager(target_parameters=target_parameters, capture_size=CAPTURE_SIZE)
    
    # ---------------------------------------------------------------------------------------------------------------------------
    
    # collect images and store in images folder under calibration_data
    if ARGS.collect_images:
        logger.info(" --- Collecting images ...")
        image_manager.collect_images(
            num_images_to_collect=ARGS.num_frames_to_collect,
            frame_collection_interval=ARGS.frame_collection_interval
        )
    
    # ---------------------------------------------------------------------------------------------------------------------------
    
    # find targets in all collected image and store in extracted_targets folder
    if ARGS.find_targets:
        logger.info(" --- Finding calibration targets ...")
        target_datasets = target_manager.extract_target_data_from_images(cameras=cameras)
        for dataset in target_datasets:
            dataset.save(EXTRACTED_TARGET_SAVE_PATH)
    
    # ---------------------------------------------------------------------------------------------------------------------------
    
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
            optim_iterations = ARGS.calibration_iterations
        )
        for intrinsics in calibration_intrinsics:
            intrinsics.save(INTRINSICS_SAVE_PATH)
    
    # ---------------------------------------------------------------------------------------------------------------------------
    
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
            optim_iterations=ARGS.calibration_iterations
        )
        for extrinsic in extrinsics:
            extrinsic.save(EXTRINSICS_SAVE_PATH)