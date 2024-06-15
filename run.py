from json import dump
from logging import basicConfig, getLogger
from json import load as json_load
from argparse import ArgumentParser
from device_capture_system.deviceIO import load_all_devices_from_config
from device_capture_system.datamodel import FramePreprocessing

from calibrate_camera_system.datamodel import ModelIntrinsics
from calibrate_camera_system.targets import TargetManager
from calibrate_camera_system.calibration import IntrinsicsCalibrator, ExtrinsicsCalibrator

# ------------ cmd args -------------

arg_parser = ArgumentParser()
arg_parser.add_argument("--collect-images", action="store_true")
arg_parser.add_argument("--num-frames-to-collect", type=int, default=100)
arg_parser.add_argument("--frame-collection-interval", type=float, default=0.1)

arg_parser.add_argument("--find-targets", action="store_true")
arg_parser.add_argument("--remove-images-without-targets", action="store_true")

arg_parser.add_argument("--calibrate-intrinsics", action="store_true")
arg_parser.add_argument("--calibrate-extrinsics", action="store_true")
arg_parser.add_argument("--calibration-iterations", type=int, default=3)
arg_parser.add_argument("--calibration-batch-size", type=int, default=40)
arg_parser.add_argument("--point-top-std-exclusion-percentle", type=float, default=5)
arg_parser.add_argument("--target_top_inverse_distance_exclusion_percentile", type=float, default=5)

arg_parser.add_argument("--capture-size", type=tuple, default=(1080, 1920))

ARGS = arg_parser.parse_args()

if not any(vars(ARGS).values()):
    arg_parser.print_help()
    exit(0)

# ----------- logging ------------

basicConfig(level=ARGS.logging_level.upper())
logger = getLogger(__name__)

# ----------- main ------------

if __name__ == "__main__":
    
    # load camera devices
    cameras = load_all_devices_from_config("video", config_file="./devices.json")
    
    
    # collect images and store in ./images
    if ARGS.collect_images:
        logger.info(" --- Collecting images ...")
        
        from calibrate_camera_system.core import ImageManager
        
        image_manager = ImageManager(
            cameras=cameras,
            frame_preprocessings=[
                FramePreprocessing.ROTATE_90_CLOCKWISE,
                FramePreprocessing.ROTATE_90_CLOCKWISE,
                FramePreprocessing.ROTATE_90_COUNTERCLOCKWISE
            ]
        )
        
        image_manager.collect_images(
            num_images_to_collect=ARGS.num_frames_to_collect,
            frame_collection_interval=ARGS.frame_collection_interval
        )
    
    
    target_manager = TargetManager(cameras=cameras)
    
    if ARGS.find_targets:
        logger.info(" --- Finding calibration targets ...")
        extracted_targets = target_manager.extract_target_data_from_images()
        target_manager.save_extracted_targets(extracted_targets)
        
    if ARGS.calibrate_intrinsics:
        logger.info(" --- Calibrating intrinsics ...")
        
        # load targets
        extracted_targets = target_manager.load_extracted_targets()
        
        intrinsics = {}
        for cam in cameras:
            capture_size = ARGS.capture_size,
            calibrator = IntrinsicsCalibrator(camera=cam, capture_size=capture_size, calibration_target=target_manager.calibration_target)
            
            calibrator.calibrate(
                target_data=extracted_targets[cam.name], 
                batch_size=ARGS.calibration_batch_size,
                optim_iterations=ARGS.calibration_iterations,
                save=True
            )
    
    if ARGS.calibrate_extrinsics:
        logger.info(" --- Calibrating extrinsics ...")
        
        # load targets
        extracted_targets = target_manager.load_extracted_targets()
        
        # load intrinsics
        intrinsics = {}
        for cam in cameras:
            intrinsics[cam.name] = ModelIntrinsics.load_from_file(f"./calibration_data/intrinsics/{cam.name}.json")
        
        calibrator = ExtrinsicsCalibrator(
            cameras=cameras,
            calibration_target=target_manager.calibration_target,
            capture_size=ARGS.capture_size
        )
        
        calibrator.calibrate(
            camera_intrinsics=intrinsics,
            target_data=extracted_targets,
            batch_size=ARGS.calibration_batch_size,
            optim_iterations=ARGS.calibration_iterations,
        )