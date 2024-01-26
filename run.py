from core import collect_images, find_all_calibration_targets, remove_images_without_targets, calibrate_intrinsics, calibrate_extrinsics
from cv2 import rotate, ROTATE_90_CLOCKWISE
from logging import basicConfig, getLogger
from json import load as json_load
from camera_capture_system.core import load_all_cameras_from_config
from argparse import ArgumentParser


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

args = arg_parser.parse_args()

if not any(vars(args).values()):
    arg_parser.print_help()
    exit(0)

# ----------- logging ------------

basicConfig(level="INFO")
logger = getLogger(__name__)

# ----------- main ------------

# ! some defult parameters are hard-coded here, change them if needed
camera_frame_transforms = {
    "cam0": "ROTATE_90_COUNTERCLOCKWISE",
    "cam1": "ROTATE_90_CLOCKWISE",
    "cam2": "ROTATE_90_COUNTERCLOCKWISE",
}
output_capture_size = (1080, 1920)


if __name__ == "__main__":
    
    # load target data and camera configs
    with open("target_data/target.json", "r") as f:
        target_data = json_load(f)
    cameras = load_all_cameras_from_config("cameras_configs.json")
    
    
    # collect images and store in ./images
    if args.collect_images:
        logger.info(" --- Collecting images ...")
        collect_images(
            cameras_cofig_file="cameras_configs.json", 
            num_frames_to_collect=args.num_frames_to_collect,
            frame_collection_interval=args.frame_collection_interval,
            frame_transforms=camera_frame_transforms)
    
    
    for cam in cameras:
        
        # find calibration targets
        if args.find_targets:
            logger.info(f" --- Finding calibration targets for {cam.uuid} ...")
            find_all_calibration_targets(cam_uuid=cam.uuid, num_target_corners=target_data["num_target_corners"])
        
        if args.remove_images_without_targets:
            logger.info(f" --- Removing images without calibration targets for {cam.uuid} ...")
            remove_images_without_targets(cam.uuid)
        
        if args.calibrate_intrinsics:
            logger.info(f" --- Calibrating intrinsics for {cam.uuid} ...")
            calibrate_intrinsics(
                cam_uuid=cam.uuid,
                num_target_corners=target_data["num_target_corners"], 
                target_size=target_data["target_size_meters"], 
                capture_size=output_capture_size,
                batch_size=args.calibration_batch_size,
                optim_iterations=args.calibration_iterations,
                point_top_std_exclusion_percentle=args.point_top_std_exclusion_percentle,
                target_top_inverse_distance_exclusion_percentile=args.target_top_inverse_distance_exclusion_percentile)
    
    if args.calibrate_extrinsics:
        logger.info(" --- Calibrating extrinsics ...")
        calibrate_extrinsics(
            cameras,
            num_target_corners=target_data["num_target_corners"],
            target_size=target_data["target_size_meters"],
            capture_size=output_capture_size,
            batch_size=args.calibration_batch_size,
            optim_iterations=args.calibration_iterations,
            point_top_std_exclusion_percentle=args.point_top_std_exclusion_percentle,
            target_top_inverse_distance_exclusion_percentile=args.target_top_inverse_distance_exclusion_percentile)
    
    