from core import collect_images, find_all_calibration_targets, remove_images_without_targets, calibrate_intrinsics, calibrate_extrinsics
from cv2 import rotate, ROTATE_90_CLOCKWISE
from logging import basicConfig, getLogger
from json import load as json_load

from camera_capture_system.core import load_all_cameras_from_config


basicConfig(level="INFO")
logger = getLogger(__name__)

if __name__ == "__main__":
    
    # collect_images(
    #     cameras_cofig_file="cameras_configs.json", 
    #     num_frames_to_collect=100,
    #     frame_collection_interval=0.1,
    #     frame_transforms={
    #         "cam0": "ROTATE_90_COUNTERCLOCKWISE",
    #         "cam1": "ROTATE_90_CLOCKWISE",
    #         "cam2": "ROTATE_90_COUNTERCLOCKWISE",
    #     })
    
    # load target data and camera configs
    with open("target_data/target.json", "r") as f:
        target_data = json_load(f)
    cameras = load_all_cameras_from_config("cameras_configs.json")
    
    
    # for cam in cameras:
        
    #     # find calibration targets
    #     logger.info(f"Finding calibration targets for {cam.uuid} ...")
    #     find_all_calibration_targets(cam_uuid=cam.uuid, num_target_corners=target_data["num_target_corners"])
        
    #     remove_images_without_targets(cam.uuid)
        
    #     calibrate_intrinsics(
    #         cam_uuid=cam.uuid,
    #         num_target_corners=target_data["num_target_corners"], 
    #         target_size=target_data["target_size_meters"], 
    #         capture_size=(1080, 1920))
    
    
    calibrate_extrinsics(cameras)
    
    