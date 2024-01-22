from core import collect_images, find_all_calibration_targets, filter_images_for_intrinsics, remove_images_without_targets, calibration_intrinsics
from cv2 import rotate, ROTATE_90_CLOCKWISE
from logging import basicConfig

basicConfig(level="INFO")


if __name__ == "__main__":
    
    # collect_images(
    #     cameras_cofig_file="cameras_configs.json", 
    #     num_frames_to_collect=100)
    
    # todo: load target specifications from target_data
    
    # find_all_calibration_targets("./images/cam0", num_target_corners=(3, 4))
    
    remove_images_without_targets("cam0")
    
    filtered_targets = filter_images_for_intrinsics("cam0")
    
    calibration_intrinsics(filtered_targets, (3, 4), (0.025, 0.025), (1920, 1080))
    