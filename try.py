import time
import cv2
import numpy as np
import json
import traceback

# define image regions (height, width), and utility function
def get_image_region(image_position: tuple[int, int], region_size: tuple[int, int]) -> tuple[int, int]:
    # estimate which region the image position is in
    region_position = int(image_position[0] / region_size[0]), int(image_position[1] / region_size[1])
    return region_position

def collect_target_points(
    video_capture: cv2.VideoCapture,
    num_regions: tuple[int, int] = (3, 2),
    per_region_requirement: int = 30,
    max_fail_count: int = 10,
    target_size: tuple[int, int] = (4, 3),
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # criteria for subpixel corner estimation
) -> tuple[np.ndarray, np.ndarray]:
    
    fail_count = 0
    
    # initialize capture
    try:
        while True:
            ret_capture, frame = video_capture.read()
            if ret_capture:
                print("Capture initialized")
                break
            max_fail_count += 1
            print(f"No frame, {fail_count}/{max_fail_count} ...")
            assert fail_count < max_fail_count, "Failed to initialize capture"
    except:
        raise
    
    # calculate region size
    region_size = int(frame.shape[1] / num_regions[0]), int(frame.shape[0] / num_regions[1])
    
    
    # prepare per region count
    per_region_conut = {}
    for i in range(num_regions[0]):
        for j in range(num_regions[1]):
            per_region_conut[i, j] = 0
    
    # collect points for all regions
    collected_image_points = []
    try:
        while True:
            
            dt = time.perf_counter()
            
            # Capture frame-by-frame
            ret_capture, frame = video_capture.read()
            
            if not ret_capture:
                max_fail_count += 1
                print(f"No frame, {fail_count}/{max_fail_count} ...")
                assert fail_count < max_fail_count, "Failed to initialize capture"
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            ret_corners, corners = cv2.findChessboardCorners(gray, target_size, None)
            if ret_corners:
                
                subpix_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)
                
                # per region calculation
                target_min = np.min(subpix_corners, axis=0).flatten()
                target_max = np.max(subpix_corners, axis=0).flatten()
                min_region = get_image_region(target_min, region_size)
                max_region = get_image_region(target_max, region_size)
                
                # only if the whole target is inside the region, count it, until region is full
                if min_region == max_region and per_region_conut[min_region] < per_region_requirement:
                    per_region_conut[min_region] += 1
                    
                    # subpix_corners = np.concatenate([subpix_corners.squeeze(), np.zeros((subpix_corners.shape[0], 1), np.float32)], axis=1)
                    # collected_image_points.append(subpix_corners)
                    collected_image_points.append(subpix_corners.squeeze())
                
                # return if all regions are full
                if all(count >= per_region_requirement for count in per_region_conut.values()):
                    print("All regions are full")
                    break
                
                # draw region
                for region, count in per_region_conut.items():
                    cv2.putText(
                        frame,
                        f"{count}/{per_region_requirement}",
                        (int(region[0] * region_size[0] + region_size[0] / 2), int(region[1] * region_size[1] + region_size[1] / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    region_top_left = int(region[0] * region_size[0]), int(region[1] * region_size[1])
                    region_bottom_right = int(region_top_left[0] + region_size[0]), int(region_top_left[1] + region_size[1])
                    
                    if count >= per_region_requirement:
                        cv2.rectangle(frame, region_top_left, region_bottom_right, (0, 0, 255), 6)
                    else:
                        cv2.rectangle(frame, region_top_left, region_bottom_right, (0, 255, 0), 6)
                
                # draw corners
                cv2.drawChessboardCorners(frame, target_size, corners, ret_corners)
            
            # Display the resulting frame
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            print(f"fps: {1/(time.perf_counter()-dt)}")
        
    except:
        raise
    
    cv2.destroyAllWindows()
    return collected_image_points


# if __name__ == "__main__":

#     num_regions = (1, 1)
#     per_region_requirement = 30
    
#     initial_focal_length = max(capture_width, capture_height)
#     camera_matrix = np.array(
#         [
#             [initial_focal_length, 0, capture_width / 2],
#             [0, initial_focal_length, capture_height / 2],
#             [0, 0, 1]
#         ]
#     )
#     dist_coeffs = np.zeros((5, 1), np.float32)
    
#     video_capture = cv2.VideoCapture(1)
#     video_capture.set(cv2.CAP_PROP_FPS, 30)
#     video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
#     video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
    
    
#     try:
        
#         # prepare object points
#         object_point = np.zeros((target_size[0] * target_size[1], 3), np.float32)
#         object_point[:, :2] = np.mgrid[:target_size[0], :target_size[1]].T.reshape(-1,2)
#         object_points = num_regions[0] * num_regions[1] * per_region_requirement * [object_point]
        
#         # collect target points
#         image_points = collect_target_points(
#             video_capture, 
#             target_size=target_size,
#             per_region_requirement=per_region_requirement, 
#             num_regions=num_regions)
        
        
#         object_points = np.array(object_points, dtype=np.float32)
#         image_points = np.array(image_points, dtype=np.float32)
        
#         print(f"shape of object points: {object_points.shape}")
#         print(f"shape of image points: {image_points.shape}")
        
        
#         out = {}
#         for i in range(object_points.shape[0]):
            
#             ret, rvec, tvec = cv2.solvePnP(
#                 objectPoints=object_points[i],
#                 imagePoints=image_points[i],
#                 cameraMatrix=camera_matrix,
#                 distCoeffs=dist_coeffs,
#                 flags=cv2.SOLVEPNP_ITERATIVE
#                 # flags=cv2.SOLVEPNP_SQPNP
#             )
            
#             if not ret:
#                 continue
        
#             reprojected_points, jacobian = cv2.projectPoints(object_points[i], rvec, tvec, camera_matrix, dist_coeffs)
            
            
            
#             out[i] = {
#                 "rvec": rvec.squeeze().tolist(),
#                 "tvec": tvec.squeeze().tolist(),
#                 "reprojected_points": reprojected_points.squeeze().tolist(),
#             }
            
#         with open("calibration.json", "w") as f:
#             json.dump(out, f)

        
#         # print("Calibrating camera ...")
#         # rmse, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, (capture_width, capture_height), None, None)
        
#         # camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (capture_width, capture_height), 1, (capture_width, capture_height))
        
#         # print(f"Camera matrix: {camera_matrix}")
#         # print(f"Distortion coefficients: {dist}")
#         # print(f"ROI: {roi}")
#         # print(f"Rvecs: {rvecs}")
#         # print(f"Tvecs: {tvecs}")
        
#     except:
#         raise
#     finally:
#         video_capture.release()
#         cv2.destroyAllWindows()



def prepare_checkerboard_model_points(
    num_targets: tuple[int, int] = (3, 4),
    targets_size = (0.05, 0.055)):
    """
    Prepare model points (object points) for checkerboard calibration.
    
    params:
        capture_size = (width, height) size of capture
        num_targets = (width, height) number of corners on checkerboard
        target_size = (width, height) size of each corner on checkerboard in meters (m)
    """
    
    model_points = np.mgrid[:num_targets[0], :num_targets[1]].T.reshape(-1,2)
    return model_points * targets_size



# PARAMETERS
capture_size = (1080, 1920) # (height, width) of the capture camera

model_points = prepare_checkerboard_model_points()


effective_focal_length = (1080, 1920) # (height, width) focal length in pixels, because pixel density might be different in height and width and are also unknown
principle_point = (int(1080/2), int(1920/2))

calibration_matrix = np.array(
    [
        [effective_focal_length[0], 0, principle_point[0]],
        [0, effective_focal_length[1], principle_point[1]],
        [0, 0, 1]
    ]
)
rotation_matrix = np.eye(3)
translation_vector = np.zeros((3, 1))


def project_world_point_to_camera(
    world_point,
    calibration_matrix,
    rotation_matrix,
    translation_vector
):
    intrinsic_matrix = np.concatenate([calibration_matrix, np.zeros((3, 1))], axis=1)
    extrinsic_matrix = np.concatenate([np.concatenate([rotation_matrix, translation_vector], axis=1), np.array([[0, 0, 0, 1]])], axis=0)
    
    return intrinsic_matrix @ extrinsic_matrix @ np.concatenate([world_point, np.array([1])], axis=0)



camera_point = project_world_point_to_camera(
    world_point=np.array([0.5, 0.55, 0]),
    calibration_matrix=calibration_matrix,
    rotation_matrix=rotation_matrix,
    translation_vector=translation_vector
)

print(camera_point)