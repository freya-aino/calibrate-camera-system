import time
import cv2
import numpy as np


# criteria for subpixel corner estimation
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points
object_points = np.zeros((6*7,3), np.float32)
object_points[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)


# define video capture
capture_width = 1920
capture_height = 1080
video_capture = cv2.VideoCapture(1)
video_capture.set(cv2.CAP_PROP_FPS, 30)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)


# define image regions (height, width), and utility function
num_regions = 3, 2
region_size = int(capture_width / num_regions[0]), int(capture_height / num_regions[1])
def get_image_region(image_position: tuple[int, int], region_size: tuple[int, int]) -> tuple[int, int]:
    # estimate which region the image position is in
    region_position = int(image_position[0] / region_size[0]), int(image_position[1] / region_size[1])
    return region_position


try:
    while True:
        
        dt = time.perf_counter()
        
        # Capture frame-by-frame
        ret_capture, frame = video_capture.read()
        
        if not ret_capture:
            print("No frame")
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        ret_corners, corners = cv2.findChessboardCorners(gray, (4, 3), None)
        if ret_corners:
            
            subpix_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            print(f"corners: {subpix_corners}")
            
            
            target_position = np.mean(subpix_corners, axis=0).flatten()
            print(f"target: {target_position}")
            
            
            region = get_image_region(target_position, region_size)
            print(f"region: {region}")
            print(f"region location: {region[0] * region_size[0]}, {region[1] * region_size[1]}")
            
            # draw current region
            region_top_left = int(region[0] * region_size[0]), int(region[1] * region_size[1])
            region_bottom_right = int(region_top_left[0] + region_size[0]), int(region_top_left[1] + region_size[1])
            cv2.rectangle(frame, region_top_left, region_bottom_right, (0, 0, 255), 6)
            
            # draw corners
            cv2.drawChessboardCorners(frame, (4, 3), corners, ret_corners)
        
        # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        print(f"fps: {1/(time.perf_counter()-dt)} ::: {frame.shape}")
    
except:
    raise
finally:
    video_capture.release()
    cv2.destroyAllWindows()