import cv2
import numpy as np
import os

from logging import getLogger
from typing import List, Tuple, Dict
from device_capture_system.datamodel import CameraDevice

from .datamodel import CalibrationTarget, ModelExtrinsics, ModelIntrinsics, TargetData


def Z_norm(array: np.ndarray, axis: int = 0) -> np.ndarray:
    return (array - array.mean(axis=axis)) / array.std(axis=axis)

def generate_model_points(calibration_target: CalibrationTarget, num_target_points: int):
    target_corners_w = calibration_target.num_target_corners_wh[0]
    target_corners_h = calibration_target.num_target_corners_wh[1]
    target_size = calibration_target.target_size_wh
    model_points = np.mgrid[:target_corners_w, :target_corners_h].T.reshape(-1,2)
    model_points = model_points * target_size
    model_points = np.concatenate([model_points, np.zeros([model_points.shape[0], 1])], axis=1).astype(np.float32)
    model_points = np.array(num_target_points * [model_points], dtype=np.float32)
    return model_points

class IntrinsicsCalibrator:
    
    def __init__(self, camera: CameraDevice, capture_size: Tuple[int, int], calibration_target: CalibrationTarget, calibration_data_path: str = "./calibration_data"):
        self.logger = getLogger(self.__class__.__name__)
        
        self.camera = camera
        self.calibration_target = calibration_target
        self.capture_size = capture_size
        self.calibration_data_path = calibration_data_path
        
        # initial calibration matrix and distortion coefficients
        effective_focal_length = self.capture_size # (width, height) focal length in pixels, because pixel density might be different in height and width and are also unknown
        principle_point = (self.capture_size[0] / 2, self.capture_size[1] / 2)    
        self.initial_calibration_matrix = np.array(
            [
                [effective_focal_length[0], 1, principle_point[0]],
                [0, effective_focal_length[1], principle_point[1]],
                [0, 0, 1]
            ], np.float32
        )
        self.initial_dist_coeffs = np.zeros((5, 1), np.float32)
        
    def calibrate(self, target_data: List[TargetData], batch_size: int = 30, optim_iterations: int = 3, save: bool = False):
        
        # generate target and initialize model points for each target
        target_points = np.stack([target.target_points for target in target_data], axis=0)
        num_target_points = len(target_data)
        model_points = generate_model_points(self.calibration_target, num_target_points)
        
        # run calibration
        self.logger.info(f"intrinsics calibration: batch size = {batch_size}, iterations = {optim_iterations}")
        avg_calibration_matrix = np.zeros_like(self.initial_calibration_matrix)
        avg_dist_coeffs = np.zeros_like(self.initial_dist_coeffs)
        avg_rmse = 0
        for i in range(optim_iterations):
            
            # select batch
            rand_batch_indexes = np.random.randint(num_target_points, size=batch_size)
            model_points_batch = model_points[rand_batch_indexes, :, :]
            target_points_batch = target_points[rand_batch_indexes, :, :]
            
            # calibrate camera
            rmse, calibration_matrix_, dist_coeffs_, r_vecs, t_vecs = cv2.calibrateCamera(
                objectPoints=model_points_batch, 
                imagePoints=target_points_batch, 
                imageSize=self.capture_size, 
                cameraMatrix=self.initial_calibration_matrix, 
                distCoeffs=self.initial_dist_coeffs)
            
            avg_calibration_matrix += calibration_matrix_
            avg_dist_coeffs += dist_coeffs_
            avg_rmse += rmse
        
        avg_calibration_matrix = avg_calibration_matrix / optim_iterations
        avg_dist_coeffs = avg_dist_coeffs / optim_iterations
        avg_rmse = avg_rmse / optim_iterations
        
        self.logger.info(f"final average rmse = {avg_rmse}")
        
        intrinsic_calibration = ModelIntrinsics(
            camera_name=self.camera.name,
            rmse=avg_rmse,
            calibration_matrix=avg_calibration_matrix,
            distortion_coefficients=avg_dist_coeffs
        )
        
        if save:
            intrinsic_calibration.save_to_file(os.path.join(self.calibration_data_path, "calibration_intrinsics", f"{self.camera.name}.json"))
            self.logger.info(f"saved calibration intrinsics for {self.camera.name}")
        
        return intrinsic_calibration
        
    def filter_targets(self, targets: List[TargetData], point_top_std_exclusion_percentle: float = 10, target_top_inverse_distance_exclusion_percentile: float = 20):
        
        # the targets are normalized locally to make their variance uniform over all targets
        # then the points with high variance for each target are flagged as outliers
        normalised_patterns = np.stack([Z_norm(target.target_points) for target in targets], axis=0)
        per_point_std = np.sqrt((normalised_patterns - normalised_patterns.mean(axis=0))**2)
        top_std_percentile_threshold = np.percentile(per_point_std, (100 - point_top_std_exclusion_percentle), axis=0)
        valid_std_mask = per_point_std <= np.expand_dims(top_std_percentile_threshold, axis=0)
        
        self.logger.debug(f"normalised_patterns: {normalised_patterns}")
        self.logger.debug(f"pattern_std: {per_point_std}")
        self.logger.debug(f"top_std_percentile_threshold: {top_std_percentile_threshold}")
        self.logger.debug(f"valid_variance_mask: {valid_std_mask}")
        
        valid_targets = []
        for i in range(len(targets)):
            if not valid_std_mask[i].all():
                continue
            valid_targets.append(targets[i])
        targets = valid_targets
        
        if len(targets) == 0:
            self.logger.warning(f"no valid targets after removing high per point std targets")
            return
        self.logger.info(f"{len(targets)} valid targets after removing high per point std targets")
        
        
        # filter out images with targets means too close to each other
        target_means = np.stack([target.target_points.mean(axis=0) for target in targets], axis=0)
        normed_target_means = Z_norm(target_means)
        avg_total_distances_inv = np.stack([np.exp( - ((normed_target_means[i] - normed_target_means)**2).mean(axis=0)).mean(axis=0) for i in range(normed_target_means.shape[0])], axis=0)
        valid_distance_mask = avg_total_distances_inv < np.percentile(avg_total_distances_inv, 100 - target_top_inverse_distance_exclusion_percentile)
        
        self.logger.debug(f"normed_target_means: {normed_target_means}")
        self.logger.debug(f"valid_distance_mask: {valid_distance_mask}")
        self.logger.debug(f"target means: {target_means[valid_distance_mask]}")
        self.logger.debug(f"avg_total_distance: {avg_total_distances_inv}")
        
        valid_targets = []
        for i in range(len(targets)):
            if not valid_distance_mask[i]:
                continue
            valid_targets.append(targets[i])
        targets = valid_targets
        
        if len(targets) == 0:
            self.logger.warning(f"no valid targets after removing high per target mean std targets")
            return
        self.logger.info(f"{len(targets)} valid targets after removing high per target mean std targets")
        
        return targets


class ExtrinsicsCalibrator:
    
    def __init__(self, cameras: List[CameraDevice], calibration_target: CalibrationTarget, capture_size: Tuple[int, int]):
        self.logger = getLogger(self.__class__.__name__)
        
        self.cameras = cameras
        self.calibration_target = calibration_target
        self.capture_size = capture_size
        
        
    def calibrate(self, camera_intrinsics: Dict[str, ModelIntrinsics], target_data: Dict[str, List[CalibrationTarget]], batch_size: int = 30, optim_iterations: int = 3):
        
        assert all([cam.name in camera_intrinsics for cam in self.cameras]), f"all cameras must have intrinsics referenced: expected {[cam.name for cam in self.cameras]}, got {camera_intrinsics.keys()}"
        assert all([cam.name in target_data for cam in self.cameras]), f"all cameras must have calibration targets referenced: expected {[cam.name for cam in self.cameras]}, got {target_data.keys()}"
        
        # load targets and change exclude targets that are not represented in all cameras
        frames_by_image_id = {}
        for cam in self.cameras:
            for target in target_data[cam.name]:
                if target.image_name not in frames_by_image_id:
                    frames_by_image_id[target.image_name] = {}
                frames_by_image_id[target.image_name][target.camera_name] = target.target_points
        all_frames = []
        for cam_frames in frames_by_image_id.values():
            if len(cam_frames) != len(self.cameras):
                continue
            all_frames.append(cam_frames)
        all_valid_frames_by_cam = {}
        for cam in self.cameras:
            all_valid_frames_by_cam[cam.name] = np.array([cam_frame[cam.name] for cam_frame in all_frames], dtype=np.float32)
        
        self.logger.info(f"found {len(all_frames)} frames with targets in all cameras")
        
        
        # prepare model points
        model_points = generate_model_points(self.calibration_target, len(all_frames))
        
        # calibrate extrinsics
        extrinsic_outputs = []
        for from_camera in self.cameras:
            
            # get from_camera intrinsics
            from_cam_matrix = camera_intrinsics[from_camera.name].calibration_matrix
            from_cam_dist_coeffs = camera_intrinsics[from_camera.name].distortion_coefficients
            
            for to_camera in self.cameras:
                
                # append inert transformation if from_camera == to_camera
                if from_camera == to_camera:
                    extrinsic_outputs.append(ModelExtrinsics(
                        from_camera=from_camera.name,
                        to_camera=to_camera.name,
                        rmse=0,
                        rotation_mat=np.eye(3),
                        translation_vec=np.zeros((3, 1), np.float32),
                        essential_mat=np.zeros((3, 3), np.float32),
                        fundamental_mat=np.zeros((3, 3), np.float32)
                    ))
                    continue
                
                # get to_camera intrinsics
                to_cam_matrix = camera_intrinsics[to_camera.name].calibration_matrix
                to_cam_dist_coeffs = camera_intrinsics[to_camera.name].distortion_coefficients
                
                self.logger.info(f"stereo calibrating {from_camera.name} with camera {to_camera.name} stereo with {batch_size} samples per iteration for {optim_iterations} iterations ...")
                avg_rmse = 0
                avg_rotation_mat = np.zeros((3, 3), np.float32)
                avg_translation_vec = np.zeros((3, 1), np.float32)
                avg_essential_mat = np.zeros((3, 3), np.float32)
                avg_fundamental_mat = np.zeros((3, 3), np.float32)
                for i in range(optim_iterations):
                    
                    # select batch
                    rand_batch_indexes = np.random.randint(model_points.shape[0], size=batch_size)
                    model_points_batch = model_points[rand_batch_indexes, :, :]
                    from_cam_target_points_batch = all_valid_frames_by_cam[from_camera.name][rand_batch_indexes, :, :]
                    to_cam_points_batch = all_valid_frames_by_cam[to_camera.name][rand_batch_indexes, :, :]
                    
                    rmse, _, _, _, _, r, t, e, f = cv2.stereoCalibrate(
                        objectPoints=model_points_batch,
                        imagePoints1=from_cam_target_points_batch,
                        imagePoints2=to_cam_points_batch,
                        imageSize=self.capture_size,
                        cameraMatrix1=from_cam_matrix,
                        distCoeffs1=from_cam_dist_coeffs,
                        cameraMatrix2=to_cam_matrix,
                        distCoeffs2=to_cam_dist_coeffs,
                        flags=cv2.CALIB_FIX_INTRINSIC
                    )
                    
                    avg_rmse += rmse
                    avg_rotation_mat += r
                    avg_translation_vec += t
                    avg_essential_mat += e
                    avg_fundamental_mat += f
                
                rmse = avg_rmse / optim_iterations
                rotation_mat = avg_rotation_mat / optim_iterations
                translation_vec = avg_translation_vec / optim_iterations
                essential_mat = avg_essential_mat / optim_iterations
                fundamental_mat = avg_fundamental_mat / optim_iterations
                self.logger.info(f"average rmse: {avg_rmse}")
                
                extrinsic_outputs.append(ModelExtrinsics(
                    from_camera=from_camera.name,
                    to_camera=to_camera.name,
                    rmse=rmse,
                    rotation_mat=rotation_mat,
                    translation_vec=translation_vec,
                    essential_mat=essential_mat,
                    fundamental_mat=fundamental_mat
                ))
            
        self.logger.info("calibrating extrinsics complete!")
        return extrinsic_outputs