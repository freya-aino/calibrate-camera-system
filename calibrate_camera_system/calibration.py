import cv2
import numpy as np
import os

from multiprocessing import Pool
from tqdm import tqdm
from logging import getLogger
from typing import List, Tuple, Dict
from device_capture_system.datamodel import CameraDevice

from .datamodel import Extrinsics, Intrinsics, TargetData, TargetParameters, TargetDataset


def Z_norm(array: np.ndarray, axis: int = 0) -> np.ndarray:
    return (array - array.mean(axis=axis)) / array.std(axis=axis)

def generate_model_points(target_parameters: TargetParameters, num_target_points: int):
    target_corners_w = target_parameters.num_target_corners_wh[0]
    target_corners_h = target_parameters.num_target_corners_wh[1]
    target_size = target_parameters.target_size_wh_m
    model_points = np.mgrid[:target_corners_w, :target_corners_h].T.reshape(-1,2)
    model_points = model_points * target_size
    model_points = np.concatenate([model_points, np.zeros([model_points.shape[0], 1])], axis=1).astype(np.float32)
    model_points = np.array(num_target_points * [model_points], dtype=np.float32)
    return model_points

class IntrinsicsCalibrationManager:
    
    def __init__(self, capture_size: Tuple[int, int], target_parameters: TargetParameters):
        self.logger = getLogger(self.__class__.__name__)
        
        self.target_parameters = target_parameters
        self.capture_size = capture_size
        
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
    
    @staticmethod
    def calibrate_camera_worker(
        model_points: np.ndarray,
        target_points: np.ndarray,
        image_size: Tuple[int, int],
        calibration_matrix: np.ndarray,
        dist_coeffs: np.ndarray):
        
        return cv2.calibrateCamera(
            objectPoints=model_points,
            imagePoints=target_points, 
            imageSize=image_size, 
            cameraMatrix=calibration_matrix, 
            distCoeffs=dist_coeffs)
    
    def calibrate(self, target_datasets: List[TargetDataset], batch_size: int = 30, optim_iterations: int = 3, multiprocessing_workers: int = 16):
        process_pool = Pool(multiprocessing_workers)
        
        self.logger.info(f"intrinsics calibration: batch size = {batch_size}, iterations = {optim_iterations} on {multiprocessing_workers} workers ...")
        
        # run calibration
        process_results = {ds.camera.name: [] for ds in target_datasets}
        try:
            for target_dataset in target_datasets:
            
                # generate target and initialize model points for each target
                target_data = target_dataset.target_data
                target_points = np.stack([target.target_points for target in target_data], axis=0)
                num_target_points = len(target_data)
                
                batches_indecies = [np.random.randint(num_target_points, size=batch_size) for _ in range(optim_iterations)]
                model_points_batch = generate_model_points(self.target_parameters, batch_size).astype(np.float32)
                target_points_batches = [target_points[batch_indexes].astype(np.float32) for batch_indexes in batches_indecies]
                
                
                for batch in target_points_batches:
                    # calibrate camera
                    res_ = process_pool.apply_async(
                        self.calibrate_camera_worker, 
                        args = (
                            model_points_batch,
                            batch,
                            self.capture_size,
                            self.initial_calibration_matrix,
                            self.initial_dist_coeffs
                        ))
                    process_results[target_dataset.camera.name].append(res_)
            
            # collect results
            results = {}
            for cam_name in process_results:
                results[cam_name] = [res.get() for res in tqdm(process_results[cam_name], desc=f"calibrating {cam_name} intrinsics")]
            
        except Exception as e:
            raise e
        finally:
            self.logger.info("closing process pool ...")
            process_pool.close()
            for cam_name in process_results:
                for res in process_results[cam_name]:
                    res.wait()
            process_pool.join()
        
        
        # format output
        outputs = []
        for ds in target_datasets:
            
            res = results[ds.camera.name]
            
            # results return in format: rmse, calibration_matrix_, dist_coeffs_, r_vecs, t_vecs
            outputs.append(Intrinsics(
                camera = ds.camera,
                rmse = sum([res[0] for res in res]) / len(res),
                calibration_matrix = sum([res[1] for res in res]) / len(res),
                distortion_coefficients = np.sum([res[2].flatten() for res in res], axis=0) / len(res)
            ))
        
        return outputs
        
    def filter_targets(self, target_dataset: TargetDataset, point_top_std_exclusion_percentle: float = 10, target_top_inverse_distance_exclusion_percentile: float = 20):
        
        # the targets are normalized locally to make their variance uniform over all targets
        # then the points with high variance for each target are flagged as outliers
        normalised_patterns = np.stack([Z_norm(target.target_points) for target in target_dataset.target_data], axis=0)
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
        
        return TargetDataset(camera=target_dataset.camera, target_data=targets)


class ExtrinsicsCalibrationManager:
    
    def __init__(self, cameras: List[CameraDevice], target_parameters: TargetParameters, capture_size: Tuple[int, int]):
        self.logger = getLogger(self.__class__.__name__)
        
        self.cameras = cameras
        self.target_parameters = target_parameters
        self.capture_size = capture_size
        
    def calibrate(self, camera_intrinsics: Dict[str, Intrinsics], target_data: TargetDataset, batch_size: int = 30, optim_iterations: int = 3):
        
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
        model_points = generate_model_points(self.target_parameters, len(all_frames))
        
        # calibrate extrinsics
        extrinsic_outputs = []
        for from_camera in self.cameras:
            
            # get from_camera intrinsics
            from_cam_matrix = camera_intrinsics[from_camera.name].calibration_matrix
            from_cam_dist_coeffs = camera_intrinsics[from_camera.name].distortion_coefficients
            
            for to_camera in self.cameras:
                
                # append inert transformation if from_camera == to_camera
                if from_camera == to_camera:
                    extrinsic_outputs.append(Extrinsics(
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
                for i in tqdm(range(optim_iterations), desc=f"calibrating extrinsics {from_camera.name} -> {to_camera.name}", unit="iteration"):
                    
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
                
                extrinsic_outputs.append(Extrinsics(
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