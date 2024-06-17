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
    
    def __init__(self, target_parameters: TargetParameters, capture_size: Tuple[int, int]):
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
    
    def calibrate(self, target_datasets: List[TargetDataset], batch_size: int = 30, optim_iterations: int = 3, multiprocessing_workers: int = 12):
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
        
        targets = target_dataset.target_data
        
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
    
    def __init__(self, target_parameters: TargetParameters, capture_size: Tuple[int, int]):
        self.logger = getLogger(self.__class__.__name__)
        
        # self.cameras = cameras
        self.target_parameters = target_parameters
        self.capture_size = capture_size
        
        self.initial_rotation_mat = np.eye(3)
        self.initial_translation_vec = np.zeros((3), np.float32)
    
    def inner_join_datasets_and_format(self, target_datasets: List[TargetDataset]):
        
        # collect points by image name
        target_points_by_image_name = {}
        for ds in target_datasets:
            for target in ds.target_data:
                if target.image_name not in target_points_by_image_name:
                    target_points_by_image_name[target.image_name] = {}
                target_points_by_image_name[target.image_name][ds.camera.name] = target.target_points
        
        # remove images without targets in all cameras
        all_targets_points = []
        for frames_at_t in target_points_by_image_name.values():
            if len(frames_at_t) != len(target_datasets):
                continue
            all_targets_points.append(frames_at_t)
        
        # collect all frames by camera
        output_datasets = []
        for ds in target_datasets:
            output_datasets.append(
                {
                    "camera": ds.camera,
                    "target_data": np.stack([target[ds.camera.name] for target in all_targets_points], axis=0)
                })
        
        return output_datasets
    
    @staticmethod
    def calibrate_stereo_worker(
        model_points: np.ndarray,
        cam_from_target_points: np.ndarray,
        cam_to_target_points: np.ndarray,
        cam_from_intrinsics: Intrinsics,
        cam_to_intrinsics: Intrinsics,
        capture_size: Tuple[int, int]
    ):
        rmse, _, _, _, _, rot, trans, _, _ = cv2.stereoCalibrate(
            objectPoints  = model_points,
            imagePoints1  = cam_from_target_points,
            imagePoints2  = cam_to_target_points,
            imageSize     = capture_size,
            cameraMatrix1 = cam_from_intrinsics.calibration_matrix,
            distCoeffs1   = cam_from_intrinsics.distortion_coefficients,
            cameraMatrix2 = cam_to_intrinsics.calibration_matrix,
            distCoeffs2   = cam_to_intrinsics.distortion_coefficients,
            flags=cv2.CALIB_FIX_INTRINSIC
        )
        return rmse, rot, trans.flatten()
    
    def calibrate(self, target_datasets: List[TargetDataset], camera_intrinsics: List[Intrinsics], batch_size: int = 30, optim_iterations: int = 3, multiprocessing_workers: int = 16):
        process_pool = Pool(multiprocessing_workers)
        
        # calibrate extrinsics
        process_results = {}
        outputs = []
        
        try:
            
            camera_intrinsics_lut = {intrinsics.camera.name: intrinsics for intrinsics in camera_intrinsics}
            inner_joined_datasets = self.inner_join_datasets_and_format(target_datasets)
            
            self.logger.info(f"calibrating extrinsics: batch size = {batch_size}, iterations = {optim_iterations} on {multiprocessing_workers} workers ...")
            self.logger.info(f"number of datapoints with points in all cameras: {len(inner_joined_datasets[0]['target_data'])}")
            
            model_points = generate_model_points(self.target_parameters, batch_size)
            
            for camera_from_dataset in inner_joined_datasets:
                for camera_to_dataset in inner_joined_datasets:
                    
                    # if same camera append null transformation
                    if camera_from_dataset["camera"] is camera_to_dataset["camera"]:
                        outputs.append(
                            Extrinsics(
                                camera_from = camera_from_dataset["camera"],
                                camera_to = camera_to_dataset["camera"],
                                rmse = 0,
                                rotation_matrix = self.initial_rotation_mat,
                                translation_vector = self.initial_translation_vec
                            ))
                        continue
                    
                    batches_indicies = [np.random.randint(camera_from_dataset["target_data"].shape[0], size=batch_size) for _ in range(optim_iterations)]
                    
                    camera_from_batches = [camera_from_dataset["target_data"][batch_i].astype(np.float32) for batch_i in batches_indicies]
                    camera_to_batches = [camera_to_dataset["target_data"][batch_i].astype(np.float32) for batch_i in batches_indicies]
                    
                    for i in range(optim_iterations):
                        res_ = process_pool.apply_async(
                            self.calibrate_stereo_worker,
                            args = (
                                model_points,
                                camera_from_batches[i],
                                camera_to_batches[i],
                                camera_intrinsics_lut[camera_from_dataset["camera"].name],
                                camera_intrinsics_lut[camera_to_dataset["camera"].name],
                                self.capture_size
                            ))
                        key = (camera_from_dataset["camera"].name, camera_to_dataset["camera"].name)
                        if key not in process_results:
                            process_results[key] = []
                        process_results[key].append(res_)
                    
            
            # collect results and format into Extrinsics
            for k in process_results:
                results = [res.get() for res in tqdm(process_results[k], desc=f"calibrating  extrinsics {k}")]
                
                # results return in format: rmse, rotation_mat, translation_vec
                outputs.append(
                    Extrinsics(
                        camera_from = camera_intrinsics_lut[k[0]].camera,
                        camera_to = camera_intrinsics_lut[k[1]].camera,
                        rmse = sum([res[0] for res in results]) / len(results),
                        rotation_matrix = np.sum([res[1] for res in results], axis=0) / len(results),
                        translation_vector = np.sum([res[2] for res in results], axis=0) / len(results)
                    )
                )
            
        except Exception as e:
            raise e
        finally:
            self.logger.info("closing process pool ...")
            process_pool.close()
            for k in process_results:
                for res in process_results[k]:
                    res.wait()
            process_pool.join()
        
        return outputs