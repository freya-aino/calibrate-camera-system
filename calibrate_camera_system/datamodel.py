import os
from json import load, dump, JSONEncoder
from numpy import ndarray, array
from pydantic import BaseModel, field_validator
from typing import Tuple, Any, List, Dict
from enum import Enum

from device_capture_system.datamodel import CameraDevice

class NumpyJsonCodec(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
    @staticmethod
    def decode_dict(dct):
        for key, value in dct.items():
            if isinstance(value, list):
                dct[key] = array(value)
        return dct

class TargetType(Enum):
    CHECKERBOARD = "CHECKERBOARD"
    # CIRCLES_GRID = "CIRCLES_GRID"
    # ASYMMETRIC_CIRCLES_GRID = "ASYMMETRIC_CIRCLES_GRID"

class TargetParameters(BaseModel):
    target_type: TargetType
    num_target_corners_wh: Tuple[int, int] # (width, height) number of targets on the checkerboard
    target_size_wh_m: Tuple[float, float] # (width, height) of each target on the checkerboard in meters (m)
    
    @staticmethod
    def load(target_parameter_dir: str, target_type: str):
        with open(os.path.join(target_parameter_dir, target_type, "target.json"), "r") as f:
            j_obj = load(f)
            return TargetParameters(
                target_type = TargetType(j_obj["target_type"].upper()),
                num_target_corners_wh=tuple(j_obj["num_target_corners_wh"]),
                target_size_wh_m=tuple(j_obj["target_size_wh_m"])
            )

class TargetData(BaseModel):
    image_name: str
    target_points: Any
    
    @field_validator("target_points")
    def check_target_points(cls, value):
        if not isinstance(value, ndarray):
            raise TypeError("target_points must be a numpy array")
        return value

class TargetDataset(BaseModel):
    camera: CameraDevice
    target_data: List[TargetData]
    
    def save(self, file_path: str):
        with open(os.path.join(file_path, f"{self.camera.name}.json"), "w") as f:
            targets = [data.model_dump() for data in self.target_data]
            dump(targets, f, cls=NumpyJsonCodec)
    
    @staticmethod
    def load(file_path: str, camera: CameraDevice):
        with open(os.path.join(file_path, f"{camera.name}.json"), "r") as f:
            raw = load(f, object_hook=NumpyJsonCodec.decode_dict)
        return TargetDataset(camera = camera, target_data = [TargetData(**data) for data in raw])

class Intrinsics(BaseModel):
    camera: CameraDevice
    rmse: float
    calibration_matrix: Any
    distortion_coefficients: Any
    
    def save(self, file_path: str):
        with open(os.path.join(file_path, f"{self.camera.name}.json"), "w") as f:
            dump(self.model_dump(), f, cls=NumpyJsonCodec)
    
    @staticmethod
    def load(file_path: str, camera: CameraDevice):
        with open(os.path.join(file_path, f"{camera.name}.json"), "r") as f:
            return Intrinsics(**load(f, object_hook=NumpyJsonCodec.decode_dict))
    
    @field_validator("calibration_matrix")
    def check_calibration_matrix(cls, value):
        if not isinstance(value, ndarray):
            raise TypeError("calibration_matrix must be a numpy array")
        return value
    
    @field_validator("distortion_coefficients")
    def check_distortion_coefficients(cls, value):
        if not isinstance(value, ndarray):
            raise TypeError("distortion_coefficients must be a numpy array")
        return value

class Extrinsics(BaseModel):
    camera_from: CameraDevice
    camera_to: CameraDevice
    rmse: float
    rotation_matrix: Any
    translation_vector: Any
    # essential_mat: Any
    # fundamental_mat: Any
    
    def save(self, file_path: str):
        with open(os.path.join(file_path, f"{self.camera_from.name}-{self.camera_to.name}.json"), "w") as f:
            dump(self.model_dump(), f, cls=NumpyJsonCodec)
    
    @staticmethod
    def load(file_path: str, camera_from: CameraDevice, camera_to: CameraDevice):
        with open(os.path.join(file_path, f"{camera_from.name}-{camera_to.name}.json"), "r") as f:
            return Extrinsics(**load(f, object_hook=NumpyJsonCodec.decode_dict))
    
    @field_validator("rotation_matrix")
    def check_rotation_matrix(cls, value):
        if not isinstance(value, ndarray):
            raise TypeError("rotation_matrix must be a numpy array")
        return value
    
    @field_validator("translation_vector")
    def check_translation_vector(cls, value):
        if not isinstance(value, ndarray):
            raise TypeError("translation_vector must be a numpy array")
        return value
    
    # @field_validator("essential_mat")
    # def check_essential_matrix(cls, value):
    #     if not isinstance(value, ndarray):
    #         raise TypeError("essential_matrix must be a numpy array")
    #     return value
    
    # @field_validator("fundamental_mat")
    # def check_fundamental_matrix(cls, value):
    #     if not isinstance(value, ndarray):
    #         raise TypeError("fundamental_matrix must be a numpy array")
    #     return value


# class CameraModel(BaseModel):
#     camera: CameraDevice
#     intrinsics: Intrinsics
#     extrinsics: Dict[str, Extrinsics]



