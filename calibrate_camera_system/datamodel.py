from json import load, dump, JSONEncoder
from numpy import ndarray, array
from pydantic import BaseModel, field_validator
from typing import Enum, Tuple, Any, List

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

class CalibrationTarget(BaseModel):
    target_type: TargetType
    num_target_corners_wh: Tuple[int, int] # (width, height) number of targets on the checkerboard
    target_size_wh_m: Tuple[float, float] # (width, height) of each target on the checkerboard in meters (m)

class TargetData(BaseModel):
    image_name: str
    camera_name: str
    target_points: Any
    
    @field_validator("target_points")
    def check_target_points(cls, value):
        if not isinstance(value, ndarray):
            raise TypeError("target_points must be a numpy array")
        return value

class ModelIntrinsics(BaseModel):
    camera_name: str
    rmse: float
    calibration_matrix: Any
    distortion_coefficients: Any
    
    def save_to_file(self, file_path: str):
        with open(file_path, "w") as f:
            dump(self.model_dump(), f, cls=NumpyJsonCodec)
            
    @staticmethod
    def load_from_file(file_path: str):
        with open(file_path, "r") as f:
            return ModelIntrinsics(**load(f, object_hook=NumpyJsonCodec.decode_dict))
    
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

class ModelExtrinsics(BaseModel):
    from_camera: str
    to_camera: str
    rmse: float
    rotation_mat: Any
    translation_vec: Any
    essential_mat: Any
    fundamental_mat: Any
    
    @field_validator("rotation_mat")
    def check_rotation_matrix(cls, value):
        if not isinstance(value, ndarray):
            raise TypeError("rotation_matrix must be a numpy array")
        return value
    
    @field_validator("translation_vec")
    def check_translation_vector(cls, value):
        if not isinstance(value, ndarray):
            raise TypeError("translation_vector must be a numpy array")
        return value
    
    @field_validator("essential_mat")
    def check_essential_matrix(cls, value):
        if not isinstance(value, ndarray):
            raise TypeError("essential_matrix must be a numpy array")
        return value
    
    @field_validator("fundamental_mat")
    def check_fundamental_matrix(cls, value):
        if not isinstance(value, ndarray):
            raise TypeError("fundamental_matrix must be a numpy array")
        return value
    