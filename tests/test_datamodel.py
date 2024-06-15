import json
import pytest
import numpy as np

from calibrate_camera_system import datamodel

def test_Intrinsics():
    datamodel.Extrinsics(
        from_camera="camera1",
        to_camera="camera2",
        rmse=0.1,
        rotation_mat=np.random.rand(3, 3),
        translation_vec=np.random.rand(3),
        essential_mat=np.random.rand(3, 3),
        fundamental_mat=np.random.rand(3, 3)
    )

@pytest.fixture
def test_json_path(tmp_path):
    with open(tmp_path / "test.json", "w") as f:
        json.dump(
            {
                "camera_name": "camera1",
                "rmse": 0.1,
                "calibration_matrix": np.random.rand(3, 3).tolist(),
                "distortion_coefficients": np.random.rand(3).tolist()
            },
            f
        )
    return tmp_path / "test.json"

def test_Intrinsics_load(test_json_path):
    
    datamodel.Intrinsics.load_from_file(test_json_path)
