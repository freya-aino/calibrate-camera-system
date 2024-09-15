import os
import json
import pytest
import numpy as np
from unittest.mock import Mock

from calibrate_camera_system import datamodel

# --------------------- JSON CODEC ---------------------

def test_encode_numpy_array():
    encoded = json.dumps(
        obj = {'data': np.array([[1, 2, 3], [4, 5, 6]])}, 
        cls = datamodel.NumpyJsonCodec
    )
    
    expected_json = '{"data": [[1, 2, 3], [4, 5, 6]]}'
    assert encoded == expected_json

def test_decode_numpy_array():
    json_str = '{"data": [[1, 2, 3], [4, 5, 6]]}'
    expected_array = np.array([[1, 2, 3], [4, 5, 6]])
    
    decoded = json.loads(json_str, object_hook=datamodel.NumpyJsonCodec.decode_dict)
    
    assert 'data' in decoded
    assert isinstance(decoded['data'], np.ndarray)
    assert np.array_equal(decoded['data'], expected_array)

# --------------------- TARGET PARAMETERS ---------------------

@pytest.fixture
def target_json(tmp_path):
    target_dir = tmp_path / "target_dir"
    target_type = "checkerboard"
    
    target_dir.mkdir()
    target_type_dir = target_dir / target_type
    target_type_dir.mkdir()
    
    target_json_path = target_type_dir / "target.json"
    target_data = {
        "target_type": target_type,
        "num_target_corners_wh": [9, 6],
        "target_size_wh_m": [0.025, 0.025]
    }
    
    with open(target_json_path, "w") as f:
        json.dump(target_data, f)
    
    return target_json_path

def test_load_target_parameters(target_json):
    target_parameter_dir = os.path.dirname(os.path.dirname(target_json))
    target_type = "checkerboard"
    
    target_params = datamodel.TargetParameters.load(target_parameter_dir, target_type)
    
    assert target_params.target_type == datamodel.TargetType.CHECKERBOARD
    assert target_params.num_target_corners_wh == (9, 6)
    assert target_params.target_size_wh_m == (0.025, 0.025)

# --------------------- TARGET DATA ---------------------

def test_target_data_validation():
    valid_points = np.array([[1, 2], [3, 4]])
    type_exceptions = [
        [[1, 2], [3, 4]]
    ]
    value_exceptions = [
        np.array([[1, 2, 3], [4, 5, 6]]),
        np.array([[0], [1], [2], [3]]),
        np.arange(10)
    ]
    
    # Valid target points
    target_data = datamodel.TargetData(image_name="test_image.jpg", target_points=valid_points)
    assert isinstance(target_data.target_points, np.ndarray)
    
    # Invalid target points
    for p in type_exceptions:
        with pytest.raises(TypeError):
            datamodel.TargetData(image_name="test_image.jpg", target_points=p)
    for p in value_exceptions:
        with pytest.raises(ValueError):
            datamodel.TargetData(image_name="test_image.jpg", target_points=p)

# --------------------- TARGET DATASET ---------------------

@pytest.fixture
def camera_device():
    mock_camera = Mock(spec=datamodel.CameraDevice)
    mock_camera.name = "test_camera"
    return mock_camera

@pytest.fixture
def target_data_list():
    return [
        datamodel.TargetData(image_name="image1.jpg", target_points=np.random.rand(10, 2)),
        datamodel.TargetData(image_name="image2.jpg", target_points=np.random.rand(10, 2)),
    ]

def test_target_dataset_save_load(tmp_path, camera_device, target_data_list):
    dataset = datamodel.TargetDataset(camera=camera_device, target_data=target_data_list)
    
    # Save the dataset
    dataset.save(tmp_path)
    
    # Load the dataset
    loaded_dataset = datamodel.TargetDataset.load(tmp_path, camera_device)
    
    assert loaded_dataset.camera.name == dataset.camera.name
    assert len(loaded_dataset.target_data) == len(dataset.target_data)
    
    for original, loaded in zip(dataset.target_data, loaded_dataset.target_data):
        assert original.image_name == loaded.image_name
        assert np.array_equal(original.target_points, loaded.target_points)

# --------------------- INTRINSICS / EXTRINSICS / CAMERA MODEL INFO ---------------------

@pytest.fixture
def mock_camera_devices():
    mock_cameras = [
        Mock(spec=datamodel.CameraDevice),
        Mock(spec=datamodel.CameraDevice),
    ]
    for i in range(len(mock_cameras)):
        mock_cameras[i].name = f"camera{i}"
        mock_cameras[i].device_id = str(i)
        mock_cameras[i].width = 1920
        mock_cameras[i].height = 1080
        mock_cameras[i].fps = 30
    
    return mock_cameras

def test_intrinsics_save_load(tmp_path, mock_camera_devices):
    intrinsics = datamodel.Intrinsics(
        camera=mock_camera_devices[0],
        rmse=0.1,
        calibration_matrix=np.random.rand(3, 3),
        distortion_coefficients=np.random.rand(5)
    )
    
    # Save the intrinsics
    intrinsics.save(tmp_path)
    
    # Load the intrinsics
    loaded_intrinsics = datamodel.Intrinsics.load(tmp_path, mock_camera_devices[0])
    
    assert loaded_intrinsics.camera.name == intrinsics.camera.name
    assert loaded_intrinsics.rmse == intrinsics.rmse
    assert np.array_equal(loaded_intrinsics.calibration_matrix, intrinsics.calibration_matrix)
    assert np.array_equal(loaded_intrinsics.distortion_coefficients, intrinsics.distortion_coefficients)


def test_extrinsics_save_load(tmp_path, mock_camera_devices):
    camera_from, camera_to = mock_camera_devices
    extrinsics = datamodel.Extrinsics(
        camera_from=camera_from,
        camera_to=camera_to,
        rmse=0.2,
        rotation_matrix=np.random.rand(3, 3),
        translation_vector=np.random.rand(3)
    )
    
    # Save the extrinsics
    extrinsics.save(tmp_path)
    
    # Load the extrinsics
    loaded_extrinsics = datamodel.Extrinsics.load(tmp_path, camera_from, camera_to)
    
    assert loaded_extrinsics.camera_from.name == extrinsics.camera_from.name
    assert loaded_extrinsics.camera_to.name == extrinsics.camera_to.name
    assert loaded_extrinsics.rmse == extrinsics.rmse
    assert np.array_equal(loaded_extrinsics.rotation_matrix, extrinsics.rotation_matrix)
    assert np.array_equal(loaded_extrinsics.translation_vector, extrinsics.translation_vector)


def test_camera_model_information_load(tmp_path, mock_camera_devices):
    
    # Create and save intrinsics for each camera
    for camera in mock_camera_devices:
        intrinsics = datamodel.Intrinsics(
            camera=camera,
            rmse=0.1,
            calibration_matrix=np.random.rand(3, 3),
            distortion_coefficients=np.random.rand(5)
        )
        intrinsics.save(tmp_path)
    
    # Create and save extrinsics for each camera pair
    for camera_from in mock_camera_devices:
        for camera_to in mock_camera_devices:
            extrinsics = datamodel.Extrinsics(
                camera_from=camera_from,
                camera_to=camera_to,
                rmse=0.2,
                rotation_matrix=np.random.rand(3, 3),
                translation_vector=np.random.rand(3)
            )
            extrinsics.save(tmp_path)
    
    # Load the camera model information
    camera_model_info = datamodel.CameraModelInformation.load(tmp_path, tmp_path, mock_camera_devices)
    
    assert len(camera_model_info.intrinsics) == len(mock_camera_devices)
    assert len(camera_model_info.extrinsics) == len(mock_camera_devices) ** 2
    
    for intrinsics in camera_model_info.intrinsics:
        assert intrinsics.camera.name in [camera.name for camera in mock_camera_devices]
    
    for extrinsics in camera_model_info.extrinsics:
        assert extrinsics.camera_from.name in [camera.name for camera in mock_camera_devices]
        assert extrinsics.camera_to.name in [camera.name for camera in mock_camera_devices]