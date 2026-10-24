# Camera System Calibration

A Python toolkit for calibrating a multi-camera setup using a checkerboard. It captures calibration images through [device-capture-system](https://github.com/freya-aino/device-capture-system), detects checkerboard corners with OpenCV, estimates camera intrinsics and relative camera poses, and saves the results as JSON. The library also supports undistorting points, stereo triangulation, and reprojection into another camera's image.

The project is a work in progress. The command-line workflow contains configuration specific to the original three-camera setup; adapt it to your hardware before collecting data. Automated tests currently cover data models and serialization, not calibration accuracy or camera capture.

## Requirements and installation

- Python 3.10 or later, below Python 4 (the declared Python constraint).
- [Poetry](https://python-poetry.org/) for dependency management.
- Cameras supported by `device-capture-system`; use at least two for stereo calibration and triangulation.
- A printed, flat checkerboard with measured corner spacing.
- A graphical desktop for the OpenCV windows used by `--test-calibration`.

Runtime dependencies include OpenCV, `device-capture-system`, and `tqdm`. NumPy and Pydantic 2 are also used directly and currently arrive through the capture package's dependencies. Pytest is a development dependency.

Run commands from the repository root. There is no installed console entry point; the CLI is `run.py`.

### Resolve the capture dependency

The committed dependency metadata needs attention before a fresh installation:

- `pyproject.toml` points to a GitHub `blob` URL for `device_capture_system-1.1.0.tar.gz`, which is a file-view page rather than a direct archive URL.
- `poetry.lock` references a local archive at `../device-capture-system/dist/device_capture_system-1.1.0.tar.gz` instead.

Obtain the version 1.1.0 source archive from the capture project, then use its actual local path to align the dependency declaration and lock file:

```bash
poetry env use python3.10
poetry add /absolute/path/to/device_capture_system-1.1.0.tar.gz
poetry install --with dev
poetry run python run.py --help
```

Replace the archive path with the file on your machine. `poetry add` updates `pyproject.toml` and `poetry.lock`. Alternatively, configure a working direct-download URL and regenerate the lock file before installing. Capture backend and operating-system support depend on `device-capture-system`; the supplied device identifiers are Windows-specific examples.

## Configure the cameras

Edit [`devices.json`](devices.json). Each entry contains:

| Field | Meaning |
| --- | --- |
| `device_id` | Backend-specific camera identifier accepted by the capture library. |
| `name` | Unique, filesystem-safe camera name, also used for output directories and calibration filenames. |
| `device_type` | `"video"` for cameras. |
| `width`, `height` | Requested capture resolution before preprocessing. |
| `fps` | Requested capture frame rate. |

The supplied configuration describes three cameras capturing at 1920 × 1080 and 30 FPS. Replace their identifiers with your own, using the capture project's device discovery/configuration facilities. Keep camera names consistent between capture, detection, and calibration.

Also edit these constants in [`run.py`](run.py):

```python
FRAME_PREPROCESSINGS = [
    FramePreprocessing.ROTATE_90_CLOCKWISE,
    FramePreprocessing.ROTATE_90_CLOCKWISE,
    FramePreprocessing.ROTATE_90_COUNTERCLOCKWISE,
]
CAPTURE_SIZE = (1080, 1920)
```

Match the preprocessing entries to the cameras in configuration order. `CAPTURE_SIZE` is **(width, height) after preprocessing**; the default accounts for the 90-degree rotations. The CLI uses one shared capture size for all cameras. Camera configuration is always read from `./devices.json`, even when the calibration data directory is changed.

## Configure the checkerboard

Only checkerboards are implemented. A printable pattern can be generated with the [calib.io pattern generator](https://calib.io/pages/camera-calibration-pattern-generator). The repository includes a PDF and configuration under [`calibration_data/target_parameters/checkerboard/`](calibration_data/target_parameters/checkerboard/).

Edit `target.json` to match the physical target:

```json
{
  "target_type": "checkerboard",
  "num_target_corners_wh": [3, 4],
  "target_size_wh_m": [0.05, 0.055]
}
```

- `num_target_corners_wh` counts **inner corners**, in width/height order, not checkerboard squares. The supplied target has 12 detected points.
- `target_size_wh_m` gives horizontal and vertical spacing between adjacent corners in **metres**, not the total board size. The supplied spacing is 50 mm × 55 mm.
- Print without unintended scaling, mount the target flat, and measure the actual spacing. The PDF is a printing aid; the program only loads the JSON.

The CLI multiplies the configured spacing by 1000 before calibration. Consequently, CLI-generated translations and triangulated coordinates use **millimetres**. When using calibration classes directly, output length units follow the spacing supplied to them; the library does not perform that conversion automatically.

## Calibration workflow

Keep cameras fixed throughout capture and calibration. Move the checkerboard through different positions, distances, and tilts, including near image edges. Keep the full pattern visible, well lit, and sharp. For extrinsics, collect many frames in which every camera sees the same target pose. Capture is delegated to the capture library; this project does not establish hardware synchronization, so avoid rapid target motion.

### 1. Collect images

```bash
poetry run python run.py --collect-images \
  --num-frames-to-collect 200 \
  --frame-collection-interval 0.5
```

Images are saved under `calibration_data/images/<camera-name>/`. The count is the requested number of successful capture cycles, and the interval is a delay in seconds between attempts, not a guaranteed sampling period. Repeated capture errors can stop collection early.

### 2. Detect target corners

```bash
poetry run python run.py --find-targets
```

Detection uses `cv2.findChessboardCorners` followed by subpixel refinement. Detections with incomplete corner sets or rejected corner orientation are excluded. One JSON dataset per camera is written to `calibration_data/extracted_targets/`.

Optionally delete images that have no accepted detection:

```bash
poetry run python run.py --remove-images-without-targets
```

**This deletes image files permanently.** It uses the saved detection datasets, so run detection first and repeat it after adding images. Removal is per camera; it does not enforce visibility across all cameras.

### 3. Calibrate intrinsics

```bash
poetry run python run.py --calibrate-intrinsics
```

For each camera, the program filters target observations, runs `cv2.calibrateCamera` on random batches sampled with replacement, discards high-RMSE runs, and averages the remaining camera matrices and distortion coefficients. Results are saved to `calibration_data/intrinsics/`.

### 4. Calibrate extrinsics

```bash
poetry run python run.py --calibrate-extrinsics
```

This requires saved intrinsics and target datasets for every configured camera. The program filters the datasets and joins observations by **identical image filenames across all cameras**. Imported image sets must preserve this correspondence.

For each ordered camera pair, it runs `cv2.stereoCalibrate` with fixed intrinsics on random batches, filters high-RMSE results, and averages rotations and translations. Results include identity transforms for each camera to itself and are saved to `calibration_data/extrinsics/`.

Stages may be combined:

```bash
poetry run python run.py --find-targets \
  --calibrate-intrinsics --calibrate-extrinsics
```

Flags always execute in this order, regardless of their order on the command line: capture, detection, image removal, intrinsics, extrinsics, calibration test. Existing JSON outputs with the same names are overwritten.

### 5. Inspect reprojections

```bash
poetry run python run.py --test-calibration
```

This loads saved calibrations, triangulates shared detections, and prints mean reprojection RMSE values. It then displays ten randomly selected shared frames with detected corners in blue and reprojected points in green. Press a key in each OpenCV window to advance.

At least **ten frames with detections in every camera** are needed for the visualization's random sample. This is a diagnostic using the existing calibration dataset, not a held-out accuracy test. The current visualization uses the last configured camera as its reference view.

### Use a separate data directory

```bash
poetry run python run.py --calibration-data-path ./my_calibration --find-targets
```

First create `my_calibration/target_parameters/checkerboard/target.json` and provide any images required for the selected stage. The CLI creates output directories, but it does not create target parameters or copy existing data. The target parameter directory is required for every processing stage.

## Command-line reference

Run `poetry run python run.py --help` to list options. With no action flags, the script prints help and exits.

| Option | Default | Purpose |
| --- | --- | --- |
| `--logging-level` | `info` | One of `debug`, `info`, or `warning`. |
| `--calibration-data-path` | `./calibration_data` | Root directory for target parameters, images, and results. |
| `--collect-images` | Off | Capture images from configured cameras. |
| `--num-frames-to-collect` | `100` | Requested successful capture cycles. |
| `--frame-collection-interval` | `0.1` | Delay between capture attempts, in seconds. |
| `--find-targets` | Off | Detect and save checkerboard corners. |
| `--remove-images-without-targets` | Off | Delete images absent from saved target datasets. |
| `--calibrate-intrinsics` | Off | Estimate each camera's intrinsic parameters. |
| `--calibrate-extrinsics` | Off | Estimate relative poses using saved intrinsics. |
| `--calibration-iterations` | `100` | Number of random batch calibration runs; use more than one. |
| `--calibration-batch-size` | `30` | Observations per batch, sampled with replacement. |
| `--point-top-std-exclusion-percentle` | `10` | Percentile exclusion for deviations in normalized corner patterns. |
| `--target-top-inverse-distance-exclusion-percentile` | `20` | Percentile exclusion for spatially clustered target centres. |
| `--high-rmse-cutoff-percentile` | `20` | Upper percentage of calibration RMSE results to exclude. |
| `--test-calibration` | Off | Print reprojection errors and display diagnostic overlays. |

The spelling `--point-top-std-exclusion-percentle` is intentional here: it matches the current CLI typo. Filtering uses strict thresholds in some stages, so retained counts need not equal the nominal percentages. Empty datasets or no retained calibration runs are not handled gracefully.

## Files and output formats

```text
.
├── run.py                         # Command-line workflow and setup constants
├── devices.json                   # Camera configuration
├── pyproject.toml                 # Package and dependency declarations
├── poetry.lock                    # Dependency lock file
├── calibrate_camera_system/
│   ├── calibration.py             # Filtering and intrinsic/stereo calibration
│   ├── core.py                    # Image capture, triangulation, projection
│   ├── datamodel.py               # Pydantic models and NumPy JSON encoding
│   └── targets.py                 # Checkerboard detection
├── tests/test_datamodel.py         # Validation and serialization tests
└── calibration_data/
    ├── target_parameters/checkerboard/
    │   ├── target.json
    │   └── target.pdf
    ├── images/<camera-name>/
    ├── extracted_targets/<camera-name>.json
    ├── intrinsics/<camera-name>.json
    └── extrinsics/<from-camera>-<to-camera>.json
```

Generated data directories under the default calibration root are ignored by Git.

| Output | JSON contents |
| --- | --- |
| Extracted targets | A list of records with `image_name` and `target_points` (an N × 2 array of pixel coordinates). |
| Intrinsics | `camera`, `rmse`, `calibration_matrix` (3 × 3), and `distortion_coefficients` (OpenCV ordering). |
| Extrinsics | `camera_from`, `camera_to`, `rmse`, `rotation_matrix` (3 × 3), and `translation_vector` (three components). |

Arrays are stored as JSON lists and restored to NumPy arrays by the model loaders. Saved RMSE values are means of retained calibration-run errors, rather than errors recomputed for the averaged parameters. Extrinsic transforms follow the convention `X_to = R @ X_from + t`.

## Use the calibration in Python

After producing calibration files, load the camera model:

```python
from device_capture_system.deviceIO import load_all_devices_from_config
from calibrate_camera_system.core import CameraModel
from calibrate_camera_system.datamodel import CameraModelInformation

cameras = load_all_devices_from_config("video", config_file="./devices.json")
model = CameraModel(
    CameraModelInformation.load(
        intrinsics_save_path="./calibration_data/intrinsics",
        extrinsics_save_path="./calibration_data/extrinsics",
        cameras=cameras,
    )
)
```

The loader expects an intrinsic file for each camera and extrinsic files for **all ordered pairs, including self-pairs**.

Given corresponding N × 2 NumPy arrays `points_a` and `points_b` in the calibrated image orientation and resolution:

```python
camera_a, camera_b = cameras[0].name, cameras[1].name
points_3d = model.triangulate_points(
    points_from=points_a,
    points_to=points_b,
    cam_from_name=camera_a,
    cam_to_name=camera_b,
)
reprojected_b = model.project_points_to_view(
    points_3d, cam_from_name=camera_a, cam_to_name=camera_b
)
```

`points_3d` is an N × 3 array in **camera A's coordinate system**. `undistort_points` returns undistorted pixel coordinates. `unified_3D_model(points, main_cam_name)` triangulates each other camera against the selected reference and returns a dictionary keyed by the other camera's name. Each returned array remains in that source camera's coordinate system; transform the arrays to a common frame before combining them.

## Development and limitations

Run the existing tests after installing development dependencies:

```bash
poetry run pytest
```

Tests cover NumPy JSON encoding, target parameter loading, array validation, and calibration data save/load operations. Camera hardware, detection quality, and end-to-end calibration are outside the current automated coverage.

Current implementation constraints:

- Only checkerboard targets and the standard OpenCV camera model are implemented; there is no fisheye-specific calibration workflow.
- Random batch sampling has no CLI seed, so repeated runs can produce different results.
- Rotations are averaged element by element. The resulting matrix is not guaranteed to be a valid orthonormal rotation, and independently calibrated pairs are not constrained to be globally consistent.
- Detection uses 16 worker processes; intrinsic and extrinsic calibration use 12 and 16 respectively. Worker counts can be changed through the Python APIs, but are not CLI options.
- The CLI loads device configuration and constructs capture-related objects even for processing existing data, so the capture dependency is required for those stages too.

## Troubleshooting

| Symptom | Checks |
| --- | --- |
| Dependency installation fails | Resolve the archive URL/local-path mismatch described in installation; check the capture package's platform requirements. |
| Target parameter path assertion or missing `target.json` | Provide `<data-root>/target_parameters/checkerboard/target.json`; run from the repository root. |
| Camera capture repeatedly fails | Check device identifiers, backend support, camera availability, requested resolution, and the capture library. `ImageManager` uses ports 10000 and 10001 by default. |
| No checkerboards detected | Verify inner-corner counts, focus, lighting, full board visibility, and orientation. The detector rejects some corner orderings. |
| Empty-array errors or filtering returns no dataset | Collect more varied target poses and inspect detection counts. Reduce exclusion percentages if needed; degenerate or identical observations can still fail strict filters. |
| Extrinsics has no usable observations | Check that all cameras have detections with matching filenames after filtering. Pairwise overlap alone is insufficient for the current join. |
| No calibration runs survive RMSE filtering | Use multiple iterations and a varied dataset. Equal RMSE values can all be rejected by the strict cutoff. |
| `--test-calibration` fails while sampling | Ensure at least ten frames have detected targets in all cameras. |
| OpenCV cannot display windows | Run the visualization in a graphical desktop with GUI-enabled OpenCV. |
| Incorrect scale or poor reprojection | Check measured corner spacing, metre-to-millimetre conversion, post-rotation image size, point correspondence, and whether cameras moved after capture. |
