# Camera System Calibration

## Prerequisites
- This project uses my [camera capture system]([url](https://github.com/kristina-aino/camera-capture-system)) so that whl has to be added to the current project path manually (changes in the future)
- install the other packages via the *poetry.lock*

## Usage
- Calibration target can be generated [here](https://calib.io/pages/camera-calibration-pattern-generator) (currently only checkerbard is supported)
- Add the pdf to the */target_data* folder
- change the *target.json* file to reflect the specifics of your new target
- Then you shuold be able to use `poetry run python run.py --collect-images` to start collecting images
  - use `--num-frames-to-collect` to influence the images it collects.
  - and use `--frame-collection-interval` to throttle the recording of individual images
- After that you can use the `--find-targets` flag to find the targets in all recorded images
  - make sure that the *target* is directed in the direction of the multi camera setup
  - the `--remove-images-without-targets` flag is used to remove all images without recognized targets
- Finnaly use `--calibrate-intrinsics` and `--calibrate-extrinsics` to calibrate the camera systems intrinsics and extrinsics parameters.

(This is a work in progress, functionality should be provided but its not tested at the moment)
