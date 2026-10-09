# KITTI to MCAP

`kitti_to_mcap.py` converts a KITTI raw drive into an MCAP rosbag.

## Inputs

The sequence directory must contain `image_02`, `image_03`, `velodyne_points`, and `oxts`. Each sensor directory must contain a `data` folder and a `timestamps.txt` file. The calibration directory must contain `calib_cam_to_cam.txt` and `calib_velo_to_cam.txt`.

## Topics

- `/kitti/camera_left/image_raw` and `/kitti/camera_left/camera_info`
- `/kitti/camera_right/image_raw` and `/kitti/camera_right/camera_info`
- `/kitti/velodyne/pointcloud`
- `/kitti/oxts/fix` and `/kitti/oxts/twist`
- `/tf_static` for camera-to-LiDAR calibration transforms

## Timestamps

Each sensor's sorted data files are paired by index with its own `timestamps.txt`. File and timestamp counts must match; conversion stops with an error otherwise. Each message header and MCAP record uses that sensor's timestamp. No cross-sensor matching is performed. Static transforms use the first LiDAR timestamp.

See [math_behind.md](math_behind.md) for the timestamp, point-cloud layout, and calibration transform calculations.

## Run

Set `KITTI_SEQUENCE`, `KITTI_CALIB_DIR`, and `BAG_OUTPUT` at the bottom of the script, source ROS 2 with the required Python dependencies, then run:

```bash
python3 mcap_converter/kitti_to_mcap.py
```

An existing bag at `BAG_OUTPUT` is deleted and recreated. Set this path carefully.

Run the generated-bag checks with:

```bash
pixi run python -m unittest discover -s mcap_converter -p 'test_mcap_output.py'
```

Set `KITTI_MCAP_BAG` or `KITTI_SEQUENCE` to test a different output bag or source sequence.
