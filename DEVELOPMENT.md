# Developer Guide

This guide is for changing, building, and validating the ROS 2 pipeline. For installation and first run, start with the [user README](README.md).

## Where to Work

- `ros2_ws/src/kitti_r1_perception/kitti_r1_perception/`: ROS 2 node implementations.
- `ros2_ws/src/kitti_r1_perception/launch/myperception.launch.py`: main launch, bag playback, and runtime parameters.
- `ros2_ws/src/kitti_r1_description/`: robot description, meshes, and RViz configurations.
- `mcap_converter/`: KITTI raw-data conversion and converter tests.
- `pixi.toml`: environment dependencies and the `run-pipeline` task.

The [technical report](TECHNICAL_REPORT.md) has the architecture diagram, node behavior, and a qualified historical telemetry summary. Verify behavior against node and launch sources, and use `pixi.toml` as the source of truth for commands and dependencies. Earlier reports are retained in `oldnotes/` as historical material.

## Build and Test

Install the Pixi environment and build from the repository root:

```bash
pixi install
pixi run colcon build --symlink-install
```

Run the perception package's ROS lint tests after building:

```bash
pixi run colcon test --packages-select kitti_r1_perception
pixi run colcon test-result --verbose
```

The pipeline smoke run checks the configured GPU environment, builds, and launches the default bag:

```bash
pixi run run-pipeline
```

For a different bag, pass its path to the main launch:

```bash
pixi run bash -c 'source install/setup.bash && ros2 launch kitti_r1_perception myperception.launch.py bag_path:=/absolute/path/to/bag'
```

The converter has focused output tests; see [mcap_converter/README.md](mcap_converter/README.md). They can be run with:

```bash
pixi run python -m unittest discover -s mcap_converter -p 'test_mcap_output.py'
```

## Changing the Pipeline

Keep ROS topics consistent between publishers, subscribers, the bag converter, and the launch. The current node interfaces are summarized in the [technical report](TECHNICAL_REPORT.md#ros-interfaces); confirm them in source before updating documentation.

The camera detector loads `torch_home/<model_version>.onnx` from the installed package share directory. Package assets under `kitti_r1_perception/torch_home/` are installed by `setup.py`. The detector initializes for a `1248 x 384` input, while `export_to_onnx.py` defaults to a square `640 x 640` export. Keep the exported model shape, preprocessing, and detector input assumptions aligned when changing models.

Fusion currently hard-codes camera intrinsics and LiDAR-to-camera extrinsics. The launch separately publishes placeholder `map -> velodyne` and `map -> camera` transforms. The MCAP's `CameraInfo` and `/tf_static` do not update those values yet; see the [sensor-frame notes](CompVisionKnowHow.md) for the planned calibration work.

## Documentation Guidance

- Keep this guide focused on development workflows; keep first-run instructions in [README.md](README.md).
- Use `TECHNICAL_REPORT.md` for architecture and implementation context, but verify changing details against executable sources.
- Treat recorded telemetry as observations from a specific run, not as a repeatable performance benchmark. The [benchmark plan](benchmark_action_plan.md) describes the evidence needed for benchmark claims.
