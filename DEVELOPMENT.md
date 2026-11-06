# Developer Guide

Build, test, and extend the ROS 2 pipeline here. For prerequisites and the first end-to-end run, follow the [README Quick Start](README.md#quick-start).

## Where to Work

- `ros2_ws/src/kitti_r1_perception/kitti_r1_perception/`: ROS 2 node implementations.
- `ros2_ws/src/kitti_r1_perception/launch/myperception.launch.py`: main launch, bag playback, and runtime parameters.
- `ros2_ws/src/kitti_r1_description/`: robot description, meshes, and RViz configurations.
- `mcap_converter/`: KITTI raw-data conversion and converter tests.
- `pixi.toml`: environment dependencies and the `run-pipeline` task.

Use the [technical report](TECHNICAL_REPORT.md) for architecture context. Verify implementation details against the node and launch sources; `pixi.toml` is the source of truth for dependencies and tasks.

## Build and Test

After completing the README setup, build from the repository root:

```bash
pixi run colcon build --symlink-install
```

Run the perception package checks:

```bash
pixi run colcon test --packages-select kitti_r1_perception
pixi run colcon test-result --verbose
```

Run the focused fusion callback tests with a dedicated Pixi task:

```bash
pixi run test-fusion
```

The test calls the fusion callback with synthetic ROS messages and checks both a successful association and a rejected match. It does not start DDS nodes or require a GPU.

The converter has focused output tests; see [mcap_converter/README.md](mcap_converter/README.md). They can be run with:

```bash
pixi run python -m unittest discover -s mcap_converter -p 'test_mcap_output.py'
```

## Changing the Pipeline

Keep topic names and message types consistent across publishers, subscribers, the converter, and launch files. Confirm interface changes in source and update the [ROS interface summary](TECHNICAL_REPORT.md#ros-interfaces).

For camera model setup and checksums, use the [README](README.md#camera-model). When changing the model, keep export settings, preprocessing, and detector input shape aligned. `setup.py` installs files under `kitti_r1_perception/torch_home/`.

Fusion currently hard-codes camera intrinsics and LiDAR-to-camera extrinsics. The launch separately publishes placeholder `map -> velodyne` and `map -> camera` transforms. The MCAP's `CameraInfo` and `/tf_static` do not update those values yet; see the [sensor-frame notes](CompVisionKnowHow.md) for the planned calibration work.

## Documentation Guidance

- Keep this guide focused on development workflows; keep first-run instructions in [README.md](README.md).
- Use `TECHNICAL_REPORT.md` for architecture and implementation context, but verify changing details against executable sources.
- Treat recorded telemetry as observations from a specific run, not as a repeatable performance benchmark. The [benchmark plan](benchmark_action_plan.md) describes the evidence needed for benchmark claims.
