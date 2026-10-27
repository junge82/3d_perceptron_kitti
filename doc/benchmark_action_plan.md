# Benchmark Protocol Action Plan

## Goal

Create a reproducible performance benchmark for the KITTI LiDAR-camera perception pipeline. Another person should be able to use the same input, software configuration, and run procedure to reproduce the measurements and understand their limits.

This plan is for performance evaluation. Detection accuracy requires ground-truth labels and a separate evaluation protocol; do not infer accuracy from runtime telemetry.

## Controller Usage

Run `pixi run benchmark-pipeline` to build the workspace once and perform three independent playback runs by default. The controller samples process-tree CPU/RSS, system CPU/RAM, and device-wide NVIDIA GPU utilization, VRAM, and power while the bag is playing. Playback start and exit markers bound the resource-sampling interval; the launch's clean-shutdown option is enabled only for these benchmark runs.

Each invocation creates a new timestamped directory under `benchmark_results/`. Each run has its own `pipeline.log` and `resource_usage.csv`; the directory also contains hardware/configuration metadata, a per-run CSV, a JSON summary, and a Markdown report. Use `pixi run benchmark-pipeline --help` to override run count, bag, playback rate, sampling interval, timeout, output directory, or to skip the build.

Three repetitions are an initial observational benchmark, not a statistically strong performance claim. Preserve failed/interrupted run statuses and report the successful-run count; run more repetitions for formal comparisons.

## Current Starting Point

- `compute_pipeline_telemetry.py` parses per-node timing fields and reports sample count, mean, median, p95, p99, minimum, and maximum.
- `benchmark_pipeline.py` runs repeatable bag playback, saves per-run logs and CPU/GPU resource samples, and generates JSON/CSV/Markdown summaries.
- `validate_pipeline_gpu.py` checks GPU-related dependencies and provider availability. It does not benchmark the ROS 2 pipeline or verify that every operation ran on the GPU.
- KITTI MCAP recordings are present in the repository. Select and record one exact bag as the primary benchmark input.
- The report currently describes telemetry observations but not a repeatable run procedure or accuracy results.

## Action Items

### 1. Freeze the benchmark configuration

- [ ] Select one primary KITTI MCAP recording and record its repository-relative path, file size, and checksum.
- [ ] Record the ROS 2 launch command, launch arguments, node parameters, model path, and any environment variables used.
- [ ] Record the repository commit, dirty-worktree status, Python and ROS 2 versions, package versions, ONNX Runtime providers, CUDA and driver versions, and GPU model.
- [ ] Record CPU model, memory, operating system, power mode, and whether the run is CPU-only or GPU-enabled.
- [ ] Keep input data, model, configuration, and hardware unchanged between runs being compared.

### 2. Define the run procedure

- [x] Document the benchmark controller command, artifact layout, and opt-in bag-end shutdown behavior.
- [ ] Specify bag playback rate and whether playback is throttled or accelerated; use the same setting for every run.
- [ ] Define startup handling: wait for all nodes and model sessions to initialize, then exclude startup and warm-up samples from steady-state statistics.
- [ ] Run at least five measured repetitions, restarting the pipeline between repetitions. Record failed or interrupted runs rather than silently discarding them.
- [ ] Use a consistent run duration or frame interval. Record processed message counts and playback duration so throughput can be interpreted.

### 3. Define metrics and interpretation

- [ ] Report per-stage and total callback latency with sample counts, mean, median, p95, p99, minimum, and maximum.
- [ ] Report end-to-end throughput and dropped or unprocessed messages, separately from callback latency.
- [ ] Distinguish sensor timestamp age, callback processing time, inference time, and visualization time; do not present them as interchangeable latency measures.
- [ ] Record CPU, GPU utilization, GPU memory, and system memory over each measured run where available.
- [ ] State whether each timing includes data transfer, synchronization, ROS message conversion, and publication.
- [ ] Summarize results across repetitions and identify variability; do not report only the fastest run.

### 4. Implement measurement and artifact capture

- [x] Extend `compute_pipeline_telemetry.py` to calculate percentile statistics and parse separate camera/LiDAR callback metrics.
- [x] Ensure benchmark metadata identifies the run, input recording, software revision, and configuration. Save raw per-run logs.
- [x] Save machine-readable JSON and CSV summaries alongside the Markdown report and raw run logs.
- [ ] Verify whether existing node timing fields measure the intended boundaries before using them in published claims.
- [ ] Keep GPU availability checks separate from performance evidence; verify actual provider/device use in the inference and LiDAR stages.

### 5. Validate and report results

- [ ] Perform a dry run and confirm the full pipeline receives the intended streams and produces expected output topics.
- [ ] Check that measured sample counts are plausible and that all required timing fields are present.
- [ ] Repeat one run and confirm the documented procedure produces comparable metrics within a stated tolerance.
- [ ] If comparing revisions or configurations, change one factor at a time and use the same data and environment.
- [ ] Publish the procedure, raw-log location, summarized results, environment details, and known limitations together.
- [ ] Update `oldnotes/tech_report_aligned.md` only after the protocol has been executed; label results as measured observations and avoid unsupported historical comparisons.

## Completion Criteria

The benchmark protocol is ready to cite when a clean checkout can follow the documented commands, reproduce the specified input and configuration, produce raw logs and summary artifacts, and report the required metrics for at least five valid runs. Any missing metric, failed run, or environment difference must be disclosed.

## Not Included Yet

This plan does not define an object-detection accuracy score. An accuracy evaluation needs a labeled KITTI subset, a clear matching and scoring method, class and range rules, and a separate report of those results.