# Calibration benchmark

```bash
# see what a sweep expands to, without launching anything
ros2 run annin_ar4_calibration benchmark_run --sweep sim_wp4.yaml --dry-run

# one cell first, to check the rig
ros2 run annin_ar4_calibration benchmark_run --sweep sim_wp4.yaml --limit 1

# the full sweep
ros2 run annin_ar4_calibration benchmark_run --sweep sim_wp4.yaml

# aggregate and plot
ros2 run annin_ar4_calibration benchmark_aggregate
ros2 run annin_ar4_calibration benchmark_report
```

Results land under `$ROS_HOME/annin_ar4_calibration/benchmark/` by default
(`~/.ros/...`); pass `--out-dir` to change output directory.
Shipped sweeps: `sim_wp2` (80 runs), `sim_wp3` (120),
`sim_wp4` (120, collects WP3 and WP4 together), `hardware` (3).


## The two metric families

### Family A — truth metrics (simulation only)

Gazebo spawns a robot carrying a known perturbation while every calibration
node sees the nominal model, so the true answer is known exactly. These
metrics compare estimate against truth directly.

| WP | metric |
|---|---|
| WP2 | tip offset error vs. `tcp_tip_gt` (mm) |
| WP3 | camera pose error vs. `sim_camera_color_optical_frame_gt` (mm, deg) |
| WP4 | **FK agreement** (mm) — see below; plus per-joint coefficient errors and mount-offset pose error |


### Family B — truth-free metrics (simulation *and* hardware)

Computable from the samples alone. On hardware these are the only metrics that
exist.

**B1 — held-out prediction error (K-fold).** 

**B2 — bootstrap spread.** 

**B3 — conditioning.** 

**B4 — reprojection error (px).**


## Modes

| mode | what it drives | ground truth |
|---|---|---|
| `sim` | Gazebo + MoveIt + the WP nodes | yes — both families |
| `hardware` | the WP nodes against an already-running arm | no — Family B only |
| `offline` | nothing; re-analyses archived runs | whatever the archive has |


```bash
ros2 run annin_ar4_calibration benchmark_run --mode offline --run-dir <directory containig measurement data>
```

### WP2 on hardware

The pivot method needs a human to touch the reference point, so it is not
automatable. Collect by hand with `tcp.launch.py`, then score the result with
`--mode offline`. The metrics are identical to the simulated ones.


## When a run never finishes

A run is a chain of blocking waits, and the `collect` timeout is a full hour.
```bash
# stream the child logs, heartbeat every 5 s, report each step as it lands,
# and fail in 5 minutes instead of an hour
ros2 run annin_ar4_calibration benchmark_run --sweep sim_wp2.yaml --limit 1 \
    --debug --timeout collect=300

# leave Gazebo and the nodes up after a failure so the state can be inspected
ros2 run annin_ar4_calibration benchmark_run --sweep sim_wp2.yaml --limit 1 \
    --debug --keep-alive
```

Where to look, in order:

| symptom | file | meaning |
|---|---|---|
| stuck on `TF base_link -> ee_link` | `sim.log` | controllers never activated |
| stuck on `/tcp_sim_poser/run_sequence to respond` | `tcp_sim_poser.log` | a pose is not completing — the last `pose N/M: executing` line is where |
| `SKIP collector refused` | `wp2.log` | the touch reached the collector and was rejected; the reason is on the line |
| `outcome: dry_run` in a manifest | — | the run never got past `--dry-run`; nothing was collected |


## Pitfalls

**Check you are not rendering on the CPU.** Mesa silently falls back to
`llvmpipe`, its software rasteriser, when it cannot open a GPU - the sim still
works, just entirely on CPU. `manifest.yaml` records the renderer per run:

```bash
grep -A 3 '^renderer:' <run>/manifest.yaml     # software_rendering: true?
```

Under WSL2 the fix is `GALLIUM_DRIVER=d3d12`, which routes Mesa through the
D3D12 mapping layer onto the real GPU. (`MESA_LOADER_DRIVER_OVERRIDE=d3d12`,
the more commonly suggested variable, does *not* work for this.) Measured on
one WP2 cell: 202 s on `llvmpipe`, 128 s on an Arc B580.
