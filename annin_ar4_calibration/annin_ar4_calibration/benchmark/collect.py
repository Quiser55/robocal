"""
Drive one benchmark run: launch, collect samples, compute, score, tear down.

The only script that actually runs the robot or sim.


Per run the following data is saved:

- ``perturbation.yaml`` - the injected error; the ground truth for WP4.
- ``nominal_robot.urdf`` - the model to compare the perturbed one against, and the reference for WP4.
- ``perturbed_robot.urdf`` - what Gazebo actually spawned.
- ``manifest.yaml`` - the spec, the outcome, and the environment.
"""
from __future__ import annotations

import datetime
import os
import shutil
import signal
import subprocess
import threading
import time

import numpy as np
import yaml

from annin_ar4_calibration.benchmark import spec as spec_mod
from annin_ar4_calibration.benchmark import trace
from annin_ar4_calibration.core import kinematic_model, perturbation, storage

#: Outcome values recorded in manifest.yaml
OUTCOME_OK = 'ok'
OUTCOME_NOT_READY = 'not_ready'
OUTCOME_COLLECTION_FAILED = 'collection_failed'
OUTCOME_COMPUTE_FAILED = 'compute_failed'


class RosBridge:
    """A short-lived rclpy node to wait for readiness and call services."""

    def __init__(self, use_sim_time: bool):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from tf2_ros import Buffer, TransformListener

        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
        self.node = Node('benchmark_runner',
                         parameter_overrides=[
                             rclpy.parameter.Parameter(
                                 'use_sim_time', rclpy.Parameter.Type.BOOL, use_sim_time)])
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        while not self._stop.is_set():
            self._executor.spin_once(timeout_sec=0.1)

    def shutdown(self):
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._executor.remove_node(self.node)
        self.node.destroy_node()

    # waiting 

    def wait_for_message(self, topic: str, msg_type, timeout: float) -> bool:
        received = threading.Event()
        sub = self.node.create_subscription(
            msg_type, topic, lambda _msg: received.set(), 1)
        try:
            with trace.waiting(f'first message on {topic}', timeout):
                return received.wait(timeout)
        finally:
            self.node.destroy_subscription(sub)

    def wait_for_tf(self, parent: str, child: str, timeout: float) -> bool:
        deadline = time.time() + timeout
        with trace.waiting(f'TF {parent} -> {child}', timeout):
            while time.time() < deadline:
                if self.tf_buffer.can_transform(parent, child, self._rclpy.time.Time()):
                    return True
                time.sleep(0.5)
        return False

    # calling 

    def _await(self, future, timeout: float):
        """Wait for a future by polling, letting the background executor do thespinning."""
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.1)
        if not future.done():
            future.cancel()
            return None
        return future.result()

    def call_trigger(self, name: str, timeout: float) -> tuple[bool, str]:
        from std_srvs.srv import Trigger
        client = self.node.create_client(Trigger, name)
        discovery = min(timeout, 60.0)
        try:
            with trace.waiting(f'{name} to be advertised', discovery):
                if not client.wait_for_service(timeout_sec=discovery):
                    return False, f'service {name} never appeared'
            with trace.waiting(f'{name} to respond', timeout):
                response = self._await(client.call_async(Trigger.Request()), timeout)
            if response is None:
                return False, f'service {name} timed out after {timeout:.0f}s'
            trace.debug(f'{name} -> success={response.success} {response.message!r}')
            return bool(response.success), str(response.message)
        finally:
            self.node.destroy_client(client)

    def fetch_nominal_urdf(self, timeout: float = 30.0) -> str | None:
        from rcl_interfaces.srv import GetParameters

        client = self.node.create_client(
            GetParameters, '/robot_state_publisher/get_parameters')
        try:
            if not client.wait_for_service(timeout_sec=timeout):
                return None
            request = GetParameters.Request()
            request.names = ['robot_description']
            response = self._await(client.call_async(request), timeout)
            if response is None or not response.values:
                return None
            return response.values[0].string_value or None
        except Exception:
            return None
        finally:
            self.node.destroy_client(client)


class ManagedProcess:
    """A `ros2 launch`/`ros2 run` subprocess in its own process group."""

    def __init__(self, args: list[str], env: dict, log_path: str,
                 label: str | None = None):
        self.args = args
        self.log_path = log_path
        self.label = label or os.path.splitext(os.path.basename(log_path))[0]
        resume_at = os.path.getsize(log_path) if os.path.exists(log_path) else 0
        self._log = open(log_path, 'ab')
        trace.debug(f'launching {self.label}: {" ".join(args)}')
        self._tail = (trace.LogTail(log_path, self.label, start=resume_at)
                      if trace.is_enabled() else None)
        self.process = subprocess.Popen(
            args, env=env, stdout=self._log, stderr=subprocess.STDOUT,
            start_new_session=True)
        trace.debug(f'{self.label} pid {self.process.pid}, log {log_path}')

    @staticmethod
    def _stat_fields(pid: int):
        with open(f'/proc/{pid}/stat') as f:
            fields = f.read().rsplit(') ', 1)[-1].split()
        return int(fields[1]), int(fields[19])

    def _descendants(self) -> list[tuple[int, int]]:
        children = {}
        starttimes = {}
        for entry in os.listdir('/proc'):
            if not entry.isdigit():
                continue
            try:
                ppid, starttime = self._stat_fields(int(entry))
            except (OSError, IndexError, ValueError):
                continue
            children.setdefault(ppid, []).append(int(entry))
            starttimes[int(entry)] = starttime

        found, queue = [], [self.process.pid]
        while queue:
            pid = queue.pop()
            for child in children.get(pid, []):
                found.append((child, starttimes[child]))
                queue.append(child)
        return found

    def stop(self, grace: float = 15.0) -> None:
        trace.debug(f'stopping {self.label} (pid {self.process.pid})')
        if self._tail is not None:
            self._tail.stop()
            self._tail = None
        if self.process.poll() is not None:
            self._log.close()
            return

        stragglers = self._descendants()

        pgid = os.getpgid(self.process.pid)
        os.killpg(pgid, signal.SIGINT)
        deadline = time.time() + grace
        while time.time() < deadline and self.process.poll() is None:
            time.sleep(0.5)
        if self.process.poll() is None:
            os.killpg(pgid, signal.SIGKILL)
            try:
                self.process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                pass

        time.sleep(1.0)
        for pid, starttime in stragglers:
            try:
                if self._stat_fields(pid)[1] != starttime:
                    continue        # PID recycled; this is somebody else
                os.kill(pid, signal.SIGKILL)
            except (OSError, IndexError, ValueError):
                pass
        self._log.close()


def write_perturbation(path: str, seed: int, sigma_t_mm: float, sigma_r_deg: float) -> str:
    """Generate this run's injected error."""
    n_joints = len(perturbation.DEFAULT_JOINT_NAMES)
    corrections = np.zeros((n_joints, kinematic_model.PARAMS_PER_JOINT))
    if sigma_t_mm > 0 or sigma_r_deg > 0:
        rng = np.random.default_rng(seed)
        corrections[:, 0:2] = np.clip(
            rng.normal(0.0, sigma_t_mm / 1000.0, (n_joints, 2)), -0.002, 0.002)
        r_clip = np.radians(0.15)
        corrections[:, 2:4] = np.clip(
            rng.normal(0.0, np.radians(sigma_r_deg), (n_joints, 2)), -r_clip, r_clip)

    document = perturbation.to_yaml_dict(
        corrections,
        generated_by='annin_ar4_calibration.benchmark.collect',
        description=(f'benchmark seed={seed}, sigma_t={sigma_t_mm} mm, '
                     f'sigma_r={sigma_r_deg} deg'))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(document, f, sort_keys=False, default_flow_style=None)
    return path


_SEQUENCE_BUDGET_MARGIN_SEC = 180.0


def write_extra_params(path: str, run_spec: spec_mod.RunSpec,
                       collect_timeout_sec: float | None = None) -> str:
    """Per-run node parameter overrides."""
    collection = run_spec.collection
    document = {
        'hand_eye_collector': {'ros__parameters': {
            'num_candidate_poses': int(collection['num_candidates']),
            'min_samples': int(collection['num_samples']),
        }},
        'hand_eye_calibration': {'ros__parameters': {
            'min_samples': min(10, int(collection['num_samples'])),
        }},
        'kinematic_collector': {'ros__parameters': {
            'num_candidate_configs': int(collection['num_candidates']),
            'min_samples': int(collection['num_samples']),
        }},
        'kinematic_calibration': {'ros__parameters': {
            'min_samples': max(12, int(collection['num_samples']) // 2),
        }},
    }
    if collect_timeout_sec:
        budget = max(60.0, float(collect_timeout_sec) - _SEQUENCE_BUDGET_MARGIN_SEC)
        document['kinematic_collector']['ros__parameters'][
            'max_sequence_duration_sec'] = budget

    with open(path, 'w') as f:
        yaml.safe_dump(document, f, sort_keys=False)
    return path


def archive_previous_attempt(run_dir: str) -> str | None:
    data_dir = os.path.join(run_dir, 'ros_home', 'annin_ar4_calibration')
    if not os.path.isdir(data_dir) or not os.listdir(data_dir):
        return None
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    dest = os.path.join(run_dir, 'previous_attempts', stamp)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(data_dir, dest)
    return dest


def run_environment(run_dir: str) -> dict:
    env = dict(os.environ)
    env['ROS_HOME'] = os.path.join(run_dir, 'ros_home')
    os.makedirs(os.path.join(env['ROS_HOME'], 'annin_ar4_calibration'), exist_ok=True)
    return env

_RENDER_ENV_VARS = ('GALLIUM_DRIVER', 'MESA_LOADER_DRIVER_OVERRIDE',
                    'LIBGL_ALWAYS_SOFTWARE', 'MESA_GL_VERSION_OVERRIDE')


def detect_renderer() -> dict:
    info = {name: os.environ[name] for name in _RENDER_ENV_VARS if name in os.environ}
    log_path = os.path.join(
        os.environ.get('GZ_HOME', os.path.expanduser('~')), '.gz', 'rendering', 'ogre2.log')
    try:
        with open(log_path, errors='replace') as f:
            for line in f:
                if 'GL_RENDERER' in line:
                    info['gl_renderer'] = line.split('=', 1)[1].strip()
    except OSError:
        return info
    renderer = info.get('gl_renderer', '')
    if renderer:
        # The one fact a reader needs without knowing Mesa driver names.
        info['software_rendering'] = any(
            name in renderer.lower() for name in ('llvmpipe', 'softpipe', 'swrast'))
    return info


def _git_sha(repo_dir: str) -> str:
    try:
        return subprocess.check_output(
            ['git', '-C', repo_dir, 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return 'unknown'


def _wp_launch_file(wp: str) -> str:
    return {'wp2': 'tcp.launch.py', 'wp3': 'hand_eye.launch.py',
            'wp4': 'kinematic.launch.py'}[wp]


class RunFailure(RuntimeError):
    def __init__(self, outcome: str, detail: str):
        super().__init__(detail)
        self.outcome = outcome
        self.detail = detail


def collect_run(run_spec: spec_mod.RunSpec, run_dir: str, timeouts: dict,
                dry_run: bool = False, keep_alive: bool = False) -> dict:
    """Execute one run. Returns the manifest dict (also written to disk)."""
    os.makedirs(run_dir, exist_ok=True)
    started = time.time()
    manifest = {
        'run_id': run_spec.run_id,
        'spec': run_spec.to_dict(),
        'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'git_sha': _git_sha(os.path.dirname(os.path.abspath(__file__))),
        'outcome': OUTCOME_OK,
        'steps': [],
    }

    if dry_run:
        manifest['outcome'] = 'dry_run'
        storage.save_result(os.path.join(run_dir, 'manifest.yaml'), manifest)
        return manifest

    archived = archive_previous_attempt(run_dir)
    if archived:
        manifest['previous_attempt_archived_to'] = archived
    env = run_environment(run_dir)
    collection = run_spec.collection
    processes: list[ManagedProcess] = []
    bridge = None

    perturbation_file = collection.get('perturbation_file') or write_perturbation(
        os.path.join(run_dir, 'perturbation.yaml'), run_spec.seed,
        float(collection['perturbation_sigma_t_mm']),
        float(collection['perturbation_sigma_r_deg']))
    manifest['perturbation_file'] = perturbation_file
    extra_params = write_extra_params(
        os.path.join(run_dir, 'extra_params.yaml'), run_spec,
        collect_timeout_sec=timeouts.get('collect'))

    is_sim = run_spec.mode == 'sim'
    try:
        bridge = RosBridge(use_sim_time=is_sim)

        if is_sim:
            processes.append(ManagedProcess(
                [
                    'ros2', 'launch', 'annin_ar4_gazebo', 'calibration_sim.launch.py',
                    f'calibration_type:={collection["calibration_type"]}',
                    f'ar_model:={collection["ar_model"]}',
                    'headless:=True',
                    f'perturbation_file:={perturbation_file}',
                    f'perturbed_urdf_out:={os.path.join(run_dir, "perturbed_robot.urdf")}',
                    f'image_noise_stddev:={collection["image_noise_stddev"]}',
                    f'camera_width:={collection["camera_width"]}',
                    f'camera_height:={collection["camera_height"]}',
                ],
                env, os.path.join(run_dir, 'sim.log')))
            _wait_for_sim(bridge, collection, timeouts, manifest)
            manifest['renderer'] = detect_renderer()
            if manifest['renderer'].get('software_rendering'):
                trace.emit(f'software rendering: {manifest["renderer"]["gl_renderer"]} '
                           f'(no GPU in use)')

        # Archive the nominal chain.
        urdf = bridge.fetch_nominal_urdf(timeout=timeouts['ready'])
        if urdf:
            with open(os.path.join(run_dir, 'nominal_robot.urdf'), 'w') as f:
                f.write(urdf)
        else:
            manifest['steps'].append(
                {'step': 'nominal_urdf', 'ok': False,
                 'message': 'could not fetch robot_description; WP4 analysis will skip'})

        for wp in run_spec.work_packages:
            trace.debug(f'--- {wp} ---')
            wp_args = [
                'ros2', 'launch', 'annin_ar4_calibration', _wp_launch_file(wp),
                f'use_sim_time:={"True" if is_sim else "False"}',
                f'extra_params_file:={extra_params}',
            ]
            if wp in ('wp3', 'wp4'):
                wp_args.append(f'calibration_type:={collection["calibration_type"]}')

            wp_proc = ManagedProcess(wp_args, env, os.path.join(run_dir, f'{wp}.log'))
            processes.append(wp_proc)
            try:
                _run_work_package(wp, bridge, collection, timeouts, manifest,
                                  perturbation_file, is_sim, env, run_dir,
                                  run_spec.seed)
            finally:
                wp_proc.stop()
                processes.remove(wp_proc)

        if is_sim:
            ok, message = bridge.call_trigger('/ground_truth/compare', timeouts['service'])
            manifest['steps'].append(
                {'step': 'ground_truth_compare', 'ok': ok, 'message': message})

    except RunFailure as exc:
        manifest['outcome'] = exc.outcome
        manifest['error'] = exc.detail
    except Exception as exc:
        manifest['outcome'] = OUTCOME_COLLECTION_FAILED
        manifest['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if keep_alive and manifest['outcome'] != OUTCOME_OK:
            trace.emit(f'--keep-alive: leaving {len(processes)} process group(s) up '
                       f'after {manifest["outcome"]}')
            for proc in processes:
                trace.emit(f'    {proc.label}: pid {proc.process.pid}  ({proc.log_path})')
            trace.emit('    kill them with: '
                       + '; '.join(f'kill -INT -{proc.process.pid}' for proc in processes))
        else:
            for proc in reversed(processes):
                proc.stop()
        if bridge is not None:
            bridge.shutdown()

    manifest['duration_sec'] = round(time.time() - started, 1)
    manifest['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    storage.save_result(os.path.join(run_dir, 'manifest.yaml'), manifest)
    return manifest


def _wait_for_sim(bridge: RosBridge, collection: dict, timeouts: dict,
                  manifest: dict) -> None:
    from sensor_msgs.msg import JointState

    ready = timeouts['ready']
    if not bridge.wait_for_message('/joint_states', JointState, ready):
        raise RunFailure(OUTCOME_NOT_READY,
                         'no /joint_states within timeout (controller spawner never '
                         'activated, or gz never started)')
    if not bridge.wait_for_tf('base_link', 'ee_link', ready):
        raise RunFailure(OUTCOME_NOT_READY, 'TF base_link -> ee_link never resolved')

    gt_frame = 'sim_camera_color_optical_frame_gt'
    if not bridge.wait_for_tf('base_link', gt_frame, ready):
        manifest['steps'].append(
            {'step': 'gt_frame', 'ok': False,
             'message': f'{gt_frame} never appeared; truth metrics will be missing'})

    manifest['steps'].append({'step': 'sim_ready', 'ok': True, 'message': ''})


def _run_work_package(wp: str, bridge: RosBridge, collection: dict, timeouts: dict,
                      manifest: dict, perturbation_file: str, is_sim: bool,
                      env: dict, run_dir: str, seed: int) -> None:
    step_clock = [time.time()]

    def step(name, ok, message, fatal_outcome=None):
        now = time.time()
        seconds = round(now - step_clock[0], 1)
        step_clock[0] = now
        manifest['steps'].append({'step': f'{wp}.{name}', 'ok': ok,
                                  'message': message, 'seconds': seconds})
        trace.debug(f'step {wp}.{name}: ok={ok} in {seconds:.1f}s {message!r}')
        if not ok and fatal_outcome:
            raise RunFailure(fatal_outcome, f'{wp}.{name}: {message}')

    if wp in ('wp3', 'wp4'):
        # A detected target is the precondition for every further collection step
        if not bridge.wait_for_tf('camera_color_optical_frame', 'calibration_target',
                                  timeouts['ready']):
            step('target_detected', False, 'calibration_target TF never appeared',
                 OUTCOME_NOT_READY)
        step('target_detected', True, '')

    if wp == 'wp2':
        if not is_sim:
            raise RunFailure(
                OUTCOME_COLLECTION_FAILED,
                'TCP on hardware needs a human to touch the reference point; collect'
                'the samples manually, then analyze them with --mode offline')
        poser = ManagedProcess(
            [
                'ros2', 'run', 'annin_ar4_calibration', 'tcp_sim_poser', '--ros-args',
                '-p', f'num_poses:={int(collection["num_poses"])}',
                '-p', f'touch_noise_std_m:={float(collection["touch_noise_std_m"])}',
                '-p', f'perturbation_file:={perturbation_file}',
                '-p', f'seed:={int(seed)}',
                '-p', 'use_sim_time:=true',
            ],
            env, os.path.join(run_dir, 'tcp_sim_poser.log'))
        try:
            ok, message = bridge.call_trigger(
                '/tcp_sim_poser/run_sequence', timeouts['collect'])
            step('run_sequence', ok, message, OUTCOME_COLLECTION_FAILED if not ok else None)
        finally:
            poser.stop()
        ok, message = bridge.call_trigger(
            '/calibration/compute_calibration', timeouts['service'])
        step('compute', ok, message, OUTCOME_COMPUTE_FAILED if not ok else None)
        return

    if wp == 'wp3':
        ok, message = bridge.call_trigger(
            '/hand_eye_collector/capture_seed', timeouts['service'])
        step('capture_seed', ok, message, OUTCOME_COLLECTION_FAILED if not ok else None)
        ok, message = bridge.call_trigger(
            '/hand_eye_collector/run_auto_sequence', timeouts['collect'])
        step('run_auto_sequence', ok, message,
             OUTCOME_COLLECTION_FAILED if not ok else None)
        ok, message = bridge.call_trigger(
            '/hand_eye_calibration/compute_calibration', timeouts['service'])
        step('compute', ok, message, OUTCOME_COMPUTE_FAILED if not ok else None)
        return

    ok, message = bridge.call_trigger(
        '/kinematic_collector/run_auto_sequence', timeouts['collect'])
    step('run_auto_sequence', ok, message, OUTCOME_COLLECTION_FAILED if not ok else None)
    ok, message = bridge.call_trigger(
        '/kinematic_calibration/compute_calibration', timeouts['service'])
    step('compute', ok, message, OUTCOME_COMPUTE_FAILED if not ok else None)
