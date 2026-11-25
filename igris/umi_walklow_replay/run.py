"""Run the UMI replay in the isolated vendor simulator.

  prepare  episode, targets, scene and composed vendor model (new output directory)
  labels   pose labels the robot can follow at their own clock, and the stream of them (labels.py)
  start    audit the model on the vendor libmujoco and launch the private container
  bringup  reset -> torque on -> HOME -> WALK_LOW, with the upper body held
  replay   online IK + SDK LowCmd replay inside the container
  stop     stop this run's container
  analyze  tracking and box-handling report
  render   Isaac Sim video of the recorded physics state
  all      every step above in order

``--policy CHECKPOINT`` (replay/all) lets a trained policy drive the robot instead of a recording: a policy process
(policy_server.py) is started on this machine next to the simulator container and the two exchange targets through
the run folder.  ``--no-render`` leaves the Isaac Sim video out of ``all``.

Nothing here connects to hardware: the container has no network and private IPC.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
VENDOR_ROOM = ROOT / 'igris/isaac_vendor_room'
# The Python of an Isaac Sim installation (it has pxr); by default the one this is run with.
ISAAC_PYTHON = Path(os.environ.get('ISAAC_PY', sys.executable))
# mink, qpsolvers + daqp, scipy and MuJoCo >= 3.10 for Python 3.12, installed with ``pip --target``
# (tools/install_packages.sh puts them in ``vendor_py/mink_cp312``).
# The simulator image has none of them; the folder is mounted read-only and used only by ``--ik mink``.
MINK_PY = Path(os.environ.get('UMI_MINK_PY', ROOT / 'vendor_py/mink_cp312'))
IMAGE = 'igris-sim:jazzy'
STAGED = ['room_Case', 'room_Rolltainer', 'room_Parcel_0', 'room_Parcel_1', 'room_Parcel_2']
# Vendor simulator controller the heading library was written against (native/heading.cpp): the library finds
# the walking modules by exported symbols, but reads the mode back at offsets of this build.
VENDOR_SIM = ROOT / 'igris/ws/install/igris_c_controller/lib/igris_c_controller/sim'
VENDOR_SIM_SHA256 = '5003375d46f80d7e1064b50a1933e0a1fb70f7566724ac682379ef909beb8b9b'
CAMERAS = ['/World/Cameras/UmiOverview', '/World/Cameras/UmiFront', '/World/Cameras/UmiCart']
# The renderer accumulates its own frames into each video frame.  With the 2 that render.py takes by default
# the moving arms and hands came out as a blend of old and new positions (error against a converged image 8.1
# grey levels; 2.4 with 16 at native-resolution antialiasing, 0.31 s instead of 0.13 s per frame and camera).
RENDER_SUBFRAMES = 16
RENDER_ANTIALIASING = 4      # DLAA: native resolution; 3 (DLSS) renders smaller and scales up


def vendor_run():
    """The existing launcher module; only its generic helpers are used."""
    if str(VENDOR_ROOM) not in sys.path:
        sys.path.insert(0, str(VENDOR_ROOM))
    import run as module
    return module


def execute(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def build_native(out, heading=False):
    """Audit the composed model on the vendor libmujoco and compile the libraries the simulator preloads.
    Returns the preload list.  Needs only g++ and the vendor install: it also runs inside the simulator image."""
    run = out / 'run'
    if (run / 'runtime.json').exists() or (run / 'poses.bin').exists():
        raise FileExistsError('This output was already started; prepare a new one')
    manifest = json.loads((run / 'manifest.json').read_text())
    if not manifest.get('stage_room_after_preparation') or 'umi_replay' not in manifest:
        raise ValueError('Not a prepared UMI replay output')
    include = ROOT / 'igris/ws/install/igris_c_mujoco/include'
    library = ROOT / 'igris/ws/install/igris_c_mujoco/lib/igris_c_mujoco'
    execute(['g++', '-O2', '-std=c++17', f'-I{include}', str(HERE / 'native/audit.cpp'), f'-L{library}',
             f'-Wl,-rpath,{library}', '-l:libmujoco.so.3.2.7', '-o', str(run / 'audit')])
    execute([str(run / 'audit'), str(run / 'vendor.xml'), str(run / 'scene.xml'), str(run / 'model.json')])
    shared = ['g++', '-O2', '-std=c++17', '-shared', '-fPIC', f'-I{include}']
    execute(shared + [str(VENDOR_ROOM / 'observer.cpp'), '-ldl', '-o', str(run / 'observer.so')])
    execute(shared + [str(ROOT / 'igris/hl_ik/native_clock.cpp'), '-ldl', '-o', str(run / 'clock.so')])
    execute(shared + [str(VENDOR_ROOM / 'room_setup.cpp'), '-ldl', '-o', str(run / 'room_setup.so')])
    names = [body['name'] for body in json.loads((run / 'model.json').read_text())['bodies']]
    if any(name not in names for name in STAGED):
        raise ValueError('Composed model lacks an expected scene body')
    (run / 'room_stage_bodies.txt').write_text('\n'.join(STAGED) + '\n')
    preload = [run / 'clock.so', run / 'room_setup.so', run / 'observer.so']
    if heading:
        if hashlib.sha256(VENDOR_SIM.read_bytes()).hexdigest() != VENDOR_SIM_SHA256:
            raise RuntimeError('The vendor simulator binary changed; check native/heading.cpp against it first')
        execute(shared + [str(HERE / 'native/heading.cpp'), '-ldl', '-lpthread', '-o', str(run / 'heading.so')])
        preload.append(run / 'heading.so')
    (run / 'vendor_logs').mkdir()
    return preload


def launch_script(run, preload, heading=False):
    """Shell script that starts the vendor simulation of a prepared run with the preload libraries."""
    return f'''source /opt/ros/jazzy/setup.bash
source /igris/ws/install/setup.bash
cd /tmp
export LD_PRELOAD={shlex.quote(':'.join(map(str, preload)))} IGRIS_ROOM_OUTPUT={shlex.quote(str(run))}
export IGRIS_HLIK_CLOCK={shlex.quote(str(run / 'native_clock.bin'))}
export IGRIS_ROOM_PAYLOAD=''
export IGRIS_ROOM_STAGE={shlex.quote(str(run))}
export IGRIS_ROOM_STAGE_BODIES_FILE={shlex.quote(str(run / 'room_stage_bodies.txt'))}
export IGRIS_ROOM_STAGE_SCOPE=source_and_destination
export IGRIS_ROOM_SKIP_NATIVE_DRAW=1
{('export IGRIS_HEADING_DIR=' + shlex.quote(str(run))) if heading else ''}
exec ros2 launch igris_c_controller simulation.launch.py config_file:={shlex.quote(str(run / 'robot.yaml'))} > {shlex.quote(str(run / 'vendor.log'))} 2>&1'''


def wait_ready(run, seconds=40.):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        log = (run / 'vendor.log').read_text(errors='replace') if (run / 'vendor.log').exists() else ''
        if 'MujocoRos node initialized, joint states extracted and published' in log and 'Control:' in log:
            return
        if 'aborting launch' in log:
            raise RuntimeError('Vendor launch failed; inspect run/vendor.log')
        time.sleep(.2)
    raise RuntimeError('Vendor readiness timed out; inspect run/vendor.log')


def start(out, image=IMAGE, heading=False):
    """``heading`` preloads the library that can switch on the heading mode of the vendor walking module."""
    run = out / 'run'
    preload = build_native(out, heading)
    name = 'igris-umi-' + hashlib.sha256(str(run).encode()).hexdigest()[:10]
    execute(['docker', 'run', '-d', '--name', name, '--label', f'igris.room.output={run}', '--network', 'none',
             '--ipc', 'private', '--shm-size', '2g', '--ulimit', 'memlock=-1', '--ulimit', 'rtprio=99',
             '--cap-add', 'SYS_NICE', '-e', 'DISPLAY=:99', '-e', 'HOME=/tmp', '-e', 'OMP_NUM_THREADS=1',
             '-e', 'OPENBLAS_NUM_THREADS=1', '-v', f'{ROOT}:{ROOT}:ro', '-v', f'{ROOT}/igris:/igris:ro',
             '-v', f'{ROOT}/igris/configs:/robros/configs:ro',
             # Outputs may live outside the repository (/DATA): the worker reads targets and weights from ``out``.
             *([] if out.resolve().is_relative_to(ROOT) else ['-v', f'{out.resolve()}:{out.resolve()}:ro']),
             *(['-v', f'{MINK_PY}:{MINK_PY}:ro'] if MINK_PY.is_dir() else []),
             '-v', f'{run}:{run}',
             '-v', f'{run}/vendor_logs:/robros/logs', image, 'sleep', 'infinity'], stdout=subprocess.DEVNULL)
    (run / 'runtime.json').write_text(json.dumps(dict(container=name, image=image, network='none', ipc='private',
                                                      native_draw=False, hardware_executed=False,
                                                      heading_library=bool(heading)), indent=2))
    execute(['docker', 'exec', '-d', name, 'Xvfb', ':99', '-screen', '0', '800x600x24'])
    execute(['docker', 'exec', '-d', '-u', f'{os.getuid()}:{os.getgid()}', name, 'bash', '-lc',
             launch_script(run, preload, heading)])
    wait_ready(run)
    print('STARTED', name, flush=True)


def labels(out, stream_hz):
    """Turn the prepared targets into labels and the pose stream ``replay --stream`` delivers sample by sample."""
    from . import labels as module
    print(json.dumps(module.for_run(out, stream_hz), indent=2))


def bringup(out, cpu=None):
    vendor_run().bringup(out / 'run', cpu)


def replay(out, limit_seconds=0., hold_seconds=3., anchor='pelvis', targets=None, joints=False, scene=True,
           rebase_waist=False, ik='mink', teleop_posture=True, free_orientation=True, joint_speed=None,
           joint_acceleration=None, gains=None, reach_hold=True, hold_lean=None, rate_floor=None, slack=True,
           heading_hold=False, stream=None, stream_hz=None, stream_latency=0., slack_source='measured', policy=None):
    """``stream`` (a file written by ``labels.save_stream``; ``True`` for the output's own ``stream.npz``) replays
    in pose-stream mode: the worker is handed one pose at a time at ``stream_hz`` (default: the file's rate).
    ``policy`` (a dict with any of ``waist``, ``seconds``, ``delay``, ``blend``) leaves the targets to a policy
    process that shares the run folder (policy_link.py); the stream then only gives the start pose and the length."""
    if policy is not None:
        stream = stream or True; anchor = 'none'; slack_source = 'calibrated'      # nothing of the room, nothing of the simulator
    run = out / 'run'
    if (run / 'replay_result.json').exists():
        raise FileExistsError('This output already holds a replay; prepare a new one')
    if stream:
        import numpy as np
        targets = out / 'stream.npz' if stream is True else Path(stream)
        if not targets.resolve().is_relative_to(out.resolve()):
            # The container sees only the output folder (and the repository): keep the stream with the run.
            shutil.copy2(targets, run / 'stream.npz'); targets = run / 'stream.npz'
        stream_hz = stream_hz or float(np.load(targets)['rate_hz'])
    arguments = ['--out', str(run), '--targets', str(targets or out / 'targets.npz'), '--hold-seconds', str(hold_seconds)]
    if stream:
        arguments += ['--stream-hz', str(stream_hz), '--stream-latency', str(stream_latency)]
    if slack_source != 'measured':
        arguments += ['--slack', slack_source]
    if policy is not None:
        arguments += ['--policy', '--waist', policy.get('waist') or 'pinned', '--policy-seconds', str(policy.get('seconds') or 0.),
                      '--policy-delay', str(policy.get('delay', 2)), '--policy-blend', str(policy.get('blend', 4))]
        if policy.get('step_limit') is not None:
            arguments += ['--policy-step-limit', str(policy['step_limit'])]
    if limit_seconds:
        arguments += ['--limit-seconds', str(limit_seconds)]
    arguments += ['--anchor', anchor]
    if joints:
        arguments += ['--joints']
    if not scene:
        arguments += ['--no-scene']
    if rebase_waist:
        arguments += ['--rebase-waist']
    paths = [str(ROOT)]
    if ik == 'mink':
        if not MINK_PY.is_dir():
            raise FileNotFoundError(f'{MINK_PY} is missing: install mink there (see README) or set UMI_MINK_PY')
        arguments += ['--ik', 'mink', '--spheres', str(out / 'model/collision_spheres.npz')]
        paths.append(str(MINK_PY))
    else:
        arguments += ['--ik', 'numpy']
    prior = (out / 'model/fista.npz').exists() and not joints
    # With the mink IK the learned elbow prior takes the place of the posture lookup where its prediction is
    # believed; the lookup stays as what the arm falls back on where it is not (mink_ik.ELBOW_TRUST_M).
    if teleop_posture and (out / 'model/teleop_posture.npz').exists() and not joints:
        arguments += ['--posture', str(out / 'model/teleop_posture.npz')]
    if not free_orientation:
        arguments += ['--full-orientation']
    if joint_speed is not None:
        arguments += ['--joint-speed', str(joint_speed)]
    if joint_acceleration is not None:
        arguments += ['--joint-acceleration', str(joint_acceleration)]
    if not reach_hold:
        arguments += ['--no-reach-hold']
    if not slack:
        arguments += ['--no-slack-compensation']
    if heading_hold:
        arguments += ['--heading-hold']
    if hold_lean is not None:
        arguments += ['--hold-lean', *map(str, hold_lean)]
    if rate_floor is not None:
        arguments += ['--rate-floor', str(rate_floor)]
    if gains is not None:
        # Kept with the run and visible in the container, which mounts the run folder.
        shutil.copy2(gains, run / 'replay_gains.toml')
        arguments += ['--gains', str(run / 'replay_gains.toml')]
    if prior:
        arguments += ['--prior', str(out / 'model/fista.npz')]
    script = (f'export PYTHONPATH={shlex.quote(":".join(paths))}:$PYTHONPATH\n'
              'exec python3 -m igris.umi_walklow_replay.worker ' + shlex.join(arguments)
              + f' > {shlex.quote(str(run / "replay.log"))} 2>&1')
    try:
        vendor_run().shell(run, script)
    finally:
        if (run / 'replay.log').exists():
            print((run / 'replay.log').read_text()[-3000:])


def stop(out):
    run = out / 'run'
    config = vendor_run().config(run)
    execute(['docker', 'stop', '--timeout', '5', config['container']], stdout=subprocess.DEVNULL)
    print('STOPPED', config['container'])


def analyze(out):
    from . import analyze as module
    report = module.analyze(out)
    print(json.dumps(report, indent=2))
    return report


def render(out, seconds=None, fps=15., cameras=None, name='render', subframes=RENDER_SUBFRAMES):
    import numpy as np
    run = out / 'run'
    log = np.load(run / 'replay_log.npz')
    replay_rows = log['phase'] == 'replay'
    if not replay_rows.any():
        raise ValueError('No replay phase was recorded')
    first = float(log['sim'][replay_rows][0]) - 1.
    duration = seconds or float(log['sim'][-1] - first)
    command = [str(ISAAC_PYTHON), str(VENDOR_ROOM / 'render.py'), '--out', str(run), '--scene', str(out / 'scene/scene.usda'),
               '--capture', str(run / name), '--start-sim', str(first), '--seconds', str(duration), '--fps', str(fps),
               '--subframes', str(subframes), '--antialiasing', str(RENDER_ANTIALIASING), '--interpolate']
    for camera in cameras or CAMERAS:
        command += ['--camera', camera]
    with (run / f'{name}.log').open('w') as stream:
        execute(command, stdout=stream, stderr=subprocess.STDOUT)
    print(json.dumps({k: v for k, v in json.loads((run / name / 'render_report.json').read_text()).items()
                      if k != 'source_timestamps'}, indent=2)[:2000])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['prepare', 'labels', 'start', 'bringup', 'replay', 'stop', 'analyze', 'render', 'all'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, help='Local LeRobot dataset directory (prepare/all)')
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--squeeze', type=float)
    parser.add_argument('--no-palm-correction', action='store_true')
    parser.add_argument('--no-object-centric', action='store_true')
    parser.add_argument('--raw-hands', action='store_true',
                        help='Use the recorded hand frames without the left/right calibration (hand_calibration.py)')
    parser.add_argument('--case-top', type=float, help='Height of the case top [m] (default 0.77)')
    parser.add_argument('--support-clearance', type=float,
                        help='Air kept between the hand pads and the case / loading surface [m] (default 0.02; 0 off)')
    parser.add_argument('--raw-tracking', action='store_true',
                        help='Keep the tracking jumps, stalls and sample-time jitter of the recording')
    parser.add_argument('--shift', type=float, nargs=2, metavar=('DX', 'DY'))
    parser.add_argument('--carton-height', type=float)
    parser.add_argument('--cart-setback', type=float, nargs=2, metavar=('DX', 'DY'),
                        help='Roll container offset from the recorded place: along its own -x and +y [m]')
    parser.add_argument('--toe-free', type=float, nargs=2, metavar=('LOW', 'HIGH'),
                        help='Let the IK choose the toe-in within this range [deg] (preferred 12)')
    parser.add_argument('--pad', choices=['flat', 'capsule'], default='flat', help='Contact surface of the hands')
    parser.add_argument('--toe-in', type=float, default=0., metavar='DEG',
                        help='Turn the hands about their palm contact points so the fingers converge')
    parser.add_argument('--ik', choices=['numpy', 'mink'], default='mink',
                        help='IK solver of the replay: mink (QP with joint, speed and self-collision bounds) or the numpy one')
    parser.add_argument('--joint-speed', type=float,
                        help='Joint speed limit [rad/s] (default 1.0); the playback slows to keep it.  0 replays against the clock')
    parser.add_argument('--joint-acceleration', type=float, help='Joint acceleration limit [rad/s2] (default 6.0)')
    parser.add_argument('--no-reach-hold', action='store_true',
                        help='Follow every pelvis retreat with the hand targets, strained or not (anchoring.py)')
    parser.add_argument('--hold-lean', type=float, nargs=2, metavar=('START', 'FULL'),
                        help='Shares of the trunk lean limit between which following a retreat fades out (default 0.5 0.8)')
    parser.add_argument('--rate-floor', type=float,
                        help='Slowest playback rate of the speed limit (default 0.2); 0 lets an unreachable target stall the replay')
    parser.add_argument('--gains', type=Path,
                        help='LowCmd Kp/Kd profile for the waist and arms (TOML); default: the vr_box_eval replay profile')
    parser.add_argument('--no-slack-compensation', action='store_true',
                        help='Send the IK joint angles as motor targets; the hands then hang about 2.4 cm low (slack.py)')
    parser.add_argument('--stream', nargs='?', const=True, metavar='FILE',
                        help='replay/all: hand the controller one pose at a time instead of the whole episode (stream.py). '
                             'Without a file: the labels made from this output\'s targets (labels.py)')
    parser.add_argument('--stream-hz', type=float, help='Rate of the pose stream (labels: default 15, the dataset rate)')
    parser.add_argument('--stream-latency', type=float, default=0., help='Delay of every stream sample [s]')
    parser.add_argument('--slack', choices=['measured', 'calibrated'], default='measured',
                        help='Joint slack from the simulator (default) or from calibrated constants (slack.py)')
    parser.add_argument('--policy', type=Path, metavar='CHECKPOINT',
                        help='replay/all: a policy drives the robot (act_ee.py checkpoint folder); starts policy_server.py here')
    parser.add_argument('--waist', choices=['pinned', 'reference', 'yaw', 'free'], default='pinned',
                        help='What the controller does with the waist joints the policy asks for (worker.py)')
    parser.add_argument('--policy-seconds', type=float, help='How long the policy drives (default: the label length + 10 s)')
    parser.add_argument('--policy-step-limit', type=float,
                        help='Largest move of the left hand target per sample [m] (default 0.026; 0 switches the limit off)')
    parser.add_argument('--no-render', action='store_true', help='all: stop after the analysis, no Isaac Sim video')
    parser.add_argument('--heading-hold', action='store_true',
                        help='start/replay/all: switch on the heading mode of the vendor walking module at the staged heading')
    parser.add_argument('--no-teleop-posture', action='store_true',
                        help='Keep the constant preferred arm posture instead of the one looked up from the teleop recordings')
    parser.add_argument('--full-orientation', action='store_true',
                        help='Follow the recorded hand orientation in free motion too')
    parser.add_argument('--rebase-waist', action='store_true',
                        help='Turn the waist-yaw reference with the pelvis, as the hand targets are')
    parser.add_argument('--elbow-prior', nargs='?', const='human', choices=['human', 'teleop', 'swivel'],
                        help='Place the elbows with a FiSTA prior instead of the teleop posture lookup: "teleop" and "swivel" '
                             'are trained on this robot\'s teleoperated box handling ("swivel" gives the swivel angle of each '
                             'arm, "teleop" an elbow place), "human" (also when no name is given) on retargeted human motion')
    parser.add_argument('--checkpoint', type=Path, help='HL-IK FiSTA checkpoint')
    parser.add_argument('--limit-seconds', type=float, default=0.)
    parser.add_argument('--hold-seconds', type=float, default=3.)
    parser.add_argument('--anchor', choices=['pelvis', 'support', 'feet', 'none'], default='pelvis',
                        help='Room anchoring (simulator ground truth); see worker.py')
    parser.add_argument('--physics-cpu', type=int)
    parser.add_argument('--fps', type=float, default=15.)
    parser.add_argument('--render-subframes', type=int, default=RENDER_SUBFRAMES,
                        help='Renderer frames accumulated per video frame (default 16); fewer is faster, moving parts smear')
    parser.add_argument('--render-name', default='render', help='Folder under run/ for the video; must not exist yet')
    parser.add_argument('--camera', action='append', choices=['UmiOverview', 'UmiFront', 'UmiCart', 'UmiTop'],
                        help='View to render; repeat for several (default: UmiOverview, UmiFront, UmiCart)')
    args = parser.parse_args()
    out = args.out.resolve()
    cameras = [f'/World/Cameras/{name}' for name in args.camera] if args.camera else None

    def do_prepare():
        if args.dataset is None:
            parser.error('--dataset is required to prepare')
        from . import prepare as module
        module.prepare(out, args.dataset, args.episode, speed=args.speed, squeeze=args.squeeze,
                       palm_correction=not args.no_palm_correction, object_centric=not args.no_object_centric,
                       shift=args.shift, carton_height=args.carton_height, cart_setback=args.cart_setback, use_elbow_prior=args.elbow_prior,
                       checkpoint=args.checkpoint, toe_in_deg=args.toe_in, toe_free_deg=args.toe_free,
                       pad=args.pad, clean_tracking=not args.raw_tracking, calibrate_hands=not args.raw_hands,
                       case_top=args.case_top, support_clearance=args.support_clearance)

    def do_replay():
        if (args.stream is True or args.policy) and not (out / 'stream.npz').exists():
            labels(out, args.stream_hz or 15.)
        server = None
        options = {}
        if args.policy:
            # The policy process runs here, outside the simulator container: it needs the GPU and the network of
            # neither side is shared.  It waits for the run folder, answers until the controller is done, then ends.
            environment = dict(os.environ, MUJOCO_GL=os.environ.get('MUJOCO_GL', 'egl'))
            with (out / 'policy_server.log').open('w') as stream:
                server = subprocess.Popen([sys.executable, '-m', 'igris.umi_walklow_replay.policy_server', '--run', str(out / 'run'),
                                           '--checkpoint', str(args.policy.resolve())], stdout=stream, stderr=subprocess.STDOUT,
                                          env=environment, cwd=str(ROOT))
            options = dict(policy=dict(waist=args.waist, seconds=args.policy_seconds, step_limit=args.policy_step_limit))
        try:
            replay(out, args.limit_seconds, args.hold_seconds, args.anchor, rebase_waist=args.rebase_waist, ik=args.ik,
                   teleop_posture=not args.no_teleop_posture, free_orientation=not args.full_orientation,
                   joint_speed=args.joint_speed, joint_acceleration=args.joint_acceleration, gains=args.gains,
                   reach_hold=not args.no_reach_hold, hold_lean=args.hold_lean, rate_floor=args.rate_floor,
                   slack=not args.no_slack_compensation, heading_hold=args.heading_hold,
                   stream=True if args.policy else args.stream,
                   stream_hz=args.stream_hz, stream_latency=args.stream_latency, slack_source=args.slack, **options)
        finally:
            if server is not None:
                try:
                    server.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    server.terminate()

    if args.action == 'prepare':
        do_prepare()
    elif args.action == 'labels':
        labels(out, args.stream_hz or 15.)
    elif args.action == 'start':
        start(out, heading=args.heading_hold)
    elif args.action == 'bringup':
        bringup(out, args.physics_cpu)
    elif args.action == 'replay':
        do_replay()
    elif args.action == 'stop':
        stop(out)
    elif args.action == 'analyze':
        analyze(out)
    elif args.action == 'render':
        render(out, fps=args.fps, cameras=cameras, name=args.render_name, subframes=args.render_subframes)
    else:
        do_prepare()
        if args.stream is True or args.policy:
            labels(out, args.stream_hz or 15.)          # before the robot is brought up: the solve takes a minute
        start(out, heading=args.heading_hold)
        try:
            bringup(out, args.physics_cpu)
            do_replay()
        finally:
            stop(out)
        analyze(out)
        if not args.no_render:
            render(out, fps=args.fps, cameras=cameras, name=args.render_name, subframes=args.render_subframes)


if __name__ == '__main__':
    main()
