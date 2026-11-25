"""One whole run inside the simulator container, for a host where nothing may run outside a container.

``run.py`` drives the simulator from the host: it compiles there and reaches the container with ``docker
exec``.  Here the same steps run from inside the container the simulator image was started in, and no docker
command is used at all:

  docker run --rm --network none --ipc private --shm-size 2g --ulimit memlock=-1 --ulimit rtprio=99 \\
      --cap-add SYS_NICE --cpuset-cpus 40-43 -u $(id -u):$(id -g) -e HOME=/tmp \\
      -v <repository>:<repository>:ro -v <repository>/igris:/igris:ro -v <repository>/igris/configs:/robros/configs:ro \\
      -v <data>:<data> -v <logs>:/robros/logs igris-sim:jazzy \\
      python3 -m igris.umi_walklow_replay.container_run --out <new folder> --dataset <UMI dataset> --episode 7 --stream

The container is the unit of isolation and of clean-up: the vendor simulation is started as a child of this
process and goes away with the container.  ``PYTHONPATH`` must hold the repository, the mink packages
(run.MINK_PY) and, for ``prepare``, pxr, pyarrow and PIL.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from . import run as host

PREFIX = ('source /opt/ros/jazzy/setup.bash\nsource /igris/ws/install/setup.bash\n'
          'source /igris/sim_local/tools_ws/install/setup.bash\n')
# CPUs everything but the physics thread runs on (``reserve``); empty: wherever the container allows.
OTHERS = []


def reserve(physics_cpu=None):
    """Keep one CPU, and the other thread of its core, for the physics thread alone; returns that CPU.

    The physics thread runs at real-time priority and waits for the controller's answer in a busy loop.  When the
    scheduler puts the controller on the same CPU, the controller only runs in the 5 % that real-time tasks must
    leave: on a container with six CPUs that happened in 2 of 8 parallel runs (a physics step took 11.8 ms instead
    of 0.73 ms and the robot never stood up).  Everything this module starts is therefore kept off that core.
    """
    allowed = sorted(os.sched_getaffinity(0))
    cpu = allowed[0] if physics_cpu is None else int(physics_cpu)
    if cpu not in allowed:
        raise ValueError(f'CPU {cpu} is not available to this container')
    core = {cpu}
    siblings = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list')
    if siblings.exists():
        for part in siblings.read_text().strip().split(','):
            ends = [int(value) for value in part.split('-')]
            core.update(range(ends[0], ends[-1] + 1))
    rest = [other for other in allowed if other not in core] or [other for other in allowed if other != cpu]
    OTHERS[:] = rest
    return cpu


def elsewhere(command):
    """``command`` confined to the CPUs that are not the physics thread's."""
    return ['taskset', '-c', ','.join(map(str, OTHERS)), *command] if OTHERS else list(command)


def isolated():
    """True in a container that has no network interface but the loopback."""
    return Path('/.dockerenv').exists() and sorted(os.listdir('/sys/class/net')) == ['lo']


def local_shell(out, script, **kwargs):
    """``isaac_vendor_room/run.py``'s ``shell`` without docker: the script runs here."""
    prefix = PREFIX + f'export CYCLONEDDS_URI=file://{out / "cyclonedds.xml"}\n'
    return subprocess.run(elsewhere(['bash', '-lc', prefix + script]), check=True, **kwargs)


def local_config(out):
    return json.loads((out / 'runtime.json').read_text())


def patch():
    """Point the host launcher's helpers at this container."""
    vendor = host.vendor_run()
    vendor.shell = local_shell; vendor.config = local_config
    return vendor


def start(out, heading=False):
    """Compile, start Xvfb and the vendor simulation as children of this process; returns the simulation."""
    if not isolated():
        raise RuntimeError('This runs only inside a simulator container without a network')
    run = out / 'run'
    preload = host.build_native(out, heading)
    (run / 'runtime.json').write_text(json.dumps(dict(
        container=socket.gethostname(), image=os.environ.get('IGRIS_SIM_IMAGE', host.IMAGE), network='none', ipc='private',
        native_draw=False, hardware_executed=False, heading_library=bool(heading), launched='inside the container'), indent=2))
    display = subprocess.Popen(['Xvfb', ':99', '-screen', '0', '800x600x24'], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, start_new_session=True)
    environment = dict(os.environ, DISPLAY=':99', HOME='/tmp', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    simulation = subprocess.Popen(elsewhere(['bash', '-lc', host.launch_script(run, preload, heading)]), env=environment,
                                  start_new_session=True)
    try:
        host.wait_ready(run)
    except Exception:
        stop([simulation, display]); raise
    print('STARTED inside', socket.gethostname(), flush=True)
    return [simulation, display]


def stop(processes):
    """End the process groups started by ``start``."""
    for process in processes:
        for sign in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            if process.poll() is not None:
                break
            try:
                os.killpg(process.pid, sign)
            except ProcessLookupError:
                break
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                continue


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True, help='New output folder (as for run.py)')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--stream', nargs='?', const=True, metavar='FILE',
                        help='Hand the controller one pose at a time; without a file, the labels of this output')
    parser.add_argument('--stream-hz', type=float)
    parser.add_argument('--stream-latency', type=float, default=0.)
    parser.add_argument('--slack', choices=['measured', 'calibrated'], default='measured')
    parser.add_argument('--anchor', choices=['pelvis', 'support', 'feet', 'none'], default='pelvis')
    parser.add_argument('--limit-seconds', type=float, default=0.)
    parser.add_argument('--hold-seconds', type=float, default=3.)
    parser.add_argument('--physics-cpu', type=int)
    parser.add_argument('--policy', action='store_true',
                        help='Take the hand and waist targets from a policy process that shares this output (policy_link.py)')
    parser.add_argument('--policy-seconds', type=float, help='How long the policy drives the robot (default: the label length + 10 s)')
    parser.add_argument('--waist', choices=['pinned', 'reference', 'yaw', 'free'], default='pinned',
                        help='What the controller does with the waist joints a policy asks for (worker.py)')
    parser.add_argument('--policy-delay', type=int, default=2, help='Samples after its observation from which a chunk is used')
    parser.add_argument('--policy-step-limit', type=float,
                        help='Largest move of the left hand target per sample [m] (default: policy_link.STEP_LIMIT_M); 0 switches the limit off')
    parser.add_argument('--policy-blend', type=int, default=4,
                        help='Samples over which a new chunk is faded in; more than the 4 between two chunks averages over several chunks')
    args = parser.parse_args()
    out = args.out.resolve()
    os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    from . import prepare
    started = time.monotonic()
    prepare.prepare(out, args.dataset, args.episode)
    if args.stream is True or args.policy:
        host.labels(out, args.stream_hz or 15.)
    patch()
    physics_cpu = reserve(args.physics_cpu)
    processes = start(out)
    status = 0
    try:
        host.bringup(out, physics_cpu)             # pins the physics thread there, on its own
        options = dict(policy=dict(seconds=args.policy_seconds, waist=args.waist, delay=args.policy_delay,
                                   blend=args.policy_blend, step_limit=args.policy_step_limit)) if args.policy else {}
        host.replay(out, args.limit_seconds, args.hold_seconds, args.anchor, stream=True if args.policy else args.stream,
                    stream_hz=args.stream_hz, stream_latency=args.stream_latency, slack_source=args.slack, **options)
    except Exception as error:                                   # the analysis below still says what was reached
        print('RUN_FAILED', repr(error), flush=True); status = 1
    finally:
        stop(processes)
    try:
        report = host.analyze(out)
        print('RUN_DONE', json.dumps(dict(out=str(out), episode=args.episode, boxes_stacked=report['boxes_stacked'],
                                          stack_success=report['stack_success'], wall_s=round(time.monotonic() - started, 1))),
              flush=True)
    except Exception as error:
        print('ANALYSIS_FAILED', repr(error), flush=True); status = status or 2
    sys.exit(status)


if __name__ == '__main__':
    main()
