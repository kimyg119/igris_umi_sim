"""Replay UMI hand targets through online IK and SDK LowCmd in vendor WALK_LOW.

Runs inside the private simulator container (network none, private IPC).  The
vendor walking controller keeps the legs; this process is the single LowCmd
writer for waist 3 + arms 14 (neck 2 held) and sends zero gains to the legs.

Phases, all timed on the native simulation clock:
  1. take the upper body over from the bring-up hold publisher,
  2. move to the first recorded pose (arms out, fold, reach),
  3. place the scene around the measured pelvis and let it settle,
  4. replay: every 10 ms solve IK for the current target and publish LowCmd,
  5. hold the last pose and write the log.

Room anchoring.  The recording's base frame was fixed in the room, but the
robot's pelvis is not: with the vendor contact settings the feet creep on the
floor (about 20 cm of stance widening and 10 deg of heading per minute), and
the pelvis sways over the feet by up to 10 cm and 6 deg with posture.  A box
set down 9 cm off its support tips over, so by default (``--anchor pelvis``)
every target is corrected for the measured pelvis pose, low-passed and bounded.
That correction feeds back through the walking controller - more reach, more
lean, more pelvis shift - and ran away at the lowest box until the IK's trunk
lean was limited (WAIST_* below), and still did in a third of the runs until a retreat of the
pelvis along the reach stopped being followed while the upper body is strained (anchoring.py,
``--no-reach-hold`` for the old behaviour).  ``--anchor support`` corrects foot creep
plus heading and sideways sway, ``--anchor feet`` foot creep only, and
``--anchor none`` replays in the moving pelvis frame.  Pelvis and foot poses
are simulator ground truth; a real robot would have to estimate them.

Pose stream (``--stream-hz``).  The targets file is then a stream of hand poses at that rate, handed to the
controller one sample at a time as a policy or a VR operator would (stream.py): no look ahead, the clock is
never slowed, the hand orientation always counts in full and nothing is known of grasps.  What the replay does
with its view of the whole episode has to be in the poses already (labels.py).  Room anchoring is still applied
to the incoming poses: it stands in for the eyes of whoever sends them.  ``--anchor none`` leaves that out too,
and ``--slack calibrated`` replaces the simulator's slack measurement by constants.

Policy (``--policy``).  The targets come from a policy process outside the container, through two files in the run
folder (policy_link.py): hand targets in the pelvis frame and waist joint targets, in chunks, each made from an
observation of this controller's joint state and the robot's cameras.  Nothing of the room is used then - no
anchoring, no simulator slack - and the targets file only says where the robot starts and how long the run is.
``--waist`` chooses what is done with the waist the policy asks for: ``pinned`` (default) sends the waist there
and solves the arms alone on that chest, ``reference`` makes it the waist's preferred posture in the joint IK,
``yaw`` takes only its yaw as that preference, ``free`` ignores it.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import tomllib

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from igris.hl_ik.native_clock import read_clock                      # noqa: E402
from igris.umi_walklow_replay.elbow_prior import ElbowPrior, FiSTA, PERIOD_S, ready_pose   # noqa: E402
from igris.umi_walklow_replay.anchoring import LEAN_FULL, LEAN_START, ReachHold, strain         # noqa: E402
from igris.umi_walklow_replay.heading import HeadingTarget, read_status, torso_heading         # noqa: E402
from igris.umi_walklow_replay.governor import JOINT_ACCELERATION, JOINT_SPEED, RATE_FLOOR, Governor   # noqa: E402
from igris.umi_walklow_replay.ik import IK                           # noqa: E402
from igris.umi_walklow_replay.posture_prior import TeleopPosture     # noqa: E402
from igris.umi_walklow_replay.retarget import PELVIS_TO_SURFACE      # noqa: E402
from igris.umi_walklow_replay.slack import SlackCompensation, calibrated, read_slack   # noqa: E402
from igris.umi_walklow_replay.stream import PoseStream, Recorded     # noqa: E402
from igris.umi_walklow_replay.policy_link import ControllerLink, PairLimiter, STEP_LIMIT_M, TURN_LIMIT_RAD     # noqa: E402
from igris.umi_walklow_replay.kinematics import ACTIVE, JOINTS, UpperBody   # noqa: E402

NAMESPACE = 'igris_c_001'
OWNED = [0, 1, 2, *range(15, 31)]            # waist, both arms, neck
GAINS = ROOT / 'igris/sim_local/vr_box_eval/results/20260914_synthetic_motion100_1kg/replay_gains.toml'
RATE_HZ = 100.
ANCHOR_TIME_CONSTANT = .25                   # low-pass of the pelvis pose used for room anchoring [s]
ANCHOR_LIMIT_M = .15                         # largest pelvis displacement the anchoring will follow
ANCHOR_LIMIT_RAD = math.radians(25.)         # largest pelvis rotation the anchoring will follow
# Trunk lean the IK may use.  A leaning trunk moves the centre of mass and the
# walking controller shifts the pelvis in response; with room anchoring that
# shift comes back as more reach.  A narrow range keeps the loop from running away:
# a target the arms cannot reach is left short instead.
WAIST_ROLL_LIMIT = .15
WAIST_PITCH_LIMITS = (-.15, .08)             # negative pitch leans the trunk forward
# Heading hold corrects by stepping only while the hand orientation weight of the targets is below this, i.e. in
# free motion between a release and the next approach (also with ``--full-orientation``).  Stepping with a box in the hands moved the pelvis 17-22 cm
# and cost two of three boxes in one of three runs of episode 7.
HEADING_FREE_HANDS = .5
# A pose stream is engaged gently.  Its first pose, anchored in the room, is not where the robot was brought to
# (the anchoring also undoes the pelvis tilt: 3-4 cm at the hands), and with no slowed playback to absorb it the
# arms would cover that at the solver's 2 rad/s.  For this long before the first sample counts, the controller
# goes to the first pose at no more than JOINT_SPEED.
STREAM_LEAD_IN_S = 1.
WAIST_MODES = ('pinned', 'reference', 'yaw', 'free')
POLICY_WAIT_S = 240.                         # longest the robot holds its start pose for the policy process to get ready


def quintic(u):
    u = min(1., max(0., u))
    return u ** 3 * (10 - 15 * u + 6 * u * u)


def gains(path=GAINS):
    profile = tomllib.loads(Path(path).read_text())['motion']['gains']
    kp = np.zeros(31); kd = np.zeros(31)
    for index in ACTIVE:
        name = JOINTS[index]
        name = name.replace('l_', 'left_', 1) if name.startswith('l_') else \
            name.replace('r_', 'right_', 1) if name.startswith('r_') else name
        kp[index] = profile[name]['kp']; kd[index] = profile[name]['kd']
    kp[29:] = 30.; kd[29:] = 3.
    return kp, kd


def yaw_of(quaternion):
    w, x, y, z = quaternion
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Native:
    """Read-only view of the vendor physics state written by the observer."""

    def __init__(self, out):
        self.out = out
        names = [b['name'] for b in json.loads((out / 'model.json').read_text())['bodies']]
        self.count = len(names)
        self.pelvis = names.index('pelvis')
        self.feet = [names.index('l_foot_original'), names.index('r_foot_original')]
        self.hands = [names.index('end_effector_left'), names.index('end_effector_right')]
        self.parcels = [names.index(n) for n in sorted(n for n in names if n.startswith('room_Parcel_'))]
        self.supports = [names.index(n) for n in ('room_Case', 'room_Rolltainer')]

    def poses(self):
        row = np.frombuffer((self.out / 'latest.bin').read_bytes(), dtype=np.float64)
        if len(row) != 2 + 7 * self.count or not np.isfinite(row).all() or not 0 <= time.time() - row[0] < .5:
            raise RuntimeError('Missing, invalid or stale native physics state')
        return row[2:].reshape(-1, 7)

    def sim_time(self):
        return read_clock(self.out / 'native_clock.bin')['sim']


def feet_frame(poses, feet):
    """Level support frame: midpoint of the feet and their mean heading."""
    position = poses[feet, :2].mean(axis=0)
    headings = [yaw_of(poses[foot, 3:]) for foot in feet]
    heading = math.atan2(sum(map(math.sin, headings)), sum(map(math.cos, headings)))
    return np.array([position[0], position[1], heading])


def planar(frame, z=0.):
    """4x4 transform of a level frame given as x, y, heading."""
    c, s = math.cos(frame[2]), math.sin(frame[2])
    out = np.eye(4)
    out[:2, :2] = [[c, -s], [s, c]]; out[:3, 3] = [frame[0], frame[1], z]
    return out


def run(out, targets_path, limit_seconds, hold_seconds, anchor_mode, prior_path=None, joints=False, scene=True,
        rebase_waist=False, ik_backend='numpy', spheres=None, posture_path=None, free_orientation=True,
        joint_speed=JOINT_SPEED, joint_acceleration=JOINT_ACCELERATION, gains_path=None, reach_hold=True,
        hold_lean=(LEAN_START, LEAN_FULL), rate_floor=RATE_FLOOR, slack=True, heading_hold=False,
        stream_hz=None, stream_latency=0., slack_source='measured', policy=None):
    """Replay targets.  ``joints`` sends the recorded joint angles (``q`` of the targets file) instead of
    solving IK; ``scene=False`` leaves the scene objects parked and takes the targets as given in the
    pelvis frame (no height offset, no room anchoring).  ``slack`` corrects the motor targets for the
    measured joint slack of the vendor model (slack.py); recorded joint angles are sent as they are.
    ``heading_hold`` switches on the heading mode of the vendor walking module so that it keeps the pelvis at
    its staged heading (heading.py; the simulator must have been started with the heading library).
    ``stream_hz`` takes the targets file as a pose stream of that rate, delivered one sample at a time
    (``stream_latency`` seconds late), and switches off everything that needs more than the present pose: the
    slowed playback, the look ahead and the per-sample orientation weight.  ``joint_speed`` is then a bound of
    the IK (mink) instead of a pace.  ``slack_source`` is 'measured'
    (simulator ground truth) or 'calibrated' (constants, slack.py).  ``policy`` (a dict: ``waist`` mode,
    ``seconds``, ``delay`` and ``blend`` in samples) hands the targets over to a policy process (module text)."""
    if policy:
        if not stream_hz or joints or ik_backend != 'mink':
            raise ValueError('A policy drives the mink IK at the rate of a pose stream')
        if policy['waist'] not in WAIST_MODES:
            raise ValueError(f"Unknown waist mode {policy['waist']!r}")
        anchor_mode = 'none'                     # the targets are relative to the robot's own hands: nothing to anchor
    if slack_source not in ('measured', 'calibrated'):
        raise ValueError(f'Unknown slack source {slack_source!r}')
    if stream_hz and joints:
        raise ValueError('A pose stream cannot be replayed as joint angles')
    if not Path('/.dockerenv').exists():
        raise RuntimeError('The replay worker must run in the private simulator container')
    runtime = json.loads((out / 'runtime.json').read_text())
    if runtime['network'] != 'none' or runtime['ipc'] != 'private':
        raise RuntimeError('Private simulation required')
    if heading_hold and not runtime.get('heading_library'):
        raise RuntimeError('Heading hold needs a simulator started with the heading library (run.py start --heading-hold)')
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from igris_c_sdk.msg import LowCmd, LowState, RobotState

    data = np.load(targets_path)
    left_targets = data['left'].copy(); right_targets = data['right'].copy()
    head_yaw = data['head_yaw']; stamps = data['t']
    surface_top = float(data['surface_top'])
    # Extra vertical shift of the targets near the pick: the case is not as high as the loading surface (prepare).
    pick_shift = (float(data['pick_offset_z']) * data['pick_weight'] if 'pick_weight' in data.files and not joints
                  else np.zeros(len(stamps)))
    joint_targets = data['q'].copy() if joints else None
    toe_hold = data['toe_hold'].copy() if 'toe_hold' in data.files else None
    # How much the hand orientation matters at each sample (prepare): 1 near and during a grasp, 0 in free motion.
    orientation = data['orientation'].copy() if free_orientation and 'orientation' in data.files and not stream_hz \
        else np.ones(len(stamps))
    # Recorded time each stream sample shows (labels.py): the box outcomes are judged against the recording.
    source_time = data['source_t'].copy() if stream_hz and 'source_t' in data.files else stamps.copy()
    # The same profile tells where the hands are busy with a box (heading hold steps only where they are not).
    hands_busy = data['orientation'].copy() if 'orientation' in data.files else np.zeros(len(stamps))
    if not scene:
        anchor_mode = 'none'
    if limit_seconds:
        keep = stamps <= limit_seconds
        left_targets, right_targets, head_yaw, stamps = left_targets[keep], right_targets[keep], head_yaw[keep], stamps[keep]
        if joints:
            joint_targets = joint_targets[keep]
        if toe_hold is not None:
            toe_hold = toe_hold[keep]
        orientation = orientation[keep]; pick_shift = pick_shift[keep]; hands_busy = hands_busy[keep]
        source_time = source_time[keep]
    model = UpperBody.from_mjcf()
    toe = dict(toe_range=data['toe_range'], toe_preference=float(data['toe_preference']), toe_pivot=data['toe_pivot']) \
        if 'toe_range' in data.files and not joints else {}
    if ik_backend == 'mink':
        if toe:
            raise ValueError('The mink IK does not take a free toe-in')
        from igris.umi_walklow_replay.collision import SelfCollision
        from igris.umi_walklow_replay.mink_ik import MinkIK
        from igris.umi_walklow_replay.mink_ik import ARM_POSTURE_COST_WITH_LOOKUP
        # A stream's clock cannot be slowed, so there the joint speed limit is a bound of the QP itself: where the
        # poses ask for more (a chunk boundary of a policy, the pelvis moving under room-anchored targets) the hands
        # fall behind for a moment.  Offline that costs nothing on the labels and at most 7 mm on a policy's output.
        bound = dict(velocity_limit=joint_speed, waist_velocity_limit=joint_speed) if stream_hz and joint_speed else {}
        ik = MinkIK(model, dt=1. / RATE_HZ, waist_roll_limit=WAIST_ROLL_LIMIT, waist_pitch_limits=WAIST_PITCH_LIMITS,
                    self_collision=SelfCollision(spheres) if spheres else None,
                    arm_posture_cost=ARM_POSTURE_COST_WITH_LOOKUP if posture_path else None, **bound)
    else:
        ik = IK(model, dt=1. / RATE_HZ, waist_roll_limit=WAIST_ROLL_LIMIT, waist_pitch_limits=WAIST_PITCH_LIMITS, **toe)
    native = Native(out)
    kp, kd = gains(gains_path or GAINS)
    # HL-IK elbow prior: predicts where a human elbow would be; a secondary IK task.
    network = FiSTA(prior_path) if prior_path else None
    prior = ElbowPrior(model, network) if network is not None else None
    lookahead = int(round(PERIOD_S * RATE_HZ))
    # Joint speed and acceleration limits: the playback slows where the recorded motion is too fast (governor.py).
    pace = Governor(stamps, speed=joint_speed, acceleration=joint_acceleration, floor=rate_floor) \
        if joint_speed and not joints and not stream_hz else None
    progress = {'path': 0., 'shortfall': 0., 'strain': 0.}
    # A pelvis retreat along the reach is not passed on to the hand targets while the upper body is strained.
    retreat = ReachHold() if reach_hold and anchor_mode == 'pelvis' and not joints else None
    # Preferred arm posture looked up from the teleoperated robot recordings for the hand poses being commanded.
    posture_prior = TeleopPosture(posture_path) if posture_path and not joints else None
    # The links stand at motor angle + joint slack: ask the motors for the IK's angles minus the measured slack.
    compensation = SlackCompensation(model.lower, model.upper) if slack and not joints else None
    ready_ticks = 3000 if network is None else 1500      # per round; the prior's ready pose takes several rounds
    # Boxes the elbow side of the arms keeps clear of while following the prior (simulator ground truth).
    parcel_half = np.array(json.loads((out / 'manifest.json').read_text())['umi_replay']['layout']['parcel_size_m']) / 2

    rclpy.init(); node = rclpy.create_node('umi_walklow_replay')
    state = {'q': None, 'dq': None, 'tau': None, 'received': 0., 'mode': None, 'mode_at': 0., 'environment': None}

    def on_state(sample):
        joints = sample.joint_state
        state['q'] = np.array([joints[i].q for i in range(31)], dtype=float)
        state['dq'] = np.array([joints[i].dq for i in range(31)], dtype=float)
        state['tau'] = np.array([joints[i].tau_est for i in range(31)], dtype=float)
        state['received'] = time.monotonic()

    def on_mode(sample):
        if sample.environment != RobotState.ROBOT_ENV_SIM:
            raise RuntimeError('Simulation RobotState required; refusing to command anything else')
        state['mode'] = int(sample.state); state['mode_at'] = time.monotonic()

    node.create_subscription(LowState, f'{NAMESPACE}/rt/lowstate', on_state, qos_profile_sensor_data)
    node.create_subscription(RobotState, f'{NAMESPACE}/rt/robotstate', on_mode, qos_profile_sensor_data)
    publisher = node.create_publisher(LowCmd, f'{NAMESPACE}/rt/lowcmd', qos_profile_sensor_data)

    def refresh():
        for _ in range(4):
            rclpy.spin_once(node, timeout_sec=0)

    def walking():
        return state['mode'] == RobotState.ROBOT_STATE_WALK_LOW

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        refresh()
        if state['q'] is not None and walking():
            break
        time.sleep(.01)
    else:
        raise RuntimeError(f"Fresh WALK_LOW required; mode={state['mode']}")

    sequence = 0
    sent = np.zeros(31)                          # motor targets of the last LowCmd

    def motor(command, dt):
        """Motor angles that put the links at ``command`` (the command itself without slack compensation)."""
        if compensation is None:
            return command
        if slack_source == 'calibrated':
            compensation.update(calibrated(), dt)
        else:
            measured = read_slack(out / 'joint_offsets.bin', time.time())
            compensation.update(measured[1] if measured else None, dt)
        return compensation.motor(command)

    heading_file = None                          # target of the vendor heading mode, once it is switched on
    staged_heading = 0.
    heading_wanted = {'pelvis': 0.}              # pelvis heading the walking module is asked to keep

    def hold_heading():
        """Tell the walking module which torso heading goes with the pelvis facing ``heading_wanted``."""
        if heading_file is not None:
            links = state['q'] + (compensation.filtered if compensation is not None else 0.)
            heading_file.send(True, torso_heading(model, links, heading_wanted['pelvis']))

    def publish(q, dq):
        nonlocal sequence
        sent[:] = q
        hold_heading()
        message = LowCmd(); message.header.seq = sequence; message.header.frame_id = ''
        message.kinematic_modes = [1] * 5                       # joint-space (PJS) on all parallel mechanisms
        for i, motor in enumerate(message.motors):
            motor.id = i
            owned = i in OWNED
            motor.q = float(q[i]) if owned else 0.; motor.dq = float(dq[i]) if owned else 0.; motor.tau = 0.
            motor.kp = float(kp[i]) if owned else 0.; motor.kd = float(kd[i]) if owned else 0.
        publisher.publish(message); sequence += 1

    def check():
        if (out / 'stop.request').exists():
            raise RuntimeError('Stop requested: ' + (out / 'stop.request').read_text())
        now = time.monotonic()
        if not walking() or now - state['received'] > .25 or now - state['mode_at'] > .5:
            raise RuntimeError(f"Stale state or WALK_LOW lost: mode={state['mode']}")
        poses = native.poses()
        if poses[native.pelvis, 2] < .7:
            raise RuntimeError('Robot fell below 0.7 m')
        return poses

    # 1. Single-writer hand-over from the bring-up hold publisher.
    if (out / 'upper_hold.ready').exists():
        (out / 'upper_hold.stop').write_text('umi replay taking ownership')
        deadline = time.monotonic() + 2
        while not (out / 'upper_hold.stopped').exists():
            if time.monotonic() > deadline:
                raise RuntimeError('Upper hold did not release command ownership')
            time.sleep(.005)
        hold = json.loads((out / 'upper_hold_result.json').read_text())
        if hold['error']:
            raise RuntimeError('Upper hold failed: ' + hold['error'])

    link = None                                  # the policy process, once it drives the robot
    log = {key: [] for key in ('wall', 'sim', 'phase', 'index', 'q_command', 'q_motor', 'q_measured', 'heading_wanted', 'tau', 'position_error', 'policy_observed',
                               'rotation_error', 'elbow_error', 'elbow_trust', 'toe', 'gap', 'relaxed', 'path', 'shortfall', 'strain', 'saturated', 'pelvis', 'hands', 'parcels', 'supports', 'feet',
                               'target_left', 'target_right', 'target_left_pose', 'target_right_pose')}
    result = {'pass': False, 'hardware_executed': False, 'targets': str(targets_path), 'samples': int(len(stamps)),
              'rate_hz': RATE_HZ, 'anchor': anchor_mode, 'joints': bool(joints), 'scene': bool(scene),
              'rebase_waist': bool(rebase_waist), 'ik': ik_backend, 'gains': str(gains_path or GAINS),
              'waist_kp_kd': [kp[:3].tolist(), kd[:3].tolist()], 'teleop_posture': bool(posture_path and not joints),
              'free_orientation': bool(free_orientation),
              'joint_speed_rad_s': float(joint_speed) if pace is not None or (stream_hz and joint_speed) else None,
              'joint_speed_by': 'playback pace' if pace is not None else 'QP bound' if stream_hz and joint_speed else None,
              'joint_acceleration_rad_s2': float(joint_acceleration) if pace is not None else None,
              'rate_floor': float(rate_floor) if pace is not None else None,
              'reach_hold': retreat is not None, 'slack_compensation': compensation is not None,
              'slack_source': slack_source if compensation is not None else None,
              'stream_hz': float(stream_hz) if stream_hz else None,
              'stream_latency_s': float(stream_latency) if stream_hz else None,
              'heading_hold': bool(heading_hold),
              'reach_hold_lean': list(map(float, hold_lean)) if retreat is not None else None, 'self_collision': bool(ik_backend == 'mink' and spheres), 'toe_range_rad': toe['toe_range'].tolist() if toe else None, 'elbow_prior': str(prior_path) if prior_path else None,
              'elbow_prior_sha256': network.sha256 if network is not None else None}

    def record(phase, index, command, info, poses, left, right):
        log['wall'].append(time.time()); log['sim'].append(native.sim_time()); log['phase'].append(phase)
        log['index'].append(index); log['q_command'].append(command.copy()); log['q_measured'].append(state['q'].copy())
        log['q_motor'].append(sent.copy()); log['heading_wanted'].append(heading_wanted['pelvis'])
        log['policy_observed'].append(link.used if link is not None else -1)        # observation behind the target in use
        log['tau'].append(state['tau'].copy())
        log['position_error'].append(info['position_error'] if info else np.zeros(2))
        log['rotation_error'].append(info['rotation_error'] if info else np.zeros(2))
        log['elbow_error'].append(info['elbow_error'] if info else np.zeros(2))
        log['elbow_trust'].append(info.get('elbow_trust', np.zeros(2)) if info else np.zeros(2))
        log['toe'].append(info['toe'] if info and 'toe' in info else np.zeros(2))
        log['gap'].append(info.get('gap', np.nan) if info else np.nan); log['relaxed'].append(info.get('relaxed', 0) if info else 0)
        log['path'].append(progress['path']); log['shortfall'].append(progress['shortfall'])
        log['strain'].append(progress['strain'])
        log['saturated'].append(bool(info['saturated']) if info else False)
        log['pelvis'].append(poses[native.pelvis].copy()); log['hands'].append(poses[native.hands].copy())
        log['parcels'].append(poses[native.parcels].copy()); log['supports'].append(poses[native.supports].copy())
        log['feet'].append(poses[native.feet].copy())
        log['target_left'].append(left[:3, 3].copy()); log['target_right'].append(right[:3, 3].copy())
        # The whole hand targets the IK was given (pelvis frame): the action labels of a dataset are made from them.
        log['target_left_pose'].append(np.array(left, dtype=float)); log['target_right_pose'].append(np.array(right, dtype=float))

    def tick(next_wall):
        delay = next_wall - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        refresh()
        return max(next_wall, time.monotonic() - .5 / RATE_HZ) + 1. / RATE_HZ

    def offset_targets(pelvis_z):
        """Shift every target so the recorded surfaces meet the simulated ones."""
        dz = surface_top + PELVIS_TO_SURFACE - pelvis_z if scene else 0.
        left = left_targets.copy(); right = right_targets.copy()
        left[:, 2, 3] += dz + (pick_shift if scene else 0.); right[:, 2, 3] += dz + (pick_shift if scene else 0.)
        return left, right, dz

    def first_pose(q_now, left, right, rounds):
        """Configuration at the first sample: the recorded joints, or the IK solution for the first targets."""
        if joints:
            first = q_now.copy(); first[ACTIVE] = joint_targets[0][ACTIVE]
            return first, dict(position_error=np.zeros(2), rotation_error=np.zeros(2))
        if posture_prior is not None:
            # Start in the posture the lookup prefers for the first hand poses, not in the constant one.
            return ik.converge(q_now, left, right, waist_yaw_reference=-head_yaw[0],
                               posture=posture_prior.reference(model, q_now, left, right), ticks=ready_ticks)
        return ready_pose(ik, model, q_now, left, right, -head_yaw[0], network, rounds=rounds, ticks=ready_ticks)

    def move(q_from, waypoints, phase, left, right, speed=.5):
        """Quintic joint-space segments; durations from the largest joint change."""
        command = q_from.copy(); previous = command.copy()
        points = [q_from, *waypoints]
        next_wall = time.monotonic()
        for a, b in zip(points[:-1], points[1:]):
            duration = max(2., float(np.max(abs(b - a))) * 1.875 / speed)
            start = native.sim_time(); last = start
            while True:
                next_wall = tick(next_wall)
                poses = check()
                now = native.sim_time(); u = (now - start) / duration
                command = a + quintic(u) * (b - a)
                dt = max(now - last, 1e-4); last = now
                publish(command, np.clip((command - previous) / dt, -2., 2.)); previous = command.copy()
                record(phase, -1, command, None, poses, left, right)
                if u >= 1.:
                    break
        return command

    try:
        refresh()
        initial = state['q'].copy()
        poses = check()
        left, right, dz = offset_targets(poses[native.pelvis, 2])
        seed = initial.copy()
        ready, info = first_pose(seed, left[0], right[0], 4)
        spread = initial.copy(); spread[16] = .9; spread[23] = -.9          # arms out before folding
        folded = spread.copy(); folded[18] = ready[18]; folded[25] = ready[25]
        command = move(initial, [spread, folded, ready], 'ready', left[0], right[0])
        # 2b. The pelvis settles differently with the arms raised: re-reference once.
        next_wall = time.monotonic()
        for _ in range(100):
            next_wall = tick(next_wall); poses = check(); publish(command, np.zeros(31))
        left, right, dz = offset_targets(poses[native.pelvis, 2])
        ready, info = first_pose(command.copy(), left[0], right[0], 2)
        command = move(command, [ready], 'ready', left[0], right[0])
        result.update(base_offset_z_m=dz, pick_offset_z_m=float(data['pick_offset_z']) if 'pick_offset_z' in data.files else 0., ready_position_error_m=info['position_error'].tolist(),
                      ready_rotation_error_rad=info['rotation_error'].tolist(),
                      pelvis_at_staging=poses[native.pelvis].tolist())

        # 3. Stage the scene around the measured pelvis (reference pose: origin, yaw 0).
        pelvis = poses[native.pelvis]
        if scene:
            request = out / 'room_layout.request.tmp'
            request.write_text(f'{pelvis[0]} {pelvis[1]} {yaw_of(pelvis[3:])}\n')
            request.replace(out / 'room_layout.request')
        if heading_hold:
            # From here on the vendor walking module turns the base back once the pelvis is 0.1 rad off this heading.
            staged_heading = heading_wanted['pelvis'] = yaw_of(pelvis[3:])
            heading_file = HeadingTarget(out / 'heading_target.bin')
        start = native.sim_time(); next_wall = time.monotonic()
        while native.sim_time() - start < 2.5 or (heading_hold and not (out / 'heading_applied.json').exists()
                                                  and native.sim_time() - start < 15.):
            # The slack correction comes in here, over 2 s: the hands rise to where the IK has them.
            # (Heading mode: the library first has to find the walking module in the controller's memory.)
            next_wall = tick(next_wall); poses = check(); publish(motor(command, 1. / RATE_HZ), np.zeros(31))
            record('settle', -1, command, None, poses, left[0], right[0])
        if scene and not (out / 'room_layout_applied.json').exists():
            raise RuntimeError('Scene staging was not acknowledged')
        if heading_hold:
            if not (out / 'heading_applied.json').exists():
                raise RuntimeError('Heading mode request was not acknowledged')
            answer = json.loads((out / 'heading_applied.json').read_text())
            on = [name for name, module in answer['modules'].items() if module['enabled'] == 1]
            if answer['error'] or not on:
                raise RuntimeError(f'Heading mode was not switched on: {answer}')
            result['heading_mode'] = dict(answer, modules_on=on, pelvis_heading_rad=staged_heading)
        # The recorded base frame is level and fixed in the room: anchor it at the
        # staged pelvis position and heading, without the pelvis tilt.
        heading = yaw_of(pelvis[3:])
        anchor = np.array([*pelvis[:3], math.cos(heading / 2), 0., 0., math.sin(heading / 2)])
        tracked = poses[native.pelvis].copy()
        anchor_planar = planar([pelvis[0], pelvis[1], heading])
        feet_at_staging = feet_frame(poses, native.feet)
        tracked_feet = feet_at_staging.copy()
        tracked_pelvis = np.array([pelvis[0], pelvis[1], heading])
        correction = np.eye(4)
        result['feet_at_staging'] = feet_at_staging.tolist()
        result['parcels_after_settle'] = poses[native.parcels].tolist()

        if policy:
            # The policy process loads its network and its cameras while the robot is brought up; its clock starts
            # only when it says it can answer (policy_server.py writes ``policy_ready``).
            waited = native.sim_time(); next_wall = time.monotonic()
            while not (out / 'policy_ready').exists():
                if native.sim_time() - waited > POLICY_WAIT_S:
                    raise RuntimeError('The policy process did not become ready')
                next_wall = tick(next_wall); poses = check(); publish(motor(command, 1. / RATE_HZ), np.zeros(31))
                record('settle', -1, command, None, poses, left[0], right[0])
            result['policy_wait_s'] = float(native.sim_time() - waited)

        # 4. Replay with online IK.
        last = native.sim_time(); start = last + (STREAM_LEAD_IN_S if stream_hz else 0.); next_wall = time.monotonic()
        previous = command.copy(); index = 0; anchor_limited = False
        cosine, sine = math.cos(yaw_of(anchor[3:])), math.sin(yaw_of(anchor[3:]))
        to_anchor = np.array([[cosine, sine, 0.], [-sine, cosine, 0.], [0., 0., 1.]])     # room -> staging pelvis axes
        reach_error = 0.
        if prior is not None:
            prior.reset(command, 0.)
        feed = live = None
        if policy:
            # Until the policy's first chunk is in use the robot stays at the first pose of the labels it was brought to.
            limiter = PairLimiter(left[0], policy.get('step_limit', STEP_LIMIT_M), policy.get('turn_limit', TURN_LIMIT_RAD)) \
                if policy.get('step_limit', STEP_LIMIT_M) > 0 else None
            link = ControllerLink(out / 'policy', stream_hz, delay=policy['delay'], blend=policy['blend'], limiter=limiter)
            live = PoseStream(1. / stream_hz)
            live.push(-STREAM_LEAD_IN_S, left[0], right[0], command[[0, 1, 2]].copy())
            policy_seconds = float(policy['seconds'] or stamps[-1] + 10.)
        elif stream_hz:
            # The poses arrive one at a time from here on.  The first is the one the robot was brought to.
            feed = Recorded(dict(t=stamps, left=left, right=right, head_yaw=head_yaw), stream_latency)
            live = PoseStream(1. / stream_hz)
            for sample in feed.due(stamps[0] + stream_latency):
                live.push(-STREAM_LEAD_IN_S, *sample)
            if prior is not None:
                prior.reset(command, -STREAM_LEAD_IN_S)
        result['replay_start_sim'] = float(start)             # simulation time of stream sample 0 (after the lead-in)
        print('REPLAY_STARTED', start, len(stamps), flush=True)
        while True:
            next_wall = tick(next_wall)
            poses = check()
            now = native.sim_time(); elapsed = now - start
            index = min(len(stamps) - 1, int(np.searchsorted(stamps, elapsed, side='right')) - 1)
            index = max(index, 0)
            ik.dt = float(np.clip(now - last, 1e-4, .05)); last = now
            waist_wanted = None
            if link is not None:
                link.publish(sequence, now, elapsed, state['q'], command)
                for sample in link.due(elapsed):
                    live.push(elapsed, *sample)
                target_left, target_right, waist_wanted = live.target(elapsed)
                heading = -float(waist_wanted[0]); weight = 1.
                index = int(np.clip(link.plan.sample(elapsed), 0, len(stamps) - 1))
            elif live is not None:
                for sample in feed.due(elapsed):
                    live.push(elapsed, *sample)
                target_left, target_right, heading = live.target(elapsed)
                index = max(feed.next - 1, 0); weight = 1.
            elif pace is not None:
                # Targets a full step ahead on the recorded path; the step is shortened below if it is too fast.
                index, (target_left, target_right, heading, weight) = pace.sample(pace.ahead(ik.dt), left, right, head_yaw, orientation)
            else:
                target_left, target_right, heading, weight = left[index], right[index], head_yaw[index], orientation[index]
            turn = 0.                  # how far the pelvis has turned since staging, as applied to the hand targets [rad]
            blend = 1. - math.exp(-ik.dt / ANCHOR_TIME_CONSTANT)
            if anchor_mode in ('feet', 'support'):
                # Creep of the feet since staging (low-passed, bounded).
                now_feet = feet_frame(poses, native.feet)
                tracked_feet[:2] += blend * (now_feet[:2] - tracked_feet[:2])
                tracked_feet[2] += blend * math.atan2(math.sin(now_feet[2] - tracked_feet[2]), math.cos(now_feet[2] - tracked_feet[2]))
                drift = planar(tracked_feet) @ np.linalg.inv(planar(feet_at_staging))
                follow = drift @ anchor_planar                   # where foot creep alone has taken the base frame
                if anchor_mode == 'support':
                    # Add the pelvis sway over the feet, except its part along the reach.
                    tracked_pelvis[:2] += blend * (poses[native.pelvis, :2] - tracked_pelvis[:2])
                    heading_now = yaw_of(poses[native.pelvis, 3:])
                    tracked_pelvis[2] += blend * math.atan2(math.sin(heading_now - tracked_pelvis[2]), math.cos(heading_now - tracked_pelvis[2]))
                    sway = tracked_pelvis[:2] - follow[:2, 3]
                    middle = (target_left[:3, 3] + target_right[:3, 3]) / 2
                    c, s = math.cos(tracked_pelvis[2]), math.sin(tracked_pelvis[2])
                    reach = np.array([c * middle[0] - s * middle[1], s * middle[0] + c * middle[1]])
                    if np.linalg.norm(reach) > .1:
                        reach /= np.linalg.norm(reach)
                        sway = sway - np.dot(sway, reach) * reach
                    follow = planar([follow[0, 3] + sway[0], follow[1, 3] + sway[1], tracked_pelvis[2]])
                moved = np.linalg.norm(follow[:2, 3] - anchor_planar[:2, 3])
                turned = abs(math.atan2((follow[:2, :2] @ anchor_planar[:2, :2].T)[1, 0], (follow[:2, :2] @ anchor_planar[:2, :2].T)[0, 0]))
                if moved > ANCHOR_LIMIT_M or turned > ANCHOR_LIMIT_RAD:
                    anchor_limited = True
                else:
                    correction = np.linalg.inv(follow) @ anchor_planar
                target_left, target_right = correction @ target_left, correction @ target_right
            elif anchor_mode == 'pelvis':
                # Keep the targets fixed in the room: undo all pelvis motion since staging.
                tracked[:3] += blend * (poses[native.pelvis, :3] - tracked[:3])
                sign = 1. if np.dot(tracked[3:], poses[native.pelvis, 3:]) >= 0 else -1.
                tracked[3:] += blend * (sign * poses[native.pelvis, 3:] - tracked[3:])
                tracked[3:] /= np.linalg.norm(tracked[3:])
                followed = tracked.copy()
                offset = followed[:3] - anchor[:3]
                distance = np.linalg.norm(offset)
                if distance > ANCHOR_LIMIT_M:
                    followed[:3] = anchor[:3] + offset * ANCHOR_LIMIT_M / distance
                if retreat is not None:
                    # Do not chase a retreat along the reach while the trunk is leaning hard or the hands are short.
                    level = strain(command[1], command[2], reach_error, roll_limit=WAIST_ROLL_LIMIT,
                                   forward_pitch_limit=WAIST_PITCH_LIMITS[0], start=hold_lean[0], full=hold_lean[1])
                    middle = (target_left[:2, 3] + target_right[:2, 3]) / 2
                    followed[:3] = anchor[:3] + to_anchor.T @ retreat.update(to_anchor @ (followed[:3] - anchor[:3]),
                                                                             middle, level, ik.dt)
                    progress['shortfall'], progress['strain'] = retreat.shortfall, level
                relative = quaternion_multiply(quaternion_conjugate(anchor[3:]), followed[3:])
                angle = 2 * math.acos(min(1., abs(relative[0])))
                if angle > ANCHOR_LIMIT_RAD:
                    axis = relative[1:] / np.linalg.norm(relative[1:]) * (1. if relative[0] >= 0 else -1.)
                    limited = np.r_[math.cos(ANCHOR_LIMIT_RAD / 2), math.sin(ANCHOR_LIMIT_RAD / 2) * axis]
                    followed[3:] = quaternion_multiply(anchor[3:], limited)
                anchor_limited = anchor_limited or distance > ANCHOR_LIMIT_M or angle > ANCHOR_LIMIT_RAD
                target_left, target_right = (rebase(anchor, followed, t) for t in (target_left, target_right))
                turn = yaw_of(followed[3:]) - yaw_of(anchor[3:])
                turn = math.atan2(math.sin(turn), math.cos(turn))
            # The recorded trunk heading is fixed in the room like the hand targets.  Left as recorded
            # while the pelvis has turned, the trunk no longer faces the hands and the far arm reaches
            # across the chest (offline: arm joints at a limit 9 % -> 30 % of the time for a 25 deg turn).
            waist_reference = -heading + (turn if rebase_waist else 0.)
            elbows = keep_out = None
            if prior is not None:
                if live is not None:
                    goal_left, goal_right = live.newest[:2]              # no look ahead: the sample the target is moving to
                else:
                    ahead = min(len(stamps) - 1, index + lookahead)
                    goal_left, goal_right = left[ahead], right[ahead]
                if anchor_mode in ('feet', 'support'):
                    goal_left, goal_right = correction @ goal_left, correction @ goal_right
                elif anchor_mode == 'pelvis':
                    goal_left, goal_right = (rebase(anchor, followed, t) for t in (goal_left, goal_right))
                elbows = prior.targets(command, elapsed, goal_left, goal_right)
                to_pelvis = np.linalg.inv(pose_matrix(poses[native.pelvis]))
                boxes = [to_pelvis @ pose_matrix(row) for row in poses[native.parcels]]
                keep_out = [(box[:3, 3], box[:3, :3], parcel_half) for box in boxes]
            if joints:
                command = command.copy(); command[ACTIVE] = joint_targets[index][ACTIVE]; info = None
            else:
                posture = posture_prior.reference(model, command, target_left, target_right, ik.dt) \
                    if posture_prior is not None else None
                waist = {}
                if link is not None:
                    waist = dict(waist_yaw_reference=None) if policy['waist'] == 'free' else \
                        dict(waist_target=waist_wanted, waist_pinned=policy['waist'] == 'pinned') if policy['waist'] != 'yaw' else {}
                wanted, info = ik.step(command, target_left, target_right, **{'waist_yaw_reference': waist_reference, **waist},
                                       elbow_targets=elbows, keep_out=keep_out,
                                       toe_hold=0. if toe_hold is None else float(toe_hold[index]),
                                       posture=posture, orientation=float(weight))
                if live is not None and elapsed < 0.:
                    step = wanted - command                          # lead-in: toward the first pose, slowly
                    command = command + step / max(1., float(np.max(np.abs(step[ACTIVE]))) / (JOINT_SPEED * ik.dt))
                else:
                    command = pace.settle(command, wanted, ik.dt) if pace is not None else wanted
                reach_error = float(np.max(info['position_error']))
            if link is not None:
                progress['path'] = max(elapsed, 0.)                  # a policy keeps no recorded time: its own clock
            elif live is not None:
                # Recorded time the target shows: the stream as it was one sample period (and the latency) ago.
                progress['path'] = float(np.interp(elapsed - stream_latency - 1. / stream_hz, stamps, source_time))
            else:
                progress['path'] = pace.tau if pace is not None else float(min(elapsed, stamps[-1]))
            # The walking module corrects the heading by stepping.  It may do so while the hands are free; near
            # and during a grasp it is asked for the heading the pelvis has, which is no correction at all.
            heading_wanted['pelvis'] = staged_heading if hands_busy[index] < HEADING_FREE_HANDS \
                else yaw_of(poses[native.pelvis, 3:])
            publish(motor(command, ik.dt), np.clip((command - previous) / ik.dt, -2.5, 2.5)); previous = command.copy()
            record('approach' if live is not None and elapsed < 0. else 'replay', index, command, info, poses, target_left, target_right)
            if link is not None:
                if elapsed >= policy_seconds:
                    link.publish(sequence, now, elapsed, state['q'], command, done=True)
                    break
            elif live is not None:
                if elapsed >= stamps[-1] + stream_latency + 1. / stream_hz:
                    break
            elif pace.done if pace is not None else elapsed >= stamps[-1]:
                break
        result['replay_sim_seconds'] = float(native.sim_time() - start)
        result['recorded_seconds'] = float(stamps[-1])
        result['anchor_limit_reached'] = bool(anchor_limited)
        if link is not None:
            result['policy'] = dict(link.report(), waist=policy['waist'], seconds=policy_seconds)
        result['reach_shortfall_max_m'] = float(np.max(log['shortfall'])) if log['shortfall'] else 0.
        if compensation is not None:
            result['slack_correction_rad'] = {JOINTS[j]: float(compensation.correction[j]) for j in ACTIVE}

        # 5. Hold so the last placement can be judged.
        start = native.sim_time(); next_wall = time.monotonic()
        while native.sim_time() - start < hold_seconds:
            heading_wanted['pelvis'] = staged_heading
            next_wall = tick(next_wall); poses = check(); publish(motor(command, 1. / RATE_HZ), np.zeros(31))
            record('hold', len(stamps) - 1, command, None, poses, left[-1], right[-1])
        result['pass'] = True
    except Exception as error:
        result['error'] = str(error)
        raise
    finally:
        if link is not None and state['q'] is not None:
            link.publish(sequence, native.sim_time(), -1., state['q'], sent, done=True)      # the policy process may stop
        if heading_file is not None:
            result['heading_status'] = read_status(out / 'heading_status.bin')
            heading_file.send(False, 0.)             # leave the walking module as it was found
            time.sleep(.05)
            result['heading_status_after_release'] = read_status(out / 'heading_status.bin')
            heading_file.close(); heading_file = None
        if log['wall']:
            phases = np.array(log.pop('phase'))
            np.savez_compressed(out / 'replay_log.npz', phase=phases, joints=np.array(JOINTS),
                                **{key: np.array(value) for key, value in log.items()})
        result['commands_sent'] = sequence
        (out / 'replay_result.json').write_text(json.dumps(result, indent=2) + '\n')
        print('REPLAY_RESULT', json.dumps(result), flush=True)
        node.destroy_node(); rclpy.shutdown()


def quaternion_multiply(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quaternion_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def pose_matrix(pose):
    w, x, y, z = pose[3:]
    out = np.eye(4)
    out[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                   [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                   [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    out[:3, 3] = pose[:3]
    return out


def rebase(anchor, pelvis, target):
    """Express a target recorded for the staging pelvis in the current pelvis frame."""
    return np.linalg.inv(pose_matrix(pelvis)) @ pose_matrix(anchor) @ target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='Prepared vendor run directory')
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--limit-seconds', type=float, default=0., help='Replay only this much of the episode')
    parser.add_argument('--hold-seconds', type=float, default=3.)
    parser.add_argument('--anchor', choices=['pelvis', 'support', 'feet', 'none'], default='pelvis',
                        help='Room anchoring from simulator ground truth: full pelvis pose (default), foot creep plus '
                             'heading/sideways sway, foot creep only, or the moving pelvis frame')
    parser.add_argument('--prior', type=Path, help='Exported HL-IK FiSTA weights; omit to leave the elbows to the joint posture')
    parser.add_argument('--joints', action='store_true', help="Send the targets file's recorded joint angles; no IK")
    parser.add_argument('--ik', choices=['numpy', 'mink'], default='numpy', dest='ik_backend')
    parser.add_argument('--joint-speed', type=float, default=JOINT_SPEED,
                        help='Joint speed limit [rad/s]; the playback slows to keep it.  0 replays against the clock')
    parser.add_argument('--joint-acceleration', type=float, default=JOINT_ACCELERATION, help='Joint acceleration limit [rad/s2]')
    parser.add_argument('--posture', type=Path, help='Teleop posture table (prepare); omit for the constant preferred posture')
    parser.add_argument('--full-orientation', action='store_true', help='Follow the recorded hand orientation in free motion too')
    parser.add_argument('--spheres', type=Path, help='Collision spheres (prepare) for the self-collision bounds of --ik mink')
    parser.add_argument('--rebase-waist', action='store_true',
                        help='Turn the waist-yaw reference with the pelvis, as the hand targets are (pelvis anchor only)')
    parser.add_argument('--gains', type=Path, help='LowCmd Kp/Kd profile (TOML, same layout as the default one)')
    parser.add_argument('--no-reach-hold', action='store_true',
                        help='Follow every pelvis retreat with the hand targets, strained or not (pelvis anchor)')
    parser.add_argument('--hold-lean', type=float, nargs=2, default=(LEAN_START, LEAN_FULL), metavar=('START', 'FULL'),
                        help='Shares of the trunk lean limit between which following a retreat fades out')
    parser.add_argument('--rate-floor', type=float, default=RATE_FLOOR,
                        help='Slowest playback rate of the speed limit; 0 lets an unreachable target stall the replay')
    parser.add_argument('--no-slack-compensation', action='store_true',
                        help='Send the IK joint angles as motor targets; the hands then hang about 2.4 cm low (slack.py)')
    parser.add_argument('--heading-hold', action='store_true',
                        help='Switch on the heading mode of the vendor walking module at the staged heading')
    parser.add_argument('--stream-hz', type=float,
                        help='Take --targets as a pose stream of this rate, delivered one sample at a time (stream.py)')
    parser.add_argument('--stream-latency', type=float, default=0., help='Delay of every stream sample [s]')
    parser.add_argument('--slack', choices=['measured', 'calibrated'], default='measured', dest='slack_source',
                        help='Joint slack from the simulator (default) or from calibrated constants')
    parser.add_argument('--policy', action='store_true',
                        help='Take hand and waist targets from a policy process through <out>/policy (policy_link.py)')
    parser.add_argument('--policy-seconds', type=float, default=0., help='How long the policy drives (default: the targets\' length + 10 s)')
    parser.add_argument('--policy-delay', type=int, default=2, help='Samples after its observation from which a chunk is used')
    parser.add_argument('--policy-blend', type=int, default=4, help='Samples over which a new chunk is faded in')
    parser.add_argument('--waist', choices=WAIST_MODES, default='pinned', help='What is done with the waist a policy asks for')
    parser.add_argument('--policy-step-limit', type=float, default=STEP_LIMIT_M,
                        help='Largest move of the left hand target per stream sample [m], the pair kept (policy_link.py); 0 hands the targets out as they come')
    parser.add_argument('--policy-turn-limit', type=float, default=math.degrees(TURN_LIMIT_RAD), help='... and its largest turn [deg]')
    parser.add_argument('--no-scene', action='store_true',
                        help='Leave the scene objects parked; targets are pelvis-frame as given (no offset, no anchoring)')
    args = parser.parse_args()
    run(args.out.resolve(), args.targets.resolve(), args.limit_seconds, args.hold_seconds, args.anchor,
        args.prior.resolve() if args.prior else None, joints=args.joints, scene=not args.no_scene,
        rebase_waist=args.rebase_waist, ik_backend=args.ik_backend, spheres=args.spheres.resolve() if args.spheres else None,
        posture_path=args.posture.resolve() if args.posture else None, free_orientation=not args.full_orientation,
        joint_speed=args.joint_speed, joint_acceleration=args.joint_acceleration,
        gains_path=args.gains.resolve() if args.gains else None, reach_hold=not args.no_reach_hold,
        hold_lean=tuple(args.hold_lean), rate_floor=args.rate_floor, slack=not args.no_slack_compensation,
        heading_hold=args.heading_hold, stream_hz=args.stream_hz, stream_latency=args.stream_latency,
        slack_source=args.slack_source,
        policy=dict(waist=args.waist, seconds=args.policy_seconds, delay=args.policy_delay, blend=args.policy_blend,
                    step_limit=args.policy_step_limit, turn_limit=math.radians(args.policy_turn_limit)) if args.policy else None)


if __name__ == '__main__':
    main()
