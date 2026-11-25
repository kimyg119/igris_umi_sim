"""Replay a teleoperated IGRIS-C episode recorded in joint space, and use it as a reference.

Dataset: ``kimyg119/Igris_Teleop_Box_Palletizing`` at tag ``full-signals-15hz`` (LeRobot v3, 15 Hz),
whose state and action hold all 31 SDK joints; the default revision keeps only the 14 arm joints.
The robot stood in WALK_LOW while an operator drove the waist and arms, so the recorded commands
are motions this robot and controller have already carried out.

Two replays share one targets file:
- joints: the recorded waist and arm commands are sent as they are (``--joints``);
- end effectors: the hand poses of those commands (forward kinematics, pelvis frame) and the
  recorded waist yaw go through the replay IK, as a policy's end-effector output would.
Neither stages scene objects: where the real boxes stood is not recorded.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .kinematics import ACTIVE, JOINTS, UpperBody
from .posture import arm_posture

RATE_HZ = 100.


def sdk_name(name):
    """Dataset joint name ('left_shoulder_pitch.pos') as the SDK name ('l_shoulder_pitch')."""
    name = name.removesuffix('.pos')
    for long, short in (('left_', 'l_'), ('right_', 'r_')):
        if name.startswith(long):
            name = short + name[len(long):]
    return name


def load_episode(root, episode):
    """Commands and measurements of one episode: ``t`` (N), ``action`` and ``state`` (N x 31), ``task``."""
    import pyarrow.parquet as pq
    root = Path(root)
    info = json.loads((root / 'meta/info.json').read_text())
    for feature in ('action', 'observation.state'):
        names = [sdk_name(name) for name in info['features'][feature]['names']]
        if names != JOINTS:
            raise ValueError(f'{feature} is not the 31-joint SDK order; use the full-signals-15hz revision')
    rows = [row for path in sorted((root / 'meta/episodes').rglob('*.parquet'))
            for row in pq.read_table(path, columns=['episode_index', 'tasks', 'length', 'data/chunk_index',
                                                    'data/file_index']).to_pylist()]
    row = next((row for row in rows if row['episode_index'] == episode), None)
    if row is None:
        raise ValueError(f'Episode {episode} is not in the dataset (0..{len(rows) - 1})')
    path = root / info['data_path'].format(chunk_index=row['data/chunk_index'], file_index=row['data/file_index'])
    table = pq.read_table(path, columns=['episode_index', 'timestamp', 'action', 'observation.state'])
    keep = np.flatnonzero(table.column('episode_index').to_numpy() == episode)
    if len(keep) != row['length']:
        raise ValueError(f'Episode {episode}: {len(keep)} rows found, {row["length"]} expected')
    table = table.take(keep)
    return dict(t=table.column('timestamp').to_numpy().astype(float),
                action=np.array(table.column('action').to_pylist(), dtype=float),
                state=np.array(table.column('observation.state').to_pylist(), dtype=float),
                task=row['tasks'][0] if row['tasks'] else '', fps=float(info['fps']), episode=int(episode))


def check(data, model):
    """Units, limits and speeds of the recorded commands, as numbers a reader can judge."""
    action, state, t = data['action'], data['state'], data['t']
    lower, upper = model.lower[ACTIVE], model.upper[ACTIVE]
    owned = action[:, ACTIVE]
    outside = np.maximum(lower - owned, owned - upper)
    speed = np.abs(np.diff(owned, axis=0)) / np.diff(t)[:, None]
    worst = int(np.argmax(outside.max(axis=0)))
    return dict(
        frames=int(len(t)), seconds=float(t[-1] - t[0]), fps=data['fps'], task=data['task'],
        command_beyond_model_limit_max_rad=float(max(0., outside.max())), command_beyond_limit_joint=JOINTS[ACTIVE[worst]],
        command_beyond_limit_fraction=float((outside > 1e-6).any(axis=1).mean()),
        command_speed_max_rad_s=float(speed.max()), command_speed_p99_rad_s=float(np.percentile(speed, 99)),
        command_minus_measured_rms_rad=float(np.sqrt(np.mean((owned - state[:, ACTIVE]) ** 2))),
        waist_range_rad={JOINTS[i]: [float(action[:, i].min()), float(action[:, i].max())] for i in range(3)},
        leg_measured_mean_rad={JOINTS[i]: float(state[:, i].mean()) for i in (3, 6, 7)},
        neck_command_rad=[float(action[:, 29].mean()), float(action[:, 30].mean())])


def targets(data, model, rate_hz=RATE_HZ):
    """100 Hz commands and their hand poses.  Legs and neck are not replayed (set to zero here)."""
    t = np.arange(0., data['t'][-1] - data['t'][0], 1. / rate_hz)
    q = np.zeros((len(t), len(JOINTS)))
    for joint in ACTIVE:
        q[:, joint] = np.interp(t, data['t'] - data['t'][0], data['action'][:, joint])
    left = np.empty((len(t), 4, 4)); right = np.empty((len(t), 4, 4))
    for i, row in enumerate(q):
        left[i], right[i] = model.fk(row)
    # The worker takes the waist-yaw reference as minus ``head_yaw`` (the UMI convention).
    return dict(t=t, q=q, left=left, right=right, head_yaw=-q[:, 0], surface_top=0., speed=1.,
                carry_frames=np.zeros((0, 2), dtype=int))


def prepare(out, teleop_root, episode, scene_dataset):
    """Scene, model and run folder from the UMI preparation, plus the teleop targets."""
    from . import prepare as prepare_module
    out = Path(out)
    prepare_module.prepare(out, scene_dataset, 0)
    model = UpperBody.from_mjcf()
    data = load_episode(teleop_root, episode)
    report = check(data, model)
    np.savez_compressed(out / 'teleop_targets.npz', **targets(data, model))
    np.savez_compressed(out / 'teleop_episode.npz', t=data['t'], action=data['action'], state=data['state'])
    (out / 'teleop.json').write_text(json.dumps(dict(dataset=str(teleop_root), episode=int(episode), **report), indent=2) + '\n')
    return report


def analyze(out):
    """Tracking, balance and posture of a teleop replay; for the end-effector replay also IK against the recording."""
    out = Path(out); run = out / 'run'
    log = np.load(run / 'replay_log.npz'); result = json.loads((run / 'replay_result.json').read_text())
    target = np.load(out / 'teleop_targets.npz')
    replay = log['phase'] == 'replay'
    if not replay.any():
        raise ValueError('No replay phase was recorded')
    model = UpperBody.from_mjcf()
    sim = log['sim'][replay] - log['sim'][replay][0]; wall = log['wall'][replay]; index = log['index'][replay]
    command = log['q_command'][replay]; measured = log['q_measured'][replay]; pelvis = log['pelvis'][replay]
    error = (measured - command)[:, ACTIVE]
    worst = int(np.argmax(np.abs(error).max(axis=0)))
    recorded = target['q'][index]
    # Hands reached in simulation against the recorded hand poses (pelvis frame).
    hand_error = np.array([[np.linalg.norm(pose[:3, 3] - wanted[i][:3, 3]) for pose, wanted in
                            zip(model.fk(q), (target['left'], target['right']))] for q, i in zip(measured[::5], index[::5])])
    w, x, y, z = pelvis[:, 3:].T
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
    lower, upper = model.lower[ACTIVE] + .035, model.upper[ACTIVE] - .035
    posture = arm_posture(model, command[::3], lower, upper)[0]
    reference = arm_posture(model, recorded[::3], lower, upper)
    report = dict(
        output=str(out), mode='joints' if result.get('joints') else 'end_effectors', hardware_executed=False,
        worker_completed=bool(result.get('pass')), worker_error=result.get('error'),
        timing=dict(replay_sim_s=float(sim[-1]), realtime_factor=float(sim[-1] / (wall[-1] - wall[0])),
                    command_rate_hz=float(len(sim) / (wall[-1] - wall[0]))),
        joint_tracking=dict(rms_rad=float(np.sqrt(np.mean(error ** 2))), max_rad=float(np.abs(error).max()),
                            worst_joint=JOINTS[ACTIVE[worst]], p99_rad=float(np.percentile(np.abs(error), 99))),
        hand_position_error_m=dict(mean=float(hand_error.mean()), p95=float(np.percentile(hand_error, 95)),
                                   max=float(hand_error.max())),
        balance=dict(pelvis_min_z_m=float(pelvis[:, 2].min()), pelvis_xy_range_m=np.ptp(pelvis[:, :2], axis=0).tolist(),
                     pelvis_tilt_max_deg=float(tilt.max()), pelvis_tilt_mean_deg=float(tilt.mean())),
        arm_posture=posture, recorded_arm_posture=reference[0])
    if not result.get('joints'):
        difference = (command - recorded)[:, ACTIVE]
        elbows = arm_posture(model, command[::3], lower, upper)[1]
        report['ik_against_recording'] = dict(
            joint_rms_rad=float(np.sqrt(np.mean(difference ** 2))), joint_max_rad=float(np.abs(difference).max()),
            waist_rms_rad=np.sqrt(np.mean(difference[:, :3] ** 2, axis=0)).tolist(),
            elbow_distance_mean_m=np.linalg.norm(elbows - reference[1], axis=-1).mean(axis=0).tolist(),
            elbow_distance_max_m=np.linalg.norm(elbows - reference[1], axis=-1).max(axis=0).tolist(),
            ik_position_error_p99_m=float(np.percentile(log['position_error'][replay], 99)),
            ik_limited_fraction=float(log['saturated'][replay].mean()))
    (run / 'teleop_analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    from . import run as run_module
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['prepare', 'replay', 'analyze'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--teleop-root', type=Path, help='Local copy of the full-signals-15hz revision')
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--scene-dataset', type=Path, help='Local UMI dataset; its episode 0 supplies the room scene')
    parser.add_argument('--end-effectors', action='store_true', help='Replay the hand poses through the IK instead of the joints')
    parser.add_argument('--limit-seconds', type=float, default=0.)
    parser.add_argument('--hold-seconds', type=float, default=2.)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.action == 'prepare':
        if not args.teleop_root or not args.scene_dataset:
            parser.error('prepare needs --teleop-root and --scene-dataset')
        print(json.dumps(prepare(out, args.teleop_root, args.episode, args.scene_dataset), indent=2))
    elif args.action == 'replay':
        run_module.replay(out, args.limit_seconds, args.hold_seconds, 'none', targets=out / 'teleop_targets.npz',
                          joints=not args.end_effectors, scene=False,
                          # The teleop commands press the hands together; this check keeps the plain numpy IK.
                          ik='numpy', teleop_posture=False, free_orientation=False, joint_speed=0.)
    else:
        print(json.dumps(analyze(out), indent=2))


if __name__ == '__main__':
    main()
