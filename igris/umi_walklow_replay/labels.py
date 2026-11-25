"""Pose labels the robot can follow at their own clock: the replay's view of the whole episode, moved into the data.

A replay may look at the whole recording.  A controller that is handed one pose at a time - the action of a
policy trained on these poses, the hand of a VR operator - may not.  Everything the replay does with that view
is done here instead, once, before training:

  shaping      grasp intervals, the two hands as one rigid pair, level at pick and set-down, grip height,
               clearance of free hands and of the support surfaces (``prepare.shape``)
  pace         the recording slowed where the joints would pass their speed and acceleration limits (governor.py)
  orientation  free hands turned the way the arm prefers instead of the operator's wrist (mink_ik.FREE_ORIENTATION)
  limits       joint ranges, coupled limits, self-collision (mink_ik.py)

The shaped targets are solved offline exactly as the replay solves them, and the label is the pose the robot
model then has.  The labels are therefore reachable, at the label clock, by a joint path inside every limit, and
a controller that receives them needs neither the grasp intervals nor a look ahead nor a slower clock
(``stream.py`` is that controller's input side).  ``source_t`` names the recorded time each label sample shows:
camera frames have to be re-timed with it, because the label clock runs slower than the recording in places.

  python -m igris.umi_walklow_replay.labels --dataset <UMI dataset> --out <new folder> [--episodes 0 7 ...]

writes per episode ``episode_XXX.npz`` (100 Hz labels and the joint path behind them, the same at the dataset
rate in the dataset's own state layout, and the recording with constants only for comparison) and ``manifest.json``.
"""
import argparse
import json
from multiprocessing import Pool
import os
from pathlib import Path

import numpy as np

from . import hand_calibration, retarget, scene
from .episode import continuous_quaternions
from .governor import blend_pose
from .kinematics import ACTIVE, UpperBody, matrix_quat

RATE_HZ = 100.
# Height of the recorded base frame above the robot's pelvis frame while it stands in WALK_LOW: loading surface
# 0.80 m + 0.10 m - pelvis height 0.8685 m.  The replay worker measures the pelvis height of each run instead.
BASE_OFFSET_Z = .0315


def feasible(targets, *, self_collision=None, posture=None, base_offset_z=BASE_OFFSET_Z, governor=None):
    """Labels for shaped 100 Hz targets (the ``made`` and ``profile`` of ``prepare.shape``, or ``targets.npz``).

    Returns ``t`` (label clock, 100 Hz), ``source_t`` (recorded time shown), ``left`` / ``right`` (N x 4 x 4 hand
    poses in the pelvis frame, without ``base_offset_z``, as in a targets file), ``head_yaw`` (minus the trunk
    heading of the solution, where a targets file has the operator's head), ``q`` (N x 31 joint path) and the
    posture ``report`` of the solve.  ``posture`` is a ``posture_prior.TeleopPosture``, ``self_collision`` a
    ``collision.SelfCollision``; ``governor`` holds options of ``governor.Governor``.
    """
    from . import offline
    from .mink_ik import ARM_POSTURE_COST_WITH_LOOKUP
    options = dict(posture_prior=posture, arm_posture_cost=ARM_POSTURE_COST_WITH_LOOKUP) if posture is not None else {}
    q, report, extra = offline.solve(targets, None, base_offset_z, solver='mink', self_collision=self_collision,
                                     free_orientation=True, governor=dict(governor or {}), **options)
    model = UpperBody.from_mjcf()
    left = np.empty((len(q), 4, 4)); right = np.empty((len(q), 4, 4))
    for index, row in enumerate(q):
        left[index], right[index] = model.fk(row)
    left[:, 2, 3] -= base_offset_z; right[:, 2, 3] -= base_offset_z
    return dict(t=np.arange(len(q)) / RATE_HZ, source_t=np.asarray(extra['path'], dtype=float), left=left, right=right,
                head_yaw=-q[:, 0], q=q, base_offset_z=float(base_offset_z), report=report)


def resample(stream, rate_hz):
    """The stream at a uniform rate from its start: poses along the straight path between neighbours, the rest linear."""
    stamps = np.asarray(stream['t'], dtype=float)
    out_t = np.arange(int(np.floor((stamps[-1] - stamps[0]) * rate_hz + 1e-9)) + 1) / float(rate_hz)
    wanted = out_t + stamps[0]
    index = np.clip(np.searchsorted(stamps, wanted, side='right') - 1, 0, len(stamps) - 2)
    share = np.clip((wanted - stamps[index]) / (stamps[index + 1] - stamps[index]), 0., 1.)
    out = dict(t=out_t)
    for key in ('left', 'right'):
        out[key] = np.array([blend_pose(stream[key][i], stream[key][i + 1], w) for i, w in zip(index, share)])
    for key in ('source_t', 'head_yaw', 'q'):
        if key in stream:
            series = np.asarray(stream[key], dtype=float)
            out[key] = ((1. - share) * series[index].T + share * series[index + 1].T).T
    return out


def state_rows(stream):
    """N x 15 rows in the layout of the dataset's ``observation.state``: left and right hand xyz + xyzw quaternion
    in the pelvis frame, then the trunk heading where the dataset has the operator's head yaw."""
    rows = np.empty((len(stream['t']), 15))
    for column, key in ((0, 'left'), (7, 'right')):
        rows[:, column:column + 3] = stream[key][:, :3, 3]
        quaternions = continuous_quaternions(np.array([matrix_quat(pose[:3, :3]) for pose in stream[key]]))
        rows[:, column + 3:column + 7] = quaternions[:, [1, 2, 3, 0]]
    rows[:, 14] = stream['head_yaw']
    return rows


def unshaped(episode, shift=scene.DEFAULT_SHIFT):
    """The recorded palm poses as robot hand targets with constants only: the calibration of the two hand frames,
    the offset from the palm to the robot's EE frame, and where the robot stands.  No repair of tracking faults,
    no grasp shaping, the recorded clock - what a source hands over that has not been preprocessed (a VR
    operator's hand, a policy trained on the recording as it is).  ``episode`` is ``episode.load_episode``'s."""
    calibrated = hand_calibration.apply(episode)
    config = json.loads(json.dumps(retarget.DEFAULT))
    config.update(palm_normal_correction=False, clearance_m=0., squeeze_m=0., base_offset=[-float(shift[0]), -float(shift[1]), 0.])
    return dict(t=np.asarray(episode['t'], dtype=float), source_t=np.asarray(episode['t'], dtype=float),
                left=retarget.ee_targets(calibrated['left'], 'left', config),
                right=retarget.ee_targets(calibrated['right'], 'right', config),
                head_yaw=np.asarray(episode['head_yaw'], dtype=float))


def save_stream(path, stream, rate_hz, **extra):
    """Write a stream (``t`` uniform at ``rate_hz``, ``left``, ``right``, ``head_yaw``, optionally ``source_t``)
    as a file the replay worker takes with ``--stream-hz``; ``extra`` must name ``surface_top`` for a scene."""
    keep = {key: stream[key] for key in ('t', 'left', 'right', 'head_yaw', 'source_t') if key in stream}
    if not np.allclose(np.diff(keep['t']), 1. / rate_hz):
        raise ValueError(f'The stream is not uniform at {rate_hz} Hz')
    np.savez_compressed(path, rate_hz=float(rate_hz), **keep, **extra)


def for_run(out, rate_hz, governor=None):
    """Labels for a prepared replay output: ``labels.npz`` (100 Hz, with the joint path) and ``stream.npz`` (the
    labels at ``rate_hz``, for ``replay --stream``).  Returns the summary also written to ``labels.json``."""
    from . import collision, posture_prior
    out = Path(out)
    if (out / 'labels.npz').exists():
        raise FileExistsError(out / 'labels.npz')
    targets = dict(np.load(out / 'targets.npz'))
    table = out / 'model/teleop_posture.npz'
    label = feasible(targets, self_collision=collision.SelfCollision(out / 'model/collision_spheres.npz'),
                     posture=posture_prior.TeleopPosture(table) if table.is_file() else None, governor=governor)
    report = label.pop('report')
    np.savez_compressed(out / 'labels.npz', surface_top=targets['surface_top'], **label)
    save_stream(out / 'stream.npz', resample(label, rate_hz), rate_hz, surface_top=targets['surface_top'])
    speed = np.abs(np.diff(label['q'][:, ACTIVE], axis=0)) * RATE_HZ
    summary = dict(recorded_s=float(targets['t'][-1]), label_s=float(label['t'][-1]), stream_hz=float(rate_hz),
                   joint_speed_max_rad_s=float(speed.max()), hand_error_p99_m=report['hand_position_error_p99_m'],
                   limited_fraction=report['limited_fraction'], base_offset_z_m=label['base_offset_z'])
    (out / 'labels.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def _episode(job):
    """One episode of the dataset: shaped, solved, summarised.  Runs in a worker process."""
    dataset, episode_index, out, spheres = job
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from . import collision, posture_prior, prepare
    try:
        shaped = prepare.shape(dataset, episode_index)
    except ValueError as error:                                       # e.g. no bimanual carry found
        return dict(episode=episode_index, error=str(error))
    targets = dict(shaped['made'], **shaped['profile'])
    table = posture_prior.TeleopPosture() if posture_prior.TABLE.is_file() else None
    label = feasible(targets, self_collision=collision.SelfCollision(spheres), posture=table)
    report = label.pop('report')
    coarse = resample(label, shaped['episode']['fps'])
    raw = unshaped(shaped['recorded'] if shaped['recorded'] is not None else shaped['episode'], shaped['shift'])
    speed = np.abs(np.diff(label['q'][:, ACTIVE], axis=0)) * RATE_HZ
    held = np.interp(label['source_t'], targets['t'], targets['orientation']) > .999
    wanted = {side: _at(targets['t'], targets[side], label['source_t'], targets, label['base_offset_z']) for side in ('left', 'right')}
    miss = np.max([np.linalg.norm(label[side][:, :3, 3] - wanted[side][:, :3, 3], axis=1) for side in ('left', 'right')], axis=0)
    summary = dict(episode=episode_index, boxes=shaped['report']['boxes'], recorded_s=float(targets['t'][-1]),
                   label_s=float(label['t'][-1]), stretch=float(label['t'][-1] / targets['t'][-1]),
                   joint_speed_max_rad_s=float(speed.max()), joint_speed_p99_rad_s=float(np.percentile(speed.max(axis=1), 99)),
                   # How far the labels are from the shaped targets where the hands are at a box (elsewhere they differ by design).
                   near_box_miss_p99_m=float(np.percentile(miss[held], 99)) if held.any() else None,
                   near_box_miss_max_m=float(miss[held].max()) if held.any() else None,
                   arm_joints_at_limit_fraction=report.get('any_arm_joint_at_limit_fraction'),
                   hand_error_p99_m=report.get('hand_position_error_p99_m'), limited_fraction=report.get('limited_fraction'), tracking_events=len(shaped['cleaning']['events']))
    np.savez_compressed(Path(out) / f'episode_{episode_index:03d}.npz',
                        **{key: label[key] for key in ('t', 'source_t', 'left', 'right', 'head_yaw', 'q')},
                        base_offset_z=label['base_offset_z'], surface_top=shaped['layout']['surface_top_m'],
                        carry_frames=np.array(shaped['report']['carry_frames']), speed=shaped['speed'],
                        state=state_rows(coarse), state_t=coarse['t'], state_source_t=coarse['source_t'], state_q=coarse['q'],
                        fps=shaped['episode']['fps'],
                        raw_state=state_rows(raw), raw_t=raw['t'],
                        shaped_left=targets['left'], shaped_right=targets['right'], shaped_t=targets['t'],
                        shaped_orientation=targets['orientation'], shaped_head_yaw=targets['head_yaw'],
                        shaped_lift_z=float(targets['pick_offset_z']) * targets['pick_weight'])
    return summary


def _at(stamps, poses, wanted, targets, base_offset_z):
    """Shaped targets at recorded times ``wanted``, lifted as the solve lifts them, minus the base height."""
    index = np.clip(np.searchsorted(stamps, wanted, side='right') - 1, 0, len(stamps) - 2)
    share = np.clip((wanted - stamps[index]) / (stamps[index + 1] - stamps[index]), 0., 1.)
    out = np.array([blend_pose(poses[i], poses[i + 1], w) for i, w in zip(index, share)])
    out[:, 2, 3] += np.interp(wanted, stamps, float(targets['pick_offset_z']) * targets['pick_weight'])
    return out


def build(dataset, out, episodes=None, jobs=16):
    """Labels for every episode (or the ones named) into a new folder ``out``; returns the manifest."""
    from . import collision
    dataset, out = Path(dataset), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    total = json.loads((dataset / 'meta/info.json').read_text())['total_episodes']
    episodes = list(range(total)) if episodes is None else [int(episode) for episode in episodes]
    spheres = collision.export(out / 'collision_spheres.npz')
    work = [(str(dataset), episode, str(out), str(spheres)) for episode in episodes]
    if jobs > 1:
        with Pool(min(jobs, len(work))) as pool:
            done = pool.map(_episode, work, chunksize=1)
    else:
        done = [_episode(job) for job in work]
    good = [row for row in done if 'error' not in row]
    manifest = dict(
        dataset=str(dataset), episodes=done, base_offset_z_m=BASE_OFFSET_Z,
        frames=dict(hands='robot EE frames in the pelvis frame; add base_offset_z to z for a robot standing in WALK_LOW',
                    state='left xyz + xyzw, right xyz + xyzw, trunk heading (in place of head.yaw), at the dataset rate',
                    source_t='recorded time shown by each label sample: re-time the camera frames with it'),
        made_by='prepare.shape -> offline solve (mink QP, speed limit by pace, teleop posture lookup, self-collision) -> FK',
        totals=dict(episodes=len(good), failed=len(done) - len(good),
                    recorded_s=float(sum(row['recorded_s'] for row in good)), label_s=float(sum(row['label_s'] for row in good)),
                    joint_speed_max_rad_s=float(max(row['joint_speed_max_rad_s'] for row in good)) if good else None))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, required=True, help='Local copy of the UMI dataset')
    parser.add_argument('--out', type=Path, required=True, help='New folder for the labels')
    parser.add_argument('--episodes', type=int, nargs='*', help='Episodes to label (default: all)')
    parser.add_argument('--jobs', type=int, default=16)
    args = parser.parse_args()
    manifest = build(args.dataset, args.out, args.episodes, args.jobs)
    print(json.dumps(manifest['totals'], indent=2))
    for row in manifest['episodes']:
        print(json.dumps(row))


if __name__ == '__main__':
    main()
