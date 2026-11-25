"""Solve an episode's targets offline, without the simulator, and describe the arm posture.

Uses the same IK settings as the replay worker.  With ``--prior`` the HL-IK
elbow prior runs exactly as in the worker (30 Hz history of its own commands).
This checks reach and posture only: no balance, contact or pelvis motion.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .elbow_prior import ElbowPrior, FiSTA, PERIOD_S, ready_pose
from .ik import IK, POSTURE
from .kinematics import ACTIVE, JOINTS, UpperBody
from .posture import arm_posture
from .worker import RATE_HZ, WAIST_PITCH_LIMITS, WAIST_ROLL_LIMIT


def solve(targets, prior_path=None, base_offset_z=0., pin_waist_yaw=False, waist_roll_limit=WAIST_ROLL_LIMIT,
          waist_pitch_limits=WAIST_PITCH_LIMITS, solver='numpy', posture_prior=None, free_orientation=False,
          governor=None, **ik_options):
    """Roll the IK over 100 Hz targets.  Returns commands (T x 31), the posture report and diagnostics.

    ``governor`` is a dict of ``governor.Governor`` options (``{}`` for its defaults): the playback then slows
    where joint speed or acceleration would exceed the limits, and T grows accordingly.
    """
    left, right = targets['left'].copy(), targets['right'].copy()
    lift = base_offset_z + (float(targets['pick_offset_z']) * targets['pick_weight'] if 'pick_weight' in targets else 0.)
    left[:, 2, 3] += lift; right[:, 2, 3] += lift
    yaw = targets['head_yaw']; stamps = targets['t']
    orientation = targets['orientation'] if free_orientation and 'orientation' in targets else np.ones(len(stamps))
    model = UpperBody.from_mjcf()
    if solver == 'mink':
        from .mink_ik import MinkIK
        ik = MinkIK(model, dt=1. / RATE_HZ, waist_roll_limit=waist_roll_limit, waist_pitch_limits=waist_pitch_limits, **ik_options)
    else:
        ik = IK(model, dt=1. / RATE_HZ, waist_roll_limit=waist_roll_limit, waist_pitch_limits=waist_pitch_limits,
                pin_waist_yaw=pin_waist_yaw, **ik_options)
    network = FiSTA(prior_path) if prior_path else None
    q = np.zeros(len(JOINTS)); q[ACTIVE] = POSTURE
    q, info = ready_pose(ik, model, q, left[0], right[0], -yaw[0], network)
    prior = ElbowPrior(model, network) if network is not None else None
    if prior is not None:
        prior.reset(q, 0.)
    lookahead = int(round(PERIOD_S * RATE_HZ))
    path = stamps
    if governor is not None:
        from .governor import Governor
        from .kinematics import rotation_vector
        pace = Governor(stamps, **governor); dt = 1. / RATE_HZ
        rows, position, rotation, limited, path, elbow_rows = [], [], [], [], [], []
        while not pace.done and len(rows) < 20 * len(stamps):
            index, (target_left, target_right, heading, weight) = pace.sample(pace.ahead(dt), left, right, yaw, orientation)
            posture = posture_prior.reference(model, q, target_left, target_right, dt) if posture_prior is not None else None
            targets_elbow = None
            if prior is not None:                                        # as in the worker: its clock is the control clock
                ahead = min(len(stamps) - 1, index + lookahead)
                targets_elbow = prior.targets(q, len(rows) * dt, left[ahead], right[ahead])
            wanted, info = ik.step(q, target_left, target_right, waist_yaw_reference=-heading, posture=posture,
                                   orientation=float(weight), elbow_targets=targets_elbow)
            q = pace.settle(q, wanted, dt)
            _, reached = pace.sample(pace.tau, left, right)              # where on the path the shortened step ended
            poses = model.fk(q)
            position.append([np.linalg.norm(a[:3, 3] - b[:3, 3]) for a, b in zip(reached, poses)])
            rotation.append([np.linalg.norm(rotation_vector(a[:3, :3] @ b[:3, :3].T)) for a, b in zip(reached, poses)])
            rows.append(q); limited.append(info['saturated']); path.append(pace.tau); elbow_rows.append(info['elbow_error'])
        commands = np.array(rows); position = np.array(position); rotation = np.array(rotation)
        limited = np.array(limited); path = np.array(path); elbow = np.array(elbow_rows)
    else:
        commands = np.empty((len(stamps), len(JOINTS))); position = np.empty((len(stamps), 2)); rotation = np.empty((len(stamps), 2))
        elbow = np.zeros((len(stamps), 2)); limited = np.zeros(len(stamps), dtype=bool)
    for i in range(len(stamps) if governor is None else 0):
        targets_elbow = None
        if prior is not None:
            ahead = min(len(stamps) - 1, i + lookahead)
            targets_elbow = prior.targets(q, float(stamps[i]), left[ahead], right[ahead])
        posture = posture_prior.reference(model, q, left[i], right[i], 1. / RATE_HZ) if posture_prior is not None else None
        q, info = ik.step(q, left[i], right[i], waist_yaw_reference=-yaw[i], elbow_targets=targets_elbow, posture=posture,
                          orientation=float(orientation[i]))
        commands[i] = q; position[i] = info['position_error']; rotation[i] = info['rotation_error']
        elbow[i] = info['elbow_error']; limited[i] = info['saturated']
    report, elbows, swivel = arm_posture(model, commands[::3], ik.lower, ik.upper)
    report.update(prior=str(prior_path) if prior_path else None,
                  hand_position_error_max_m=float(position.max()), hand_position_error_p99_m=float(np.percentile(position, 99)),
                  hand_rotation_error_max_deg=float(np.degrees(rotation.max())),
                  elbow_target_error_mean_m=float(elbow.mean()) if prior is not None else None,
                  limited_fraction=float(limited.mean()),
                  joint_speed_max_rad_s=float(np.abs(np.diff(commands[:, ACTIVE], axis=0)).max() * RATE_HZ),
                  waist_pitch_range_rad=[float(commands[:, 2].min()), float(commands[:, 2].max())],
                  waist_roll_abs_max_rad=float(np.abs(commands[:, 1]).max()))
    return commands, report, dict(elbows=elbows, swivel=swivel, path=path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', type=Path, required=True, help='targets.npz written by prepare')
    parser.add_argument('--prior', type=Path, help='Exported FiSTA weights (model/fista.npz); omit for no elbow prior')
    parser.add_argument('--base-offset-z', type=float, default=.078, help='Vertical target shift the worker applies [m]')
    parser.add_argument('--out', type=Path, help='Write commands and the report here (new .npz/.json pair)')
    args = parser.parse_args()
    commands, report, extra = solve(np.load(args.targets), args.prior, args.base_offset_z)
    print(json.dumps(report, indent=2))
    if args.out:
        if args.out.with_suffix('.npz').exists():
            raise FileExistsError(args.out.with_suffix('.npz'))
        np.savez_compressed(args.out.with_suffix('.npz'), q=commands, **extra)
        args.out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
