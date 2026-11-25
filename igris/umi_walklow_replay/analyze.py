"""Judge one replay: command tracking, balance, and what happened to each box.

Every number comes from the native physics state and LowState recorded by the
replay worker.  Command tracking, IK residuals and task outcome are reported
separately; a finished replay is not a successful palletizing.
"""
import json
from pathlib import Path

import numpy as np

from .kinematics import ACTIVE, JOINTS, UpperBody
from .posture import arm_posture

FPS = 15.
LIFT_M = .03            # a box counts as lifted once it rises this far above where it stood
PLACE_XY_M = .10        # allowed horizontal miss of the recorded set-down point
PLACE_Z_M = .03
PLACE_TILT_DEG = 15.


def yaw(quaternion):
    w, x, y, z = np.asarray(quaternion).T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def tilt_deg(quaternion):
    w, x, y, z = np.asarray(quaternion).T
    return np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))


def box_outcomes(sim, parcels, carries, expected, speed=1.):
    """Per carried box: lift, transport and placement, from recorded poses.

    ``parcels`` is T x 3 x 7 (bottom, middle, top box); boxes are taken top first.
    ``carries`` holds [pick_frame, place_frame] pairs at the recording rate and
    ``expected`` the world set-down centre of each carry.  All verdicts use the
    last recorded pose, so a box that is set down and later knocked over fails:
    ``stacked`` means upright at its stack level (the level is 15 cm tall, so
    the right height implies it rests on the box below); ``placed`` also wants
    it within PLACE_XY_M of the recorded set-down point.  ``set_down_ok`` is the
    ``placed`` test 1.5 s after the hands let go.
    """
    def judge(pose, goal):
        miss_xy = float(np.linalg.norm(pose[:2] - goal[:2])); miss_z = float(pose[2] - goal[2])
        lean = float(tilt_deg(pose[3:]))
        upright = bool(abs(miss_z) <= PLACE_Z_M and lean <= PLACE_TILT_DEG)
        return miss_xy, miss_z, lean, upright, bool(upright and miss_xy <= PLACE_XY_M)

    outcomes = []
    for order, ((first, last), goal) in enumerate(zip(carries, expected)):
        box = parcels[:, parcels.shape[1] - 1 - order]
        start, end = first / FPS / speed, last / FPS / speed
        before = np.searchsorted(sim, start); after = min(len(sim) - 1, np.searchsorted(sim, end))
        if before >= len(sim) - 1:
            outcomes.append(dict(box=order, reached=False))
            continue
        held = slice(before, after + 1)
        rise = float(box[held, 2].max() - box[before, 2])
        complete = after < len(sim) - 1
        released = judge(box[min(len(sim) - 1, np.searchsorted(sim, end + 1.5))], goal)
        miss_xy, miss_z, lean, upright, on_target = judge(box[-1], goal)
        outcomes.append(dict(box=order, reached=True, carry_complete_in_log=bool(complete), lifted=bool(rise >= LIFT_M),
                             max_rise_m=rise, max_tilt_while_held_deg=float(tilt_deg(box[held, 3:]).max()),
                             set_down_ok=bool(complete and released[4]),
                             final_xy_miss_m=miss_xy, final_z_miss_m=miss_z, final_tilt_deg=lean,
                             stacked=bool(complete and upright), placed=bool(complete and on_target)))
    return outcomes


def analyze(out):
    out = Path(out)
    run = out / 'run'
    log = np.load(run / 'replay_log.npz')
    result = json.loads((run / 'replay_result.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())['umi_replay']
    targets = np.load(out / 'targets.npz')
    phase = log['phase']; replay = phase == 'replay'
    if not replay.any():
        raise ValueError('No replay phase was recorded')
    sim = log['sim'][replay] - log['sim'][replay][0]; wall = log['wall'][replay]
    # The motors are judged against what they were sent: with slack compensation that is not the IK's angle.
    sent = log['q_motor'] if 'q_motor' in log.files else log['q_command']
    measured = log['q_measured'][replay][:, ACTIVE]
    error = measured - sent[replay][:, ACTIVE]
    pelvis = log['pelvis'][replay]
    layout = manifest['layout']
    # World set-down points: the scene was staged at the pelvis pose measured before replay.
    staged = np.array(result['pelvis_at_staging'])
    heading = float(yaw(staged[3:])); c, s = np.cos(heading), np.sin(heading)
    place = np.array(layout['place']['xy'])
    place_world = staged[:2] + np.array([c * place[0] - s * place[1], s * place[0] + c * place[1]])
    height = layout['parcel_size_m'][2]
    expected = [np.array([*place_world, layout['surface_top_m'] + (level + .5) * height]) for level in range(3)]
    after = np.concatenate([log['parcels'][replay], log['parcels'][phase == 'hold']])
    # Carries are found by recorded time: with the speed limit the playback runs slower than the clock.
    path = log['path'][replay] if 'path' in log.files else sim
    after_sim = np.concatenate([path, path[-1] + (log['sim'][phase == 'hold'] - log['sim'][replay][-1])])
    boxes = box_outcomes(after_sim, after, targets['carry_frames'].tolist(), expected, float(targets['speed']))
    model = UpperBody.from_mjcf()
    posture = arm_posture(model, log['q_command'][replay][::3], model.lower[ACTIVE] + .035, model.upper[ACTIVE] - .035)[0]
    report = dict(
        output=str(out), episode=manifest['episode'], hardware_executed=False,
        elbow_prior=result.get('elbow_prior'), arm_posture=posture,
        worker_completed=bool(result.get('pass')), worker_error=result.get('error'),
        timing=dict(replay_sim_s=float(sim[-1]), replay_wall_s=float(wall[-1] - wall[0]),
                    realtime_factor=float(sim[-1] / (wall[-1] - wall[0])), command_rate_hz=float(replay.sum() / (wall[-1] - wall[0])),
                    commands_sent=result.get('commands_sent')),
        joint_tracking=dict(rms_rad={JOINTS[j]: float(v) for j, v in zip(ACTIVE, np.sqrt((error ** 2).mean(0)))},
                            max_rad=float(abs(error).max()), worst_joint=JOINTS[ACTIVE[int(abs(error).max(0).argmax())]]),
        ik=dict(position_error_max_m=float(log['position_error'][replay].max()),
                rotation_error_max_deg=float(np.degrees(log['rotation_error'][replay].max())),
                position_error_p95_m=float(np.percentile(log['position_error'][replay], 95)),
                limited_fraction=float(log['saturated'][replay].mean())),
        balance=dict(pelvis_min_z_m=float(pelvis[:, 2].min()), pelvis_xy_range_m=np.ptp(pelvis[:, :2], axis=0).tolist(),
                     pelvis_yaw_range_deg=float(np.degrees(np.ptp(np.unwrap(yaw(pelvis[:, 3:]))))),
                     pelvis_tilt_max_deg=float(tilt_deg(pelvis[:, 3:]).max()), base_offset_z_m=result.get('base_offset_z_m')),
        support_motion_m=np.ptp(log['supports'][replay][:, :, :3], axis=0).max(axis=1).tolist(),
        boxes=boxes,
        boxes_lifted=int(sum(b.get('lifted', False) for b in boxes)),
        boxes_stacked=int(sum(b.get('stacked', False) for b in boxes)),
        boxes_placed=int(sum(b.get('placed', False) for b in boxes)),
        stack_success=bool(len(boxes) == len(targets['carry_frames']) and all(b.get('stacked', False) for b in boxes)),
        task_success=bool(len(boxes) == len(targets['carry_frames']) and all(b.get('placed', False) for b in boxes)),
    )
    (run / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
