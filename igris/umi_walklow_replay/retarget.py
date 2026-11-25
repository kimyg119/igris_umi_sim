"""Map recorded UMI palm poses to IGRIS-C end-effector targets.

The recorded frame has robot EE axes (thumb +x, fingers -z, palm normal -y on
the left hand and +y on the right) but its origin sits on the operator's palm,
not on the robot EE origin, and the palm normal carries a constant per-hand
recording error.  Both are corrected with constants, never per frame.
Only numpy is used.
"""
import json
import math
from pathlib import Path

import numpy as np

from .kinematics import axis_rotation, quat_matrix, rotation_vector

# Box long axis (right palm -> left palm) seen in each recorded hand frame,
# averaged over every bimanual carry of all 34 episodes.  A palm lying flat on
# the box end face would give [0, 1, 0]; the left hand is 12.3 deg and the
# right hand 34.3 deg away, with a sample spread of about 5 deg.
MEASURED_BOX_AXIS = {'left': [-0.0761, 0.9769, 0.1997], 'right': [-0.4326, 0.8263, -0.3607]}

PELVIS_TO_SURFACE = .10      # [m] recorded work surfaces sit this far below the pelvis origin (target heights -> room heights)

DEFAULT = {
    # Robot palm contact centre in its EE frame [m].  The palm pad surface is at
    # |y| = 0.030; z lies between the MCP axis (0) and the pad centre (0.068).
    'contact': {'left': [0., -0.030, 0.035], 'right': [0., 0.030, 0.035]},
    'squeeze_m': 0.022,                 # while holding, each palm is commanded this far into the box face
    # In free motion each palm stays this much further out than recorded.  The
    # operator opened only 5 cm per side; the robot's hands arrive up to 8 cm off
    # sideways, and a hand that lands on top of the box cannot close.
    'clearance_m': 0.03,
    'palm_normal_correction': True,
    # While both palms hold the box (tactile contact), command them as one rigid
    # pair: parallel, facing each other, box_length apart.  The pair's position
    # and attitude still come from the recording.
    'object_centric': True,
    'box_length_m': 0.27,
    # The operator held each box at a different height (10.7, 4.2 and 7.4 cm above
    # its bottom in episode 0).  A rigid 8 cm pad placed that low also clamps the
    # box underneath, so each carry is shifted vertically to the box mid-height.
    # The shift comes from the recording and the box height only.
    'centre_grip_height': True,
    'box_height_m': 0.15,
    'grasp_blend_in_s': 0.6,
    # The attitude of the pair is taken up earlier than its position.  Both hands then turn level together while
    # they come down to the box instead of in the last 0.6 s (20-45 deg there, most at the lowest box).
    'grasp_turn_in_s': 1.5,
    'grasp_blend_out_s': 0.4,
    'measured_box_axis': MEASURED_BOX_AXIS,
    # Rigid shift of every target (and of the scene) in the pelvis frame [m].
    'base_offset': [0., 0., 0.],
    # Turn each hand about its palm contact point so the fingers converge by this angle [deg].
    # The teleoperated robot holds boxes that way (fingers 17-26 deg inward); with the palms
    # exactly parallel and narrower than the shoulders the forearms must run parallel too,
    # which folds the elbows in and puts the shoulder roll on its inward limit.
    'toe_in_deg': 0.,
    # Empty hands do not need the operator's hand orientation.  Followed exactly, it winds the wrist
    # yaw to its limit and swings the elbow out (left swivel 58 deg in free motion against 8 deg while
    # holding; the teleoperated robot stays within 4-39 deg).  The orientation counts fully from
    # ``orientation_lead_s`` before the approach to a box until ``orientation_lag_s`` after the release.
    'orientation_lead_s': 1.0,
    'orientation_lag_s': 1.0,
    # Toe-in chosen by the IK instead: [lowest, highest] angle [deg], or None.  The IK holds the palm
    # point and the two other turns and spends the freedom on keeping arm joints off their limits;
    # inside the range the angle is drawn weakly to the preferred value.
    'toe_free_deg': None,
    'toe_preference_deg': 12.,
    # Contact surface of the hands: 'flat' (slab, turned to suit the fixed or preferred toe-in) or
    # 'capsule' (rounded, the same contact at any toe-in).
    'pad': 'flat',
    # Air kept between a hand's contact pad and the surfaces the boxes stand on (clearance.py); 0 switches it off.
    # The recorded palm comes down next to the lowest box closer than the robot's longer, rigid hand allows.
    'support_clearance_m': 0.02,
}


def palm_correction(side, config=DEFAULT):
    """Constant rotation C with R_corrected = R_recorded @ C."""
    if not config.get('palm_normal_correction', True):
        return np.eye(3)
    measured = np.asarray(config['measured_box_axis'][side], dtype=float)
    measured /= np.linalg.norm(measured)
    axis = np.cross([0., 1., 0.], measured)
    sine = np.linalg.norm(axis)
    if sine < 1e-12:
        return np.eye(3)
    return axis_rotation(axis / sine, np.arctan2(sine, measured[1]))


def pose_matrices(pose):
    """N x 4 x 4 transforms from xyz + xyzw quaternion rows."""
    pose = np.atleast_2d(np.asarray(pose, dtype=float))
    out = np.tile(np.eye(4), (len(pose), 1, 1))
    out[:, :3, 3] = pose[:, :3]
    for i, row in enumerate(pose):
        out[i, :3, :3] = quat_matrix(row[[6, 3, 4, 5]])
    return out


def ee_targets(pose, side, config=DEFAULT):
    """Free-motion robot EE targets (N x 4 x 4, pelvis frame) for recorded palm poses."""
    if side not in ('left', 'right'):
        raise ValueError(side)
    return _to_ee(_palms(pose, side, config), side, config)


def _palms(pose, side, config):
    """Recorded palm frames with the constant palm-normal correction applied."""
    palms = pose_matrices(pose)
    palms[:, :3, :3] = palms[:, :3, :3] @ palm_correction(side, config)
    return palms


def _to_ee(palms, side, config, weight=None):
    """EE frames for palm frames; ``weight`` is 1 where the box is held, 0 in free motion."""
    inward = np.array([0., -1., 0.]) if side == 'left' else np.array([0., 1., 0.])
    weight = np.zeros(len(palms)) if weight is None else np.asarray(weight, dtype=float)
    travel = weight * float(config['squeeze_m']) - (1 - weight) * float(config.get('clearance_m', 0.))
    local = -np.asarray(config['contact'][side], dtype=float)[None] + travel[:, None] * inward[None]
    out = palms.copy()
    out[:, :3, 3] += np.einsum('nij,nj->ni', out[:, :3, :3], local) + np.asarray(config.get('base_offset', [0, 0, 0]), dtype=float)
    return out


def toe_in_transform(side, degrees, contact):
    """EE-frame transform that turns a hand by ``degrees`` about its x axis through the contact point.

    With the fingers along -z and the palm normal along -/+y (left/right), a positive angle
    brings the fingertips of both hands toward each other.  The contact point does not move.
    """
    angle = math.radians(-degrees if side == 'left' else degrees)
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.eye(4); rotation[1:3, 1:3] = [[c, -s], [s, c]]
    there = np.eye(4); there[:3, 3] = contact
    back = np.eye(4); back[:3, 3] = -np.asarray(contact, dtype=float)
    return there @ rotation @ back


def toe_in(transforms, side, config=DEFAULT):
    """EE targets with the configured toe-in applied (a copy when the angle is zero)."""
    degrees = float(config.get('toe_in_deg', 0.))
    if not degrees:
        return transforms.copy()
    return transforms @ toe_in_transform(side, degrees, config['contact'][side])


def grasp_weight(t, segments, config):
    """0 in free motion, 1 while the box is held, smooth in between."""
    t = np.asarray(t, dtype=float)
    weight = np.zeros(len(t))
    before, after = float(config['grasp_blend_in_s']), float(config['grasp_blend_out_s'])
    for segment in segments:
        start, end = t[segment['pick_index']], t[segment['place_index']]
        rise = np.clip((t - (start - before)) / before, 0., 1.)
        fall = np.clip(((end + after) - t) / after, 0., 1.)
        ramp = np.minimum(rise, fall)
        weight = np.maximum(weight, ramp * ramp * (3 - 2 * ramp))
    return weight


def orientation_weight(t, segments, config):
    """1 where the hand orientation matters (approach, hold, release), 0 in the middle of free motion."""
    t = np.asarray(t, dtype=float)
    weight = np.zeros(len(t))
    lead, lag = float(config.get('orientation_lead_s', 1.)), float(config.get('orientation_lag_s', 1.))
    for segment in segments:
        start = t[segment['pick_index']] - float(config['grasp_blend_in_s'])      # the approach begins here
        end = t[segment['place_index']] + float(config['grasp_blend_out_s'])
        ramp = np.minimum(np.clip((t - (start - lead)) / lead, 0., 1.), np.clip(((end + lag) - t) / lag, 0., 1.))
        weight = np.maximum(weight, ramp * ramp * (3 - 2 * ramp))
    return weight


def _blend(a, b, weight, turn=None):
    """Interpolate a single 4x4 pose from ``a`` to ``b``; ``turn`` is the share of the rotation if it differs."""
    turn = weight if turn is None else turn
    out = np.eye(4)
    out[:3, 3] = (1 - weight) * a[:3, 3] + weight * b[:3, 3]
    delta = rotation_vector(b[:3, :3] @ a[:3, :3].T)
    angle = np.linalg.norm(delta)
    out[:3, :3] = a[:3, :3] if angle < 1e-12 else axis_rotation(delta / angle, turn * angle) @ a[:3, :3]
    return out


def _pair_rotation(left, right):
    """Rotation of the recorded hand pair: EE axes thumb, palm axis, -fingers."""
    axis = left[:3, 3] - right[:3, 3]
    axis /= np.linalg.norm(axis)                       # right palm -> left palm
    up = left[:3, 0] + right[:3, 0]                    # thumbs: shared roll about the axis
    up -= axis * np.dot(up, axis)
    up /= np.linalg.norm(up)
    return np.column_stack((up, axis, np.cross(up, axis)))


def _level(rotation):
    """Same heading, but with the palm axis horizontal and the thumbs straight up."""
    heading = np.arctan2(rotation[1, 1], rotation[0, 1])
    axis = np.array([np.cos(heading), np.sin(heading), 0.])
    up = np.array([0., 0., 1.])
    return np.column_stack((up, axis, np.cross(up, axis)))


def _mean_rotation(rotations):
    u, _, vt = np.linalg.svd(np.mean(rotations, axis=0))
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    return u @ vt


def _slerp(a, b, fraction):
    delta = rotation_vector(b @ a.T)
    angle = np.linalg.norm(delta)
    return a if angle < 1e-12 else axis_rotation(delta / angle, fraction * angle) @ a


def grip_height_shifts(segments, box_height):
    """Vertical shift per carry that puts the palms at the box mid-height.

    Boxes leave a stack top first, so carry k takes level n-1-k.  With a common
    grip height the pick heights minus their level heights would be equal; the
    shift removes each carry's deviation from their mean.
    """
    count = len(segments)
    if count < 2:
        return np.zeros(count)
    reduced = np.array([segment['pick_mid'][2] - (count - 1 - k) * box_height for k, segment in enumerate(segments)])
    # A box is 15 cm tall: a larger correction means the pick was misdetected, not held oddly.
    return np.clip(np.median(reduced) - reduced, -box_height / 2, box_height / 2)


def bimanual_targets(episode, segments, config=DEFAULT):
    """Left and right EE targets (N x 4 x 4) at the recorded frame rate.

    While the box is held the two palms are commanded as one rigid pair: parallel,
    facing each other, ``box_length_m`` apart, centred on the recorded palm
    midpoint.  The pair follows the recorded attitude changes, corrected so that
    it is level at the pick and at the set-down, where the box rests on a
    horizontal surface.
    """
    left, right = _palms(episode['left'], 'left', config), _palms(episode['right'], 'right', config)
    weight = np.zeros(len(left))
    if config.get('object_centric', True) and segments:
        weight = grasp_weight(episode['t'], segments, config)
        turn_in = max(float(config.get('grasp_turn_in_s', 0.)), float(config['grasp_blend_in_s']))
        turn = grasp_weight(episode['t'], segments, {**config, 'grasp_blend_in_s': turn_in})
        half = float(config['box_length_m']) / 2
        count = len(weight)
        fps = 1. / float(np.median(np.diff(episode['t'])))
        window = max(1, int(round(fps / 3)))
        shifts = (grip_height_shifts(segments, float(config['box_height_m']))
                  if config.get('centre_grip_height', True) else np.zeros(len(segments)))
        for segment, shift in zip(segments, shifts):
            first, last = segment['pick_index'], segment['place_index']
            lead = int(np.ceil(turn_in * fps)) + 1
            tail = int(np.ceil(float(config['grasp_blend_out_s']) * fps)) + 1
            recorded = {i: _pair_rotation(left[i], right[i]) for i in range(max(0, first - lead), min(count, last + tail + 1))}
            at_pick = _mean_rotation([recorded[i] for i in range(first, min(first + window, last) + 1)])
            at_place = _mean_rotation([recorded[i] for i in range(max(last - window, first), last + 1)])
            fix_pick = _level(at_pick) @ at_pick.T
            fix_place = _level(at_place) @ at_place.T
            for i, rotation in recorded.items():
                if turn[i] <= 0:
                    continue
                fraction = float(np.clip((i - first) / max(last - first, 1), 0., 1.))
                pair = _slerp(fix_pick, fix_place, fraction) @ rotation
                middle = (left[i, :3, 3] + right[i, :3, 3]) / 2 + np.array([0., 0., shift])
                for palms, sign in ((left, 1.), (right, -1.)):
                    held = np.eye(4); held[:3, :3] = pair; held[:3, 3] = middle + sign * half * pair[:, 1]
                    palms[i] = _blend(palms[i], held, weight[i], turn[i])
    return _to_ee(left, 'left', config, weight), _to_ee(right, 'right', config, weight)


def contact_points(targets, side, config=DEFAULT):
    """Where the robot palm contact centre ends up for given EE targets."""
    return targets[:, :3, 3] + targets[:, :3, :3] @ np.asarray(config['contact'][side], dtype=float)


def resample(t, transforms, rate_hz, speed=1.0):
    """Uniform-rate targets: Catmull-Rom positions, shortest-arc rotations."""
    t = np.asarray(t, dtype=float) / float(speed)
    if len(t) != len(transforms) or len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError('Need increasing timestamps matching the transforms')
    out_t = np.arange(0., t[-1] + 1e-9, 1. / rate_hz)
    index = np.clip(np.searchsorted(t, out_t, side='right') - 1, 0, len(t) - 2)
    u = (out_t - t[index]) / (t[index + 1] - t[index])
    p = transforms[:, :3, 3]
    before = p[np.maximum(index - 1, 0)]; a = p[index]; b = p[index + 1]; after = p[np.minimum(index + 2, len(t) - 1)]
    u1 = u[:, None]
    position = .5 * ((2 * a) + (-before + b) * u1 + (2 * before - 5 * a + 4 * b - after) * u1 ** 2
                     + (-before + 3 * a - 3 * b + after) * u1 ** 3)
    out = np.tile(np.eye(4), (len(out_t), 1, 1))
    out[:, :3, 3] = position
    for k, (i, fraction) in enumerate(zip(index, u)):
        start = transforms[i, :3, :3]
        delta = rotation_vector(transforms[i + 1, :3, :3] @ start.T)
        angle = np.linalg.norm(delta)
        out[k, :3, :3] = start if angle < 1e-12 else axis_rotation(delta / angle, fraction * angle) @ start
    return out_t, out


def save_config(path, config):
    Path(path).write_text(json.dumps(config, indent=2) + '\n')
