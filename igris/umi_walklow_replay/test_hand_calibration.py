"""Calibration of the recorded hand frames from held boxes."""
import numpy as np
import pytest

from igris.umi_walklow_replay import hand_calibration
from igris.umi_walklow_replay.kinematics import axis_rotation, matrix_quat, quat_matrix, rotation_vector


def held_boxes(left_error, right_error, carries=6, frames=40, seed=0):
    """True palm poses of both hands on level boxes, recorded through constant per-hand errors."""
    rng = np.random.default_rng(seed)
    left, right, segments = [], [], []
    for carry in range(carries):
        heading = rng.uniform(-1., 1.); pitch = rng.uniform(-.9, .3)             # the hands pitch with the box height
        middle = np.array([.4, 0., .1 * carry])
        for frame in range(frames):
            held = 5 <= frame < frames - 5
            wobble = 0. if frame < 12 or frame >= frames - 12 else .2 * np.sin(frame)   # the box tilts while carried
            pair = axis_rotation([0, 0, 1], heading) @ axis_rotation([1, 0, 0], wobble) @ axis_rotation([0, 1, 0], pitch)
            for side, sign, store, error in (('left', 1., left, left_error), ('right', -1., right, right_error)):
                true = np.eye(4); true[:3, :3] = pair
                true[:3, 3] = middle + sign * (.135 if held else .2) * pair[:, 1]
                store.append(true @ np.linalg.inv(error))                         # recorded = true @ error^-1
        segments.append(dict(pick_index=carry * frames + 5, place_index=carry * frames + frames - 6))
    return np.array(left), np.array(right), segments


def error(normal_tilt, roll, offset):
    out = np.eye(4)
    out[:3, :3] = hand_calibration.correction(dict(normal=normal_tilt, roll_deg=roll)); out[:3, 3] = out[:3, :3] @ offset
    return out


def test_measure_recovers_the_palm_axes_the_left_turn_and_the_origin_error():
    left_error = error([-.05, .93, .36], -25., np.array([.03, 0., .01]))
    right_error = error([-.2, .92, -.33], 0., np.array([-.03, 0., -.01]))
    left, right, segments = held_boxes(left_error, right_error)
    constants, report = hand_calibration.measure([(left, right, segments)])
    for side, wanted in (('left', [-.05, .93, .36]), ('right', [-.2, .92, -.33])):
        wanted = np.array(wanted) / np.linalg.norm(wanted)
        assert np.degrees(np.arccos(np.clip(np.dot(constants[side]['normal'], wanted), -1, 1))) < .5
    assert constants['left']['roll_deg'] == pytest.approx(-25., abs=.5) and constants['right']['roll_deg'] == 0.
    assert np.allclose(constants['left']['offset_m'], [.03, 0., .01], atol=2e-3)
    assert np.allclose(constants['right']['offset_m'], [-.03, 0., -.01], atol=2e-3)
    assert abs(report['left_minus_right_height_at_rest_m']['corrected'][0]) < 2e-3


def test_apply_makes_the_two_hands_agree_on_a_held_box():
    left_error = error([-.05, .93, .36], -25., np.array([.03, 0., .01]))
    right_error = error([-.2, .92, -.33], 0., np.array([-.03, 0., -.01]))
    left, right, segments = held_boxes(left_error, right_error)
    constants, _ = hand_calibration.measure([(left, right, segments)])
    rows = lambda poses: np.array([[*p[:3, 3], *matrix_quat(p[:3, :3])[[1, 2, 3, 0]]] for p in poses])
    episode = hand_calibration.apply(dict(left=rows(left), right=rows(right), t=np.arange(len(left)) / 15.), constants)
    held = segments[2]['pick_index'] + 10
    rotation = lambda row: quat_matrix(row[[6, 3, 4, 5]])
    assert np.linalg.norm(rotation_vector(rotation(episode['left'][held]).T @ rotation(episode['right'][held]))) < np.radians(1.)
    between = episode['left'][held, :3] - episode['right'][held, :3]
    assert np.linalg.norm(between) == pytest.approx(.27, abs=3e-3)
    assert np.allclose(between / np.linalg.norm(between), rotation(episode['right'][held])[:, 1], atol=.02)   # along the palm axis
    assert episode['t'] is not None and np.allclose(np.linalg.norm(episode['left'][:, 3:], axis=1), 1.)


def test_too_little_data_is_refused():
    left, right, _ = held_boxes(np.eye(4), np.eye(4), carries=1, frames=12)
    with pytest.raises(ValueError):
        hand_calibration.measure([(left, right, [])])
