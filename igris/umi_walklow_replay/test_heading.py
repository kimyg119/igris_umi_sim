import math

import numpy as np

from igris.umi_walklow_replay.heading import HeadingTarget, read_status, torso_heading
from igris.umi_walklow_replay.kinematics import JOINTS, UpperBody

MODEL = UpperBody.from_mjcf()
WAIST_YAW, WAIST_ROLL, WAIST_PITCH = (JOINTS.index(name) for name in ('waist_yaw', 'waist_roll', 'waist_pitch'))


def hand_line_heading(q):
    """Direction of the line from the right hand to the left hand, seen from above (pelvis frame)."""
    left, right = MODEL.fk(q)
    line = left[:3, 3] - right[:3, 3]
    return math.atan2(line[1], line[0])


def test_torso_faces_where_the_pelvis_faces_with_a_straight_waist():
    assert math.isclose(torso_heading(MODEL, np.zeros(31), .4), .4, abs_tol=1e-12)


def test_waist_yaw_turns_the_torso_heading_as_it_turns_the_hands():
    q = np.zeros(31); q[WAIST_YAW] = .5
    turned = hand_line_heading(q) - hand_line_heading(np.zeros(31))
    assert math.isclose(abs(turned), .5, abs_tol=1e-9)          # the arms hang from the torso
    assert math.isclose(torso_heading(MODEL, q, 0.), turned, abs_tol=1e-9)


def test_pelvis_heading_adds_to_the_waist_turn():
    q = np.zeros(31); q[WAIST_YAW] = -.7; q[WAIST_PITCH] = -.1; q[WAIST_ROLL] = .05
    assert math.isclose(torso_heading(MODEL, q, .3) - torso_heading(MODEL, q, 0.), .3, abs_tol=1e-12)


def test_heading_stays_between_minus_pi_and_pi():
    q = np.zeros(31); q[WAIST_YAW] = -1.2
    for pelvis in (3., -3., 2.5):
        assert -math.pi <= torso_heading(MODEL, q, pelvis) <= math.pi


def test_each_target_is_one_record_that_adds_up(tmp_path):
    path = tmp_path / 'heading_target.bin'
    target = HeadingTarget(path)
    target.send(True, .25)
    first = np.fromfile(path)
    target.send(True, -.5)
    second = np.fromfile(path)
    target.send(False, 0.)
    third = np.fromfile(path)
    target.close()
    for record, enable, heading in ((first, 1., .25), (second, 1., -.5), (third, 0., 0.)):
        assert record.shape == (4,) and record[1] == enable and record[2] == heading
        assert record[3] == record[0] + record[1] + record[2]   # the library drops a record that does not add up
    assert first[0] < second[0] < third[0]                       # and one it has already seen


def test_status_of_the_library_is_read_back(tmp_path):
    path = tmp_path / 'heading_status.bin'
    assert read_status(path) is None
    np.array([312., 8000., 1., -.21]).tofile(path)
    assert read_status(path) == dict(setter_calls=312, last_sequence=8000, enabled=True, target_yaw_rad=-.21)
