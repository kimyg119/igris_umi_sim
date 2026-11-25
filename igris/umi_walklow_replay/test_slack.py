import numpy as np
import pytest

from igris.umi_walklow_replay.slack import SlackCompensation, read_slack

DT = .01
LOWER, UPPER = np.full(31, -1.), np.full(31, 1.)


def record(path, rows):
    """Observer file: rows of the wall time followed by the 31 slack angles."""
    np.array(rows, dtype=np.float64).tofile(path)


def row(stamp, value):
    return [stamp, *([value] * 31)]


def settled(slack, seconds=3.):
    compensation = SlackCompensation(LOWER, UPPER)
    for _ in range(int(seconds / DT)):
        compensation.update(slack, DT)
    return compensation


def test_reader_returns_the_last_complete_row(tmp_path):
    path = tmp_path / 'joint_offsets.bin'
    record(path, [row(100., .01), row(100.03, .02)])
    with path.open('ab') as stream:
        stream.write(b'\0' * 100)                       # the observer is in the middle of the next row
    stamp, slack = read_slack(path, 100.05)
    assert stamp == 100.03 and slack.shape == (31,) and np.allclose(slack, .02)


def test_reader_gives_nothing_for_a_stale_missing_or_broken_record(tmp_path):
    path = tmp_path / 'joint_offsets.bin'
    assert read_slack(path, 100.) is None
    record(path, [row(100., .01)])
    assert read_slack(path, 100.2) is None              # older than 0.15 s
    assert read_slack(path, 99.9) is None               # stamped in the future
    record(path, [row(100., np.nan)])
    assert read_slack(path, 100.05) is None


def test_link_reaches_the_wanted_angle_once_the_slack_is_steady():
    slack = np.zeros(31); slack[15] = .02; slack[18] = -.015      # the link is at motor + slack
    wanted = np.full(31, .3)
    motor = settled(slack).motor(wanted)
    assert np.allclose(motor + slack, wanted, atol=1e-5)


def test_correction_comes_in_gradually():
    slack = np.full(31, .02)
    compensation = SlackCompensation(LOWER, UPPER)
    compensation.update(slack, DT)
    first = np.zeros(31) - compensation.motor(np.zeros(31))
    assert 0 < first.max() < .002                       # a tenth of the slack at most after one tick
    for _ in range(50):
        compensation.update(slack, DT)
    assert .002 < (np.zeros(31) - compensation.motor(np.zeros(31))).max() < .02


def test_a_flipping_slack_does_not_shake_the_motor_target():
    """Near balance the slack joint rattles between its two sides; the motor must not follow each flip."""
    compensation = settled(np.full(31, .01))
    before = compensation.motor(np.zeros(31))
    compensation.update(np.full(31, -.01), DT)
    assert np.abs(compensation.motor(np.zeros(31)) - before).max() < .002


def test_motor_target_stays_inside_the_joint_limits():
    compensation = settled(np.full(31, -.02))
    motor = compensation.motor(np.full(31, 1.))         # link at its upper limit, motor would have to pass it
    assert np.all(motor <= UPPER) and np.allclose(motor, 1.)


def test_last_slack_is_kept_while_the_record_is_briefly_stale():
    compensation = settled(np.full(31, .02))
    before = compensation.motor(np.zeros(31))
    for _ in range(20):
        compensation.update(None, DT)                   # 0.2 s without a fresh record
    assert np.allclose(compensation.motor(np.zeros(31)), before)


def test_a_record_that_stays_stale_stops_the_replay():
    compensation = settled(np.full(31, .02))
    with pytest.raises(RuntimeError, match='slack'):
        for _ in range(100):
            compensation.update(None, DT)


def test_only_the_owned_joints_are_corrected():
    compensation = SlackCompensation(LOWER, UPPER, joints=[0, 15])
    for _ in range(300):
        compensation.update(np.full(31, .02), DT)
    motor = compensation.motor(np.zeros(31))
    assert np.allclose(motor[[0, 15]], -.02, atol=1e-5) and np.allclose(np.delete(motor, [0, 15]), 0.)


def test_calibrated_slack_is_the_gravity_side_of_the_weight_bearing_joints():
    """Without the simulator's measurement: constants for the joints that sit on one side all the time."""
    from igris.umi_walklow_replay.kinematics import JOINTS
    from igris.umi_walklow_replay.slack import calibrated
    constant = calibrated()
    named = {JOINTS[index]: value for index, value in enumerate(constant) if value}
    assert sorted(named) == ['l_elbow_pitch', 'l_shoulder_pitch', 'r_elbow_pitch', 'r_shoulder_pitch']
    assert all(np.radians(.9) < value < np.radians(1.3) for value in named.values())
    compensation = SlackCompensation(LOWER, UPPER)
    for _ in range(300):
        compensation.update(calibrated(), DT)                  # every tick the same: no file is read
    wanted = np.full(31, .2)
    assert np.allclose(compensation.motor(wanted), wanted - constant, atol=1e-4)
