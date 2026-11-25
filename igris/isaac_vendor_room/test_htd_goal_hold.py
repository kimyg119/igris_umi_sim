"""Real CPU WorldXY feedback tests; no native or Isaac simulation."""
import copy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def helper():
    path = Path(__file__).with_name('htd_goal_hold.py')
    assert path.exists(), 'GoalHold helper is not implemented'
    spec = importlib.util.spec_from_file_location('htd_goal_hold', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GoalHold()


def pose(x=0., y=0., yaw=0., z=.94):
    return np.array([x, y, z, math.cos(yaw/2), 0., 0., math.sin(yaw/2)])


def sample(t=10., xy=(2., 3.), yaw=0., stamp=None):
    return SimpleNamespace(native_time=t, pose_native_time=t if stamp is None else stamp,
                           torso_pose=pose(*xy, yaw=-.7),
                           pelvis_pose=pose(9., -8., yaw=yaw, z=.87))


def test_capture_uses_torso_position_and_pelvis_heading_without_mutating_input():
    h = helper(); s = sample(yaw=math.pi/2); before = copy.deepcopy(s)
    h.capture(s)
    np.testing.assert_array_equal(h.metrics['goal_xy_world'], [2., 3.])
    assert h.metrics['heading_goal_rad'] == pytest.approx(math.pi/2)
    assert h.metrics['velocity_xy_world'] == [0., 0.]
    assert h.metrics['yaw_rate_rad_s'] == 0.
    assert h.metrics['velocity_estimate'] == 'zero_at_capture'
    np.testing.assert_array_equal(s.torso_pose, before.torso_pose)
    np.testing.assert_array_equal(s.pelvis_pose, before.pelvis_pose)
    assert not h.success


def test_world_drift_commands_return_direction_in_pelvis_frame():
    h = helper(); h.capture(sample(yaw=math.pi/2))
    cmd = h.step(sample(10.02, xy=(2.08, 3.), yaw=math.pi/2))
    assert abs(cmd[0]) < 1e-7 and cmd[1] > 0.
    assert np.linalg.norm(cmd[:2]) <= .003001
    assert h.metrics['position_error_world'] == pytest.approx([-.08, 0.])
    assert h.metrics['goal_xy_world'] == [2., 3.]


def test_heading_feedback_opposes_drift_even_inside_position_arrival():
    h = helper(); h.capture(sample())
    cmd = h.step(sample(10.02, yaw=.2))
    np.testing.assert_array_equal(cmd[:2], [0., 0.])
    assert cmd[2] == pytest.approx(-.01)
    assert h.metrics['heading_error_rad'] == pytest.approx(-.2)
    assert not h.success


def test_velocity_uses_pose_time_and_repeated_stamp_keeps_last_derivative():
    h = helper(); h.capture(sample())
    h.step(sample(10.04, xy=(2.002, 2.996), yaw=.01, stamp=10.02))
    assert h.metrics['velocity_xy_world'] == pytest.approx([.1, -.2])
    assert h.metrics['yaw_rate_rad_s'] == pytest.approx(.5)
    h.step(sample(10.06, xy=(2.002, 2.996), yaw=.01, stamp=10.02))
    assert h.metrics['velocity_xy_world'] == pytest.approx([.1, -.2])
    assert h.metrics['yaw_rate_rad_s'] == pytest.approx(.5)
    assert h.metrics['velocity_estimate'] == 'cached_repeated_pose_stamp'
    assert not h.success


def test_wrapped_yaw_derivative_crosses_pi_without_false_full_turn():
    h = helper(); h.capture(sample(yaw=math.pi-.01))
    h.step(sample(10.02, yaw=-math.pi+.01))
    assert h.metrics['yaw_rate_rad_s'] == pytest.approx(1.)
    assert h.metrics['heading_error_rad'] == pytest.approx(-.02)


def test_real_controller_speed_ellipse_and_acceleration_hold_over_large_drift():
    h = helper(); h.capture(sample())
    previous = np.zeros(3)
    for i in range(1, 130):
        cmd = h.step(sample(10.+i*.02, xy=(12., -7.), yaw=.8))
        assert np.linalg.norm(cmd[:2]/[.1, .07]) <= 1.00001
        assert abs(cmd[2]) <= .250001
        assert np.linalg.norm(cmd[:2]-previous[:2]) <= .003001
        assert abs(cmd[2]-previous[2]) <= .010001
        previous = cmd.copy()
    assert np.linalg.norm(previous[:2]) > .07
    assert previous[2] == pytest.approx(-.25)


def test_settled_success_requires_elapsed_updates_and_clears_on_motion():
    h = helper(); h.capture(sample())
    for i in range(1, 50):
        np.testing.assert_array_equal(h.step(sample(10+i*.02)), [0., 0., 0.])
    assert not h.success
    for i in (50, 51): h.step(sample(10+i*.02))
    assert h.success and h.metrics['success']
    h.step(sample(11.04, xy=(2.005, 3.)))
    assert not h.success  # 5mm/20ms movement exceeds settled speed.
    assert h.metrics['settled_s'] == 0.


def test_capture_is_once_and_step_requires_capture():
    h = helper()
    with pytest.raises(RuntimeError, match='capture'): h.step(sample())
    h.capture(sample())
    with pytest.raises(RuntimeError, match='capture'): h.capture(sample(10.1))
    assert h.metrics['goal_xy_world'] == [2., 3.]


@pytest.mark.parametrize('which', ['torso_pose', 'pelvis_pose'])
@pytest.mark.parametrize('invalid', ['nan', 'short', 'quaternion'])
def test_invalid_pose_rejected_before_capture(which, invalid):
    h = helper(); s = sample(); value = getattr(s, which).copy()
    if invalid == 'nan': value[0] = np.nan
    elif invalid == 'short': value = value[:6]
    else: value[3:] *= 2.
    setattr(s, which, value)
    with pytest.raises(ValueError): h.capture(s)
    h.capture(sample())  # Rejected sample did not consume capture.


@pytest.mark.parametrize('native,stamp', [(10., 10.001), (10., 9.949), (float('nan'), 10.),
                                          (10., float('inf')), (-1., -1.)])
def test_future_stale_or_nonfinite_pose_time_rejected(native, stamp):
    h = helper()
    with pytest.raises(ValueError): h.capture(sample(native, stamp=stamp))


def test_backward_clocks_and_inconsistent_repeated_pose_rejected_without_rebase():
    h = helper(); h.capture(sample()); h.step(sample(10.02))
    for s in [sample(10.01), sample(10.04, stamp=10.01), sample(10.04, xy=(2.1, 3.), stamp=10.02)]:
        with pytest.raises((ValueError, RuntimeError)): h.step(s)
    h.step(sample(10.04))
    assert h.metrics['velocity_xy_world'] == [0., 0.]
    assert h.metrics['goal_xy_world'] == [2., 3.]


@pytest.mark.parametrize('dt', [0., -.02, float('nan'), .04])
def test_update_requires_declared_actor_period(dt):
    h = helper(); h.capture(sample())
    with pytest.raises(ValueError): h.step(sample(10.02), dt=dt)


def test_metrics_and_return_are_detached_from_controller_and_input():
    h = helper(); s = sample(); h.capture(s)
    s.torso_pose[:2] = 999.
    cmd = h.step(sample(10.02, xy=(2.08, 3.)))
    cmd[:] = 999.
    assert h.metrics['goal_xy_world'] == [2., 3.]
    assert max(abs(x) for x in h.metrics['command_xyz']) <= .1
    h.step(sample(10.04, xy=(2.08, 3.)))
    assert max(abs(x) for x in h.metrics['command_xyz']) <= .1
    assert h.events[0]['event'] == 'captured_goal'
    assert h.events[-1]['event'] == 'goal_hold_update'
