import numpy as np
import pytest

from lowering_targets import WorldPositionLowering
from walking_transfer import WalkingTransfer, pose_matrix
from test_walking_transfer import fixture


def test_world_horizontal_alignment_preserves_height_before_descent():
    _, p, torso_pose, _, world = fixture()
    torso = pose_matrix(torso_pose)
    local = np.linalg.inv(torso)[None] @ world
    delta = np.array([-.022, -.041, -.117])
    path = WorldPositionLowering(local, torso, delta, np.zeros((2, 3)))
    moved = torso.copy()
    moved[:3, 3] += [.008, -.006, .003]
    aligned = moved[None] @ path.targets(moved, 0., 0., np.zeros((2, 3)), xy_fraction=1.)
    np.testing.assert_allclose(aligned[:, :3, 3], world[:, :3, 3] + delta * [1, 1, 0], atol=1e-12)
    lowered = moved[None] @ path.targets(moved, .5, 0., np.zeros((2, 3)), xy_fraction=1.)
    np.testing.assert_allclose(lowered[:, :3, 3], world[:, :3, 3] + delta * [1, 1, .5], atol=1e-12)


@pytest.mark.parametrize('bad', [-.01, 1.01, float('nan')])
def test_horizontal_fraction_is_bounded(bad):
    _, _, torso_pose, _, world = fixture()
    torso = pose_matrix(torso_pose)
    path = WorldPositionLowering(np.linalg.inv(torso)[None] @ world, torso, [0, 0, -.1], np.zeros((2, 3)))
    with pytest.raises(ValueError):
        path.targets(torso, 0., 0., np.zeros((2, 3)), xy_fraction=bad)


def arrive(dwell):
    _, p, t, b, h = fixture()
    ctl = WalkingTransfer([1.16, -.26, .758], -np.pi/4, .1, 4.,
        heading_preserving_route=True, torso_relative_lowering=True,
        world_position_lowering=True, xy_first_lowering=True, arrival_settle_seconds=dwell)
    ctl.start(0., p, t, b, h)
    original = [v.copy() for v in (p, t, b)]
    stamp = 0.
    for offset in ([0, .07], [.38, .085], [.9, 0]):
        for v, old in zip((p, t, b), original):
            v[:2] = old[:2] + offset
        for _ in range(250):
            stamp += .02
            state = ctl.update(stamp, p, t, b, [6, 0, 6, 0], 0.)
            if state['phase'] == 'lower':
                return ctl, state, p, t, b, stamp
    raise AssertionError('Did not arrive')


def test_shorter_arrival_dwell_keeps_velocity_gate_and_measured_release():
    fast, state, p, t, b, start = arrive(.3)
    slow, _, _, _, _, slow_start = arrive(1.5)
    assert slow_start-start == pytest.approx(1.2, abs=.021)
    assert np.linalg.norm(fast.velocity) < .03 and abs(fast.yaw_rate) < .06
    assert state['lowering_fraction'] == state['lowering_xy_fraction'] == 0.
    for elapsed, xy, z in [(2., .5, 0.), (4., 1., 0.), (10., 1., .5), (16., 1., 1.)]:
        state = fast.update(start+elapsed, p, t, b, [6, 0, 6, 0], 0.)
        assert state['phase'] == 'lower' and state['release_fraction'] == 0.
        assert state['lowering_xy_fraction'] == pytest.approx(xy)
        assert state['lowering_fraction'] == pytest.approx(z)
        assert state['command_xyz'] == [0., 0., 0.]
    assert fast.lower_duration == 16.
    # Finished path without receiving support must not open.
    with pytest.raises(RuntimeError, match='destination support'):
        fast.update(start+25.1, p, t, b, [6, 0, 6, 0], 0.)


@pytest.mark.parametrize('kwargs', [
    {'arrival_settle_seconds': .29}, {'arrival_settle_seconds': float('nan')},
    {'arrival_settle_seconds': .3}, {'xy_first_lowering': True},
])
def test_relaxations_require_bounded_reviewed_heading_route(kwargs):
    with pytest.raises(ValueError):
        WalkingTransfer([1.16, -.26, .758], -np.pi/4, .1, 4., **kwargs)


@pytest.mark.parametrize('kwargs', [{'xy_first_lowering': True}, {'arrival_settle_seconds': .3}])
def test_launcher_rejects_unreviewed_relaxation_before_creating_scene(tmp_path, kwargs):
    from transport_trial import run
    with pytest.raises(ValueError, match='heading cart'):
        run(tmp_path/'new', tmp_path/'missing_motion.npz', **kwargs)
    assert not (tmp_path/'new').exists()
