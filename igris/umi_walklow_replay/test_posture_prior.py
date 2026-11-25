import numpy as np
import pytest

from igris.umi_walklow_replay import posture_prior, retarget
from igris.umi_walklow_replay.ik import IK, POSTURE
from igris.umi_walklow_replay.kinematics import ACTIVE, JOINTS, UpperBody


@pytest.fixture(scope='module')
def model():
    return UpperBody.from_mjcf()


def table(tmp_path, model, configurations):
    """A small lookup table from the given 31-joint configurations (both arms, right mirrored)."""
    keys, joints = [], []
    for q in configurations:
        keys.extend(posture_prior.features(model, q)); joints.extend((q[15:22], q[22:29] * posture_prior.MIRROR))
    path = tmp_path / 'table.npz'
    np.savez(path, keys=np.array(keys), joints=np.array(joints), episodes=np.array([0]), meta='{}')
    return path


def test_mirrored_arms_share_one_key_and_one_posture(model):
    q = np.zeros(len(JOINTS)); q[ACTIVE] = POSTURE                      # POSTURE is left-right symmetric
    q[ACTIVE[3]] = -.8; q[ACTIVE[10]] = -.8
    keys = posture_prior.features(model, q)
    assert np.allclose(keys[0], keys[1], atol=1e-4)
    assert np.allclose(q[15:22], q[22:29] * posture_prior.MIRROR)


def test_lookup_returns_the_recorded_arm_for_a_recorded_hand_pose(model, tmp_path):
    rng = np.random.default_rng(2)
    lower, upper = model.lower[ACTIVE] + .2, model.upper[ACTIVE] - .2
    recorded = []
    for _ in range(60):
        q = np.zeros(len(JOINTS)); q[ACTIVE] = rng.uniform(lower, upper); q[:3] = 0.
        recorded.append(q)
    prior = posture_prior.TeleopPosture(table(tmp_path, model, recorded), neighbours=1)
    q = recorded[7]
    left, right = model.fk(q)
    wanted = prior.reference(model, q, left, right)
    assert np.allclose(wanted[:7], q[15:22], atol=1e-9) and np.allclose(wanted[7:], q[22:29], atol=1e-9)
    # Smoothing: a new target is approached over the time constant, not jumped to.
    other = recorded[8]
    step = prior.reference(model, other, *model.fk(other), dt=.01)
    assert np.linalg.norm(step - wanted) < .2 * np.linalg.norm(np.r_[other[15:22], other[22:29]] - wanted)


def test_ik_follows_the_given_posture_without_moving_the_hands(model):
    start = np.zeros(len(JOINTS)); start[ACTIVE] = POSTURE
    left, right = model.fk(start)
    wanted = start[15:29].copy(); wanted[1] += .25; wanted[8] -= .25      # both elbows further out
    plain, _ = IK(model).converge(start, left, right, ticks=500)
    shaped, info = IK(model).converge(start, left, right, posture=wanted, ticks=500)
    assert info['position_error'].max() < 1e-3
    assert shaped[16] > plain[16] + .03 and shaped[23] < plain[23] - .03


def test_orientation_counts_only_around_a_grasp():
    t = np.arange(0., 30., 1 / 15.)
    segment = dict(pick_index=150, place_index=300)                      # carry from 10 s to 20 s
    weight = retarget.orientation_weight(t, [segment], retarget.DEFAULT)
    blend_in, lead = retarget.DEFAULT['grasp_blend_in_s'], retarget.DEFAULT['orientation_lead_s']
    assert weight[0] == 0. and weight[-1] == 0.
    assert weight[np.searchsorted(t, 10. - blend_in)] > .99              # full when the approach begins
    assert weight[np.searchsorted(t, 10. - blend_in - lead - .1)] == 0.
    assert np.all(weight[150:300] == 1.) and weight[np.searchsorted(t, 22.)] == 0.
