"""State and relative actions of the chest and the two hands (ee_features.py)."""
import numpy as np
import pytest

from igris.umi_walklow_replay import ee_features
from igris.umi_walklow_replay.ik import POSTURE
from igris.umi_walklow_replay.kinematics import ACTIVE, JOINTS, UpperBody, axis_rotation


def pose(rng, reach=.3):
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    out = np.eye(4); out[:3, :3] = axis_rotation(axis, rng.uniform(-2., 2.)); out[:3, 3] = rng.normal(size=3) * reach
    return out


@pytest.fixture(scope='module')
def model():
    return UpperBody.from_mjcf()


def test_pose9_round_trip_and_repair_of_a_rotation_that_is_not_one():
    rng = np.random.default_rng(0)
    poses = np.array([pose(rng) for _ in range(5)])
    assert np.allclose(ee_features.from9(ee_features.pose9(poses)), poses, atol=1e-12)
    bent = ee_features.pose9(poses) + rng.normal(size=(5, 9)) * .02            # what a network gives back
    repaired = ee_features.from9(bent)[:, :3, :3]
    assert np.allclose(repaired @ np.swapaxes(repaired, 1, 2), np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(repaired), 1.)


def test_inverse_of_a_rigid_transform():
    rng = np.random.default_rng(1)
    poses = np.array([pose(rng) for _ in range(4)])
    assert np.allclose(ee_features.inverse(poses) @ poses, np.eye(4), atol=1e-12)


def test_state_does_not_change_when_the_whole_robot_is_moved_in_the_room(model):
    q = np.zeros(len(JOINTS)); q[ACTIVE] = POSTURE; q[0] = .4; q[2] = -.1
    hands, chest, waist = ee_features.body(model, q)
    here = ee_features.state(hands, chest, waist)
    assert here.shape == (ee_features.STATE_SIZE,) and len(ee_features.STATE_NAMES) == ee_features.STATE_SIZE
    pelvis = pose(np.random.default_rng(2))
    assert np.allclose(ee_features.state(pelvis @ hands, pelvis @ chest, waist), here, atol=1e-12)
    assert np.allclose(here[27:], q[:3])
    turned = q.copy(); turned[0] += .3                              # the waist alone turns: only the waist values change
    assert np.allclose(ee_features.state(*ee_features.body(model, turned))[:27], here[:27], atol=1e-12)
    # The pair: the right hand as the left hand sees it.
    assert np.allclose(ee_features.from9(here[18:27]), ee_features.inverse(hands[0]) @ hands[1], atol=1e-12)
    many = ee_features.state(np.stack([hands, hands]), np.stack([chest, chest]), np.stack([waist, waist]))
    assert many.shape == (2, ee_features.STATE_SIZE) and np.allclose(many[1], here)


def test_measured_angles_are_read_with_the_calibrated_slack(model):
    q = np.zeros(len(JOINTS)); q[ACTIVE] = POSTURE
    hands, _, waist = ee_features.body(model, q)
    assert np.allclose(waist, q[:3])
    assert not np.allclose(hands, model.fk(q))                       # the links stand lower than the motors say
    assert np.allclose(hands, model.fk(ee_features.links(q)))


def test_actions_are_undone_by_absolute_targets_and_do_not_depend_on_the_frame():
    rng = np.random.default_rng(3)
    now = np.array([pose(rng), pose(rng)]); waist = rng.normal(size=3) * .2
    targets = np.array([[pose(rng), pose(rng)] for _ in range(6)]); waists = rng.normal(size=(6, 3)) * .2
    for paired in (True, False):
        actions = ee_features.relative_actions(now, waist, targets, waists, paired)
        assert actions.shape == (6, ee_features.ACTION_SIZE) and len(ee_features.ACTION_NAMES) == ee_features.ACTION_SIZE
        again, again_waist = ee_features.absolute_targets(now, waist, actions, paired)
        assert np.allclose(again, targets, atol=1e-12) and np.allclose(again_waist, waists)
        room = pose(rng)                                            # the same motion seen from the room
        assert np.allclose(ee_features.relative_actions(room @ now, waist, room @ targets, waists, paired), actions, atol=1e-12)


def test_an_error_in_the_left_hand_action_moves_the_pair_and_keeps_the_squeeze():
    """Two hands 286 mm apart on a box.  The left hand's action is 2 cm off: with paired hands both targets move
    and stay 286 mm apart; with a relative action for each hand the gap changes by the error."""
    left = np.eye(4); right = np.eye(4); right[1, 3] = -.286
    now = np.array([left, right]); targets = now[None].copy(); targets[:, :, 0, 3] += .05      # both 5 cm ahead
    for paired, kept in ((True, True), (False, False)):
        actions = ee_features.relative_actions(now, np.zeros(3), targets, np.zeros((1, 3)), paired)
        actions[0, 1] -= .02                                         # the left hand's sideways value, 2 cm wrong
        got, _ = ee_features.absolute_targets(now, np.zeros(3), actions, paired)
        gap = np.linalg.norm(got[0, 0, :3, 3] - got[0, 1, :3, 3])
        assert np.isclose(gap, .286) == kept


def test_a_hand_that_stays_where_it_is_gives_the_identity_action():
    rng = np.random.default_rng(4)
    now = np.array([pose(rng), pose(rng)])
    actions = ee_features.relative_actions(now, np.zeros(3), now[None], np.zeros((1, 3)), paired=False)
    still = np.r_[0., 0., 0., 1., 0., 0., 0., 1., 0.]
    assert np.allclose(actions[0], np.r_[still, still, 0., 0., 0.], atol=1e-12)
    paired = ee_features.relative_actions(now, np.zeros(3), now[None], np.zeros((1, 3)))
    assert np.allclose(paired[0, :9], still, atol=1e-12)
    assert np.allclose(ee_features.from9(paired[0, 9:18]), ee_features.inverse(now[0]) @ now[1], atol=1e-12)


def test_labels_made_in_the_room_are_executed_in_the_pelvis_frame():
    """Labels: hand now and target both in the room.  Execution: the action on the hand in the pelvis frame.
    With a pelvis that has not moved the hand is sent to the labelled place."""
    rng = np.random.default_rng(5)
    pelvis = pose(rng)
    hands_pelvis = np.array([pose(rng), pose(rng)]); goals_room = np.array([[pose(rng), pose(rng)]])
    actions = ee_features.relative_actions(pelvis @ hands_pelvis, np.zeros(3), goals_room, np.zeros((1, 3)))
    executed, _ = ee_features.absolute_targets(hands_pelvis, np.zeros(3), actions)
    assert np.allclose(pelvis @ executed, goals_room, atol=1e-12)
