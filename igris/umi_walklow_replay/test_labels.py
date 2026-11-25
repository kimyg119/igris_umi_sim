"""Pose labels a robot can follow at their own clock (labels.py).  The offline solve needs mink (see README)."""
import numpy as np
import pytest

pytest.importorskip('mink')

from igris.umi_walklow_replay import labels, retarget                 # noqa: E402
from igris.umi_walklow_replay.governor import JOINT_SPEED             # noqa: E402
from igris.umi_walklow_replay.ik import POSTURE                       # noqa: E402
from igris.umi_walklow_replay.kinematics import ACTIVE, JOINTS, UpperBody, axis_rotation, rotation_vector   # noqa: E402


@pytest.fixture(scope='module')
def model():
    return UpperBody.from_mjcf()


def fast_reach(model, seconds=.6, distance=.30):
    """100 Hz targets: the left hand goes 30 cm up and forward in 0.6 s (the arm would need about 2 rad/s), then waits."""
    q = np.zeros(len(JOINTS)); q[ACTIVE] = POSTURE
    left, right = model.fk(q)
    stamps = np.arange(0., seconds + .5, .01)
    travel = np.clip(stamps / seconds, 0., 1.)
    travel = travel * travel * (3 - 2 * travel) * distance
    lefts = np.repeat(left[None], len(stamps), axis=0)
    lefts[:, :3, 3] += travel[:, None] * np.array([.6, 0., .8])
    return dict(t=stamps, left=lefts, right=np.repeat(right[None], len(stamps), axis=0), head_yaw=np.zeros(len(stamps)))


@pytest.fixture(scope='module')
def made(model):
    targets = fast_reach(model)
    return targets, labels.feasible(targets, base_offset_z=.03)


def test_labels_are_the_hand_poses_of_one_joint_path(model, made):
    _, label = made
    for index in (0, len(label['t']) // 2, len(label['t']) - 1):
        left, right = model.fk(label['q'][index])
        left[2, 3] -= .03; right[2, 3] -= .03                     # the file leaves the base height to the replay
        assert np.allclose(label['left'][index], left, atol=1e-12) and np.allclose(label['right'][index], right, atol=1e-12)
    assert np.allclose(label['head_yaw'], -label['q'][:, 0])       # the trunk heading the solution has, not the operator's head
    assert np.allclose(np.diff(label['t']), .01) and label['base_offset_z'] == .03


def test_the_joint_path_keeps_the_speed_limit_by_taking_longer(made):
    targets, label = made
    speed = np.abs(np.diff(label['q'][:, ACTIVE], axis=0)).max() * 100.
    assert speed <= JOINT_SPEED + 1e-6
    assert label['t'][-1] > targets['t'][-1] + .1                  # the recording asked for about twice that speed
    # Every label sample names the recorded time it shows; the whole recording is covered, in order.
    assert np.all(np.diff(label['source_t']) >= 0) and np.isclose(label['source_t'][-1], targets['t'][-1])


def test_labels_end_where_the_recording_ends(model, made):
    targets, label = made
    wanted = targets['left'][-1].copy(); wanted[2, 3] += .03
    assert np.linalg.norm(model.fk(label['q'][-1])[0][:3, 3] - wanted[:3, 3]) < 3e-3


def test_resample_gives_uniform_stamps_and_poses_in_between(made):
    _, label = made
    coarse = labels.resample(label, 15.)
    assert np.allclose(coarse['t'], np.arange(len(coarse['t'])) / 15.) and coarse['t'][-1] <= label['t'][-1]
    for index in (1, len(coarse['t']) // 2):
        before = int(np.floor(coarse['t'][index] * 100. + 1e-9))
        low, high = label['left'][before][:3, 3], label['left'][before + 1][:3, 3]
        share = coarse['t'][index] * 100. - before
        assert np.allclose(coarse['left'][index][:3, 3], (1 - share) * low + share * high, atol=1e-12)
    assert set(coarse) >= {'t', 'source_t', 'left', 'right', 'head_yaw', 'q'}
    assert len(coarse['q']) == len(coarse['t']) == len(coarse['source_t'])


def test_state_rows_have_the_layout_of_the_dataset(made):
    _, label = made
    coarse = labels.resample(label, 15.)
    rows = labels.state_rows(coarse)
    assert rows.shape == (len(coarse['t']), 15)                    # left xyz + xyzw, right xyz + xyzw, trunk heading
    assert np.allclose(retarget.pose_matrices(rows[:, :7]), coarse['left'], atol=1e-9)
    assert np.allclose(retarget.pose_matrices(rows[:, 7:14]), coarse['right'], atol=1e-9)
    assert np.allclose(rows[:, 14], coarse['head_yaw'])
    assert np.all(np.sum(rows[1:, 3:7] * rows[:-1, 3:7], axis=1) > 0)        # no sign flips between neighbours


def test_unshaped_stream_applies_constants_only():
    """Each output pose is the recorded palm pose times one constant transform, moved by where the robot stands."""
    rng = np.random.default_rng(3)
    count = 20
    pose = np.zeros((count, 7)); pose[:, :3] = rng.normal(size=(count, 3)) * .2
    for row in pose:
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        angle = rng.uniform(0., 1.)
        row[3:6] = np.sin(angle / 2) * axis; row[6] = np.cos(angle / 2)
    episode = dict(t=np.arange(count) / 15., fps=15., left=pose, right=pose.copy(), head_yaw=np.linspace(0., .5, count))
    stream = labels.unshaped(episode, shift=(.07, -.03))
    assert np.allclose(stream['t'], episode['t']) and np.allclose(stream['head_yaw'], episode['head_yaw'])
    recorded = retarget.pose_matrices(pose)
    for side in ('left', 'right'):
        moved = stream[side].copy(); moved[:, :3, 3] += [.07, -.03, 0.]
        constant = [np.linalg.inv(a) @ b for a, b in zip(recorded, moved)]
        assert np.allclose(constant, constant[0], atol=1e-9)
        assert np.linalg.norm(rotation_vector(constant[0][:3, :3])) > .01 or np.linalg.norm(constant[0][:3, 3]) > .01
