"""The pose-only stand-in policy (pose_policy.py): past hand poses in, a chunk of future hand poses out."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from igris.umi_walklow_replay import pose_policy                     # noqa: E402
from igris.umi_walklow_replay.kinematics import axis_rotation, matrix_quat   # noqa: E402


def gliding(count=120, turn=.01, seed=0):
    """State rows (N x 15) of two hands moving at a constant velocity and turning at a constant rate about z."""
    rng = np.random.default_rng(seed)
    velocity = rng.normal(size=(2, 3)) * .004
    start = rng.normal(size=(2, 3)) * .1
    rows = np.zeros((count, 15))
    for index in range(count):
        for hand in range(2):
            rows[index, 7 * hand:7 * hand + 3] = start[hand] + index * velocity[hand]
            w, x, y, z = matrix_quat(axis_rotation(np.array([0., 0., 1.]), (1 - 2 * hand) * turn * index))
            rows[index, 7 * hand + 3:7 * hand + 7] = [x, y, z, w]
        rows[index, 14] = .002 * index
    return rows


def test_chunk_targets_are_undone_by_apply():
    rows = gliding()
    latest = 40
    chunk = pose_policy.chunk_targets(rows, latest)
    assert chunk.shape == (pose_policy.HORIZON, 13)
    future = pose_policy.apply(rows[latest], chunk)
    assert np.allclose(future, rows[latest + 1:latest + 1 + pose_policy.HORIZON], atol=1e-9)


def test_past_the_end_of_an_episode_the_chunk_stays_at_the_last_pose():
    rows = gliding(count=50)
    chunk = pose_policy.chunk_targets(rows, 45)
    future = pose_policy.apply(rows[45], chunk)
    assert np.allclose(future[:4], rows[46:50], atol=1e-9) and np.allclose(future[4:], rows[49], atol=1e-9)


def test_observation_holds_only_the_history_up_to_the_latest_sample():
    rows = gliding()
    changed = rows.copy(); changed[41:] += 5.
    assert np.array_equal(pose_policy.observation(rows, 40), pose_policy.observation(changed, 40))
    assert pose_policy.observation(rows, 40).shape == pose_policy.observation(rows, 2).shape     # the start is padded
    assert not np.array_equal(pose_policy.observation(rows, 40), pose_policy.observation(rows, 41))


@pytest.fixture(scope='module')
def trained():
    episodes = {index: gliding(seed=index) for index in range(12)}
    return episodes, pose_policy.train(episodes, held_out=(10, 11), epochs=150, seed=0, width=128)


def test_trained_network_continues_a_steady_motion(trained):
    episodes, saved = trained
    rows = episodes[11]                                             # not trained on
    chunk = pose_policy.predict(saved, rows, 60)
    assert chunk.shape == (pose_policy.HORIZON, 15)
    error = np.linalg.norm(chunk[:, :3] - rows[61:61 + pose_policy.HORIZON, :3], axis=1)
    still = np.linalg.norm(rows[60, :3] - rows[61:61 + pose_policy.HORIZON, :3], axis=1)        # error of not moving at all
    assert error[-1] < .5 * still[-1]
    assert saved['report']['held_out'] == [10, 11] and saved['report']['val_loss'] < saved['report']['first_val_loss']


def test_executed_stream_uses_no_sample_later_than_the_delay_allows(trained):
    """What arrives at sample j was computed from observations at least ``delay`` + 1 samples older."""
    episodes, saved = trained
    rows = episodes[10]
    stream, source = pose_policy.executed(saved, rows, replan=4, delay=2, details=True)
    assert stream.shape == rows.shape
    assert np.array_equal(stream[:3], np.repeat(rows[:1], 3, axis=0))        # until the first chunk arrives: the start pose, held
    assert np.all(np.arange(3, len(rows)) - source[3:] >= 3) and np.all(np.diff(source[3:]) % 4 == 0)
    changed = rows.copy(); changed[70:] += .3
    other = pose_policy.executed(saved, changed, replan=4, delay=2)
    first = int(np.flatnonzero(np.any(stream != other, axis=1))[0])
    assert first >= 70 + 3


def test_stream_file_has_what_the_replay_worker_reads(trained, tmp_path):
    episodes, saved = trained
    stream = pose_policy.executed(saved, episodes[10], replan=4, delay=2)
    path = tmp_path / 'stream.npz'
    pose_policy.save(path, stream, rate_hz=15., surface_top=.8, source_t=np.arange(len(stream)) / 15.)
    data = np.load(path)
    assert data['left'].shape == (len(stream), 4, 4) and np.allclose(data['left'][:, :3, 3], stream[:, :3])
    assert np.allclose(data['head_yaw'], stream[:, 14]) and float(data['rate_hz']) == 15. and 'source_t' in data.files


def test_blended_chunks_cross_fade_over_the_overlap_without_using_newer_observations(trained):
    """A new chunk is faded in over its own run against what the chunk before it said about the same samples."""
    episodes, saved = trained
    rows = episodes[10]
    plain, source = pose_policy.executed(saved, rows, replan=4, delay=2, details=True)
    mixed, mixed_source = pose_policy.executed(saved, rows, replan=4, delay=2, details=True, blend=True)
    assert np.array_equal(source, mixed_source)                       # the same observations at the same times
    older, newer = pose_policy.predict(saved, rows, 36), pose_policy.predict(saved, rows, 40)
    for offset, share in enumerate((.25, .5, .75, 1.)):
        sample = 40 + 2 + 1 + offset                                 # the run of the chunk observed at 40
        wanted = (1 - share) * older[sample - 37, :3] + share * newer[sample - 41, :3]
        assert np.allclose(mixed[sample, :3], wanted, atol=1e-9)
        assert np.isclose(np.linalg.norm(mixed[sample, 3:7]), 1.) and np.isclose(np.linalg.norm(mixed[sample, 10:14]), 1.)
    assert np.allclose(mixed[46], plain[46], atol=1e-9)               # at the end of its run the new chunk stands alone
    changed = rows.copy(); changed[70:] += .3
    other = pose_policy.executed(saved, changed, replan=4, delay=2, blend=True)
    assert int(np.flatnonzero(np.any(mixed != other, axis=1))[0]) >= 70 + 3
