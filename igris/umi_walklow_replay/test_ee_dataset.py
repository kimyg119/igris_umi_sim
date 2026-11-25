"""Frames of a simulated replay as training data (ee_dataset.py)."""
import json

import numpy as np
import pytest

from igris.umi_walklow_replay import ee_dataset, ee_features
from igris.umi_walklow_replay.ik import POSTURE
from igris.umi_walklow_replay.kinematics import ACTIVE, JOINTS, UpperBody, axis_rotation, matrix_quat

RATE = 15.
START = 30.


def fake_run(tmp_path, seconds=2., tick=.011, with_poses=True, **result):
    """A run folder holding only what ``frames`` reads: 1 s of lead-in, ``seconds`` of replay, 1 s of hold."""
    run = tmp_path / 'out/run'; run.mkdir(parents=True)
    sim = np.arange(START - 1., START + seconds + 1., tick)
    phase = np.where(sim < START, 'approach', np.where(sim < START + seconds, 'replay', 'hold'))
    q = np.zeros((len(sim), len(JOINTS))); q[:, ACTIVE] = POSTURE
    q[:, 0] = .1 * (sim - START)                                    # the waist turns at 0.1 rad/s
    command = q.copy(); command[:, 0] += .01
    pelvis = np.zeros((len(sim), 7)); pelvis[:, 2] = .87; pelvis[:, 0] = .02 * (sim - START)      # the pelvis creeps ahead
    for index, time in enumerate(sim):
        pelvis[index, 3:] = matrix_quat(axis_rotation(np.array([0., 0., 1.]), .05 * (time - START)))
    left = np.tile(np.eye(4), (len(sim), 1, 1)); right = left.copy()
    left[:, 0, 3] = .3 + .1 * (sim - START); left[:, 1, 3] = .2; right[:, 0, 3] = .3; right[:, 1, 3] = -.2
    left[phase == 'hold', 0, 3] = 9.                                # what the hold logs is not a followed target
    log = dict(sim=sim, phase=phase, q_measured=q, q_command=command, pelvis=pelvis, parcels=np.zeros((len(sim), 3, 7)))
    if with_poses:
        log.update(target_left_pose=left, target_right_pose=right)
    np.savez(run / 'replay_log.npz', **log)
    (run / 'replay_result.json').write_text(json.dumps({'pass': True, 'stream_hz': RATE, 'replay_start_sim': START, **result}))
    (run / 'manifest.json').write_text(json.dumps(dict(umi_replay=dict(episode=7))))
    (run / 'analysis.json').write_text(json.dumps(dict(boxes_stacked=3, stack_success=True)))
    return tmp_path / 'out', sim


@pytest.fixture(scope='module')
def model():
    return UpperBody.from_mjcf()


def test_one_frame_per_stream_sample_with_what_was_known_then(tmp_path, model):
    out, sim = fake_run(tmp_path)
    data = ee_dataset.frames(out, model)
    count = len(data['t'])
    assert count == 29 or count == 30                              # 2 s at 15 Hz, the last sample has to be reached
    assert np.allclose(data['t'], np.arange(count) / RATE) and np.allclose(data['sim'], START + data['t'])
    assert data['episode'] == 7 and data['stack_success'] and data['boxes_stacked'] == 3
    # State: the joint angles of the newest tick, never a later one.
    waist = data['q_measured'][:, 0]
    assert np.all(waist <= .1 * data['t'] + 1e-9) and np.all(waist >= .1 * (data['t'] - .011) - 1e-9)
    for index in (0, count // 2, count - 1):
        hands, chest, angles = ee_features.body(model, data['q_measured'][index])
        assert np.allclose(data['ee_pelvis'][index], hands) and np.allclose(data['chest_pelvis'][index], chest)
        assert np.allclose(data['state'][index], ee_features.state(hands, chest, angles))


def test_action_of_a_frame_is_the_target_one_sample_period_later(tmp_path, model):
    out, sim = fake_run(tmp_path)
    data = ee_dataset.frames(out, model)
    later = data['t'] + 1. / RATE
    reached = data['target_pelvis'][:, 0, 0, 3]                     # left hand target x: 0.3 + 0.1 * time
    assert np.all(reached <= .3 + .1 * later + 1e-9) and np.all(reached >= .3 + .1 * (later - .012) - 1e-9)
    assert np.all(reached < 5.)                                     # never a row of the hold after the replay
    assert np.allclose(data['waist_target'][:, 0], .1 * (reached - .3) / .1 + .01, atol=1e-9)
    # In the room: through the pelvis pose of the tick the target was read at.
    for index in (0, len(later) - 1):
        time = (reached[index] - .3) / .1
        pelvis = np.eye(4); pelvis[:3, :3] = axis_rotation(np.array([0., 0., 1.]), .05 * time); pelvis[:3, 3] = [.02 * time, 0., .87]
        assert np.allclose(data['target_room'][index, 0], pelvis @ data['target_pelvis'][index, 0], atol=1e-9)
    assert np.allclose(data['ee_room'][3], data['pelvis_room'][3] @ data['ee_pelvis'][3])


def test_a_run_that_is_not_a_finished_stream_replay_is_refused(tmp_path, model):
    out, _ = fake_run(tmp_path / 'a', with_poses=False)
    with pytest.raises(ValueError, match='hand target poses'):
        ee_dataset.frames(out, model)
    out, _ = fake_run(tmp_path / 'b')
    (out / 'run/replay_result.json').write_text(json.dumps({'pass': False, 'stream_hz': RATE, 'replay_start_sim': START}))
    with pytest.raises(ValueError, match='not a finished'):
        ee_dataset.frames(out, model)


def test_small_pictures_come_back_from_the_jpeg_store(tmp_path):
    pytest.importorskip('cv2')
    store = ee_dataset.JpegStore(tmp_path / 'cam.jpgs')
    pictures = []
    for shade in (40, 120, 200):
        picture = np.full((480, 640, 3), shade, dtype=np.uint8); picture[:, :320, 0] = 250       # red on the left
        pictures.append(picture); store.add(picture)
    offsets = store.close()
    blob = (tmp_path / 'cam.jpgs').read_bytes()
    assert len(offsets) == 4 and offsets[-1] == len(blob)
    for index, shade in enumerate((40, 120, 200)):
        small = ee_dataset.read_jpeg(blob, offsets, index)
        assert small.shape == (ee_dataset.SMALL[1], ee_dataset.SMALL[0], 3)
        assert abs(int(small[100, 250, 1]) - shade) < 6 and small[100, 40, 0] > 235 and abs(int(small[100, 40, 2]) - shade) < 12
