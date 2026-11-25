import json

import numpy as np

from igris.umi_walklow_replay import tracking as module

FPS = 15.


def quaternion(axis, angle):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    return np.r_[np.sin(angle / 2) * axis, np.cos(angle / 2)]          # xyzw, as recorded


def truth(t):
    """Two hands 27 cm apart moving together at up to 0.27 m/s, as in a carry; rows are xyz + xyzw."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    centre = np.c_[.35 + .12 * np.sin(2 * np.pi * .15 * t), .20 * np.sin(2 * np.pi * .12 * t + .4),
                   1. + .15 * np.sin(2 * np.pi * .2 * t + 1.)]
    turn = .5 * np.sin(2 * np.pi * .2 * t)
    left = np.c_[centre + [0., .135, 0.], [quaternion([0, 0, 1], angle) for angle in turn]]
    right = np.c_[centre - [0., .135, 0.], [quaternion([0, 1, 0], -angle) for angle in turn]]
    return left, right, .6 * np.sin(2 * np.pi * .15 * t)


def recording(count=300, sample_times=None):
    t = np.arange(count) / FPS
    left, right, head_yaw = truth(t if sample_times is None else sample_times)
    return dict(t=t, fps=FPS, left=left, right=right, head_yaw=head_yaw,
                tactile_left=np.zeros(count), tactile_right=np.zeros(count))


def steps(pose, first, last):
    return np.linalg.norm(np.diff(pose[first:last + 1, :3], axis=0), axis=1)


def turns_deg(pose, first, last):
    dot = np.abs(np.sum(pose[first:last, 3:] * pose[first + 1:last + 1, 3:], axis=1))
    return np.degrees(2 * np.arccos(np.clip(dot, 0., 1.)))


def events(report):
    return [(event['hand'], event['kind'], event['frame'], event['frames']) for event in report['events']]


def test_a_recording_without_faults_stays_within_a_millimetre():
    episode = recording()
    cleaned, report = module.clean(episode)
    for side in ('left', 'right'):
        assert np.linalg.norm(cleaned[side][:, :3] - episode[side][:, :3], axis=1).max() < .001
        assert np.allclose(np.linalg.norm(cleaned[side][:, 3:], axis=1), 1., atol=1e-9)
        assert np.sum(cleaned[side][:, 3:] * episode[side][:, 3:], axis=1).min() > np.cos(np.radians(.2) / 2)
        assert report['moved_mm'][side]['max'] < 1.
    assert np.abs(cleaned['head_yaw'] - episode['head_yaw']).max() < np.radians(.2)
    assert report['events'] == [] and report['repaired_frames'] == {'left': 0, 'right': 0}
    json.dumps(report)                      # the report goes into episode.json and the manifest


def test_the_recorded_episode_is_not_modified():
    episode = recording()
    episode['left'][100:102, :3] += [0., .12, .08]
    before = {key: np.array(value, copy=True) for key, value in episode.items()}
    module.clean(episode)
    for key, value in before.items():
        assert np.array_equal(episode[key], value), key


def test_a_two_frame_spike_is_replaced_by_the_path_between_its_neighbours():
    episode = recording()
    expected = episode['left'].copy()
    episode['left'][100:102, :3] += [0., .12, .08]
    cleaned, report = module.clean(episode)
    assert np.linalg.norm(cleaned['left'][97:105, :3] - expected[97:105, :3], axis=1).max() < .005
    assert np.linalg.norm(cleaned['right'][:, :3] - episode['right'][:, :3], axis=1).max() < .001
    assert events(report) == [('left', 'spike', 100, 2)]
    assert report['repaired_frames'] == {'left': 2, 'right': 0}


def test_a_hand_that_freezes_and_then_jumps_is_bridged_at_an_even_pace():
    episode = recording()
    episode['left'][150:159, :3] = episode['left'][149, :3] + 1e-4 * np.arange(1, 10)[:, None]
    episode['left'][150:159, 3:] = episode['left'][149, 3:]
    jump = steps(episode['left'], 158, 159)[0]
    assert jump > .08                       # the hand kept moving while its tracker stood still
    cleaned, report = module.clean(episode)
    assert steps(cleaned['left'], 147, 161).max() < .25 * jump
    assert events(report) == [('left', 'stall', 150, 9)]


def test_a_bit_identical_repeat_of_the_previous_frame_is_treated_as_missing():
    episode = recording()
    expected = {side: episode[side].copy() for side in ('left', 'right')}
    for side in ('left', 'right'):
        episode[side][62] = episode[side][61]
        assert np.linalg.norm(episode[side][62, :3] - expected[side][62, :3]) > .015
    cleaned, report = module.clean(episode)
    for side in ('left', 'right'):
        assert np.linalg.norm(cleaned[side][62, :3] - expected[side][62, :3]) < .002
    assert events(report) == [('left', 'repeat', 62, 1), ('right', 'repeat', 62, 1)]


def test_a_jump_that_does_not_come_back_is_spread_over_the_neighbouring_frames():
    episode = recording()
    episode['left'][200:, :3] += [0., .09, 0.]
    before = steps(episode['left'], 195, 205).max()
    cleaned, report = module.clean(episode)
    assert steps(cleaned['left'], 195, 205).max() < .5 * before
    assert np.allclose(cleaned['left'][[190, 210], :3], episode['left'][[190, 210], :3], atol=.002)
    assert events(report) == [('left', 'step', 200, 4)]


def test_an_orientation_jump_without_a_position_jump_is_spread_too():
    episode = recording()
    tilt = quaternion([1, 0, 0], np.radians(10.))
    x, y, z, w = episode['right'][120:, 3:].T
    # Hamilton product recorded * tilt, both xyzw.
    episode['right'][120:, 3:] = np.c_[w * tilt[0] + x * tilt[3] + y * tilt[2] - z * tilt[1],
                                       w * tilt[1] - x * tilt[2] + y * tilt[3] + z * tilt[0],
                                       w * tilt[2] + x * tilt[1] - y * tilt[0] + z * tilt[3],
                                       w * tilt[3] - x * tilt[0] - y * tilt[1] - z * tilt[2]]
    assert turns_deg(episode['right'], 115, 125).max() > 9.
    cleaned, report = module.clean(episode)
    assert turns_deg(cleaned['right'], 115, 125).max() < 6.
    assert events(report) == [('right', 'step', 120, 4)]


def test_a_fast_reach_is_not_mistaken_for_a_jump():
    episode = recording(120)
    u = np.clip((episode['t'] - 3.) / 1., 0., 1.)
    reach = .6 * (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5)          # 0.6 m in one second, 1.1 m/s at its peak
    for side in ('left', 'right'):
        episode[side][:, 0] += reach
    assert steps(episode['left'], 40, 65).max() > .06
    cleaned, report = module.clean(episode)
    assert report['events'] == []
    assert np.linalg.norm(cleaned['left'][:, :3] - episode['left'][:, :3], axis=1).max() < .005


def test_sample_time_jitter_common_to_both_hands_is_smoothed():
    count = 300
    t = np.arange(count) / FPS
    late = np.random.default_rng(3).choice([-1 / 30, 0., 1 / 30], size=count, p=[.15, .7, .15])
    episode = recording(count, sample_times=t + late)
    exact = recording(count)

    def unevenness(pose):
        distance = steps(pose, 0, count - 1)
        return np.median(np.abs(np.diff(distance)) / ((distance[1:] + distance[:-1]) / 2))

    def error(pose, side):
        return np.sqrt(np.mean(np.sum((pose[4:-4, :3] - exact[side][4:-4, :3]) ** 2, axis=1)))

    cleaned, report = module.clean(episode)
    # A sample taken late and the next one taken early are the same pose twice, as in the recording.
    assert {event['kind'] for event in report['events']} <= {'repeat'}
    for side in ('left', 'right'):
        assert unevenness(cleaned[side]) < .4 * unevenness(episode[side])
        assert error(cleaned[side], side) < .75 * error(episode[side], side)
    yaw = lambda values: np.sqrt(np.mean((values[4:-4] - exact['head_yaw'][4:-4]) ** 2))
    assert yaw(cleaned['head_yaw']) < .75 * yaw(episode['head_yaw'])


def test_one_hand_moving_quickly_on_its_own_keeps_its_motion():
    episode = recording()
    wiggle = .03 * np.sin(2 * np.pi * 1. * episode['t'])           # 1 Hz, 3 cm
    episode['left'][:, 2] += wiggle
    cleaned, report = module.clean(episode)
    assert report['events'] == []
    kept = np.dot(cleaned['left'][10:-10, 2] - cleaned['right'][10:-10, 2], wiggle[10:-10]) / np.dot(wiggle[10:-10], wiggle[10:-10])
    assert .95 < kept < 1.02


def test_a_fault_at_the_end_of_the_episode_holds_the_last_good_sample():
    episode = recording(60)
    for side in ('left', 'right'):
        episode[side][57:] = episode[side][56]
    cleaned, report = module.clean(episode)
    assert all(np.isfinite(cleaned[side]).all() for side in ('left', 'right'))
    assert np.linalg.norm(cleaned['left'][59, :3] - episode['left'][56, :3]) < .01
    assert events(report) == [('left', 'repeat', 57, 3), ('right', 'repeat', 57, 3)]


def test_an_episode_shorter_than_the_smoothing_window_is_returned_as_recorded():
    episode = recording(5)
    cleaned, report = module.clean(episode)
    assert np.array_equal(cleaned['left'], episode['left']) and np.array_equal(cleaned['right'], episode['right'])
    assert report['events'] == []
