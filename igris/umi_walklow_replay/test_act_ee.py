"""The data side of the ACT training (act_ee.py): relative chunks, statistics, pictures, error measures."""
import numpy as np

from igris.umi_walklow_replay import act_ee, ee_features
from igris.umi_walklow_replay.kinematics import axis_rotation


def moving(count=40, seed=0):
    """Hands gliding and turning steadily, in a room frame; targets a little ahead of the hands."""
    rng = np.random.default_rng(seed)
    now = np.tile(np.eye(4), (count, 2, 1, 1)); targets = now.copy()
    for index in range(count):
        for hand in range(2):
            now[index, hand, :3, :3] = axis_rotation(np.array([0., 0., 1.]), .02 * index * (1 - 2 * hand))
            now[index, hand, :3, 3] = [.3 + .01 * index, .2 - .4 * hand, 1.]
            targets[index, hand, :3, :3] = axis_rotation(np.array([0., 0., 1.]), .02 * (index + 1) * (1 - 2 * hand))
            targets[index, hand, :3, 3] = [.3 + .01 * (index + 1), .2 - .4 * hand, 1.]
    waist = np.column_stack([.01 * np.arange(count), np.zeros(count), np.zeros(count)])
    return now, waist, targets, waist + .01


def test_chunk_of_a_frame_is_what_ee_features_gives_for_that_frame():
    now, waist, targets, waist_targets = moving()
    for paired in (True, False):
        actions, pad = act_ee.chunks(now, waist, targets, waist_targets, length=8, paired=paired)
        assert actions.shape == (40, 8, ee_features.ACTION_SIZE) and pad.shape == (40, 8) and not pad[:32].any()
        for frame in (0, 17, 31):
            wanted = ee_features.relative_actions(now[frame], waist[frame], targets[frame:frame + 8], waist_targets[frame:frame + 8], paired)
            assert np.allclose(actions[frame], wanted, atol=1e-6)


def test_absolute_chunks_give_the_targets_back():
    now, waist, targets, waist_targets = moving()
    for paired in (True, False):
        actions, pad = act_ee.absolute_chunks(targets, waist_targets, length=8, paired=paired)
        back = ee_features.pair_targets(actions[5]) if paired else np.stack(
            [ee_features.from9(actions[5][:, :9]), ee_features.from9(actions[5][:, 9:18])], axis=1)
        assert np.allclose(back, targets[5:13], atol=1e-6) and np.allclose(actions[5][:, 18:], waist_targets[5:13], atol=1e-6)


def test_past_the_end_the_last_target_is_repeated_and_marked():
    now, waist, targets, waist_targets = moving()
    actions, pad = act_ee.chunks(now, waist, targets, waist_targets, length=8)
    assert pad[36].tolist() == [False] * 4 + [True] * 4
    last = ee_features.relative_actions(now[36], waist[36], targets[39:40], waist_targets[39:40])[0]
    assert np.allclose(actions[36, 3:], last, atol=1e-6)


def test_chunks_do_not_change_when_the_room_frame_does():
    now, waist, targets, waist_targets = moving()
    room = np.eye(4); room[:3, :3] = axis_rotation(np.array([0., 1., 0.]), .7); room[:3, 3] = [3., -2., .5]
    one, _ = act_ee.chunks(now, waist, targets, waist_targets, length=6)
    two, _ = act_ee.chunks(room @ now, waist, room @ targets, waist_targets, length=6)
    assert np.allclose(one, two, atol=1e-5)


def test_statistics_leave_the_padding_out():
    class Stub:
        def __init__(self, seed):
            now, waist, targets, waist_targets = moving(seed=seed)
            self.actions, self.pad = act_ee.chunks(now, waist, targets, waist_targets, length=8)
            self.actions[self.pad] = 1e6                              # whatever is in the padding must not count
            self.state = np.random.default_rng(seed).normal(size=(40, ee_features.STATE_SIZE)).astype(np.float32)
    stats = act_ee.statistics([Stub(0), Stub(1)])
    assert np.all(np.abs(stats['action_mean']) < 10.) and np.all(stats['action_std'] >= 1e-4)
    assert stats['state_mean'].shape == (ee_features.STATE_SIZE,)


def test_training_picture_is_shifted_without_losing_its_size_and_stays_bytes():
    picture = np.zeros((240, 320, 3), dtype=np.uint8); picture[100:140, 150:170] = 255
    moved = act_ee.vary_picture(picture, shift=(5, -3))
    assert moved.shape == picture.shape and moved.dtype == np.uint8
    assert np.array_equal(moved[97:137, 155:175], picture[100:140, 150:170]) and moved[100, 152].max() == 0
    brighter = act_ee.vary_picture(picture, gain=(1., .1))
    assert brighter.dtype == np.uint8 and 24 <= brighter[0, 0, 0] <= 27 and brighter[120, 160, 0] == 255
    assert np.array_equal(act_ee.vary_picture(picture), picture)


def test_pictures_are_normalised_on_the_device_of_the_network():
    import pytest
    torch = pytest.importorskip('torch')
    picture = np.zeros((2, 240, 320, 3), dtype=np.uint8); picture[:, 100:140, 150:170] = 255
    out = act_ee.normalise(torch.as_tensor(picture))
    assert tuple(out.shape) == (2, 3, 240, 320) and out.dtype == torch.float32
    assert np.isclose(float(out[0, 0, 0, 0]), (0. - .485) / .229, atol=1e-5) and np.isclose(float(out[1, 2, 120, 160]), (1. - .406) / .225, atol=1e-5)


def test_error_measures_are_in_metres_and_degrees_over_the_executed_steps():
    now, waist, targets, waist_targets = moving()
    wanted, pad = act_ee.chunks(now, waist, targets, waist_targets, length=12)
    off = wanted.copy()
    off[:, :, 0] += .01                                              # left hand 1 cm off along its x, right hand exact
    off[:, :, 18] += np.radians(2.)                                  # waist yaw 2 degrees off
    report = act_ee.errors(off, wanted, pad)
    assert np.isclose(report['first_position_mean_m'], .01, atol=1e-6) and report['second_position_mean_m'] < 1e-6
    assert report['first_rotation_mean_deg'] < 1e-3
    assert np.isclose(report['waist_mean_deg'], 2. / 3., atol=1e-4)
    assert act_ee.errors(wanted, wanted, pad)['first_position_mean_m'] < 1e-9
