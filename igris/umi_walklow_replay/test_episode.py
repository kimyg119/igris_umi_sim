import json

import numpy as np
import pytest

from igris.umi_walklow_replay import episode as module


def synthetic(carries=((30, 90), (150, 210), (270, 330)), fps=15., height=.15):
    """Three boxes taken top first from a stack at (0.4, -0.3) and set down at (0.15, 0.4)."""
    count = 360
    t = np.arange(count) / fps
    left = np.tile([.4, .2, .4, 0., 0., 0., 1.], (count, 1)).astype(float)
    right = np.tile([.4, -.2, .4, 0., 0., 0., 1.], (count, 1)).astype(float)
    tactile_left = np.full(count, 1.); tactile_right = np.full(count, 1.)
    for order, (first, last) in enumerate(carries):
        level = len(carries) - 1 - order
        for i in range(first, last + 1):
            # As in the recording, the hands rest about a second at the pick and at the set-down.
            u = float(np.clip((i - first - 15) / (last - first - 30), 0., 1.))
            centre = np.array([.4, -.3, level * height + .075]) * (1 - u) + np.array([.15, .4, order * height + .075]) * u
            left[i, :3] = centre + [0, .135, 0]; right[i, :3] = centre - [0, .135, 0]
        tactile_left[first:last + 1] = 40.; tactile_right[first:last + 1] = 40.
    return dict(t=t, fps=fps, left=left, right=right, head_yaw=np.zeros(count),
                tactile_left=tactile_left, tactile_right=tactile_right)


def test_carry_segments_and_geometry():
    episode = synthetic()
    segments = module.carry_segments(episode)
    assert [(s['pick_index'], s['place_index']) for s in segments] == [(30, 90), (150, 210), (270, 330)]
    assert np.allclose(segments[0]['pick_mid'][:2], [.4, -.3], atol=.01)
    assert np.allclose(segments[0]['place_mid'][:2], [.15, .4], atol=.01)
    assert segments[0]['separation'] == pytest.approx(.27, abs=1e-6)
    report = module.geometry(episode)
    assert report['boxes'] == 3
    assert report['box_height_m'] == pytest.approx(.15, abs=.005)
    assert report['source_minus_destination_top_m'] == pytest.approx(0., abs=.005)
    assert report['pick_axis_yaw_rad'] == pytest.approx(np.pi / 2, abs=1e-6)
    json.dumps(report)                      # the report must be plain JSON


def test_short_or_one_sided_contact_is_not_a_carry():
    episode = synthetic(carries=((30, 50),))            # 1.3 s: below the 3 s minimum
    assert module.carry_segments(episode) == []
    episode = synthetic()
    episode['tactile_right'][:] = 1.                    # only the left palm presses
    assert module.carry_segments(episode) == []
    with pytest.raises(ValueError):
        module.geometry(episode)


def test_continuous_quaternions_removes_sign_flips():
    q = np.tile([0., 0., np.sin(.2), np.cos(.2)], (6, 1))
    q[2] *= -1; q[4:] *= -1
    fixed = module.continuous_quaternions(q)
    assert np.allclose(fixed, fixed[0])


def test_load_episode_rejects_bad_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        module.load_episode(tmp_path, 0)
    (tmp_path / 'data').mkdir(); (tmp_path / 'meta').mkdir()
    names = [f'{s}_ee.{c}' for s in ('left', 'right') for c in ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')] + ['head.yaw']
    (tmp_path / 'meta/info.json').write_text(json.dumps(
        {'total_episodes': 2, 'fps': 15, 'features': {'observation.state': {'names': names}}}))
    with pytest.raises(ValueError, match='outside'):
        module.load_episode(tmp_path, 5)


def test_light_grasp_is_one_carry():
    episode = synthetic(carries=((30, 150),))
    episode['tactile_left'][60:120] = 5.5               # weak palm between the on and off thresholds
    assert [(s['pick_index'], s['place_index']) for s in module.carry_segments(episode)] == [(30, 150)]


def test_late_pressure_does_not_move_the_pick():
    episode = synthetic(carries=((30, 150),))
    episode['tactile_left'][30:80] = 1.; episode['tactile_right'][30:80] = 1.     # pressure appears mid-carry
    segment = module.carry_segments(episode)[0]
    assert segment['pick_index'] <= 31                    # found from the constant palm distance
    assert np.allclose(segment['pick_mid'][:2], [.4, -.3], atol=.01)
