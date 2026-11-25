from pathlib import Path

import os
import numpy as np
import pytest

from igris.umi_walklow_replay import teleop
from igris.umi_walklow_replay.ik import POSTURE
from igris.umi_walklow_replay.kinematics import ACTIVE, JOINTS, UpperBody

FULL_SIGNALS = Path(os.environ.get('UMI_TELEOP_DATASET', 'datasets/Igris_Teleop_Box_Palletizing_full_signals_15hz'))


def synthetic(frames=46, fps=15.):
    """A recording that sweeps the left elbow and the waist yaw; legs stand bent as in WALK_LOW."""
    t = np.arange(frames) / fps
    action = np.zeros((frames, len(JOINTS))); action[:, ACTIVE] = POSTURE
    action[:, 0] = .5 * np.sin(t); action[:, ACTIVE[6]] = -1. + .3 * np.sin(2 * t)
    action[:, 3] = -.4; action[:, 6] = .67
    return dict(t=100. + t, action=action, state=action + .01, task='synthetic', fps=fps, episode=0)


def test_dataset_names_map_to_the_sdk_order():
    assert teleop.sdk_name('left_shoulder_pitch.pos') == 'l_shoulder_pitch'
    assert teleop.sdk_name('right_ankle_roll.pos') == 'r_ankle_roll'
    assert teleop.sdk_name('waist_yaw.pos') == 'waist_yaw' and teleop.sdk_name('neck_pitch') == 'neck_pitch'


def test_targets_hold_the_commands_and_their_hand_poses():
    model = UpperBody.from_mjcf(); data = synthetic()
    target = teleop.targets(data, model)
    assert target['t'][0] == 0. and np.allclose(np.diff(target['t']), .01)
    assert np.all(target['q'][:, 3:15] == 0.) and np.all(target['q'][:, 29:] == 0.)       # legs and neck are not replayed
    at = int(np.searchsorted(target['t'], 2.))                                           # a recorded sample time (30 frames)
    assert np.allclose(target['q'][at, ACTIVE], data['action'][30, ACTIVE], atol=1e-6)
    left, right = model.fk(target['q'][at])
    assert np.allclose(target['left'][at], left) and np.allclose(target['right'][at], right)
    assert np.allclose(target['head_yaw'], -target['q'][:, 0])                            # the worker's waist reference is its negative


def test_check_reports_limits_speed_and_standing_legs():
    model = UpperBody.from_mjcf(); data = synthetic()
    report = teleop.check(data, model)
    assert report['command_beyond_limit_fraction'] == 0. and report['frames'] == 46
    assert report['command_speed_max_rad_s'] == pytest.approx(.6, abs=.05)                # elbow: 0.3 * 2 rad/s
    assert report['command_minus_measured_rms_rad'] == pytest.approx(.01)
    assert report['leg_measured_mean_rad']['l_knee_pitch'] == pytest.approx(.68)
    data['action'][5, ACTIVE[6]] = model.upper[ACTIVE[6]] + .2
    assert teleop.check(data, model)['command_beyond_model_limit_max_rad'] == pytest.approx(.2)


@pytest.mark.skipif(not (FULL_SIGNALS / 'meta/info.json').is_file(), reason='teleop dataset is not on this machine')
def test_recorded_episode_loads_with_all_joints_and_rejects_unknown_episodes():
    data = teleop.load_episode(FULL_SIGNALS, 744)
    assert data['action'].shape == (653, 31) and data['state'].shape == (653, 31) and data['fps'] == 15.
    assert 'place' in data['task'] and np.all(np.diff(data['t']) > 0)
    with pytest.raises(ValueError):
        teleop.load_episode(FULL_SIGNALS, 99999)
