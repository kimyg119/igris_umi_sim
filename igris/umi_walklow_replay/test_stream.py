"""A controller that is handed one pose at a time (stream.py).  The rollout needs mink (see README)."""
import numpy as np
import pytest

from igris.umi_walklow_replay.kinematics import ACTIVE, JOINTS, UpperBody, axis_rotation, rotation_vector
from igris.umi_walklow_replay.stream import PoseStream, Recorded

PERIOD = 1. / 15.


def pose(x, angle=0.):
    out = np.eye(4); out[0, 3] = x; out[:3, :3] = axis_rotation(np.array([0., 0., 1.]), angle)
    return out


def test_first_sample_is_the_target_at_once():
    stream = PoseStream(PERIOD)
    assert stream.target(0.) is None                              # nothing has arrived: nothing to follow
    stream.push(0., pose(.1), pose(.2), .3)
    left, right, yaw = stream.target(0.)
    assert np.allclose(left, pose(.1)) and np.allclose(right, pose(.2)) and yaw == .3


def test_target_goes_to_the_newest_sample_over_one_period_and_no_further():
    stream = PoseStream(PERIOD)
    stream.push(0., pose(0.), pose(0.), 0.)
    stream.push(PERIOD, pose(.03, .2), pose(-.03), .1)
    left, right, yaw = stream.target(PERIOD)
    assert np.allclose(left, pose(0.)) and yaw == 0.              # the new sample has only just arrived
    left, right, yaw = stream.target(1.5 * PERIOD)
    assert np.allclose(left[:3, 3], [.015, 0., 0.]) and np.allclose(rotation_vector(left[:3, :3]), [0., 0., .1])
    assert np.allclose(right[:3, 3], [-.015, 0., 0.]) and np.isclose(yaw, .05)
    for later in (2 * PERIOD, 5 * PERIOD):                        # no sample after it: the target waits there
        left, _, yaw = stream.target(later)
        assert np.allclose(left, pose(.03, .2)) and np.isclose(yaw, .1)
    assert np.allclose(stream.newest[0], pose(.03, .2))


def test_a_late_or_early_sample_does_not_make_the_target_jump():
    stream = PoseStream(PERIOD)
    stream.push(0., pose(0.), pose(0.), 0.)
    stream.push(PERIOD, pose(.03), pose(0.), 0.)
    early = 1.4 * PERIOD                                          # the next sample comes before the last was reached
    before = stream.target(early)[0]
    stream.push(early, pose(.06), pose(0.), 0.)
    assert np.allclose(stream.target(early)[0], before)
    assert np.allclose(stream.target(early + PERIOD)[0], pose(.06))


def test_recorded_stream_hands_out_each_sample_once_when_its_time_has_come():
    stamps = np.arange(4) * PERIOD
    source = dict(t=stamps, left=np.array([pose(k * .01) for k in range(4)]), right=np.array([pose(0.)] * 4),
                  head_yaw=np.arange(4) * .1)
    feed = Recorded(source)
    assert [round(sample[2], 1) for sample in feed.due(0.)] == [0.]
    assert list(feed.due(.5 * PERIOD)) == []
    assert [round(sample[2], 1) for sample in feed.due(2.2 * PERIOD)] == [.1, .2] and not feed.finished
    assert [round(sample[2], 1) for sample in feed.due(9.)] == [.3] and feed.finished
    late = Recorded(source, latency=.2)                           # e.g. the time a policy takes to answer
    assert list(late.due(.19)) == [] and len(list(late.due(.2 + 1e-6))) == 1


# ---- with the IK ------------------------------------------------------------------------------------------------

mink = pytest.importorskip('mink')

from igris.umi_walklow_replay import stream as stream_module          # noqa: E402
from igris.umi_walklow_replay.ik import POSTURE                       # noqa: E402
from igris.umi_walklow_replay.mink_ik import MinkIK                   # noqa: E402


@pytest.fixture(scope='module')
def model():
    return UpperBody.from_mjcf()


def swaying(model, seconds=3., rate=15.):
    """Hand poses of a joint path the robot can follow (both elbows and the waist yaw swinging, at most 0.5 rad/s)."""
    stamps = np.arange(0., seconds, 1. / rate)
    q = np.zeros((len(stamps), len(JOINTS))); q[:, ACTIVE] = POSTURE
    swing = .25 * (1 - np.cos(2 * np.pi * stamps / seconds))
    q[:, ACTIVE[6]] -= swing; q[:, ACTIVE[13]] -= swing; q[:, ACTIVE[0]] += .5 * swing
    hands = [model.fk(row) for row in q]
    return dict(t=stamps, left=np.array([h[0] for h in hands]), right=np.array([h[1] for h in hands]), head_yaw=-q[:, 0])


def test_rollout_follows_the_stream_one_period_behind(model):
    source = swaying(model)
    run = stream_module.rollout(source, MinkIK(model), model, period=PERIOD)
    assert np.allclose(np.diff(run['t']), .01) and run['t'][-1] >= source['t'][-1] + PERIOD - 1e-9
    report = stream_module.summary(run, source, PERIOD)
    assert report['hand_error_p99_m'] < 3e-3                      # against where the stream was one period ago
    assert report['hand_error_now_p99_m'] > report['hand_error_p99_m']      # against the source's present: the delay shows
    assert report['joint_speed_max_rad_s'] < 1.5 and report['relaxed_ticks'] == 0
    final = model.fk(run['q'][-1])[0]
    assert np.linalg.norm(final[:3, 3] - source['left'][-1][:3, 3]) < 2e-3


def test_commands_never_depend_on_samples_that_have_not_arrived(model):
    """Cut the stream short: every command up to the last sample that both versions delivered is bit-identical."""
    source = swaying(model)
    keep = 20
    short = {key: value[:keep] for key, value in source.items()}
    whole = stream_module.rollout(source, MinkIK(model), model, period=PERIOD)
    cut = stream_module.rollout(short, MinkIK(model), model, period=PERIOD)
    shared = int(np.floor(source['t'][keep] * 100. - 1e-9)) + 1                # ticks before sample ``keep`` arrives
    assert np.array_equal(whole['q'][:shared], cut['q'][:shared])
    assert not np.array_equal(whole['q'][shared + 5], cut['q'][min(shared + 5, len(cut['q']) - 1)])


def test_latency_delays_the_motion_and_changes_nothing_else(model):
    source = swaying(model)
    prompt = stream_module.rollout(source, MinkIK(model), model, period=PERIOD)
    late = stream_module.rollout(source, MinkIK(model), model, period=PERIOD, latency=.1)
    assert np.allclose(late['q'][10:len(prompt['q'])], prompt['q'][:len(prompt['q']) - 10], atol=1e-9)


def test_base_height_is_added_to_the_targets(model):
    source = swaying(model, seconds=1.)
    lowered = dict(source, left=source['left'].copy(), right=source['right'].copy())
    lowered['left'][:, 2, 3] -= .03; lowered['right'][:, 2, 3] -= .03
    plain = stream_module.rollout(source, MinkIK(model), model, period=PERIOD)
    lifted = stream_module.rollout(lowered, MinkIK(model), model, period=PERIOD, base_offset_z=.03)
    assert np.allclose(plain['q'], lifted['q'], atol=1e-9)
