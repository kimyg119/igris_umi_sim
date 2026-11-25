import numpy as np

from igris.umi_walklow_replay.governor import Governor, blend_pose
from igris.umi_walklow_replay.kinematics import ACTIVE, JOINTS, axis_rotation, rotation_vector


def ramp(governor, wanted_speed, ticks=400, dt=.01):
    """Drive one joint toward a target that always sits ``wanted_speed * dt`` ahead; return its steps and the path times."""
    q = np.zeros(len(JOINTS)); steps = []; times = []
    for _ in range(ticks):
        governor.ahead(dt)
        wanted = q.copy(); wanted[ACTIVE[5]] += wanted_speed * dt
        new = governor.settle(q, wanted, dt)
        steps.append(new[ACTIVE[5]] - q[ACTIVE[5]]); times.append(governor.tau); q = new
    return np.array(steps), np.array(times)


def test_step_is_cut_to_the_speed_limit_and_the_path_slows_by_the_same_share():
    stamps = np.arange(0., 20., .01)
    governor = Governor(stamps, speed=1., acceleration=1e9, rise=1e9)
    steps, times = ramp(governor, 3.)                      # the path asks for 3 rad/s
    assert np.allclose(steps[5:], 1. * .01)                # the joint moves at 1 rad/s
    assert np.allclose(np.diff(times)[5:], .01 / 3.)       # and the path advances at a third of real time
    assert np.isclose(governor.rate, 1 / 3.)
    slow, times = ramp(Governor(stamps, speed=1., acceleration=1e9, rise=1e9), .4)
    assert np.allclose(slow[5:], .4 * .01) and np.allclose(np.diff(times)[5:], .01)        # within the limit: untouched


def test_speeding_up_is_limited_and_slowing_down_is_not():
    stamps = np.arange(0., 20., .01)
    governor = Governor(stamps, speed=1., acceleration=4., rise=1e9)
    steps, _ = ramp(governor, 3., ticks=60)
    speed = steps / .01
    assert np.all(np.diff(speed) <= 4. * .01 + 1e-9)       # at most 4 rad/s2
    assert speed[10] < .5 and np.isclose(speed[-1], 1.)    # still climbing after 0.1 s; at the speed limit after 0.25 s
    q = np.zeros(len(JOINTS)); governor.ahead(.01)
    stopped = governor.settle(q, q, .01)                   # the path stops: so does the joint, at once
    assert np.array_equal(stopped, q)


def test_playback_rate_recovers_gradually():
    stamps = np.arange(0., 20., .01)
    governor = Governor(stamps, speed=1., acceleration=1e9, rise=2.)
    ramp(governor, 5., ticks=50)
    assert governor.rate < .25
    _, times = ramp(governor, .1, ticks=100)               # an easy stretch: the rate climbs by 2 per second
    rates = np.diff(times) / .01
    assert rates[-1] > .99 and np.all(np.diff(rates) <= 2. * .01 + 1e-9)


def test_sampling_interpolates_poses_and_values():
    stamps = np.array([0., 1., 2.])
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[1, :3, 3] = [1., 0., 0.]; poses[1, :3, :3] = axis_rotation(np.array([0., 0., 1.]), 1.)
    poses[2] = poses[1]
    governor = Governor(stamps)
    index, (pose, value) = governor.sample(.25, poses, np.array([0., 4., 8.]))
    assert index == 0 and np.allclose(pose[:3, 3], [.25, 0., 0.]) and np.isclose(value, 1.)
    assert np.allclose(rotation_vector(pose[:3, :3]), [0., 0., .25])
    assert np.allclose(blend_pose(poses[0], poses[1], 1.), poses[1])
    index, (pose,) = governor.sample(5., poses)            # past the end: the last pose
    assert np.allclose(pose, poses[2])


def test_path_keeps_a_minimum_rate_when_the_solver_reverses_every_tick():
    stamps = np.arange(0., 20., .01)

    def chatter(governor):
        q = np.zeros(len(JOINTS)); steps = []
        for k in range(300):                                 # 3 s of a solver that wants 30 mrad one way, then the other
            governor.ahead(.01)
            wanted = q.copy(); wanted[ACTIVE[5]] += .03 * (-1) ** k
            new = governor.settle(q, wanted, .01)
            steps.append(new[ACTIVE[5]] - q[ACTIVE[5]]); q = new
        return np.array(steps)

    stalled = Governor(stamps, speed=1., acceleration=6., floor=0.)
    chatter(stalled)
    assert stalled.tau < .1                                  # without a floor the path stands still
    governor = Governor(stamps, speed=1., acceleration=6., floor=.2)
    steps = chatter(governor)
    assert governor.tau > .5 and np.isclose(governor.rate, .2)             # a fifth of real time
    assert np.all(np.abs(steps) <= 6. * .01 * .01 + 1e-12)                 # the joints still keep their limits
