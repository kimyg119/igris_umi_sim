import numpy as np

from igris.umi_walklow_replay.analyze import box_outcomes, tilt_deg, yaw


def recorded(carries, goals, drop=None, fps=15., rate=100.):
    """Poses of three stacked boxes that are carried one by one to ``goals``."""
    end = carries[-1][1] / fps + 4.
    sim = np.arange(0., end, 1. / rate)
    parcels = np.zeros((len(sim), 3, 7)); parcels[:, :, 3] = 1.
    for level in range(3):
        parcels[:, level, :3] = [.4, -.3, .8 + level * .15 + .075]
    for order, ((first, last), goal) in enumerate(zip(carries, goals)):
        index = 2 - order
        start, stop = first / fps, last / fps
        for i, time in enumerate(sim):
            if time < start:
                continue
            u = min(1., (time - start) / (stop - start))
            arc = .25 * np.sin(np.pi * u)
            parcels[i, index, :3] = parcels[0, index, :3] * (1 - u) + np.asarray(goal) * u + [0, 0, arc]
            if drop == order and time > start + .5 * (stop - start):
                parcels[i, index, :3] = [.3, 0., .075]; parcels[i, index, 3:] = [np.sqrt(.5), np.sqrt(.5), 0., 0.]
    return sim, parcels


CARRIES = [[30, 120], [180, 270], [330, 420]]
GOALS = [np.array([.15, .4, .8 + (level + .5) * .15]) for level in range(3)]


def test_all_boxes_placed():
    sim, parcels = recorded(CARRIES, GOALS)
    outcomes = box_outcomes(sim, parcels, CARRIES, GOALS)
    assert [o['placed'] for o in outcomes] == [True, True, True]
    assert all(o['lifted'] and o['max_rise_m'] > .03 for o in outcomes)


def test_dropped_box_is_reported_not_placed():
    sim, parcels = recorded(CARRIES, GOALS, drop=1)
    outcomes = box_outcomes(sim, parcels, CARRIES, GOALS)
    assert [o['placed'] for o in outcomes] == [True, False, True]
    assert outcomes[1]['final_tilt_deg'] > 80 and outcomes[1]['final_z_miss_m'] < -.5


def test_log_that_ends_before_a_carry_reports_it_unreached():
    sim, parcels = recorded(CARRIES, GOALS)
    cut = sim < CARRIES[1][0] / 15.
    outcomes = box_outcomes(sim[cut], parcels[cut], CARRIES, GOALS)
    assert outcomes[0]['placed'] and outcomes[1] == dict(box=1, reached=False) and not outcomes[2]['reached']


def test_angles():
    assert tilt_deg(np.array([[1., 0., 0., 0.]]))[0] == 0.
    half = np.deg2rad(30) / 2
    assert np.isclose(tilt_deg(np.array([[np.cos(half), np.sin(half), 0., 0.]]))[0], 30.)
    assert np.isclose(yaw(np.array([[np.cos(half), 0., 0., np.sin(half)]]))[0], np.deg2rad(30))


def test_box_knocked_over_after_set_down_is_not_placed():
    sim, parcels = recorded(CARRIES, GOALS)
    fallen = sim > CARRIES[2][1] / 15. + 2.5            # third box tips after the hands have left
    parcels[fallen, 0, :3] = [.3, .3, .9]; parcels[fallen, 0, 3:] = [np.sqrt(.5), np.sqrt(.5), 0., 0.]
    outcomes = box_outcomes(sim, parcels, CARRIES, GOALS)
    assert outcomes[2]['set_down_ok'] and not outcomes[2]['placed']
    assert outcomes[0]['placed'] and outcomes[1]['placed']


def test_stacked_but_off_the_recorded_point():
    goals = [g + np.array([.13, 0., 0.]) for g in GOALS]            # a 13 cm sideways miss of every set-down point
    sim, parcels = recorded(CARRIES, GOALS)
    outcomes = box_outcomes(sim, parcels, CARRIES, goals)
    assert all(o['stacked'] and not o['placed'] for o in outcomes)
