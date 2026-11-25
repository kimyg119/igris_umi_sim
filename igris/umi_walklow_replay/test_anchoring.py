import numpy as np

from igris.umi_walklow_replay.anchoring import ReachHold, strain

DT = .01
AHEAD = np.array([.4, 0.])          # middle of the hand targets, 40 cm ahead of the staging pelvis
LIMITS = dict(roll_limit=.15, forward_pitch_limit=-.15)


def retreat(hold, distance, level, start=0., ticks=100):
    """Move the pelvis back by ``distance`` in equal steps at a constant strain; return the offsets passed on."""
    return np.array([hold.update(np.array([start - distance * k / ticks, 0., .01]), AHEAD, level, DT)
                     for k in range(1, ticks + 1)])


def test_retreat_is_followed_while_the_arms_have_reach():
    hold = ReachHold()
    followed = retreat(hold, .05, 0.)
    assert np.allclose(followed[-1], [-.05, 0., .01]) and hold.shortfall == 0.


def test_retreat_under_full_strain_is_left_as_hand_shortfall():
    hold = ReachHold()
    followed = retreat(hold, .05, 1.)
    assert np.allclose(followed[:, 0], 0.)               # the targets do not move away along the reach
    assert np.allclose(followed[:, 2], .01)              # the height is followed as before
    assert np.isclose(hold.shortfall, .05)


def test_sideways_motion_is_followed_even_under_strain():
    hold = ReachHold()
    for k in range(1, 101):
        followed = hold.update(np.array([0., .03 * k / 100, 0.]), AHEAD, 1., DT)
    assert np.allclose(followed, [0., .03, 0.], atol=3e-3)


def test_pelvis_coming_back_is_taken_out_of_the_shortfall_first():
    hold = ReachHold()
    retreat(hold, .05, 1.)
    back = retreat(hold, -.03, 1., start=-.05)           # 3 cm back toward the hands, still strained
    assert np.allclose(back[:, 0], 0.) and np.isclose(hold.shortfall, .02)
    ahead = retreat(hold, -.04, 1., start=-.02)          # on to 2 cm ahead of where it started
    assert np.isclose(hold.shortfall, 0.) and np.isclose(ahead[-1, 0], .02)


def test_turning_the_hands_does_not_turn_an_earlier_pelvis_offset_into_a_shortfall():
    hold = ReachHold()
    pelvis = np.array([.08, 0., 0.])                     # 8 cm toward the first targets, reached without strain
    hold.update(pelvis, pelvis[:2] + [.4, 0.], 0., DT)
    for k in range(1, 101):                              # strained, the hands swing 120 deg round; the pelvis stays
        angle = np.radians(120 * k / 100)
        followed = hold.update(pelvis, pelvis[:2] + .4 * np.array([np.cos(angle), np.sin(angle)]), 1., DT)
    assert hold.shortfall == 0. and np.allclose(followed, pelvis)


def test_a_shortfall_stays_where_the_pelvis_retreated_when_the_hands_turn():
    hold = ReachHold(release=0.)
    retreat(hold, .05, 1.)                               # 5 cm back from hands that are straight ahead
    followed = hold.update(np.array([-.05, 0., .01]), np.array([-.05, .4]), 1., DT)      # the hands are now to the left
    assert np.allclose(followed, [0., 0., .01]) and np.isclose(hold.shortfall, .05)


def test_shortfall_is_given_back_slowly_once_the_strain_is_gone():
    hold = ReachHold(release=.02)                        # 2 cm/s
    retreat(hold, .05, 1.)
    steady = np.array([hold.update(np.array([-.05, 0., 0.]), AHEAD, 0., DT) for _ in range(100)])     # 1 s
    assert np.isclose(hold.shortfall, .03) and np.isclose(steady[-1, 0], -.02)
    assert np.all(np.diff(steady[:, 0]) < 0.) and np.all(np.diff(steady[:, 0]) >= -.02 * DT - 1e-12)
    for _ in range(300):
        last = hold.update(np.array([-.05, 0., 0.]), AHEAD, 0., DT)
    assert hold.shortfall == 0. and np.allclose(last, [-.05, 0., 0.])


def test_partial_strain_follows_part_of_the_retreat():
    hold = ReachHold(release=0.)
    followed = retreat(hold, .04, .75)
    assert np.isclose(hold.shortfall, .03) and np.isclose(followed[-1, 0], -.01)


def test_strain_rises_with_the_trunk_lean_between_the_two_thresholds():
    thresholds = dict(start=.5, full=.8, **LIMITS)
    assert strain(0., -.075, 0., **thresholds) == 0.                    # half the forward lean limit
    assert np.isclose(strain(0., -.0975, 0., **thresholds), .5)         # 65 % of it
    assert strain(0., -.15, 0., **thresholds) == 1.
    assert strain(.12, 0., 0., **thresholds) == 1. and strain(-.12, 0., 0., **thresholds) == 1.   # roll, either side


def test_leaning_back_is_not_strain():
    assert strain(0., .08, 0., **LIMITS) == 0.           # the trunk leans back to carry a box high: reach to spare


def test_strain_rises_with_a_hand_target_the_solver_cannot_reach():
    assert strain(0., 0., .004, **LIMITS) == 0.
    assert strain(0., 0., .03, **LIMITS) == 1.
