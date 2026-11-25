import numpy as np
import pytest

from igris.umi_walklow_replay.ik import IK, POSTURE, project_polygon
from igris.umi_walklow_replay.kinematics import ACTIVE, JOINTS, SHOULDERS, UpperBody, coupled_halfspaces, rotation_vector


@pytest.fixture(scope='module')
def model():
    return UpperBody.from_mjcf()


def posture_q():
    q = np.zeros(len(JOINTS)); q[ACTIVE] = POSTURE
    return q


def test_reaches_poses_of_feasible_configurations(model):
    ik = IK(model)
    rng = np.random.default_rng(3)
    for _ in range(8):
        goal = posture_q()
        goal[ACTIVE] += rng.uniform(-.35, .35, len(ACTIVE))
        goal[ACTIVE] = np.clip(goal[ACTIVE], ik.lower + .05, ik.upper - .05)
        goal[[JOINTS.index(n) for n in ('waist_roll', 'l_wrist_roll', 'l_wrist_pitch', 'r_wrist_roll', 'r_wrist_pitch')]] *= .3
        left, right = model.fk(goal)
        q, info = ik.converge(posture_q(), left, right, ticks=600)
        assert info['position_error'].max() < 1e-3
        assert info['rotation_error'].max() < np.deg2rad(.5)


def test_single_step_respects_velocity_and_limits(model):
    ik = IK(model, dt=.01, velocity_limit=2., waist_velocity_limit=1.2)
    goal = posture_q(); goal[ACTIVE] += .5
    goal[ACTIVE] = np.clip(goal[ACTIVE], ik.lower, ik.upper)
    left, right = model.fk(goal)
    start = posture_q()
    q, _ = ik.step(start, left, right)
    change = abs(q - start)[ACTIVE]
    assert np.all(change <= ik.velocity * ik.dt + 1e-9)
    assert np.array_equal(q[3:15], start[3:15]) and np.array_equal(q[29:], start[29:])   # legs and neck untouched


def test_unreachable_target_stays_bounded(model):
    ik = IK(model)
    left, right = model.fk(posture_q())
    left = left.copy(); right = right.copy()
    left[0, 3] += 1.; right[0, 3] += 1.
    q, info = ik.converge(posture_q(), left, right, ticks=500)
    assert np.isfinite(q).all()
    assert np.all(q[ACTIVE] >= ik.lower - 1e-9) and np.all(q[ACTIVE] <= ik.upper + 1e-9)
    for joints, normal, offset in coupled_halfspaces():
        assert np.all(normal @ q[joints] <= offset + 1e-9)
    assert info['position_error'].min() > .3 and info['saturated']


def test_waist_yaw_reference_is_followed_when_hands_allow(model):
    ik = IK(model)
    q = posture_q()
    # Hands rigidly attached to a trunk turned by 0.5 rad: the reference picks that solution.
    turned = q.copy(); turned[0] = .5
    left, right = model.fk(turned)
    solved, info = ik.converge(q, left, right, waist_yaw_reference=.5, ticks=800)
    assert info['position_error'].max() < 1e-3
    assert abs(solved[0] - .5) < .05


def test_polygon_projection():
    joints, normal, offset = coupled_halfspaces()[0]
    inside = project_polygon([0., 0.], normal, offset)
    assert np.allclose(inside, 0)
    outside = project_polygon([0., 1.], normal, offset)
    assert np.all(normal @ outside <= offset + 1e-9) and outside[1] < .3


def test_trunk_lean_range_can_be_narrowed(model):
    ik = IK(model, waist_roll_limit=.15, waist_pitch_limits=(-.15, .08))
    left, right = model.fk(posture_q())
    left = left.copy(); right = right.copy()
    left[:3, 3] += [.4, 0., -.5]; right[:3, 3] += [.4, 0., -.5]        # far and low: asks for a deep lean
    q, _ = ik.converge(posture_q(), left, right, ticks=500)
    assert -.15 - 1e-9 <= q[2] <= .08 + 1e-9 and abs(q[1]) <= .15 + 1e-9


def test_elbow_targets_move_the_elbows_without_moving_the_hands(model):
    ik = IK(model)
    start = posture_q()
    left, right = model.fk(start)
    _, elbows, _, _ = model.arm_frames(start)
    # Ask both elbows 6 cm further out to the sides; the hands must stay where they are.
    wanted = np.array([elbows[0][:3, 3] + [0, .06, 0], elbows[1][:3, 3] - [0, .06, 0]])
    q, info = ik.converge(start, left, right, elbow_targets=wanted, ticks=600)
    assert info['position_error'].max() < 1e-3 and info['rotation_error'].max() < np.deg2rad(.5)
    moved = model.arm_frames(q)[1]
    assert moved[0][1, 3] - elbows[0][1, 3] > .03 and elbows[1][1, 3] - moved[1][1, 3] > .03
    assert np.allclose(q[:3], start[:3], atol=.02)          # the trunk is not used to place the elbows
    assert info['elbow_error'].max() < np.linalg.norm([0, .06, 0])


def test_keep_out_box_stops_the_swivel_and_pushes_the_elbow_away(model):
    from igris.umi_walklow_replay.ik import ELBOW_KEEP_OUT
    ik = IK(model)
    start = posture_q()
    left, right = model.fk(start)
    _, elbows, hands, _ = model.arm_frames(start)
    wanted = np.array([elbows[0][:3, 3] + [0, .08, 0], elbows[1][:3, 3] - [0, .08, 0]])
    half = np.array([.135, .09, .075])

    def box(centre):
        return [(np.asarray(centre, dtype=float), np.eye(3), half)]

    def gap(q, boxes):
        return min(ik._arm_gap(q, side, hands[side][:3, 3], boxes) for side in range(2))

    free, _ = ik.converge(start, left, right, elbow_targets=wanted, ticks=600)
    # A box whose face is 10 cm outside the resting left elbow: the free swivel would end inside the margin.
    outside = box(elbows[0][:3, 3] + [0, .10 + half[1], 0])
    assert gap(free, outside) < ELBOW_KEEP_OUT - .01
    guarded, info = ik.converge(start, left, right, elbow_targets=wanted, keep_out=outside, ticks=600)
    assert gap(guarded, outside) >= ELBOW_KEEP_OUT - 2e-3
    assert info['position_error'].max() < 1e-3                       # the hands are not disturbed
    assert guarded[22:29] == pytest.approx(free[22:29], abs=2e-2)    # the other arm still follows its target
    # A box already 4 cm from the elbow: the arm swivels away from it although the target is on the far side.
    close = box(elbows[0][:3, 3] + [0, .04 + half[1], 0])
    before = gap(start, close)
    escaped, info = ik.converge(start, left, right, elbow_targets=wanted, keep_out=close, ticks=600)
    assert gap(escaped, close) > before + .02 and info['position_error'].max() < 1e-3


def test_partial_kinematics_match_the_full_walk(model):
    rng = np.random.default_rng(3)
    for _ in range(5):
        q = np.zeros(31); q[ACTIVE] = rng.uniform(model.lower[ACTIVE], model.upper[ACTIVE])
        torso, elbows, _, _ = model.arm_frames(q)
        assert np.allclose(model.torso(q), torso)
        for side in range(2):
            shoulder, elbow = model.upper_arm(q, side)
            assert np.allclose(elbow, elbows[side][:3, 3])
            assert np.allclose(shoulder, torso[:3, :3] @ SHOULDERS[side] + torso[:3, 3])


def test_swivel_stops_short_of_joint_limits_and_backs_out_of_them(model):
    from igris.umi_walklow_replay.ik import ELBOW_LIMIT_MARGIN
    ik = IK(model)
    start = posture_q()
    left, right = model.fk(start)
    _, elbows, _, _ = model.arm_frames(start)
    left_arm = slice(3, 10)

    def room(q):
        active = q[ACTIVE][left_arm]
        return np.minimum(active - ik.lower[left_arm], ik.upper[left_arm] - active).min()

    assert room(start) > ELBOW_LIMIT_MARGIN
    # An elbow target far out to the side would wind the arm into a limit; the swivel stops at the margin.
    far = np.array([elbows[0][:3, 3] + [0, .40, .25], elbows[1][:3, 3]])
    q, info = ik.converge(start, left, right, elbow_targets=far, ticks=1500)
    assert room(q) > ELBOW_LIMIT_MARGIN - .02 and info['position_error'].max() < 1e-3
    assert np.linalg.norm(model.arm_frames(q)[1][0][:3, 3] - elbows[0][:3, 3]) > .03     # it did follow as far as allowed
    # Pushed along its swivel until a joint is 3 cm from its bound, with the elbow target where the
    # elbow then is: the arm swivels back out although the prior asks for nothing.
    def swivel_direction(configuration):
        return np.linalg.svd(model.fk_jacobian(configuration)[2][:6, left_arm])[2][-1]

    reference = swivel_direction(q)
    for sign in (1., -1.):
        trial = q.copy()
        for _ in range(400):
            direction = swivel_direction(trial)
            trial[ACTIVE[3:10]] += sign * .002 * direction * np.sign(direction @ reference)
            if room(trial) < .02:
                break
        if room(trial) < .03:
            break
    assert room(trial) < .03
    here = np.array([elbow[:3, 3] for elbow in model.arm_frames(trial)[1]])
    out, info = ik.converge(trial, *model.fk(trial), elbow_targets=here, ticks=600)
    assert room(out) > room(trial) + .03 and info['position_error'].max() < 1e-3


def test_pinned_waist_yaw_follows_its_reference_and_the_hands_stay_on_target(model):
    start = posture_q()
    left, right = model.fk(start)
    pinned, info = IK(model, pin_waist_yaw=True).converge(start, left, right, waist_yaw_reference=.4, ticks=600)
    assert pinned[0] == pytest.approx(.4, abs=1e-6)                   # commanded directly, not traded against the arms
    assert info['position_error'].max() < 1e-3 and info['rotation_error'].max() < np.deg2rad(.5)
    one, _ = IK(model, pin_waist_yaw=True).step(start, left, right, waist_yaw_reference=.4)
    assert abs(one[0] - start[0]) <= 1.2 * .01 + 1e-9                 # still bound by the waist speed


def test_free_toe_in_keeps_the_pivot_and_stays_inside_its_range(model):
    pivot = np.array([[0., -.030, .035], [0., .030, .035]])
    low, high = 0., np.deg2rad(25.)
    ik = IK(model, toe_range=(low, high), toe_pivot=pivot)
    start = posture_q()
    left, right = model.fk(start)
    q, info = ik.converge(start, left, right, ticks=800)
    assert info['position_error'].max() < 1e-3 and info['rotation_error'].max() < np.deg2rad(.5)
    assert np.all(info['toe'] > low - 1e-3) and np.all(info['toe'] < high + 1e-3)
    assert np.all(info['toe'] > np.deg2rad(2.))                         # the freedom is used: drawn toward the preferred angle
    for pose, target, point in zip(model.fk(q), (left, right), pivot):
        assert np.linalg.norm((pose @ np.r_[point, 1.] - target @ np.r_[point, 1.])[:3]) < 1e-3     # the palm point stays
        turn = rotation_vector(target[:3, :3].T @ pose[:3, :3])
        assert np.linalg.norm(turn[1:]) < np.deg2rad(.5) and abs(turn[0]) > np.deg2rad(2.)          # a turn about x only
    # A target turned far outside the range is followed only up to the bound.
    c, s = np.cos(np.deg2rad(60.)), np.sin(np.deg2rad(60.))
    spin = np.eye(4); spin[1:3, 1:3] = [[c, -s], [s, c]]                 # 60 deg toe-out for the left hand
    there = np.eye(4); there[:3, 3] = pivot[0]; back = np.eye(4); back[:3, 3] = -pivot[0]
    q2, info2 = ik.converge(q, left @ there @ spin @ back, right, ticks=800)
    assert low - np.deg2rad(1.) < info2['toe'][0] < high + np.deg2rad(1.)
    with pytest.raises(ValueError):
        IK(model, toe_range=(low, high))


def test_toe_hold_fixes_the_free_toe_in_at_its_preferred_angle(model):
    pivot = np.array([[0., -.030, .035], [0., .030, .035]])
    preferred = np.deg2rad(12.)
    ik = IK(model, toe_range=(0., np.deg2rad(25.)), toe_preference=preferred, toe_pivot=pivot)
    start = posture_q()
    left, right = model.fk(start)
    q = start
    for _ in range(800):
        q, info = ik.step(q, left, right, toe_hold=1.)
    assert np.allclose(info['toe'], preferred, atol=np.deg2rad(.5)) and info['position_error'].max() < 1e-3
