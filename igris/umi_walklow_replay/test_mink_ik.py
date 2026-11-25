"""The mink IK must agree with the numpy kinematics and keep its bounds.  Needs mink (see README)."""
import numpy as np
import pytest

pytest.importorskip('mink')
pytest.importorskip('yaml')

from igris.umi_walklow_replay import collision                      # noqa: E402
from igris.umi_walklow_replay.ik import IK, POSTURE                 # noqa: E402
from igris.umi_walklow_replay.kinematics import ACTIVE, JOINTS, UpperBody   # noqa: E402
from igris.umi_walklow_replay.mink_ik import COLLISION_GAP_M, MinkIK        # noqa: E402


@pytest.fixture(scope='module')
def model():
    return UpperBody.from_mjcf()


@pytest.fixture(scope='module')
def spheres(tmp_path_factory):
    return collision.SelfCollision(collision.export(tmp_path_factory.mktemp('spheres') / 'spheres.npz'))


def posture_q():
    q = np.zeros(len(JOINTS)); q[ACTIVE] = POSTURE
    return q


def test_reduced_model_has_the_same_kinematics(model):
    solver = MinkIK(model)
    rng = np.random.default_rng(1)
    for _ in range(5):
        q31 = np.zeros(31); q31[ACTIVE] = rng.uniform(solver.lower, solver.upper)
        q = np.empty(17); q[solver.columns] = q31[ACTIVE]
        solver.configuration.update(q)
        for name, pose in zip(('end_effector_left', 'end_effector_right'), model.fk(q31)):
            assert np.allclose(solver.configuration.get_transform_frame_to_world(name, 'body').as_matrix(), pose, atol=1e-9)


def test_reaches_a_target_like_the_numpy_solver(model):
    start = posture_q()
    left, right = model.fk(start)
    left = left.copy(); left[:3, 3] += [.05, .03, .08]
    q, info = MinkIK(model).converge(start, left, right, ticks=400)
    reference, _ = IK(model).converge(start, left, right, ticks=400)
    assert info['position_error'].max() < 2e-4 and info['rotation_error'].max() < np.deg2rad(.1)
    assert np.abs(q - reference)[ACTIVE].max() < .05 and info['relaxed'] == 0


def test_single_step_respects_speed_and_joint_limits(model):
    solver = MinkIK(model)
    start = posture_q()
    left, right = model.fk(start)
    left = left.copy(); left[:3, 3] += [.3, .2, .3]                    # far away: wants the largest step
    q, info = solver.step(start, left, right)
    assert np.all(np.abs(q - start)[ACTIVE] <= solver.velocity * solver.dt + 1e-9) and info['saturated']
    for _ in range(600):
        q, info = solver.step(q, left, right)
    assert np.all(q[ACTIVE] >= solver.lower - 1e-9) and np.all(q[ACTIVE] <= solver.upper + 1e-9)
    assert np.isfinite(q).all()


def test_self_collision_bound_keeps_the_arm_off_the_trunk(model, spheres):
    start = posture_q()
    left, right = model.fk(start)
    across = right.copy(); across[:3, 3] = [.18, .12, .22]              # right hand asked across the chest, into the trunk
    free, _ = MinkIK(model).converge(start, left, across, ticks=500)
    guarded, info = MinkIK(model, self_collision=spheres).converge(start, left, across, ticks=500)
    assert spheres.clearance(free)[0] < 0.                              # without the bound the spheres overlap
    assert spheres.clearance(guarded)[0] > COLLISION_GAP_M - 1.5e-3 and info['relaxed'] == 0
    assert np.isclose(info['gap'], spheres.clearance(guarded)[0], atol=2e-3)


def test_clearance_matches_a_direct_comparison_of_all_spheres(spheres):
    q = posture_q(); q[ACTIVE[10]] = -.6; q[ACTIVE[13]] = -1.6           # right arm bent toward the trunk
    gap, link_a, link_b = spheres.clearance(q)
    poses = spheres.tree.poses(q)
    best = np.inf
    for a, b in spheres.pairs:
        pa = spheres.centres[a] @ poses[spheres.bodies[a]][:3, :3].T + poses[spheres.bodies[a]][:3, 3]
        pb = spheres.centres[b] @ poses[spheres.bodies[b]][:3, :3].T + poses[spheres.bodies[b]][:3, 3]
        best = min(best, (np.linalg.norm(pa[:, None] - pb[None], axis=2) - spheres.radii[a][:, None] - spheres.radii[b][None]).min())
    assert np.isclose(gap, min(best, spheres.cutoff), atol=1e-9) and (link_a is None) == (best >= spheres.cutoff)


def elbow_points(model, q):
    return np.array([elbow[:3, 3] for elbow in model.arm_frames(q)[1]])


def other_elbows(model, q, left, right):
    """Elbow places of another arm posture with the same hand poses: somewhere the elbows can go with the hands held."""
    reference = POSTURE[3:].copy(); reference[1] += .6; reference[8] -= .6          # both shoulders rolled outward
    other, info = MinkIK(model, arm_posture_cost=.3).converge(q, left, right, posture=reference, ticks=1500)
    assert info['position_error'].max() < 5e-3                         # the strong posture pull costs the hands 3 mm
    return elbow_points(model, other)


def test_elbow_follows_its_target_while_the_hands_stay(model):
    start = posture_q()
    left, right = model.fk(start)
    wanted = elbow_points(model, start)
    wanted[0] = other_elbows(model, start, left, right)[0]              # only the left elbow is asked to move
    before = np.linalg.norm(wanted[0] - elbow_points(model, start)[0])
    q, info = MinkIK(model).converge(start, left, right, elbow_targets=wanted, ticks=800)
    after = np.linalg.norm(wanted - elbow_points(model, q), axis=1)
    assert before > .02 and after[0] < .2 * before and after[1] < .01
    assert info['position_error'].max() < 2e-3 and info['rotation_error'].max() < np.deg2rad(1.)
    assert np.allclose(info['elbow_error'], after, atol=1e-6) and info['relaxed'] == 0
    assert np.all(info['elbow_trust'] > .9)                              # a reachable target is believed


def test_elbow_target_far_from_anything_the_arm_can_do_is_not_chased(model):
    """A prediction the arm cannot get near comes from hand poses the prior was not trained on (in simulation: the
    third box carried high, elbow asked 6-18 cm from where it could go, the arm swivelled to 114 deg and its joint
    limits).  Such a target is left alone instead of being followed as far as the swivel goes."""
    start = posture_q()
    left, right = model.fk(start)
    wanted = elbow_points(model, start)
    wanted[1] += [0., -.30, .20]                                        # right elbow asked 36 cm out and up
    q, info = MinkIK(model).converge(start, left, right, elbow_targets=wanted, ticks=600)
    assert info['position_error'].max() < 1e-3 and np.isfinite(q).all()
    assert np.linalg.norm(elbow_points(model, q) - elbow_points(model, start), axis=1).max() < .01


def test_elbow_moves_at_a_bounded_pace_toward_a_new_target(model):
    """The prior's prediction can step; the arm must not jump after it within one tick."""
    start = posture_q()
    left, right = model.fk(start)
    wanted = elbow_points(model, start); wanted[0] = other_elbows(model, start, left, right)[0]
    q, _ = MinkIK(model).step(start, left, right, elbow_targets=wanted)
    moved = np.linalg.norm(elbow_points(model, q)[0] - elbow_points(model, start)[0])
    assert 0 < moved < .2 * np.linalg.norm(wanted[0] - elbow_points(model, start)[0])


def test_trunk_is_not_moved_to_bring_an_elbow_to_its_target(model):
    """The prior places the elbow relative to the torso; turning the waist cannot bring it nearer and only moves the body."""
    start = posture_q()
    left, right = model.fk(start)
    wanted = elbow_points(model, start)
    wanted[1] += [0., -.30, .20]                                        # far out of reach
    q, _ = MinkIK(model).converge(start, left, right, elbow_targets=wanted, ticks=600)
    assert np.abs(q[:3] - start[:3]).max() < .01


def test_untrusted_elbow_prediction_falls_back_to_the_posture_reference(model):
    """Where the prediction cannot be trusted the arm still needs a posture to lean on: the one it would have had
    without the prior (the looked-up teleop posture in the replay)."""
    start = posture_q()
    left, right = model.fk(start)
    reference = POSTURE[3:].copy(); reference[1] += .4; reference[8] -= .4          # some other preferred posture
    far = elbow_points(model, start); far += [[0., .30, .20], [0., -.30, .20]]       # both predictions out of reach
    plain, _ = MinkIK(model, arm_posture_cost=.03).converge(start, left, right, posture=reference, ticks=1500)
    hybrid, info = MinkIK(model, arm_posture_cost=.03).converge(start, left, right, posture=reference, elbow_targets=far, ticks=1500)
    assert np.abs(plain - start)[ACTIVE].max() > .05                                 # the reference does move the arms
    assert np.abs(hybrid - plain)[ACTIVE].max() < .02 and info['position_error'].max() < 1e-3
    assert np.all(info['elbow_trust'] == 0.)


# ---- a waist that is given from outside (a policy's third "end effector") --------------------------------------

def test_pinned_waist_goes_to_its_target_at_its_speed_limit_whatever_the_hands_ask(model):
    solver = MinkIK(model, velocity_limit=1., waist_velocity_limit=1.)
    start = posture_q()
    left, right = model.fk(start)                                   # hand targets that want the waist to stay
    waist = np.array([.3, .05, -.08])
    q, info = solver.step(start, left, right, waist_target=waist, waist_pinned=True)
    assert np.allclose(q[:3] - start[:3], np.sign(waist - start[:3]) * 1. * solver.dt)       # one full step each
    for _ in range(60):
        q, info = solver.step(q, left, right, waist_target=waist, waist_pinned=True)
    assert np.allclose(q[:3], waist, atol=1e-12)
    # The arms made up for the turned and leaning chest: the hands are where they were asked to be.
    assert info['position_error'].max() < 2e-3 and info['relaxed'] == 0


def test_pinned_waist_is_not_pulled_over_by_a_target_out_of_reach(model):
    start = posture_q()
    left, right = model.fk(start)
    far_left = left.copy(); far_left[:3, 3] += [.25, 0., -.35]       # low and far ahead: the free IK leans for it
    far_right = right.copy(); far_right[:3, 3] += [.25, 0., -.35]
    free = MinkIK(model, velocity_limit=1., waist_velocity_limit=1.)
    q_free = start
    for _ in range(300):
        q_free, _ = free.step(q_free, far_left, far_right)
    pinned = MinkIK(model, velocity_limit=1., waist_velocity_limit=1.)
    q_pinned = start
    for _ in range(300):
        q_pinned, info = pinned.step(q_pinned, far_left, far_right, waist_target=start[:3], waist_pinned=True)
    assert abs(q_free[2] - start[2]) > .05                          # the free solver bent the trunk forward
    assert np.allclose(q_pinned[:3], start[:3], atol=1e-12)         # the pinned one did not move it at all
    assert info['position_error'].max() > 1e-3                      # and left the hands short instead


def test_waist_target_as_reference_is_followed_where_the_hands_allow_it(model):
    solver = MinkIK(model)
    start = posture_q()
    left, right = model.fk(start)
    turned = start.copy(); turned[0] = .25
    left_turned, right_turned = model.fk(turned)                    # the same hands on a chest turned by 0.25 rad
    q = start
    for _ in range(800):
        q, info = solver.step(q, left_turned, right_turned, waist_target=np.array([.25, 0., 0.]))
    assert abs(q[0] - .25) < .02 and info['position_error'].max() < 1e-3
    q = start                                                        # without it the waist turns less: the arms take a share
    for _ in range(800):
        q, _ = solver.step(q, left_turned, right_turned)
    assert abs(q[0] - .25) > .03
