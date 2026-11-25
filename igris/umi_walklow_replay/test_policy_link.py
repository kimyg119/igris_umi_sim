"""The files between the controller and a policy process, and the plan made of the chunks (policy_link.py)."""
import numpy as np

from igris.umi_walklow_replay import policy_link
from igris.umi_walklow_replay.kinematics import axis_rotation, rotation_vector
from igris.umi_walklow_replay.policy_link import ControllerLink, PairLimiter, Plan

RATE = 15.


def chunk(observed, length=16, shift=0.):
    """Hand targets whose x says which sample they are for (plus ``shift``), waist yaw likewise."""
    hands = np.tile(np.eye(4), (length, 2, 1, 1))
    hands[:, 0, 0, 3] = observed + np.arange(length) + shift
    hands[:, 1, 0, 3] = -(observed + np.arange(length) + shift)
    waist = np.zeros((length, 3)); waist[:, 0] = observed + np.arange(length) + shift
    return hands, waist


def test_state_and_chunk_files_round_trip(tmp_path):
    assert policy_link.read_state(tmp_path) is None and policy_link.read_chunk(tmp_path) is None
    q = np.arange(31.) * .01; command = q + .5
    policy_link.write_state(tmp_path, 7, 31.25, 1.25, RATE, q, command)
    state = policy_link.read_state(tmp_path)
    assert state['sequence'] == 7 and state['sim'] == 31.25 and state['elapsed'] == 1.25 and not state['done']
    assert np.array_equal(state['q_measured'], q) and np.array_equal(state['q_command'], command)
    hands, waist = chunk(12)
    policy_link.write_chunk(tmp_path, 3, 12, hands, waist)
    sequence, observed, read_hands, read_waist = policy_link.read_chunk(tmp_path)
    assert (sequence, observed) == (3, 12) and np.array_equal(read_hands, hands) and np.array_equal(read_waist, waist)
    assert not list(tmp_path.glob('*.tmp'))


def test_a_chunk_is_used_from_delay_samples_after_its_observation_however_early_it_arrives():
    plan = Plan(RATE, delay=2, blend=0)
    assert plan.accept(4, *chunk(4), elapsed=4.2 / RATE) == 6          # arrived within the same sample period
    assert plan.due(5.9 / RATE) == []                                 # samples 0..5: nothing to follow yet
    due = plan.due(7.5 / RATE)
    assert [sample for sample, *_ in due] == [6, 7] and due[0][1][0, 3] == 6. and due[1][3][0] == 7.


def test_a_late_chunk_starts_at_the_sample_after_its_arrival():
    plan = Plan(RATE, delay=2, blend=0)
    assert plan.accept(4, *chunk(4), elapsed=9.4 / RATE) == 10
    assert [sample for sample, *_ in plan.due(10.01 / RATE)] == [10]


def test_new_chunk_is_faded_in_against_the_one_before_it():
    plan = Plan(RATE, delay=2, blend=4)
    plan.accept(0, *chunk(0), elapsed=0.)
    plan.due(5.5 / RATE)                                              # samples 2..5 from the first chunk
    plan.accept(4, *chunk(4, shift=.4), elapsed=4.5 / RATE)           # the second one says 0.4 more everywhere
    got = {sample: (left[0, 3], waist[0]) for sample, left, right, waist in plan.due(12.5 / RATE)}
    for sample, share in ((6, .25), (7, .5), (8, .75), (9, 1.), (10, 1.)):
        assert np.isclose(got[sample][0], sample + .4 * share) and np.isclose(got[sample][1], sample + .4 * share)
    assert plan.skipped == 0 and plan.chunks == 2


def test_samples_handed_out_are_never_taken_back_and_missing_ones_are_counted():
    plan = Plan(RATE, delay=2, blend=0)
    plan.accept(0, *chunk(0, length=6), elapsed=0.)
    assert [sample for sample, *_ in plan.due(8.2 / RATE)] == [2, 3, 4, 5]          # 6, 7, 8: the chunk has ended
    assert plan.skipped == 3
    assert plan.accept(4, *chunk(4), elapsed=8.3 / RATE) == 9
    assert [sample for sample, *_ in plan.due(9.1 / RATE)] == [9]


def test_a_newer_chunk_replaces_what_older_ones_said_about_later_samples():
    plan = Plan(RATE, delay=2, blend=0)
    plan.accept(0, *chunk(0, length=32), elapsed=0.)
    plan.accept(4, *chunk(4, length=8, shift=.4), elapsed=4.1 / RATE)                 # reaches only to sample 11
    got = [sample for sample, *_ in plan.due(20. / RATE)]
    assert got == list(range(2, 12))                                 # nothing of the first chunk after the second ends


def test_controller_link_reads_each_chunk_once(tmp_path):
    link = ControllerLink(tmp_path / 'policy', RATE, delay=2, blend=0)
    link.publish(0, 30., 0., np.zeros(31), np.zeros(31))
    assert policy_link.read_state(tmp_path / 'policy')['rate_hz'] == RATE
    assert link.due(.01) == []
    policy_link.write_chunk(tmp_path / 'policy', 1, 0, *chunk(0))
    assert link.due(1.2 / RATE) == []                               # read during sample 1: usable from sample 2
    first = link.due(2.2 / RATE)
    assert len(first) == 1 and first[0][0][0, 3] == 2. and link.used == 0
    assert link.due(2.3 / RATE) == [] and link.plan.chunks == 1
    report = link.report()
    assert report['chunks'] == 1 and report['skipped_samples'] == 0 and report['arrival_samples_after_observation_max'] == 1.


def hands(x, angle=0., gap=.286):
    left = np.eye(4); left[:3, :3] = axis_rotation(np.array([0., 0., 1.]), angle); left[:3, 3] = [x, .143, 1.]
    right = left.copy(); right[:3, 3] = left[:3, 3] + left[:3, :3] @ [0., -gap, 0.]
    return left, right


def test_targets_within_the_limit_are_handed_out_as_they_are():
    limiter = PairLimiter(hands(0.)[0], step=.026, turn=np.radians(3.6))
    for index in range(1, 6):
        left, right = hands(.02 * index, .05 * index)
        got_left, got_right = limiter.apply(left, right)
        assert np.array_equal(got_left, left) and np.array_equal(got_right, right)
    assert limiter.limited == 0


def test_a_jump_of_the_left_target_is_spread_over_samples_and_the_pair_stays_a_pair():
    """The left target jumps 10 cm and 20 degrees: each handed-out sample moves at most the limit, both hands stay
    286 mm apart and facing as they were given, and the target is reached in the end."""
    limiter = PairLimiter(hands(0.)[0], step=.026, turn=np.radians(3.6))
    left, right = hands(.10, np.radians(20.))
    wanted_pair = np.linalg.inv(left) @ right
    previous = hands(0.)[0]
    for count in range(1, 12):
        got_left, got_right = limiter.apply(left, right)
        assert np.linalg.norm(got_left[:3, 3] - previous[:3, 3]) <= .026 + 1e-9
        assert np.linalg.norm(rotation_vector(previous[:3, :3].T @ got_left[:3, :3])) <= np.radians(3.6) + 1e-9
        assert np.allclose(np.linalg.inv(got_left) @ got_right, wanted_pair, atol=1e-12)
        previous = got_left
        if np.allclose(got_left, left, atol=1e-12):
            break
    assert 4 <= count <= 7 and limiter.limited == count - 1          # 20 degrees at 3.6 per sample: six samples


def test_controller_link_limits_what_it_hands_out(tmp_path):
    limiter = PairLimiter(np.eye(4), step=.026, turn=1.)
    link = ControllerLink(tmp_path / 'policy', RATE, delay=2, blend=0, limiter=limiter)
    policy_link.write_chunk(tmp_path / 'policy', 1, 0, *chunk(0))      # left x = sample index: metres per sample
    link.due(1.2 / RATE)
    out = link.due(4.2 / RATE)                                        # samples 2, 3, 4 want x = 2, 3, 4
    assert np.allclose([left[0, 3] for left, _, _ in out], [.026, .052, .078])
    assert link.report()['limited_samples'] == 3
