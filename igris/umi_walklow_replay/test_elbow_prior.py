"""The numpy FiSTA must reproduce the torch network of igris/hl_ik."""
import numpy as np
import pytest

from igris.umi_walklow_replay import elbow_prior
from igris.umi_walklow_replay.ik import POSTURE
from igris.umi_walklow_replay.kinematics import ACTIVE, JOINTS, SHOULDERS, UpperBody


@pytest.fixture(scope='module')
def exported(tmp_path_factory):
    pytest.importorskip('torch')
    if not elbow_prior.CHECKPOINT.is_file():
        pytest.skip('HL-IK checkpoint is not on this machine')
    return elbow_prior.export(elbow_prior.CHECKPOINT, tmp_path_factory.mktemp('fista') / 'fista.npz')


def random_inputs(network, rng):
    """History and goal near the training distribution, with mixed quaternion signs."""
    history = network.mean[None] + rng.normal(size=(5, 2, 14)) * network.std[None] * .5
    for start in (3, 10):
        history[:, :, start:start + 4] /= np.linalg.norm(history[:, :, start:start + 4], axis=-1, keepdims=True)
    history[2, 0, 3:7] *= -1                     # hemisphere flips must not change the answer
    goal = history[-1, :, :7] + rng.normal(size=(2, 7)) * .02
    goal[:, 3:] /= np.linalg.norm(goal[:, 3:], axis=-1, keepdims=True)
    goal[1, 3:] *= -1
    return history, goal


def test_numpy_matches_torch(exported):
    import torch
    from igris.hl_ik.network import BimanualFiSTA, NetworkConfig, canonicalize, decode
    saved = torch.load(elbow_prior.CHECKPOINT, map_location='cpu', weights_only=False)
    reference = BimanualFiSTA(NetworkConfig(**saved['config']['network'])); reference.load_state_dict(saved['network']); reference.eval()
    network = elbow_prior.FiSTA(exported)
    rng = np.random.default_rng(0)
    for _ in range(20):
        history, goal = random_inputs(network, rng)
        with torch.no_grad():
            h = torch.as_tensor(history[None], dtype=torch.float32); g = torch.as_tensor(goal[None], dtype=torch.float32)
            h, g, _ = canonicalize(h, g)
            expected = decode(reference((h - saved['mean']) / saved['std'], (g - saved['mean'][:, :7]) / saved['std'][:, :7]),
                              saved['mean'], saved['std'])[0].numpy()
        assert np.allclose(network.predict(history, goal), expected, atol=2e-5)


def test_history_runs_at_thirty_hertz_and_targets_are_plausible(exported):
    model = UpperBody.from_mjcf()
    prior = elbow_prior.ElbowPrior(model, elbow_prior.FiSTA(exported))
    q = np.zeros(len(JOINTS)); q[ACTIVE] = POSTURE
    hands = model.fk(q)
    prior.reset(q, 0.)
    first = prior.targets(q, .01, *hands)                 # 10 ms later: no new history sample yet
    assert np.allclose(np.array(prior.history)[0], prior.state(q)) and len(prior.history) == 5
    before = np.array(prior.history).copy()
    prior.targets(q, .02, *hands)
    assert np.array_equal(np.array(prior.history), before)
    second = prior.targets(q, .034, *hands)               # one period has passed
    assert prior.last_time == pytest.approx(elbow_prior.PERIOD_S)
    # Holding still, the predicted elbow stays close to the current elbow.
    _, elbows, _, _ = model.arm_frames(q)
    for side in range(2):
        assert np.linalg.norm(second[side] - elbows[side][:3, 3]) < .03
        assert np.linalg.norm(first[side] - elbows[side][:3, 3]) < 1e-9


def test_seed_elbows_sit_below_and_outside_the_shoulders(exported):
    seed = elbow_prior.seed_elbows(elbow_prior.FiSTA(exported)) - SHOULDERS
    assert seed[0, 1] > .05 and seed[1, 1] < -.05 and np.all(seed[:, 2] < -.1)
    with pytest.raises(ValueError):
        elbow_prior.FiSTA(exported).predict(np.zeros((4, 2, 14)), np.zeros((2, 7)))


def test_ready_pose_starts_the_elbows_near_the_training_mean_and_keeps_the_hands(exported):
    from igris.umi_walklow_replay.ik import IK
    model = UpperBody.from_mjcf(); network = elbow_prior.FiSTA(exported)
    start = np.zeros(len(JOINTS)); start[ACTIVE] = POSTURE
    hands = model.fk(start)

    def seed_miss(q):
        torso, elbows, _, _ = model.arm_frames(q)
        local = [(np.linalg.inv(torso) @ elbow)[:3, 3] for elbow in elbows]
        return np.linalg.norm(np.array(local) - elbow_prior.seed_elbows(network), axis=1)

    plain, _ = elbow_prior.ready_pose(IK(model), model, start, *hands, 0.)
    seeded, info = elbow_prior.ready_pose(IK(model), model, start, *hands, 0., network)
    assert info['position_error'].max() < 2e-3 and np.degrees(info['rotation_error'].max()) < 1.
    assert np.all(seed_miss(seeded) < seed_miss(plain) + 1e-6)
    assert np.abs(seeded[:3] - plain[:3]).max() < .02          # the trunk must not turn to let an elbow move
