import json

import numpy as np

from igris.umi_walklow_replay.elbow_data import double_rate, pack, split_sessions, states, window_starts
from igris.umi_walklow_replay.elbow_prior import ElbowPrior
from igris.umi_walklow_replay.kinematics import UpperBody

MODEL = UpperBody.from_mjcf()


def motion(frames, seed=0):
    """A smooth 31-joint motion inside the joint ranges."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0., 1., frames)[:, None]
    q = np.zeros((frames, 31))
    q[:, 15:29] = [-.6, .3, 0., -1., 0., 0., 0.] * 2 + .2 * np.sin(2 * np.pi * (t + rng.random(14)))
    return q


def test_doubling_the_rate_keeps_the_samples_and_fills_in_between():
    q = motion(20)
    doubled = double_rate(q)
    assert doubled.shape == (39, 31)
    assert np.array_equal(doubled[::2], q)
    ramp = np.linspace(0., 1., 12)[:, None] * np.ones((1, 31))
    assert np.allclose(double_rate(ramp)[1::2], (ramp[:-1] + ramp[1:]) / 2)          # a straight line stays straight
    assert np.abs(doubled[1::2] - (q[:-1] + q[1:]) / 2).max() < .01                   # close to the midpoints of a smooth motion


def test_states_are_what_the_prior_sees_at_run_time():
    q = motion(6)
    prior = ElbowPrior(MODEL, None)
    got = states(MODEL, q)
    assert got.shape == (6, 2, 14) and got.dtype == np.float32
    assert np.allclose(got[3], prior.state(q[3]), atol=1e-6)


def test_windows_take_six_frames_of_one_episode():
    starts = window_starts([8, 5, 7])                # episodes laid end to end
    assert starts.tolist() == [0, 1, 2, 13, 14]      # 8 frames give 3 windows, 5 give none, 7 give 2


def test_sessions_are_split_whole_and_the_same_way_every_time():
    frames = {session: 1000 + 37 * session for session in range(40)}
    first = split_sessions(frames)
    assert first == split_sessions(dict(reversed(list(frames.items()))))
    assert set(first) == set(frames) and set(first.values()) == {'train', 'val', 'test'}
    total = sum(frames.values())
    for name, low, high in (('val', .08, .14), ('test', .08, .14)):
        share = sum(frames[s] for s, where in first.items() if where == name) / total
        assert low <= share <= high


def test_pack_writes_the_layout_the_trainer_reads(tmp_path):
    episodes = [dict(episode=k, session=k // 2, action=motion(12 + k, seed=k)) for k in range(8)]
    split = {0: 'train', 1: 'train', 2: 'val', 3: 'test'}
    summary = pack(MODEL, episodes, split, tmp_path / 'packed/teleop')
    for name in ('train', 'val', 'test'):
        state = np.load(tmp_path / f'packed/teleop/{name}/states.npy')
        starts = np.load(tmp_path / f'packed/teleop/{name}/starts.npy')
        clips = json.loads((tmp_path / f'packed/teleop/{name}/clips.json').read_text())
        assert state.shape[1:] == (2, 14) and state.dtype == np.float32
        assert len(state) == sum(clip['frames'] for clip in clips) == summary[name]['frames']
        assert len(starts) == summary[name]['windows'] and starts.max() + 5 < len(state)
        for clip in clips:                               # every window lies inside one episode
            inside = starts[(starts >= clip['start']) & (starts < clip['start'] + clip['frames'])]
            assert len(inside) == clip['windows'] and (inside + 5 < clip['start'] + clip['frames']).all()
    train = json.loads((tmp_path / 'packed/teleop/train/clips.json').read_text())
    assert [clip['episode'] for clip in train] == [0, 1, 2, 3]
    assert train[0]['frames'] == 2 * 12 - 1                                           # stored at twice the recorded rate
    assert json.loads((tmp_path / 'packed/teleop/manifest.json').read_text())['rate_hz'] == 30.


def test_packed_windows_train_and_compare_end_to_end(tmp_path):
    """One epoch on a toy pack: the trainer reads what ``pack`` writes and the comparison reads the checkpoint."""
    import pytest
    pytest.importorskip('torch'); pytest.importorskip('scipy')
    from igris.umi_walklow_replay import elbow_train
    episodes = [dict(episode=k, session=k // 2, action=motion(40, seed=k)) for k in range(8)]
    pack(MODEL, episodes, {0: 'train', 1: 'train', 2: 'val', 3: 'test'}, tmp_path / 'packed/toy')
    best = elbow_train.train(tmp_path, 'toy', seed=1, device='cpu', epochs=1, batch=64)
    assert best.is_file() and (tmp_path / 'training/toy/seed_1/complete.json').is_file()
    report = elbow_train.compare(tmp_path, 'toy', {'toy': best}, 'cpu')
    assert report['toy']['samples'] == 2 * (2 * 40 - 1 - 5)                           # the two test episodes
    assert report['toy']['position_mean_m'] > 0 and report['toy']['persistence_position_mean_m'] > 0


def test_swivel_noise_turns_only_the_elbow_about_the_shoulder_hand_line():
    """The training augmentation: the arm's one free turn is put off by a random angle, as a rollout would leave it."""
    import pytest
    torch = pytest.importorskip('torch')
    from igris.umi_walklow_replay.elbow_train import swivel_noise
    history = torch.as_tensor(np.stack([states(MODEL, motion(5, seed=k)) for k in range(64)]))       # windows x 5 x 2 x 14
    same = swivel_noise(history, 0., torch.Generator().manual_seed(0))
    assert torch.equal(same, history)
    turned = swivel_noise(history, np.radians(20.), torch.Generator().manual_seed(0))
    assert torch.equal(turned[..., :7], history[..., :7])                                 # the hand is not touched
    hand, elbow, new = history[..., :3], history[..., 7:10], turned[..., 7:10]
    assert torch.allclose(new.norm(dim=-1), elbow.norm(dim=-1), atol=1e-5)                 # same distance from the shoulder
    assert torch.allclose((new - hand).norm(dim=-1), (elbow - hand).norm(dim=-1), atol=1e-5)   # and from the hand
    assert torch.allclose(turned[..., 10:14].norm(dim=-1), torch.ones(64, 5, 2), atol=1e-5)
    moved = (new - elbow).norm(dim=-1)
    assert moved.max() > .03 and moved.median() > .005                                     # centimetres, not noise in the last digit
    # One angle per window and arm: the five history frames turn together.
    axis = hand / hand.norm(dim=-1, keepdim=True)
    radial_old = elbow - axis * (elbow * axis).sum(-1, keepdim=True); radial_new = new - axis * (new * axis).sum(-1, keepdim=True)
    cosine = (radial_old * radial_new).sum(-1) / (radial_old.norm(dim=-1) * radial_new.norm(dim=-1))
    sine = (torch.cross(radial_old, radial_new, dim=-1) * axis).sum(-1) / (radial_old.norm(dim=-1) * radial_new.norm(dim=-1))
    angle = torch.atan2(sine, cosine)
    assert torch.allclose(angle, angle[:, :1].expand_as(angle), atol=1e-3)


def test_training_with_swivel_noise_writes_its_own_variant(tmp_path):
    import pytest
    pytest.importorskip('torch'); pytest.importorskip('scipy')
    from igris.umi_walklow_replay import elbow_train
    episodes = [dict(episode=k, session=k // 2, action=motion(40, seed=k)) for k in range(8)]
    pack(MODEL, episodes, {0: 'train', 1: 'train', 2: 'val', 3: 'test'}, tmp_path / 'packed/toy')
    best = elbow_train.train(tmp_path, 'toy', seed=1, device='cpu', epochs=1, batch=64, swivel_noise_deg=20., variant='toy_s20')
    assert best == tmp_path / 'training/toy_s20/seed_1/best.pt' and best.is_file()
    config = json.loads((tmp_path / 'training/toy_s20/seed_1/config.json').read_text())
    assert config['history_swivel_noise_deg'] == 20. and config['scope'] == 'toy'
    report = elbow_train.compare(tmp_path, 'toy', {'noisy': best}, 'cpu')                 # judged on clean windows
    assert report['noisy']['samples'] == 2 * (2 * 40 - 1 - 5)


def test_hands_only_training_does_not_look_at_the_elbow_history(tmp_path):
    """With the elbow history masked the prediction is a function of the hands alone: nothing to drift on."""
    import pytest
    torch = pytest.importorskip('torch'); pytest.importorskip('scipy')
    from igris.umi_walklow_replay import elbow_prior, elbow_train
    episodes = [dict(episode=k, session=k // 2, action=motion(40, seed=k)) for k in range(8)]
    pack(MODEL, episodes, {0: 'train', 1: 'train', 2: 'val', 3: 'test'}, tmp_path / 'packed/toy')
    best = elbow_train.train(tmp_path, 'toy', seed=1, device='cpu', epochs=1, batch=64, mask_elbow_history=True, variant='toy_hands')
    config = json.loads((tmp_path / 'training/toy_hands/seed_1/config.json').read_text())
    assert config['elbow_history'] == 'masked'
    report = elbow_train.compare(tmp_path, 'toy', {'hands': best}, 'cpu')
    assert report['hands']['samples'] == 2 * (2 * 40 - 1 - 5) and report['hands']['position_mean_m'] > 0
    # The exported network, as the replay runs it: other elbows in the history, same answer.
    network = elbow_prior.FiSTA(elbow_prior.export(best, tmp_path / 'fista.npz'))
    history = states(MODEL, motion(5, seed=3)).astype(float); goal = states(MODEL, motion(6, seed=3))[5, :, :7].astype(float)
    other = history.copy(); other[:, :, 7:10] += .05
    assert np.allclose(network.predict(history, goal), network.predict(other, goal))
    # And it is the torch network with the same inputs.
    saved = torch.load(best, map_location='cpu', weights_only=False)
    from igris.hl_ik.network import BimanualFiSTA, NetworkConfig, canonicalize
    net = BimanualFiSTA(NetworkConfig(**saved['config']['network'])); net.load_state_dict(saved['network']); net.eval()
    h, g, _ = canonicalize(torch.as_tensor(history, dtype=torch.float32)[None], torch.as_tensor(goal, dtype=torch.float32)[None])
    h = (h - saved['mean']) / saved['std']; g = (g - saved['mean'][:, :7]) / saved['std'][:, :7]; h[..., 7:] = 0.
    with torch.no_grad():
        expected = (net(h, g)[0] * saved['std'][:, 7:] + saved['mean'][:, 7:]).numpy()
    expected[:, 3:] /= np.linalg.norm(expected[:, 3:], axis=1, keepdims=True)
    assert np.allclose(network.predict(history, goal), expected, atol=2e-5)


def test_ready_pose_of_a_hands_only_prior_aims_at_its_own_prediction(tmp_path):
    """A hands-only network already knows where the elbows belong for the first hand poses; the feedback one starts
    from its training mean instead."""
    import pytest
    pytest.importorskip('torch'); pytest.importorskip('scipy')
    from igris.umi_walklow_replay import elbow_prior, elbow_train
    from igris.umi_walklow_replay.ik import IK, POSTURE
    from igris.umi_walklow_replay.kinematics import ACTIVE
    episodes = [dict(episode=k, session=k // 2, action=motion(40, seed=k)) for k in range(8)]
    pack(MODEL, episodes, {0: 'train', 1: 'train', 2: 'val', 3: 'test'}, tmp_path / 'packed/toy')
    best = elbow_train.train(tmp_path, 'toy', seed=1, device='cpu', epochs=2, batch=64, mask_elbow_history=True, variant='toy_hands')
    network = elbow_prior.FiSTA(elbow_prior.export(best, tmp_path / 'fista.npz'))
    start = np.zeros(31); start[ACTIVE] = POSTURE
    left, right = MODEL.fk(start)
    wanted = elbow_prior.held_elbows(MODEL, network, start, left, right)
    assert wanted.shape == (2, 3) and np.isfinite(wanted).all()

    def miss(q):
        return np.linalg.norm([elbow[:3, 3] for elbow in MODEL.arm_frames(q)[1]] - wanted, axis=1)

    q, info = elbow_prior.ready_pose(IK(MODEL), MODEL, start, left, right, 0., network, rounds=2, ticks=400)
    assert info['position_error'].max() < 2e-3
    assert np.all(miss(q) <= miss(start) + 1e-6) and miss(q).sum() < miss(start).sum()


def test_torch_swivel_angle_is_the_posture_measure():
    import pytest
    torch = pytest.importorskip('torch')
    from igris.umi_walklow_replay.elbow_train import swivel_angle
    from igris.umi_walklow_replay.posture import swivel_deg
    recorded = states(MODEL, motion(30, seed=5))
    got = torch.rad2deg(swivel_angle(torch.as_tensor(recorded[..., :3]), torch.as_tensor(recorded[..., 7:10]))).numpy()
    for frame in range(30):
        for side in range(2):
            assert abs(got[frame, side] - swivel_deg(recorded[frame, side, 7:10], recorded[frame, side, :3], side)) < 1e-3


def test_swivel_angle_becomes_an_elbow_place_the_arm_can_take():
    """What the replay does with a predicted swivel angle: the elbow where it is, turned about the shoulder-hand
    line toward that angle, by no more than the step."""
    from igris.umi_walklow_replay import elbow_prior
    from igris.umi_walklow_replay.posture import swivel_deg
    state = states(MODEL, motion(1, seed=2))[0].astype(float)
    now = np.array([swivel_deg(state[side, 7:10], state[side, :3], side) for side in range(2)])
    points = elbow_prior.swivel_elbows(state, np.radians(now + [10., -40.]), step=np.radians(15.))
    after = np.array([swivel_deg(points[side], state[side, :3], side) for side in range(2)])
    assert np.allclose(after - now, [10., -15.], atol=1e-6)                       # the second is cut to the step
    for side in range(2):                                                          # still on the arm's circle
        assert np.isclose(np.linalg.norm(points[side]), np.linalg.norm(state[side, 7:10]))
        assert np.isclose(np.linalg.norm(points[side] - state[side, :3]), np.linalg.norm(state[side, 7:10] - state[side, :3]))


def test_swivel_network_trains_and_runs_as_an_angle_inside_the_recorded_range(tmp_path):
    import pytest
    torch = pytest.importorskip('torch'); pytest.importorskip('scipy')
    from igris.umi_walklow_replay import elbow_prior, elbow_train
    from igris.umi_walklow_replay.ik import POSTURE
    from igris.umi_walklow_replay.kinematics import ACTIVE
    episodes = [dict(episode=k, session=k // 2, action=motion(40, seed=k)) for k in range(8)]
    pack(MODEL, episodes, {0: 'train', 1: 'train', 2: 'val', 3: 'test'}, tmp_path / 'packed/toy')
    best = elbow_train.train(tmp_path, 'toy', seed=1, device='cpu', epochs=2, batch=64, output='swivel')
    assert best == tmp_path / 'training/toy_swivel/seed_1/best.pt' and best.is_file()
    config = json.loads((tmp_path / 'training/toy_swivel/seed_1/config.json').read_text())
    assert config['output'] == 'swivel' and config['elbow_history'] == 'masked'
    report = elbow_train.compare(tmp_path, 'toy', {'swivel': best}, 'cpu')
    assert report['swivel']['samples'] == 2 * (2 * 40 - 1 - 5) and report['swivel']['swivel_error_mean_deg'] > 0
    network = elbow_prior.FiSTA(elbow_prior.export(best, tmp_path / 'fista.npz'))
    assert network.output == 'swivel' and network.mask_elbow_history
    history = states(MODEL, motion(5, seed=3)).astype(float); goal = states(MODEL, motion(6, seed=3))[5, :, :7].astype(float)
    angles = network.predict(history, goal)
    assert angles.shape == (2,)
    assert np.all(angles >= network.swivel_range[:, 0]) and np.all(angles <= network.swivel_range[:, 1])
    # The torch network, with the same inputs, before the cut to the recorded range.
    saved = torch.load(best, map_location='cpu', weights_only=False)
    net = elbow_train.build(saved['config']); net.load_state_dict(saved['network']); net.eval()
    from igris.hl_ik.network import canonicalize
    h, g, _ = canonicalize(torch.as_tensor(history, dtype=torch.float32)[None], torch.as_tensor(goal, dtype=torch.float32)[None])
    h = (h - saved['mean']) / saved['std']; g = (g - saved['mean'][:, :7]) / saved['std'][:, :7]; h[..., 7:] = 0.
    with torch.no_grad():
        expected = (net(h, g)[0, :, 0] * saved['swivel_std'] + saved['swivel_mean']).numpy()
    assert np.allclose(angles, np.clip(expected, network.swivel_range[:, 0], network.swivel_range[:, 1]), atol=2e-5)
    # In the replay it is an elbow place near the arm's own, for the ready pose and for every tick.
    q = np.zeros(31); q[ACTIVE] = POSTURE
    prior = elbow_prior.ElbowPrior(MODEL, network); prior.reset(q)
    here = np.array([elbow[:3, 3] for elbow in MODEL.arm_frames(q)[1]])
    for points in (prior.targets(q, 1. / 30, *MODEL.fk(q)), elbow_prior.held_elbows(MODEL, network, q, *MODEL.fk(q))):
        assert points.shape == (2, 3) and np.linalg.norm(points - here, axis=1).max() < .08
