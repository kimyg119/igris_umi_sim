"""HL-IK elbow prior (FiSTA) for the replay IK, in numpy.

FiSTA predicts, per arm, where a human elbow would be at the next 30 Hz step
from the last five hand and elbow poses and the next hand goal.  Poses are
xyz + wxyz in the torso frame, measured from the shoulder origin; the network
and its conventions are ``igris/hl_ik`` (network.py, solver.py).  The simulator
container has no torch, so the trained weights are exported once on the host
and evaluated here with numpy.

The prior is a one-step predictor fed with its own results.  On the UMI box
episode its rollout forgets the starting posture within about ten seconds, so
the seed (``seed_elbows``) only shapes the first moments.  It is a secondary IK
task, never against the hands: with the hand pose fixed, only the arm's swivel
is left for it.

Limits found on this data (2026-11-23, simulation), which is why the replay
leaves it off unless asked:
- The hands are above the robot's shoulders 65-73 % of the time and their
  orientation is 2.5-3.5 sigma from the training mean, so the network works at
  the edge of what it saw.
- It moves the elbows out and up.  Under the vendor walking controller that
  shifts the pelvis more, and episode 0 stacked all three boxes in 5 of 9 runs
  against 17 of 18 without it.
- Clipping the predicted elbow to the training range (all axes, or the height
  alone) was worse (1 of 6 with the height limit); not kept.
"""
from collections import deque
import os
import hashlib
from pathlib import Path

import numpy as np

from .kinematics import SHOULDERS, axis_rotation, matrix_quat
from .posture import swivel_deg

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / 'outputs/hl_ik_20260915/training/full/seed_42/best.pt'   # default of the HL-IK VR teleop
# The same network trained on this robot's teleoperated box handling instead of retargeted human motion
# (elbow_data.py, elbow_train.py; 2026-11-24), in its hands-only form: it is not given its own elbow history,
# which is what let a rollout drift.  Seed 42 is the representative seed, fixed before training.
TELEOP_CHECKPOINT = Path(os.environ.get('UMI_TELEOP_PRIOR', ROOT / 'checkpoints/hl_ik_teleop/teleop_hands/seed_42/best.pt'))
# The same again with the swivel angle of each arm as the output instead of an elbow place (always something the
# arm can do; elbow_train.py ``--output swivel``).
SWIVEL_CHECKPOINT = Path(os.environ.get('UMI_SWIVEL_PRIOR', ROOT / 'checkpoints/hl_ik_teleop/teleop_swivel/seed_42/best.pt'))
CHECKPOINTS = {'human': CHECKPOINT, 'teleop': TELEOP_CHECKPOINT, 'swivel': SWIVEL_CHECKPOINT}
SWIVEL_STEP = np.radians(15.)            # largest turn asked for at once from a predicted swivel angle
PERIOD_S = 1. / 30.                      # history spacing the network was trained with


def export(checkpoint, target):
    """Write the torch checkpoint as plain arrays.  Needs torch; run on the host."""
    import torch
    checkpoint = Path(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    network = saved['config']['network']
    if network['history'] != 5 or network['arm_mode'] != 'independent_single_arm_text_spec':
        raise ValueError(f'Unsupported FiSTA configuration: {network}')
    arrays = {name.replace('.', '/'): value.numpy().astype(np.float64) for name, value in saved['network'].items()}
    output = saved['config'].get('output', 'pose')
    if output == 'swivel':
        arrays.update({key: saved[key].numpy().astype(np.float64) for key in ('swivel_mean', 'swivel_std', 'swivel_range')})
    np.savez(target, mean=saved['mean'].numpy().astype(np.float64), std=saved['std'].numpy().astype(np.float64),
             hidden=network['hidden'], heads=network['heads'], history=network['history'],
             # A hands-only network (elbow_train.py) was trained without the elbow history and must not be given it.
             mask_elbow_history=saved['config'].get('elbow_history') == 'masked',
             output=output,                        # 'pose': elbow pose per arm; 'swivel': swivel angle per arm
             checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
             **arrays)
    return target


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


class FiSTA:
    """Both single-arm networks of ``igris.hl_ik.network.BimanualFiSTA`` in evaluation mode."""

    def __init__(self, path):
        data = np.load(path)
        self.mean, self.std = data['mean'], data['std']
        self.hidden, self.heads = int(data['hidden']), int(data['heads'])
        self.sha256 = str(data['checkpoint_sha256'])
        self.mask_elbow_history = bool(data['mask_elbow_history']) if 'mask_elbow_history' in data.files else False
        self.output = str(data['output']) if 'output' in data.files else 'pose'
        if self.output == 'swivel':
            self.swivel_mean, self.swivel_std, self.swivel_range = data['swivel_mean'], data['swivel_std'], data['swivel_range']
        self.arms = [{name.split('/', 2)[2]: data[name] for name in data.files if name.startswith(f'arms/{arm}/')}
                     for arm in range(2)]
        if self.mean.shape != (2, 14) or any('temporal/weight_ih_l0' not in arm for arm in self.arms):
            raise ValueError('Not an exported FiSTA checkpoint')

    @staticmethod
    def _linear(w, name, x):
        return x @ w[name + '/weight'].T + w[name + '/bias']

    def _arm(self, w, history, goal):
        h = self.hidden
        # GRU over the five history steps (PyTorch gate order: reset, update, new).
        state = np.zeros(h)
        for step in history:
            gi = w['temporal/weight_ih_l0'] @ step + w['temporal/bias_ih_l0']
            gh = w['temporal/weight_hh_l0'] @ state + w['temporal/bias_hh_l0']
            reset = _sigmoid(gi[:h] + gh[:h]); update = _sigmoid(gi[h:2 * h] + gh[h:2 * h])
            new = np.tanh(gi[2 * h:] + reset * gh[2 * h:])
            state = (1 - update) * new + update * state
        # Self-attention between the latest hand token and elbow token.
        tokens = history[-1].reshape(2, 7)
        tokens = self._linear(w, 'spatial_pre/2', np.maximum(self._linear(w, 'spatial_pre/0', tokens), 0))
        projected = tokens @ w['attention/in_proj_weight'].T + w['attention/in_proj_bias']
        query, key, value = (projected[:, i * h:(i + 1) * h].reshape(2, self.heads, h // self.heads) for i in range(3))
        attended = np.empty((2, self.heads, h // self.heads))
        for head in range(self.heads):
            score = query[:, head] @ key[:, head].T / np.sqrt(h // self.heads)
            weight = np.exp(score - score.max(axis=1, keepdims=True))
            weight /= weight.sum(axis=1, keepdims=True)
            attended[:, head] = weight @ value[:, head]
        spatial = attended.reshape(2, h) @ w['attention/out_proj/weight'].T + w['attention/out_proj/bias']
        spatial = self._linear(w, 'spatial_post/2', np.maximum(self._linear(w, 'spatial_post/0', spatial.reshape(-1)), 0))
        # The goal modulates the temporal feature (FiLM).
        film = self._linear(w, 'goal/2', np.maximum(self._linear(w, 'goal/0', goal), 0))
        conditioned = _sigmoid(film[:h]) * state + film[h:]
        hidden = np.maximum(self._linear(w, 'output/0', np.r_[conditioned, spatial]), 0)
        return self._linear(w, 'output/3', hidden)

    def predict(self, history, goal):
        """Next elbow poses (2 x 7) for history (5 x 2 x 14) and hand goals (2 x 7); for a swivel network the swivel
        angle of each arm [rad], cut to the range of its training labels."""
        history = np.array(history, dtype=float, copy=True); goal = np.array(goal, dtype=float, copy=True)
        if history.shape != (5, 2, 14) or goal.shape != (2, 7) or not np.isfinite(history).all() or not np.isfinite(goal).all():
            raise ValueError('Expected finite history (5, 2, 14) and goal (2, 7)')
        # Quaternion hemispheres follow the latest history sample, as in training.
        for start in (3, 10):
            latest = history[-1, :, start:start + 4].copy()
            latest *= np.where(latest[:, :1] < 0, -1., 1.)
            sign = np.where(np.sum(history[:, :, start:start + 4] * latest[None], axis=-1, keepdims=True) < 0, -1., 1.)
            history[:, :, start:start + 4] *= sign
        goal[:, 3:] *= np.where(np.sum(goal[:, 3:] * history[-1, :, 3:7], axis=-1, keepdims=True) < 0, -1., 1.)
        history = (history - self.mean) / self.std
        if self.mask_elbow_history:
            history[:, :, 7:] = 0.                       # hands only: the elbow channels stand at their training mean
        goal = (goal - self.mean[:, :7]) / self.std[:, :7]
        out = np.stack([self._arm(self.arms[arm], history[:, arm], goal[arm]) for arm in range(2)])
        if self.output == 'swivel':
            return np.clip(out[:, 0] * self.swivel_std + self.swivel_mean, self.swivel_range[:, 0], self.swivel_range[:, 1])
        pose = out * self.std[:, 7:] + self.mean[:, 7:]
        norm = np.linalg.norm(pose[:, 3:], axis=1, keepdims=True)
        if np.any(norm < 1e-8):
            raise ValueError('Near-zero predicted quaternion')
        pose[:, 3:] /= norm
        return pose


def pose7(transform):
    return np.r_[transform[:3, 3], matrix_quat(transform[:3, :3])]


def swivel_elbows(state, wanted, step=SWIVEL_STEP):
    """Elbow places (2 x 3, from the shoulder, torso axes) for wanted swivel angles [rad]: each elbow of ``state``
    (2 x 14) turned about its shoulder-hand line toward the wanted angle, by at most ``step``.  The place keeps its
    distance from the shoulder and from the hand, so it is one the arm can take with the hand where it is."""
    points = np.empty((2, 3))
    for side in range(2):
        hand, elbow = state[side, :3], state[side, 7:10]
        now = np.radians(swivel_deg(elbow, hand, side))
        turn = float(np.clip(np.arctan2(np.sin(wanted[side] - now), np.cos(wanted[side] - now)), -step, step))
        # + is outward: about the shoulder-hand line that is a right-hand turn for the left arm, a left-hand one for the right.
        points[side] = axis_rotation(hand / np.linalg.norm(hand), turn if side == 0 else -turn) @ elbow
    return points


def seed_elbows(network):
    """Training-set mean elbow position per arm, torso frame [m].

    Used once to start from a posture the prior knows: on average the human elbow
    sits 14 cm below, 9 cm outside and 3-4 cm ahead of the shoulder.
    """
    return network.mean[:, 7:10] + SHOULDERS


def held_elbows(model, network, q, left, right):
    """Elbow places (2 x 3, pelvis frame) the network gives for holding the hand poses ``left`` and ``right``
    (4x4, pelvis frame) with the trunk of ``q``: its history is the same state five times."""
    state = ElbowPrior(model, network).state(q)
    torso = model.torso(q); inverse = np.linalg.inv(torso)
    for side, goal in enumerate((left, right)):
        local = inverse @ goal
        local[:3, 3] -= SHOULDERS[side]
        state[side, :7] = pose7(local)                 # the hands as asked for, not as they are
    prediction = network.predict(np.array([state] * 5), state[:, :7])
    if network.output == 'swivel':
        prediction = swivel_elbows(ElbowPrior(model, network).state(q), prediction)        # from the arm as it is
    return np.array([torso[:3, :3] @ (prediction[side, :3] + SHOULDERS[side]) + torso[:3, 3] for side in range(2)])


def ready_pose(ik, model, q, left, right, waist_yaw_reference, network=None, rounds=4, ticks=800):
    """Configuration that holds the first hand targets.  With a network the elbows start where it wants them: a
    hands-only network says so for these hand poses; one that reads its own elbow history starts at its mean place."""
    info = None
    for _ in range(rounds if network is not None else 1):
        elbows = None
        if network is not None and network.mask_elbow_history:
            elbows = held_elbows(model, network, q, left, right)
        elif network is not None:
            torso = model.arm_frames(q)[0]
            elbows = (torso[:3, :3] @ seed_elbows(network).T).T + torso[:3, 3]
        q, info = ik.converge(q, left, right, waist_yaw_reference=waist_yaw_reference, elbow_targets=elbows, ticks=ticks)
    return q, info


class ElbowPrior:
    """30 Hz history of commanded poses and the elbow targets predicted from it."""

    def __init__(self, model, network, period=PERIOD_S):
        self.model, self.network, self.period = model, network, float(period)
        self.history = deque(maxlen=5)
        self.last_time = None
        self.prediction = None          # 2 x 7, relative to the shoulders, torso axes

    def state(self, q):
        """Hand and elbow poses (2 x 14) of a 31-joint configuration, as the network sees them."""
        torso, elbows, hands, _ = self.model.arm_frames(q)
        inverse = np.linalg.inv(torso)
        rows = []
        for side in range(2):
            row = []
            for frame in (hands[side], elbows[side]):
                local = inverse @ frame
                local[:3, 3] -= SHOULDERS[side]
                row.extend(pose7(local))
            rows.append(row)
        return np.array(rows)

    def reset(self, q, time=0.):
        state = self.state(q)
        self.history.clear()
        for _ in range(5):
            self.history.append(state.copy())
        self.last_time = float(time)
        self.prediction = state[:, 7:].copy()

    def targets(self, q, time, goal_left, goal_right):
        """Elbow positions (2 x 3, pelvis frame) for the command ``q`` and the next hand goals.

        ``goal_*`` are 4x4 pelvis-frame hand targets one history period ahead.  The
        history advances whenever a period of ``time`` has passed.
        """
        if self.last_time is None:
            self.reset(q, time)
        torso = self.model.torso(q)
        if time - self.last_time >= self.period - 1e-9:
            # Never jump more than one sample: a stalled clock must not skip history.
            self.last_time = time if time - self.last_time > 2 * self.period else self.last_time + self.period
            self.history.append(self.state(q))
            inverse = np.linalg.inv(torso)
            goals = []
            for side, goal in enumerate((goal_left, goal_right)):
                local = inverse @ goal
                local[:3, 3] -= SHOULDERS[side]
                goals.append(pose7(local))
            self.prediction = self.network.predict(np.array(self.history), np.array(goals))
            if self.network.output == 'swivel':
                self.prediction = swivel_elbows(self.history[-1], self.prediction)
        return np.array([torso[:3, :3] @ (self.prediction[side, :3] + SHOULDERS[side]) + torso[:3, 3] for side in range(2)])
