"""Between the controller in the simulator container and a policy process outside it: two small files.

The simulator container has no network and private IPC; the run folder is the one thing both sides see.

  state.bin   controller -> policy, rewritten every control tick: the simulation time, the time since the policy
              took over, the measured and the commanded joint angles
  chunk.bin   policy -> controller, rewritten when the policy has answered an observation: the stream sample the
              observation was taken at, and targets for that sample and the ones after it - both hands in the
              pelvis frame and the three waist joints

Samples are counted on the controller's clock: sample ``k`` is due ``k / rate`` after the policy took over.  A
chunk observed at sample ``k`` holds targets for ``k, k + 1, ...`` - the action of a training frame is the
sample handed over at that frame (ee_dataset.py).  It can only be used from the sample after its arrival, and
never before ``k + delay``, so that a fast answer and a slow one are executed alike.  Where it takes over from
the chunk before it, the two are cross-faded over ``blend`` samples (pose_policy.py found chunk boundaries
to be the rough part of a policy's output).  A sample no chunk has a target for is skipped: the controller
then stays at the last one it was given.

The targets are handed out no faster than the demonstrations moved (``PairLimiter``).  A chunk is anchored at the
measured hands of its own observation, so where one chunk takes over from another the left hand's target can jump
by centimetres within a sample (1.4 m/s during one lift; the demonstrations: 0.4 m/s).  The joints then follow at
their speed bound, each arm as far as its own joints let it, and two hands whose targets are a perfect pair can
come apart in reality.  With the limit the commands of a policy are as smooth as the demonstrations' (IK error p99
56 -> 14 mm, joint acceleration p99 103 -> 45 rad/s2 over 12 runs each); the number of boxes placed did not change.
"""
import os
from pathlib import Path

import numpy as np

from .governor import blend_pose
from .kinematics import rotation_vector

STATE_SIZE = 5 + 31 + 31
ROW = 16 + 16 + 3                 # one sample of a chunk: left 4x4, right 4x4, waist
# Largest change of the left hand's target from one stream sample to the next: the 99.9th percentile of the 94
# demonstrations at 15 Hz (25.7 mm and 3.6 degrees; their 99th is 20.9 mm and 2.8 degrees, the largest 62 mm and 5.6).
STEP_LIMIT_M = .026
TURN_LIMIT_RAD = float(np.radians(3.6))


def write_atomic(path, values):
    """Replace ``path`` by the float64 array in one step, so a reader sees the old or the new file, never a part."""
    path = Path(path)
    scratch = path.with_name(path.name + f'.{os.getpid()}.tmp')
    scratch.write_bytes(np.ascontiguousarray(values, dtype=np.float64).tobytes())
    os.replace(scratch, path)


def read_values(path):
    """The float64 array in ``path``; None when there is no file yet."""
    try:
        return np.frombuffer(Path(path).read_bytes(), dtype=np.float64)
    except FileNotFoundError:
        return None


def write_state(folder, sequence, sim, elapsed, rate_hz, q_measured, q_command, done=False):
    write_atomic(Path(folder) / 'state.bin', np.r_[sequence, sim, elapsed, rate_hz, 1. if done else 0., q_measured, q_command])


def read_state(folder):
    """The controller's last state as a dict, or None before the first one."""
    values = read_values(Path(folder) / 'state.bin')
    if values is None or len(values) != STATE_SIZE:
        return None
    return dict(sequence=int(values[0]), sim=float(values[1]), elapsed=float(values[2]), rate_hz=float(values[3]),
                done=bool(values[4]), q_measured=values[5:36].copy(), q_command=values[36:67].copy())


def write_chunk(folder, sequence, observed, hands, waist):
    """``hands``: K x 2 x 4 x 4 (pelvis frame), ``waist``: K x 3, for the samples ``observed``, ``observed`` + 1, ..."""
    hands = np.asarray(hands, dtype=float); waist = np.asarray(waist, dtype=float)
    rows = np.concatenate([hands.reshape(len(hands), 32), waist.reshape(len(hands), 3)], axis=1)
    write_atomic(Path(folder) / 'chunk.bin', np.r_[sequence, observed, len(rows), rows.ravel()])


def read_chunk(folder):
    """(sequence, observed sample, hands K x 2 x 4 x 4, waist K x 3) of the newest chunk, or None."""
    values = read_values(Path(folder) / 'chunk.bin')
    if values is None or len(values) < 3 or len(values) != 3 + int(values[2]) * ROW:
        return None
    rows = values[3:].reshape(int(values[2]), ROW)
    return int(values[0]), int(values[1]), rows[:, :32].reshape(-1, 2, 4, 4).copy(), rows[:, 32:].copy()


class PairLimiter:
    """Hands out hand targets that move no further per sample than ``step`` [m] and ``turn`` [rad], the two hands
    as the pair they were given as: the left hand's target is held back on the way to where it is wanted, and the
    right hand's target keeps its place in the left one's frame."""

    def __init__(self, start_left, step=STEP_LIMIT_M, turn=TURN_LIMIT_RAD):
        self.left = np.array(start_left, dtype=float); self.step, self.turn = float(step), float(turn)
        self.limited = 0

    def apply(self, left, right):
        left = np.asarray(left, dtype=float); right = np.asarray(right, dtype=float)
        distance = float(np.linalg.norm(left[:3, 3] - self.left[:3, 3]))
        angle = float(np.linalg.norm(rotation_vector(self.left[:3, :3].T @ left[:3, :3])))
        share = min(1., self.step / distance if distance > self.step else 1., self.turn / angle if angle > self.turn else 1.)
        if share < 1.:
            self.limited += 1
            held = blend_pose(self.left, left, share)
            right = held @ np.linalg.inv(left) @ right
            left = held
        self.left = left.copy()
        return left, right


class Plan:
    """Targets per stream sample, from the chunks as they arrive."""

    def __init__(self, rate_hz, delay=2, blend=4):
        self.rate, self.delay, self.blend = float(rate_hz), int(delay), int(blend)
        self.targets = {}             # sample -> (left 4x4, right 4x4, waist 3)
        self.source = {}              # sample -> observation the target came from
        self.next = 0                 # first sample not handed out yet
        self.chunks = 0; self.skipped = 0; self.arrival = []      # (observed sample, sample it arrived in)
        self.started = False          # a target has been handed out: from then on a sample without one is a gap

    def sample(self, elapsed):
        """Index of the newest sample that is due ``elapsed`` seconds after the policy took over."""
        return int(np.floor(elapsed * self.rate + 1e-9))

    def accept(self, observed, hands, waist, elapsed):
        """Take in a chunk that arrived at ``elapsed``.  Returns the first sample it is used for."""
        first = max(observed + self.delay, self.sample(elapsed) + 1, self.next)
        fresh = {}
        for offset in range(first - observed, len(hands)):
            sample = observed + offset
            new = (hands[offset, 0], hands[offset, 1], waist[offset])
            share = min(1., (sample - first + 1.) / self.blend) if self.blend > 0 else 1.
            old = self.targets.get(sample)
            if old is not None and share < 1.:
                new = (blend_pose(old[0], new[0], share), blend_pose(old[1], new[1], share), (1. - share) * old[2] + share * new[2])
            fresh[sample] = new
        # What the older chunks said beyond this one's reach is dropped: a newer observation outranks them.
        self.targets = {sample: target for sample, target in self.targets.items() if sample < first}
        self.targets.update(fresh)
        for sample in fresh:
            self.source[sample] = observed
        self.chunks += 1; self.arrival.append((int(observed), self.sample(elapsed)))
        return first

    def due(self, elapsed):
        """Targets of the samples that have come due and were not handed out before, oldest first, as
        (sample, left, right, waist).  Once the first target is out, a due sample without one is skipped and counted."""
        out = []
        newest = self.sample(elapsed)
        while self.next <= newest:
            target = self.targets.pop(self.next, None)
            if target is not None:
                out.append((self.next, *target)); self.started = True
            elif self.started:
                self.skipped += 1
            self.next += 1
        return out


class ControllerLink:
    """The controller's side: publish the state, take in chunks, hand out due targets."""

    def __init__(self, folder, rate_hz, delay=2, blend=4, limiter=None):
        self.folder = Path(folder); self.folder.mkdir(parents=True, exist_ok=True)
        for name in ('state.bin', 'chunk.bin'):
            (self.folder / name).unlink(missing_ok=True)
        self.plan = Plan(rate_hz, delay, blend)
        self.limiter = limiter        # a PairLimiter, or None to hand the targets out as they come
        self.rate = float(rate_hz); self.sequence = -1; self.stamp = None; self.used = -1

    def publish(self, sequence, sim, elapsed, q_measured, q_command, done=False):
        write_state(self.folder, sequence, sim, elapsed, self.rate, q_measured, q_command, done)

    def due(self, elapsed):
        """Read a new chunk if there is one, then the due targets as (left, right, waist) tuples."""
        path = self.folder / 'chunk.bin'
        try:
            stamp = path.stat().st_mtime_ns
        except FileNotFoundError:
            stamp = None
        if stamp is not None and stamp != self.stamp:
            chunk = read_chunk(self.folder)
            if chunk is not None and chunk[0] != self.sequence:
                self.sequence = chunk[0]; self.stamp = stamp
                self.plan.accept(chunk[1], chunk[2], chunk[3], elapsed)
        out = []
        for sample, left, right, waist in self.plan.due(elapsed):
            self.used = self.plan.source.pop(sample, self.used)
            if self.limiter is not None:
                left, right = self.limiter.apply(left, right)
            out.append((left, right, waist))
        return out

    def report(self):
        arrival = np.array(self.plan.arrival, dtype=float).reshape(-1, 2)
        late = arrival[:, 1] - arrival[:, 0] if len(arrival) else np.zeros(0)
        return dict(chunks=int(self.plan.chunks), skipped_samples=int(self.plan.skipped),
                    arrival_samples_after_observation_mean=float(late.mean()) if len(late) else None,
                    arrival_samples_after_observation_max=float(late.max()) if len(late) else None,
                    delay=self.plan.delay, blend=self.plan.blend,
                    limited_samples=int(self.limiter.limited) if self.limiter is not None else None,
                    step_limit_m=self.limiter.step if self.limiter is not None else None,
                    turn_limit_rad=self.limiter.turn if self.limiter is not None else None)
