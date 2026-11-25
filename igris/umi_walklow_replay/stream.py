"""The input side of a controller that is handed one pose at a time.

The replay worker indexes its targets by time: it can read ahead, slow its own clock, and it knows from the
whole episode where a grasp begins.  A policy's actions or a VR operator's hands arrive differently - one sample
after the other, at the source's rate, and nothing is known of the next one.  ``PoseStream`` is all the
controller has in that case: samples are pushed in as they arrive and the target of *now* is read out.  Between
samples the target moves in a straight line from where it is to the newest sample, over one sample period, so
it runs one period behind the source (67 ms at 15 Hz) and can never be ahead of what has arrived.

``Recorded`` plays a stored stream out in that manner, and ``rollout`` is the controller without a simulator:
100 Hz IK on the stream, the hand orientation always in full, no slowed clock, the arm posture from the lookup
or the learned prior (both use only the present).  The replay worker uses the same two classes (``--stream``).
"""
import numpy as np

from .governor import blend_pose
from .ik import POSTURE
from .kinematics import ACTIVE, JOINTS, rotation_vector


class PoseStream:
    """Hand targets and trunk heading from samples that arrive one at a time.  The heading may also be an
    array - the three waist joint angles a policy asks for - and is blended the same way."""

    def __init__(self, period):
        self.period = float(period)
        self.start = None            # where the target was when the newest sample arrived
        self.newest = None           # (left 4x4, right 4x4, heading): the sample the target is moving to
        self.arrived = 0.

    def push(self, now, left, right, heading):
        """A sample has arrived at time ``now``."""
        sample = (np.array(left, dtype=float), np.array(right, dtype=float),
                  np.array(heading, dtype=float) if np.ndim(heading) else float(heading))
        self.start = sample if self.newest is None else self.target(now)
        self.newest = sample; self.arrived = float(now)

    def target(self, now):
        """(left, right, heading) to follow at ``now``; None before the first sample."""
        if self.newest is None:
            return None
        share = float(np.clip((now - self.arrived) / self.period, 0., 1.))
        return (blend_pose(self.start[0], self.newest[0], share), blend_pose(self.start[1], self.newest[1], share),
                (1. - share) * self.start[2] + share * self.newest[2])


class Recorded:
    """A stored stream (``t``, ``left``, ``right``, ``head_yaw``) handed out sample by sample: each one is
    released at its own time plus ``latency`` (the time a source takes to answer), once."""

    def __init__(self, stream, latency=0.):
        self.stamps = np.asarray(stream['t'], dtype=float) + float(latency)
        self.left, self.right, self.heading = stream['left'], stream['right'], np.asarray(stream['head_yaw'], dtype=float)
        self.next = 0

    @property
    def finished(self):
        return self.next >= len(self.stamps)

    def due(self, now):
        """Samples that have arrived by ``now`` and were not handed out before, oldest first."""
        out = []
        while self.next < len(self.stamps) and self.stamps[self.next] <= now + 1e-9:
            out.append((self.left[self.next], self.right[self.next], float(self.heading[self.next])))
            self.next += 1
        return out


def lifted(pose, height):
    out = np.array(pose, dtype=float, copy=True); out[2, 3] += height
    return out


def rollout(source, ik, model, *, period, posture=None, prior=None, base_offset_z=0., latency=0.):
    """The controller's commands for ``source`` arriving in real time, without a simulator.

    ``source`` holds ``t``, ``left``, ``right`` (pelvis frame, N x 4 x 4) and ``head_yaw`` at its own rate;
    ``period`` is that rate's sample period.  ``posture`` is a ``posture_prior.TeleopPosture``, ``prior`` an
    ``elbow_prior.ElbowPrior`` (its goal is the newest sample: the place the target is moving to).  The robot
    starts in the posture for the first sample, as it is brought there before a replay.  Returns ``t``, the
    commands ``q`` (T x 31), the targets followed and the IK diagnostics per tick.
    """
    dt = ik.dt
    stamps = np.asarray(source['t'], dtype=float)
    feed = Recorded(dict(source, t=stamps - stamps[0]), latency)
    stream = PoseStream(period)
    for sample in feed.due(latency):                                # the first sample: known before the motion starts
        stream.push(0., lifted(sample[0], base_offset_z), lifted(sample[1], base_offset_z), sample[2])
    left, right, heading = stream.target(0.)
    q = np.zeros(len(JOINTS)); q[ACTIVE] = POSTURE
    if posture is not None:
        q, _ = ik.converge(q, left, right, waist_yaw_reference=-heading, posture=posture.reference(model, q, left, right), ticks=1500)
    else:
        from .elbow_prior import ready_pose
        q, _ = ready_pose(ik, model, q, left, right, -heading, prior.network if prior is not None else None)
    if prior is not None:
        prior.reset(q, 0.)
    ticks = int(np.ceil((stamps[-1] - stamps[0] + latency + period) / dt - 1e-9)) + 1
    rows = dict(q=[], target_left=[], target_right=[], heading=[], position_error=[], rotation_error=[], saturated=[],
                gap=[], relaxed=[], elbow_trust=[])
    for index in range(ticks):
        now = index * dt
        for sample in feed.due(now):
            stream.push(now, lifted(sample[0], base_offset_z), lifted(sample[1], base_offset_z), sample[2])
        left, right, heading = stream.target(now)
        reference = posture.reference(model, q, left, right, dt) if posture is not None else None
        elbows = prior.targets(q, now, stream.newest[0], stream.newest[1]) if prior is not None else None
        q, info = ik.step(q, left, right, waist_yaw_reference=-heading, posture=reference, orientation=1., elbow_targets=elbows)
        rows['q'].append(q); rows['target_left'].append(left); rows['target_right'].append(right); rows['heading'].append(heading)
        for key in ('position_error', 'rotation_error', 'saturated', 'gap', 'relaxed', 'elbow_trust'):
            rows[key].append(info[key])
    out = {key: np.array(value) for key, value in rows.items()}
    out['t'] = np.arange(ticks) * dt
    return out


def summary(run, source, period, model=None, velocity=None, lower=None, upper=None):
    """How well a rollout did: hand errors against the target it was given (the source one period ago) and against
    the source's present, joint speeds, bounds.  ``velocity`` / ``lower`` / ``upper`` are the IK's 17 bounds."""
    q = run['q'][:, ACTIVE]
    speed = np.abs(np.diff(q, axis=0)) / float(run['t'][1] - run['t'][0])
    stamps = np.asarray(source['t'], dtype=float) - float(source['t'][0])
    index = np.clip(np.searchsorted(stamps, run['t'], side='right') - 1, 0, len(stamps) - 2)
    share = np.clip((run['t'] - stamps[index]) / (stamps[index + 1] - stamps[index]), 0., 1.)
    offset = run['target_left'][-1][2, 3] - source['left'][-1][2, 3]             # base height the rollout added
    now = np.zeros((len(run['t']), 2))
    for column, (key, target) in enumerate((('left', 'target_left'), ('right', 'target_right'))):
        present = (1. - share)[:, None] * source[key][index][:, :3, 3] + share[:, None] * source[key][index + 1][:, :3, 3]
        present[:, 2] += offset
        # The hand is where its target is minus the IK error; the IK error is small, so measure from the target.
        now[:, column] = np.linalg.norm(run[target][:, :3, 3] - present, axis=1) + run['position_error'][:, column]
    report = dict(
        seconds=float(run['t'][-1]), period_s=float(period),
        hand_error_p50_m=float(np.percentile(run['position_error'], 50)), hand_error_p99_m=float(np.percentile(run['position_error'], 99)),
        hand_error_max_m=float(run['position_error'].max()),
        hand_rotation_error_p99_deg=float(np.degrees(np.percentile(run['rotation_error'], 99))),
        hand_rotation_error_max_deg=float(np.degrees(run['rotation_error'].max())),
        hand_error_now_p50_m=float(np.percentile(now, 50)), hand_error_now_p99_m=float(np.percentile(now, 99)),
        joint_speed_p99_rad_s=float(np.percentile(speed.max(axis=1), 99)), joint_speed_max_rad_s=float(speed.max()),
        ticks_above_1_rad_s_fraction=float((speed.max(axis=1) > 1.0 + 1e-6).mean()),
        relaxed_ticks=int(np.count_nonzero(run['relaxed'])),
        gap_min_m=float(np.nanmin(run['gap'])) if np.isfinite(run['gap']).any() else None)
    if velocity is not None:
        report['ticks_at_speed_bound_fraction'] = float((speed >= np.asarray(velocity) - 1e-6).any(axis=1).mean())
    if lower is not None and upper is not None:
        report['arm_joint_at_limit_fraction'] = float(((q[:, 3:] <= lower[3:] + 1e-6) | (q[:, 3:] >= upper[3:] - 1e-6)).any(axis=1).mean())
        report['waist_lean_at_limit_fraction'] = float(((q[:, 1:3] <= lower[1:3] + 1e-6) | (q[:, 1:3] >= upper[1:3] - 1e-6)).any(axis=1).mean())
    return report
