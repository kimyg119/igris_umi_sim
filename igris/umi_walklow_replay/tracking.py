"""Repair tracking faults in a recorded UMI episode before it becomes IK targets.

Found in the 34 episodes of the dataset (15 Hz, hand poses xyz + xyzw in the pelvis frame):
  repeat  a pose bit-identical to the previous frame: nothing new arrived (88 frames)
  stall   a hand that stands still for 0.2-0.6 s and then jumps 2-8 cm to catch up (11, left hand)
  spike   a hand that jumps 5-15 cm away and is back on its course within a few frames
  step    a jump of 5-9 cm, or of 7-11 deg, that stays
The frames of a fault are replaced by the straight path between their good neighbours.

Both hands and the head yaw also share a sample-time error of 12.6 ms rms, up to one 30 Hz
period: the distance covered per frame alternates (8 / 16 / 24 mm) although the hand moved
evenly.  A centred quadratic fit over 0.47 s takes that out; it keeps 99 % of a 1 Hz motion.
Only numpy is used.
"""
import numpy as np

# A stall: at least STALL_FRAMES steps below these sizes, ended by a step above STALL_JUMP_M.
STALL_STEP_M = 5e-4
STALL_STEP_RAD = np.radians(.1)
STALL_FRAMES = 3
STALL_JUMP_M = .02
# A jump: one step above JUMP_M or JUMP_RAD that is also JUMP_RATIO times the median of the
# NEIGHBOURS steps on either side.  The recorded hands move at most 0.55 m/s, 3.7 cm per frame.
JUMP_M = .04
JUMP_RAD = np.radians(6.)
JUMP_RATIO = 3.
NEIGHBOURS = 5
SPIKE_FRAMES = 6            # longest excursion that still counts as a spike
BRIDGE_FRAMES = 2           # frames replaced on each side of a jump that stays
SMOOTH_WINDOW = 7           # frames


def _turns(rotation):
    """Angle between neighbouring quaternions [rad]."""
    return 2 * np.arccos(np.clip(np.abs(np.sum(rotation[:-1] * rotation[1:], axis=1)), 0., 1.))


def _runs(mask):
    """First and last index of every run of True."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2] - 1))


def _event(kind, frame, first, last, jump_m, jump_rad):
    return dict(kind=kind, frame=int(frame), frames=int(last - first + 1), jump_m=float(jump_m),
                jump_deg=float(np.degrees(jump_rad)))


def _stale(pose):
    """Frames in which the tracker delivered nothing new: stalls that end in a jump, and exact repeats."""
    count = len(pose)
    bad = np.zeros(count, dtype=bool); events = []
    if count < 3:
        return bad, events
    distance = np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1); turn = _turns(pose[:, 3:])
    for first, last in _runs((distance < STALL_STEP_M) & (turn < STALL_STEP_RAD)):
        leaving = last + 1                  # the step that ends the run
        if last - first + 1 >= STALL_FRAMES and leaving < count - 1 and distance[leaving] > STALL_JUMP_M:
            bad[first + 1:leaving + 1] = True
            events.append(_event('stall', first + 1, first + 1, leaving, distance[leaving], turn[leaving]))
    same = np.r_[False, np.all(pose[1:] == pose[:-1], axis=1)] & ~bad
    for first, last in _runs(same):
        leaving = min(last, count - 2)
        events.append(_event('repeat', first, first, last, distance[leaving], turn[leaving]))
    return bad | same, events


def _jumps(pose):
    """Frames around a jump: the excursion of a spike, or BRIDGE_FRAMES on each side of a jump that stays."""
    count = len(pose)
    bad = np.zeros(count, dtype=bool); events = []
    if count < 3:
        return bad, events
    position = pose[:, :3]
    move = np.diff(position, axis=0); distance = np.linalg.norm(move, axis=1); turn = _turns(pose[:, 3:])
    step = 0
    while step < count - 1:
        near = np.r_[max(0, step - NEIGHBOURS):step, step + 1:min(count - 1, step + 1 + NEIGHBOURS)]
        far = distance[step] > max(JUMP_M, JUMP_RATIO * np.median(distance[near]))
        twisted = turn[step] > max(JUMP_RAD, JUMP_RATIO * np.median(turn[near]))
        if far:
            # Back on the course the hand had before the jump?
            course = np.median(move[max(0, step - NEIGHBOURS):step], axis=0) if step else np.zeros(3)
            back = next((frame for frame in range(step + 2, min(count, step + 2 + SPIKE_FRAMES))
                         if np.linalg.norm(position[frame] - position[step] - (frame - step) * course)
                         < .5 * distance[step]), None)
            if back is not None:
                bad[step + 1:back] = True
                events.append(_event('spike', step + 1, step + 1, back - 1, distance[step], turn[step]))
                step = back
                continue
        if far or twisted:
            first, last = max(0, step - BRIDGE_FRAMES + 1), min(count - 1, step + BRIDGE_FRAMES)
            bad[first:last + 1] = True
            events.append(_event('step', step + 1, first, last, distance[step], turn[step]))
        step += 1
    return bad, events


def _slerp(a, b, fraction):
    dot = float(a @ b)
    if dot < 0:
        b, dot = -b, -dot
    if dot > .9995:
        out = a + fraction * (b - a)
    else:
        angle = np.arccos(dot)
        out = (np.sin((1 - fraction) * angle) * a + np.sin(fraction * angle) * b) / np.sin(angle)
    return out / np.linalg.norm(out)


def _fill(pose, bad):
    """Bad frames replaced by the straight path between their good neighbours (held at the ends)."""
    if not bad.any():
        return pose
    out = pose.copy()
    good = np.flatnonzero(~bad)
    for axis in range(3):
        out[:, axis] = np.interp(np.arange(len(pose)), good, pose[good, axis])
    for frame in np.flatnonzero(bad):
        after = np.searchsorted(good, frame)
        if after in (0, len(good)):
            out[frame, 3:] = pose[good[min(after, len(good) - 1)], 3:]
        else:
            a, b = good[after - 1], good[after]
            out[frame, 3:] = _slerp(pose[a, 3:], pose[b, 3:], (frame - a) / (b - a))
    return out


def _smooth(values):
    """Centred quadratic fit over SMOOTH_WINDOW frames; the ends use the nearest full window."""
    values = np.asarray(values, dtype=float)
    count, half = len(values), SMOOTH_WINDOW // 2
    fit = np.linalg.pinv(np.vander(np.arange(-half, half + 1), 3, increasing=True))     # coefficients of 1, k, k^2
    out = np.empty_like(values)
    for frame in range(count):
        centre = min(max(frame, half), count - 1 - half)
        constant, slope, bend = fit @ values[centre - half:centre + half + 1]
        offset = frame - centre
        out[frame] = constant + slope * offset + bend * offset * offset
    return out


def clean(episode):
    """The episode with faults repaired and sample-time jitter smoothed, and a JSON report of what changed.

    ``episode`` is what ``episode.load_episode`` returns and is left untouched.  Every event
    names its hand, kind, the first frame that shows the fault, the number of frames replaced
    and the size of the jump.
    """
    out = dict(episode)
    smooth = len(episode['t']) >= SMOOTH_WINDOW
    events = []; repaired = {}; moved = {}
    for side in ('left', 'right'):
        recorded = np.asarray(episode[side], dtype=float)
        stale, found = _stale(recorded)
        pose = _fill(recorded, stale)
        jumped, more = _jumps(pose)
        pose = _fill(pose, jumped)
        if smooth:
            rotation = _smooth(pose[:, 3:])
            pose = np.c_[_smooth(pose[:, :3]), rotation / np.linalg.norm(rotation, axis=1)[:, None]]
        kept = ~(stale | jumped)
        shift = np.linalg.norm(pose[kept, :3] - recorded[kept, :3], axis=1)
        events += [dict(hand=side, time_s=float(episode['t'][event['frame']]), **event)
                   for event in sorted(found + more, key=lambda event: event['frame'])]
        repaired[side] = int(np.sum(~kept))
        moved[side] = dict(mean=float(1e3 * shift.mean()), max=float(1e3 * shift.max()))
        out[side] = pose
    if smooth:
        out['head_yaw'] = _smooth(episode['head_yaw'])
    return out, dict(events=events, repaired_frames=repaired, moved_mm=moved,
                     smoothing_window_frames=SMOOTH_WINDOW if smooth else 0)
