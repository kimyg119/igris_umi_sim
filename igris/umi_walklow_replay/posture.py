"""Arm-posture measures for a joint trajectory: where the elbows are and how often limits are met."""
import numpy as np

from .kinematics import ACTIVE, JOINTS, SHOULDERS


def swivel_deg(elbow, hand, side):
    """Elbow angle about the shoulder-to-hand line: 0 = hanging straight down, + = out to the side.

    ``elbow`` and ``hand`` are positions relative to the shoulder in torso axes.
    """
    axis = hand / np.linalg.norm(hand)
    down = np.array([0., 0., -1.])
    reference = down - axis * np.dot(axis, down)
    if np.linalg.norm(reference) < 1e-6:
        return 0.
    reference /= np.linalg.norm(reference)
    radial = elbow - axis * np.dot(elbow, axis)
    if np.linalg.norm(radial) < 1e-6:
        return 0.
    radial /= np.linalg.norm(radial)
    outward = np.cross(axis, reference) * (1. if side == 0 else -1.)
    return float(np.degrees(np.arctan2(np.dot(radial, outward), np.dot(radial, reference))))


def arm_posture(model, q_series, lower, upper):
    """Summary of a (T x 31) command series; ``lower``/``upper`` are the 17 IK joint bounds."""
    q_series = np.asarray(q_series, dtype=float)
    elbows = np.empty((len(q_series), 2, 3)); swivel = np.empty((len(q_series), 2))
    for k, q in enumerate(q_series):
        torso, elbow_frames, hand_frames, _ = model.arm_frames(q)
        inverse = np.linalg.inv(torso)
        for side in range(2):
            elbow = (inverse @ elbow_frames[side])[:3, 3] - SHOULDERS[side]
            hand = (inverse @ hand_frames[side])[:3, 3] - SHOULDERS[side]
            elbows[k, side] = elbow; swivel[k, side] = swivel_deg(elbow, hand, side)
    active = q_series[:, ACTIVE]
    at_limit = ((active <= lower + 1e-6) | (active >= upper - 1e-6)).mean(axis=0)
    outward = elbows[:, :, 1] * np.array([1., -1.])
    report = dict(
        samples=int(len(q_series)),
        swivel_mean_deg=swivel.mean(axis=0).tolist(),
        swivel_p05_deg=np.percentile(swivel, 5, axis=0).tolist(), swivel_p95_deg=np.percentile(swivel, 95, axis=0).tolist(),
        swivel_left_right_difference_mean_deg=float(np.mean(np.abs(swivel[:, 0] - swivel[:, 1]))),
        elbow_above_shoulder_fraction=(elbows[:, :, 2] > 0).mean(axis=0).tolist(),
        elbow_inside_shoulder_line_fraction=(outward < 0).mean(axis=0).tolist(),
        joints_at_limit_fraction={JOINTS[j]: float(v) for j, v in zip(ACTIVE, at_limit) if v > .01},
        any_arm_joint_at_limit_fraction=float(((active[:, 3:] <= lower[3:] + 1e-6) | (active[:, 3:] >= upper[3:] - 1e-6)).any(axis=1).mean()),
    )
    return report, elbows, swivel
