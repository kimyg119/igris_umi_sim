"""Bounded shared wrist-target rotation from observed payload orientation."""
import numpy as np
from scipy.spatial.transform import Rotation


class BoxOrientationFeedback:
    def __init__(self):
        self.correction = Rotation.identity()
        self.error_deg = 0.

    def update(self, expected, measured, yaw, dt, active):
        if not np.isfinite(yaw) or not 0 < dt <= .1:
            raise ValueError('Invalid orientation feedback time/yaw')
        expected=np.asarray(expected,float);measured=np.asarray(measured,float)
        if expected.shape!=(3,3) or measured.shape!=(3,3) or not np.isfinite([expected,measured]).all():
            raise ValueError('Expected finite payload rotation matrices')
        heading=Rotation.from_euler('z',yaw)
        error=Rotation.from_matrix(expected@measured.T).as_rotvec()
        self.error_deg=float(np.rad2deg(np.linalg.norm(error)))
        if active:
            # Integrate in the frame before intentional yaw. A stored correction
            # then follows the carried object as the waist turns.
            velocity=.8*heading.inv().apply(error)
            velocity*=min(1.,np.deg2rad(.3)/max(np.linalg.norm(velocity),1e-12))
            candidate=Rotation.from_rotvec(velocity*dt)*self.correction
            vector=candidate.as_rotvec()
            vector*=min(1.,np.deg2rad(10)/max(np.linalg.norm(vector),1e-12))
            self.correction=Rotation.from_rotvec(vector)
        return (heading*self.correction*heading.inv()).as_matrix()


def rotate_target(pose, center, rotation):
    target=np.asarray(pose).copy()
    target[:3,3]=center+rotation@(target[:3,3]-center)
    target[:3,:3]=rotation@target[:3,:3]
    return target
