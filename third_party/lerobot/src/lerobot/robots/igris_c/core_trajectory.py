"""OnlineJointTrajectory reused from igris_core.

Source: igris-core/igris_core/igris_core/joint_trajectory.py
Revision: d5a00dcb545ae5f91b73d82128ce82d0894c006b
Only the online class is retained; transport remains in LeRobot.
"""

import math

import numpy as np


class OnlineJointTrajectory:
    """Causal C3 segments with an optional moving endpoint and a stopping tail.

    Preserve q, velocity, acceleration and jerk when replacing a segment.
    Moving targets pass through at the supplied velocity; missing updates run
    the precomputed braking tail, then hold. Limits and collision checks remain
    the executor's responsibility. All times use the caller's single clock.
    """

    _endpoint_inverse = np.array(
        [
            [35.0, -15.0, 2.5, -1 / 6],
            [-84.0, 39.0, -7.0, 0.5],
            [70.0, -34.0, 6.5, -0.5],
            [-20.0, 10.0, -2.0, 1 / 6],
        ]
    )

    def __init__(self, position):
        q = np.array(position, dtype=float, copy=True)
        if q.shape != (31,) or not np.isfinite(q).all():
            raise ValueError("finite (31,) initial position required")
        self.start_s = 0.0
        self.duration_s = 1.0
        self.coefficients = np.zeros((8, 31))
        self.coefficients[0] = q
        self.stop_coefficients = self.coefficients.copy()

    @classmethod
    def _segment(cls, start, end, duration):
        c = np.zeros((8, 31))
        c[:4] = start * np.array([1.0, duration, duration**2 / 2, duration**3 / 6])[:, None]
        residual = end * np.array([1.0, duration, duration**2, duration**3])[:, None]
        residual -= np.array([c[:4].sum(axis=0), c[1] + 2 * c[2] + 3 * c[3], 2 * c[2] + 6 * c[3], 6 * c[3]])
        c[4:] = cls._endpoint_inverse @ residual
        return c

    def _evaluate(self, coefficients, elapsed, derivative):
        u = np.clip(elapsed / self.duration_s, 0.0, 1.0)
        c = coefficients.copy()
        for _ in range(derivative):
            c = c[1:] * np.arange(1, len(c))[:, None] / self.duration_s
        return np.polynomial.polynomial.polyval(u, c).T

    def sample(self, seconds, derivative=0):
        t = np.asarray(seconds, dtype=float)
        if not np.isfinite(t).all() or np.any(t < self.start_s - 1e-9) or derivative not in (0, 1, 2, 3):
            raise ValueError("finite time after segment start and derivative 0..3 required")
        elapsed = t - self.start_s
        value = self._evaluate(self.coefficients, elapsed, derivative)
        if np.any(elapsed > self.duration_s):
            stopped = self._evaluate(self.stop_coefficients, elapsed - self.duration_s, derivative)
            value = np.where((elapsed > self.duration_s)[..., None], stopped, value)
        if derivative:
            value = np.where((elapsed >= 2 * self.duration_s)[..., None], 0.0, value)
        return value

    def retarget(self, seconds, position, duration_s, *, velocity=None):
        target = np.asarray(position, dtype=float)
        velocity = np.zeros(31) if velocity is None else np.asarray(velocity, dtype=float)
        if (
            target.shape != (31,)
            or not np.isfinite(target).all()
            or velocity.shape != (31,)
            or not np.isfinite(velocity).all()
            or not math.isfinite(duration_s)
            or duration_s <= 0
        ):
            raise ValueError("finite (31,) target/velocity and positive duration required")
        start = np.array([self.sample(seconds, d) for d in range(4)])
        h = float(duration_s)
        end = np.zeros((4, 31))
        end[0] = target
        end[1] = velocity
        stopped = np.zeros((4, 31))
        stopped[0] = target + 0.5 * h * velocity
        self.coefficients = self._segment(start, end, h)
        self.stop_coefficients = self._segment(end, stopped, h)
        self.start_s = float(seconds)
        self.duration_s = h
