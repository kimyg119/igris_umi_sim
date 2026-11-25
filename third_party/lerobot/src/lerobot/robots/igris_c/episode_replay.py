"""Recorded absolute position actions as the direct sender's low-rate source."""

import json
from pathlib import Path

import numpy as np


class EpisodeReplay:
    def __init__(self, path, contract):
        payload = json.loads(Path(path).read_text())
        rows = payload["rows"]
        self.timestamps = np.asarray(rows["timestamp"], float)
        self.actions = np.asarray(rows["action"], float)
        states = np.asarray(rows["observation.state"], float)
        n = len(self.timestamps)
        if (
            self.timestamps.shape != (n,)
            or n < 2
            or self.actions.shape != (n, len(contract.action_indices))
            or states.shape != (n, len(contract.state_indices))
            or not all(np.isfinite(a).all() for a in (self.timestamps, self.actions, states))
            or len(rows["episode_index"]) != n
            or any(i != payload["episode"] for i in rows["episode_index"])
        ):
            raise ValueError("episode must contain one finite, schema-matching numeric trajectory")
        self.period = 1 / contract.fps
        if not np.allclose(np.diff(self.timestamps), self.period, rtol=0.01, atol=1e-5):
            raise ValueError("episode timestamps must increase at dataset fps without gaps")
        self.timestamps -= self.timestamps[0]
        self.actions = np.asarray([contract.action(a) for a in self.actions])
        self.initial = np.zeros(31)
        self.initial[list(contract.state_indices)] = states[0]
        self.episode = payload["episode"]
        self.origin = None
        self.next_index = 0

    def update(self, target, now, measured):
        if self.origin is None:
            self.origin = now
        while True:
            index = min(self.next_index, len(self.timestamps) - 1)
            # Continue supplying the final position at source fps while the interpolator settles.
            extra = max(0, self.next_index - index) * self.period
            stamp = self.origin + self.timestamps[index] + extra
            if stamp > now:
                return
            target.submit(self.next_index, stamp, now, self.actions[index], measured, now)
            self.next_index += 1
