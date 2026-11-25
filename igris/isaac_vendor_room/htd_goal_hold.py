"""Pure opt-in SIM feedback hold using the project's existing WorldXY controller.

Capture a qualified torso world-XY goal and pelvis heading once. The caller
owns due-actor scheduling, grasp/pose safety, command publication and duration.
This helper performs no IO, simulation, service calls or physical state writes.
"""
from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
import sys

import numpy as np
import torch


WBC_SRC = Path(__file__).resolve().parents[1]/'igris-c-wbc-base/src'
if str(WBC_SRC) not in sys.path:
    sys.path.insert(0, str(WBC_SRC))
from wbc_base import world_xy

if Path(world_xy.__file__).resolve() != WBC_SRC/'wbc_base/world_xy.py':
    raise ImportError('GoalHold requires this project wbc_base.world_xy source')


def _yaw(pose):
    w, x, y, z = pose[3:]
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class GoalHold:
    def __init__(self):
        cfg = world_xy.WorldXYConfig()
        self._controller = world_xy.WorldXYController(1, device='cpu', cfg=cfg)
        self.contract = dict(schema='htd_goal_hold/v1', scope='SIM bounded goal-hold diagnostic',
            controller='wbc_base.world_xy.WorldXYController',
            controller_source=str(WBC_SRC/'wbc_base/world_xy.py'), config=asdict(cfg),
            goal_frame='captured torso world XY', heading_frame='captured pelvis world yaw',
            command_frame='pelvis yaw', quaternion_order='wxyz', actor_dt_s=.02,
            velocity_source='successive torso world XY and wrapped pelvis yaw / native pose dt',
            repeated_pose_stamp='cache previous velocity and yaw rate; require identical poses',
            initial_velocity='zero at capture; no pre-capture velocity estimate',
            maximum_pose_native_age_s=.05, unit_quaternion_tolerance=1e-5,
            capture='once after caller qualification; no rebasing or physical reset',
            success='current WorldXY settled predicate; not grasp, delivery or navigation success')
        self.metrics = {}
        self.events = []
        self._captured = False
        self._last_native = self._pose_time = None
        self._torso = self._pelvis = None
        self._velocity = np.zeros(2)
        self._yaw_rate = 0.

    @staticmethod
    def _pose(value, name):
        p = np.asarray(value, dtype=float)
        if p.shape != (7,) or not np.isfinite(p).all():
            raise ValueError('Finite XYZ/WXYZ '+name+' required')
        if abs(np.linalg.norm(p[3:])-1.) > 1e-5:
            raise ValueError('Unit quaternion required for '+name)
        return p.copy()

    def _read(self, snapshot):
        native, stamp = float(snapshot.native_time), float(snapshot.pose_native_time)
        if not np.isfinite([native, stamp]).all() or min(native, stamp) < 0:
            raise ValueError('Finite nonnegative native pose clocks required')
        if not -1e-9 <= native-stamp <= .05+1e-9:
            raise ValueError('Native pose is future or older than .05 seconds')
        if self._last_native is not None and (native < self._last_native or stamp < self._pose_time):
            raise ValueError('Native or pose clock moved backward')
        torso = self._pose(snapshot.torso_pose, 'torso_pose')
        pelvis = self._pose(snapshot.pelvis_pose, 'pelvis_pose')
        if self._pose_time == stamp and (not np.array_equal(torso, self._torso)
                                         or not np.array_equal(pelvis, self._pelvis)):
            raise ValueError('Changed pose at an identical native pose stamp')
        return native, stamp, torso, pelvis

    @property
    def success(self):
        return self._captured and bool(self._controller.success[0])

    def _record(self, event, native, stamp, torso, pelvis, command, estimate):
        yaw = _yaw(pelvis)
        error = self._goal-torso[:2]
        self.metrics = dict(native_time=native, pose_native_time=stamp, pose_native_age_s=native-stamp,
            torso_pose=torso.tolist(), pelvis_pose=pelvis.tolist(), goal_xy_world=self._goal.tolist(),
            heading_goal_rad=self._heading, pelvis_yaw_rad=yaw,
            velocity_xy_world=self._velocity.tolist(), yaw_rate_rad_s=self._yaw_rate,
            velocity_estimate=estimate, position_error_world=error.tolist(),
            position_error_m=float(np.linalg.norm(error)), heading_error_rad=_wrap(self._heading-yaw),
            command_xyz=command.tolist(), arrived=bool(self._controller.arrived[0]),
            settled_s=float(self._controller.settled_s[0]), success=self.success)
        self.events.append(dict(event=event, **deepcopy(self.metrics)))

    def capture(self, snapshot):
        if self._captured:
            raise RuntimeError('Goal capture is allowed once; rebasing is forbidden')
        native, stamp, torso, pelvis = self._read(snapshot)
        self._goal, self._heading = torso[:2].copy(), _yaw(pelvis)
        self._controller.reset([0], torch.tensor(self._goal[None], dtype=torch.float32),
                               torch.tensor([self._heading], dtype=torch.float32))
        self._last_native, self._pose_time = native, stamp
        self._torso, self._pelvis = torso, pelvis
        self._captured = True
        self._record('captured_goal', native, stamp, torso, pelvis, np.zeros(3), 'zero_at_capture')

    def step(self, snapshot, dt=.02):
        if not self._captured:
            raise RuntimeError('Goal capture is required before step')
        if not math.isfinite(dt) or abs(dt-.02) > 1e-12:
            raise ValueError('GoalHold requires the declared .02-second actor period')
        native, stamp, torso, pelvis = self._read(snapshot)
        estimate = 'cached_repeated_pose_stamp'
        if stamp > self._pose_time:
            pose_dt = stamp-self._pose_time
            self._velocity = (torso[:2]-self._torso[:2])/pose_dt
            self._yaw_rate = _wrap(_yaw(pelvis)-_yaw(self._pelvis))/pose_dt
            estimate = 'native_pose_difference'
        command = self._controller.update(
            torch.tensor(torso[None, :2], dtype=torch.float32),
            torch.tensor([_yaw(pelvis)], dtype=torch.float32),
            torch.tensor(self._velocity[None], dtype=torch.float32),
            torch.tensor([self._yaw_rate], dtype=torch.float32), dt)[0].numpy().copy()
        self._last_native, self._pose_time = native, stamp
        self._torso, self._pelvis = torso, pelvis
        self._record('goal_hold_update', native, stamp, torso, pelvis, command, estimate)
        return command
