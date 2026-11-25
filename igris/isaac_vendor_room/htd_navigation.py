"""Offline-tested goal adapter; not yet enabled in a simulator transport scope.

The caller must qualify actual grasp and source clearance before activation.
It owns no actuator, timing, grasp, IK, placement, or completion decision.
WorldXY commands remain torso-feedback velocities in the pelvis-yaw frame;
source-clearance planes are measured at the pelvis. No V5 command shaping.
"""
from dataclasses import replace
import numpy as np
import torch

from heading_route import HeadingRoute
from htd_goal_hold import GoalHold, _yaw, _wrap, world_xy


def navigation_config():
    """Explicit transport config evaluated on64 captured-stance rollouts.

    The15/30mm hysteresis restarts before drift can sit outside the35mm
    measured pelvis release region. Default ten-second hold stays unchanged.
    """
    return replace(world_xy.WorldXYConfig(), position_gain=2.5,
                   position_filter_s=.2, heading_gain=4.,
                   arrival_radius=.015, departure_radius=.03)


class NavigationGoals:
    def __init__(self, pelvis_pose, torso_pose, payload_pose, target_payload_xyz,
                 reviewed_torso_heading):
        pelvis=GoalHold._pose(pelvis_pose,'pelvis_pose')
        torso=GoalHold._pose(torso_pose,'torso_pose')
        payload=GoalHold._pose(payload_pose,'payload_pose')
        target=np.asarray(target_payload_xyz,dtype=float)
        if target.shape!=(3,) or not np.isfinite(target).all():
            raise ValueError('Finite target payload XYZ required')
        if (not np.isfinite(reviewed_torso_heading)
                or abs(_wrap(_yaw(torso)-reviewed_torso_heading))>.12):
            raise ValueError('Measured torso heading differs from reviewed approach')
        self.heading=_yaw(pelvis)
        self.torso_offset=torso[:2]-pelvis[:2]
        self.payload_offset=payload[:2]-pelvis[:2]
        self.final_pelvis_xy=target[:2]-self.payload_offset
        self.final_torso_xy=self.final_pelvis_xy+self.torso_offset
        self.route=HeadingRoute(pelvis[:2],self.final_pelvis_xy,self.heading)
        self.goal_torso_xy=self.route.goal+self.torso_offset
        self.recovering=False
        self.target_payload_xyz=target.copy()
        self.capture=dict(pelvis_pose=pelvis.tolist(),torso_pose=torso.tolist(),
                          payload_pose=payload.tolist(),reviewed_torso_heading=float(reviewed_torso_heading))
        self._original=world_xy.WorldXYConfig()
        self._default=navigation_config()
        self._transit=self._default
        self._applied=None

    def update(self,pelvis_pose):
        pelvis=GoalHold._pose(pelvis_pose,'pelvis_pose')
        advanced=self.route.advance(pelvis[:2],settled=False)
        recovery=self.route.clearance_recovery_goal(pelvis[:2])
        self.recovering=recovery is not None
        root_goal=self.route.goal if recovery is None else recovery
        self.goal_torso_xy=root_goal+self.torso_offset
        return advanced

    def apply(self,controller):
        """Change only goal/latch/config; preserve command/filter/derivative history."""
        if (not isinstance(controller,world_xy.WorldXYController)
                or tuple(controller.goal.shape)!=(1,2)
                or controller.cfg not in (self._original,self._default,self._transit)):
            raise ValueError('Single WorldXY controller with reviewed defaults required')
        cfg=self._default if self.route.final else self._transit
        signature=(id(controller),*self.goal_torso_xy,float(self.heading),cfg)
        if signature==self._applied:
            return
        controller.goal[0]=torch.as_tensor(self.goal_torso_xy,dtype=controller.goal.dtype,
                                            device=controller.goal.device)
        controller.heading[0]=self.heading
        controller.cfg=cfg
        controller.arrived[:]=False
        controller.settled_s[:]=0.
        self._applied=signature

    def evidence(self):
        return dict(schema='htd_navigation_goals/v1',scope='pure goal adapter; no transport validation',
            capture=self.capture,heading_frame='pelvis yaw',feedback_frame='torso world XY',
            measured_route_frame='pelvis world XY',fixed_torso_offset_xy=self.torso_offset.tolist(),
            fixed_payload_offset_xy=self.payload_offset.tolist(),heading_goal_rad=self.heading,
            target_payload_xyz=self.target_payload_xyz.tolist(),final_pelvis_xy=self.final_pelvis_xy.tolist(),
            final_torso_xy=self.final_torso_xy.tolist(),goal_torso_xy=self.goal_torso_xy.tolist(),
            source_recovery=self.recovering,route=self.route.evidence(),
            transit_arrival_radius_m=.015,final_arrival_radius_m=.015,departure_radius_m=.03,
            limitation='Fixed torso/pelvis offsets are approximate; caller must check actual root/torso arrival separately')
