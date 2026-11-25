"""Pure route feedback for explicit SIM loaded navigation.

Caller qualifies grasp/source clearance and chooses activation. The actor/PD
clock, physical guards, support and release remain caller responsibilities.
The reviewed approach is opt-in; legacy route/controller behavior is the default.
"""
import math
import copy
from dataclasses import asdict
import numpy as np
from htd_goal_hold import GoalHold
from htd_navigation import NavigationGoals, navigation_config


class NavigationFeedback(GoalHold):
    def __init__(self,target_payload_xyz,reviewed_torso_heading,*,reviewed_approach=False,prealignment_coast_seconds=None):
        if type(reviewed_approach) is not bool:
            raise ValueError('reviewed_approach must be an explicit bool')
        from htd_reviewed_approach import coast_descriptor
        coast=coast_descriptor(prealignment_coast_seconds)
        if coast is not None and not reviewed_approach:
            raise ValueError('Coast requires explicit reviewed approach')
        self._coast_seconds=prealignment_coast_seconds
        self._coast_last_step=None
        super().__init__()
        self.reviewed_approach=reviewed_approach
        if reviewed_approach:
            from htd_reviewed_approach import ReviewedApproachController,geometry_review
            self._controller=ReviewedApproachController(1,device='cpu',prealignment_coast_seconds=prealignment_coast_seconds)
            self.contract.update(controller='htd_reviewed_approach.ReviewedApproachController',
                                 reviewed_approach=dict(enabled=True,geometry_review=geometry_review(),
                                    final_positive_desired_vx_m_s=.04,reverse_vx_m_s=.1,
                                    command_acceleration_m_s2=.15))
        if coast is not None: self.contract['prealignment_coast']=coast
        self._controller.cfg=navigation_config()
        self.navigation=None
        self.target=np.asarray(target_payload_xyz,dtype=float).copy()
        if self.target.shape!=(3,) or not np.isfinite(self.target).all() or not math.isfinite(reviewed_torso_heading):
            raise ValueError('Finite reviewed navigation destination required')
        self.reviewed_heading=float(reviewed_torso_heading)
        self.contract.update(schema='htd_navigation_feedback/v1',scope='SIM explicit loaded navigation feedback',
            config=asdict(self._controller.cfg),
            target_payload_xyz=self.target.tolist(),reviewed_torso_heading=self.reviewed_heading,
            goal_frame='Fixed captured torso/pelvis offset plus measured pelvis route waypoints',
            activation='Caller-qualified grasp and source clearance; route begins on first due actor tick',
            waypoint_transition='Measured pelvis planes; preserve command and pose-derivative history',
            transit_arrival_radius_m=.015,
            final_arrival_radius_m=.015,departure_radius_m=.03,
            success='Final waypoint and WorldXY settled; caller additionally checks actual pelvis arrival/support')
        if reviewed_approach:
            self.contract['waypoint_transition']='Measured source planes, then actual-root prealignment dwell; preserve controller history'

    @property
    def success(self):
        return (self.navigation is not None and self.navigation.route.final
                and (not self.reviewed_approach or self.navigation.route.failure is None)
                and not self.navigation.recovering and super().success)

    def capture(self,snapshot):
        if self._captured:
            raise RuntimeError('Navigation capture is allowed once')
        self._read(snapshot)
        if self.reviewed_approach:
            from htd_reviewed_approach import ReviewedNavigationGoals
            nav=ReviewedNavigationGoals(snapshot.pelvis_pose,snapshot.torso_pose,snapshot.payload_pose,
                self.target,self.reviewed_heading,native_time=snapshot.native_time,
                pose_native_time=snapshot.pose_native_time)
        else:
            nav=NavigationGoals(snapshot.pelvis_pose,snapshot.torso_pose,snapshot.payload_pose,
                                self.target,self.reviewed_heading)
        super().capture(snapshot)
        self.navigation=nav
        self.events.append(dict(event='captured_navigation',**nav.evidence()))

    def step(self,snapshot,dt=.02):
        if not self._captured or self.navigation is None:
            raise RuntimeError('Qualified navigation capture required')
        # Validate before mutating route/goal; GoalHold performs identical checks again.
        self._read(snapshot)
        if not math.isfinite(dt) or abs(dt-.02)>1e-12:
            raise ValueError('Navigation requires the declared .02-second actor period')
        if self._coast_seconds is not None and self._coast_last_step==snapshot.native_time:
            if snapshot.pose_native_time!=self._pose_time:
                raise ValueError('Changed pose stamp at repeated coast actor timestamp')
            return self._controller.command[0].numpy().copy()
        old=(self.navigation.route.index,self.navigation.recovering)
        if self.reviewed_approach:
            from htd_reviewed_approach import ReviewedApproachError
            try:
                self.navigation.update(snapshot.pelvis_pose,native_time=snapshot.native_time,
                                       pose_native_time=snapshot.pose_native_time)
            except ReviewedApproachError as error:
                self.censor_coast(str(error),snapshot.native_time)
                self.metrics['success']=False
                self.metrics['navigation']=self.navigation.evidence()
                self.events.append(dict(event='reviewed_approach_rejected',
                    native_time=snapshot.native_time,error=str(error),**self.navigation.evidence()))
                raise
        else:
            self.navigation.update(snapshot.pelvis_pose)
        self.navigation.apply(self._controller)
        self._goal=self.navigation.goal_torso_xy.copy()
        self._heading=self.navigation.heading
        if old!=(self.navigation.route.index,self.navigation.recovering):
            self.events.append(dict(event='navigation_waypoint_or_recovery',native_time=snapshot.native_time,
                                    **self.navigation.evidence()))
        command=super().step(snapshot,dt)
        self.metrics['navigation']=self.navigation.evidence()
        if self._coast_seconds is not None:
            self._coast_last_step=snapshot.native_time
            self.metrics['prealignment_coast']=self._controller.coast_evidence()
            self.events[-1]['prealignment_coast']=copy.deepcopy(self.metrics['prealignment_coast'])
        return command

    def censor_coast(self,reason,native_time=None):
        if self._coast_seconds is None: return
        self._controller.censor_coast(reason,native_time)
        self.metrics['prealignment_coast']=self._controller.coast_evidence()
        self.events.append(dict(event='prealignment_coast_terminal',reason=str(reason),
            native_time=native_time,prealignment_coast=copy.deepcopy(self.metrics['prealignment_coast'])))

    def arrival_signal(self,control_time_s,snapshot):
        if (not self._captured or self.navigation is None or not math.isfinite(control_time_s)
                or control_time_s<0 or not self.metrics
                or not 0<=snapshot.native_time-self.metrics['native_time']<=.05):
            raise ValueError('A fresh due-actor navigation measurement is required')
        self._read(snapshot)
        return dict(sample_time_s=float(control_time_s),
                    goal_xy_world=self.navigation.final_pelvis_xy.tolist(),
                    heading_goal_rad=self.navigation.heading,
                    route_final=self.navigation.route.final and not self.navigation.recovering,
                    feedback_settled=bool(self.success))
