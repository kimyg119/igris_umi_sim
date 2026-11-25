"""Explicit SIM contact grace guarded by measured force and relative box pose.

This is a relaxed diagnostic, not a force-closure certificate. No attachment,
state reset, target modification, or contact-force invention is performed.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from htd_goal_hold import GoalHold


class GraspRetention:
    def __init__(self,mass_kg,friction,*,qualified_translation_limit_m=.015):
        if not np.isfinite([mass_kg,friction]).all() or min(mass_kg,friction)<=0:
            raise ValueError('Positive finite mass and friction required')
        self.required_each=1.5*mass_kg*9.81/(2*friction)
        self.required_total=mass_kg*9.81/friction
        if qualified_translation_limit_m not in (.015,.025):
            raise ValueError('Explicit relative translation limit must be15 or25mm')
        self.contract=dict(schema='htd_grasp_retention/v1',scope='SIM opt-in loaded goal hold only',
            regular_rule='each hand palm+tip meets1.5 weight/friction reserve',
            transient_rule='sum of all hand normal forces meets full weight/friction, reserve1.0',
            required_each_N=self.required_each,required_total_during_grace_N=self.required_total,
            maximum_continuous_contact_grace_s=.1,maximum_relative_translation_m=.015,
            qualified_translation_limit_m=qualified_translation_limit_m,
            wider_translation_condition='Both hands simultaneously meet regular friction reserve; no wider contact grace',
            maximum_relative_rotation_rad=float(np.deg2rad(15)),maximum_native_sample_age_s=.05,
            reference='actual payload relative to each measured EE, captured once before LOW',
            reset_interval='both hands simultaneously qualified; changing missing side does not reset',
            limitation='scalar friction reserve and bounded pose are diagnostic evidence, not force closure')
        self.events=[];self.metrics={};self.captured=False;self._bad_since=None
        self._last_native=None;self._last_pose=None;self._last_force=None

    def _read(self,s):
        native=float(s.native_time)
        if not np.isfinite(native) or native<0:
            raise ValueError('Finite native time required')
        times=[]
        for name,previous in [('pose',self._last_pose),('force',self._last_force)]:
            t=float(getattr(s,name+'_native_time'))
            if (not np.isfinite(t) or t<0 or not -1e-9<=native-t<=.05
                    or previous is not None and t<previous):
                raise ValueError('Fresh monotonic native '+name+' sample required')
            times.append(t)
        if self._last_native is not None and native<self._last_native:
            raise ValueError('Native time moved backward')
        force=np.asarray(s.palm_tip_force_N,dtype=float)
        if force.shape!=(4,) or not np.isfinite(force).all() or np.any(force<0):
            raise ValueError('Finite nonnegative hand forces required')
        box=GoalHold._pose(s.payload_pose,'payload_pose')
        hands=np.asarray(s.hand_poses,dtype=float)
        if hands.shape!=(2,7):
            raise ValueError('Two actual hand poses required')
        hands=np.stack([GoalHold._pose(x,'hand_pose') for x in hands])
        hr=Rotation.from_quat(hands[:,3:].copy(),scalar_first=True)
        br=Rotation.from_quat(box[3:].copy(),scalar_first=True)
        xyz=hr.inv().apply(box[:3]-hands[:,:3]);rot=hr.inv()*br
        return native,times,force.reshape(2,2).sum(axis=1),xyz,rot

    def capture(self,s):
        if self.captured:raise RuntimeError('Grasp reference may be captured only once')
        native,times,force,xyz,rot=self._read(s)
        if np.any(force<self.required_each):
            raise RuntimeError('Qualified contact on both hands required at capture')
        self._xyz,self._rot=xyz.copy(),rot
        self.captured=True
        self._last_native=native;self._last_pose,self._last_force=times
        self.events.append(dict(event='captured_actual_grasp',native_time=native,pose_native_time=times[0],
            force_native_time=times[1],relative_xyz_m=xyz.tolist(),relative_wxyz=rot.as_quat(scalar_first=True).tolist()))
        self.check(s)

    def check(self,s):
        if not self.captured:raise RuntimeError('Actual grasp reference required')
        native,times,force,xyz,rot=self._read(s)
        translation=np.linalg.norm(xyz-self._xyz,axis=1)
        rotation=(self._rot.inv()*rot).magnitude()
        both=bool(np.all(force>=self.required_each))
        limit=self.contract['qualified_translation_limit_m'] if both else self.contract['maximum_relative_translation_m']
        if np.any(translation>limit):
            raise RuntimeError(f'Payload relative translation exceeds{limit*1000:g}mm')
        if np.any(rotation>self.contract['maximum_relative_rotation_rad']):
            raise RuntimeError('Payload relative rotation exceeds15degrees')
        if self._bad_since is not None and native-self._bad_since>.1+1e-9:
            raise RuntimeError('Unqualified hand contact persisted beyond100ms')
        if not both:
            if force.sum()<self.required_total:
                raise RuntimeError('Transient grasp lacks total measured friction support')
            since=times[1] if self._bad_since is None else self._bad_since
            if native-since>.1+1e-9:
                raise RuntimeError('Unqualified hand contact persisted beyond100ms')
            if self._bad_since is None:
                self.events.append(dict(event='contact_grace_started',native_time=native,
                    force_native_time=times[1],force_per_hand_N=force.tolist()))
            self._bad_since=since
        elif self._bad_since is not None:
            self.events.append(dict(event='both_hands_recovered',native_time=native,
                                    duration_s=native-self._bad_since))
            self._bad_since=None
        self._last_native=native;self._last_pose,self._last_force=times
        self.metrics=dict(native_time=native,pose_native_time=times[0],force_native_time=times[1],
            force_per_hand_N=force.tolist(),total_force_N=float(force.sum()),both_hands_qualified=both,
            relative_translation_m=translation.tolist(),relative_rotation_rad=rotation.tolist(),
            active_translation_limit_m=limit,
            contact_grace_active=not both,contact_grace_elapsed_s=0. if both else native-self._bad_since)
