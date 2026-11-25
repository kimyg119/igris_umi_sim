"""Pure simulation-only near-placement sequencing; no transport or state writes.

Every WalkTwist is zero. Waist references must be installed identically in the
SDK upper-body publisher and live IK caller axes, without modifying the motion
archive. The pelvis remains physically free under vendor balance control.

Lower/open targets use the same two-stage contract as WalkingTransfer's hybrid
mode: this class returns the local baseline and world-delta metadata; AFTER
bounded force preload, the live worker must use WorldPositionLowering with its
last accepted local goals and filtered torso. Thus world_hand_targets below
are baseline inputs, not the final hybrid IK targets during lower/open.
"""
import numpy as np
from walking_transfer import pose_matrix, smooth, wrap, yaw

WAIST_LIMITS=np.array([[-1.570796327,1.570796327],[-.314159265,.314159265],[-.872664626,.287979327]])


def _waist(value):
    q=np.asarray(value,dtype=float)
    if q.shape!=(3,) or not np.isfinite(q).all() or np.any(q<WAIST_LIMITS[:,0]) or np.any(q>WAIST_LIMITS[:,1]):
        raise ValueError('Invalid caller waist yaw/roll/pitch target or measurement')
    return q.copy()


def _hands(value):
    h=np.asarray(value,dtype=float)
    if h.shape!=(2,4,4) or not np.isfinite(h).all():raise ValueError('Expected accepted local EE pair')
    for m in h:
        if (not np.allclose(m[3],[0,0,0,1],atol=1e-9,rtol=0)
                or not np.allclose(m[:3,:3].T@m[:3,:3],np.eye(3),atol=1e-6,rtol=0)
                or np.linalg.det(m[:3,:3])<0):raise ValueError('Invalid local EE transform')
    if np.linalg.norm(h[0,:3,3]-h[1,:3,3])<.001:raise ValueError('Coincident hands')
    return h.copy()


class InplaceTransfer:
    def __init__(self,target_xyz,mass,mu,*,waist_end_yaw=-.6200024167299453):
        self.target=np.asarray(target_xyz,dtype=float).copy()
        if self.target.shape!=(3,) or not np.isfinite([*self.target,mass,mu,waist_end_yaw]).all() or min(mass,mu)<=0:
            raise ValueError('Invalid near target or payload')
        _waist([waist_end_yaw,0.,0.])
        self.mass=float(mass);self.required=1.5*mass*9.81/(2*mu)
        self.waist_end_yaw=float(waist_end_yaw)
        self.phase='pickup';self.started=None;self.command=np.zeros(3)

    def start(self,sim_time,pelvis,torso,payload,torso_relative_hands,waist_target,palm_tip_force,*,pickup_qualified):
        if self.started is not None:raise RuntimeError('In-place transfer already started')
        if not np.isfinite(sim_time) or sim_time<0:raise ValueError('Invalid simulation time')
        f=np.asarray(palm_tip_force,dtype=float)
        if (pickup_qualified is not True or f.shape!=(4,) or not np.isfinite(f).all()
                or np.any(f[[0,2]]<self.required)):raise ValueError('Force-qualified pickup is required')
        t=pose_matrix(torso);pose_matrix(pelvis);pose_matrix(payload)
        h=_hands(torso_relative_hands);w=_waist(waist_target)
        self.local_hands=h;self.clearance_start_hands=h.copy()
        self.clearance_direction=t[:3,:3].T@np.array([0.,0.,.04])
        self.clearance_start_z=float(payload[2]);self.waist_start=w;self.waist_target=w.copy()
        self.waist_end=w.copy();self.waist_end[0]=self.waist_end_yaw
        self.local_box_rotation=t[:3,:3].T@pose_matrix(payload)[:3,:3]
        self.last_xy=np.asarray(pelvis[:2]).copy();self.last_yaw=yaw(pelvis)
        self.last_box=np.asarray(payload[:3]).copy();self.last_waist=w.copy()
        self.velocity=np.zeros(2);self.yaw_rate=0.;self.box_velocity=np.zeros(3);self.waist_velocity=np.zeros(3)
        self.started=self.last_time=self.phase_started=float(sim_time);self.phase='clearance'
        self.clearance_ready_s=self.settled_s=self.supported_s=self.lost_palm_s=0.

    def update(self,sim_time,pelvis,torso,payload,waist_measured,palm_tip_force,source_support_z_N,destination_support_z_N):
        if self.started is None:raise RuntimeError('In-place transfer not started')
        if not np.isfinite(sim_time) or sim_time<self.last_time:raise ValueError('Nonmonotonic simulation time')
        dt=float(sim_time-self.last_time)
        if dt>.2:raise RuntimeError('In-place state sample gap exceeds 200 ms')
        if sim_time-self.started>85.:raise RuntimeError('In-place transfer exceeded 85 simulation seconds')
        t=pose_matrix(torso);pose_matrix(pelvis);pose_matrix(payload);w=_waist(waist_measured)
        p=np.asarray(pelvis);b=np.asarray(payload);f=np.asarray(palm_tip_force,dtype=float)
        if f.shape!=(4,) or not np.isfinite([*f,source_support_z_N,destination_support_z_N]).all():
            raise ValueError('Invalid actual native contact forces')
        if dt>0:
            alpha=dt/(.2+dt)
            self.velocity+=alpha*((p[:2]-self.last_xy)/dt-self.velocity)
            self.yaw_rate+=alpha*(wrap(yaw(p)-self.last_yaw)/dt-self.yaw_rate)
            self.box_velocity+=alpha*((b[:3]-self.last_box)/dt-self.box_velocity)
            self.waist_velocity+=alpha*((w-self.last_waist)/dt-self.waist_velocity)
        self.last_xy=p[:2].copy();self.last_yaw=yaw(p);self.last_box=b[:3].copy();self.last_waist=w
        self.last_time=float(sim_time);elapsed=sim_time-self.phase_started
        if self.phase in ('clearance','waist_arc'):
            self.lost_palm_s=0. if np.all(f[[0,2]]>=self.required) else self.lost_palm_s+dt
            if self.lost_palm_s>.6:raise RuntimeError('In-place carry lost friction-qualified palm contact')
        if self.phase=='clearance':
            self.local_hands=self.clearance_start_hands.copy()
            self.local_hands[:,:3,3]+=smooth(elapsed/8.)*self.clearance_direction
            ready=(elapsed>=8. and b[2]-self.clearance_start_z>=.032 and np.linalg.norm(self.box_velocity)<.02
                   and np.all(f[[0,2]]>=self.required) and source_support_z_N<.2*self.mass*9.81)
            self.clearance_ready_s=self.clearance_ready_s+dt if ready else 0.
            if self.clearance_ready_s>=.5:
                self.phase='waist_arc';self.phase_started=sim_time;elapsed=0.
            elif elapsed>20.:raise RuntimeError('No stable measured clearance lift before waist arc')
        if self.phase=='waist_arc':
            fraction=smooth(elapsed/20.)
            self.waist_target=self.waist_start+fraction*(self.waist_end-self.waist_start)
            settled=(elapsed>=20. and np.max(abs(w-self.waist_end))<.03
                     and np.max(abs(self.waist_velocity))<.04 and np.linalg.norm(self.velocity)<.03
                     and abs(self.yaw_rate)<.06 and np.linalg.norm(self.box_velocity)<.02)
            self.settled_s=self.settled_s+dt if settled else 0.
            if self.settled_s>=1.:
                self.phase='lower';self.phase_started=sim_time;elapsed=0.
                self.lower_local_start=self.local_hands.copy();self.lower_torso=t.copy()
                self.lower_delta=self.target-b[:3];self.lower_local_delta=t[:3,:3].T@self.lower_delta
            elif elapsed>30.:raise RuntimeError('Actual waist/body did not settle after in-place arc')
        local=self.local_hands.copy();release=0.;lower_fraction=None
        if self.phase in ('lower','open','done'):
            self.waist_target=self.waist_end.copy()
            lower_fraction=smooth(elapsed/12.) if self.phase=='lower' else 1.
            local=self.lower_local_start.copy();local[:,:3,3]+=lower_fraction*self.lower_local_delta
            if self.phase=='lower':
                supported=(elapsed>=12. and destination_support_z_N>.5*self.mass*9.81
                           and np.linalg.norm(b[:2]-self.target[:2])<.03 and abs(b[2]-self.target[2])<.025
                           and np.linalg.norm(self.box_velocity)<.02)
                self.supported_s=self.supported_s+dt if supported else 0.
                if self.supported_s>=.5:
                    self.phase='open';self.phase_started=sim_time;elapsed=0.
                elif elapsed>25.:raise RuntimeError('No stable destination support before release')
            release=smooth(elapsed/4.) if self.phase=='open' else 1. if self.phase=='done' else 0.
            normal=self.lower_local_start[0,:3,3]-self.lower_local_start[1,:3,3]
            normal/=np.linalg.norm(normal)
            local[:,:3,3]+=np.array([1.,-1.])[:,None]*.055*release*normal
            if self.phase=='open' and elapsed>=7.:self.phase='done'
        return dict(phase=self.phase,command_xyz=[0.,0.,0.],vendor_command_before_filter=[0.,0.,0.],
            waist_target_rad=self.waist_target.tolist(),nominal_waist_target_rad=self.waist_target.tolist(),
            waist_measured_rad=w.tolist(),waist_tracking_error_rad=(w-self.waist_target).tolist(),
            caller_sdk_indices=[0,1,2],caller_reference_schema='explicit_inplace_waist/v1',
            world_hand_targets=(t[None]@local).tolist(),torso_relative_hand_targets=local.tolist(),
            target_payload_xyz=self.target.tolist(),release_fraction=release,done=self.phase=='done',
            expected_payload_rotation_world=(t[:3,:3]@self.local_box_rotation).tolist(),
            world_position_lowering=True,lowering_frame='world_position_torso_orientation',lowering_duration_s=12.,
            lowering_fraction=lower_fraction,lowering_delta_world_m=self.lower_delta.tolist() if lower_fraction is not None else None,
            hand_target_contract='local_baseline_then_force_preload_then_live_WorldPositionLowering',
            clearance_lift_m=.04,clearance_measured_rise_m=float(b[2]-self.clearance_start_z),
            clearance_ready_s=self.clearance_ready_s,lost_palm_s=self.lost_palm_s,settled_s=self.settled_s,supported_s=self.supported_s)
