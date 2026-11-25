"""Simulation-only loaded navigation and measured-support release sequencing.

Targets are real EE goals. This module never edits object or robot state.
Carry goals follow the measured torso; lowering defaults to fixed-world goals.
An explicit local-lowering experiment preserves torso-relative orientation and
uses a slower12s path; actual contact/placement gates still control release.
The caller supplies contact force from the authoritative native physics step.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def pose_matrix(pose):
    p=np.asarray(pose,dtype=float)
    if p.shape!=(7,) or not np.isfinite(p).all() or abs(np.linalg.norm(p[3:])-1)>.001:
        raise ValueError('Expected finite unit-wxyz pose')
    result=np.eye(4);result[:3,3]=p[:3]
    result[:3,:3]=Rotation.from_quat(p[3:].copy(),scalar_first=True).as_matrix()
    return result


def yaw(pose):
    r=pose_matrix(pose)[:3,:3]
    return float(np.arctan2(r[1,0],r[0,0]))


def wrap(angle):return float(np.arctan2(np.sin(angle),np.cos(angle)))


def smooth(value):
    u=float(np.clip(value,0,1));return u**3*(10-15*u+6*u*u)


class WalkingTransfer:
    def __init__(self,target_xyz,torso_heading,mass,mu,*,clearance_lift=0.,torso_relative_lowering=False,world_position_lowering=False,heading_preserving_route=False,xy_first_lowering=False,arrival_settle_seconds=1.5,clearance_minimum_rise=None,stationary_only=False,arrival_brake_radius=None,adaptive_clearance=False):
        self.target=np.asarray(target_xyz,dtype=float)
        if self.target.shape!=(3,) or not np.isfinite([*self.target,torso_heading,mass,mu]).all() or min(mass,mu)<=0:
            raise ValueError('Invalid walking destination or payload')
        self.heading=float(torso_heading);self.mass=mass;self.required=1.5*mass*9.81/(2*mu)
        if not isinstance(heading_preserving_route,bool):raise ValueError('Heading route option must be boolean')
        self.heading_preserving_route=heading_preserving_route;self.route=None
        if not isinstance(stationary_only,bool):raise ValueError('Stationary diagnostic option must be boolean')
        self.stationary_only=stationary_only
        if arrival_brake_radius is not None and (not heading_preserving_route
                or not np.isfinite(arrival_brake_radius) or not .015<=arrival_brake_radius<=.025):
            raise ValueError('Explicit arrival braking requires heading route and15..25mm radius')
        self.arrival_brake_radius=arrival_brake_radius
        if not np.isfinite(clearance_lift)or not 0<=clearance_lift<=.06:raise ValueError('Clearance lift must be between0 and60mm')
        self.clearance_lift=float(clearance_lift);self.clearance_ready_s=0.
        if clearance_minimum_rise is not None and (not heading_preserving_route
                or not np.isfinite(clearance_minimum_rise) or not 0<clearance_minimum_rise<=.8*clearance_lift):
            raise ValueError('Explicit minimum clearance requires heading route and bounded positive measured rise')
        self.clearance_minimum_rise=.8*self.clearance_lift if clearance_minimum_rise is None else float(clearance_minimum_rise)
        if (type(adaptive_clearance) is not bool or adaptive_clearance and
                (not heading_preserving_route or clearance_lift<=0 or self.clearance_minimum_rise<.02)):
            raise ValueError('Adaptive clearance requires reviewed heading route, positive lift and minimum20mm rise')
        self.adaptive_clearance=adaptive_clearance
        self.clearance_requested_fraction=0.
        self.clearance_requested_cap_fraction=None
        self.clearance_cap_trigger=None
        if not isinstance(torso_relative_lowering,bool):raise ValueError('Local lowering option must be boolean')
        self.torso_relative_lowering=torso_relative_lowering
        if not isinstance(world_position_lowering,bool) or world_position_lowering and not torso_relative_lowering:
            raise ValueError('World-position lowering requires torso-following orientations')
        self.world_position_lowering=world_position_lowering
        if (not isinstance(xy_first_lowering,bool) or xy_first_lowering and
                not (world_position_lowering and heading_preserving_route)):
            raise ValueError('XY-first lowering requires reviewed heading route and world-position lowering')
        if (not np.isfinite(arrival_settle_seconds) or not .3<=arrival_settle_seconds<=1.5
                or arrival_settle_seconds!=1.5 and not heading_preserving_route):
            raise ValueError('Short arrival dwell requires reviewed heading route and0.3..1.5seconds')
        self.xy_first_lowering=xy_first_lowering;self.arrival_settle_seconds=float(arrival_settle_seconds)
        self.lower_duration=16. if xy_first_lowering else 12. if torso_relative_lowering else 8.
        self.phase='pickup';self.command=np.zeros(3);self.started=None
        self.external_goal=None

    def _navigation_goal(self,pelvis,torso,payload):
        if self.stationary_only:
            self.goal_xy=np.asarray(pelvis[:2],dtype=float).copy()
            self.goal_yaw=yaw(pelvis)
            return
        if self.heading_preserving_route:
            from heading_route import HeadingRoute
            if abs(wrap(self.heading-yaw(torso)))>.12:
                raise ValueError('Measured grasp heading differs from reviewed cart approach')
            self.goal_yaw=yaw(pelvis)
            final=self.target[:2]-(np.asarray(payload[:2])-np.asarray(pelvis[:2]))
            self.route=HeadingRoute(pelvis[:2],final,self.goal_yaw)
            self.goal_xy=self.route.goal
            return
        turn=wrap(self.heading-yaw(torso));self.goal_yaw=wrap(yaw(pelvis)+turn)
        r=Rotation.from_euler('z',turn).as_matrix()
        self.goal_xy=self.target[:2]-(r@(np.asarray(payload[:3])-np.asarray(pelvis[:3])))[:2]

    def start(self,sim_time,pelvis,torso,payload,world_hands,*,torso_relative_hands=None):
        if self.started is not None:raise RuntimeError('Walking transfer already started')
        if not np.isfinite(sim_time) or sim_time<0:raise ValueError('Invalid simulation time')
        h=np.asarray(world_hands,dtype=float)
        if h.shape!=(2,4,4) or not np.isfinite(h).all():raise ValueError('Expected two world EE transforms')
        for matrix in h:
            if not np.allclose(matrix[3],[0,0,0,1]) or not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-6):
                raise ValueError('Invalid world EE transform')
        self.local_hands=np.linalg.inv(pose_matrix(torso))[None]@h
        if torso_relative_hands is not None:
            relative=np.asarray(torso_relative_hands,dtype=float)
            if relative.shape!=(2,4,4) or not np.isfinite(relative).all():
                raise ValueError('Expected two validated torso-relative EE transforms')
            for matrix in relative:
                if not np.allclose(matrix[3],[0,0,0,1]) or not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-6):
                    raise ValueError('Invalid torso-relative EE transform')
            # Preserve the exact local goals that the live solver accepted.
            # Reconstructing them through the raw torso would import the
            # filtered-world target's frame lag into the carried grasp.
            self.local_hands=relative.copy()
        self.local_box_rotation=pose_matrix(torso)[:3,:3].T@pose_matrix(payload)[:3,:3]
        pose_matrix(payload);pose_matrix(pelvis)
        self._navigation_goal(pelvis,torso,payload)
        self.clearance_start_hands=self.local_hands.copy()
        self.clearance_direction=pose_matrix(torso)[:3,:3].T@np.array([0.,0.,self.clearance_lift])
        self.clearance_start_box_z=float(payload[2])
        self.started=self.last_time=float(sim_time);self.phase='clearance'if self.clearance_lift else'carry'
        self.last_xy=np.array(pelvis[:2]);self.velocity=np.zeros(2)
        self.last_yaw=yaw(pelvis);self.yaw_rate=0.;self.turning=True;self.translating=True
        self.translation_heading_ready=False
        self.source_recovery=False
        self.arrival_coast_until=-1.
        self.last_box=np.array(payload[:3]);self.box_velocity=np.zeros(3)
        self.lost_palm_s=0.;self.settled_s=0.;self.supported_s=0.;self.phase_started=sim_time

    def _begin_lower(self,sim_time,world_hands,torso_matrix,payload):
        self.phase='lower';self.phase_started=sim_time;self.command[:]=0
        self.lower_start=world_hands.copy();self.lower_delta=self.target-payload[:3]
        self.lower_local_start=self.local_hands.copy()
        self.lower_local_delta=torso_matrix[:3,:3].T@self.lower_delta
        self.lower_torso_rotation=torso_matrix[:3,:3].copy()

    def update(self,sim_time,pelvis,torso,payload,palm_tip_force,support_z_N,*,hold_carry=False,combined_carry_contact=False,external_navigation=None,ik_position_error_m=None):
        if self.started is None:raise RuntimeError('Walking transfer is not started')
        # The sole live-HL owner must supply the latest accepted segment's error.
        # Validate before mutating clocks/filters; missing evidence cannot silently
        # fall back to the nominal path, even after the cap has been captured.
        if self.adaptive_clearance and self.phase=='clearance':
            if (isinstance(ik_position_error_m,(bool,np.bool_))
                    or not isinstance(ik_position_error_m,(int,float,np.integer,np.floating))
                    or not np.isfinite(ik_position_error_m) or ik_position_error_m<0):
                raise ValueError('Adaptive clearance requires finite nonnegative accepted IK position error')
        if not isinstance(hold_carry,bool):raise ValueError('Carry hold must be boolean')
        hold_carry=hold_carry or self.stationary_only
        external=external_navigation is not None
        if self.external_goal is not None and not external:
            raise ValueError('External navigation cannot fall back to vendor control')
        if external:
            if (hold_carry or not self.heading_preserving_route or not self.xy_first_lowering
                    or self.phase=='clearance' or not isinstance(external_navigation,dict)):
                raise ValueError('External navigation requires active reviewed transport carry scope')
            signal=external_navigation
            goal=np.asarray(signal.get('goal_xy_world'),dtype=float)
            values=[signal.get('sample_time_s',np.nan),signal.get('heading_goal_rad',np.nan)]
            if (goal.shape!=(2,) or not np.isfinite(goal).all() or not np.isfinite(values).all()
                    or not 0<=sim_time-values[0]<=.05
                    or type(signal.get('route_final')) is not bool
                    or type(signal.get('feedback_settled')) is not bool):
                raise ValueError('Fresh external navigation goal and qualification required')
            fixed=np.r_[goal,values[1]]
            if self.external_goal is not None and not np.array_equal(self.external_goal,fixed):
                raise ValueError('External final goal must stay fixed after capture')
            self.external_goal=fixed.copy()
            self.goal_xy=goal.copy();self.goal_yaw=float(values[1])
        if not isinstance(combined_carry_contact,bool) or (combined_carry_contact and not (hold_carry or external)):
            raise ValueError('Combined contact requires explicit held carry scope')
        if not np.isfinite(sim_time) or sim_time<self.last_time:raise ValueError('Nonmonotonic simulation time')
        dt=float(sim_time-self.last_time)
        p=np.asarray(pelvis,dtype=float);b=np.asarray(payload,dtype=float)
        torso_matrix=pose_matrix(torso);pose_matrix(p);pose_matrix(b)
        f=np.asarray(palm_tip_force,dtype=float)
        if f.shape!=(4,) or not np.isfinite([*f,support_z_N]).all() or np.any(f<0):raise ValueError('Invalid native forces')
        if sim_time-self.started>180:raise RuntimeError('Walking transfer exceeded 180 simulation seconds')
        if dt>0:
            alpha=dt/(.2+dt)
            self.velocity+=alpha*((p[:2]-self.last_xy)/dt-self.velocity)
            self.yaw_rate+=alpha*(wrap(yaw(p)-self.last_yaw)/dt-self.yaw_rate)
            self.box_velocity+=alpha*((b[:3]-self.last_box)/dt-self.box_velocity)
        self.last_yaw=yaw(p)
        self.last_xy=p[:2].copy();self.last_box=b[:3].copy();self.last_time=float(sim_time)
        relative_hands=None;lower_fraction=None;lower_xy_fraction=None
        if self.phase in ('clearance','carry'):
            self.expected_box_rotation=torso_matrix[:3,:3]@self.local_box_rotation
            qualified=(f.reshape(2,2).sum(axis=1) if combined_carry_contact and self.phase=='carry' else f[[0,2]])
            self.lost_palm_s=0. if np.all(qualified>=self.required) else self.lost_palm_s+dt
            if self.lost_palm_s>.6:raise RuntimeError('Loaded walking lost friction-qualified palm contact')
            if self.phase=='clearance':
                elapsed=sim_time-self.phase_started
                self.command[:]=0.
                measured_ready=(b[2]-self.clearance_start_box_z>=self.clearance_minimum_rise
                       and np.linalg.norm(self.box_velocity)<.02 and np.all(f[[0,2]]>=self.required)
                       and support_z_N<.2*self.mass*9.81)
                if (self.adaptive_clearance and self.clearance_requested_cap_fraction is None
                        and ik_position_error_m>=.010 and measured_ready):
                    # Freeze the PREVIOUS request once; neither advance to this
                    # tick's path point nor rebase the request on measured box Z.
                    self.clearance_requested_cap_fraction=self.clearance_requested_fraction
                    self.clearance_cap_trigger=dict(native_time_s=float(sim_time),
                        ik_position_error_m=float(ik_position_error_m),
                        measured_rise_m=float(b[2]-self.clearance_start_box_z),
                        requested_fraction=self.clearance_requested_fraction,
                        effective_ee_rise_m=self.clearance_requested_fraction*self.clearance_lift)
                fraction=smooth(elapsed/8.)
                if self.clearance_requested_cap_fraction is not None:
                    fraction=min(fraction,self.clearance_requested_cap_fraction)
                self.clearance_requested_fraction=fraction
                self.local_hands=self.clearance_start_hands.copy()
                self.local_hands[:,:3,3]+=fraction*self.clearance_direction
                ready=elapsed>=8. and measured_ready
                self.clearance_ready_s=self.clearance_ready_s+dt if ready else 0.
                if self.clearance_ready_s>=.5:
                    self._navigation_goal(p,torso,b);self.phase='carry';self.phase_started=sim_time
                elif elapsed>20.:raise RuntimeError('No stable measured clearance lift before walking')
            world_hands=torso_matrix[None]@self.local_hands
            relative_hands=self.local_hands
        if self.phase=='carry' and hold_carry:
            self.command[:]=0.
            self.settled_s=0.
        if self.phase=='carry' and external:
            self.command[:]=0.
            distance=float(np.linalg.norm(self.goal_xy-p[:2]))
            settled=(signal['route_final'] and signal['feedback_settled'] and distance<.035
                     and abs(wrap(self.goal_yaw-yaw(p)))<.08
                     and np.linalg.norm(self.velocity)<.03 and abs(self.yaw_rate)<.06)
            self.settled_s=self.settled_s+dt if settled else 0.
            if self.settled_s>=self.arrival_settle_seconds:
                self._begin_lower(sim_time,world_hands,torso_matrix,b)
        if self.phase=='carry' and not hold_carry and not external:
            recovery_goal=None
            if self.route is not None:
                if self.route.advance(p[:2],settled=False):
                    self.goal_xy=self.route.goal;self.settled_s=0.;self.translating=True
                    self.command[:2]=0.
                recovery_goal=self.route.clearance_recovery_goal(p[:2])
                recovering=recovery_goal is not None
                if recovering!=self.source_recovery:self.command[:2]=0.
                self.source_recovery=recovering
            steering_goal=self.goal_xy if recovery_goal is None else recovery_goal
            error=steering_goal-p[:2];angle=yaw(p);heading_error=wrap(self.goal_yaw-angle)
            c,s=np.cos(angle),np.sin(angle)
            distance=float(np.linalg.norm(error))
            # Vendor V5 scales raw lateral input by .5, then applies .2
            # planar/yaw deadbands before and after its .96/.04 filter.
            # Use explicit move/stop hysteresis; sub-deadband creep cannot
            # converge. Turn first to reduce simultaneous loaded motions.
            if abs(heading_error)<.035+.12*abs(self.yaw_rate):self.turning=False
            elif abs(heading_error)>.08:self.turning=True
            approach_speed=float(np.dot(error,self.velocity)/max(distance,1e-12))
            routed_arrival=self.route is not None and self.route.final
            stop_radius=(min(.065,.025+.35*max(0.,approach_speed)) if routed_arrival
                         else .025+.12*max(0.,approach_speed))
            if routed_arrival and self.arrival_brake_radius is not None:
                stop_radius=self.arrival_brake_radius
            arrival_radius=.035 if routed_arrival else .045
            transit=self.route is not None and not self.route.final
            if transit:self.translating=True
            elif routed_arrival:
                if self.translating and distance<stop_radius:
                    self.translating=False;self.arrival_coast_until=sim_time+2.5
                elif sim_time>=self.arrival_coast_until and distance>arrival_radius:
                    self.translating=True
            elif distance<stop_radius:self.translating=False
            elif distance>arrival_radius:self.translating=True
            arrival_coasting=routed_arrival and sim_time<self.arrival_coast_until
            if arrival_coasting:self.turning=False
            # Qualify the end of a turn once. Normal gait yaw oscillation must
            # not repeatedly zero the forward ramp below the vendor deadband.
            # A new turn or XY stop clears qualification; final settling below
            # still continuously requires small measured angular velocity.
            if self.turning or not self.translating or abs(heading_error)>=.08:self.translation_heading_ready=False
            elif abs(heading_error)<.08 and abs(self.yaw_rate)<.06:self.translation_heading_ready=True
            desired_xy=np.zeros(2)
            if self.translating and not self.turning and self.translation_heading_ready:
                desired_xy=np.array([[c,s],[-s,c]])@error*(.23/max(distance,1e-12))
            desired_wz=float(np.copysign(.23,heading_error)) if self.turning else 0.
            delta=desired_xy-self.command[:2]
            self.command[:2]+=delta*min(1.,.10*dt/max(np.linalg.norm(delta),1e-12))
            self.command[2]+=np.clip(desired_wz-self.command[2],-.20*dt,.20*dt)
            if not np.any(desired_xy):self.command[:2]=0.
            if desired_wz==0.:self.command[2]=0.
            settled=not transit and distance<arrival_radius and abs(heading_error)<.08 and np.linalg.norm(self.velocity)<.03 and abs(self.yaw_rate)<.06
            self.settled_s=self.settled_s+dt if settled else 0.
            if self.settled_s>=self.arrival_settle_seconds:
                self._begin_lower(sim_time,world_hands,torso_matrix,b)
        if self.phase in ('lower','open','done'):
            self.command[:]=0.
            path_hands=self.lower_local_start.copy() if self.torso_relative_lowering else self.lower_start.copy()
            delta=self.lower_local_delta if self.torso_relative_lowering else self.lower_delta
            elapsed=sim_time-self.phase_started
            if self.xy_first_lowering:
                lower_fraction=smooth((elapsed-4.)/12.) if self.phase=='lower' else 1.
                lower_xy_fraction=smooth(elapsed/4.) if self.phase=='lower' else 1.
                path_delta=self.lower_delta*np.array([lower_xy_fraction,lower_xy_fraction,lower_fraction])
                path_hands[:,:3,3]+=self.lower_torso_rotation.T@path_delta
            else:
                lower_fraction=smooth(elapsed/self.lower_duration) if self.phase=='lower' else 1.
                lower_xy_fraction=lower_fraction
                path_hands[:,:3,3]+=lower_fraction*delta
            if self.phase=='lower':
                supported=(sim_time-self.phase_started>=self.lower_duration and support_z_N>.5*self.mass*9.81
                           and np.linalg.norm(b[:2]-self.target[:2])<.03
                           and abs(b[2]-self.target[2])<.025 and np.linalg.norm(self.box_velocity)<.02)
                self.supported_s=self.supported_s+dt if supported else 0.
                if self.supported_s>=.5:self.phase='open';self.phase_started=sim_time
                elif sim_time-self.phase_started>25:raise RuntimeError('No stable destination support before release')
            release=smooth((sim_time-self.phase_started)/4.) if self.phase=='open' else 1. if self.phase=='done' else 0.
            normal=path_hands[0,:3,3]-path_hands[1,:3,3];normal/=np.linalg.norm(normal)
            path_hands[0,:3,3]+=.055*release*normal;path_hands[1,:3,3]-=.055*release*normal
            if self.torso_relative_lowering:
                relative_hands=path_hands
                world_hands=torso_matrix[None]@path_hands
                self.expected_box_rotation=torso_matrix[:3,:3]@self.local_box_rotation
            else:
                relative_hands=None;world_hands=path_hands
            if self.phase=='open' and sim_time-self.phase_started>=7:self.phase='done'
        else:release=0.
        return dict(phase=self.phase,command_xyz=(self.command*np.array([1.,2.,1.])).tolist(),
                    vendor_command_before_filter=self.command.tolist(),world_hand_targets=world_hands.tolist(),
                    torso_relative_hand_targets=None if relative_hands is None else relative_hands.tolist(),
                    goal_xy=self.goal_xy.tolist(),goal_yaw=self.goal_yaw,target_payload_xyz=self.target.tolist(),
                    release_fraction=release,done=self.phase=='done',position_error_m=float(np.linalg.norm(self.goal_xy-p[:2])),
                    expected_payload_rotation_world=self.expected_box_rotation.tolist(),
                    lost_palm_s=self.lost_palm_s,settled_s=self.settled_s,supported_s=self.supported_s,
                    translation_heading_ready=self.translation_heading_ready,
                    source_clearance_recovery=self.source_recovery,
                    arrival_coasting=self.phase=='carry' and sim_time<self.arrival_coast_until,
                    navigation_route=None if external or self.route is None else self.route.evidence(),
                    navigation_owner='external_htd' if external else 'vendor_v5',
                    lowering_frame='world_position_torso_orientation' if self.world_position_lowering else 'torso' if self.torso_relative_lowering else 'world',lowering_duration_s=self.lower_duration,
                    world_position_lowering=self.world_position_lowering,lowering_fraction=lower_fraction,
                    lowering_xy_fraction=lower_xy_fraction,xy_first_lowering=self.xy_first_lowering,
                    arrival_settle_seconds=self.arrival_settle_seconds,arrival_brake_radius_m=self.arrival_brake_radius,
                    lowering_delta_world_m=self.lower_delta.tolist() if lower_fraction is not None else None,
                    clearance_lift_m=self.clearance_lift,clearance_measured_rise_m=float(b[2]-self.clearance_start_box_z),
                    clearance_ready_s=self.clearance_ready_s,payload_speed_m_s=float(np.linalg.norm(self.box_velocity)),
                    clearance_minimum_rise_m=self.clearance_minimum_rise,
                    adaptive_clearance=self.adaptive_clearance,
                    clearance_requested_fraction=self.clearance_requested_fraction,
                    clearance_requested_cap_fraction=self.clearance_requested_cap_fraction,
                    clearance_effective_ee_rise_m=self.clearance_requested_fraction*self.clearance_lift,
                    clearance_cap_trigger=None if self.clearance_cap_trigger is None else self.clearance_cap_trigger.copy())
