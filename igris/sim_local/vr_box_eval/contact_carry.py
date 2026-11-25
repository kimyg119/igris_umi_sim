"""Contact-driven box task; inputs are observed vendor physics and IK acknowledgements."""
import numpy as np

PALM_SEAT_LIMIT = 4.0  # Virtual angular preload up to 10.6 degrees; measured contact stops it.
PALM_GEOMETRY_ARC = 2.0  # Initial placement checks the unloaded 5.3-degree surface arc.
BOX_TILT_LIMIT_RAD = np.deg2rad(10.)
LANDING_TILT_LIMIT_RAD = np.deg2rad(5.)


class ContactCarry:
    def __init__(self, mass, friction=.8, *, palm_first=False, preload_limit_m=.035, palm_force_N=None, tip_force_N=None):
        if not np.isfinite([mass,friction]).all() or mass<=0 or friction<=0:
            raise ValueError('mass and friction must be finite and positive')
        if not np.isfinite(preload_limit_m) or not 0<preload_limit_m<=.05:
            raise ValueError('preload limit must be in (0, 0.05] m per hand')
        self.required_force = 1.5 * mass * 9.81 / (2 * friction)
        self.preload_force = max(1., self.required_force)
        if palm_force_N is not None:
            if not palm_first or not np.isfinite(palm_force_N) or palm_force_N<=0:
                raise ValueError('palm force requires palm-first contact and a finite positive value')
            self.preload_force = max(self.preload_force, float(palm_force_N))
        self.contact_threshold_N = min(.5, .1*mass*9.81)
        self.surface_force_N = max(.1,self.contact_threshold_N)
        self.secondary_force_N = min(.2,self.surface_force_N) if palm_first else self.surface_force_N
        if tip_force_N is not None:
            if not palm_first or not np.isfinite(tip_force_N) or tip_force_N<=0:
                raise ValueError('tip force requires palm-first contact and a finite positive value')
            self.secondary_force_N = max(self.secondary_force_N, float(tip_force_N))
        self.palm_first = palm_first
        self.seat_limit = PALM_SEAT_LIMIT
        self.phase = 'approach'
        self.value = np.array([0., -.02, 0., -np.pi/4, 0., 0.])  # clock, preload, lift, yaw, left/right seating
        self.proposal = self.value.copy()
        self.force = np.zeros(2)
        self.patch = np.zeros(4)  # left/right palm and selected secondary surface (plate or distal)
        self.stable = self.phase_age = self.probe_settled = 0.
        self.landing_stable = 0.
        self.rotation_unstable = self.landing_contact_age = 0.
        self.step_scale = 1.
        self.ceiling = float(preload_limit_m)
        self.retry_delay = 0.
        self.reason = None
        self.rejections = 0
        self.relieve_preload = False
        self.events = []

    def acknowledge(self, proposal, accepted, *, progressing=False):
        if accepted:
            self.value = np.array(proposal, copy=True)
            self.step_scale = min(1., self.step_scale*1.1)
            self.relieve_preload = False
        else:
            self.rejections += 1
            if progressing:
                return  # MoveL is still approaching the endpoint; this is not an IK rejection.
            self.relieve_preload = self.palm_first and self.phase in ('probe_lift','lift','carry_left')
            self.step_scale = max(.01, self.step_scale*.5)
            self.retry_delay = .15

    def transition(self, phase):
        self.events.append(dict(phase=phase,preload_m=float(self.value[1]),lift_m=float(self.value[2])))
        self.phase = phase
        self.phase_age = self.stable = 0.

    def restore_surface_contact(self, q, step):
        seeking_secondary = self.palm_first and np.min(self.patch[[0,2]])>=self.surface_force_N and np.min(self.patch[[1,3]])<self.secondary_force_N
        seat_exhausted = self.palm_first and any(
            self.patch[2*i+1]<self.secondary_force_N and self.patch[2*i]>=self.surface_force_N and q[4+i]>=self.seat_limit
            for i in range(2))
        # Once tilt is exhausted, bounded normal preload must still recover a weak tip.
        support_force=np.min(self.patch[[0,2]]) if self.palm_first else np.min(self.force)
        if (support_force<self.preload_force and not seeking_secondary) or seat_exhausted:
            q[1]=min(self.ceiling,q[1]+.002*step)
        for side in range(2):
            palm,pad=self.patch[2*side:2*side+2]
            if palm<self.surface_force_N and self.force[side]>=self.preload_force:
                q[4+side]=max(0.,q[4+side]-.5*step)
            elif pad<self.secondary_force_N and palm>=self.surface_force_N:
                q[4+side]=min(self.seat_limit,q[4+side]+.5*step)

    def update(self, dt, *, fresh, forces, patches, box_lift, supported, box_tilt_rad=0.):
        dt = min(max(float(dt),0.),.05)
        self.phase_age += dt
        if not fresh:
            self.stable = 0.
            self.landing_stable = 0.
            self.rotation_unstable = self.landing_contact_age = 0.
            self.proposal = self.value.copy()
            return self.proposal
        blend = 1-np.exp(-dt/.15)
        self.force += blend*(np.asarray(forces)-self.force)
        self.patch += blend*(np.asarray(patches)-self.patch)
        palm_force = float(np.min(self.patch[[0,2]]))
        seated = bool(np.min(self.patch)>=self.surface_force_N and
                      np.min(self.patch.reshape(2,2).sum(1))>=self.preload_force)
        if self.palm_first:
            seated=bool(palm_force>=self.preload_force and np.min(self.patch[[1,3]])>=self.secondary_force_N)
        secure = palm_force>=(self.surface_force_N if self.palm_first else self.preload_force) if self.phase in ('approach','squeeze') else seated
        self.stable = self.stable+dt if secure else 0.
        q = self.value.copy();step = 0. if self.retry_delay>0 else 2*dt*self.step_scale
        self.retry_delay = max(0.,self.retry_delay-dt)
        # Normal force alone cannot detect a cube rotating around the palm bumps.
        rotating = self.phase in ('probe_lift','lift','carry_left') and box_tilt_rad>BOX_TILT_LIMIT_RAD
        self.rotation_unstable = self.rotation_unstable+dt if rotating else 0.
        if rotating:
            if self.rotation_unstable>=.15:
                self.reason='box_rotation_unstable';self.transition('failed')
            self.proposal=q
            return q.copy()
        if (self.relieve_preload and self.phase in ('probe_lift','lift','carry_left') and
                seated and palm_force>1.1*self.preload_force and q[1]>0):
            # Recover IK room from excess virtual squeeze before requesting more lift/yaw.
            # Contact feedback and the normal IK gates must accept this small retreat too.
            q[1]=max(0.,q[1]-.0005)
            self.proposal=q
            return q.copy()
        if self.phase == 'approach':
            q[0] = min(19., q[0]+step)
            if self.value[0]>=19.:self.transition('squeeze')
        elif self.phase == 'squeeze':
            # Stop preload at sufficient palm force; fingertips are supplementary.
            if self.stable>=.3:
                self.transition('seat_palm')
            elif not secure:
                q[1] = min(self.ceiling, q[1]+.004*step)
            if self.phase_age>35.:
                self.reason='insufficient_contact_within_ik_range';self.transition('failed')
        elif self.phase == 'seat_palm':
            # Restore palm pressure before tilting; preload still requires IK acknowledgement.
            if self.stable>=.3:
                self.transition('probe_lift')
            else:
                self.restore_surface_contact(q,step)
            if self.phase_age>20.:
                self.reason='palm_tip_not_seated' if self.palm_first else 'broad_palm_not_seated';self.transition('failed')
        elif self.phase == 'probe_lift':
            probe_limit = .08 if self.palm_first else .04
            if seated:
                q[2] = min(probe_limit,q[2]+.01*step)
            else:
                self.restore_surface_contact(q,step)
            self.probe_settled = self.probe_settled+dt if self.value[2]>=probe_limit-1e-6 else 0.
            if box_lift>.012 and not supported and self.stable>=.3:
                self.transition('lift')
            elif self.probe_settled>2. or self.phase_age>16.:
                self.reason='probe_lift_not_secure';self.transition('failed')
        elif self.phase in ('lift','carry_left'):
            if box_lift<.005:
                self.reason='contact_lost';self.transition('failed')
            elif not seated:
                # Hold lift/yaw; missing broad contact needs seating, not more palm force.
                self.restore_surface_contact(q,step)
            elif self.phase=='lift':
                q[2]=min(.12 if self.palm_first else .08,q[2]+.02*step)
                clear = not self.palm_first or (box_lift>.035 and not supported and self.stable>=.3)
                if self.value[2]>=.08 and clear:self.transition('carry_left')
            elif self.palm_first and supported:
                # Compliant deflection can leave a tilted cube touching its support.
                q[2]=min(.12,q[2]+.01*step)
            else:
                q[3]=min(np.pi/4,q[3]+np.pi/24*step)
                if self.value[3]>=np.pi/4:self.transition('lower')
        elif self.phase == 'lower':
            # Stop pushing down at physical support contact. A zero virtual
            # height can otherwise press a tilted cube into a table corner.
            if not supported:q[2]=max(0.,q[2]-.02*step)
            self.landing_contact_age=self.landing_contact_age+dt if supported else 0.
            self.landing_stable=self.landing_stable+dt if supported and abs(box_lift)<.01 and box_tilt_rad<=LANDING_TILT_LIMIT_RAD else 0.
            if self.landing_stable>=.5:
                self.transition('open')
            elif self.landing_contact_age>3.:
                self.reason='unstable_landing';self.transition('failed')
        elif self.phase == 'open':
            # Landed release moves away from contact; a failed lower goal must not
            # shrink every outward proposal to micrometres. IK still acknowledges it.
            q[1]=max(-.02,q[1]-.012*(2*dt if self.palm_first else step))
            if self.value[1]<=-.02 and np.max(self.force)<self.contact_threshold_N:self.transition('done')
        if self.phase in ('failed','done'):
            q=self.value.copy()
        self.proposal=q
        return q.copy()
