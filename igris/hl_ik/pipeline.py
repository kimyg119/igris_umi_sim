"""EE-priority, bounded QP refinement of the existing HL-IK interface.

The neural prior remains an elbow pose predictor. Both QP modes share the full
configured collision model, limits and final nonlinear corridor gate. No I/O.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
import proxsuite
import daqp
import mink
from .common import matrix7, pose7, Rotation
from .solver import HLIK, IKResult
from .retarget_quality import Distances


@dataclass
class PipelineResult(IKResult):
    accepted: bool
    prior_status: str
    qp_ms: float
    collision_ms: float
    primary_fallbacks: int
    qp_fallbacks: int


class PipelineIK(HLIK):
    def __init__(self, prior=None, *, mode='priority', acceleration_limit=30.,
                 collision_margin=.0005,joint_margin=.0001,
                 joint_limit_cost=0.,soft_joint_margin=.15,collision_braking=False,
                 warm_start_primary=True,primary_max_iter=50,primary_max_iter_in=50,
                 pair_buffer_overrides=None,**kwargs):
        if type(warm_start_primary) is not bool:
            raise ValueError('warm_start_primary must be a boolean')
        self.warm_start_primary=warm_start_primary
        for name,value in (('primary_max_iter',primary_max_iter),
                           ('primary_max_iter_in',primary_max_iter_in)):
            if type(value) is not int or value<=0:
                raise ValueError(f'{name} must be a positive integer')
        self.primary_max_iter=primary_max_iter
        self.primary_max_iter_in=primary_max_iter_in
        super().__init__(prior, **kwargs)
        if mode not in ('priority', 'weighted'): raise ValueError('unknown QP mode')
        if acceleration_limit is not None and (not np.isfinite(acceleration_limit) or acceleration_limit<=0):
            raise ValueError('finite positive acceleration limit required')
        self.mode=mode; self.acceleration_limit=acceleration_limit
        if not np.isfinite([collision_margin,joint_margin]).all() or min(collision_margin,joint_margin)<0:
            raise ValueError('nonnegative finite interior margins required')
        self.collision_margin=collision_margin;self.joint_margin=joint_margin
        if not np.isfinite([joint_limit_cost,soft_joint_margin]).all() or min(joint_limit_cost,soft_joint_margin)<0:
            raise ValueError('Nonnegative finite joint-limit preference required')
        self.joint_limit_cost=joint_limit_cost;self.soft_joint_margin=soft_joint_margin
        self.collision_braking=bool(collision_braking)
        if self.geometry is not None:
            self.geometry=Distances(self.model,hand_envelope=kwargs.get('hand_envelope'),
                                    pair_buffer_overrides=pair_buffer_overrides)
        self.velocity=np.zeros(14);self.acceleration=np.zeros(14);self._qps={}
        self._observed_arm=None;self._observed_velocity=np.zeros(14)
        self._qp_fallbacks=0

    def reset(self,q):
        super().reset(q);self.velocity=np.zeros(14);self.acceleration=np.zeros(14)
        self._observed_arm=np.asarray(q)[15:29].copy();self._observed_velocity=np.zeros(14)

    def observe(self,q):
        super().observe(q)
        arm=np.asarray(q)[15:29]
        velocity=np.zeros(14) if self._observed_arm is None else (arm-self._observed_arm)*30
        self.acceleration=(velocity-self._observed_velocity)*30
        self.velocity=velocity;self._observed_velocity=velocity.copy();self._observed_arm=arm.copy()

    def _qp(self,h,g,c,lower,upper,stage,retry=True):
        key=(stage,len(g),len(c))
        # Explicit transport opt-in avoids repeated primary warm-start dual
        # stalls. Keep the secondary cache, all tolerances and retries intact.
        qp=self._qps.get(key) if stage!=0 or self.warm_start_primary else None
        args=dict(H=(h+h.T)/2,g=g,C=c,l=lower,u=upper)
        if qp is None:
            qp=proxsuite.proxqp.dense.QP(len(g),0,len(c))
            qp.settings.eps_abs=1e-8;qp.settings.eps_rel=0.
            qp.settings.max_iter=self.primary_max_iter if stage==0 else 50
            qp.settings.max_iter_in=self.primary_max_iter_in if stage==0 else 50
            qp.init(**args);self._qps[key]=qp
        else:
            qp.settings.initial_guess=proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
            qp.update(**args,update_preconditioner=True)
        qp.solve();x=qp.results.x.copy()
        def feasible(value):
            return (np.isfinite(value).all() and np.all(c@value>=lower-2e-8)
                    and np.all(c@value<=upper+2e-8))
        ok=(qp.results.info.status==proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED
            and feasible(x))
        if not ok and retry:
            self._qps.pop(key)
            return self._qp(h,g,c,lower,upper,stage,retry=False)
        if not ok:
            # One bounded active-set retry handles feasible, ill-conditioned
            # EE problems on which ProxQP's dual residual plateaus. Both solvers
            # must pass the SAME original-unit constraints and nonlinear gate.
            self._qp_fallbacks+=1
            x,_,flag,_=daqp.solve(args['H'],g,c,upper,lower,
                primal_tol=1e-9,dual_tol=1e-9,iter_limit=100)
            ok=flag==1 and feasible(x)
        return x,ok

    def solve(self,targets,q_previous,*,position_tolerance=.005,
              orientation_tolerance=np.deg2rad(2),record_history=True,dt=1/30,velocity_limit=3.):
        start=perf_counter();q_previous=np.asarray(q_previous,float).copy()
        self._qp_fallbacks=0
        if not(np.isfinite(dt) and dt>0 and np.isfinite(velocity_limit) and velocity_limit>0):
            raise ValueError('positive finite timestep and speed limit required')
        if abs(dt-1/30)>1e-6:raise ValueError('pipeline history and rate bounds require 30 Hz')
        targets=np.asarray(targets,float)
        if targets.shape!=(2,7):raise ValueError('two torso-frame targets required')
        matrices=[matrix7(p) for p in targets];targets=np.array([pose7(t) for t in matrices])
        self.set_q(q_previous);base=self.configuration.q.copy();previous=base[self.qadr].copy()
        predicted=None;prior_status='disabled';qp_ms=collision_ms=0.;fallbacks=0;iterations=0
        for task,t in zip(self.tasks[:2],matrices):task.set_target(mink.SE3.from_matrix(t))

        def result(q,accepted,reason):
            self.set_q(q);errors=[]
            for task,t in zip(self.tasks[:2],matrices):
                actual=self.configuration.get_transform_frame_to_world(task.frame_name,'body').as_matrix()
                errors.append((np.linalg.norm(actual[:3,3]-t[:3,3]),Rotation.from_matrix(t[:3,:3].T@actual[:3,:3]).magnitude()))
            pe,ae=np.max(errors,axis=0)
            success=bool(accepted and pe<=position_tolerance and ae<=orientation_tolerance)
            if accepted:
                velocity=(q[15:29]-q_previous[15:29])/dt
                self.acceleration=(velocity-self.velocity)/dt;self.velocity=velocity
                # Feasible tracking ticks and strict convergence are separate.
                if record_history:self.observe(q)
            return PipelineResult(q.copy(),success,float(pe),float(ae),iterations,
                reason if not accepted else 'ok' if success else 'tracking_residual',
                (perf_counter()-start)*1000,0.,predicted,bool(accepted),prior_status,qp_ms,collision_ms,fallbacks,self._qp_fallbacks)

        if np.any(previous<self.limits[:,0]-1e-8) or np.any(previous>self.limits[:,1]+1e-8) or np.any(self.A@previous>self.b+1e-8):
            return result(q_previous,False,'invalid_seed_limits')
        if self.geometry is not None:
            stamp=perf_counter();hit=self.geometry.collision(self.configuration.data);collision_ms+=(perf_counter()-stamp)*1000
            if hit:return result(q_previous,False,'invalid_seed_collision')
        if not self.history:self.reset(q_previous)
        if self.prior is not None:
            goal=targets.copy();goal[:,:3]-=self.shoulder
            try:
                predicted=np.asarray(self.prior.predict(np.array(self.history),goal),float)
                if predicted.shape!=(2,7) or not np.isfinite(predicted).all():raise ValueError('invalid prior pose')
                transforms=[matrix7(p) for p in predicted]
                for s,(task,t) in enumerate(zip(self.tasks[2:],transforms)):
                    t[:3,3]+=self.shoulder[s];task.set_target(mink.SE3.from_matrix(t))
                prior_status='used'
            except (ValueError,FloatingPointError):predicted=None;prior_status='invalid_prediction'

        # Bounds apply to the WHOLE command tick, not each optimizer iteration.
        lower_v=np.full(14,-velocity_limit);upper_v=-lower_v
        if self.acceleration_limit is not None:
            lower_v=np.maximum(lower_v,self.velocity-self.acceleration_limit*dt)
            upper_v=np.minimum(upper_v,self.velocity+self.acceleration_limit*dt)
        lower=np.maximum(self.limits[:,0]+self.joint_margin-previous,lower_v*dt)
        upper=np.minimum(self.limits[:,1]-self.joint_margin-previous,upper_v*dt)
        coupled_upper=self.b-self.joint_margin-self.A@previous
        if self.acceleration_limit is not None:
            # Reserve a two-tick reaction distance before braking at the joint
            # and coupled limits. A position box alone can leave the next tick
            # infeasible while the previous command still has outward velocity.
            def stopping_speed(distance,acceleration):
                reaction=2*acceleration*dt
                return np.sqrt(reaction**2+2*acceleration*np.maximum(distance,0))-reaction
            lower=np.maximum(lower,-dt*stopping_speed(previous-self.limits[:,0],self.acceleration_limit))
            upper=np.minimum(upper,dt*stopping_speed(self.limits[:,1]-previous,self.acceleration_limit))
            coupled_acceleration=self.acceleration_limit*np.abs(self.A).sum(1)
            coupled_upper=np.minimum(coupled_upper,dt*stopping_speed(coupled_upper,coupled_acceleration))
        if np.any(lower>upper+1e-10):return result(q_previous,False,'infeasible_rate_bounds')
        fixed_c=np.vstack((np.eye(14),self.A));fixed_l=np.r_[lower,np.full(len(self.b),-np.inf)]
        fixed_u=np.r_[upper,coupled_upper]
        def residual(displacement):
            full=base.copy();full[self.qadr]=previous+displacement;self.configuration.update(full)
            tasks=self.tasks[:4 if predicted is not None else 2]
            errors=np.concatenate([-task.compute_error(self.configuration) for task in tasks])
            return errors*self.weights[:len(errors)]
        displacement=np.zeros(14);last_cost=np.inf;feasible=None
        for iteration in range(min(self.max_iterations,6)):
            iterations=iteration+1;arm=previous+displacement
            residual(displacement)
            errors=[];jacs=[]
            for task in self.tasks[:4 if predicted is not None else 2]:
                errors.append(-task.compute_error(self.configuration))
                jacs.append(-task.compute_jacobian(self.configuration)[:,self.dadr])
            e=np.concatenate(errors);j=np.vstack(jacs)
            weights=self.weights[:len(e)];e*=weights;j*=weights[:,None]
            # Linearize in total displacement from the caller's current posture.
            offset=e-j@displacement;primary_j=j[:12];primary_e=offset[:12]
            c=fixed_c;l=fixed_l;u=fixed_u
            if self.geometry is not None:
                stamp=perf_counter();pair_thresholds=None
                if self.collision_braking and self.acceleration_limit is not None:
                    pair_thresholds=self.geometry.braking_thresholds(self.configuration.data,self.dadr,
                        np.maximum(abs(lower),abs(upper))/dt,dt,self.acceleration_limit,self.collision_margin)
                contacts=self.geometry.contacts(self.configuration.data,
                    threshold=.15 if self.collision_braking else .02,per_pair=4,
                    active_dofs=self.dadr if self.collision_braking else None,pair_thresholds=pair_thresholds)
                collision_ms+=(perf_counter()-stamp)*1000
                if contacts:
                    gradients=np.array([r[1][self.dadr-6] for r in contacts]);gaps=np.array([r[0] for r in contacts])
                    # Add an interior margin for resampling/numerical roundoff.
                    # Caller-owned body pairs with zero arm derivative cannot
                    # be moved by this solver; the final full audit still sees them.
                    movable=np.linalg.norm(gradients,axis=1)>1e-10
                    gradients=gradients[movable];gaps=gaps[movable]
                    clearance=self.collision_margin-gaps+gradients@displacement
                    if self.collision_braking and self.acceleration_limit is not None:
                        # A legal next position can still be too close to stop
                        # under the acceleration bound. Begin slowing earlier,
                        # using a conservative normal-acceleration estimate and
                        # the same two-tick reaction allowance as joint limits.
                        normal_acceleration=.5*self.acceleration_limit*np.abs(gradients).sum(1)
                        available=gaps-gradients@displacement-self.collision_margin
                        clearance=np.maximum(clearance,-dt*stopping_speed(available,normal_acceleration))
                    c=np.vstack((c,gradients));l=np.r_[l,clearance];u=np.r_[u,np.full(len(gaps),np.inf)]
            smooth=self.velocity*dt
            if self.mode=='weighted':
                h=j.T@j+.35*np.eye(14);g=j.T@offset
            else:
                h=primary_j.T@primary_j+1e-6*np.eye(14);g=primary_j.T@primary_e-1e-6*smooth
            stamp=perf_counter();primary,ok=self._qp(h,g,c,l,u,0);qp_ms+=(perf_counter()-stamp)*1000
            if not ok:
                if feasible is not None:displacement=feasible;break
                return result(q_previous,False,'qp_failed')
            candidate=primary.copy()
            if self.mode=='priority':
                _,singular,vh=np.linalg.svd(primary_j,full_matrices=True)
                null=vh[np.count_nonzero(singular>1e-7):].T
                secondary_h=j[12:].T@j[12:]+2.35*np.eye(14)
                secondary_g=j[12:].T@offset[12:]-2.*smooth
                if self.joint_limit_cost:
                    # A posture preference in the EE null space, separate from
                    # the unchanged hard limits. Anticipate coupled wrist limits
                    # before they eliminate the local EE solution branch.
                    faces=np.vstack((np.eye(14),-np.eye(14),self.A))
                    bounds=np.r_[self.limits[:,1],-self.limits[:,0],self.b]
                    active=faces@arm>bounds-self.soft_joint_margin
                    f=faces[active];offset_limit=f@previous-bounds[active]+self.soft_joint_margin
                    secondary_h+=self.joint_limit_cost*(f.T@f)
                    secondary_g+=self.joint_limit_cost*(f.T@offset_limit)
                if null.shape[1]:
                    cn=c@null;shift=c@primary
                    # Roundoff-legal primary is a feasible zero correction.
                    ln=np.minimum(l-shift,0.);un=np.maximum(u-shift,0.)
                    stamp=perf_counter();z,ok=self._qp(null.T@secondary_h@null,null.T@(secondary_h@primary+secondary_g),cn,ln,un,1)
                    qp_ms+=(perf_counter()-stamp)*1000
                    if ok:candidate=primary+null@z
                    else:fallbacks+=1
                # Exact nonlinear EE cost must stay within 0.02 mm-equivalent
                # of the primary candidate; reduce only the secondary movement.
                r=residual(primary)[:12];primary_norm=np.linalg.norm(r)
                for k in range(8):
                    r=residual(candidate)[:12]
                    if np.linalg.norm(r)<=primary_norm+np.sqrt(50)*.00002:break
                    candidate=primary+(candidate-primary)*.5
                else:candidate=primary;fallbacks+=1
            if np.any(c@candidate<l-3e-8) or np.any(c@candidate>u+3e-8):
                return result(q_previous,False,'constraint_residual')
            residual(candidate)
            if self.geometry is not None:
                stamp=perf_counter();hit=self.geometry.corridor(self.configuration.data,base,self.configuration.q.copy(),max_step=.01)
                collision_ms+=(perf_counter()-stamp)*1000
                if hit:
                    # Re-linearize at the rejected candidate before giving up.
                    # It is never committed to history or returned as accepted.
                    displacement=candidate
                    continue
            feasible=candidate.copy()
            r=residual(candidate)
            cost=float(r[:12]@r[:12]) if self.mode=='priority' else float(r@r)
            change=np.max(abs(candidate-displacement));displacement=candidate
            if change<1e-6 or abs(last_cost-cost)<1e-10:break
            last_cost=cost
        if feasible is None:return result(q_previous,False,'collision_corridor')
        displacement=feasible
        q=q_previous.copy();q[15:29]=previous+displacement
        return result(q,True,'ok')
