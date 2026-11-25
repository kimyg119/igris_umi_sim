"""Learned elbow prior plus bounded SE(3) Levenberg-Marquardt IK.

The numerical objective follows HL-IK Eq.2-6. Collision and IGRIS limits are
additional gates. No transport, robot connection or actuator command exists here.
"""
import time
from collections import deque
from dataclasses import dataclass
import numpy as np
import mujoco
import mink
import torch
from .common import *
from .network import BimanualFiSTA,NetworkConfig,canonicalize,decode
from .geometry import SphereGeometry,coupled_halfspaces

class ElbowPrior:
    def __init__(self,checkpoint,device='cpu'):
        self.device=device;torch.set_num_threads(1)
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        self.model=BimanualFiSTA(NetworkConfig(**saved['config']['network'])).to(device)
        self.model.load_state_dict(saved['network']);self.model.eval()
        self.mean=saved['mean'].to(device);self.std=saved['std'].to(device)
        self.checkpoint=str(checkpoint);self.sha256=sha256(checkpoint)

    @torch.no_grad()
    def predict(self,history,goal):
        history=np.asarray(history);goal=np.asarray(goal)
        if history.shape!=(5,2,14) or goal.shape!=(2,7) or not np.isfinite(history).all() or not np.isfinite(goal).all():
            raise ValueError('expected finite history (5,2,14) and goal (2,7)')
        h=torch.as_tensor(history[None],device=self.device,dtype=torch.float32)
        g=torch.as_tensor(goal[None],device=self.device,dtype=torch.float32)
        h,g,_=canonicalize(h,g)
        pred=self.model((h-self.mean)/self.std,(g-self.mean[:,:7])/self.std[:,:7])
        return decode(pred,self.mean,self.std)[0].cpu().numpy()

@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    position_error_m: float
    orientation_error_rad: float
    iterations: int
    reason: str
    elapsed_ms: float
    objective: float
    predicted_elbow: np.ndarray | None

class HLIK:
    def __init__(self,prior=None,*,prior_scale=1.,collision=True,max_iterations=30,hand_envelope=None,model_path=None):
        self.model_path=Path(model_path) if model_path is not None else OUTPUT/'model/igris_c_rcv_hand_torso.xml'
        self.model=mujoco.MjModel.from_xml_path(str(self.model_path))
        self.configuration=mink.Configuration(self.model)
        self.prior=prior;self.prior_scale=prior_scale;self.max_iterations=max_iterations
        self.geometry=SphereGeometry(self.model,hand_envelope=hand_envelope) if collision else None
        self.qadr=np.array([self.model.joint(n).qposadr[0] for n in ARM_NAMES])
        self.dadr=np.array([self.model.joint(n).dofadr[0] for n in ARM_NAMES])
        self.limits=self.model.jnt_range[[self.model.joint(n).id for n in ARM_NAMES]]
        self.A,self.b=coupled_halfspaces(list(ARM_NAMES))
        self.tasks=[mink.FrameTask(frame_name=n,frame_type='body',position_cost=1.,orientation_cost=1.) for n in (*EE_NAMES,*ELBOW_NAMES)]
        self.shoulder=np.array([[0,.171,.266],[0,-.171,.266]])
        self.history=deque(maxlen=5)
        self.weights=np.r_[np.tile(np.sqrt([50,50,50,40,40,40]),2),np.tile(np.sqrt([20,20,20,5,5,5])*np.sqrt(prior_scale),2),np.full(14,np.sqrt(.35))]

    def set_q(self,q):
        q=np.asarray(q,dtype=float)
        if q.shape!=(31,) or not np.isfinite(q).all():raise ValueError('finite SDK q[31] required')
        if np.max(np.abs(q[29:]))>1e-8:raise ValueError('this torso model requires the neck locked at zero')
        full=self.model.qpos0.copy()
        for i,n in enumerate(SDK_NAMES[:29]):full[self.model.joint(n).qposadr[0]]=q[i]
        self.configuration.update(full)

    def q_sdk(self):
        q=np.zeros(31)
        for i,n in enumerate(SDK_NAMES[:29]):q[i]=self.configuration.q[self.model.joint(n).qposadr[0]]
        return q

    def states(self):
        result=[]
        for side in range(2):
            poses=[]
            for name in (EE_NAMES[side],ELBOW_NAMES[side]):
                t=self.configuration.get_transform_frame_to_world(name,'body').as_matrix()
                t[:3,3]-=self.shoulder[side];poses.extend(pose7(t))
            result.append(poses)
        return np.asarray(result)

    def reset(self,q):
        self.set_q(q);state=self.states();self.history.clear()
        for _ in range(5):self.history.append(state.copy())

    def observe(self,q):
        self.set_q(q);self.history.append(self.states())

    def residual(self,arm,previous,base,*,jacobian=True):
        q=base.copy();q[self.qadr]=arm;self.configuration.update(q)
        count=4 if self.prior is not None else 2
        errors=[];jacs=[]
        for task in self.tasks[:count]:
            # Mink returns Log(current^-1 target). Negate to get paper Eq.2.
            errors.append(-task.compute_error(self.configuration))
            if jacobian:jacs.append(-task.compute_jacobian(self.configuration)[:,self.dadr])
        errors.append(arm-previous)
        weights=np.r_[self.weights[:count*6],self.weights[-14:]]
        error=np.concatenate(errors)*weights
        if not jacobian:return error
        jacs.append(np.eye(14));return error,np.vstack(jacs)*weights[:,None]

    def solve(self,targets,q_previous,*,position_tolerance=.005,orientation_tolerance=np.deg2rad(2),record_history=True,dt=1/30,velocity_limit=3.):
        start=time.perf_counter();targets=np.asarray(targets,dtype=float)
        if not np.isfinite(dt) or dt<=0 or not np.isfinite(velocity_limit) or velocity_limit<=0:raise ValueError('positive finite dt and velocity limit required')
        if self.prior is not None and abs(dt-1/30)>1e-6:raise ValueError('trained history requires 30 Hz; interpolate commands separately')
        if targets.shape!=(2,7):raise ValueError('targets must be two torso-frame xyz+wxyz poses')
        matrices=[matrix7(p) for p in targets]
        targets=np.array([pose7(t) for t in matrices]) # The prior and LM use the same normalized rotations.
        if not self.history:self.reset(q_previous)
        self.set_q(q_previous);base=self.configuration.q.copy();arm=base[self.qadr].copy();previous=arm.copy()
        if np.any(arm<self.limits[:,0]-1e-6) or np.any(arm>self.limits[:,1]+1e-6) or np.max(self.A@arm-self.b)>1e-6:
            raise ValueError('previous arm configuration violates limits')
        predicted=None
        for task,t in zip(self.tasks[:2],matrices):task.set_target(mink.SE3.from_matrix(t))
        if self.prior is not None:
            goal=targets.copy();goal[:,:3]-=self.shoulder
            predicted=self.prior.predict(np.array(self.history),goal)
            for s,task in enumerate(self.tasks[2:]):
                t=matrix7(predicted[s]);t[:3,3]+=self.shoulder[s];task.set_target(mink.SE3.from_matrix(t))
        damping=.001;iterations=0
        for iteration in range(self.max_iterations):
            iterations=iteration+1;res,jac=self.residual(arm,previous,base);cost=float(res@res/2)
            step=np.linalg.solve(jac.T@jac+damping*np.eye(14),-jac.T@res)
            if np.max(np.abs(step))<1e-6:break
            # Do not clip a candidate across a coupled-limit or collision boundary.
            alpha=1.;accepted=False
            for _ in range(12):
                cand=arm+alpha*step
                legal=(np.all(cand>=self.limits[:,0]) and np.all(cand<=self.limits[:,1]) and np.max(self.A@cand-self.b)<=1e-8
                       and np.max(np.abs(cand-previous))<=velocity_limit*dt)
                if legal:
                    r=self.residual(cand,previous,base,jacobian=False);new_cost=float(r@r/2)
                    if new_cost<cost:
                        if self.geometry is None or self.geometry.collision(self.configuration.data) is None:
                            arm=cand;accepted=True;damping=max(1e-7,damping/3);break
                alpha*=.5
            if not accepted:
                damping*=10
                if damping>1e6:break
            elif cost-new_cost<1e-10:break
        res=self.residual(arm,previous,base,jacobian=False)
        errors=[]
        for task,t in zip(self.tasks[:2],matrices):
            actual=self.configuration.get_transform_frame_to_world(task.frame_name,'body').as_matrix()
            errors.append((np.linalg.norm(actual[:3,3]-t[:3,3]),Rotation.from_matrix(t[:3,:3].T@actual[:3,:3]).magnitude()))
        pe=max(e[0] for e in errors);ae=max(e[1] for e in errors)
        hit=None if self.geometry is None else self.geometry.corridor(self.configuration.data,base,self.configuration.q.copy())
        reason='collision' if hit else 'position_residual' if pe>position_tolerance else 'orientation_residual' if ae>orientation_tolerance else 'ok'
        success=reason=='ok';result=self.q_sdk()
        # Failed candidates are returned for diagnostics, never inserted in accepted history.
        if record_history and success:self.history.append(self.states())
        return IKResult(result,success,float(pe),float(ae),iterations,reason,(time.perf_counter()-start)*1000,float(res@res/2),predicted)
