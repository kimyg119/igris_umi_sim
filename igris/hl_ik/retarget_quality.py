"""Offline sequence refinement of GMR output; independent of the active pipeline.

Keeps original timestamps, hand frames, collision spheres/ignore list, and
comparison thresholds. Corrects torso anchoring and the elbow hinge target.
Optimizes all
29 joint trajectories together. Collision constraints have explicit slack during
recovery; only a separate final audit decides whether a frame can train a model.
Nothing in this module sends robot commands or changes the original dataset.
"""
import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Slerp
import osqp
import mujoco
from qpsolvers import solve_qp
from .common import OUTPUT, HERE, SDK_NAMES, EE_NAMES, ELBOW_NAMES, sha256, save_json, Rotation
from .geometry import SphereGeometry, coupled_halfspaces
from . import retarget as gmr


@dataclass
class Settings:
    iterations: int = 25
    velocity_limit: float = 3.
    acceleration_limit: float = 60.
    velocity_cost: float = .15
    acceleration_cost: float = .015
    collision_activation: float = .025
    collision_clearance: float = .006
    collision_slack_cost: float = 1e7
    trust_region: float = .05
    corridor_step: float = .01
    optimization_corridor_step: float = .02
    enforce_timing: bool = True
    window_frames: int = 90
    window_overlap: int = 8
    collision_constraints_per_pair: int = 1
    max_corridor_samples: int = 8
    corridor_repair_iterations: int = 4


class Distances(SphereGeometry):
    """Closest buffered-sphere gaps and analytic joint-space derivatives."""
    def enable_native_contacts(self):
        from .native_contacts import NativeContacts
        self._native_contacts=NativeContacts()

    def braking_thresholds(self,data,active_dofs,speed_bound,dt,acceleration,margin):
        """Pair horizons that retain every potentially binding braking row.

        If |gradient_j| <= B_j and each total displacement is bounded by
        dt*s_j, a gap above margin + sum B_j*(s_j**2/a + 3*dt*s_j)
        cannot bind the current braking QP. Cauchy-Schwarz bounds the stopping
        term; the third dt term covers displacement of the linearization point.
        The final nonlinear collision/corridor gates remain exhaustive.
        """
        speed=np.asarray(speed_bound,float);dofs=np.asarray(active_dofs,int)
        if speed.shape!=dofs.shape or not np.isfinite(speed).all() or np.any(speed<0) or dt<=0 or acceleration<=0:
            raise ValueError('Invalid braking displacement bound')
        rot=data.xmat[self.ids].reshape(-1,3,3);pos=data.xpos[self.ids]
        centers=np.einsum('bij,bj->bi',rot,self.center)+pos
        candidates=np.flatnonzero(np.linalg.norm(centers[self.ia]-centers[self.ib],axis=1)
            <self.bound[self.ia]+self.bound[self.ib]+.15)
        if not hasattr(self,'_gradient_reach'):
            self._gradient_reach=np.array([np.linalg.norm(v,axis=1).max() for v in self.local])
        weights=speed**2/acceleration+3*dt*speed
        horizons=np.full(len(self.pairs),.02);jacs={}
        for k in candidates:
            a,b=self.pairs[k]
            for j in (a,b):
                if j not in jacs:
                    jp=np.empty((3,self.model.nv));jr=jp.copy()
                    mujoco.mj_jacBody(self.model,data,jp,jr,int(self.ids[j]))
                    jacs[j]=(jp[:,dofs],self._gradient_reach[j]*np.linalg.norm(jr[:,dofs],axis=0))
            bound=np.linalg.norm(jacs[a][0]-jacs[b][0],axis=0)+jacs[a][1]+jacs[b][1]
            horizons[k]=np.clip(margin+bound@weights,.02,.15)
        return horizons

    def contacts(self, data, threshold=.025, derivatives=True, per_pair=4,stop_at_first=False,active_dofs=None,pair_thresholds=None):
        rot=data.xmat[self.ids].reshape(-1,3,3);pos=data.xpos[self.ids]
        centers=np.einsum('bij,bj->bi',rot,self.center)+pos
        ranges=threshold
        if pair_thresholds is not None:
            ranges=np.asarray(pair_thresholds,float)
            if ranges.shape!=(len(self.pairs),) or not np.isfinite(ranges).all() or np.any(ranges<0):
                raise ValueError('Expected one finite nonnegative horizon per body pair')
            ranges=np.minimum(ranges,threshold)
        candidates=np.flatnonzero(np.linalg.norm(centers[self.ia]-centers[self.ib],axis=1)
                                 <self.bound[self.ia]+self.bound[self.ib]+ranges)
        if active_dofs is not None:
            # Distances between bodies moved rigidly by the same controlled
            # joints have exactly zero derivative for those joints. Skip their
            # expensive sphere/Jacobian work in a restricted-arm QP. Full-body
            # collision/corridor audits omit this option and remain exhaustive.
            key=tuple(map(int,active_dofs))
            if not hasattr(self,'_active_pair_masks'):self._active_pair_masks={}
            if key not in self._active_pair_masks:
                controlled=set(map(int,self.model.dof_bodyid[list(key)]));ancestors=[]
                for body in self.ids:
                    chain=set();body=int(body)
                    while body:
                        chain.add(body);body=int(self.model.body_parentid[body])
                    ancestors.append(chain)
                self._active_pair_masks[key]=np.array([bool(controlled&(ancestors[a]^ancestors[b])) for a,b in self.pairs])
            candidates=candidates[self._active_pair_masks[key][candidates]]
        kernel=getattr(self,'_native_contacts',None)
        if kernel is not None and per_pair>0:
            points=[v@r.T+p for v,r,p in zip(self.local,rot,pos)]
            limits=np.full(len(candidates),threshold) if pair_thresholds is None else ranges[candidates]
            # Select in the original buffered-gap coordinates. Negative limits
            # are supported by the kernel, including its stop-at-first path.
            limits=limits-self.pair_gap_offsets[candidates]
            native=kernel.select(points,self.radius,np.asarray(self.pairs)[candidates],limits,per_pair,stop_at_first)
            if native is not None:
                result=[];jacs={};cursor=0
                while cursor<len(native):
                    k=native[cursor][0];end=cursor+1
                    while end<len(native) and native[end][0]==k:end+=1
                    selected=native[cursor:end];a,b=self.pairs[candidates[k]]
                    ii=np.array([hit[1] for hit in selected]);jj=np.array([hit[2] for hit in selected])
                    gradients=None
                    if derivatives:
                        x,y=points[a][ii],points[b][jj];delta=x-y
                        normals=delta/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-10)
                        for j in (a,b):
                            if j not in jacs:
                                jp=np.empty((3,self.model.nv));jr=jp.copy()
                                mujoco.mj_jacBody(self.model,data,jp,jr,int(self.ids[j]));jacs[j]=(jp[:,6:],jr[:,6:])
                        ap,ar=jacs[a];bp,br=jacs[b]
                        gradients=normals@(ap-bp)+np.cross(x-pos[a],normals)@ar-np.cross(y-pos[b],normals)@br
                    for index,(_,i,j,gap) in enumerate(selected):
                        if self.pair_gap_offsets[candidates[k]]:gap+=self.pair_gap_offsets[candidates[k]]
                        result.append((gap,None if gradients is None else gradients[index],(self.names[a],self.names[b]),(i,j)))
                    cursor=end
                return result
        # Exact gap ties use the historical NumPy quicksort ordering below.
        points={};boxes={};jacs={};result=[]
        for k in candidates:
            a,b=self.pairs[k]
            pair_threshold=threshold if pair_thresholds is None else ranges[k]
            pair_threshold-=self.pair_gap_offsets[k]
            for j in (a,b):
                if j not in points:
                    points[j]=self.local[j]@rot[j].T+pos[j]
                    boxes[j]=((points[j]-self.radius[j][:,None]).min(0),(points[j]+self.radius[j][:,None]).max(0))
            pa,pb=points[a],points[b];ra,rb=self.radius[a],self.radius[b]
            la,ha=boxes[a];lb,hb=boxes[b]
            if np.any(la>hb+pair_threshold) or np.any(lb>ha+pair_threshold):continue
            ka=np.flatnonzero(((pa+ra[:,None]+pair_threshold>=lb)&(pa-ra[:,None]-pair_threshold<=hb)).all(1))
            kb=np.flatnonzero(((pb+rb[:,None]+pair_threshold>=la)&(pb-rb[:,None]-pair_threshold<=ha)).all(1))
            gaps=cdist(pa[ka],pb[kb])-ra[ka,None]-rb[None,kb]
            selected=np.flatnonzero(gaps.ravel()<pair_threshold)
            if not len(selected):continue
            values=gaps.ravel()[selected]
            if 0<per_pair<len(selected):
                nearest=np.argpartition(values,per_pair-1)[:per_pair]
                nearest=nearest[np.argsort(values[nearest])]
                # Preserve the previous full-sort ordering for exact ties,
                # including a tie that straddles the selection boundary.
                if (np.any(np.diff(values[nearest])==0)
                        or np.count_nonzero(values==values[nearest[-1]])>1):
                    nearest=np.argsort(values)[:per_pair]
            else:nearest=np.argsort(values)[:per_pair]
            selected=selected[nearest]
            ii,jj=np.unravel_index(selected,gaps.shape)
            x,y=pa[ka[ii]],pb[kb[jj]]
            delta=x-y
            normals=delta/np.maximum(np.linalg.norm(delta,axis=1,keepdims=True),1e-10)
            gradients=None
            if derivatives:
                for j in (a,b):
                    if j not in jacs:
                        jp=np.empty((3,self.model.nv));jr=jp.copy()
                        mujoco.mj_jacBody(self.model,data,jp,jr,int(self.ids[j]));jacs[j]=(jp[:,6:],jr[:,6:])
                # Compute the same selected sphere-pair derivatives as a batch.
                # The broad collision-braking horizon otherwise spends most of
                # its time dispatching tiny NumPy cross products in Python.
                ap,ar=jacs[a];bp,br=jacs[b]
                gradients=normals@(ap-bp)+np.cross(x-pos[a],normals)@ar-np.cross(y-pos[b],normals)@br
            for index,(i,j) in enumerate(zip(ii,jj)):
                gap=float(gaps[i,j]);grad=None if gradients is None else gradients[index]
                if self.pair_gap_offsets[k]:gap+=self.pair_gap_offsets[k]
                result.append((gap,grad,(self.names[a],self.names[b]),(int(ka[i]),int(kb[j]))))
                if stop_at_first:return result
        return result

    def collision(self,data):
        # A solve audits the same start/end geometry several times. Cache exact
        # body transforms only; changed arm, leg or base poses get a fresh audit.
        # Bound memory and return a new result object to avoid caller mutation.
        # Pair offsets are immutable per instance, as are its configured spheres.
        key=data.xpos[self.ids].tobytes()+data.xmat[self.ids].tobytes()
        if not hasattr(self,'_collision_cache'):self._collision_cache={}
        if key in self._collision_cache:
            cached=self._collision_cache[key]
            return None if cached is None else dict(pair=list(cached[0]),margin_m=cached[1])
        hits=self.contacts(data,threshold=0.,derivatives=False,per_pair=1,stop_at_first=True)
        cached=(tuple(hits[0][2]),hits[0][0]) if hits else None
        if len(self._collision_cache)>=64:self._collision_cache.pop(next(iter(self._collision_cache)))
        self._collision_cache[key]=cached
        return None if cached is None else dict(pair=list(cached[0]),margin_m=cached[1])


def fk(model,data,q):
    data.qpos[:]=q;mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)


def differences(n,order,dt):
    coefficients=([-1,1] if order==1 else [1,-2,1])
    d=sp.diags([np.full(n-order,c) for c in coefficients],np.arange(order+1),shape=(n-order,n))/dt**order
    return sp.kron(d,sp.eye(29),format='csc')


class Refiner:
    def __init__(self, settings=None):
        self.settings=settings or Settings();gmr.initialize()
        self.model=gmr.model;self.data=mujoco.MjData(self.model)
        self.geometry=Distances(self.model);self.A,self.b=coupled_halfspaces(gmr.names)
        assert tuple(gmr.names)==SDK_NAMES[:29]
        self.links=[entry[0] for entry in gmr.retargeter.ik_match_table2.items()]
        self.human_names=[entry[0] for entry in gmr.retargeter.ik_match_table2.values()]
        self.ids=[self.model.body(name).id for name in self.links]
        # Metre and radian residuals need different scales. GMR's inherited
        # 10:5 costs gave a 0.1 m miss less weight than a 0.3 rad rotation miss.
        self.position_cost=np.array([100 if 'wrist' in n else 80 if 'elbow' in n else
                                     80 if 'foot' in n else 40 if n=='pelvis' else 20 for n in self.links])
        self.rotation_cost=np.array([3 if 'wrist' in n else 2 if 'elbow' in n else 3 for n in self.links])

    def elbow_frames(self,positions,rotations,directions):
        result=rotations.copy()
        for side,s in enumerate(('l','r')):
            i=self.links.index(s+'_elbow_bracket');previous=None
            # IGRIS elbow_pitch is about local +Y. Forearm axial twist is carried
            # by the downstream wrist_yaw; SMPL elbow axial twist must not be
            # imposed on this upstream bracket as an independent orientation.
            local_f=gmr.reference_pos[s+'_wrist_connector']-gmr.reference_pos[s+'_elbow_bracket']
            h=np.array([0.,1.,0.]);f=local_f-h*(h@local_f);f/=np.linalg.norm(f)
            local=np.column_stack((np.cross(h,f),h,f))
            for t in range(len(positions)):
                upper,forearm=directions[t,side];hinge=-np.cross(upper,forearm)
                if np.linalg.norm(hinge)<.08:
                    hinge=rotations[t,i,:,1] if previous is None else previous
                    hinge=hinge-forearm*(hinge@forearm)
                if np.linalg.norm(hinge)<1e-8:raise ValueError('degenerate elbow hinge frame')
                hinge/=np.linalg.norm(hinge);forearm=forearm-hinge*(hinge@forearm);forearm/=np.linalg.norm(forearm)
                world=np.column_stack((np.cross(hinge,forearm),hinge,forearm))
                result[t,i]=world@local.T;previous=hinge
        return result

    def targets(self, clip):
        n=len(clip['time']);positions=[];rotations=[];directions=[]
        for t in range(n):
            human={name:(clip['human_positions'][t,j],clip['human_quaternions'][t,j])
                   for j,name in enumerate(gmr.JOINT_NAMES)}
            gmr.morphology_targets(human)
            scaled=gmr.retargeter.scaled_human_data
            positions.append([scaled[k][0] for k in self.human_names])
            rotations.append([scaled[k][1] for k in self.human_names])
            dirs=[]
            for side in ('left','right'):
                v=np.diff([scaled[side+'_'+j][0] for j in ('shoulder','elbow','wrist')],axis=0)
                dirs.append(v/np.linalg.norm(v,axis=1,keepdims=True))
            directions.append(dirs)
        p=np.array(positions);r=Rotation.from_quat(np.array(rotations).reshape(-1,4),scalar_first=True).as_matrix().reshape(n,-1,3,3);d=np.array(directions)
        return p,self.elbow_frames(p,r,d),d

    def pose_objective(self,q,target_p,target_r):
        fk(self.model,self.data,q);js=[];es=[]
        for i,b in enumerate(self.ids):
            p=self.data.xpos[b];r=self.data.xmat[b].reshape(3,3)
            jp=np.empty((3,self.model.nv));jr=jp.copy();mujoco.mj_jacBody(self.model,self.data,jp,jr,b)
            # World position residual plus chordal rotation residual (three axes).
            # This avoids mixing translation with rotation in a single SE(3) log.
            es.append((p-target_p[i])*self.position_cost[i]);js.append(jp[:,6:]*self.position_cost[i])
            for axis in range(3):
                v=r[:,axis];es.append((v-target_r[i,:,axis])*self.rotation_cost[i]/np.sqrt(2))
                cross=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
                js.append(-cross@jr[:,6:]*self.rotation_cost[i]/np.sqrt(2))
        return np.concatenate(es),np.vstack(js)

    def optimization_contacts(self):
        s=self.settings
        contacts=self.geometry.contacts(self.data,s.collision_activation,per_pair=max(4,s.collision_constraints_per_pair))
        counts={};kept=[]
        for contact in contacts:
            gap,_,pair,_=contact;count=counts.get(pair,0);counts[pair]=count+1
            # Several sphere normals can be active on one mesh link. Keeping
            # only the closest lets the QP alternate between overlapping spheres.
            if count<s.collision_constraints_per_pair or gap<s.collision_clearance+.005:kept.append(contact)
        return kept

    def refine(self,clip,log):
        n=len(clip['time']);targets=self.targets(clip)
        if n<=self.settings.window_frames:return self.refine_window(clip,log,targets)
        q=clip['qpos'].astype(float).copy();history=[];begin=0
        while begin<n:
            end=min(n,begin+self.settings.window_frames)
            chunk={k:(v[begin:end] if k in ('qpos','time','human_positions','human_quaternions') else v) for k,v in clip.items()}
            fixed=q[begin:begin+3].copy() if begin else None
            part,_,records=self.refine_window(chunk,log,tuple(v[begin:end] for v in targets),fixed)
            q[begin:end]=part;history.extend([dict(window_start=begin,window_end=end,**r) for r in records])
            if end==n:break
            begin=end-self.settings.window_overlap
        return q,targets,history

    def refine_window(self,clip,log,targets,fixed=None):
        s=self.settings;q=clip['qpos'].astype(float).copy();n=len(q)
        ts=clip['time'];dt=float(np.median(np.diff(ts)))
        if n<6 or not np.allclose(np.diff(ts),dt,atol=1e-7):raise ValueError('at least six uniformly spaced frames required')
        p,r,directions=targets
        # The old soft torso task can leave the shoulder anchor far from its
        # intended location. Root directly at the same morphology-derived torso
        # target, then fit the limbs; shoulder-relative targets become reachable.
        torso=self.links.index('torso');q[:,:3]=p[:,torso]
        q[:,3:7]=Rotation.from_matrix(r[:,torso]).as_quat(scalar_first=True)
        # Refit from a coherent torso-root branch before adding trajectory
        # constraints. A wrong old torso yaw can leave both arms folded through
        # the body; collision recovery alone cannot reliably change that branch.
        previous=np.zeros(29);previous[15:29]=[0,.35,0,-1.0,0,0,0,0,-.35,0,-1.0,0,0,0]
        inequality=np.vstack((np.eye(29),-np.eye(29),self.A))
        for t in range(n):
            if fixed is not None and t<len(fixed):q[t]=fixed[t];previous=q[t,7:].copy();continue
            q[t,7:]=previous
            for _ in range(30 if t==0 else 8):
                error,jac=self.pose_objective(q[t],p[t],r[t]);x=q[t,7:]
                upper=np.r_[np.minimum(self.model.jnt_range[1:,1]-x,.2),
                            np.minimum(x-self.model.jnt_range[1:,0],.2),self.b-self.A@x]
                continuity=0. if t==0 else 16.
                h=jac.T@jac+np.eye(29)*(.5+continuity);c=jac.T@error+continuity*(x-previous)
                # Starting inside the pelvis can trap later trajectory recovery
                # on the wrong side of a hand/forearm obstacle. Include the same
                # sphere constraints while following targets from the clear seed.
                contacts=self.geometry.contacts(self.data,s.collision_activation,per_pair=4)
                if contacts:
                    gaps=np.array([v[0] for v in contacts]);grad=np.array([v[1] for v in contacts]);m=len(gaps)
                    mat=np.block([[inequality,np.zeros((len(upper),m))],[-grad,-np.eye(m)],
                                  [np.zeros((m,29)),-np.eye(m)]])
                    h=sp.block_diag((h,np.eye(m)*s.collision_slack_cost)).toarray();c=np.r_[c,np.full(m,100.)]
                    step=solve_qp(h,c,mat,np.r_[upper,gaps-s.collision_clearance,np.zeros(m)],solver='daqp')
                else:step=solve_qp(h,c,inequality,upper,solver='daqp')
                if step is None:raise RuntimeError(f'pose initialization infeasible at frame {t}')
                step=step[:29]
                q[t,7:]+=step
                if abs(step).max()<1e-4:break
            previous=q[t,7:].copy()
        q_original=q[:,7:].ravel().copy();dim=n*29;eye=sp.eye(dim,format='csc')
        d1= differences(n,1,dt);d2=differences(n,2,dt)
        smooth=s.velocity_cost**2*(d1.T@d1)+s.acceleration_cost**2*(d2.T@d2)
        coupled=sp.kron(sp.eye(n),sp.csc_matrix(self.A),format='csc')
        joint_min=np.tile(self.model.jnt_range[1:,0],n);joint_max=np.tile(self.model.jnt_range[1:,1],n)
        history=[];witnesses={}
        for iteration in range(s.iterations+s.corridor_repair_iterations):
            start=time.monotonic();x=q[:,7:].ravel();h=[];c=[];cr=[];cc=[];cv=[];bounds=[];min_gap=1.
            def add_contacts(t,alpha,contacts):
                nonlocal min_gap
                for gap,grad,_,_ in contacts:
                    row=len(bounds);cols=np.arange(29)+29*t
                    cr.extend([row]*29);cc.extend(cols);cv.extend((1-alpha)*grad)
                    if alpha:
                        cr.extend([row]*29);cc.extend(cols+29);cv.extend(alpha*grad)
                    bounds.append(s.collision_clearance-gap);min_gap=min(min_gap,gap)
            for t in range(n):
                error,jac=self.pose_objective(q[t],p[t],r[t]);h.append(jac.T@jac+np.eye(29)*.5);c.append(jac.T@error)
                add_contacts(t,0.,self.optimization_contacts())
                if t+1<n:
                    steps=max(2,int(np.ceil(np.max(np.abs(q[t+1,7:]-q[t,7:]))/s.optimization_corridor_step)))
                    steps=min(steps,s.max_corridor_samples)
                    alphas=sorted(set((np.arange(1,steps)/steps).tolist()+witnesses.get(t,[])))
                    for alpha in alphas:
                        middle=q[t].copy();middle[7:]=(1-alpha)*q[t,7:]+alpha*q[t+1,7:]
                        fk(self.model,self.data,middle)
                        add_contacts(t,float(alpha),self.optimization_contacts())
            h=sp.block_diag(h,format='csc')+smooth
            c=np.concatenate(c)+smooth@x+.05*(x-q_original)
            trust=1.0 if iteration==0 and s.enforce_timing else s.trust_region
            base_a=sp.vstack([eye,d1,d2,coupled],format='csc')
            vmax=s.velocity_limit if s.enforce_timing else np.inf;amax=s.acceleration_limit if s.enforce_timing else np.inf
            lower=np.r_[np.maximum(joint_min-x,-trust),-vmax-d1@x,-amax-d2@x,np.full(coupled.shape[0],-np.inf)]
            upper=np.r_[np.minimum(joint_max-x,trust),vmax-d1@x,amax-d2@x,np.tile(self.b,n)-coupled@x]
            if fixed is not None:lower[:len(fixed)*29]=0;upper[:len(fixed)*29]=0
            count=len(bounds);collision=sp.csc_matrix((cv,(cr,cc)),shape=(count,dim))
            identity=sp.eye(count,format='csc')
            mat=sp.bmat([[base_a,None],[collision,identity],[None,identity]],format='csc')
            lo=np.r_[lower,bounds,np.zeros(count)];hi=np.r_[upper,np.full(2*count,np.inf)]
            objective=sp.block_diag([h,s.collision_slack_cost*identity],format='csc')
            objective_c=np.r_[c,np.full(count,100.)]
            qp=osqp.OSQP();qp.setup(P=sp.triu(objective,format='csc'),q=objective_c,A=mat,l=lo,u=hi,
                verbose=False,eps_abs=1e-5,eps_rel=1e-5,max_iter=30000,polishing=True)
            solution=qp.solve(raise_error=False)
            if solution.info.status_val not in (1,2):raise RuntimeError(f'OSQP {solution.info.status}; iteration {iteration}')
            delta=solution.x[:dim].reshape(n,29);q[:,7:]+=delta
            row=dict(iteration=iteration,seconds=time.monotonic()-start,constraints=count,
                minimum_linearization_gap_m=min_gap,max_joint_update_rad=float(abs(delta).max()),
                max_collision_slack_m=max(0.,float(solution.x[dim:].max())) if count else 0.,qp_status=solution.info.status)
            row.update(qp_seconds=solution.info.run_time)
            stop=iteration>=4 and abs(delta).max()<.003 and min_gap>=.0005
            if stop or iteration>=s.iterations-1:
                added=0
                for t in range(n-1):
                    fk(self.model,self.data,q[t]);left_hit=self.geometry.collision(self.data)
                    fk(self.model,self.data,q[t+1]);right_hit=self.geometry.collision(self.data)
                    if left_hit or right_hit:continue
                    steps=max(1,int(np.ceil(np.max(np.abs(q[t+1,7:]-q[t,7:]))/s.corridor_step)))
                    for alpha in np.arange(1,steps)/steps:
                        middle=q[t].copy();middle[7:]=(1-alpha)*q[t,7:]+alpha*q[t+1,7:]
                        fk(self.model,self.data,middle)
                        if self.geometry.collision(self.data):
                            if all(abs(alpha-a)>1e-8 for a in witnesses.get(t,[])):
                                witnesses.setdefault(t,[]).append(float(alpha));added+=1
                            break
                row['added_corridor_witnesses']=added
                if added:stop=False
            history.append(row);print(json.dumps(row),flush=True);log.write(json.dumps(row)+'\n');log.flush()
            if stop:break
        return q,(p,r,directions),history

    def audit(self,q,targets,ts):
        p,r,directions=targets;n=len(q);flags=np.zeros((n,11),bool);pe=np.zeros((n,2,2));ae=pe.copy();de=pe.copy();states=[];pairs={}
        for t in range(n):
            fk(self.model,self.data,q[t]);states.append(gmr.local_states(self.data))
            flags[t,0]=not np.isfinite(q[t]).all()
            flags[t,1]=bool(np.any(q[t,7:]<self.model.jnt_range[1:,0]-1e-5) or np.any(q[t,7:]>self.model.jnt_range[1:,1]+1e-5))
            flags[t,2]=np.max(self.A@q[t,7:]-self.b)>1e-5
            hit=self.geometry.collision(self.data)
            if hit:flags[t,3]=True;pair='/'.join(hit['pair']);pairs[pair]=pairs.get(pair,0)+1
            for side,s in enumerate(('l','r')):
                for k,name in enumerate((s+'_elbow_bracket',s+'_wrist_connector')):
                    i=self.links.index(name);b=self.ids[i]
                    pe[t,side,k]=np.linalg.norm(self.data.xpos[b]-p[t,i])
                    ae[t,side,k]=Rotation.from_matrix(r[t,i].T@self.data.xmat[b].reshape(3,3)).magnitude()
                points=[self.data.xpos[self.model.body(s+'_'+name).id] for name in ('shoulder_frame','elbow_bracket','wrist_connector')]
                v=np.diff(points,axis=0);v/=np.linalg.norm(v,axis=1,keepdims=True)
                de[t,side]=np.arccos(np.clip((v*directions[t,side]).sum(1),-1,1))
            flags[t,4]=pe[t].max()>.10;flags[t,5]=ae[t,:,1].max()>np.deg2rad(40);flags[t,6]=de[t].max()>np.deg2rad(40)
            flags[t,10]=ae[t,:,0].max()>np.deg2rad(40)
        velocity=np.diff(q[:,7:],axis=0)/np.diff(ts)[:,None]
        accel=np.diff(velocity,axis=0)/np.diff(ts)[1:,None]
        bad=abs(velocity[:,15:29]).max(1)>3+1e-4;flags[:-1,7]|=bad;flags[1:,7]|=bad
        bad=abs(accel[:,15:29]).max(1)>60+1e-3;flags[:-2,8]|=bad;flags[1:-1,8]|=bad;flags[2:,8]|=bad
        collision_connections=0;interior_only=0
        for t in range(n-1):
            if self.geometry.corridor(self.data,q[t],q[t+1],max_step=self.settings.corridor_step):
                flags[t:t+2,9]=True;collision_connections+=1
                interior_only+=not flags[t:t+2,3].any()
        valid=~flags.any(1);window=np.convolve(valid.astype(int),np.ones(6,int),'valid')==6
        strict_valid=valid&(pe.max((1,2))<=.03)&(ae.max((1,2))<=np.deg2rad(20))&(de.max((1,2))<=np.deg2rad(20))
        strict_windows=np.convolve(strict_valid.astype(int),np.ones(6,int),'valid')==6
        jerk=np.diff(accel,axis=0)/np.diff(ts)[2:,None]
        reasons=gmr.REASONS+['elbow_orientation']
        metrics=dict(frames=n,valid_frames=int(valid.sum()),valid_fraction=float(valid.mean()),valid_windows=int(window.sum()),
            rejection_counts=dict(zip(reasons,flags.sum(0).tolist())),collision_pairs=pairs,
            position_mean_m=float(pe.mean()),position_p95_m=float(np.percentile(pe,95)),position_max_m=float(pe.max()),
            wrist_position_mean_m=float(pe[:,:,1].mean()),elbow_position_mean_m=float(pe[:,:,0].mean()),
            orientation_mean_deg=float(np.rad2deg(ae).mean()),orientation_p95_deg=float(np.percentile(np.rad2deg(ae),95)),
            wrist_orientation_mean_deg=float(np.rad2deg(ae[:,:,1]).mean()),elbow_orientation_mean_deg=float(np.rad2deg(ae[:,:,0]).mean()),
            direction_mean_deg=float(np.rad2deg(de).mean()),direction_p95_deg=float(np.percentile(np.rad2deg(de),95)),
            velocity_max_rad_s=float(abs(velocity[:,15:29]).max()),velocity_p95_rad_s=float(np.percentile(abs(velocity[:,15:29]),95)),
            acceleration_max_rad_s2=float(abs(accel[:,15:29]).max()),acceleration_p95_rad_s2=float(np.percentile(abs(accel[:,15:29]),95)),
            jerk_p95_rad_s3=float(np.percentile(abs(jerk[:,15:29]),95)),joint_step_max_rad=float(abs(np.diff(q[:,22:],axis=0)).max()),
            collision_connections=collision_connections,interior_only_collision_connections=int(interior_only),
            strict_valid_windows=int(strict_windows.sum()),strict_30mm_20deg_fraction=float(strict_valid.mean()))
        return metrics,dict(state=np.array(states,dtype=np.float32),valid=valid,strict_valid=strict_valid,reject_flags=flags,reason_names=np.array(reasons),
            target_position_error=pe,target_angle_error=ae,arm_angle_error=de)


def retime(clip,q):
    """Linear joint path keeps every sampled collision corridor unchanged.

    A piecewise-linear path has velocity corners. At fixed output 30 Hz, its
    discrete acceleration bound scales conservatively by 1/factor, not 1/factor².
    Original-time results are retained so slowdown cannot masquerade as fidelity.
    """
    ts=clip['time'];dt=float(np.median(np.diff(ts)))
    velocity=np.diff(q[:,7:],axis=0)/dt;accel=np.diff(velocity,axis=0)/dt
    factor=max(1.,float(abs(velocity).max())/2.8,float(abs(accel).max())/55.)*1.002
    duration=np.ceil((ts[-1]-ts[0])*factor/dt)*dt;factor=duration/(ts[-1]-ts[0])
    times=np.arange(round(duration/dt)+1)*dt;source=ts[0]+times/factor
    source=np.clip(source,ts[0],ts[-1])
    def interp(values):
        flat=values.reshape(len(ts),-1)
        return np.column_stack([np.interp(source,ts,flat[:,i]) for i in range(flat.shape[1])]).reshape((len(times),)+values.shape[1:])
    result={k:v for k,v in clip.items() if k in ('joint_names','source_sha256')}
    result.update(time=times,source_time=source,human_positions=interp(clip['human_positions']))
    hq=clip['human_quaternions']
    result['human_quaternions']=np.stack([Slerp(ts,Rotation.from_quat(hq[:,i],scalar_first=True))(source).as_quat(scalar_first=True) for i in range(hq.shape[1])],axis=1)
    output=interp(q);output[:,3:7]=Slerp(ts,Rotation.from_quat(q[:,3:7],scalar_first=True))(source).as_quat(scalar_first=True)
    return result,output,factor


def run(path,out,settings,refiner=None,row=None):
    source=OUTPUT/'data/clips'/path
    if not source.is_file():raise FileNotFoundError(source)
    dest=Path(out);dest.mkdir(parents=True,exist_ok=False)
    implementation_sha256=sha256(__file__)
    with np.load(source) as z:clip={k:z[k] for k in z.files}
    refiner=refiner or Refiner(settings)
    # Recover the existing per-clip scale, preserving exactly the original targets.
    if row is None:
        rows={r['path']:r for r in map(json.loads,(OUTPUT/'data/retarget_results.jsonl').read_text().splitlines())};row=rows[path]
    refiner_settings=asdict(settings)
    gmr.retargeter.human_scale_table={k:v*row['estimated_height_m']/1.8 for k,v in gmr.default_scale.items()}
    before_hash={str(p):sha256(p) for p in (source,HERE/'retarget.py',HERE/'geometry.py',HERE/'human.py')}
    with (dest/'iterations.jsonl').open('w') as log:q,targets,iterations=refiner.refine(clip,log)
    baseline,_=refiner.audit(clip['qpos'],targets,clip['time']);refined,arrays=refiner.audit(q,targets,clip['time'])
    comparison=dict(source=str(source),source_path=path,split=row['split'],settings=refiner_settings,
        baseline=baseline,refined=refined,iterations=iterations,collision=refiner.geometry.provenance(),
        source_hashes=before_hash,implementation_sha256=implementation_sha256,
        timestamps_preserved=True,root_trajectory_preserved=False,
        root_policy='anchor torso to original morphology target; optimize 29 joints with fixed per-frame torso',hardware_executed=False,
        audit='all input frames and every corridor, both methods; no frame subsampling; 0.01 rad corridor sampling',
        quality_criteria='same 100 mm/40 deg/3 rad/s/60 rad/s2, plus strict 30 mm/20 deg indicator')
    comparison['elbow_target']='IGRIS hinge frame from upper/lower-arm directions; SMPL axial twist remains in wrist target'
    comparison['baseline_original_elbow_orientation_mean_deg']=float(np.rad2deg(clip['target_angle_error'][:,:,0]).mean())
    if not settings.enforce_timing:
        slowed,slow_q,factor=retime(clip,q)
        # Match source precision before auditing the values that will be saved.
        for key in ('human_positions','human_quaternions'):slowed[key]=slowed[key].astype(np.float32)
        slow_targets=refiner.targets(slowed)
        slow_metrics,slow_arrays=refiner.audit(slow_q,slow_targets,slowed['time'])
        slowed.update(slow_arrays,qpos=slow_q)
        np.savez_compressed(dest/'retimed_clip.npz',**slowed)
        comparison.update(retimed=slow_metrics,time_scale=factor,original_duration_s=float(clip['time'][-1]-clip['time'][0]),retimed_duration_s=float(slowed['time'][-1]))
    for p,h in before_hash.items():
        if sha256(p)!=h:raise RuntimeError(f'source changed during refinement: {p}')
    clip.update(arrays);clip['qpos']=q # Retain float64 for constraint verification.
    np.savez_compressed(dest/'clip.npz',**clip)
    save_json(dest/'comparison.json',comparison)
    print(json.dumps(dict(path=path,baseline=baseline,refined=refined)),flush=True)
    return comparison


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('path');parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=25);parser.add_argument('--retime',action='store_true');args=parser.parse_args()
    settings=Settings(iterations=args.iterations,enforce_timing=not args.retime)
    if args.retime:settings.velocity_cost=.08;settings.acceleration_cost=.01
    run(args.path,args.out,settings)
