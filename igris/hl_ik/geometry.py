"""Full configured sphere collision and vendor coupled limits on torso MJCF."""
import json
from collections.abc import Mapping
from numbers import Real
from types import MappingProxyType
import numpy as np
import mujoco
import yaml
from scipy.spatial.distance import cdist
from .common import *

def coupled_halfspaces(names):
    config=json.loads((IGRIS/'igris_curobo/src/igris_ik/data/vendor/joint_limits.json').read_text())
    rows=[];bounds=[]
    for pair in config['coupled_limits']:
        if not all(n in names for n in pair['joints']):continue
        v=np.asarray(pair['vertices']);v=v.mean(0)+.95*(v-v.mean(0))
        e=np.roll(v,-1,axis=0)-v;n=np.column_stack((e[:,1],-e[:,0]));n/=np.linalg.norm(n,axis=1,keepdims=True)
        b=(n*v).sum(1);a=np.zeros((len(v),len(names)));a[:,[names.index(j) for j in pair['joints']]]=n
        assert np.max(v@n.T-b)<1e-8
        rows.extend(a);bounds.extend(b)
    return np.asarray(rows).reshape(-1,len(names)),np.asarray(bounds)

class SphereGeometry:
    def __init__(self,model,buffer=.0025,hand_envelope=None,pair_buffer_overrides=None):
        if isinstance(buffer,(bool,np.bool_)) or not isinstance(buffer,Real):
            raise ValueError('Expected finite nonnegative sphere buffer and pair offset')
        try:buffer_value=float(buffer)
        except (TypeError,ValueError,OverflowError) as error:
            raise ValueError('Expected finite nonnegative sphere buffer and pair offset') from error
        if not np.isfinite(buffer_value) or not 0<=buffer_value<=float(np.finfo(float).max)/2:
            raise ValueError('Expected finite nonnegative sphere buffer and pair offset')
        self.model=model;self.buffer=buffer
        self.source=IGRIS/'igris_curobo/src/igris_ik/data/curobo/igris_c_rcv_hand_pelvis.yml'
        cfg=yaml.safe_load(self.source.read_text())['kinematics']
        self.hand_envelope=hand_envelope
        if hand_envelope:
            override=json.loads(Path(hand_envelope).read_text())['spheres']
            if set(override)!={'l_hand_root','r_hand_root'}:raise ValueError('unexpected hand envelope links')
            cfg['collision_spheres'].update(override)
        self.names=cfg['collision_link_names'];self.ids=np.array([model.body(n).id for n in self.names])
        self.local=[];self.radius=[]
        for name in self.names:
            v=np.array([s['center']+[s['radius']] for s in cfg['collision_spheres'][name]])
            self.local.append(v[:,:3]+PELVIS_TO_TORSO_LINK_OFFSET.get(name,np.zeros(3)))
            self.radius.append(v[:,3]+buffer)
        ignore=cfg['self_collision_ignore']
        self.pairs=[(i,j) for i,a in enumerate(self.names) for j,b in enumerate(self.names) if i<j
            and b not in ignore.get(a,[]) and a not in ignore.get(b,[])]
        self.ia=np.array([p[0] for p in self.pairs]);self.ib=np.array([p[1] for p in self.pairs])
        # Retain the configured radii and broad phase. A pair-specific reduction
        # of padding adds a constant to its gap; it never excludes a pair or
        # shrinks the underlying raw spheres. Freeze a copy for collision caches.
        configured={(self.names[a],self.names[b]):k for k,(a,b) in enumerate(self.pairs)}
        overrides={}
        if pair_buffer_overrides is not None:
            if not isinstance(pair_buffer_overrides,Mapping):
                raise ValueError('Expected pair_buffer_overrides mapping')
            for pair,value in pair_buffer_overrides.items():
                if not isinstance(pair,tuple) or len(pair)!=2 or not all(isinstance(n,str) for n in pair):
                    raise ValueError('Expected body-name tuple for pair buffer override')
                canonical=pair if pair in configured else pair[::-1]
                if canonical not in configured:
                    raise ValueError(f'Pair buffer override is not a configured collision pair: {pair}')
                if canonical in overrides:
                    raise ValueError(f'Duplicate reversed pair buffer override: {pair}')
                if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real):
                    raise ValueError('Pair buffer must be finite and between zero and the global buffer')
                try:numeric=float(value)
                except (TypeError,ValueError,OverflowError) as error:
                    raise ValueError('Pair buffer must be finite and between zero and the global buffer') from error
                if not np.isfinite(numeric) or not 0<=numeric<=buffer:
                    raise ValueError('Pair buffer must be finite and between zero and the global buffer')
                overrides[canonical]=numeric
        self.pair_buffer_overrides=MappingProxyType({p:overrides[p] for p in configured if p in overrides})
        self.pair_gap_offsets=np.zeros(len(self.pairs))
        for pair,value in self.pair_buffer_overrides.items():
            self.pair_gap_offsets[configured[pair]]=2.*(buffer-value)
        self.pair_gap_offsets.setflags(write=False)
        # Bounding spheres provide an inexpensive first stage before exact spheres.
        self.center=np.array([v.mean(0) for v in self.local])
        self.bound=np.array([np.max(np.linalg.norm(v-c,axis=1)+r) for v,c,r in zip(self.local,self.center,self.radius)])

    def collision(self,data):
        rot=data.xmat[self.ids].reshape(-1,3,3);pos=data.xpos[self.ids]
        centers=np.einsum('bij,bj->bi',rot,self.center)+pos
        candidates=np.flatnonzero(np.linalg.norm(centers[self.ia]-centers[self.ib],axis=1)<self.bound[self.ia]+self.bound[self.ib])
        transformed={}
        for k in candidates:
            a,b=self.pairs[k]
            for j in (a,b):
                if j not in transformed:transformed[j]=self.local[j]@rot[j].T+pos[j]
            pa,pb=transformed[a],transformed[b];ra,rb=self.radius[a],self.radius[b]
            la,ha=(pa-ra[:,None]).min(0),(pa+ra[:,None]).max(0)
            lb,hb=(pb-rb[:,None]).min(0),(pb+rb[:,None]).max(0)
            if np.any(la>hb) or np.any(lb>ha):continue
            ka=((pa+ra[:,None]>=lb)&(pa-ra[:,None]<=hb)).all(1)
            kb=((pb+rb[:,None]>=la)&(pb-rb[:,None]<=ha)).all(1)
            gap=cdist(pa[ka],pb[kb])-ra[ka,None]-rb[None,kb]
            if self.pair_gap_offsets[k]:gap=gap+self.pair_gap_offsets[k]
            if gap.size and np.min(gap)<0:return dict(pair=[self.names[a],self.names[b]],margin_m=float(np.min(gap)))
        return None

    def corridor(self,data,a,b,max_step=.02):
        # ponytail: sampled joint corridor; upgrade to swept collision for hardware use.
        n=max(1,int(np.ceil(np.max(np.abs(b[7:]-a[7:]))/max_step)))
        old=data.qpos.copy()
        try:
            for t in np.linspace(0,1,n+1):
                data.qpos[:]=a;data.qpos[7:]=(1-t)*a[7:]+t*b[7:]
                mujoco.mj_kinematics(self.model,data)
                hit=self.collision(data)
                if hit:return hit
        finally:
            data.qpos[:]=old;mujoco.mj_kinematics(self.model,data)
        return None

    def provenance(self):
        return dict(source=str(self.source),sha256=sha256(self.source),spheres=sum(map(len,self.local)),
                    hand_envelope=str(self.hand_envelope) if self.hand_envelope else None,
                    link_count=len(self.names),pair_count=len(self.pairs),buffer_per_sphere_m=self.buffer,
                    pair_buffer_overrides=[dict(pair=list(pair),buffer_per_sphere_m=value,
                        gap_offset_m=2.*(self.buffer-value)) for pair,value in self.pair_buffer_overrides.items()],
                    ignore_policy='unchanged configured cuRobo ignore list; no automatic removal of failing pairs',
                    frame_offsets={k:v.tolist() for k,v in PELVIS_TO_TORSO_LINK_OFFSET.items()})
