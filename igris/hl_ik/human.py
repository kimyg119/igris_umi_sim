"""AMASS SMPL-X skeletal FK, without unnecessary posed vertex skinning.

For the first 22 skeletal joints this is algebraically identical to SMPL-X LBS:
shape-dependent rest joints followed by the kinematic tree. Tested against smplx.
"""
import argparse
from functools import lru_cache
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .common import *

JOINT_NAMES=('pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2',
 'left_ankle','right_ankle','spine3','left_foot','right_foot','neck','left_collar',
 'right_collar','head','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist')

def build_rigs(out=OUTPUT/'model'):
    for gender in ('NEUTRAL','MALE','FEMALE'):
        source=GMR/f'assets/body_models/smplx/SMPLX_{gender}.npz'
        dest=out/f'smplx_{gender.lower()}_rig.npz'
        if dest.exists():continue
        with np.load(source) as f:
            reg=f['J_regressor'][:22];base=reg@f['v_template']
            shape=np.einsum('jv,vck->jck',reg,f['shapedirs'][:,:,:16],optimize=True)
            parents=f['kintree_table'][0,:22];parents[0]=-1
        np.savez(dest,base=base,shape=shape,parents=parents,source_sha256=sha256(source))

@lru_cache(maxsize=3)
def rig(gender):
    with np.load(OUTPUT/'model'/f'smplx_{gender}_rig.npz') as f:return {k:f[k] for k in f.files}

def skeletal_fk(betas,rotvec,trans,gender):
    r=rig(gender);rest=r['base']+np.einsum('jck,k->jc',r['shape'],np.asarray(betas).reshape(-1)[:16])
    local=Rotation.from_rotvec(rotvec.reshape(-1,3)).as_matrix().reshape(-1,22,3,3)
    pos=np.empty((len(rotvec),22,3));rot=np.empty((len(rotvec),22,3,3))
    pos[:,0]=rest[0]+trans;rot[:,0]=local[:,0]
    for j in range(1,22):
        p=r['parents'][j];rot[:,j]=rot[:,p]@local[:,j]
        pos[:,j]=pos[:,p]+np.einsum('bij,j->bi',rot[:,p],rest[j]-rest[p])
    return pos,rot

def load_clip(path,fps=30.):
    with np.load(path) as f:
        rate=float(f['mocap_frame_rate']);body=f['pose_body'];root=f['root_orient'];trans=f['trans'];betas=f['betas'];gender=str(f['gender']).lower()
    if gender not in ('male','female','neutral'):raise ValueError(f'unsupported gender {gender}')
    if not np.isfinite(rate) or rate<=0:raise ValueError('invalid mocap_frame_rate')
    if body.ndim!=2 or body.shape[1]!=63 or root.shape!=(len(body),3) or trans.shape!=(len(body),3):raise ValueError('invalid AMASS SMPL-X shapes')
    if not all(np.isfinite(x).all() for x in (body,root,trans,betas)):raise ValueError('nonfinite source')
    if len(body)<2:raise ValueError('fewer than two source frames')
    # Use exact time axes, including native rates not divisible by 30.
    src=np.arange(len(body))/rate;ts=np.arange(int(np.floor(src[-1]*fps+1e-8))+1)/fps
    rv=np.concatenate((root[:,None,:],body.reshape(-1,21,3)),axis=1)
    sampled=np.stack([Slerp(src,Rotation.from_rotvec(rv[:,j]))(ts).as_rotvec() for j in range(22)],axis=1)
    sampled_trans=np.column_stack([np.interp(ts,src,trans[:,j]) for j in range(3)])
    p,r=skeletal_fk(betas,sampled,sampled_trans,gender)
    quat=Rotation.from_matrix(r.reshape(-1,3,3)).as_quat(scalar_first=True).reshape(len(ts),22,4)
    return p,quat,ts,dict(gender=gender,source_fps=rate,source_frames=len(body),
        estimated_height_m=float(1.66+.1*np.asarray(betas).reshape(-1)[0]))

def verify(path):
    import smplx,torch
    with np.load(path) as f:
        b=f['betas'][:16];root=f['root_orient'][:3];body=f['pose_body'][:3];tr=f['trans'][:3];gender=str(f['gender'])
    bm=smplx.create(GMR/'assets/body_models','smplx',gender=gender,use_pca=False,num_betas=16)
    with torch.no_grad():
        result=bm(betas=torch.tensor(b[None],dtype=torch.float32),global_orient=torch.tensor(root,dtype=torch.float32),
          body_pose=torch.tensor(body,dtype=torch.float32),transl=torch.tensor(tr,dtype=torch.float32),
          left_hand_pose=torch.zeros(3,45),right_hand_pose=torch.zeros(3,45),jaw_pose=torch.zeros(3,3),
          leye_pose=torch.zeros(3,3),reye_pose=torch.zeros(3,3),return_full_pose=True)
    p,r=skeletal_fk(b,np.concatenate((root[:,None],body.reshape(-1,21,3)),axis=1),tr,gender)
    err=float(np.max(np.linalg.norm(p-result.joints[:,:22].numpy(),axis=-1)))
    assert err<2e-6,err
    return dict(source=str(path),joint_position_max_error_m=err,joints=22,frames=3)

if __name__=='__main__':
    build_rigs();p=next((IGRIS/'retargeting/AMASS_dataset/CMU').rglob('*stageii.npz'))
    result=verify(p);save_json(OUTPUT/'model/smplx_fk_validation.json',result);print(result)
