"""Kinematic EE/elbow frame parity between learned torso and native models."""
import sys
import numpy as np
import mujoco
from .common import *

def main():
    sys.path.insert(0,str(IGRIS/'sim_local/vr_box_eval'));sys.path.insert(0,str(IGRIS/'igris_curobo/src'))
    from render_results import model_for
    a=model_for(IGRIS/'sim_local/vr_box_eval/results/20260914_synthetic_motion100_1kg')
    b=mujoco.MjModel.from_xml_path(str(OUTPUT/'model/igris_c_rcv_hand_torso.xml'))
    ad=mujoco.MjData(a);bd=mujoco.MjData(b);rng=np.random.default_rng(42);errors=[]
    for _ in range(64):
        for name in SDK_NAMES[:29]:
            j=b.joint(name).id;lo,hi=b.jnt_range[j];q=rng.uniform(max(lo,-.6),min(hi,.6))
            ad.qpos[a.joint(name).qposadr[0]]=q;bd.qpos[b.joint(name).qposadr[0]]=q
        mujoco.mj_kinematics(a,ad);mujoco.mj_kinematics(b,bd)
        torso=a.body('torso').id;rt=ad.xmat[torso].reshape(3,3).T
        for name in (*EE_NAMES,*ELBOW_NAMES):
            x=a.body(name).id;y=b.body(name).id;p=rt@(ad.xpos[x]-ad.xpos[torso]);r=rt@ad.xmat[x].reshape(3,3)
            errors.append([np.linalg.norm(p-bd.xpos[y]),Rotation.from_matrix(r.T@bd.xmat[y].reshape(3,3)).magnitude()])
    e=np.array(errors);result=dict(configurations=64,frames=list((*EE_NAMES,*ELBOW_NAMES)),
        max_position_m=float(e[:,0].max()),max_orientation_rad=float(e[:,1].max()),
        scope='kinematic frame parity with native passive offsets at zero; not dynamic closed-chain validation')
    save_json(OUTPUT/'model/native_frame_validation.json',result);print(result)
    assert e[:,0].max()<1e-5 and e[:,1].max()<2e-5

if __name__=='__main__':main()
