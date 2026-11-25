"""Separate planned IK error from measured native hand tracking error."""
import argparse,sys
import numpy as np
import mujoco
from scipy.spatial.transform import Slerp
from .common import *

def analyze(trial,plan,*,exact_clock=False):
    sys.path.insert(0,str(IGRIS/'sim_local/vr_box_eval'));sys.path.insert(0,str(IGRIS/'igris_curobo/src'))
    from render_results import model_for
    out=OUTPUT/'native'/trial;inputs=json.loads((out/'input.json').read_text())
    f=np.load(out/'physics.npz');t=f['t'];qp=f['qpos']
    it=np.array([v['t'] for v in inputs]);progress=np.interp(t,it,[v.get('source_motion_time_s',v['t']) for v in inputs])
    if exact_clock:
        import struct
        meta=json.loads((out/'metadata.json').read_text());result=json.loads((out/'result.json').read_text())
        raw=(out/'physics.bin').read_bytes();nq,nv=struct.unpack('ii',raw[:8]);rows=np.frombuffer(raw[8:],np.float64).reshape(-1,2+nq+nv+7)
        relative=rows[:,0]-meta['start_wall'];rows=rows[(relative>=0)&(relative<=meta['duration'])]
        if len(rows)!=len(t) or not np.allclose(rows[:,0]-meta['start_wall'],t):raise ValueError('physics clocks do not match')
        progress=rows[:,1]-result['start_sim']
    z=np.load(OUTPUT/'box_plans'/plan/'ik.npz');pt=z['time'];poses=z['goals'];progress=np.clip(progress,pt[0],pt[-1])
    targets=np.stack([np.column_stack([np.interp(progress,pt,poses[:,s,j]) for j in range(3)]) for s in range(2)],axis=1)
    rotations=[Slerp(pt,Rotation.from_quat(poses[:,s,3:],scalar_first=True))(progress) for s in range(2)]
    cmds=np.load(OUTPUT/'box_plans'/plan/'motion.npz');q=np.column_stack([np.interp(progress,cmds['time'],cmds['q'][:,j]) for j in range(31)])
    if exact_clock:
        sent=np.load(out/'commands.npz')['records'];indices=np.searchsorted(sent[:,0],meta['start_wall']+t,side='right')-1
        if np.any(indices<0):raise ValueError('physics samples precede first command')
        q=sent[indices,3:]
    native=model_for(out);nd=mujoco.MjData(native);torso=native.body('torso').id
    kin=mujoco.MjModel.from_xml_path(str(OUTPUT/'model/igris_c_rcv_hand_torso.xml'));kd=mujoco.MjData(kin)
    addresses=[kin.joint(n).qposadr[0] for n in SDK_NAMES[:29]]
    error=[]
    for i in range(len(t)):
        nd.qpos[:]=qp[i];mujoco.mj_kinematics(native,nd);rt=nd.xmat[torso].reshape(3,3).T;origin=nd.xpos[torso]
        kd.qpos[addresses]=q[i,:29];mujoco.mj_kinematics(kin,kd);values=[]
        for s,name in enumerate(EE_NAMES):
            a=native.body(name).id;b=kin.body(name).id
            actual=rt@(nd.xpos[a]-origin);actual_r=Rotation.from_matrix(rt@nd.xmat[a].reshape(3,3));planned=kd.xpos[b];planned_r=Rotation.from_matrix(kd.xmat[b].reshape(3,3))
            values.append([np.linalg.norm(actual-targets[i,s]),(rotations[s][i].inv()*actual_r).magnitude(),
                np.linalg.norm(actual-planned),(planned_r.inv()*actual_r).magnitude(),np.linalg.norm(planned-targets[i,s]),(rotations[s][i].inv()*planned_r).magnitude()])
        error.append(values)
    error=np.array(error);phase=np.array([inputs[min(np.searchsorted(it,sec),len(it)-1)]['stage'] for sec in t])
    summary={}
    for label,mask in [('all',np.ones(len(t),bool)),*[(p,phase==p) for p in sorted(set(phase))]]:
        v=error[mask];summary[label]=dict(frames=len(v),measured_goal_position_mean_m=float(v[:,:,0].mean()),measured_goal_orientation_mean_deg=float(np.rad2deg(v[:,:,1]).mean()),
            measured_command_position_mean_m=float(v[:,:,2].mean()),measured_command_orientation_mean_deg=float(np.rad2deg(v[:,:,3]).mean()),
            planned_goal_position_mean_m=float(v[:,:,4].mean()),planned_goal_orientation_mean_deg=float(np.rad2deg(v[:,:,5]).mean()))
    prefix='tracking_exact' if exact_clock else 'tracking'
    definition=('goals at actual recorded physics time minus native start_sim; last published q at each recorded wall time (no future commands); ' if exact_clock else '')
    save_json(out/(prefix+'.json'),dict(definition=definition+'measured native EE FK in measured torso frame; command FK from torso model; means weight recorded frames equally (30 Hz wall-clock sampling), not simulation-time intervals',metrics=summary))
    np.savez_compressed(out/(prefix+'.npz'),time=t,motion_time=progress,error=error,phase=phase)
    print(summary['all'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('trial');ap.add_argument('plan');ap.add_argument('--exact-clock',action='store_true');a=ap.parse_args();analyze(a.trial,a.plan,exact_clock=a.exact_clock)
