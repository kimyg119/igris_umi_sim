"""Summarize recorded gain trials after render_results.py --metrics-only (no physics steps)."""
import csv,json,sys
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from render_results import model_for
from igris_ik.constants import URDF_JOINT_NAMES


def summarize(out):
    meta=json.loads((out/'metadata.json').read_text())
    frames=json.loads((out/'input.json').read_text())
    rows=list(csv.DictReader(next(out.glob('*.csv')).open()))
    t=np.array([float(r['t_wall'])-meta['start_wall'] for r in rows])
    ft=np.array([f['t'] for f in frames])
    fi=np.clip(np.searchsorted(ft,t,side='right')-1,0,len(frames)-1)
    wrists=[19,20,21,26,27,28]
    def values(prefix,indices):return np.array([[float(r[f'{prefix}_{i}']) for i in indices] for r in rows])
    active=(t>=3)&(t<=meta['duration']-2)
    assert np.allclose(values('kp',wrists)[active],meta['wrist_kp'])
    assert np.allclose(values('kd',wrists)[active],3.)
    # Compare the same stage and near-maximum requested preload, excluding released holds.
    mask=((t>=25)&(t<40) if meta.get('squeeze_hold') else
          np.array([frames[i]['stage']=='squeeze' and frames[i]['task_value'][1]>=.03 for i in fi]))&active
    physics=np.load(out/'physics.npz')
    patches=np.fromfile(out/'physics.bin.contacts',np.float64).reshape(-1,7)
    pi=np.clip(np.searchsorted(patches[:,0],t+meta['start_wall']),0,len(patches)-1)
    zi=np.clip(np.searchsorted(physics['t'],t),0,len(physics['t'])-1)
    patch=patches[pi][:,[1,2,4,5]]
    joint_error=np.rad2deg(values('cmd_q',wrists)-values('q',wrists))
    model=model_for(out);actual=mujoco.MjData(model);command=mujoco.MjData(model)
    indices=[0,1,2,*range(15,29)]
    addresses=[model.jnt_qposadr[model.joint(URDF_JOINT_NAMES[i]).id] for i in indices]
    cq=values('cmd_q',indices);orientation=[];position=[]
    for k in np.flatnonzero(mask)[::10]:
        actual.qpos[:]=physics['qpos'][zi[k]];mujoco.mj_forward(model,actual)
        command.qpos[:]=actual.qpos;command.qpos[addresses]=cq[k];mujoco.mj_forward(model,command)
        ori=[];pos=[]
        for side in ('l','r'):
            b=model.body(side+'_wrist_connector').id
            a=Rotation.from_matrix(actual.xmat[b].reshape(3,3));c=Rotation.from_matrix(command.xmat[b].reshape(3,3))
            ori.append(np.rad2deg((c.inv()*a).magnitude()))
            pos.append(1000*np.linalg.norm(command.xpos[b]-actual.xpos[b]))
        orientation.append(ori);position.append(pos)
    result=dict(trial=out.name,wrist_kp=meta['wrist_kp'],wrist_kd=3.,
                task=json.loads((out/'contact_task.json').read_text()) if (out/'contact_task.json').exists() else None,
                metrics=json.loads((out/'metrics.json').read_text()),window_samples=int(mask.sum()))
    if np.any(mask):
        result.update(window_start_s=float(t[mask][0]),window_end_s=float(t[mask][-1]),
          wrist_joint_error_p95_deg=float(np.percentile(abs(joint_error[mask]),95)),
          wrist_joint_error_rmse_deg=float(np.sqrt(np.mean(joint_error[mask]**2))),
          wrist_ik_command_error_p95_deg=float(np.percentile(abs(np.rad2deg(values('ik_q',wrists)-values('cmd_q',wrists)))[mask],95)),
          wrist_speed_p95_deg_s=float(np.percentile(abs(np.rad2deg(values('dq',wrists)))[mask],95)),
          wrist_command_speed_p95_deg_s=float(np.percentile(abs(np.rad2deg(np.gradient(values('cmd_q',wrists),t,axis=0)))[mask],95)),
          palm_tip_simultaneous_fraction=float(np.mean(np.all(patch[mask]>.2,axis=1))),
          palm_tip_mean_N=patch[mask].mean(0).tolist(),
          bilateral_force_mean_N=physics['forces'][zi[mask],:2].mean(0).tolist(),
          wrist_fk_orientation_mean_deg=np.mean(orientation,axis=0).tolist(),
          wrist_fk_position_mean_mm=np.mean(position,axis=0).tolist())
    (out/'wrist_comparison.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    print(json.dumps([summarize(Path(p).resolve()) for p in sys.argv[1:]],indent=2))
