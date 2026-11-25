"""Measured tracking, jitter, effort and cadence; never integrates replacement physics."""
import argparse
import json
from pathlib import Path
import struct

import numpy as np
from scipy.signal import butter,sosfiltfilt
from run import NAMES,OWNED

def rms(x,axis=0):return np.sqrt(np.mean(np.square(x),axis=axis))

def analyze(out):
    result=json.loads((out/'result.json').read_text())
    if not result['completed']:
        (out/'score.json').write_text(json.dumps(result,indent=2))
        return result
    z=np.load(out/'signals.npz');cmd=z['commands'];states=z['states']
    cmd=cmd[cmd[:,3]>0]
    with (out/'physics.bin').open('rb') as f:
        nq,nv,nu=struct.unpack('iii',f.read(12));width=2+nq+nv+2*nu+16
    data=np.memmap(out/'physics.bin',mode='r',dtype=np.float64,offset=12)
    assert len(data)%width==0
    data=np.asarray(data.reshape(-1,width))
    data=data[(data[:,0]>=cmd[0,0])&(data[:,0]<=cmd[-1,0])]
    # Drop no replay/hold segments. Uniform wall-time resampling is for spectral metrics only.
    assert np.all(np.diff(data[:,0])>0) and np.all(np.diff(data[:,1])>0)
    ci=np.searchsorted(cmd[:,0],data[:,0],side='right')-1
    target=cmd[ci,4:];moving=cmd[ci,3]==1
    hold_start=cmd[cmd[:,3]==2,0][0]+2.
    held=data[:,0]>=hold_start
    jointmap={};actuators=[]
    for line in (out/'physics.bin.map').read_text().splitlines():
        v=line.split('\t')
        if v[0]=='joint':jointmap[v[1]]=(int(v[2]),int(v[3]))
        elif v[0]=='actuator':actuators.append((v[1],int(v[2]),float(v[3]),float(v[4])))
    joint_q=np.array([data[:,2+jointmap[n][0]] for n in np.array(NAMES)[OWNED]]).T
    backlash=np.array([data[:,2+jointmap[n+'_backlash'][0]] for n in np.array(NAMES)[OWNED]]).T
    # Native output serial hinges and their collinear passive backlash hinges.
    physical_q=joint_q+backlash
    motor_error=np.rad2deg(joint_q-target);physical_error=np.rad2deg(physical_q-target)
    times=data[:,0];grid=np.arange(times[0],times[-1],1/300.)
    errors=np.column_stack([np.interp(grid,times,physical_error[:,j]) for j in range(17)])
    native=np.column_stack([np.interp(grid,times,motor_error[:,j]) for j in range(17)])
    band=butter(4,[8,45],btype='bandpass',fs=300,output='sos')
    hf=sosfiltfilt(band,errors,axis=0);motor_hf=sosfiltfilt(band,native,axis=0)
    trimmed=(grid>grid[0]+1)&(grid<grid[-1]-1)
    hf_hold=(grid>hold_start+1)&(grid<grid[-1]-1)
    # SDK uses inferred PJS angles for parallel mechanisms; retain independently.
    states=states[(states[:,0]>=cmd[0,0])&(states[:,0]<=cmd[-1,0])]
    si=np.searchsorted(cmd[:,0],states[:,0],side='right')-1
    sdk_error=np.rad2deg(states[:,2+OWNED]-cmd[si,4:])
    sdk_move=cmd[si,3]==1
    force=data[:,2+nq+nv+nu:2+nq+nv+2*nu]
    ctrl=data[:,2+nq+nv:2+nq+nv+nu]
    efforts={}
    for n,a,lo,hi in actuators:
        efforts[n]=dict(max_abs_force=float(np.max(np.abs(force[:,a]))),rms_force=float(rms(force[:,a])),ctrl_limit=[lo,hi],
            saturation_fraction=float(np.mean((ctrl[:,a]<=lo+1e-5)|(ctrl[:,a]>=hi-1e-5))) if lo<hi else 0.)
    per_joint={}
    for j,i in enumerate(OWNED):
        per_joint[NAMES[i]]=dict(kp=float(json.loads((out/'trial.json').read_text())['gains'][j][0]),
            kd=float(json.loads((out/'trial.json').read_text())['gains'][j][1]),
            sdk_replay_rms_deg=float(rms(sdk_error[sdk_move,j])),
            native_replay_rms_deg=float(rms(motor_error[moving,j])),
            output_replay_rms_deg=float(rms(physical_error[moving,j])),
            output_replay_p95_deg=float(np.percentile(np.abs(physical_error[moving,j]),95)),
            native_jitter_8_45hz_rms_deg=float(rms(motor_hf[trimmed,j])),
            output_jitter_8_45hz_rms_deg=float(rms(hf[trimmed,j])),
            output_hold_jitter_8_45hz_rms_deg=float(rms(hf[hf_hold,j])),
            output_hold_mean_error_deg=float(np.mean(physical_error[held,j])),
            output_hold_std_deg=float(np.std(physical_error[held,j])),
            output_hold_p99_span_deg=float(np.percentile(physical_error[held,j],99.5)-np.percentile(physical_error[held,j],.5)),
            backlash_abs_p95_deg=float(np.percentile(np.abs(np.rad2deg(backlash[:,j])),95)))
    quat=data[:,5:9];tilt=np.rad2deg(np.arccos(np.clip(1-2*(quat[:,1]**2+quat[:,2]**2),-1,1)))
    cadence=np.diff(cmd[:,0]);native_dt=np.diff(data[:,0]);late=cmd[:,0]-cmd[:,1]
    result.update(physics_samples=len(data),command_count=len(cmd),
        command_hz=float((len(cmd)-1)/(cmd[-1,0]-cmd[0,0])),
        command_interval_ms=dict(p95=float(np.percentile(cadence,95)*1000),p99=float(np.percentile(cadence,99)*1000),max=float(max(cadence)*1000)),
        publisher_lateness_ms=dict(p99=float(np.percentile(late,99)*1000),max=float(max(late)*1000)),
        physics_rtf=float((data[-1,1]-data[0,1])/(times[-1]-times[0])),
        physics_observed_hz=float((len(data)-1)/(times[-1]-times[0])),
        physics_interval_ms=dict(p99=float(np.percentile(native_dt,99)*1000),max=float(max(native_dt)*1000)),
        base_tilt_max_deg=float(max(tilt)),base_z_min_m=float(min(data[:,4])),self_contact_samples=int(np.sum(data[:,-2]>0)),
        per_joint=per_joint,efforts=efforts)
    for group,js in [('waist',np.arange(3)),('arms',np.arange(3,17)),('wrist',np.array([7,8,9,14,15,16])),('shoulder_elbow',np.array([3,4,5,6,10,11,12,13]))]:
        result[group]=dict(native_tracking_rms_deg=float(rms(motor_error[moving][:,js],axis=None)),
            output_tracking_rms_deg=float(rms(physical_error[moving][:,js],axis=None)),
            sdk_tracking_rms_deg=float(rms(sdk_error[sdk_move][:,js],axis=None)),
            jitter_8_45hz_rms_deg=float(rms(hf[trimmed][:,js],axis=None)),
            hold_jitter_8_45hz_rms_deg=float(rms(hf[hf_hold][:,js],axis=None)),
            max_hold_std_deg=float(np.max(np.std(physical_error[held][:,js],axis=0))))
    result['timing_valid']=bool(.97<=result['physics_rtf']<=1.03 and abs(result['command_hz']-100)<.5 and result['physics_interval_ms']['max']<50)
    (out/'score.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(out/'analysis.npz',t=times-times[0],phase=cmd[ci,3],target_q=target,
        native_q=joint_q,output_q=physical_q,grid_t=grid-times[0],jitter_deg=hf,base_tilt_deg=tilt,ee_pose=data[:,-16:-2].reshape(-1,2,7))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('trial',type=Path);args=parser.parse_args()
    result=analyze(args.trial)
    print(json.dumps({k:result[k] for k in ('profile','mass_kg','completed','error','timing_valid','physics_rtf','command_hz','base_tilt_max_deg','arms','wrist') if k in result},indent=2))
