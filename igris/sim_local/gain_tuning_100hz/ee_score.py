"""Same-base, same-time FK against commanded q, using native MuJoCo 3.2.7."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.signal import butter,sosfiltfilt,welch
from run import HERE,container_path
from score import rms

def analyze(out):
    score=json.loads((out/'score.json').read_text())
    if not score.get('completed'):return
    z=np.load(out/'signals.npz');c=z['commands'];c=c[c[:,3]>0]
    np.column_stack([c[:,0],c[:,4:]]).astype('f8').tofile(out/'commands.bin')
    subprocess.run(['docker','exec','igris-sim',container_path(HERE/'fk'),container_path(out)],check=True)
    x=np.fromfile(out/'ee.bin').reshape(-1,30);ci=np.searchsorted(c[:,0],x[:,0],side='right')-1
    actual=x[:,2:16].reshape(-1,2,7);target=x[:,16:30].reshape(-1,2,7)
    delta=(actual[:,:,:3]-target[:,:,:3])*1000
    position=np.linalg.norm(delta,axis=-1)
    angle=np.rad2deg(2*np.arccos(np.clip(np.abs(np.sum(actual[:,:,3:]*target[:,:,3:],axis=-1)),0,1)))
    moving=c[ci,3]==1;hold=x[:,0]>=c[c[:,3]==2,0][0]+2
    grid=np.arange(x[0,0],x[-1,0],1/300)
    uniform=np.column_stack([np.interp(grid,x[:,0],delta.reshape(-1,6)[:,j]) for j in range(6)])
    filtered=sosfiltfilt(butter(4,[8,45],btype='bandpass',fs=300,output='sos'),uniform,axis=0).reshape(-1,2,3)
    mask=(grid>c[c[:,3]==2,0][0]+3)&(grid<grid[-1]-1)
    ee={}
    for j,side in enumerate(['left','right']):
        ee[side]=dict(replay_position_mean_mm=float(position[moving,j].mean()),replay_position_p95_mm=float(np.percentile(position[moving,j],95)),
            replay_orientation_rms_deg=float(rms(angle[moving,j])),replay_orientation_p95_deg=float(np.percentile(angle[moving,j],95)),
            hold_position_mean_mm=float(position[hold,j].mean()),hold_position_std_mm=float(rms(np.std(delta[hold,j],axis=0),axis=None)*np.sqrt(3)),
            hold_jitter_8_45hz_vector_rms_mm=float(rms(np.linalg.norm(filtered[mask,j],axis=-1))))
    a=np.load(out/'analysis.npz');qerr=np.rad2deg(a['output_q']-a['target_q'])
    u=np.column_stack([np.interp(grid-x[0,0],a['t'],qerr[:,j]) for j in range(17)])
    high=sosfiltfilt(butter(4,[45,120],btype='bandpass',fs=300,output='sos'),u,axis=0)
    freq,power=welch(u[mask],fs=300,nperseg=min(1024,int(mask.sum())),axis=0)
    band=(freq>=8)&(freq<=120)
    for j,(name,entry) in enumerate(score['per_joint'].items()):
        entry['hold_jitter_45_120hz_rms_deg']=float(rms(high[mask,j]))
        entry['hold_peak_frequency_8_120hz']=float(freq[band][np.argmax(power[band,j])])
    score['ee']=ee
    (out/'score.json').write_text(json.dumps(score,indent=2))
    np.savez_compressed(out/'ee_analysis.npz',t=x[:,0]-x[0,0],position_error_mm=position,orientation_error_deg=angle,actual_pose=actual,target_pose=target)
    print(out.name,json.dumps(ee),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('trial',type=Path);a=p.parse_args();analyze(a.trial.resolve())
