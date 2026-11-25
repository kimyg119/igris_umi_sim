"""Common subject-held-out prediction tests and generated-history IK rollouts."""
import argparse,json,time
from collections import Counter
import numpy as np
import torch
from .common import *
from .train import load_split,evaluate as prediction_metrics,result_rows
from .network import BimanualFiSTA,NetworkConfig
from .solver import HLIK,ElbowPrior

def common_test():
    torch.set_num_threads(2)
    states,starts=load_split('full','test','cuda');results=[]
    for scope in ('paper','full'):
        train_groups=set(json.loads((OUTPUT/'packed'/scope/'manifest.json').read_text())['train']['groups'])
        test_groups=set(json.loads((OUTPUT/'packed/full/manifest.json').read_text())['test']['groups'])
        assert not train_groups&test_groups
        for seed in (42,123,789,1024):
            path=OUTPUT/'training'/scope/f'seed_{seed}/best.pt'
            saved=torch.load(path,map_location='cuda',weights_only=False)
            net=BimanualFiSTA(NetworkConfig(**saved['config']['network'])).cuda();net.load_state_dict(saved['network'])
            result=dict(scope=scope,seed=seed,checkpoint_sha256=sha256(path),**prediction_metrics(net,states,starts,saved['mean'].cuda(),saved['std'].cuda()))
            save_json(path.parent/'test_common.json',result);results.append(result);print(json.dumps(result),flush=True)
    save_json(OUTPUT/'evaluation/common_prediction.json',dict(test_manifest_sha256=sha256(OUTPUT/'packed/full/manifest.json'),
        protocol='all valid full-data subject-held-out test windows, teacher-forced history; neither model trained on test subjects',results=results))

def fixtures():
    """One deterministic held-out clip per corpus; choose before loading models."""
    if not (OUTPUT/'packed/full/manifest.json').exists():raise RuntimeError('complete full-data pack before choosing final rollout fixtures')
    rows=sorted([r for r in result_rows().values() if r['status']=='processed' and r['split']=='test' and r['valid_windows']>=120],key=lambda r:r['path'])
    found={}
    for r in rows:
        if r['corpus'] in found:continue
        with np.load(r['output']) as f:
            valid=f['valid']&(f['target_angle_error'][:,:,0].max(1)<=np.deg2rad(40))
            if len(valid)<125:continue
            indices=np.flatnonzero(np.convolve(valid.astype(int),np.ones(125,dtype=int),'valid')==125)
            if len(indices):found[r['corpus']]=dict(path=r['path'],output=r['output'],group=r['group'],start=int(indices[0]),steps=120)
    save_json(OUTPUT/'evaluation/rollout_fixtures.json',dict(selection='first lexicographic subject-held-out clip per corpus with 125 consecutive QC-passing frames; earliest such interval; 5 initial observations then 120 predicted steps',clips=list(found.values())))

def rollouts(scope,checkpoint_path=None,output=OUTPUT):
    fixtures_path=OUTPUT/'evaluation/rollout_fixtures.json'
    if not fixtures_path.exists():fixtures()
    selected=json.loads(fixtures_path.read_text())['clips'];rows=[]
    checkpoint=checkpoint_path or (None if scope=='numerical' else OUTPUT/'training'/scope/'seed_42/best.pt')
    prior=ElbowPrior(checkpoint) if checkpoint else None
    out=output/'evaluation/rollouts'/scope;out.mkdir(parents=True,exist_ok=True)
    for clip in selected:
        with np.load(clip['output']) as f:
            start=clip['start'];names=f['joint_names'].tolist();qsource=f['qpos'][start:start+125]
            qs=np.zeros((125,31))
            for j,name in enumerate(SDK_NAMES[:29]):qs[:,j]=qsource[:,7+names.index(name)]
            states=f['state'][start:start+125]
        solver=HLIK(prior);solver.reset(qs[0])
        for q in qs[1:5]:solver.observe(q)
        current=qs[4].copy();records=[];commands=[]
        for i in range(5,125):
            current[:15]=qs[i,:15];goals=states[i,:,:7].copy();goals[:,:3]+=solver.shoulder
            result=solver.solve(goals,current,record_history=False)
            if result.reason!='collision':current=result.q
            solver.observe(current)
            actual=solver.states();elbow_error=np.linalg.norm(actual[:,7:10]-states[i,:,7:10],axis=1)
            orientation=(Rotation.from_quat(actual[:,10:14],scalar_first=True).inv()*Rotation.from_quat(states[i,:,10:14],scalar_first=True)).magnitude()
            # Score the retained state, including a collision hold, against the requested goal.
            ee_position=np.linalg.norm(actual[:,:3]-states[i,:,:3],axis=1).max()
            ee_orientation=(Rotation.from_quat(actual[:,3:7],scalar_first=True).inv()*Rotation.from_quat(states[i,:,3:7],scalar_first=True)).magnitude().max()
            records.append([result.success,ee_position,ee_orientation,result.elapsed_ms,*elbow_error,*orientation,result.reason=='collision'])
            commands.append(current.copy())
        v=np.asarray(records);key=clip['path'].split('/')[0]
        np.savez_compressed(out/(key+'.npz'),q=commands,metrics=v,source_q=qs,states=states)
        row=dict(scope=scope,**clip,success_fraction=float(v[:,0].mean()),ee_position_mean_m=float(v[:,1].mean()),ee_position_max_m=float(v[:,1].max()),
            ee_orientation_mean_deg=float(np.rad2deg(v[:,2]).mean()),elbow_position_mean_m=float(v[:,4:6].mean()),elbow_orientation_mean_deg=float(np.rad2deg(v[:,6:8]).mean()),
            collision_holds=int(v[:,8].sum()),solver_ms_median=float(np.median(v[:,3])),solver_ms_p95=float(np.percentile(v[:,3],95)))
        rows.append(row);print(json.dumps(row),flush=True)
    save_json(out/'metrics.json',dict(scope=scope,history='own generated FK at 30 Hz after 5 ground-truth warmup frames; teacher-forced lower body; collision failure holds arms; errors score retained q; all residual failures counted',
        fixture_sha256=sha256(fixtures_path),checkpoint_sha256=sha256(checkpoint) if checkpoint else None,clips=rows))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['common','fixtures','rollouts']);ap.add_argument('--scope')
    ap.add_argument('--checkpoint',type=Path);ap.add_argument('--output',type=Path,default=OUTPUT);a=ap.parse_args()
    if a.command=='common':common_test()
    elif a.command=='fixtures':fixtures()
    else:rollouts(a.scope,a.checkpoint,a.output)
