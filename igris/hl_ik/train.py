"""Pack non-overlapping subject splits; train paper-subset and full-data FiSTA."""
import argparse
import json
import os
import time
from dataclasses import asdict
from collections import Counter
import numpy as np
import torch
from .common import *
from .network import *
from .splits import curated_manifest

def result_rows():
    rows={}
    for line in (OUTPUT/'data/retarget_results.jsonl').read_text().splitlines():
        r=json.loads(line);rows[r['path']]=r
    for source in curated_manifest():
        if source['path'] in rows:
            rows[source['path']].update({k:source[k] for k in ('group','split','corpus','paper')})
    return rows

def pack(scope):
    manifest=curated_manifest();results=result_rows()
    selected=[r for r in manifest if scope=='full' or r['paper']]
    missing=[r['path'] for r in selected if r['status']=='pending' and r['path'] not in results]
    if missing:raise RuntimeError(f'{scope}: {len(missing)} files still pending; cannot silently train a partial population')
    out=OUTPUT/'packed'/scope;out.mkdir(parents=True,exist_ok=True)
    summary={}
    for split in ('train','val','test'):
        rows=[results[r['path']] for r in selected if r['split']==split and r['path'] in results
              and results[r['path']]['status']=='processed' and results[r['path']]['valid_windows']>0]
        rows.sort(key=lambda r:r['path'])
        if not rows:raise ValueError(f'empty {scope}/{split}')
        total=sum(r['frames'] for r in rows);windows=sum(r['valid_windows'] for r in rows)
        dest=out/split;dest.mkdir(exist_ok=True)
        state=np.lib.format.open_memmap(dest/'states.npy',mode='w+',dtype=np.float32,shape=(total,2,14))
        starts=[]
        cursor=idx=0;clips=[];elbow_rejected=0
        for r in rows:
            with np.load(r['output']) as f:
                s=f['state'];valid=f['valid'].copy();n=len(s)
                elbow_bad=f['target_angle_error'][:,:,0].max(1)>np.deg2rad(40)
                elbow_rejected+=int((valid&elbow_bad).sum());valid&=~elbow_bad
                begins=np.flatnonzero(np.convolve(valid.astype(int),np.ones(6,dtype=int),'valid')==6)
                state[cursor:cursor+n]=s;starts.append(cursor+begins)
                clips.append(dict(path=r['path'],group=r['group'],corpus=r['corpus'],start=cursor,frames=n,windows=len(begins)))
                cursor+=n;idx+=len(begins)
        assert cursor==total and idx<=windows
        state.flush();np.save(dest/'starts.npy',np.concatenate(starts))
        save_json(dest/'clips.json',clips)
        summary[split]=dict(clips=len(rows),frames=total,windows=idx,additional_elbow_orientation_rejected_frames=elbow_rejected,groups=sorted({r['group'] for r in rows}))
    assert not(set(summary['train']['groups'])&set(summary['val']['groups']) or set(summary['train']['groups'])&set(summary['test']['groups']))
    summary['source_manifest_sha256']=sha256(OUTPUT/'data/source_manifest.json')
    summary['split_code_sha256']=sha256(HERE/'splits.py')
    summary['scope']=scope;summary['additional_quality_gate']='elbow orientation <= 40 deg, both arms, all six window frames'
    save_json(out/'manifest.json',summary);print(json.dumps(summary,indent=2))

def load_split(scope,split,device,output=OUTPUT):
    p=output/'packed'/scope/split
    return torch.as_tensor(np.load(p/'states.npy'),device=device),torch.as_tensor(np.load(p/'starts.npy'),device=device)

def stats(states,starts):
    total=torch.zeros(2,14,device=states.device,dtype=torch.float64);square=total.clone();count=0
    # Exactly training window states, after the same hemisphere transform as inference.
    for ids in starts.split(16384):
        h=states[ids[:,None]+torch.arange(5,device=states.device)[None]];f=states[ids+5]
        h,g,y=canonicalize(h,f[:,:,:7],f[:,:,7:]);x=torch.cat((h,torch.cat((g,y),dim=-1)[:,None]),dim=1).double()
        total+=x.sum((0,1));square+=x.square().sum((0,1));count+=x.shape[0]*x.shape[1]
    mean=total/count;std=(square/count-mean.square()).clamp_min(.05**2).sqrt()
    return mean.float(),std.float()

@torch.no_grad()
def evaluate(net,states,starts,mean,std,batch=4096):
    net.eval();total_loss=0;pos=[];angle=[];persist=[];persist_angle=[]
    for ids in starts.split(batch):
        h,g,y,raw=batch_from_states(states,ids,mean,std);pred=net(h,g)
        total_loss+=float((pred-y).square().sum())
        p=decode(pred,mean,std)
        pos.append((p[:,:,:3]-raw[:,:,:3]).norm(dim=-1).cpu().numpy())
        a=raw[:,:,3:]/raw[:,:,3:].norm(dim=-1,keepdim=True)
        angle.append((2*torch.acos((p[:,:,3:]*a).sum(-1).abs().clamp(max=1))).cpu().numpy())
        last=states[ids+4,:,7:]
        persist.append((last[:,:,:3]-raw[:,:,:3]).norm(dim=-1).cpu().numpy())
        persist_angle.append((2*torch.acos((last[:,:,3:]*a).sum(-1).abs().clamp(max=1))).cpu().numpy())
    p,a=np.concatenate(pos),np.concatenate(angle);pp,pa=np.concatenate(persist),np.concatenate(persist_angle)
    return dict(normalized_mse=total_loss/(len(starts)*14),samples=len(starts),
        position_mean_m=float(p.mean()),position_p95_m=float(np.percentile(p,95)),position_rmse_m=float(np.sqrt((p*p).mean())),
        orientation_mean_rad=float(a.mean()),orientation_p95_rad=float(np.percentile(a,95)),
        persistence_position_mean_m=float(pp.mean()),persistence_orientation_mean_rad=float(pa.mean()))

def train(scope,seed,epochs,batch,device,output=OUTPUT):
    torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(2)
    config=NetworkConfig();net=BimanualFiSTA(config).to(device)
    train_s,train_i=load_split(scope,'train',device,output);val_s,val_i=load_split(scope,'val',device,output)
    mean,std=stats(train_s,train_i)
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=1e-5)
    out=output/'training'/scope/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'best.pt').exists():raise FileExistsError(f'checkpoint exists: {out}; use a new experiment path instead of overwriting')
    cfg=dict(scope=scope,seed=seed,epochs=epochs,batch=batch,optimizer='AdamW',lr=.001,weight_decay=.0001,
        schedule='cosine to 1e-5',early_stopping_patience=10,normalization='training windows only, per-arm standardization with std floor .05',
        loss='coordinate MSE on normalized xyz/wxyz; window hemisphere alignment; output quaternion normalized for pose metrics',
        network=asdict(config),parameter_count=sum(p.numel() for p in net.parameters()),device=device,
        paper='arXiv:2509.20263v3; unpublished hyperparameters are explicit IGRIS reconstruction choices',
        data_manifest_sha256=sha256(output/'packed'/scope/'manifest.json'))
    cfg['source_sha256']={name:sha256(HERE/name) for name in ('network.py','train.py')}
    save_json(out/'config.json',cfg)
    best=float('inf');wait=0;history=[];start=time.monotonic()
    for epoch in range(1,epochs+1):
        net.train();total=0.;permutation=train_i[torch.randperm(len(train_i),device=device)]
        for ids in permutation.split(batch):
            h,g,y,_=batch_from_states(train_s,ids,mean,std);opt.zero_grad(set_to_none=True)
            pred=net(h,g);loss=(pred-y).square().mean();loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step();total+=float(loss.detach())*len(ids)
        schedule.step();metrics=evaluate(net,val_s,val_i,mean,std)
        row=dict(epoch=epoch,train_mse=total/len(train_i),**metrics,elapsed_s=time.monotonic()-start)
        history.append(row);save_json(out/'history.json',history)
        improved=metrics['normalized_mse']<best-1e-6
        if improved:
            best=metrics['normalized_mse'];wait=0
            torch.save(dict(network=net.state_dict(),config=cfg,mean=mean.cpu(),std=std.cpu(),epoch=epoch,validation=metrics),out/'best.pt')
        else:wait+=1
        print(json.dumps(dict(scope=scope,seed=seed,**row)),flush=True)
        if wait>=10:break
    saved=torch.load(out/'best.pt',map_location=device,weights_only=False);net.load_state_dict(saved['network'])
    test_s,test_i=load_split(scope,'test',device,output);test=evaluate(net,test_s,test_i,mean,std)
    save_json(out/'test_own_split.json',test);save_json(out/'complete.json',dict(best_epoch=saved['epoch'],epochs_run=len(history),elapsed_s=time.monotonic()-start,test=test))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['pack','train']);ap.add_argument('--scope',choices=['paper','full'],required=True)
    ap.add_argument('--seeds',type=int,nargs='+',default=[42,123,789,1024]);ap.add_argument('--epochs',type=int,default=60)
    ap.add_argument('--batch',type=int,default=2048);ap.add_argument('--device',default='cuda');a=ap.parse_args()
    if a.command=='pack':pack(a.scope)
    else:
        for seed in a.seeds:train(a.scope,seed,a.epochs,a.batch,a.device)

if __name__=='__main__':main()
