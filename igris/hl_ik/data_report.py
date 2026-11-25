"""Account for every source file and every saved retarget frame."""
from collections import Counter,defaultdict
import numpy as np
from .common import *
from .splits import curated_manifest
from .train import result_rows

def summarize():
    manifest=curated_manifest();results=result_rows();corpora={};failures=[];pairs=Counter();errors=defaultdict(list)
    completed=[]
    for source in manifest:
        row=results.get(source['path'],source)
        if row['status']=='pending':raise RuntimeError('source remains unaccounted: '+source['path'])
        c=corpora.setdefault(source['corpus'],dict(files=0,status=Counter(),frames=0,accepted_frames=0,accepted_windows=0,
            rejection_counts=Counter(),splits=Counter()))
        c['files']+=1;c['status'][row['status']]+=1;c['splits'][source['split']]+=1
        if row['status']!='processed':
            failures.append(dict(path=row['path'],status=row['status'],reason=row.get('reason')));continue
        with np.load(row['output']) as f:
            valid=f['valid'].copy();extra=f['target_angle_error'][:,:,0].max(1)>np.deg2rad(40)
            c['rejection_counts']['elbow_orientation_additional']+=int((valid&extra).sum());valid&=~extra
            assert np.isfinite(f['qpos']).all() and np.isfinite(f['state']).all()
            assert not f['reject_flags'][valid].any()
            c['frames']+=len(valid);c['accepted_frames']+=int(valid.sum())
            windows=int((np.convolve(valid.astype(int),np.ones(6,dtype=int),'valid')==6).sum());c['accepted_windows']+=windows
            c['rejection_counts'].update(row['rejection_counts']);pairs.update(row['collision_pairs'])
            for name,key,scale in [('target_position_m','target_position_error',1.),('target_orientation_deg','target_angle_error',180/np.pi),('limb_direction_deg','arm_angle_error',180/np.pi)]:
                errors[name].append(f[key][valid]*scale)
            completed.append(dict(path=row['path'],group=source['group'],split=source['split'],frames=len(valid),accepted_frames=int(valid.sum()),windows=windows))
    distributions={}
    for name,values in errors.items():
        v=np.concatenate(values)
        distributions[name]=dict(mean=float(v.mean()),p50=float(np.percentile(v,50)),p95=float(np.percentile(v,95)),max=float(v.max()),
            layout='both arms; last axis elbow/wrist or upper/lower segment',per_point_mean=v.mean((0,1)).tolist())
    for c in corpora.values():
        for key in ('status','rejection_counts','splits'):c[key]=dict(c[key])
    result=dict(source_files=len(manifest),status_counts=dict(Counter(results.get(r['path'],r)['status'] for r in manifest)),
        total_frames=sum(c['frames'] for c in corpora.values()),accepted_frames=sum(c['accepted_frames'] for c in corpora.values()),
        accepted_windows=sum(c['accepted_windows'] for c in corpora.values()),corpora=corpora,exceptions=failures,
        accepted_frame_errors=distributions,rejected_collision_pairs=dict(pairs.most_common()),
        quality_scope='all recorded frames and all training windows; original GMR flags plus both-elbow 40 deg gate',
        limits=['sphere geometry and configured exclusions approximate self collision',
            'sampled corridors use max 0.02 rad steps, not continuous swept-volume proof',
            'leg retargets provide collision context; gait balance, foot-slip and ground-contact tracking are not training acceptance claims',
            'a filtered training set does not guarantee safe learned predictions; runtime collision and residual gates remain required'])
    save_json(OUTPUT/'evaluation/data_quality.json',result);save_json(OUTPUT/'evaluation/accepted_clips.json',completed)
    print(json.dumps({k:result[k] for k in ('source_files','status_counts','total_frames','accepted_frames','accepted_windows')},indent=2))

if __name__=='__main__':summarize()
