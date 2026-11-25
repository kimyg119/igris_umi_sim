"""Frozen refinement snapshots and matched-source FiSTA comparisons.

The ongoing retarget batch is read only. Snapshot completion and source split
coverage remain explicit; a partial snapshot is never labelled the full corpus.
"""
import argparse,json,shutil
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from .common import OUTPUT,HERE,SDK_NAMES,sha256,save_json
from .train import train,load_split,evaluate
from .network import BimanualFiSTA,NetworkConfig

VARIANTS=('original','refined_original_time','refined_retimed')

def starts_for(clip,variant):
    if variant=='original':
        valid=clip['valid']&(clip['target_angle_error'][:,:,0].max(1)<=np.deg2rad(40))
    else:valid=clip['strict_valid']
    if len(valid)<6:return np.empty(0,np.int64)
    starts=np.flatnonzero(np.convolve(valid.astype(int),np.ones(6,int),'valid')==6)
    if variant=='refined_retimed' and len(starts):
        # Keep six consecutive 30 Hz observations, but do not oversample a
        # source phase merely because one clip was slowed down more than another.
        phase=np.floor(clip['source_time'][starts+5]*30+1e-5).astype(np.int64)
        _,indices=np.unique(phase,return_index=True);starts=starts[np.sort(indices)]
    return starts

def freeze(batch,out,require_complete=False):
    if (out/'snapshot.json').exists():raise FileExistsError(out/'snapshot.json')
    config=json.loads((batch/'config.json').read_text())
    for path,digest in config['dependencies'].items():
        if sha256(path)!=digest:raise ValueError(f'retarget dependency changed: {path}')
    raw=(batch/'results.jsonl').read_bytes()
    if not raw.endswith(b'\n'):raw=raw.rsplit(b'\n',1)[0]+b'\n'
    records=[json.loads(line) for line in raw.splitlines() if line]
    if require_complete and (len(records)!=config['count'] or any(r['status']!='processed' for r in records)):
        raise ValueError('requested complete refinement population is not ready')
    split_path=OUTPUT/'data/split_manifest.json';splits={r['path']:r for r in json.loads(split_path.read_text())['files']}
    entries=[]
    for row in records:
        if row['status']!='processed':continue
        comparison=Path(row['comparison']).resolve();meta=json.loads(comparison.read_text());source=splits[row['path']]
        if meta['implementation_sha256']!=config['implementation_sha256'] or meta['settings']!=config['settings']:
            raise ValueError(f'mixed refinement recipe: {comparison}')
        files={'original':OUTPUT/'data/clips'/row['path'],
               'refined_original_time':comparison.parent/'clip.npz','refined_retimed':comparison.parent/'retimed_clip.npz'}
        hashes={v:sha256(p) for v,p in files.items()}
        if hashes['original']!=meta['source_hashes'][str(files['original'])]:raise ValueError('original clip changed')
        entries.append(dict(path=row['path'],group=source['group'],split=source['split'],corpus=source['corpus'],
            files={k:str(v) for k,v in files.items()},sha256=hashes,time_scale=meta['time_scale'],
            comparison=str(comparison),comparison_sha256=sha256(comparison)))
    entries.sort(key=lambda r:r['path'])
    for split in ('train','val','test'):
        if not any(r['split']==split for r in entries):raise ValueError(f'no completed {split} source')
    snapshot=dict(batch=str(batch.resolve()),expected_population=config['count'],recorded_results=len(records),
        processed_sources=len(entries),complete=len(entries)==config['count'],
        status_counts=dict(Counter(r['status'] for r in records)),corpora=dict(Counter(r['corpus'] for r in entries)),
        source_split_counts=dict(Counter(r['split'] for r in entries)),
        split_manifest_sha256=sha256(split_path),batch_config_sha256=sha256(batch/'config.json'),batch_config=config,
        entries=entries,protocol='fixed completed-source snapshot; original source-ID split unchanged; no model/test-based clip selection')
    save_json(out/'snapshot.json',snapshot);print(json.dumps({k:v for k,v in snapshot.items() if k not in ('entries','batch_config')},indent=2))

def pack_one(out,snapshot,variant,split,destination):
    rows=[r for r in snapshot['entries'] if r['split']==split];arrays=[];begins=[];clips=[];cursor=0
    for row in rows:
        path=Path(row['files'][variant])
        if sha256(path)!=row['sha256'][variant]:raise ValueError(f'frozen clip changed: {path}')
        with np.load(path) as z:
            if tuple(z['joint_names'])!=SDK_NAMES[:29]:raise ValueError('joint-order mismatch')
            if not np.allclose(np.diff(z['time']),1/30,atol=1e-7):raise ValueError('network requires 30 Hz')
            s=z['state'];starts=starts_for(z,variant)
            if not np.isfinite(s).all():raise ValueError('nonfinite state')
            if not len(starts):continue
            arrays.append(s);begins.append(cursor+starts)
            clips.append(dict(path=row['path'],group=row['group'],corpus=row['corpus'],start=cursor,
                frames=len(s),windows=len(starts),variant=variant,file=str(path),sha256=row['sha256'][variant]))
            cursor+=len(s)
    if not arrays:raise ValueError(f'empty {variant}/{split}')
    destination.mkdir(parents=True,exist_ok=False)
    np.save(destination/'states.npy',np.concatenate(arrays));np.save(destination/'starts.npy',np.concatenate(begins))
    save_json(destination/'clips.json',clips)
    return dict(clips=len(clips),frames=cursor,windows=sum(c['windows'] for c in clips),
                groups=sorted({c['group'] for c in clips}),data_variant=variant)

def pack(out):
    snapshot=json.loads((out/'snapshot.json').read_text());summaries={}
    # Same improved held-out validation/test targets for all matched-source runs.
    # This permits comparing the original labels against corrected labels without
    # giving each model a different validation target. No test fitting occurs.
    common={s:pack_one(out,snapshot,'refined_retimed',s,out/'common'/s) for s in ('val','test')}
    for variant in VARIANTS:
        folder=out/'packed'/variant
        m={'train':pack_one(out,snapshot,variant,'train',folder/'train')}
        for split in ('val','test'):
            (folder/split).symlink_to(out.resolve()/'common'/split,target_is_directory=True);m[split]=common[split]
        groups=[set(m[s]['groups']) for s in ('train','val','test')]
        assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        m.update(scope=variant,snapshot_sha256=sha256(out/'snapshot.json'),adapter_sha256=sha256(__file__),
            normalization='training windows only',validation='same refined retimed strict validation for every matched run',
            quality='original: old 100mm/40deg gates; refined: strict 30mm/20deg and all saved audit gates',
            retimed_sampling='at most one valid output window per source 30Hz phase; window itself remains 6 consecutive output frames')
        save_json(folder/'manifest.json',m);summaries[variant]=m
    save_json(out/'packed/summary.json',summaries);print(json.dumps(summaries,indent=2))

def fit(out):
    for variant in VARIANTS:
        for seed in (42,123,789,1024):
            folder=out/'training'/variant/f'seed_{seed}'
            if (folder/'complete.json').exists():continue
            train(variant,seed,60,2048,'cuda',output=out)

def compare(out):
    torch.set_num_threads(2);rows=[]
    tests={'refined_snapshot':(out,'refined_retimed'),'original_full_test':(OUTPUT,'full')}
    variants=[(out,v,v) for v in VARIANTS]+[(OUTPUT,'paper','previous_paper'),(OUTPUT,'full','previous_full')]
    for test_name,(root,scope) in tests.items():
        states,starts=load_split(scope,'test','cuda',root)
        test_groups=set(json.loads((root/'packed'/scope/'manifest.json').read_text())['test']['groups'])
        for train_root,variant,label in variants:
            training=json.loads((train_root/'packed'/variant/'manifest.json').read_text())
            assert not set(training['train']['groups'])&test_groups
            for seed in (42,123,789,1024):
                path=train_root/'training'/variant/f'seed_{seed}/best.pt'
                saved=torch.load(path,map_location='cuda',weights_only=False)
                net=BimanualFiSTA(NetworkConfig(**saved['config']['network'])).cuda();net.load_state_dict(saved['network'])
                r=dict(test=test_name,model=label,seed=seed,checkpoint_sha256=sha256(path),
                    **evaluate(net,states,starts,saved['mean'].cuda(),saved['std'].cuda()))
                rows.append(r);print(json.dumps(r),flush=True)
        del states,starts
    save_json(out/'evaluation/prediction.json',dict(snapshot_sha256=sha256(out/'snapshot.json'),results=rows,
        protocol='all listed test windows; original source-ID groups excluded from every model train; teacher-forced history'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['freeze','pack','train','compare'])
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--batch',type=Path);ap.add_argument('--require-complete',action='store_true');a=ap.parse_args()
    if a.command=='freeze':freeze(a.batch,a.out,a.require_complete)
    elif a.command=='pack':pack(a.out)
    elif a.command=='train':fit(a.out)
    else:compare(a.out)
