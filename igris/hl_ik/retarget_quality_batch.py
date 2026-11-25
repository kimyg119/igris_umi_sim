"""Accounted paper/full experiments with one frozen refinement configuration."""
import argparse,contextlib,json,multiprocessing as mp,time,traceback,sys
from importlib.metadata import version
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from .common import OUTPUT,HERE,sha256,save_json
from .retarget_quality import Refiner,Settings,run


def initialize(settings,dependencies,reuse):
    global worker_settings,worker_refiner,worker_dependencies,worker_reuse
    worker_settings=settings;worker_dependencies=dependencies;worker_reuse=reuse
    for path,digest in dependencies.items():
        if sha256(path)!=digest:raise RuntimeError(f'input changed before worker initialization: {path}')
    worker_refiner=Refiner(settings)


def process(row,out):
    for path,digest in worker_dependencies.items():
        if sha256(path)!=digest:raise RuntimeError(f'frozen experiment input changed: {path}')
    if row['path'] in worker_reuse:
        old=worker_reuse[row['path']];source=OUTPUT/'data/clips'/row['path']
        if sha256(source)==old['result']['source_hashes'][str(source)]:return dict(old,reused_from_paper=True)
    target=out/'clips'/Path(row['path']).with_suffix('')
    if all((target/name).exists() for name in ('comparison.json','clip.npz','retimed_clip.npz')):
        result=json.loads((target/'comparison.json').read_text())
        if result['implementation_sha256']!=sha256(HERE/'retarget_quality.py') or result['settings']!=asdict(worker_settings):
            raise RuntimeError(f'existing result uses a different implementation: {target}')
        source=OUTPUT/'data/clips'/row['path']
        if sha256(source)!=result['source_hashes'][str(source)]:raise RuntimeError(f'source changed since saved result: {source}')
        return dict(status='processed',path=row['path'],comparison=str(target/'comparison.json'),result=result,reused=True)
    if target.exists():
        # Preserve interrupted evidence as a distinct attempt.
        i=1
        while target.with_name(target.name+f'.attempt{i}').exists():i+=1
        target.rename(target.with_name(target.name+f'.attempt{i}'))
    log=out/'logs'/Path(row['path']).with_suffix('.log');log.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    try:
        with log.open('w') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
            result=run(row['path'],target,worker_settings,worker_refiner,row)
        return dict(status='processed',path=row['path'],comparison=str(target/'comparison.json'),result=result,elapsed_s=time.monotonic()-start)
    except Exception as e:
        return dict(status='failed',path=row['path'],error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc(),elapsed_s=time.monotonic()-start)


def summary(rows,config):
    good=[x for x in rows if x['status']=='processed'];totals={}
    for stage in ('baseline','refined','retimed'):
        weighted={};rejections=Counter();frames=0;source_frames=0
        for item in good:
            r=item['result'];m=r[stage];weight=r['baseline']['frames'];source_frames+=weight;frames+=m['frames']
            for key in ('valid_fraction','position_mean_m','orientation_mean_deg','direction_mean_deg','strict_30mm_20deg_fraction',
                        'wrist_position_mean_m','elbow_position_mean_m','wrist_orientation_mean_deg','elbow_orientation_mean_deg'):
                if key in m:weighted[key]=weighted.get(key,0.)+weight*m[key]
            for key,value in m['rejection_counts'].items():rejections[key]+=weight*value/m['frames']
        totals[stage]=dict(output_frames=frames,source_frames=source_frames,
            **{k:v/source_frames for k,v in weighted.items()} if source_frames else {})
        totals[stage]['rejection_fraction']={k:v/source_frames for k,v in rejections.items()} if source_frames else {}
    checks={};a=totals['baseline'];b=totals['retimed']
    if good:
        for k in ('self_collision','corridor_collision','velocity','acceleration'):
            checks[k]=b['rejection_fraction'][k]<=a['rejection_fraction'][k]
        for k in ('position_mean_m','direction_mean_deg','orientation_mean_deg'):
            checks[k]=b[k]<a[k]
        checks['valid_fraction']=b['valid_fraction']>a['valid_fraction']
    failures=[{k:v for k,v in x.items() if k!='traceback'} for x in rows if x['status']=='failed']
    scales=[x['result']['time_scale'] for x in good]
    regressions=[dict(path=x['path'],metrics=[k for k in ('valid_fraction','position_mean_m','orientation_mean_deg','direction_mean_deg')
        if (x['result']['retimed'][k]<x['result']['baseline'][k] if k=='valid_fraction' else
            x['result']['retimed'][k]>x['result']['baseline'][k])]) for x in good]
    return dict(config=config,status_counts=dict(Counter(r['status'] for r in rows)),completed=len(rows)==config['count'],
        stages=totals,full_application_checks=checks,
        ready_for_full=len(rows)==config['count'] and not config['truncated'] and config['scope']=='paper' and config['split']=='all' and not failures and bool(checks) and all(checks.values()),
        failures=failures,regressed_clips=[r for r in regressions if r['metrics']],
        clips_without_valid_six_frame_windows=[x['path'] for x in good if x['result']['retimed']['valid_windows']==0],
        clips_without_strict_six_frame_windows=[x['path'] for x in good if x['result']['retimed'].get('strict_valid_windows',0)==0],
        time_scale=dict(zip(('minimum','median','p95','maximum'),np.percentile(scales,[0,50,95,100]).tolist())) if scales else {},
        aggregation='equal source-time weighting; retimed output density does not increase a clip weight')


def batch(scope,split,out,jobs,paper_evidence=None,limit=None):
    settings=Settings(enforce_timing=False,velocity_cost=.08,acceleration_cost=.01,collision_constraints_per_pair=1,
                      trust_region=.05,collision_clearance=.006)
    implementation=sha256(HERE/'retarget_quality.py')
    runtime=dict(python=sys.version,packages={name:version(name) for name in ('numpy','scipy','mujoco','mink','osqp','daqp')})
    igris=HERE.parent
    dependencies={str(p):sha256(p) for p in [HERE/'retarget_quality_batch.py',HERE/'retarget_quality.py',HERE/'retarget.py',HERE/'geometry.py',HERE/'human.py',HERE/'common.py',
        OUTPUT/'model/igris_c_rcv_hand_torso.xml',OUTPUT/'model/gmr_config.json',
        igris/'igris_curobo/src/igris_ik/data/vendor/joint_limits.json',igris/'igris_curobo/src/igris_ik/data/curobo/igris_c_rcv_hand_pelvis.yml']}
    reuse={}
    if scope=='full' and paper_evidence is None:raise ValueError('full experiment requires completed paper-subset evidence')
    if paper_evidence is not None:
        evidence=json.loads(paper_evidence.read_text());previous=evidence['config']
        if scope=='full' and (not evidence['ready_for_full'] or previous['scope']!='paper' or previous['split']!='all'):
            raise ValueError('complete paper-subset quality gate has not passed')
        if previous['implementation_sha256']!=implementation or previous['settings']!=asdict(settings):raise ValueError('full configuration differs from paper-test configuration')
        if previous.get('dependencies')!=dependencies:raise ValueError('full experiment dependencies differ from paper test')
        if previous.get('runtime')!=runtime:raise ValueError('runtime packages differ from paper experiment')
        reuse={r['path']:r for r in map(json.loads,(paper_evidence.parent/'results.jsonl').read_text().splitlines()) if r['status']=='processed'}
    source=json.loads((OUTPUT/'data/source_manifest.json').read_text())['files']
    results={r['path']:r for r in map(json.loads,(OUTPUT/'data/retarget_results.jsonl').read_text().splitlines())}
    selected=[r for r in source if (scope=='full' or r['paper']) and (split=='all' or r['split']==split)]
    missing=[r['path'] for r in selected if r['status']=='pending' and r['path'] not in results]
    if missing:raise RuntimeError(f'{len(missing)} initial GMR files are not complete')
    rows=[results[r['path']] for r in selected if r['path'] in results and results[r['path']]['status']=='processed']
    rows.sort(key=lambda r:r['path'])
    population_count=len(rows)
    if limit:rows=rows[:limit]
    config=dict(scope=scope,split=split,count=len(rows),settings=asdict(settings),implementation_sha256=implementation,
        population_count=population_count,truncated=len(rows)!=population_count,
        dependencies=dependencies,runtime=runtime,
        source_manifest_sha256=sha256(OUTPUT/'data/source_manifest.json'),paths=[r['path'] for r in rows],
        input_status_counts=dict(Counter(results.get(r['path'],r)['status'] for r in selected)),
        excluded_inputs=[dict(path=r['path'],status=results.get(r['path'],r)['status'],reason=results.get(r['path'],r).get('reason'))
                         for r in selected if results.get(r['path'],r)['status']!='processed'],
        jobs=jobs,paper_evidence=str(paper_evidence) if paper_evidence else None,
        note='retarget quality experiment; no learning or physical execution; all frames and corridors of every selected clip are audited')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config:raise ValueError('existing experiment configuration differs')
    save_json(out/'config.json',config)
    complete=[];start=time.monotonic();last_summary=start
    with ProcessPoolExecutor(max_workers=jobs,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(settings,dependencies,reuse)) as pool:
        pending={pool.submit(process,row,out):row['path'] for row in rows}
        with (out/'results.jsonl').open('w',buffering=1) as log:
            for future in as_completed(pending):
                try:r=future.result()
                except Exception as e:r=dict(status='failed',path=pending[future],error=str(e))
                complete.append(r);log.write(json.dumps(r,allow_nan=False)+'\n')
                if len(complete)%25==0 or time.monotonic()-last_summary>30:
                    save_json(out/'summary.json',summary(complete,config));last_summary=time.monotonic()
                print(json.dumps(dict(done=len(complete),total=len(rows),path=r['path'],status=r['status'],elapsed_s=time.monotonic()-start,error=r.get('error'))),flush=True)
    for path,digest in dependencies.items():
        if sha256(path)!=digest:raise RuntimeError(f'input changed during experiment; results cannot authorize full application: {path}')
    final=summary(complete,config);save_json(out/'summary.json',final);return final


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scope',choices=['paper','full','staged'],required=True);p.add_argument('--split',choices=['all','train','val','test'],default='all')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--jobs',type=int,default=6);p.add_argument('--paper-evidence',type=Path);p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.scope=='staged':
        if a.split!='all' or a.limit is not None or a.paper_evidence is not None:p.error('staged runs require both complete populations without external evidence')
        batch('paper','all',a.out/'paper',a.jobs)
        batch('full','all',a.out/'full',a.jobs,a.out/'paper/summary.json')
    else:batch(a.scope,a.split,a.out,a.jobs,a.paper_evidence,a.limit)
