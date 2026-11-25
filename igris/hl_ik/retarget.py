"""GMR AMASS conversion with complete per-file and per-frame quality accounting."""
import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .common import *
from .human import JOINT_NAMES,load_clip
from .geometry import SphereGeometry,coupled_halfspaces

PAPER={'ACCAD','CMU','SFU'}
TEST_CORPUS={'BMLhandball'}
REASONS=['nonfinite','joint_limit','coupled_limit','self_collision','target_position',
         'target_orientation','arm_direction','velocity','acceleration','corridor_collision']

def header(z,key):
    with z.open(key+'.npy') as f:
        version=np.lib.format.read_magic(f)
        return np.lib.format._read_array_header(f,version)[0]

def make_manifest(source):
    rows=[]
    for p in sorted(source.rglob('*.npz')):
        rel=p.relative_to(source);corpus=rel.parts[0]
        # Group all takes from one subject. Both arms always remain in same split.
        subject=rel.parts[1]
        if corpus=='ACCAD':
            match=re.match(r'(Female\d+|Male\d+)',subject)
            if match:subject=match.group(0)
        if corpus=='DanceDB':subject=re.sub(r'^\d+_','',subject)
        group=corpus+'/'+subject
        split='train'
        row=dict(path=str(rel),corpus=corpus,group=group,split=split,paper=corpus in PAPER,bytes=p.stat().st_size)
        try:
            with zipfile.ZipFile(p) as z:
                keys=set(z.namelist())
                if p.name.endswith('_stagei.npz'):row.update(status='excluded',reason='stage_i_shape_fit_not_motion')
                elif not all(k+'.npy' in keys for k in ('pose_body','root_orient','trans','betas','gender','mocap_frame_rate')):
                    row.update(status='excluded',reason='unsupported_non_smplx_schema')
                else:row.update(status='pending',source_frames=int(header(z,'pose_body')[0]))
        except Exception as e:row.update(status='failed',reason=f'{type(e).__name__}: {e}')
        rows.append(row)
    # Stratify subject groups per corpus, so validation is present even in SFU.
    splits={}
    for corpus in sorted({r['corpus'] for r in rows}):
        groups=sorted({r['group'] for r in rows if r['corpus']==corpus},key=lambda g:hashlib.sha256(g.encode()).hexdigest())
        hold=max(1,round(len(groups)*.1)) if len(groups)>=3 else 0
        for i,g in enumerate(groups):splits[g]='test' if corpus in TEST_CORPUS or i<hold else 'val' if i<2*hold else 'train'
    for r in rows:r['split']=splits[r['group']]
    save_json(OUTPUT/'data/source_manifest.json',dict(source=str(source),files=rows,count=len(rows),
        split='stratified subject groups 80/10/10, common test for both models; BMLhandball absent locally',paper_corpora=sorted(PAPER)))
    print('MANIFEST',len(rows),dict(Counter(r['status'] for r in rows)),dict(Counter(r['corpus'] for r in rows)),flush=True)
    return rows

def initialize():
    global retargeter,geometry,A,b,model,names,default_scale,reference,reference_pos,limb_lengths
    sys.path.insert(0,str(GMR))
    import mink
    from mink.limits import Constraint
    with contextlib.redirect_stdout(io.StringIO()):
        import general_motion_retargeting as gmr
        gmr.ROBOT_XML_DICT['igris_hlik_torso']=OUTPUT/'model/igris_c_rcv_hand_torso.xml'
        gmr.IK_CONFIG_DICT['smplx']['igris_hlik_torso']=OUTPUT/'model/gmr_config.json'
        retargeter=gmr.GeneralMotionRetargeting('smplx','igris_hlik_torso',verbose=False)
    model=retargeter.model
    names=[model.joint(i).name for i in range(1,model.njnt)]
    A,b=coupled_halfspaces(names)
    class CoupledLimit(mink.Limit):
        def compute_qp_inequalities(self,configuration,dt):
            # Mink QP variable is delta q, not velocity (solve_ik divides by dt).
            return Constraint(G=np.pad(A,((0,0),(6,0))),h=.95*(b-A@configuration.q[7:]))
    retargeter.ik_limits.append(CoupledLimit())
    default_scale=dict(retargeter.human_scale_table)
    geometry=SphereGeometry(model)
    reference=mujoco.MjData(model);mujoco.mj_kinematics(model,reference)
    reference_pos={model.body(i).name:reference.xpos[i].copy() for i in range(1,model.nbody)}
    limb_lengths={}
    for s in ('l','r'):
        for part,chain in [('arm',(f'{s}_shoulder_frame',f'{s}_elbow_bracket',f'{s}_wrist_connector')),
                           ('leg',(f'{s}_hip_bracket',f'{s}_lower_leg',f'{s}_foot_original'))]:
            limb_lengths[(s,part)]=np.linalg.norm(np.diff([reference_pos[k] for k in chain],axis=0),axis=1)

def morphology_targets(human):
    """GMR targets with robot segment lengths; preserve human limb directions.

    Existing global scaling alone misplaces shoulders by ~20 cm for this torso
    topology. Construct shoulder anchors and each limb using actual URDF lengths.
    This robot adaptation replaces the paper's unavailable fitted beta retargeter.
    """
    import mink
    positions={k:np.asarray(v[0]) for k,v in human.items()}
    rotations={k:Rotation.from_quat(human[k][1],scalar_first=True)*retargeter.rot_offsets1[k]
               for k in retargeter.rot_offsets1}
    pelvis=positions['pelvis']*retargeter.human_scale_table['pelvis']
    rt=rotations['spine3'];rp=rotations['pelvis']
    waist=(rt.inv()*rp).as_euler('ZXY')
    reference.qpos[:]=model.qpos0
    for n,v in zip(('waist_yaw','waist_roll','waist_pitch'),waist):
        j=model.joint(n).id;reference.qpos[model.jnt_qposadr[j]]=np.clip(v,*model.jnt_range[j])
    mujoco.mj_kinematics(model,reference)
    torso=pelvis-rt.apply(reference.xpos[model.body('pelvis').id])
    target={'pelvis':pelvis,'spine3':torso,'head':torso+rt.apply(reference_pos['head'])}
    for side,hs in [('l','left'),('r','right')]:
        target[hs+'_shoulder']=torso+rt.apply(reference_pos[side+'_shoulder_frame'])
        hip_local=reference_pos[side+'_hip_bracket']-reference_pos['pelvis']
        target[hs+'_hip']=pelvis+rp.apply(hip_local)
        for part,joints in [('arm',('shoulder','elbow','wrist')),('leg',('hip','knee','ankle'))]:
            lengths=limb_lengths[(side,part)]
            for k in range(2):
                a,c=hs+'_'+joints[k],hs+'_'+joints[k+1]
                direction=positions[c]-positions[a]
                target[c]=target[a]+direction/max(np.linalg.norm(direction),1e-9)*lengths[k]
    scaled={k:[target[k],v.as_quat(scalar_first=True)] for k,v in rotations.items()}
    retargeter.scaled_human_data=scaled
    for table in (retargeter.human_body_to_task1,retargeter.human_body_to_task2):
        for name,task in table.items():
            p,q=scaled[name];task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(q),p))

def solve_frame(human):
    import mink
    morphology_targets(human)
    for tasks,enabled in [(retargeter.tasks1,retargeter.use_ik_match_table1),(retargeter.tasks2,retargeter.use_ik_match_table2)]:
        if not enabled:continue
        prev=np.inf
        for iteration in range(retargeter.max_iter+1):
            err=np.linalg.norm(np.concatenate([t.compute_error(retargeter.configuration) for t in tasks]))
            if prev-err<=.001:break
            prev=err
            # Keywords are essential: installed Mink added safety_break before limits.
            vel=mink.solve_ik(retargeter.configuration,tasks,model.opt.timestep,solver='daqp',
                damping=retargeter.damping,limits=retargeter.ik_limits,safety_break=True)
            retargeter.configuration.integrate_inplace(vel,model.opt.timestep)
    return retargeter.configuration.q.copy()

def local_states(data):
    root=model.body('torso').id;rt=data.xmat[root].reshape(3,3).T;pt=data.xpos[root]
    result=[]
    for side,ee,elbow in zip(('l','r'),EE_NAMES,ELBOW_NAMES):
        # Shoulder origin before the first arm joint; axes fixed in torso frame.
        shoulder=model.body(f'{side}_shoulder_frame').id
        ps=rt@(data.xpos[shoulder]-pt)
        poses=[]
        for name in (ee,elbow):
            i=model.body(name).id;t=np.eye(4);t[:3,3]=rt@(data.xpos[i]-pt)-ps
            t[:3,:3]=rt@data.xmat[i].reshape(3,3);poses.extend(pose7(t))
        result.append(poses)
    return np.asarray(result)

def process(row,source):
    start=time.monotonic();path=source/row['path'];dest=OUTPUT/'data/clips'/Path(row['path']).with_suffix('.npz')
    result={**row,'source_sha256':sha256(path)}
    try:
        if row.get('source_frames',2)<2:
            result.update(status='excluded',reason='fewer_than_two_source_frames');return result
        human_pos,human_quat,ts,meta=load_clip(path);n=len(ts)
        if n<6:
            result.update(status='excluded',reason='fewer_than_6_frames_at_30hz',**meta);return result
        q0=model.qpos0.copy()
        for name,value in [('l_shoulder_roll',.25),('r_shoulder_roll',-.25),('l_elbow_pitch',-.8),('r_elbow_pitch',-.8)]:
            q0[model.joint(name).qposadr[0]]=value
        retargeter.configuration.update(q0)
        # Existing IGRIS GMR scaling, with original estimated-height convention.
        ratio=meta['estimated_height_m']/1.8
        retargeter.human_scale_table={k:v*ratio for k,v in default_scale.items()}
        q=np.empty((n,model.nq));state=np.empty((n,2,14));flags=np.zeros((n,len(REASONS)),dtype=bool)
        target_pos_error=np.zeros((n,2,2));target_angle_error=np.zeros((n,2,2));arm_angle_error=np.zeros((n,2,2))
        collisions=Counter()
        for t in range(n):
            human={name:(human_pos[t,j],human_quat[t,j]) for j,name in enumerate(JOINT_NAMES)}
            q[t]=solve_frame(human)
            data=retargeter.configuration.data
            state[t]=local_states(data)
            flags[t,0]=not np.isfinite(q[t]).all()
            flags[t,1]=bool(np.any(q[t,7:]<model.jnt_range[1:,0]-1e-6) or np.any(q[t,7:]>model.jnt_range[1:,1]+1e-6))
            flags[t,2]=bool(np.max(A@q[t,7:]-b)>1e-6)
            hit=geometry.collision(data)
            if hit:flags[t,3]=True;collisions[' / '.join(hit['pair'])]+=1
            scaled=retargeter.scaled_human_data
            for s,side in enumerate(('l','r')):
                hside='left' if s==0 else 'right'
                for k,part in enumerate(('elbow','wrist')):
                    link=ELBOW_NAMES[s] if part=='elbow' else f'{side}_wrist_connector'
                    i=model.body(link).id;hp,hq=scaled[f'{hside}_{part}']
                    target_pos_error[t,s,k]=np.linalg.norm(data.xpos[i]-hp)
                    target_angle_error[t,s,k]=(Rotation.from_quat(hq,scalar_first=True).inv()*Rotation.from_matrix(data.xmat[i].reshape(3,3))).magnitude()
                robot_points=np.array([data.xpos[model.body(l).id] for l in (f'{side}_shoulder_frame',ELBOW_NAMES[s],f'{side}_wrist_connector')])
                hp=np.array([scaled[f'{hside}_{part}'][0] for part in ('shoulder','elbow','wrist')])
                ru=np.diff(robot_points,axis=0);hu=np.diff(hp,axis=0)
                cos=(ru*hu).sum(-1)/np.maximum(np.linalg.norm(ru,axis=1)*np.linalg.norm(hu,axis=1),1e-9)
                arm_angle_error[t,s]=np.arccos(np.clip(cos,-1,1))
            flags[t,4]=np.max(target_pos_error[t])>.10
            flags[t,5]=np.max(target_angle_error[t,:,1])>np.deg2rad(40)
            flags[t,6]=np.max(arm_angle_error[t])>np.deg2rad(40)
        # Preserve original frame flags; velocity/acceleration invalidate both ends.
        velocity=np.diff(q[:,7:],axis=0)*30
        bad=np.max(np.abs(velocity[:,15:29]),axis=1)>3
        flags[:-1,7]|=bad;flags[1:,7]|=bad
        bad=np.max(np.abs(np.diff(velocity[:,15:29],axis=0)*30),axis=1)>60
        flags[:-2,8]|=bad;flags[1:-1,8]|=bad;flags[2:,8]|=bad
        prelim=~flags.any(axis=1)
        for t in range(1,n):
            if prelim[t-1] and prelim[t]:
                hit=geometry.corridor(retargeter.configuration.data,q[t-1],q[t])
                if hit:flags[t-1:t+1,9]=True;collisions['corridor: '+' / '.join(hit['pair'])]+=1
        # Quaternion signs cannot jump inside a sequence. Align each pose separately.
        for s in range(2):
            for off in (3,10):
                if state[0,s,off]<0:state[0,s,off:off+4]*=-1
                for t in range(1,n):
                    if np.dot(state[t-1,s,off:off+4],state[t,s,off:off+4])<0:state[t,s,off:off+4]*=-1
        valid=~flags.any(axis=1)
        windows=np.convolve(valid.astype(int),np.ones(6,dtype=int),'valid')==6
        dest.parent.mkdir(parents=True,exist_ok=True)
        tmp=dest.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,qpos=q.astype(np.float32),state=state.astype(np.float32),time=ts,
            valid=valid,reject_flags=flags,joint_names=np.array(names),reason_names=np.array(REASONS),
            human_positions=human_pos.astype(np.float32),human_quaternions=human_quat.astype(np.float32),
            target_position_error=target_pos_error.astype(np.float32),target_angle_error=target_angle_error.astype(np.float32),
            arm_angle_error=arm_angle_error.astype(np.float32),source_sha256=result['source_sha256'])
        tmp.replace(dest)
        result.update(status='processed',output=str(dest),frames=n,valid_frames=int(valid.sum()),valid_windows=int(windows.sum()),
            rejection_counts=dict(zip(REASONS,flags.sum(0).astype(int).tolist())),collision_pairs=dict(collisions),
            mean_wrist_target_error_m=float(target_pos_error[:,:,1].mean()),mean_arm_direction_error_rad=float(arm_angle_error.mean()),**meta)
    except Exception as e:
        result.update(status='failed',reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc())
    result['elapsed_s']=time.monotonic()-start
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=IGRIS/'retargeting/AMASS_dataset')
    ap.add_argument('--jobs',type=int,default=12);ap.add_argument('--limit',type=int);ap.add_argument('--manifest-only',action='store_true');ap.add_argument('--retry-failed',action='store_true')
    args=ap.parse_args();out=OUTPUT/'data';out.mkdir(parents=True,exist_ok=True)
    mf=out/'source_manifest.json'
    rows=json.loads(mf.read_text())['files'] if mf.exists() else make_manifest(args.source)
    if args.manifest_only:return
    log=out/'retarget_results.jsonl';previous={}
    if log.exists():
        for line in log.read_text().splitlines():
            r=json.loads(line);previous[r['path']]=r
    rows=sorted(rows,key=lambda r:(0 if r['paper'] else 1 if r['corpus'] in TEST_CORPUS else 2,r['path']))
    pending=[r for r in rows if r['status']=='pending' and
        (r['path'] not in previous or (args.retry_failed and previous[r['path']]['status']=='failed'))]
    if args.limit:pending=pending[:args.limit]
    save_json(out/'run_config.json',dict(fps=30.,history=5,jobs=args.jobs,quality_reasons=REASONS,
        position_threshold_m=.10,wrist_orientation_threshold_deg=40,arm_direction_threshold_deg=40,
        arm_velocity_limit_rad_s=3,arm_acceleration_limit_rad_s2=60,corridor_step_rad=.02,
        source_manifest_sha256=sha256(mf),model_sha256=sha256(OUTPUT/'model/igris_c_rcv_hand_torso.xml')))
    initialize();save_json(out/'collision_model.json',geometry.provenance())
    started=time.monotonic();done=0
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=mp.get_context('fork'),initializer=initialize) as pool,log.open('a',buffering=1) as fh:
        futures={pool.submit(process,row,args.source):row['path'] for row in pending}
        for f in as_completed(futures):
            result=f.result();fh.write(json.dumps(result,allow_nan=False)+'\n');previous[result['path']]=result;done+=1
            print(json.dumps(dict(done=done,total=len(pending),path=result['path'],status=result['status'],frames=result.get('frames'),valid=result.get('valid_frames'),seconds=round(time.monotonic()-started,1),error=result.get('reason'))),flush=True)
    for r in rows:
        if r['path'] not in previous:previous[r['path']]=r
    save_json(out/'retarget_summary.json',dict(files=list(previous.values()),status_counts=dict(Counter(r['status'] for r in previous.values())),elapsed_s=time.monotonic()-started))

if __name__=='__main__':main()
