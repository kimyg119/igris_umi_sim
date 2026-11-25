"""Arrange and evaluate the room for an explicitly selected 1 kg q archive."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
MOTION=ROOT/'outputs/synthetic_box_motion_20260914/motion_100hz.npz'
SOURCE=ROOT/'igris/sim_local/vr_box_eval/results/20260914_synthetic_motion100_1kg'
PAYLOAD='/World/Boxes/Small_RightFront'


def load_motion(path):
    with np.load(path,allow_pickle=False) as data:
        t=data['command_t_s'].copy();q=data['q_command_rad'].copy();indices=data['command_sdk_indices'].copy()
        if t.ndim!=1 or data['mode'].shape!=t.shape or data['trip'].shape!=t.shape:raise ValueError('Invalid archive timeline')
        if not np.array_equal(indices,np.r_[0:3,15:29]):raise ValueError('Expected waist and both arms only')
        if q.shape!=(len(t),17) or len(t)<2 or not np.isfinite(q).all():raise ValueError('Invalid archived q')
        if not np.allclose(t,np.arange(len(t))*.01,atol=1e-9,rtol=0):raise ValueError('Expected complete 100 Hz timeline')
        if np.any(data['mode']!='FOLLOWING') or np.any(data['trip']!=''):raise ValueError('Inactive or tripped archive')
        if np.max(abs(np.diff(q,axis=0)))/.01>.5:raise ValueError('Unexpected command velocity')
    return t,q,indices


def layout(out,offset=(0.,0.,0.),compliant_hands=False,motion=MOTION,vendor_box=None,destination=(.33,.20),source_base_clearance=0.,source_base_normal=0.,narrow_source_support=False,payload_mass=1.):
    from pxr import Gf,Usd,UsdGeom,UsdPhysics,UsdShade
    from prepare import SCENE,prepare
    if not np.isfinite(offset).all():raise ValueError('Non-finite placement offset')
    if not np.isfinite(payload_mass) or payload_mass<=0:raise ValueError('Positive finite payload mass required')
    if not np.isfinite([source_base_clearance,source_base_normal]).all() or not 0<=source_base_clearance<=.06 or abs(source_base_normal)>.08:
        raise ValueError('Source base offsets exceed the supported stack footprint')
    if vendor_box is None:vendor_box=compliant_hands
    if vendor_box and not compliant_hands:raise ValueError('Vendor box comparison requires the compliant hand model')
    if narrow_source_support and not compliant_hands:raise ValueError('Narrow source support requires compliant hands')
    motion=Path(motion).resolve();t,q,indices=load_motion(motion)
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    stage=Usd.Stage.Open(str(SCENE));stage.SetEditTarget(stage.GetSessionLayer())
    cart=UsdGeom.Xformable(stage.GetPrimAtPath('/World/Rolltainer'))
    cart_matrix=cart.GetLocalTransformation()
    cart_matrix.SetTranslateOnly(cart_matrix.ExtractTranslation()+Gf.Vec3d(0,1.,0))
    cart.MakeMatrixXform().Set(cart_matrix)
    pallet=UsdGeom.Xformable(stage.GetPrimAtPath('/World/Pallet'))
    pallet_matrix=pallet.GetLocalTransformation()
    pallet_matrix.SetTranslateOnly(pallet_matrix.ExtractTranslation()+Gf.Vec3d(.1/np.sqrt(2),-.1/np.sqrt(2),0))
    if compliant_hands:pallet_matrix=Gf.Matrix4d().SetTranslate(Gf.Vec3d(2,-2,0))
    pallet.MakeMatrixXform().Set(pallet_matrix)
    changes={}
    def place(name,xyz,yaw=0.,roll=0.):
        prim=stage.GetPrimAtPath('/World/Boxes/'+name)
        matrix=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1,0,0),roll))
        matrix=matrix*Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),yaw))
        matrix.SetTranslateOnly(Gf.Vec3d(*map(float,xyz)))
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(matrix)
        changes[name]={'xyz_m':list(map(float,xyz)),'yaw_deg':yaw,'roll_deg':roll}
    xy=np.array([.37/np.sqrt(2),-.37/np.sqrt(2)])+np.asarray(offset[:2])
    top=0. if compliant_hands else .15
    source_names=['Pallet_C1_L1','Pallet_C1_L2','Pallet_C2_L1','Pallet_C1_L3'] if compliant_hands else [f'Pallet_C1_L{i}' for i in (1,2,3)]
    if narrow_source_support:
        # Original assets, rotated/repositioned only: .23+.23+.10+.18=.74 m.
        # Upright small top box has a .10 m width beneath the .15 m payload.
        source_names=['Pallet_C1_L1','Pallet_C1_L2','Small_Center_L1','Small_Center_L2']
    for i,name in enumerate(source_names):
        last=i==len(source_names)-1;height=.23 if last else .17
        radial=[.165,.145,.12,.095][i] if compliant_hands else (.095 if last else .17)
        roll=90 if last else 0
        if narrow_source_support:
            height=[.23,.23,.10,.18][i];radial=[.165,.145,.04,0.][i];roll=[90,90,0,90][i]
        if compliant_hands and i==0:radial+=source_base_clearance
        at=xy+np.array([1.,-1.])*radial/np.sqrt(2)
        if compliant_hands and i==0:at+=np.array([1.,1.])*source_base_normal/np.sqrt(2)
        place(name,[*at,top+height/2+.0002],-45,roll)
        top+=height+.0002
    # The destination is deliberately below release height so the saved descent
    # can finish before the box settles onto the large boxes.
    destination=np.asarray(destination,float);destination_names=[]
    if destination.shape!=(2,) or not np.isfinite(destination).all():raise ValueError('Finite destination XY required')
    destination_shift=destination-np.array([.33,.20])
    for column,y in [(3,.065),(2,.295)]:
        dest_top=0.
        for i,name in enumerate([f'Roll_C{column}_R1_L1',f'Roll_C{column}_R1_L2',f'Roll_C{column}_R1_L3',f'Roll_C{column}_R2_L1']):
            place(name,[* (np.array([.46 if i==3 else .51,y])+destination_shift),dest_top+.085+.0002])
            destination_names.append('room_Boxes__'+name)
            dest_top+=.17+.0002
    payload=stage.GetPrimAtPath(PAYLOAD)
    if vendor_box:
        scale=Gf.Matrix4d().SetScale(Gf.Vec3d(.15/.18,.15/.18,.15/.10))
        for child in payload.GetChildren():
            if child.IsA(UsdGeom.Xformable):
                transform=UsdGeom.Xformable(child);matrix=transform.GetLocalTransformation()*scale
                transform.MakeMatrixXform().Set(matrix)
    # Apply the same body orientation to the visual mesh and grasp collider.
    UsdPhysics.MassAPI(payload).GetMassAttr().Set(float(payload_mass))
    material=UsdShade.Material.Define(stage,'/World/Materials/ReplayPayload')
    physics=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateDynamicFrictionAttr(1.2);physics.CreateStaticFrictionAttr(1.2)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(PAYLOAD+'/Body_Collider')).Bind(material,materialPurpose='physics')
    place(PAYLOAD.rsplit('/',1)[-1],[*xy,top+(.075 if vendor_box else .09)+.0003+offset[2]],45,90)
    # Keep all other boxes dynamic and collidable, outside the arm sweep.
    extras=[p for p in stage.GetPrimAtPath('/World/Boxes').GetChildren() if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetName() not in changes]
    for i,p in enumerate(extras):
        height=.10 if p.GetName().startswith('Small_') else .17
        place(p.GetName(),[1.35+.40*(i%4),-1.0+.30*(i//4),height/2+.0003])
    usd=out/'room_replay.usdc';stage.Flatten().Export(str(usd))
    manifest=prepare(out/'run',scene=usd,compliant_hands=compliant_hands)
    manifest['payload']={'usd_path':PAYLOAD,'body':'room_Boxes__'+PAYLOAD.rsplit('/',1)[-1],'mass_kg':float(payload_mass),'size_m':[.15]*3 if vendor_box else [.18,.18,.10],'friction':1.2,'condim':4 if vendor_box else 3}
    manifest['payload']['geom']=next(c['geom'] for c in manifest['room_colliders'] if c['body']==manifest['payload']['body'])
    manifest['destination_bodies']=destination_names
    manifest['source_support_bodies']=['room_Boxes__'+name for name in source_names]
    if compliant_hands:
        ref=json.loads(MOTION.with_name('source_metadata.json').read_text());x,y,z,w=ref['base_quat_xyzw']
        manifest.update(stage_room_after_preparation=True,room_reference_base_xyyaw=[*ref['base_position'][:2],float(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))])
    if vendor_box:
        import xml.etree.ElementTree as ET
        tree=ET.parse(out/'run/scene.xml');geom=tree.find(f".//geom[@name='{manifest['payload']['geom']}']")
        if not np.allclose(2*np.fromstring(geom.get('size'),sep=' '),.15,atol=1e-7,rtol=0):raise ValueError('Incorrect baked box size')
        original=ET.parse(SOURCE/'scene.xml').find(".//geom[@name='eval_box_geom']")
        for field in ('size','friction','condim'):geom.set(field,original.get(field))
        tree.write(out/'run/scene.xml',encoding='unicode')
    (out/'run/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report={'source_usd':str(SCENE),'scene':str(usd),'motion':str(MOTION),'motion_sha256':hashlib.sha256(MOTION.read_bytes()).hexdigest(),'samples':len(t),'duration_sim_s':float(t[-1]),'q_modified':False,'robot_physics_modified':False,'payload':manifest['payload'],'source_support_top_m':top,'destination_support_top_m':dest_top,'destination_xy_m':destination.tolist(),'rolltainer_translation_m':[0,1.,0],'pallet_translation_m':[.1/np.sqrt(2),-.1/np.sqrt(2),0],'boxes':changes,'limits':'Original replay used different finger compliance; unchanged q does not establish a successful grasp in this model.'}
    if compliant_hands:
        report.update(robot_physics_modified=True,authorized_robot_changes='Existing MCP servo and passive DIP spring; 0.6 kg per hand; non-hand XML unchanged',limits='Full-hand grasp requires measured palm and distal contact during unsupported carry.')
        report.pop('pallet_translation_m')
    report.update(motion=str(motion),motion_sha256=hashlib.sha256(motion.read_bytes()).hexdigest(),q_modified=hashlib.sha256(motion.read_bytes()).digest()!=hashlib.sha256(MOTION.read_bytes()).digest())
    report.update(narrow_source_support=narrow_source_support,source_support_bodies=manifest['source_support_bodies'])
    if compliant_hands:report.update(pallet_world_xyz_m=[2,-2,0],pallet_yaw_deg=0,stage_room_after_preparation=True)
    if vendor_box:report['payload_shape_source']=str(SOURCE/'scene.xml')
    (out/'layout.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def analyze(out):
    from run import JOINTS
    manifest=json.loads((out/'manifest.json').read_text());model=json.loads((out/'model.json').read_text())
    payload=manifest['payload']['body'];index=next(i for i,b in enumerate(model['bodies']) if b['name']==payload)
    raw=np.fromfile(out/'poses.bin');width=2+7*len(model['bodies']);rows=raw[:len(raw)//width*width].reshape(-1,width)
    trials=sorted(out.glob('command_*.json'));command=json.loads(trials[-1].read_text());begin=command.get('replay_start_wall',rows[-1,0])
    end=command['samples'][-1]['wall']
    rows=rows[(rows[:,0]>=begin)&(rows[:,0]<=end)];box=rows[:,2+7*index:2+7*(index+1)]
    t,q,owned=load_motion(Path(command['motion']));samples=[s for s in command['samples'] if s['archive_index']>=0]
    if not samples:
        report={'pass':False,'hardware_executed':False,'command_pass':False,'command_error':command.get('error'),'unique_q_samples_sent':0,'all_q_samples_sent':False,'stage':'preparation; archive replay never started'}
        (out/'replay_result.json').write_text(json.dumps(report,indent=2)+'\n');return report
    seen={s['archive_index'] for s in samples}
    delta=max(float(np.max(abs(np.asarray(s['target'])[owned]-q[s['archive_index']]))) for s in samples)
    reference_valid=delta==0;live_validation=None
    if command.get('live_hl_ik'):
        reference_delta=max(float(np.max(abs(np.asarray(s['nominal_target'])[owned]-q[s['archive_index']]))) for s in samples)
        archive_nominal_delta=reference_delta
        if command.get('inplace_transfer'):
            from inplace_reference import apply_waist_reference
            errors=[]
            for sample in samples:
                archived=np.zeros(31);archived[owned]=q[sample['archive_index']]
                expected=apply_waist_reference(archived,sample.get('walking'))
                errors.append(float(np.max(abs(np.asarray(sample['nominal_target'])[owned]-expected[owned]))))
            reference_delta=max(errors)
        live_path=out/'live_ik_result.json'
        live_validation=json.loads(live_path.read_text()) if live_path.exists() else {'error':'missing worker result'}
        reference_valid=reference_delta==0 and live_validation.get('all_accepted',False) and not live_validation.get('error')
    all_events=[json.loads(line) for line in (out/'contacts.jsonl').read_text().splitlines()]
    all_events=[e for e in all_events if begin<=e['wall']<=end]
    events=[e for e in all_events if e['room_body']==payload]
    left={e['wall'] for e in events if e['robot_geom'].startswith('l_hand')};right={e['wall'] for e in events if e['robot_geom'].startswith('r_hand')}
    height=box[:,2];raised=height>height[0]+.025
    support_events=[json.loads(line) for line in (out/'payload_contacts.jsonl').read_text().splitlines()]
    final_start=rows[-1,0]-1.
    destination_bodies=set(manifest.get('destination_bodies',['room_Boxes__'+n for n in ['Roll_C3_R1_L1','Roll_C3_R1_L2','Roll_C3_R1_L3','Roll_C3_R2_L1']]))
    destination_geoms={c['geom'] for c in manifest['room_colliders'] if c['body'] in destination_bodies}
    from placement_support import final_support_evidence
    destination_support=final_support_evidence(rows[:,:2],support_events,destination_geoms,manifest['payload']['mass_kg'])
    final_start=destination_support['start_wall']
    environment_geoms={c['geom'] for c in manifest['room_colliders']} - {manifest['payload']['geom']}
    grounded={e['wall'] for e in support_events if e['normal_force_N']>.1 and (environment_geoms.intersection(e['geoms']) or any('floor' in g or 'ground' in g for g in e['geoms']))}
    held=sorted((left&right)-grounded)
    sim_at={e['wall']:e['sim'] for e in events}
    unsupported_duration=longest=0.
    for previous,current in zip(held,held[1:]):
        unsupported_duration=unsupported_duration+sim_at[current]-sim_at[previous] if current-previous<.1 else 0.
        longest=max(longest,unsupported_duration)
    destination=np.asarray(json.loads((out.parent/'layout.json').read_text())['destination_xy_m'])
    if (out/'room_layout_applied.json').exists() and manifest.get('room_staging',{}).get('scope')!='source_only':
        placement=json.loads((out/'room_layout_applied.json').read_text());yaw=placement.get('yaw_rad',0.);c,s=np.cos(yaw),np.sin(yaw)
        destination=np.array([[c,-s],[s,c]])@destination+placement.get('translation_xy',[0,0])
    # Actual receiving-box contact determines landing; measure settling in sim
    # time rather than accepting a transient contact while the object falls.
    last=np.flatnonzero(rows[:,1]>=rows[-1,1]-1.)
    final_speed=float(np.linalg.norm(box[-1,:3]-box[last[0],:3])/(rows[-1,1]-rows[last[0],1]))
    at_destination=destination_support['pass'] and box[-1,2]>.6 and final_speed<.03
    target_z=None
    if command.get('inplace_transfer'):target_z=manifest['inplace_transfer']['target_scene_xyz'][2]
    elif command.get('walking_transfer'):target_z=manifest['walking_transfer']['target_payload_xyz'][2]
    destination_z_error=None if target_z is None else abs(float(box[-1,2])-target_z)
    if destination_z_error is not None:at_destination=bool(at_destination and destination_z_error<.025)
    final_robot_forces={}
    for e in events:
        if e['wall']>=final_start:final_robot_forces[e['wall']]=final_robot_forces.get(e['wall'],0.)+e['normal_force_N']
    maximum_final_robot_force=max(final_robot_forces.values(),default=0.)
    released=maximum_final_robot_force<.05*manifest['payload']['mass_kg']*9.81
    report={'pass':False,'hardware_executed':False,'command_pass':command['pass'],'command_error':command.get('error'),'source_q_max_difference_rad':delta,'source_q_sha256':command['motion_sha256'],'unique_q_samples_sent':len(seen),'all_q_samples_sent':len(q)==len(seen),'box_initial_xyz':box[0,:3].tolist(),'box_final_xyz':box[-1,:3].tolist(),'box_max_height_m':float(height.max()),'box_lift_m':float(height.max()-height[0]),'two_hand_contact_frames':len(left&right),'payload_contact_records':len(events),'raised_frames':int(raised.sum()),'unsupported_two_hand_frames':len(held),'longest_unsupported_two_hand_sim_s':longest,'destination_xy_error_m':float(np.linalg.norm(box[-1,:2]-destination)),'final_support_contact':destination_support['pass'],'destination_support_evidence':destination_support,'final_speed_m_s':final_speed,'released':released,'limits':'Contact sampled at up to 30 Hz; no palm/fingertip pressure-quality or hardware claim.'}
    sequence_complete=len(seen)==len(q)
    report['destination_z_error_m']=destination_z_error
    report['maximum_final_robot_payload_normal_force_N']=maximum_final_robot_force
    report['final_robot_payload_contact_frames']=len(final_robot_forces)
    if command.get('walking_transfer'):
        walk_samples=[s for s in samples if s.get('walking')]
        sequence_complete=bool(command.get('walking_result',{}).get('done'))
        pelvis_xyz=np.array([s['pelvis'][:3] for s in walk_samples])
        traveled=float(np.linalg.norm(pelvis_xyz[-1,:2]-pelvis_xyz[0,:2])) if len(pelvis_xyz)>1 else 0.
        feet={}
        for name in ('l_foot_original','r_foot_original'):
            i=next(i for i,b in enumerate(model['bodies']) if b['name']==name)
            foot=rows[:,2+7*i:5+7*i]
            feet[name]=dict(vertical_range_m=float(np.ptp(foot[:,2])),xy_displacement_m=float(np.linalg.norm(foot[-1,:2]-foot[0,:2])))
        report.update(walking_transfer=True,walking_result=command.get('walking_result'),pelvis_displacement_during_walk_m=traveled,foot_motion=feet)
        sequence_complete=bool(sequence_complete and traveled>.5 and all(v['vertical_range_m']>.008 and v['xy_displacement_m']>.4 for v in feet.values()))
    if command.get('inplace_transfer'):
        transfer_samples=[s for s in samples if s.get('walking')]
        pelvis_xyz=np.array([s['pelvis'][:3]for s in transfer_samples])
        displacement=float(np.max(np.linalg.norm(pelvis_xyz[:,:2]-pelvis_xyz[0,:2],axis=1))) if len(pelvis_xyz) else None
        zero_commands=bool(transfer_samples) and all(np.array_equal(s['walking']['command_xyz'],[0.,0.,0.]) for s in transfer_samples)
        sequence_complete=bool(command.get('inplace_result',{}).get('done') and zero_commands and displacement is not None and displacement<.10)
        report.update(inplace_transfer=True,inplace_result=command.get('inplace_result'),zero_walking_commands=zero_commands,
            pelvis_max_displacement_during_inplace_m=displacement,inplace_pelvis_limit_m=.10,
            nominal_reference_schema='explicit_inplace_waist/v1',archive_nominal_difference_max_rad=archive_nominal_delta)
    destination_valid=report['destination_xy_error_m']<.03 if command.get('walking_transfer') or command.get('inplace_transfer') else True
    report['pass']=bool(command['pass'] and sequence_complete and reference_valid and raised.any() and longest>1. and at_destination and released and destination_valid)
    report['basic_transfer_pass']=report['pass']
    if live_validation is not None:
        report.update(live_hl_ik=live_validation,nominal_reference_difference_max_rad=reference_delta,
                      live_ik_active_fraction=float(np.mean([s['live_ik_active'] for s in samples])))
    report['replay_realtime_ratio']=float((rows[-1,1]-rows[0,1])/(rows[-1,0]-rows[0,0]))
    report['non_hand_payload_force_max_N']=max([e['normal_force_N'] for e in events if not e['robot_geom'].startswith(('l_hand','r_hand'))],default=0.)
    report['non_hand_room_force_max_N']=max([e['normal_force_N'] for e in all_events if not e['robot_geom'].startswith(('l_hand','r_hand'))],default=0.)
    room_contacts={}
    for event in all_events:
        if event['robot_geom'].startswith(('l_hand','r_hand')):continue
        key=(event['robot_geom'],event['room_body'])
        item=room_contacts.setdefault(key,dict(robot_geom=key[0],room_body=key[1],samples=0,max_force_N=0.))
        item['samples']+=1;item['max_force_N']=max(item['max_force_N'],event['normal_force_N'])
    report['non_hand_room_contacts']=sorted(room_contacts.values(),key=lambda x:-x['max_force_N'])
    support_motion={}
    for body in manifest.get('source_support_bodies',['room_Boxes__'+name for name in ('Pallet_C1_L1','Pallet_C1_L2','Pallet_C2_L1','Pallet_C1_L3')]):
        body_index=next(i for i,b in enumerate(model['bodies']) if b['name']==body)
        xyz=rows[:,2+7*body_index:5+7*body_index]
        support_motion[body.removeprefix('room_Boxes__')]=dict(max_xy_displacement_m=float(np.linalg.norm(xyz[:,:2]-xyz[0,:2],axis=1).max()),
                                  final_displacement_m=(xyz[-1]-xyz[0]).tolist())
    report['source_support_motion']=support_motion
    if command.get('backlash_compensation_active'):
        report['backlash_compensation_active']=True
        report['command_reference_difference_max_rad']=max(float(np.max(abs(np.asarray(s['motor_target'])[owned]-q[s['archive_index']]))) for s in samples)
        error=np.array([np.asarray(s['q'])+s['joint_offsets']-s['target'] for s in samples])[:,owned]
        report['output_angle_error_max_rad']=float(np.max(abs(error)))
        report['output_angle_error_p95_rad']=float(np.percentile(abs(error),95))
    ages=[s['state_receive_age_s'] for s in samples]
    report['lowstate_receive_age_max_s']=float(max(ages))
    report['lowstate_receive_age_p95_s']=float(np.percentile(ages,95))
    report['command_wall_rate_hz']=(len(samples)-1)/(samples[-1]['wall']-samples[0]['wall']) if len(samples)>1 else None
    if manifest.get('compliant_hands'):
        motion=np.load(command['motion']);raw=np.load(MOTION.with_name('raw_targets.npz'));original=np.load(MOTION)
        phases={str(raw['stage'][i]):float(np.interp(raw['t'][i]-original['source_offset_s'],original['source_time_s'],original['command_t_s'])) for i in np.r_[0,np.flatnonzero(raw['stage'][1:]!=raw['stage'][:-1])+1]}
        progress=motion['original_command_t_s'] if 'original_command_t_s' in motion else t
        frame_progress=np.interp(rows[:,0],[s['wall'] for s in samples],[progress[s['archive_index']] for s in samples])
        carry=(frame_progress>=phases['carry_left'])&(frame_progress<phases['lower'])
        if command.get('walking_transfer') or command.get('inplace_transfer'):
            sample_times=np.array([s['wall'] for s in samples])
            nearest=np.maximum(0,np.searchsorted(sample_times,rows[:,0],side='right')-1)
            carry=np.array([samples[i].get('walking',{}).get('phase') in ('carry','waist_arc') for i in nearest])
        forces={float(f'{stamp:.9f}'):np.zeros(4) for stamp in rows[:,0]}
        for e in events:
            if e.get('patch') not in ('palm','tip') or not e['robot_geom'].startswith(('l_hand','r_hand')):continue
            key=e['wall']
            if key in forces:
                forces[key][2*int(e['robot_geom'].startswith('r_'))+int(e['patch']=='tip')]+=e['normal_force_N']
        pressure=np.array(list(forces.values()));broad=np.all(pressure>.2,axis=1)
        load=np.all(pressure[:,[0,2]]>=1.5*manifest['payload']['mass_kg']*9.81/(2*manifest['payload']['friction']),axis=1)&np.all(pressure[:,[1,3]]>=.2,axis=1)
        unsupported=np.array([float(f'{stamp:.9f}') not in grounded for stamp in rows[:,0]])
        frac=lambda a:float(np.mean(a[carry])) if carry.any() else None
        palm_load=np.all(pressure[:,[0,2]]>=1.5*manifest['payload']['mass_kg']*9.81/(2*manifest['payload']['friction']),axis=1)
        report['two_palm_supported_fraction_carry']=frac(palm_load&unsupported&(True if command.get('walking_transfer') or command.get('inplace_transfer') else raised))
        report['relaxed_pass']=bool(manifest.get('evaluation_profile')=='relaxed_transport' and report['basic_transfer_pass'] and report['two_palm_supported_fraction_carry'] is not None and report['two_palm_supported_fraction_carry']>=.95)
        report['evaluation_profile']=manifest.get('evaluation_profile','original_grasp_requirements')
        report.update(palm_tip_force_order=['left_palm','left_tip','right_palm','right_tip'],palm_tip_force_max_N=pressure.max(0).tolist(),full_hand_contact_fraction_carry=frac(broad),full_hand_supported_fraction_carry=frac(broad&load&unsupported&raised),carry_mean_world_z_m=float(np.mean(height[carry])) if carry.any() else None,carry_min_world_z_m=float(np.min(height[carry])) if carry.any() else None,carry_samples=int(carry.sum()),limits='Native contact samples up to 30 Hz; simulated motor current and spring estimate, no hardware execution.')
        report['pass']=bool(report['pass'] and report['full_hand_supported_fraction_carry'] is not None and report['full_hand_supported_fraction_carry']>=.95)
        motor=out/'hand_compliance_log.json'
        if motor.exists():
            h=json.loads(motor.read_text());hr=[v for v in h['rows'] if begin<=v['wall_time']<=end]
            report['hand_controller_error']=h['error']
            report['hand_failure']=h.get('failure')
            if hr:report.update(mcp_range_deg=np.rad2deg([min(v['q'] for v in hr),max(v['q'] for v in hr)]).tolist(),dip_range_deg=np.rad2deg([min(v['dip_q'] for v in hr),max(v['dip_q'] for v in hr)]).tolist(),simulated_supply_current_max_A=max(v['current_A'] for v in hr))
    (out/'replay_result.json').write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','analyze'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--offset',type=float,nargs=3,default=(0.,0.,0.))
    parser.add_argument('--compliant-hands',action='store_true',help='Reuse the existing MCP servo and passive DIP model')
    parser.add_argument('--vendor-box',action=argparse.BooleanOptionalAction,default=None,help='15 cm cube/contact properties; default with compliant hands. --no-vendor-box keeps the original room size')
    parser.add_argument('--motion',type=Path,default=MOTION)
    parser.add_argument('--destination',type=float,nargs=2,default=(.33,.20))
    parser.add_argument('--source-base-clearance',type=float,default=0.)
    parser.add_argument('--source-base-normal',type=float,default=0.)
    parser.add_argument('--narrow-source-support',action='store_true')
    parser.add_argument('--payload-mass',type=float,default=1.)
    args=parser.parse_args();print(json.dumps(layout(args.out,args.offset,args.compliant_hands,args.motion,args.vendor_box,args.destination,args.source_base_clearance,args.source_base_normal,args.narrow_source_support,args.payload_mass) if args.action=='prepare' else analyze(args.out),indent=2))
