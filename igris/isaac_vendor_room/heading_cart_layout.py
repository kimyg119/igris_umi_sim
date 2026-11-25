"""Build a fresh, explicitly reviewed heading-preserving rolltainer layout.

The proposal contains the original assets and offline evidence. Source poses
remain unchanged; cart and receiving assets stay fixed during source staging.
Only tangential mu4 is applied here. The trial runner owns later contact options.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from walking_transfer import pose_matrix
from rolltainer_staging import staging_contract

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'outputs/hlik_transport_20260918_2005'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_proposal(path):
    path=Path(path).resolve();value=json.loads(path.read_text())
    if value.get('schema')!='heading_cart_layout_proposal/v1':raise ValueError('Unknown heading cart proposal')
    hashes=value.get('hashes',{})
    for key in ('scene','native_geometry_reference','source_layout','source_manifest','source_scene'):
        file=Path(value[key]).resolve()
        if str(file)not in hashes or not file.is_file()or sha(file)!=hashes[str(file)]:
            raise ValueError('Proposal source hash mismatch: '+key)
    for name,expected in hashes.items():
        if not Path(name).is_file()or sha(name)!=expected:raise ValueError('Proposal evidence hash mismatch: '+name)
    room=value['room_body_poses'];source=value['source_support_bodies']+[value['payload']['body']];dest=value['destination_bodies']
    if (len(room)!=39 or len(source)!=5 or len(dest)!=6 or len(set(source+dest))!=11
            or not set(source+dest)<=set(room)):
        raise ValueError('Invalid source/destination body partition')
    for pose in room.values():
        pose=np.asarray(pose,dtype=float)
        if pose.shape!=(7,)or not np.isfinite(pose).all()or abs(np.linalg.norm(pose[3:])-1)>1e-8:
            raise ValueError('Invalid authored room pose')
    proof=value['proof'];geometry=proof['route_and_arrival']
    if (proof.get('native_version')!='3.2.7' or proof.get('physics_steps')!=0
            or proof.get('acquisition_robot_nonpayload_contact_records')!=0
            or any(geometry.get(k)!=0 for k in ('nominal_robot_nonpayload_contact_records',
                'arrival_robot_nonpayload_contact_records','lower_open_robot_nonpayload_contact_records'))):
        raise ValueError('Proposal geometry evidence is not clear')
    ik=proof['nominal_ik']
    if not ik.get('completed')or ik.get('strict_failures')!=0 or not ik.get('all_prior_used'):
        raise ValueError('Proposal nominal IK evidence is not qualified')
    target=np.asarray(value['goal_proposal']['payload_center_world_m'],dtype=float)
    if target.shape!=(3,)or not np.isfinite(target).all()or abs(target[2]-value['destination_support_top_m']-.0753)>1e-7:
        raise ValueError('Invalid placement target height')
    support=value['destination_support']
    if support.get('full_footprint_supported')is not False or not support.get('intentional_footprint_overhang'):
        raise ValueError('Partial destination support must remain explicit')
    if min(support['com_projection_edge_margins_m'])<=.02:
        raise ValueError('Reviewed COM support margin is too small')
    if value['physics_profile']['payload_mass_kg']!=.1 or value['physics_profile']['tangential_friction']!=4.:
        raise ValueError('Heading proposal requires reviewed .1kg/mu4 profile')
    value['proposal_path']=str(path);value['proposal_sha256']=sha(path)
    return value


def build(out,proposal,plan=P/'ready_plan43',*,payload_mass=.1):
    from pxr import Usd,UsdGeom,UsdPhysics
    from box_replay import layout
    from prepare import prepare
    from contact_material import apply
    out=Path(out).resolve();plan=Path(plan).resolve();config=load_proposal(proposal)
    if payload_mass not in (.01,.1):
        raise ValueError('Heading layout requires explicit100g or relaxed10g payload')
    out.mkdir(parents=True,exist_ok=False)
    staged=layout(out/'staging',compliant_hands=True,motion=plan/'motion_100hz.npz',vendor_box=True,
                  source_base_clearance=.01,narrow_source_support=True,payload_mass=payload_mass)
    staging=json.loads((out/'staging/run/manifest.json').read_text())
    source=staging['source_support_bodies']+[staging['payload']['body']]
    room={x['body']:x['usd_path']for x in staging['room_bodies']}
    if (set(room)!=set(config['room_body_poses']) or source!=config['source_support_bodies']+[config['payload']['body']]):
        raise ValueError('Fresh source body partition differs from reviewed proposal')
    original=Usd.Stage.Open(staged['scene']);stage=Usd.Stage.Open(config['scene'])
    oldcache=UsdGeom.XformCache();cache=UsdGeom.XformCache()
    for name,path in room.items():
        a=original.GetPrimAtPath(path);b=stage.GetPrimAtPath(path)
        expected_mass=(float(np.float32(payload_mass)) if name==staging['payload']['body']
                       else UsdPhysics.MassAPI(b).GetMassAttr().Get())
        if UsdPhysics.MassAPI(a).GetMassAttr().Get()!=expected_mass:
            raise ValueError('Original asset mass changed: '+name)
        np.testing.assert_allclose(np.asarray(cache.GetLocalToWorldTransform(b)).T,
            pose_matrix(config['room_body_poses'][name]),atol=1e-10,rtol=0)
        if name in source:
            np.testing.assert_allclose(cache.GetLocalToWorldTransform(b),oldcache.GetLocalToWorldTransform(a),atol=1e-12,rtol=0)
        for child in a.GetChildren():
            if child.IsA(UsdGeom.Xformable):
                other=stage.GetPrimAtPath(child.GetPath())
                np.testing.assert_array_equal(UsdGeom.Xformable(child).GetLocalTransformation(),UsdGeom.Xformable(other).GetLocalTransformation())
    usd=out/'room_replay.usdc';stage.Flatten().Export(str(usd))
    # Modify only the fresh flattened layer; never mutate the reviewed scene.
    fresh_stage=Usd.Stage.Open(str(usd))
    UsdPhysics.MassAPI(fresh_stage.GetPrimAtPath(staging['payload']['usd_path'])).GetMassAttr().Set(float(payload_mass))
    fresh_stage.GetRootLayer().Save()
    manifest=prepare(out/'run',scene=usd,compliant_hands=True)
    for key in ('payload','source_support_bodies','stage_room_after_preparation','room_reference_base_xyyaw'):
        manifest[key]=copy.deepcopy(staging[key])
    manifest['payload']['geom']=next(c['geom']for c in manifest['room_colliders']if c['body']==manifest['payload']['body'])
    manifest['destination_bodies']=list(config['destination_bodies'])
    manifest['room_staging']=staging_contract(manifest)
    manifest.update(evaluation_profile='relaxed_transport',layout_kind='heading_preserving_rolltainer_offset_landing',
        narrow_destination_support=True,destination_support=copy.deepcopy(config['destination_support']),
        heading_cart_proposal=dict(path=config['proposal_path'],sha256=config['proposal_sha256'],physics_steps=0))
    # Pure room-pose changes must not perturb native auto-inertia through USD
    # roundoff. Keep every original room collider shape/local free-body pose;
    # only the fixed cart's WORLD pose and the free bodies' root poses change.
    tree=ET.parse(out/'run/scene.xml');reference=ET.parse(out/'staging/run/scene.xml')
    oldcolliders={c['usd_path']:c for c in staging['room_colliders']}
    for item in manifest['room_colliders']:
        prior=oldcolliders[item['usd_path']]
        if (prior['body'],prior['geom'])!=(item['body'],item['geom']):raise ValueError('Collider identity changed')
        a=tree.find(f'.//geom[@name="{item["geom"]}"]');b=reference.find(f'.//geom[@name="{prior["geom"]}"]')
        if a.get('type')!=b.get('type'):raise ValueError('Collider type changed')
        for key in ('size','mesh','mass','density')+(('pos','quat')if item['body']!='world'else()):
            if b.get(key)is not None:a.set(key,b.get(key))
    for name,pose in config['room_body_poses'].items():
        a=tree.find(f'.//body[@name="{name}"]')
        if name in source:
            b=reference.find(f'.//body[@name="{name}"]')
            for key in ('pos','quat'):
                if b.get(key)is not None:a.set(key,b.get(key))
        else:
            a.set('pos',' '.join(format(x,'.17g')for x in pose[:3]));a.set('quat',' '.join(format(x,'.17g')for x in pose[3:]))
    a=tree.find(f'.//geom[@name="{manifest["payload"]["geom"]}"]');b=reference.find(f'.//geom[@name="{staging["payload"]["geom"]}"]')
    for key in ('size','mass','condim','friction'):a.set(key,b.get(key))
    tree.write(out/'run/scene.xml',encoding='unicode')
    for body in manifest['room_bodies']:body['initial_xyz']=config['room_body_poses'][body['body']][:3]
    (out/'run/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    apply(out/'run',4.)
    manifest=json.loads((out/'run/manifest.json').read_text())
    manifest['sha256'][str(out/'run/scene.xml')]=sha(out/'run/scene.xml')
    manifest['sha256'][str(usd)]=sha(usd)
    (out/'run/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    boxes=copy.deepcopy(staged['boxes'])
    for body,pose in config['room_body_poses'].items():
        if not body.startswith('room_Boxes__')or body in source:continue
        roll,pitch,yaw=Rotation.from_quat(np.asarray(pose)[3:],scalar_first=True).as_euler('xyz',degrees=True)
        if abs(pitch)>1e-7:raise ValueError('Box metadata requires zero pitch')
        boxes[body.removeprefix('room_Boxes__')].update(xyz_m=pose[:3],yaw_deg=float(yaw),roll_deg=float(roll))
    if len(boxes)!=38:raise ValueError('Expected complete38-box metadata')
    report=dict(schema='heading_cart_layout/v1',source_usd=staged['source_usd'],scene=str(usd),
        motion=staged['motion'],motion_sha256=staged['motion_sha256'],boxes=boxes,payload=manifest['payload'],
        source_support_bodies=manifest['source_support_bodies'],source_support_top_m=staged['source_support_top_m'],
        destination_bodies=manifest['destination_bodies'],destination_support_top_m=config['destination_support_top_m'],
        destination_xy_m=config['goal_proposal']['payload_center_world_m'][:2],destination_support=config['destination_support'],
        room_staging=manifest['room_staging'],stage_room_after_preparation=manifest['stage_room_after_preparation'],
        room_reference_base_xyyaw=manifest['room_reference_base_xyyaw'],narrow_destination_support=True,
        depth_frame=dict(origin_world_m=config['cart_origin_world_m'],yaw_world_rad=config['cart_inward_yaw_rad']),
        contact_material=manifest['contact_material'],goal_proposal=config['goal_proposal'],
        configuration=dict(proposal_path=config['proposal_path'],proposal_sha256=config['proposal_sha256'],builder_contact_scope='mu4_only'),
        payload_mass_override=dict(reviewed_proposal_mass_kg=.1,actual_mass_kg=payload_mass,
            changed_body=manifest['payload']['body'] if payload_mass!=.1 else None,
            scope='Mass-only relaxed SIM variant; proposal geometry unchanged, dynamics unverified'),
        geometry_checks=dict(source_poses_unchanged=True,original_room_masses_and_collider_shapes_preserved=payload_mass==.1,
            original_room_masses_preserved_except_payload=True,all_original_collider_shapes_preserved=True,
            all39_room_bodies_free=True,static_cart_initial_placement_only=True,source_only_startup_alignment=True),
        sha256={str(usd):sha(usd),str(out/'run/scene.xml'):sha(out/'run/scene.xml'),config['proposal_path']:config['proposal_sha256'],str(Path(__file__).resolve()):sha(__file__)},
        physics_steps=0,hardware_executed=False,
        limits='Fresh authoring/native compilation only. Partial footprint and COM margins are recorded in destination_support. No dynamics, grip retention, stack settling or release evidence. Trial runner owns palm torsion, pair friction timing and impratio.')
    (out/'layout.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'goal_proposal.json').write_text(json.dumps(config['goal_proposal'],indent=2)+'\n')
    (out/'reviewed_proposal.json').write_text(json.dumps(config,indent=2)+'\n')
    if sha(config['scene'])!=config['hashes'][config['scene']]:raise RuntimeError('Reviewed source USD was modified')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--proposal',type=Path,required=True);p.add_argument('--plan',type=Path,default=P/'ready_plan43')
    p.add_argument('--payload-mass',type=float,choices=[.01,.1],default=.1)
    a=p.parse_args();r=build(a.out,a.proposal,a.plan,payload_mass=a.payload_mass)
    print(json.dumps({k:r[k]for k in ('scene','destination_support_top_m','room_staging','geometry_checks','physics_steps')},indent=2))
