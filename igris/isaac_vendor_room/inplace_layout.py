"""Fresh original-asset near receiver for the reviewed in-place task1 proposal.

No runtime/global scene is replaced. Source and destination must receive the
same startup planar transform; parked bodies and static cart remain fixed.
The new staging scope requires explicit run.py integration before execution.
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
from rolltainer_staging import alignment_transform

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'outputs/hlik_transport_20260918_2005'
DEFAULT_PROPOSAL=P/'task1_near_from68_ik01/proposal.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_authored_mass(usd_value,native_value):
    # USD MassAPI stores float32; native compilation can introduce a final
    # double-precision rounding bit. Do not accept even one authored float ULP.
    return bool(np.isfinite([usd_value,native_value]).all()
                and np.float32(usd_value)==np.float32(native_value))


def near_staging_contract(manifest):
    room=[b['body']for b in manifest['room_bodies']]
    source=list(manifest['source_support_bodies'])+[manifest['payload']['body']]
    destination=list(manifest['destination_bodies']);selected=source+destination
    if (len(set(room))!=len(room) or len(set(selected))!=len(selected) or not set(selected)<=set(room)):
        raise ValueError('Near staging requires distinct existing source and destination free bodies')
    return dict(scope='source_and_destination',transform_bodies=selected,
        fixed_bodies=[b for b in room if b not in selected],transforms_destination=True,
        park_before_ready='all_room_free_bodies',ready_pose_for_unselected_bodies='model_qpos0',
        reset_velocity_for='all_room_free_bodies',static_cart_transform='unchanged',native_selector_required=True)


def align_near_goal(target_scene_xyz,reference_xyyaw,actual_xyyaw):
    target=np.asarray(target_scene_xyz,dtype=float)
    if target.shape!=(3,)or not np.isfinite(target).all():raise ValueError('Finite near target required')
    return (alignment_transform(reference_xyyaw,actual_xyyaw)@np.r_[target,1.])[:3].tolist()


def load_layout_proposal(path):
    path=Path(path).resolve();proposal=json.loads(path.read_text());layout=proposal['modified_layout']
    pose_file=ROOT/layout['all_34_non_source_pose_file']
    lines=pose_file.read_text().splitlines();count=int(lines[0]);poses={}
    inverse=np.linalg.inv(np.asarray(proposal['startup_frame']))
    for line in lines[1:]:
        fields=line.split();name=fields[0];pose=np.array(fields[1:],float)
        if name in poses:raise ValueError('Duplicate proposed room pose')
        m=inverse@pose_matrix(pose)
        poses[name]=np.r_[m[:3,3],Rotation.from_matrix(m[:3,:3]).as_quat(scalar_first=True)].tolist()
    if count!=34 or len(poses)!=count:raise ValueError('Expected all34 non-source free-body poses')
    destinations=[a['body']for a in layout['assets']]
    for asset in layout['assets']:
        np.testing.assert_allclose(pose_matrix(poses[asset['body']]),pose_matrix(asset['nominal_scene_pose_wxyz']),atol=1e-12,rtol=0)
    return dict(proposal_path=str(path),pose_file=str(pose_file),nominal_room_poses=poses,
        destination_bodies=destinations,target_scene_xyz=proposal['target_scene_xyz'],support_top_m=layout['support_top_m'],
        assets=layout['assets'],support=dict(full_footprint_supported=False,intentional_footprint_overhang=True,
            com_axis_margins_m=layout['nominal_com_axis_margins_m'],
            planned_footprint_overlap_fraction=layout['measured_planned_footprint_overlap_fraction'],
            planned_payload_overhang_negative_axis_m=layout['projected_overhang_negative_axis_m'],
            planned_payload_overhang_positive_axis_m=layout['projected_overhang_positive_axis_m'],
            middle_box_depth_overhang_m=layout['middle_box_depth_overhang_m']),
        physics_profile=proposal['physics_profile'],proposal_sha256=sha(path),pose_file_sha256=sha(pose_file))


def build(out,proposal=DEFAULT_PROPOSAL,plan=P/'ready_plan43'):
    from pxr import Gf,Usd,UsdGeom,UsdPhysics
    from box_replay import layout
    from prepare import prepare
    from contact_material import apply
    from contact_solver import apply_pair_friction,apply_palm_torsion
    out=Path(out).resolve();plan=Path(plan).resolve()
    config=load_layout_proposal(proposal)
    out.mkdir(parents=True,exist_ok=False)
    staged=layout(out/'staging',compliant_hands=True,motion=plan/'motion_100hz.npz',vendor_box=True,
                  source_base_clearance=.01,narrow_source_support=True,payload_mass=.1)
    original=json.loads((out/'staging/run/manifest.json').read_text())
    usd_source=Path(staged['scene']);source_hash=sha(usd_source)
    stage=Usd.Stage.Open(str(usd_source));stage.SetEditTarget(stage.GetSessionLayer())
    cache=UsdGeom.XformCache();room={b['body']:b['usd_path']for b in original['room_bodies']}
    source=original['source_support_bodies']+[original['payload']['body']]
    if set(config['nominal_room_poses'])!=set(room)-set(source):raise ValueError('Proposed room/source body partition differs')
    preserved={name:np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(room[name]))).copy() for name in source}
    masses={name:float(UsdPhysics.MassAPI(stage.GetPrimAtPath(path)).GetMassAttr().Get())for name,path in room.items()}
    child_transforms={str(p.GetPath()):np.asarray(UsdGeom.Xformable(p).GetLocalTransformation()).copy()
        for name,path in room.items()for p in stage.GetPrimAtPath(path).GetChildren() if p.IsA(UsdGeom.Xformable)}
    cart=np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Rolltainer'))).copy()
    for name,pose in config['nominal_room_poses'].items():
        # USD matrices multiply row vectors; pose_matrix uses column vectors.
        m=Gf.Matrix4d(*pose_matrix(pose).T.flatten().tolist())
        UsdGeom.Xformable(stage.GetPrimAtPath(room[name])).MakeMatrixXform().Set(m)
    updated=UsdGeom.XformCache()
    for name,matrix in preserved.items():
        np.testing.assert_allclose(updated.GetLocalToWorldTransform(stage.GetPrimAtPath(room[name])),matrix,atol=1e-12,rtol=0)
    np.testing.assert_allclose(updated.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Rolltainer')),cart,atol=1e-12,rtol=0)
    for name,path in room.items():
        if float(UsdPhysics.MassAPI(stage.GetPrimAtPath(path)).GetMassAttr().Get())!=masses[name]:raise ValueError('Original room mass changed')
    for path,matrix in child_transforms.items():
        np.testing.assert_array_equal(UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetLocalTransformation(),matrix)
    levels=[]
    for asset in config['assets']:
        path=room[asset['body']];collider=stage.GetPrimAtPath(path+'/Body_Collider')
        cube=UsdGeom.Cube(collider);local=UsdGeom.Xformable(collider).GetLocalTransformation()
        size=float(cube.GetSizeAttr().Get())*np.asarray(Gf.Transform(local).GetScale())
        np.testing.assert_allclose(size,asset['original_size_m'],atol=1e-12,rtol=0)
        if not same_authored_mass(masses[asset['body']],asset['mass_kg']):raise ValueError('Original destination mass differs')
        world=np.asarray(updated.GetLocalToWorldTransform(collider)).T
        extent=abs(world[:3,:3])@np.repeat(float(cube.GetSizeAttr().Get())/2,3)
        low,high=world[:3,3]-extent,world[:3,3]+extent
        levels.append(dict(body=asset['body'],bottom_m=float(low[2]),top_m=float(high[2]),original_size_m=size.tolist(),mass_kg=masses[asset['body']]))
    if abs(levels[-1]['top_m']-.6808)>1e-7:raise ValueError('Near support height differs')
    for below,above in zip(levels,levels[1:]):
        if abs(above['bottom_m']-below['top_m']-.0002)>1e-7:raise ValueError('Near vertical layer gap differs')
    usd=out/'room_replay.usdc';stage.Flatten().Export(str(usd))
    manifest=prepare(out/'run',scene=usd,compliant_hands=True)
    for key in ('payload','source_support_bodies','stage_room_after_preparation','room_reference_base_xyyaw'):
        manifest[key]=copy.deepcopy(original[key])
    manifest['payload']['geom']=next(c['geom']for c in manifest['room_colliders']if c['body']==manifest['payload']['body'])
    manifest['destination_bodies']=config['destination_bodies'];manifest['evaluation_profile']='relaxed_transport'
    manifest['room_staging']=near_staging_contract(manifest)
    manifest['inplace_transfer']=dict(target_scene_xyz=config['target_scene_xyz'],target_requires_startup_alignment=True,
        waist_yaw_start_rad=.7556667619552057,waist_yaw_end_rad=-.6200024167299453,
        clearance_lift_m=.04,waist_arc_duration_s=20.,lowering_duration_s=12.,world_position_lowering=True,
        opening_duration_s=4.,opening_per_hand_m=.055,post_open_hold_s=3.,
        runtime_integration_pending=True,zero_walk_twist=True)
    manifest['destination_support']=config['support']
    tree=ET.parse(out/'run/scene.xml');geom=tree.find(f".//geom[@name='{manifest['payload']['geom']}']")
    old_geom=ET.parse(out/'staging/run/scene.xml').find(f".//geom[@name='{original['payload']['geom']}']")
    # Preserve the exact already-validated native 150 mm payload dimensions;
    # rebaking USD float32 transforms otherwise changes automatic inertia by
    # ~8e-8 relative even though the composed USD geometry is unchanged.
    for field in ('size','condim','friction'):geom.set(field,old_geom.get(field))
    tree.write(out/'run/scene.xml',encoding='unicode')
    (out/'run/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    apply(out/'run',4.)
    apply_pair_friction(out/'run',.001,include_feet=True,hand_time_constant=.005)
    apply_palm_torsion(out/'run',.02)
    manifest=json.loads((out/'run/manifest.json').read_text())
    manifest['sha256'][str(out/'run/scene.xml')]=sha(out/'run/scene.xml')
    manifest['sha256'][str(usd)]=sha(usd)
    (out/'run/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if sha(usd_source)!=source_hash:raise ValueError('Fresh staging source was modified')
    boxes=copy.deepcopy(staged['boxes'])
    for body,pose in config['nominal_room_poses'].items():
        if not body.startswith('room_Boxes__'):continue
        roll,pitch,yaw=Rotation.from_quat(np.asarray(pose)[3:],scalar_first=True).as_euler('xyz',degrees=True)
        if abs(pitch)>1e-7:raise ValueError('Box metadata requires zero pitch')
        boxes[body.removeprefix('room_Boxes__')].update(xyz_m=pose[:3],yaw_deg=float(yaw),roll_deg=float(roll))
    report=dict(schema='inplace_near_layout/v1',scene=str(usd),configuration=config,levels=levels,boxes=boxes,
        payload=manifest['payload'],source_support_bodies=source[:-1],destination_bodies=config['destination_bodies'],
        source_support_top_m=staged['source_support_top_m'],destination_xy_m=config['target_scene_xyz'][:2],
        destination_support_top_m=levels[-1]['top_m'],room_staging=manifest['room_staging'],
        room_reference_base_xyyaw=manifest['room_reference_base_xyyaw'],
        geometry_checks=dict(source_poses_unchanged=True,all_original_room_masses_preserved=True,
            collider_and_visual_local_transforms_unchanged=True,static_cart_unchanged=True,
            all_room_bodies_remain_free=True,vertical_continuity=True),
        contact_material=manifest['contact_material'],pair_friction_solver=manifest['pair_friction_solver'],
        palm_torsion=manifest['palm_torsion'],
        sha256={str(usd):sha(usd),str(out/'run/scene.xml'):sha(out/'run/scene.xml'),str(Path(proposal).resolve()):sha(proposal),
                str(plan/'motion_100hz.npz'):sha(plan/'motion_100hz.npz'),str(Path(__file__).resolve()):sha(__file__)},
        physics_steps=0,runtime_integration_pending=True,
        limits='Static authoring/compilation only. Original four-asset receiver has intentional payload and middle-box overhang; no balance, settling or release evidence. Startup must transform the near receiver and goal together with source. Native feet/legs remain physically free.')
    (out/'layout.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'goal_proposal.json').write_text(json.dumps(manifest['inplace_transfer'],indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--proposal',type=Path,default=DEFAULT_PROPOSAL);p.add_argument('--plan',type=Path,default=P/'ready_plan43')
    a=p.parse_args();r=build(a.out,a.proposal,a.plan)
    print(json.dumps({k:r[k]for k in ('scene','destination_support_top_m','room_staging','geometry_checks','physics_steps')},indent=2))
