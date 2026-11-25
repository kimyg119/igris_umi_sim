"""Reposition the existing loads without changing any support-relative box pose."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

def rotation(degrees):
    a=math.radians(degrees)
    return np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])


def pose(prim,position,yaw=0):
    xf=UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    # A single matrix also overrides imported double/float quaternion differences.
    mat=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0,0,1),yaw))
    mat.SetRow(3,Gf.Vec4d(*position,1))
    xf.MakeMatrixXform().Set(mat)


def robot_asset(source,out):
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    src=Usd.Stage.Open(str(source))
    src.SetEditTarget(src.GetSessionLayer())
    for p in list(src.Traverse()):
        if p.IsInstance():
            p.SetInstanceable(False)
    src=Usd.Stage.Open(src.Flatten())
    # Keep the native colours, replace built-in MDL dependencies with portable USD shaders.
    for p in [p for p in src.Traverse() if p.IsA(UsdShade.Material)]:
        shaders=[c for c in p.GetChildren() if c.IsA(UsdShade.Shader)]
        color=Gf.Vec3f(.72,.72,.72)
        for sh in shaders:
            value=sh.GetAttribute('inputs:diffuse_color_constant').Get()
            if value is not None:
                color=Gf.Vec3f(*value)
            src.RemovePrim(sh.GetPath())
        for prop in p.GetProperties():
            if prop.GetName().startswith('outputs:'):
                p.RemoveProperty(prop.GetName())
        sh=UsdShade.Shader.Define(src,str(p.GetPath())+'/PreviewSurface')
        sh.CreateIdAttr('UsdPreviewSurface')
        sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(color)
        sh.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.48)
        sh.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(.15)
        UsdShade.Material(p).CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),'surface')
    for p in src.Traverse():
        if p.IsA(UsdPhysics.RevoluteJoint):
            name=p.GetName()
            leg=any(word in name for word in ('hip','knee','ankle','waist'))
            drive=UsdPhysics.DriveAPI.Apply(p,'angular')
            drive.CreateStiffnessAttr(20000 if leg else 300)
            drive.CreateDampingAttr(1000 if leg else 30)
            drive.CreateTargetPositionAttr(0)
    bb=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render']).ComputeWorldBound(src.GetDefaultPrim()).ComputeAlignedRange()
    ground_offset=-bb.GetMin()[2]
    flattened=src.Flatten()
    dst=Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(dst,1)
    UsdGeom.SetStageUpAxis(dst,'Z')
    root=UsdGeom.Xform.Define(dst,'/IGRIS_C').GetPrim()
    dst.SetDefaultPrim(root)
    model=UsdGeom.Xform.Define(dst,'/IGRIS_C/Model').GetPrim()
    model.GetReferences().AddReference(flattened.identifier,src.GetDefaultPrim().GetPath())
    pose(model,(0,0,float(ground_offset)))
    root.SetCustomData({'floor_origin':[0.,0.,0.],'pelvis_height_m':float(ground_offset),
                        'free_base':True,'joint_drive_purpose':'neutral standing pose, not a locomotion controller',
                        'source_sha256':source_hash})
    dst.Flatten().Export(str(out/'igris_c_rcv_hand.usd'))
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    return float(ground_offset)


def compose(out,source):
    old=json.loads((out/'layout.json').read_text())
    # The original layout remains the immutable reference for all 38 carton poses.
    tip=np.array([math.sqrt(2)*.20,0.])
    specs={
        'pallet':{'yaw_deg':-135.,'size_m':[1.10,1.10],'touch_corner_local_m':[-.55,-.55]},
        'rolltainer':{'yaw_deg':-45.,'size_m':[.92,.70],'touch_corner_local_m':[.46,-.35]},
    }
    stage=Usd.Stage.Open(str(out/'room_interactive.usd'))
    cache=UsdGeom.XformCache()
    relative={}
    for box in old['boxes']:
        support='/World/Pallet' if box['support']=='pallet' else '/World/Rolltainer'
        relative[box['name']]=cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Boxes/'+box['name']))*cache.GetLocalToWorldTransform(stage.GetPrimAtPath(support)).GetInverse()
    stage.SetEditTarget(stage.GetSessionLayer())
    for name,spec in specs.items():
        rot=rotation(spec['yaw_deg'])
        center=tip-rot@spec['touch_corner_local_m']
        spec['position_m']=[*center.tolist(),0.]
        group='/World/Pallet' if name=='pallet' else '/World/Rolltainer'
        pose(stage.GetPrimAtPath(group),spec['position_m'],spec['yaw_deg'])
        for b in old['boxes']:
            if b['support']!=name:
                continue
            matrix=relative[b['name']]*UsdGeom.Xformable(stage.GetPrimAtPath(group)).GetLocalTransformation()
            UsdGeom.Xformable(stage.GetPrimAtPath('/World/Boxes/'+b['name'])).MakeMatrixXform().Set(matrix)
    height=robot_asset(source,out)
    robot=UsdGeom.Xform.Define(stage,'/World/IGRIS_C').GetPrim()
    robot.GetReferences().AddReference('igris_c_rcv_hand.usd')
    pose(robot,(0.,0.,0.))
    # Move only the contextual room so its wall does not intersect the new V layout.
    room=UsdGeom.Xform.Define(stage,'/World/Room').GetPrim()
    pose(room,(1.50,0.,0.),-90)
    for name,eye,target,focal in (
        ('Overview',(-3.4,-.15,2.35),(.22,-.18,.72),35),
        ('Overhead',(.10,0,3.8),(.10,0,0),31),
        ('RobotFront',(1.8,-2.2,1.95),(0,0,.8),38),
        ('RobotBack',(-2.7,-2.6,2.2),(0,0,.75),33)):
        cam=UsdGeom.Camera.Define(stage,'/World/Cameras/'+name)
        up=Gf.Vec3d(1,0,0) if name=='Overhead' else Gf.Vec3d(0,0,1)
        matrix=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),up).GetInverse()
        UsdGeom.Xformable(cam).MakeMatrixXform().Set(matrix)
        cam.CreateFocalLengthAttr(focal)
        cam.CreateHorizontalApertureAttr(36)
        cam.CreateClippingRangeAttr(Gf.Vec2f(.01,100))
    data=dict(stage.GetRootLayer().customLayerData)
    data['cameraSettings']={'Perspective':{'position':Gf.Vec3d(-3.4,-.15,2.35),'target':Gf.Vec3d(.22,-.18,.72)}}
    stage.GetSessionLayer().customLayerData=data
    stage.GetDefaultPrim().SetCustomDataByKey('layout','corner_contact_robot_origin_20cm')
    # Export a flattened scene; geometry remains referenced internally by USD prototypes.
    # The robot is kept as a reusable companion USD as well.
    result=Usd.Stage.Open(stage.Flatten())
    # The user wants this rolltainer parked: static chassis and wheels, free cartons.
    cart=result.GetPrimAtPath('/World/Rolltainer')
    for prim in list(Usd.PrimRange(cart)):
        if prim.IsA(UsdPhysics.Joint):
            prim.SetActive(False)
        elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            prim.RemoveAPI(UsdPhysics.MassAPI)
    cart.SetCustomDataByKey('fixed_to_floor',True)
    for prim in result.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName()==Sdf.ValueTypeNames.Asset:
                value=attr.Get()
                if value and '/textures/' in value.path:
                    attr.Set(Sdf.AssetPath('textures/'+Path(value.path).name))
    result.GetRootLayer().Export(str(out/'room_corner_igris.usd'))
    report={'robot_floor_origin_m':[0.,0.,0.],'robot_pelvis_height_m':height,
        'robot_forward_axis':'+X','face_distance_m':.20,'contact_point_xy_m':tip.tolist(),
        'front_faces_angle_deg':90.,'supports':specs,
        'box_count':len(old['boxes']),'rolltainer_fixed':True,'source_layout':'layout.json',
        'distance_definition':'horizontal perpendicular distance from floor origin to finite robot-facing support edge'}
    (out/'corner_layout.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot-usd',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    compose(args.out.resolve(),args.robot_usd.resolve())
