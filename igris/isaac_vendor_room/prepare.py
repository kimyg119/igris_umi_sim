"""Compose room collision geometry into the unchanged vendor MuJoCo robot.

Run with a Python containing pxr, numpy and PyYAML. Original USD/MJCF stay read-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / 'assets/room_real2sim/room_corner_igris.usd'
VENDOR = ROOT / 'igris/sim_local/igris_c_v2_rcv_hand.xml'


def numbers(values):
    return ' '.join(format(float(v), '.17g') for v in values)


def pose(matrix):
    transform = Gf.Transform(matrix)
    scale = np.asarray(transform.GetScale())
    rotation = transform.GetRotation().GetQuat()
    quat = [rotation.GetReal(), *rotation.GetImaginary()]
    # Gf may represent a rotation with negative scales. Our scene uses positive,
    # orthogonal scales; fail if an input would need a mesh/shear approximation.
    if np.any(scale <= 0) or not np.allclose(np.asarray(matrix)[:3,:3] @ np.asarray(matrix)[:3,:3].T,
                                            np.diag(scale**2), atol=1e-6):
        raise ValueError('Unsupported reflected/sheared room collider')
    return np.asarray(transform.GetTranslation()), quat, scale


def prepare(out, scene=SCENE, vendor=VENDOR, dense=False, compliant_hands=False):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stage = Usd.Stage.Open(str(scene))
    if UsdGeom.GetStageMetersPerUnit(stage) != 1 or UsdGeom.GetStageUpAxis(stage) != 'Z':
        raise ValueError('Room must use metres and Z up')
    cache = UsdGeom.XformCache()
    root = ET.parse(vendor).getroot()
    # Resolve original container paths to host paths. Both namespaces are mounted
    # at the same absolute paths in the private simulator container.
    for element in root.iter():
        for key, value in list(element.attrib.items()):
            for prefix, target in [('/robros/configs/', ROOT/'igris/configs'), ('/igris/', ROOT/'igris')]:
                if value.startswith(prefix):
                    element.set(key, str(target/value[len(prefix):]))
                    break
    hand_metadata=None
    if compliant_hands:
        from hands import install
        ET.ElementTree(root).write(out/'manufacturer.xml',encoding='unicode')
        hand_metadata=install(root,out)
    ET.ElementTree(root).write(out/'vendor.xml',encoding='unicode')
    if not dense:root.find('option').set('jacobian','sparse')
    # User-authorized environment-only contact. Bit 2 is unused by the vendor:
    # it leaves robot self-contact and the original ground interaction unchanged.
    hand_geoms=[]
    for side in ('l','r'):
        hand=root.find(f".//body[@name='{side}_hand_rcv']")
        if hand is None:raise ValueError(f'Missing {side} RCV hand')
        for geom in hand.iter('geom'):
            if geom.get('class')!='visual':raise ValueError('Unexpected hand collision baseline')
            geom.set('contype','2');geom.set('conaffinity','0')
            hand_geoms.append(geom.get('name'))
    world = root.find('worldbody')
    bodies = {}
    manifest = {'physics_backend': 'vendor MuJoCo 3.2.7', 'isaac_role': 'rendering and visual sensors',
                'newton_solver_active': False, 'hardware_executed': False,
                'jacobian': 'vendor dense' if dense else 'sparse (user-authorized computation representation)',
                'hand_environment_contact': {'geom_names':hand_geoms,'hand_contype':2,'hand_conaffinity':0,'room_conaffinity':3},
                'source_usd': str(scene), 'source_mjcf': str(vendor),
                'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (scene,vendor)},
                'room_bodies': [], 'room_colliders': [],
                'floor': 'Original vendor plane retained; overlapping USD ground collider omitted',
                'material_conversion': 'USD dynamic friction -> MuJoCo sliding friction; torsion/rolling use MuJoCo defaults'}
    if hand_metadata:manifest['compliant_hands']=hand_metadata
    key_q = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith('/World/IGRIS_C') or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        api = UsdPhysics.RigidBodyAPI(prim)
        if api.GetKinematicEnabledAttr().Get():
            raise ValueError(f'Unsupported kinematic room body: {path}')
        pos, quat, scale = pose(cache.GetLocalToWorldTransform(prim))
        if not np.allclose(scale, 1):
            raise ValueError(f'Scaled rigid body: {path}')
        name = 'room_' + path.removeprefix('/World/').replace('/', '__')
        node = ET.SubElement(world, 'body', name=name, pos=numbers(pos), quat=numbers(quat))
        ET.SubElement(node, 'freejoint', name=name+'_free')
        mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
        if not mass or mass <= 0:
            raise ValueError(f'Missing positive room mass: {path}')
        bodies[path] = (node, cache.GetLocalToWorldTransform(prim), float(mass), [])
        manifest['room_bodies'].append({'usd_path':path,'body':name,'mass_kg':float(mass),'initial_xyz':pos.tolist()})
        key_q.extend([*pos,*quat])
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith('/World/IGRIS_C') or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        if path == '/World/Room/GroundCollider_Collider':
            continue
        ancestor = prim
        while ancestor and str(ancestor.GetPath()) not in bodies:
            ancestor = ancestor.GetParent()
        owner = bodies.get(str(ancestor.GetPath())) if ancestor else None
        transform = cache.GetLocalToWorldTransform(prim)
        if owner:
            transform = transform * owner[1].GetInverse()
        pos, quat, scale = pose(transform)
        attrs = {'name':'room_geom_'+str(len(manifest['room_colliders'])), 'pos':numbers(pos),
                 'quat':numbers(quat),'condim':'3','contype':'1','conaffinity':'3'}
        if prim.IsA(UsdGeom.Cube):
            size = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) * scale / 2
            attrs.update(type='box', size=numbers(size))
            volume = float(8*np.prod(size))
        elif prim.IsA(UsdGeom.Cylinder):
            cylinder = UsdGeom.Cylinder(prim)
            axis = str(cylinder.GetAxisAttr().Get())
            axis_index = 'XYZ'.index(axis)
            radial = np.delete(scale,axis_index)
            if not np.allclose(radial,radial[0]):
                raise ValueError(f'Elliptic cylinder: {path}')
            radius = float(cylinder.GetRadiusAttr().Get())*radial[0]
            half = float(cylinder.GetHeightAttr().Get())*scale[axis_index]/2
            rotation = Gf.Rotation(Gf.Vec3d(0,0,1), Gf.Vec3d(*(np.eye(3)[axis_index])))
            matrix = Gf.Matrix4d().SetRotate(rotation) * Gf.Matrix4d().SetRotate(Gf.Quatd(quat[0],Gf.Vec3d(*quat[1:])))
            q = Gf.Transform(matrix).GetRotation().GetQuat()
            attrs.update(type='cylinder',size=numbers([radius,half]),quat=numbers([q.GetReal(),*q.GetImaginary()]))
            volume = float(np.pi*radius**2*half*2)
        else:
            raise ValueError(f'Unsupported collider {prim.GetTypeName()}: {path}')
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')[0]
        mu = UsdPhysics.MaterialAPI(material.GetPrim()).GetDynamicFrictionAttr().Get() if material else None
        attrs['friction'] = numbers([float(mu) if mu is not None else .5, .005, .0001])
        geom = ET.SubElement(owner[0] if owner else world,'geom',attrs)
        if owner:
            owner[3].append((geom,volume))
        manifest['room_colliders'].append({'usd_path':path,'geom':attrs['name'],'body':owner[0].get('name') if owner else 'world'})
    for node, _, mass, geoms in bodies.values():
        total = sum(v for _,v in geoms)
        if not total:
            raise ValueError('Rigid body without colliders')
        for geom,volume in geoms:
            geom.set('mass',numbers([mass*volume/total]))
    # Existing robot coordinates and all original actuator/constraint settings stay intact.
    for key in root.findall('./keyframe/key'):
        key.set('qpos',key.get('qpos','')+' '+numbers(key_q))
        if key.get('qvel'):
            key.set('qvel',key.get('qvel')+' '+numbers([0]*6*len(bodies)))
    ET.indent(root)
    ET.ElementTree(root).write(out/'scene.xml',encoding='unicode')
    config = yaml.safe_load((ROOT/'igris/sim_local/robot.yaml').read_text())
    config['igris_c']['description']['mujoco']['robot_model_path'] = str(out/'scene.xml')
    config['igris_c']['description']['mujoco']['fixed'] = 'false'
    config['igris_c']['network']['cyclonedds']['uri_file'] = 'file://'+str(out/'cyclonedds.xml')
    (out/'robot.yaml').write_text(yaml.safe_dump(config))
    (out/'cyclonedds.xml').write_text('<CycloneDDS><Domain id="any"><General><Interfaces><NetworkInterface name="lo" multicast="false"/></Interfaces><AllowMulticast>false</AllowMulticast></General><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>100</MaxAutoParticipantIndex><Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(out),'dynamic_room_bodies':len(bodies),'room_colliders':len(manifest['room_colliders'])}))
    return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--dense',action='store_true',help='Use the original dense representation for comparison')
    args=parser.parse_args()
    prepare(args.out,dense=args.dense)
