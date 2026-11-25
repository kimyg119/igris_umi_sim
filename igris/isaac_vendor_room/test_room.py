"""One runnable check for scene conversion, with rotated cubes and cylinders."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from prepare import prepare, SCENE, VENDOR
from box_replay import layout,load_motion,MOTION,PAYLOAD


def matrix(element):
    xyz=np.fromstring(element.get('pos','0 0 0'),sep=' ')
    q=np.fromstring(element.get('quat','1 0 0 0'),sep=' ')
    m=Gf.Matrix4d().SetRotate(Gf.Quatd(float(q[0]),Gf.Vec3d(*q[1:])))
    return m.SetTranslateOnly(Gf.Vec3d(*xyz))


def check():
    with TemporaryDirectory() as directory:
        out=Path(directory)/'room'
        report=prepare(out,dense=True)
        source=ET.parse(out/'vendor.xml').getroot();room=ET.parse(out/'scene.xml').getroot()
        for tag in ['option','actuator','equality','contact','default']:
            assert ET.canonicalize(ET.tostring(source.find(tag),encoding='unicode'),strip_text=True)==ET.canonicalize(ET.tostring(room.find(tag),encoding='unicode'),strip_text=True),tag
        assert len(report['room_bodies'])==39
        assert len(report['room_colliders'])==214
        assert report['jacobian']=='vendor dense'
        sparse_out=Path(directory)/'sparse'
        prepare(sparse_out)
        sparse=ET.parse(sparse_out/'scene.xml').getroot()
        assert sparse.find('option').get('jacobian')=='sparse'
        sparse.find('option').set('jacobian',room.find('option').get('jacobian'))
        assert ET.canonicalize(ET.tostring(sparse,encoding='unicode'),strip_text=True)==ET.canonicalize(ET.tostring(room,encoding='unicode'),strip_text=True)
        hands=report['hand_environment_contact']['geom_names']
        assert len(hands)==76
        for name in hands:
            a=source.find(f".//geom[@name='{name}']");b=room.find(f".//geom[@name='{name}']")
            assert b.get('contype')=='2' and b.get('conaffinity')=='0'
            assert {k:v for k,v in b.attrib.items() if k not in ('contype','conaffinity')}==a.attrib
        assert source.find(".//body[@name='pelvis']/freejoint") is not None or source.find(".//body[@name='pelvis']/joint[@type='free']") is not None
        usd=Usd.Stage.Open(str(SCENE));cache=UsdGeom.XformCache()
        count={'Cube':0,'Cylinder':0}
        for item in report['room_colliders']:
            geom=room.find(f".//geom[@name='{item['geom']}']")
            assert geom.get('contype')=='1' and geom.get('conaffinity')=='3'
            m=matrix(geom)
            if item['body']!='world':m=m*matrix(room.find(f".//body[@name='{item['body']}']"))
            prim=usd.GetPrimAtPath(item['usd_path']);original=cache.GetLocalToWorldTransform(prim)
            assert np.allclose(m.ExtractTranslation(),original.ExtractTranslation(),atol=1e-7),item
            size=np.fromstring(geom.get('size'),sep=' ')
            if prim.IsA(UsdGeom.Cube):
                original_size=UsdGeom.Cube(prim).GetSizeAttr().Get()/2
                a=np.array([original.Transform(Gf.Vec3d(*(np.array(v)*original_size))) for v in [(1,0,0),(0,1,0),(0,0,1)]])
                b=np.array([m.Transform(Gf.Vec3d(*(np.array(v)*size))) for v in [(1,0,0),(0,1,0),(0,0,1)]])
                assert np.allclose(a,b,atol=1e-7),item
                count['Cube']+=1
            else:
                cylinder=UsdGeom.Cylinder(prim);axis='XYZ'.index(cylinder.GetAxisAttr().Get())
                tip=original.Transform(Gf.Vec3d(*(np.eye(3)[axis]*cylinder.GetHeightAttr().Get()/2)))
                assert np.allclose(tip,m.Transform(Gf.Vec3d(0,0,float(size[1]))),atol=1e-7),item
                count['Cylinder']+=1
        for body in report['room_bodies']:
            node=room.find(f".//body[@name='{body['body']}']")
            assert node.find('freejoint') is not None
            assert abs(sum(float(g.get('mass')) for g in node.findall('geom'))-body['mass_kg'])<1e-8
        try:prepare(out)
        except FileExistsError:pass
        else:raise AssertionError('Must preserve existing experiment outputs')
        print('PASS: original dynamics/options, environment-only hand masks, all 214 collider frames/dimensions, 39 masses/free bodies, no overwrite',count)
        variant=Path(directory)/'replay'
        report=layout(variant)
        times,qs,indices=load_motion(MOTION)
        assert qs.shape==(8061,17) and times[-1]==80.6 and report['q_modified'] is False
        x=ET.parse(variant/'run/scene.xml')
        payload=x.find(f".//body[@name='room_Boxes__{PAYLOAD.rsplit('/',1)[-1]}']/geom")
        assert float(payload.get('mass'))==1.
        assert np.allclose(np.fromstring(payload.get('size'),sep=' '),[.09,.09,.05],atol=1e-8)
        assert report['destination_support_top_m']<report['source_support_top_m']
        assert len(report['boxes'])==38
        # Reject a changed/invalid archive before opening a command transport.
        with np.load(MOTION,allow_pickle=False) as data:broken={k:data[k] for k in data.files}
        broken['q_command_rad']=qs.copy();broken['q_command_rad'][10,0]=np.nan
        invalid=Path(directory)/'invalid.npz';np.savez(invalid,**broken)
        try:load_motion(invalid)
        except ValueError:pass
        else:raise AssertionError('Invalid q must be rejected')
        print('PASS: exact archive, 1 kg room payload, 38-box layout, invalid archive rejection')
        for cube in (False,True):
            compliant=Path(directory)/('compliant_cube' if cube else 'compliant_room')
            report=layout(compliant,compliant_hands=True,**({} if cube else {'vendor_box':False}))
            meta=json.loads((compliant/'run/manifest.json').read_text());xml=ET.parse(compliant/'run/scene.xml')
            geom=xml.find(f".//geom[@name='{meta['payload']['geom']}']")
            np.testing.assert_allclose(2*np.fromstring(geom.get('size'),sep=' '),meta['payload']['size_m'],atol=1e-7)
            stage=Usd.Stage.Open(report['scene']);body=stage.GetPrimAtPath(PAYLOAD)
            cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_])
            for child in ('Body','Body_Collider'):
                bounds=cache.ComputeRelativeBound(stage.GetPrimAtPath(PAYLOAD+'/'+child),body).ComputeAlignedRange()
                np.testing.assert_allclose(bounds.GetMax(),np.asarray(meta['payload']['size_m'])/2,atol=1e-7)
                np.testing.assert_allclose(bounds.GetMin(),-np.asarray(meta['payload']['size_m'])/2,atol=1e-7)
            assert geom.get('condim')==('4' if cube else '3') and float(geom.get('mass'))==1
            assert meta['stage_room_after_preparation'] and report['robot_physics_modified']
            assert len(meta['room_bodies'])==39 and len(report['boxes'])==38
            assert meta['compliant_hands']['non_hand_manufacturer_xml_unchanged']
            assert xml.find(".//body[@name='l_hand_mcp']/joint") is not None
        print('PASS: compliant hands, original/cube payload dimensions, staged room, preserved non-hand model')


if __name__=='__main__':check()
