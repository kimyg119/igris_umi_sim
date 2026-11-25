"""Explicit easier-load profile must preserve the original grasp requirements."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from pxr import Usd,UsdPhysics
from grasp_feedback import GraspFeedback
from box_replay import layout,PAYLOAD


def test_relaxed_grasp_still_requires_both_load_bearing_palms():
    f=GraspFeedback(.25,1.2,target=3.,tip_min_force=0.)
    assert abs(f.required-1.5328125)<1e-10
    for _ in range(90):_,ready=f.update([3.,0.,0.,0.],1/30)
    assert not ready
    for _ in range(90):_,ready=f.update([3.,0.,3.,0.],1/30)
    assert ready
    for _ in range(90):_,ready=f.update([3.,0.,.1,0.],1/30)
    assert not ready
    old=GraspFeedback(1.,.6)
    for _ in range(90):_,ready=old.update([14.,0.,14.,0.],1/30)
    assert not ready


def test_reject_invalid_force_configuration():
    import pytest
    for options in [dict(tip_min_force=-1.),dict(tip_min_force=float('nan')),dict(target=float('nan')),dict(target=1.)]:
        with pytest.raises(ValueError):GraspFeedback(.25,1.2,**options)


def test_payload_mass_scales_inertia_and_leaves_robot_unchanged(tmp_path):
    def model(out):
        tree=ET.parse(out/'run/scene.xml')
        for el in tree.iter():
            if el.get('kv')=='0' and 'dampratio' in el.attrib:el.attrib.pop('kv')
        return mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(),encoding='unicode'))
    original=tmp_path/'original';easy=tmp_path/'easy'
    layout(original,compliant_hands=True,narrow_source_support=True)
    layout(easy,compliant_hands=True,narrow_source_support=True,payload_mass=.25)
    a,b=model(original),model(easy)
    manifest=json.loads((easy/'run/manifest.json').read_text());idx=b.body(manifest['payload']['body']).id
    assert manifest['payload']['mass_kg']==.25
    assert abs(b.body_mass[idx]-.25)<1e-9
    np.testing.assert_allclose(b.body_inertia[idx],a.body_inertia[idx]*.25,rtol=1e-12)
    keep=np.arange(a.nbody)!=idx
    np.testing.assert_array_equal(a.body_mass[keep],b.body_mass[keep])
    np.testing.assert_array_equal(a.body_inertia[keep],b.body_inertia[keep])
    np.testing.assert_array_equal(a.geom_size,b.geom_size)
    stage=Usd.Stage.Open(str(easy/'room_replay.usdc'))
    assert UsdPhysics.MassAPI(stage.GetPrimAtPath(PAYLOAD)).GetMassAttr().Get()==.25
