import importlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
PROPOSAL=ROOT/'outputs/hlik_transport_20260918_2005/task2_straight_cart01/offset_landing55_cart/proposal.json'

def module():
    try:return importlib.import_module('heading_cart_layout')
    except ModuleNotFoundError:pytest.fail('heading_cart_layout feature is not implemented')

def test_reviewed_proposal_has_fixed_six_destinations_and_explicit_partial_support():
    c=module().load_proposal(PROPOSAL)
    assert len(c['destination_bodies'])==6 and len(c['room_body_poses'])==39
    assert c['destination_support']['full_footprint_supported'] is False
    assert c['destination_support']['minimum_com_axis_margins_m']==[.035,.05]
    assert c['destination_support']['nominal_footprint_overlap_fraction']==pytest.approx(.4852850978517849)
    assert c['goal_proposal']['payload_center_world_m'][2]==pytest.approx(.7579000124797224)

def test_proposal_rejects_changed_source_file_and_overlapping_body_partition(tmp_path):
    m=module();raw=json.loads(PROPOSAL.read_text());changed=tmp_path/'scene.usdc';changed.write_bytes(b'wrong scene')
    raw['scene']=str(changed);path=tmp_path/'bad.json';path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='hash'):m.load_proposal(path)
    raw=json.loads(PROPOSAL.read_text());raw['destination_bodies'][0]=raw['payload']['body'];path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='partition'):m.load_proposal(path)

def test_proposal_rejects_geometry_contacts_or_unqualified_ik(tmp_path):
    m=module();raw=json.loads(PROPOSAL.read_text());raw['proof']['route_and_arrival']['nominal_robot_nonpayload_contact_records']=1
    path=tmp_path/'bad.json';path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='geometry'):m.load_proposal(path)
    raw=json.loads(PROPOSAL.read_text());raw['proof']['nominal_ik']['strict_failures']=1;path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='IK'):m.load_proposal(path)

def test_fresh_build_preserves_source_and_payload_and_applies_only_mu4(tmp_path):
    m=module();out=tmp_path/'fresh';report=m.build(out,PROPOSAL)
    manifest=json.loads((out/'run/manifest.json').read_text());staging=json.loads((out/'staging/run/manifest.json').read_text())
    assert len(report['boxes'])==38
    assert manifest['room_staging']['scope']=='source_only'
    assert manifest['room_staging']['transform_bodies']==manifest['source_support_bodies']+[manifest['payload']['body']]
    assert len(manifest['destination_bodies'])==6 and not manifest['room_staging']['transforms_destination']
    current=ET.parse(out/'run/scene.xml');original=ET.parse(out/'staging/run/scene.xml')
    for name in manifest['source_support_bodies']+[manifest['payload']['body']]:
        a=current.find(f'.//body[@name="{name}"]');b=original.find(f'.//body[@name="{name}"]')
        assert a.get('pos')==b.get('pos') and a.get('quat')==b.get('quat')
    name=manifest['payload']['geom'];a=current.find(f'.//geom[@name="{name}"]');b=original.find(f'.//geom[@name="{staging["payload"]["geom"]}"]')
    assert a.get('size')==b.get('size') and a.get('mass')==b.get('mass') and a.get('condim')==b.get('condim')
    pairs=current.findall('.//contact/pair');assert len(pairs)==76
    assert all(np.fromstring(p.get('friction'),sep=' ')[2]==.005 for p in pairs)
    assert all(p.get('solreffriction') is None for p in pairs)
    audit=json.loads((out/'run/material_audit.json').read_text());assert audit['mu']==4 and audit['scene_sha256']==m.sha(out/'run/scene.xml')
    assert not (out/'run/palm_torsion_audit.json').exists()
    assert not (out/'run/pair_friction_solver_audit.json').exists()
    assert manifest['sha256'][str(out/'run/scene.xml')]==m.sha(out/'run/scene.xml')
    with pytest.raises(FileExistsError):m.build(out,PROPOSAL)


def test_explicit_light_payload_preserves_geometry_and_all_other_room_masses(tmp_path):
    from pxr import Usd,UsdPhysics
    m=module();out=tmp_path/'light';before=m.sha(m.load_proposal(PROPOSAL)['scene'])
    report=m.build(out,PROPOSAL,payload_mass=.01)
    manifest=json.loads((out/'run/manifest.json').read_text())
    assert manifest['payload']['mass_kg']==.01
    scene=ET.parse(out/'run/scene.xml');g=scene.find(f'.//geom[@name="{manifest["payload"]["geom"]}"]')
    assert float(g.get('mass'))==pytest.approx(.01)
    assert np.fromstring(g.get('size'),sep=' ').tolist()==[.075,.075,.075]
    usd=Usd.Stage.Open(report['scene']);reference=Usd.Stage.Open(m.load_proposal(PROPOSAL)['scene'])
    for b in manifest['room_bodies']:
        mass=UsdPhysics.MassAPI(usd.GetPrimAtPath(b['usd_path'])).GetMassAttr().Get()
        expected=.01 if b['body']==manifest['payload']['body'] else UsdPhysics.MassAPI(reference.GetPrimAtPath(b['usd_path'])).GetMassAttr().Get()
        assert mass==pytest.approx(expected)
    assert m.sha(m.load_proposal(PROPOSAL)['scene'])==before
    assert report['payload_mass_override']['actual_mass_kg']==.01
