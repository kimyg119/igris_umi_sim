"""Check physical scene geometry and immutable source placement, without stepping."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics


@pytest.fixture(scope='module', params=[False, True], ids=['default', 'narrow'])
def built(tmp_path_factory, request):
    from rolltainer_layout import build
    out = tmp_path_factory.mktemp('rolltainer') / 'layout'
    report = build(out, narrow_destination_support=True) if request.param else build(out)
    return out, report, Usd.Stage.Open(str(out / 'room_replay.usdc'))


def test_stacks_are_inside_deck_and_reach_task1_height(built):
    _, report, stage = built
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    cart = stage.GetPrimAtPath('/World/Rolltainer')
    deck = bbox.ComputeRelativeBound(stage.GetPrimAtPath('/World/Rolltainer/Chassis/Deck_Collider'), cart).ComputeAlignedRange()
    for stack in report['stacks']:
        previous_top = deck.GetMax()[2]
        for name in stack['boxes']:
            bounds = bbox.ComputeRelativeBound(stage.GetPrimAtPath('/World/Boxes/' + name + '/Body_Collider'), cart).ComputeAlignedRange()
            lo, hi = np.array(bounds.GetMin()), np.array(bounds.GetMax())
            assert np.all(lo[:2] >= np.array(deck.GetMin())[:2])
            assert np.all(hi[:2] <= np.array(deck.GetMax())[:2])
            assert 0 <= lo[2] - previous_top < .00021
            previous_top = hi[2]
        assert abs(previous_top - .6808) < .002
    assert report['physics_steps'] == 0


def test_source_payload_and_original_dimensions_are_preserved(built):
    out, report, stage = built
    before = Usd.Stage.Open(str(out / 'staging/room_replay.usdc'))
    a, b = UsdGeom.XformCache(), UsdGeom.XformCache()
    for name in report['source_support_bodies'] + [report['payload']['body']]:
        path = '/World/Boxes/' + name.removeprefix('room_Boxes__')
        np.testing.assert_array_equal(a.GetLocalToWorldTransform(before.GetPrimAtPath(path)), b.GetLocalToWorldTransform(stage.GetPrimAtPath(path)))
    baseline = Usd.Stage.Open(report['baseline_scene'])
    for name in report['source_support_bodies'] + [report['payload']['body']]:
        path = '/World/Boxes/' + name.removeprefix('room_Boxes__')
        np.testing.assert_allclose(a.GetLocalToWorldTransform(baseline.GetPrimAtPath(path)), b.GetLocalToWorldTransform(stage.GetPrimAtPath(path)), atol=1e-12)
    for old in before.GetPrimAtPath('/World/Boxes').GetChildren():
        new = stage.GetPrimAtPath(old.GetPath())
        assert UsdPhysics.MassAPI(old).GetMassAttr().Get() == UsdPhysics.MassAPI(new).GetMassAttr().Get()
        np.testing.assert_array_equal(UsdGeom.Xformable(old.GetChild('Body_Collider')).GetLocalTransformation(), UsdGeom.Xformable(new.GetChild('Body_Collider')).GetLocalTransformation())
    for name in report['previous_destination_bodies']:
        path = '/World/Boxes/' + name.removeprefix('room_Boxes__')
        p = np.array(b.GetLocalToWorldTransform(stage.GetPrimAtPath(path)).ExtractTranslation())
        assert np.linalg.norm(p[:2] - [.5, .2]) > .8


def test_goal_payload_has_support_and_goal_frames_agree(built):
    _, report, stage = built
    goal = report['goal_proposal']
    cache = UsdGeom.XformCache()
    cart = stage.GetPrimAtPath('/World/Rolltainer')
    target_cart = np.array(cache.GetLocalToWorldTransform(cart).GetInverse().Transform(Gf.Vec3d(*goal['payload_center_world_m'])))
    top = stage.GetPrimAtPath('/World/Boxes/' + report['stacks'][0]['boxes'][-1] + '/Body_Collider')
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeRelativeBound(top, cart).ComputeAlignedRange()
    support = report['geometry_checks']['payload_support']
    if report.get('narrow_destination_support', False):
        assert not support['full_footprint_supported']
        assert support['intentional_footprint_overhang']
        assert support['nominal_footprint_overlap_fraction'] == pytest.approx(2/3, abs=1e-6)
        np.testing.assert_allclose(support['com_projection_edge_margins_m'], [.09, .05, .09, .05], atol=1e-7)
        np.testing.assert_allclose(sorted(support['nominal_payload_overhang_m']), [0, 0, .025, .025], atol=1e-7)
        assert np.all(target_cart[:2] >= np.array(bounds.GetMin())[:2])
        assert np.all(target_cart[:2] <= np.array(bounds.GetMax())[:2])
    else:
        assert support['full_footprint_supported']
        assert not support['intentional_footprint_overhang']
        assert np.all(target_cart[:2] - .075 >= np.array(bounds.GetMin())[:2])
        assert np.all(target_cart[:2] + .075 <= np.array(bounds.GetMax())[:2])
    assert 0 <= target_cart[2] - .075 - bounds.GetMax()[2] < .00031
    assert np.linalg.norm(np.array(goal['robot_goal_xy_world_m']) - goal['payload_center_world_m'][:2]) < .4
    transform = np.array(goal['initial_base_to_world'])
    reconstructed = transform @ np.r_[goal['robot_goal_xy_initial_base_m'], 0, 1]
    np.testing.assert_allclose(reconstructed[:2], goal['robot_goal_xy_world_m'], atol=1e-10)
    assert len(goal['placement_ee_world']) == 2


def test_native_model_keeps_free_boxes_and_updated_hashes(built):
    import mujoco
    out, report, _ = built
    manifest = json.loads((out / 'run/manifest.json').read_text())
    xml = ET.parse(out / 'run/scene.xml')
    for el in xml.iter():
        if el.get('kv') == '0' and 'dampratio' in el.attrib:
            el.attrib.pop('kv')
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml.getroot(), encoding='unicode'))
    assert model.body_mass[model.body(manifest['payload']['body']).id] == .25
    assert int(model.geom_condim[model.geom(manifest['payload']['geom']).id]) == 4
    for name in manifest['destination_bodies']:
        assert xml.find(f".//body[@name='{name}']/freejoint") is not None
    assert manifest['room_reference_base_xyyaw'] == json.loads((out / 'staging/run/manifest.json').read_text())['room_reference_base_xyyaw']
    assert manifest['stage_room_after_preparation']
    for path, digest in manifest['sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    assert manifest['contact_material']['mu'] == 1.2
    assert len(manifest['destination_bodies']) == 6
    assert manifest['room_staging']['transform_bodies'] == manifest['source_support_bodies'] + [manifest['payload']['body']]
    assert set(manifest['destination_bodies']) <= set(manifest['room_staging']['fixed_bodies'])
    assert set(manifest['parked_previous_destination_bodies']) <= set(manifest['room_staging']['fixed_bodies'])
    pairs = xml.findall('./contact/pair')
    assert len(pairs) == manifest['contact_material']['pair_count']
    assert len({p.get('name') for p in pairs}) == len(pairs)
    assert all(p.get('geom2') == manifest['payload']['geom'] for p in pairs)
    if report.get('narrow_destination_support', False):
        assert manifest['destination_bodies'][:3] == ['room_Boxes__Roll_C3_R1_L1', 'room_Boxes__Small_RightBack_L1', 'room_Boxes__Small_Center_L3']
        assert {'room_Boxes__Roll_C3_R1_L2', 'room_Boxes__Roll_C3_R1_L3'} <= set(manifest['parked_previous_destination_bodies'])
        assert report['stacks'][0]['boxes'] == ['Roll_C3_R1_L1', 'Small_RightBack_L1', 'Small_Center_L3']
        np.testing.assert_allclose([level['top_m']-level['bottom_m'] for level in report['stacks'][0]['levels']], [.23, .10, .18], atol=1e-7)
        assert report['geometry_checks']['payload_support']['full_footprint_supported'] is False


def test_existing_output_is_not_overwritten(built):
    from rolltainer_layout import build
    with pytest.raises(FileExistsError):
        build(built[0])
