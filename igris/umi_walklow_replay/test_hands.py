import xml.etree.ElementTree as ET

import numpy as np
import pytest

from igris.umi_walklow_replay import hands
from igris.umi_walklow_replay.kinematics import VENDOR_MJCF


def geoms(root):
    return {g.get('name'): (np.fromstring(g.get('pos'), sep=' '), np.fromstring(g.get('quat'), sep=' '))
            for side in ('l', 'r') for g in root.find(f".//body[@name='{side}_hand_rcv']").findall('geom')}


def test_vendor_pose_is_reproduced_exactly():
    root = ET.parse(VENDOR_MJCF).getroot()
    original = geoms(root)
    report = hands.apply_mjcf(root, {side: dict(hands.VENDOR_BAKED) for side in hands.SIDES})
    assert len(report['left']['moved_geoms']) == len(report['right']['moved_geoms']) == 16
    for name, (position, quat) in geoms(root).items():
        assert np.allclose(position, original[name][0], atol=1e-6)
        assert min(np.abs(quat - original[name][1]).max(), np.abs(quat + original[name][1]).max()) < 1e-5


def test_only_posed_links_move():
    root = ET.parse(VENDOR_MJCF).getroot()
    original = geoms(root)
    pose = {side: {'mcp': .5, 'thumb_cmc': -.5, 'thumb_ip': -.3} for side in hands.SIDES}
    report = hands.apply_mjcf(root, pose)
    moved = set(report['left']['moved_geoms']) | set(report['right']['moved_geoms'])
    for name, (position, _) in geoms(root).items():
        assert (np.linalg.norm(position - original[name][0]) > 1e-9) == (name in moved) or name in moved
        if name not in moved:
            assert np.array_equal(position, original[name][0])


def test_symmetry_and_limits():
    same = {side: {'mcp': .4, 'thumb_cmc': -.3, 'thumb_ip': -.2} for side in hands.SIDES}
    assert hands.symmetry_error(same) < 1e-4
    different = {'left': same['left'], 'right': {'mcp': .1, 'thumb_cmc': -.3, 'thumb_ip': -.2}}
    assert hands.symmetry_error(different) > 5e-3
    with pytest.raises(ValueError):
        hands.Hand('left').link_poses({'mcp': 2.0})


def test_saved_pose_is_symmetric_and_keeps_real_targets():
    import json
    data = json.loads(hands.POSE_FILE.read_text())
    assert hands.symmetry_error(hands.load_pose()) < 1e-4
    assert data['joints']['left']['mcp']['real_deg'] == 325.8984
    assert data['joints']['right']['thumb_ip']['real_deg'] == 170.9473
    for side in data['joints'].values():
        for entry in side.values():
            assert entry['zero_offset_deg'] == pytest.approx(
                entry['real_deg'] - entry['real_sign'] * np.rad2deg(entry['sim_rad']))


def test_contact_pad_meets_the_box_as_before_when_the_hand_is_toed_in():
    import xml.etree.ElementTree as ET
    from igris.umi_walklow_replay import retarget
    from igris.umi_walklow_replay.kinematics import VENDOR_MJCF, _fixed, quat_matrix

    def pad_in_ee(toe):
        root = ET.parse(VENDOR_MJCF).getroot()
        hands.add_pads(root, .6, toe)
        result = {}
        for side, short, name in (('left', 'l', 'end_effector_left'), ('right', 'r', 'end_effector_right')):
            body = root.find(f".//body[@name='{short}_hand_rcv']"); geom = body.find(f"geom[@name='{short}_hand_pad']")
            pad = np.eye(4); pad[:3, 3] = np.fromstring(geom.get('pos'), sep=' ')
            pad[:3, :3] = quat_matrix(np.fromstring(geom.get('quat', '1 0 0 0'), sep=' '))
            result[side] = np.linalg.inv(_fixed(root.find(f".//body[@name='{name}']"))) @ _fixed(body) @ pad
        return result

    toe = {side: retarget.toe_in_transform(side, 12., retarget.DEFAULT['contact'][side]) for side in ('left', 'right')}
    plain, turned = pad_in_ee(None), pad_in_ee(toe)
    for side in ('left', 'right'):
        # A toed-in EE target is the plain one times ``toe``; the pad then sits where the plain pad sat.
        assert np.allclose(toe[side] @ turned[side], plain[side], atol=1e-8)
        assert not np.allclose(turned[side], plain[side], atol=1e-3)


def test_capsule_pad_lies_across_the_palm_and_reaches_the_flat_pad_face():
    import xml.etree.ElementTree as ET
    from igris.umi_walklow_replay.kinematics import VENDOR_MJCF, _fixed

    root = ET.parse(VENDOR_MJCF).getroot()
    names = hands.add_pads(root, .6, None, .035)
    assert names == ['l_hand_pad', 'r_hand_pad']
    for side, short, name in (('left', 'l', 'end_effector_left'), ('right', 'r', 'end_effector_right')):
        body = root.find(f".//body[@name='{short}_hand_rcv']"); geom = body.find(f"geom[@name='{short}_hand_pad']")
        assert geom.get('type') == 'capsule' and geom.get('condim') == '4' and geom.get('pos') is None
        to_ee = np.linalg.inv(_fixed(root.find(f".//body[@name='{name}']"))) @ _fixed(body)
        ends = np.fromstring(geom.get('fromto'), sep=' ').reshape(2, 3)
        ends = np.array([(to_ee @ np.r_[end, 1.])[:3] for end in ends])
        assert np.allclose(ends[:, 0], [-hands.PAD_CAPSULE_HALF, hands.PAD_CAPSULE_HALF]) and np.allclose(ends[:, 2], .035)
        # Its surface touches the plane of the flat pad's palm-side face.
        assert np.allclose(np.abs(ends[:, 1]) + hands.PAD_CAPSULE_RADIUS, abs(hands.PAD_FACE_IN_EE[side]))
        assert np.allclose(hands.capsule_axis_point(side, .035), ends.mean(axis=0))


def test_pad_outline_in_the_ee_frame_matches_the_documented_slab():
    import xml.etree.ElementTree as ET
    root = ET.parse(VENDOR_MJCF).getroot()
    hands.add_pads(root, .6)
    outline = hands.pad_points_in_ee(root)
    for side in ('left', 'right'):
        points = outline[side]
        assert points.shape == (8, 3)
        assert np.allclose([points[:, 0].min(), points[:, 0].max()], [-.040, .040], atol=1e-3)       # across the palm
        assert np.isclose(np.ptp(points[:, 2]), .177, atol=1e-3)                                      # along the fingers
        face = hands.PAD_FACE_IN_EE[side]                                                             # palm-side face
        assert np.isclose(points[:, 1].min() if face < 0 else points[:, 1].max(), face, atol=1e-3)


def test_capsule_pad_outline_covers_its_round_surface():
    import xml.etree.ElementTree as ET
    root = ET.parse(VENDOR_MJCF).getroot()
    hands.add_pads(root, .6, None, .035)
    points = hands.pad_points_in_ee(root)['left']
    assert np.isclose(points[:, 2].max() - .035, hands.PAD_CAPSULE_RADIUS) and np.isclose(.035 - points[:, 2].min(), hands.PAD_CAPSULE_RADIUS)
    assert np.isclose(points[:, 0].max(), hands.PAD_CAPSULE_HALF + hands.PAD_CAPSULE_RADIUS)
