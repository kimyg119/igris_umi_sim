"""Prepare a distant rolltainer destination and geometric walking/EE proposal.

Run with the configured Isaac Python for USD and MuJoCo imports; this module
does not start Isaac, open a transport, or step physics. All coordinates in
the proposal precede the runtime room-staging transform.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from box_replay import ROOT, MOTION, PAYLOAD, layout
from prepare import prepare
from rolltainer_staging import adapt_goal, staging_contract

EXPERIMENT = ROOT / 'outputs/hlik_transport_20260918_2005'
BASELINE = EXPERIMENT / 'transfer_trial48'
PLAN = EXPERIMENT / 'ready_plan43'
CART = '/World/Rolltainer'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _bounds_in_depth_frame(stage, path):
    """Cart +Y is inward depth, cart -X is lateral left."""
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bounds = bbox.ComputeRelativeBound(stage.GetPrimAtPath(path), stage.GetPrimAtPath(CART)).ComputeAlignedRange()
    lo, hi = np.array(bounds.GetMin()), np.array(bounds.GetMax())
    return np.array([lo[1], -hi[0], lo[2]]), np.array([hi[1], -lo[0], hi[2]])


def build(out, baseline=BASELINE, plan=PLAN, payload_mass=.25, mu=1.2, *, narrow_destination_support=False):
    """Make a fresh staged source, move destination assets, and rebuild MJCF."""
    out, baseline, plan = map(lambda p: Path(p).resolve(), (out, baseline, plan))
    baseline_report = json.loads((baseline / 'layout.json').read_text())
    plan_report = json.loads((plan / 'plan_result.json').read_text())
    baseline_scene = baseline / 'room_replay.usdc'
    baseline_hash = _sha(baseline_scene)
    out.mkdir(parents=True, exist_ok=False)
    staged_report = layout(out / 'staging', compliant_hands=True,
                           motion=plan / 'motion_100hz.npz', vendor_box=True,
                           source_base_clearance=.01, narrow_source_support=True,
                           payload_mass=payload_mass)
    staged_manifest = json.loads((out / 'staging/run/manifest.json').read_text())
    stage = Usd.Stage.Open(staged_report['scene'])
    stage.SetEditTarget(stage.GetSessionLayer())
    cache = UsdGeom.XformCache()
    original = Usd.Stage.Open(str(baseline_scene))
    original_cache = UsdGeom.XformCache()
    preserved = staged_manifest['source_support_bodies'] + [staged_manifest['payload']['body']]
    for body in preserved:
        path = '/World/Boxes/' + body.removeprefix('room_Boxes__')
        if not np.allclose(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)),
                           original_cache.GetLocalToWorldTransform(original.GetPrimAtPath(path)),
                           atol=1e-12, rtol=0):
            raise ValueError('Source pose differs from immutable baseline: ' + path)

    cart_matrix = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(CART))
    cart_origin = np.array(cart_matrix.ExtractTranslation())
    inward = np.array(cart_matrix.TransformDir(Gf.Vec3d(0, 1, 0)))
    yaw = float(np.arctan2(inward[1], inward[0]))
    rotation = _rotation(yaw)
    deck_lo, deck_hi = _bounds_in_depth_frame(stage, CART + '/Chassis/Deck_Collider')
    deck_top = float(deck_hi[2])
    gap = .0002
    boxes = copy.deepcopy(staged_report['boxes'])

    def place(name, local_xyz, box_yaw, roll_deg=0.):
        xyz = cart_origin + rotation @ np.asarray(local_xyz)
        matrix = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), np.rad2deg(box_yaw)))
        if roll_deg:
            matrix = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), roll_deg)) * matrix
        matrix.SetTranslateOnly(Gf.Vec3d(*map(float, xyz)))
        UsdGeom.Xformable(stage.GetPrimAtPath('/World/Boxes/' + name)).MakeMatrixXform().Set(matrix)
        boxes[name] = dict(xyz_m=xyz.tolist(), yaw_deg=float(np.rad2deg(box_yaw)), roll_deg=float(roll_deg))

    stacks = []
    for column, depth in [(3, -.225), (2, .10)]:
        top = deck_top
        if narrow_destination_support and column == 3:
            # Existing assets only: .23 + .10 + .18 m preserves the original
            # .51 m stack height. The top's .10 m lateral span clears the
            # seated fingertips. Source support assets are never repurposed.
            depth = -.25
            specifications = [('Roll_C3_R1_L1', yaw + np.pi/2, 90.),
                              ('Small_RightBack_L1', yaw, 0.),
                              ('Small_Center_L3', yaw, 90.)]
        else:
            specifications = [(f'Roll_C{column}_R1_L{i}', yaw - np.pi/2, 0.) for i in (1, 2, 3)]
        names = [item[0] for item in specifications]
        levels = []
        for name, box_yaw, roll_deg in specifications:
            collider = stage.GetPrimAtPath('/World/Boxes/' + name + '/Body_Collider')
            cube = UsdGeom.Cube(collider)
            scale = np.array(Gf.Transform(UsdGeom.Xformable(collider).GetLocalTransformation()).GetScale())
            size = float(cube.GetSizeAttr().Get()) * scale
            expected = [.18, .18, .10] if name.startswith('Small_') else [.345, .23, .17]
            if not np.allclose(size, expected, atol=1e-7):
                raise ValueError('Unexpected original destination box dimensions')
            height = size[1] if roll_deg else size[2]
            center_z = top + gap + height / 2
            place(name, [depth, 0., center_z], box_yaw, roll_deg)
            low, high = _bounds_in_depth_frame(stage, str(collider.GetPath()))
            if np.any(low[:2] < deck_lo[:2]) or np.any(high[:2] > deck_hi[:2]):
                raise ValueError('Destination box overhangs rolltainer deck')
            levels.append(dict(name=name, bottom_m=float(low[2]), top_m=float(high[2]),
                               min_depth_frame_m=low.tolist(), max_depth_frame_m=high.tolist()))
            top = float(high[2])
        stacks.append(dict(boxes=names, center_depth_frame_xy_m=[depth, 0.],
                           support_top_m=top, levels=levels))
    selected = {'room_Boxes__' + name for stack in stacks for name in stack['boxes']}
    parked = []
    for i, body in enumerate(b for b in staged_manifest['destination_bodies'] if b not in selected):
        name = body.removeprefix('room_Boxes__')
        # Beyond the old extras' maximum X=2.55 m, away from the walking corridor.
        xyz = [3., -1. - .30 * i, .0853]
        matrix = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*xyz))
        UsdGeom.Xformable(stage.GetPrimAtPath('/World/Boxes/' + name)).MakeMatrixXform().Set(matrix)
        boxes[name] = dict(xyz_m=xyz, yaw_deg=0., roll_deg=0.)
        parked.append(body)

    support_top = stacks[0]['support_top_m']
    target_local = np.array([-.25, 0., support_top + .075 + .0003])
    target = cart_origin + rotation @ target_local
    robot_goal = cart_origin + rotation @ [-.64, 0., 0.]
    primary_low = np.array(stacks[0]['levels'][-1]['min_depth_frame_m'])
    primary_high = np.array(stacks[0]['levels'][-1]['max_depth_frame_m'])
    payload_margins = np.r_[target_local[:2] - .075 - primary_low[:2], primary_high[:2] - target_local[:2] - .075]
    com_margins = np.r_[target_local[:2] - primary_low[:2], primary_high[:2] - target_local[:2]]
    overlap = np.maximum(0., np.minimum(target_local[:2] + .075, primary_high[:2])
                         - np.maximum(target_local[:2] - .075, primary_low[:2]))
    payload_support = dict(full_footprint_supported=bool(np.all(payload_margins >= 0)),
        intentional_footprint_overhang=bool(narrow_destination_support),
        com_projection_edge_margins_m=com_margins.tolist(),
        nominal_payload_overhang_m=np.maximum(0., -payload_margins).tolist(),
        nominal_footprint_overlap_area_m2=float(np.prod(overlap)),
        nominal_footprint_overlap_fraction=float(np.prod(overlap)/(.15*.15)),
        top_footprint_depth_frame_xy_m=(primary_high[:2]-primary_low[:2]).tolist(),
        interpretation='Planar geometry for a yaw-aligned 150 mm cube; positive COM margin is not a settled-contact or dynamic stability guarantee.')
    if np.any(com_margins <= 0):
        raise ValueError('Payload COM projection is outside destination support')
    if np.any(payload_margins < 0) and not narrow_destination_support:
        raise ValueError('Payload target is not fully supported')
    if abs(support_top - baseline_report['destination_support_top_m']) > .002:
        raise ValueError('Cart stack height differs from task1 by more than 2 mm')

    metadata = json.loads(MOTION.with_name('source_metadata.json').read_text())
    reference = staged_manifest['room_reference_base_xyyaw']
    initial_to_world = np.eye(4)
    initial_to_world[:3, :3] = _rotation(reference[2])
    initial_to_world[:2, 3] = reference[:2]
    robot_goal_base = np.linalg.inv(initial_to_world) @ np.r_[robot_goal, 1.]
    source_payload = np.array(staged_report['boxes'][PAYLOAD.rsplit('/', 1)[-1]]['xyz_m'])
    grasp_yaw = np.arctan2(source_payload[1] - reference[1], source_payload[0] - reference[0])
    # Anchor radial grasp direction is exactly -45 degrees in the task1 layout.
    # Use the authored direction, not the slightly displaced initial base yaw.
    grasp_rotation = _rotation(yaw - np.deg2rad(-45.))
    source_to_place = np.eye(4)
    source_to_place[:3, :3] = grasp_rotation
    source_to_place[:3, 3] = target - grasp_rotation @ source_payload
    placement_ee = [source_to_place @ np.array(p) for p in plan_report['anchor_world_hand_poses']]
    carry_z = source_payload[2] + float(plan_report['lift_m'])
    carry_ee = [p.copy() for p in placement_ee]
    for p in carry_ee:
        p[2, 3] += carry_z - target[2]
    entry_lows, entry_highs = zip(*[_bounds_in_depth_frame(stage, CART + f'/Chassis/Side{i}_Tube0') for i in (0, 1)])
    entry_span = [float(entry_highs[1][1]), float(entry_lows[0][1])]
    ee_local = [(rotation.T @ (p[:3, 3] - cart_origin)).tolist() for p in placement_ee]

    usd = out / 'room_replay.usdc'
    stage.Flatten().Export(str(usd))
    manifest = prepare(out / 'run', scene=usd, compliant_hands=True)
    for key in ('payload', 'source_support_bodies', 'stage_room_after_preparation', 'room_reference_base_xyyaw'):
        manifest[key] = copy.deepcopy(staged_manifest[key])
    manifest['payload']['geom'] = next(c['geom'] for c in manifest['room_colliders'] if c['body'] == manifest['payload']['body'])
    manifest['destination_bodies'] = (['room_Boxes__'+name for stack in stacks for name in stack['boxes']]
        if narrow_destination_support else [b for b in staged_manifest['destination_bodies'] if b in selected])
    manifest['parked_previous_destination_bodies'] = parked
    manifest['evaluation_profile'] = 'relaxed_transport'
    manifest['room_staging'] = staging_contract(manifest)
    tree = ET.parse(out / 'run/scene.xml')
    payload_geom = tree.find(f".//geom[@name='{manifest['payload']['geom']}']")
    staged_geom = ET.parse(out / 'staging/run/scene.xml').find(f".//geom[@name='{staged_manifest['payload']['geom']}']")
    for field in ('condim', 'friction'):
        payload_geom.set(field, staged_geom.get(field))
    tree.write(out / 'run/scene.xml', encoding='unicode')
    (out / 'run/manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    from contact_material import apply
    apply(out / 'run', mu)
    manifest = json.loads((out / 'run/manifest.json').read_text())
    manifest['sha256'][str(out / 'run/scene.xml')] = _sha(out / 'run/scene.xml')
    manifest['layout_kind'] = 'far_rolltainer_narrow_destination_support' if narrow_destination_support else 'far_rolltainer_two_three_box_stacks'
    manifest['narrow_destination_support'] = bool(narrow_destination_support)
    manifest['destination_support'] = payload_support
    (out / 'run/manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    report = dict(source_usd=staged_report['source_usd'], scene=str(usd),
        baseline_scene=str(baseline_scene), baseline_scene_sha256=baseline_hash,
        motion=staged_report['motion'], motion_sha256=staged_report['motion_sha256'],
        source_support_top_m=staged_report['source_support_top_m'],
        source_support_bodies=manifest['source_support_bodies'],
        destination_bodies=manifest['destination_bodies'], previous_destination_bodies=staged_manifest['destination_bodies'],
        parked_previous_destination_bodies=parked, payload=manifest['payload'], boxes=boxes, stacks=stacks,
        destination_xy_m=target[:2].tolist(), destination_support_top_m=support_top,
        task1_destination_support_top_m=baseline_report['destination_support_top_m'],
        support_height_difference_from_task1_m=support_top-baseline_report['destination_support_top_m'],
        stack_interlayer_gap_m=gap, contact_material=manifest['contact_material'],
        stage_room_after_preparation=manifest['stage_room_after_preparation'],
        room_reference_base_xyyaw=reference, rolltainer_translation_m=[0., 1., 0.],
        depth_frame=dict(origin_world_m=cart_origin.tolist(), yaw_world_rad=yaw,
                         convention='+X into cart through open front; +Y robot left; origin at deck XY center and floor Z',
                         deck_min_m=deck_lo.tolist(), deck_max_m=deck_hi.tolist(),
                         entrance_inner_lateral_span_m=entry_span),
        geometry_checks=dict(payload_support_edge_margins_m=payload_margins.tolist(), payload_support=payload_support,
            primary_stack_deck_front_clearance_m=float(stacks[0]['levels'][0]['min_depth_frame_m'][0]-deck_lo[0]),
            nominal_robot_root_to_deck_front_m=float(deck_lo[0] + .64),
            source_pose_matches_baseline=True, original_asset_dimensions_preserved=True,
            original_box_masses_preserved_except_payload=True, all_room_boxes_free=True,
            old_destination_obstacles_removed=True),
        goal_proposal=dict(robot_goal_xy_world_m=robot_goal[:2].tolist(), robot_goal_yaw_world_rad=yaw,
            robot_goal_xy_initial_base_m=robot_goal_base[:2].tolist(), robot_goal_yaw_initial_base_rad=yaw-reference[2],
            initial_base_to_world=initial_to_world.tolist(), initial_base_position_world_m=metadata['base_position'],
            initial_base_quat_xyzw=metadata['base_quat_xyzw'],
            payload_center_world_m=target.tolist(), payload_center_depth_frame_m=target_local.tolist(),
            payload_center_robot_goal_floor_frame_m=[.39, 0., float(target[2])],
            source_grasp_radial_yaw_measured_from_initial_base_rad=float(grasp_yaw),
            source_to_placement_transform=source_to_place.tolist(),
            placement_ee_world=[p.tolist() for p in placement_ee], placement_ee_depth_frame_xyz_m=ee_local,
            carry_front_payload_center_world_m=[*target[:2], float(carry_z)],
            carry_front_ee_world=[p.tolist() for p in carry_ee],
            ee_source=str(plan / 'plan_result.json'), ee_source_sha256=_sha(plan / 'plan_result.json'),
            ee_interpretation='Initial candidate: ready_plan43 grasp-anchor rigid transforms, rotated +90 degrees and translated with payload; not solved IK or contact-qualified EE.',
            remaining_checks=['IK and joint limits at goal', 'whole-body and foot clearance along approach and placement',
                              'loaded walking balance and contact retention', 'dynamic stack settling and release']),
        sha256={str(usd): _sha(usd), str(out / 'run/scene.xml'): _sha(out / 'run/scene.xml')},
        physics_steps=0, hardware_executed=False, dynamics_validated=False,
        limits='Geometry and host model compilation only. No walking, grasp, settling, release, or robot collision validation. Cart colliders remain static; robot root and boxes remain free. Support is 1.8 mm above task1 because three unscaled original boxes stand on the existing deck.')
    report['narrow_destination_support'] = bool(narrow_destination_support)
    if narrow_destination_support:
        report['limits'] = ('Geometry and host model compilation only; no settling, walking or release validation. '
            'Unscaled original .23/.10/.18 m layers retain the .6826 m support top. '
            'Payload intentionally overhangs the .10 m lateral support by .025 m per side; '
            'nominal footprint overlap is two thirds, with COM margins .09/.05 m. '
            'The .18 m middle layer overhangs the .17 m base depth by .005 m per side. '
            'All boxes remain free; positive COM projection margins do not prove stack stability.')
        report['geometry_checks']['primary_layer_support'] = dict(
            middle_front_back_overhang_m=.005,
            middle_com_projection_edge_margins_m=[.085, .1725, .085, .1725],
            top_fully_inside_middle_footprint=True,
            payload_full_footprint_supported=False)
    report['room_staging'] = manifest['room_staging']
    report['goal_proposal'] = adapt_goal(report['goal_proposal'], reference, reference)
    if _sha(baseline_scene) != baseline_hash:
        raise RuntimeError('Immutable source scene changed during preparation')
    (out / 'layout.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'goal_proposal.json').write_text(json.dumps(report['goal_proposal'], indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--payload-mass', type=float, default=.25)
    parser.add_argument('--mu', type=float, default=1.2)
    parser.add_argument('--narrow-destination-support', action='store_true',
                        help='Use existing .23/.10/.18 m assets with a .10 m wide receiving top; intentional payload overhang')
    args = parser.parse_args()
    result = build(args.out, args.baseline, args.plan, args.payload_mass, args.mu,
                   narrow_destination_support=args.narrow_destination_support)
    print(json.dumps({key: result[key] for key in ('scene', 'destination_support_top_m', 'destination_xy_m', 'goal_proposal', 'physics_steps', 'limits')}, indent=2))
