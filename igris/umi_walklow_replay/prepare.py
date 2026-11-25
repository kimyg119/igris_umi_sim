"""Prepare one replay: episode, targets, scene and the composed vendor model.

Output layout (a new directory every time):
  episode/   episode.npz, episode.json           poses used (tracking faults repaired) and derived geometry
             recorded.npz                        poses as recorded, when they were repaired
  scene/     scene.usda, textures/, backdrop     Isaac Sim scene
  model/     vendor_posed.xml, hands.json        vendor MJCF with the RCV hand posture
  targets.npz, retarget.json                     100 Hz robot EE targets (pelvis frame)
  run/       scene.xml, robot.yaml, manifest...  input of isaac_vendor_room/run.py
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np

from . import (clearance, collision, elbow_prior, episode as episode_module, hand_calibration, hands, posture_prior,
               retarget, scene, tracking)
from .kinematics import VENDOR_MJCF

ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOM = ROOT / 'igris/isaac_vendor_room'
RATE_HZ = 100.


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


HAND_BODIES = ('l_hand_rcv', 'r_hand_rcv')


def vendor_differences(original, changed):
    """Every difference from the vendor MJCF; all must lie in the two hand bodies."""
    allowed = {id(geom) for name in HAND_BODIES for geom in changed.find(f".//body[@name='{name}']").findall('geom')}
    differences = []

    def walk(a, b, path):
        if a.tag != b.tag:
            raise ValueError(f'Vendor structure changed at {path}')
        if a.attrib != b.attrib:
            if id(b) not in allowed or set(k for k in set(a.attrib) | set(b.attrib) if a.get(k) != b.get(k)) - {'pos', 'quat', 'friction'}:
                raise ValueError(f'Unexpected vendor change at {path}: {a.attrib} -> {b.attrib}')
            differences.append(b.get('name'))
        children_a, children_b = list(a), list(b)
        extra = children_b[len(children_a):]
        if len(children_b) < len(children_a) or any(id(e) not in allowed or e.get('type') not in ('box', 'capsule') for e in extra):
            raise ValueError(f'Unexpected element count at {path}')
        differences.extend(e.get('name') for e in extra)
        for index, (x, y) in enumerate(zip(children_a, children_b)):
            walk(x, y, f'{path}/{x.tag}[{index}]')

    walk(original, changed, original.tag)
    return differences


def posed_root(pose, toe_in=None, capsule_contact_z=None):
    """Vendor MJCF tree with the fixed hand meshes moved to the requested posture and a contact pad per hand."""
    root = ET.parse(VENDOR_MJCF).getroot()
    report = hands.apply_mjcf(root, pose)
    for side in ('l', 'r'):
        for geom in root.find(f".//body[@name='{side}_hand_rcv']").findall('geom'):
            # Silicone palm on paper.  Without this the default 1.0 would win the
            # MuJoCo max-friction rule against the 0.6 scene surfaces.
            geom.set('friction', f'{scene.FRICTION} 0.005 0.0001')
    pads = hands.add_pads(root, scene.FRICTION, toe_in, capsule_contact_z)
    return root, report, pads


def posed_vendor_model(out, pose, toe_in=None, capsule_contact_z=None):
    """Write that tree; every difference from the vendor file is checked to lie inside the two hand bodies."""
    root, report, pads = posed_root(pose, toe_in, capsule_contact_z)
    changed = vendor_differences(ET.parse(VENDOR_MJCF).getroot(), root)
    out.mkdir(parents=True, exist_ok=False)
    path = out / 'vendor_posed.xml'
    ET.ElementTree(root).write(path, encoding='unicode')
    (out / 'hands.json').write_text(json.dumps(dict(
        pose_file=str(hands.POSE_FILE), pose_sha256=sha256(hands.POSE_FILE), symmetry_error_m=hands.symmetry_error(pose),
        hand_friction=scene.FRICTION, vendor_inertia_unchanged=True, contact_pads=pads,
        elements_differing_from_vendor=len(changed), all_differences_inside_hand_bodies=True,
        report=report), indent=2, default=float) + '\n')
    return path, pads


def pads_only_contact(scene_xml, pads):
    """Leave the hand meshes non-colliding, as the vendor has them; only the pads touch the scene."""
    tree = ET.parse(scene_xml)
    for name in HAND_BODIES:
        for geom in tree.getroot().find(f".//body[@name='{name}']").findall('geom'):
            if geom.get('name') in pads:
                if geom.get('contype') != '2' or geom.get('conaffinity') != '0':
                    raise ValueError('Contact pad lost its environment-only collision bits')
            else:
                geom.set('contype', '0'); geom.set('conaffinity', '0')
    tree.write(scene_xml, encoding='unicode')


def targets(episode, config, speed):
    segments = episode_module.carry_segments(episode)
    left, right = retarget.bimanual_targets(episode, segments, config)
    left, right = retarget.toe_in(left, 'left', config), retarget.toe_in(right, 'right', config)
    t, left = retarget.resample(episode['t'], left, RATE_HZ, speed)
    _, right = retarget.resample(episode['t'], right, RATE_HZ, speed)
    head_yaw = np.interp(t, episode['t'] / speed, episode['head_yaw'])
    out = dict(t=t, left=left, right=right, head_yaw=head_yaw,
               orientation=np.interp(t, episode['t'] / speed, retarget.orientation_weight(episode['t'], segments, config)))
    if config.get('toe_free_deg'):
        out.update(toe_range=np.radians(config['toe_free_deg']), toe_preference=np.radians(config['toe_preference_deg']),
                   toe_pivot=toe_pivots(config))
        if config.get('pad') != 'capsule':
            # A flat pad cannot rock on the box face: the angle is free only while nothing is held.
            out['toe_hold'] = np.interp(t, episode['t'] / speed, retarget.grasp_weight(episode['t'], segments, config))
    return out


def toe_pivots(config):
    """EE-frame point each hand turns about when its toe-in is free: the capsule axis, or the palm contact point."""
    if config.get('pad') == 'capsule':
        return np.array([hands.capsule_axis_point(side, config['contact'][side][2]) for side in ('left', 'right')])
    return np.array([config['contact'][side] for side in ('left', 'right')], dtype=float)


def height_profile(made, layout, report):
    """Where the targets belong to the pick side, and how much lower that side is than the recording has it.

    The worker lifts every target so that the recorded place surface meets the roll container's loading
    surface.  The case can stand lower or higher than that (0.77 m against 0.80 m, measured), by a different
    amount than the recording shows; targets near the pick take the difference, fading out along the way to
    the place.  ``pick_weight`` is 1 at the pick and 0 at the place; ``pick_offset_z`` is the extra shift [m].
    """
    pick, place = np.array(layout['parcels'][0]['xy']), np.array(layout['place']['xy'])
    middle = (made['left'][:, :2, 3] + made['right'][:, :2, 3]) / 2
    along = np.clip((middle - pick) @ (place - pick) / float((place - pick) @ (place - pick)), 0., 1.)
    recorded = float(report.get('source_minus_destination_top_m') or 0.)
    return dict(pick_weight=1. - along * along * (3 - 2 * along),
                pick_offset_z=float(layout['case_top_m'] - layout['surface_top_m']) - recorded)


def support_surfaces(layout):
    """Top faces the hands keep clear of, in the frame of the targets: the case and the roll container's loading
    surface (its front row of cartons)."""
    case, cart = layout['case'], layout['cart']
    length, width, _ = cart['carton']
    front = np.array(cart['xy']) + scene.rotation(cart['yaw']) @ np.array([0., -width / 2 - .01])
    return [dict(name='case', centre=list(case['xy']), yaw=case['yaw'], half=[case['size'][0] / 2, case['size'][1] / 2],
                 top=case['size'][2]),
            dict(name='loading surface', centre=front.tolist(), yaw=cart['yaw'], half=[length + .005, width / 2],
                 top=layout['surface_top_m'])]


def keep_clear(made, profile, layout, robot, margin):
    """Raise the hand targets where the contact pads would come closer than ``margin`` to a support surface.
    ``robot`` is the posed vendor tree (``posed_root``), which carries the pads."""
    outline = hands.pad_points_in_ee(robot)
    room_height = layout['surface_top_m'] + retarget.PELVIS_TO_SURFACE + profile['pick_offset_z'] * profile['pick_weight']
    lifts = {}
    for side in ('left', 'right'):
        made[side], lifts[side] = clearance.clear_supports(made[side], outline[side], room_height, support_surfaces(layout),
                                                           1. / RATE_HZ, margin=margin)
    made['clearance_lift'] = np.stack([lifts['left'], lifts['right']], axis=1)
    return {side: dict(max_lift_m=float(lift.max()), seconds_lifted=float((lift > 0).sum() / RATE_HZ))
            for side, lift in lifts.items()}


def shape(dataset, episode_index, *, speed=1., squeeze=None, palm_correction=True, object_centric=True, shift=None,
          carton_height=None, toe_in_deg=0., toe_free_deg=None, pad='flat', clean_tracking=True, cart_setback=None,
          calibrate_hands=True, case_top=None, support_clearance=None):
    """The part of a preparation that decides the hand targets, with nothing written: the episode (tracking faults
    repaired, hand frames calibrated), the scene layout it implies and the 100 Hz targets.  The targets are shaped
    with the whole episode in view - grasp intervals, pick and place geometry, support surfaces - which a replay
    may do and a controller that is handed one pose at a time may not (labels.py turns them into training labels).
    """
    recorded = episode_module.load_episode(dataset, episode_index)
    episode = recorded; cleaning = calibration = None
    if clean_tracking:
        # Jumps, stalls and sample-time jitter of the recording would go straight into the IK targets.
        episode, cleaning = tracking.clean(episode)
    if calibrate_hands:
        # The two recorded hand frames disagree by constants (hand_calibration.py); the right hand is the reference.
        episode = hand_calibration.apply(episode); calibration = hand_calibration.CALIBRATION
    report = dict(dataset=episode_module.DATASET, root=str(dataset), episode=episode_index, frames=len(episode['t']),
                  duration_s=float(episode['t'][-1]), tracking=cleaning, hand_calibration=calibration,
                  **episode_module.geometry(episode))
    config = json.loads(json.dumps(retarget.DEFAULT))
    # Calibrated poses already have the palm axis on +y; the older per-hand correction is for uncalibrated ones.
    config['palm_normal_correction'] = bool(palm_correction) and not calibrate_hands
    config['object_centric'] = bool(object_centric)
    config['box_length_m'] = scene.PARCEL[0]
    config['box_height_m'] = scene.PARCEL[2]
    if squeeze is not None:
        config['squeeze_m'] = float(squeeze)
    # Where the robot stands relative to the recorded base frame.
    shift = list(scene.DEFAULT_SHIFT) if shift is None else [float(v) for v in shift]
    config['base_offset'] = [-shift[0], -shift[1], 0.]
    config['toe_in_deg'] = float(toe_in_deg)
    config['toe_free_deg'] = [float(v) for v in toe_free_deg] if toe_free_deg else None
    if support_clearance is not None:
        config['support_clearance_m'] = float(support_clearance)
    config['pad'] = pad
    if pad not in ('flat', 'capsule'):
        raise ValueError(f'Unknown pad {pad!r}')
    if config['toe_free_deg'] and config['toe_in_deg']:
        raise ValueError('Give either a fixed toe-in or a free range, not both')
    layout = scene.layout_from_episode(report, shift, carton_height, cart_setback, case_top)
    made = targets(episode, config, speed)
    profile = height_profile(made, layout, report)
    pose = hands.load_pose()
    # A flat pad is turned back by the toe-in the hands will have: the fixed angle, or the preferred one of a free range.
    pad_angle = config['toe_preference_deg'] if config['toe_free_deg'] else config['toe_in_deg']
    toe = {side: retarget.toe_in_transform(side, pad_angle, config['contact'][side]) for side in ('left', 'right')}
    capsule = config['contact']['left'][2] if pad == 'capsule' else None
    # The targets, once the hands' contact pads are known: kept clear of the surfaces the boxes stand on.
    lifted = keep_clear(made, profile, layout, posed_root(pose, toe, capsule)[0], config['support_clearance_m']) \
        if config.get('support_clearance_m') else None
    return dict(recorded=recorded if clean_tracking or calibrate_hands else None, episode=episode, report=report,
                cleaning=cleaning, calibration=calibration, config=config, shift=shift, layout=layout, made=made,
                profile=profile, lifted=lifted, hand_pose=pose, toe=toe, capsule_contact_z=capsule, speed=speed)


def prepare(out, dataset, episode_index, *, speed=1., squeeze=None, palm_correction=True, object_centric=True,
            shift=None, carton_height=None, use_elbow_prior=False, checkpoint=None, toe_in_deg=0.,
            toe_free_deg=None, pad='flat', clean_tracking=True, cart_setback=None, calibrate_hands=True, case_top=None,
            support_clearance=None):
    """``use_elbow_prior`` is False, True ('human') or the name of a FiSTA checkpoint in ``elbow_prior.CHECKPOINTS``
    ('human': trained on retargeted human motion, 'teleop': trained on this robot's teleoperation); ``checkpoint``
    names a file instead."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    # 1. Episode, the scene quantities it implies and the robot EE targets.
    shaped = shape(dataset, episode_index, speed=speed, squeeze=squeeze, palm_correction=palm_correction,
                   object_centric=object_centric, shift=shift, carton_height=carton_height, toe_in_deg=toe_in_deg,
                   toe_free_deg=toe_free_deg, pad=pad, clean_tracking=clean_tracking, cart_setback=cart_setback,
                   calibrate_hands=calibrate_hands, case_top=case_top, support_clearance=support_clearance)
    episode, report, config, layout = shaped['episode'], shaped['report'], shaped['config'], shaped['layout']
    made, profile, lifted, pose = shaped['made'], shaped['profile'], shaped['lifted'], shaped['hand_pose']
    cleaning, calibration = shaped['cleaning'], shaped['calibration']
    (out / 'episode').mkdir()
    if shaped['recorded'] is not None:
        np.savez_compressed(out / 'episode/recorded.npz', **shaped['recorded'])
    np.savez_compressed(out / 'episode/episode.npz', **episode)
    (out / 'episode/episode.json').write_text(json.dumps(report, indent=2) + '\n')
    # 2. Scene and hand posture.
    scene.build(out / 'scene', layout, pose)
    vendor, pads = posed_vendor_model(out / 'model', pose, shaped['toe'], shaped['capsule_contact_z'])
    np.savez_compressed(out / 'targets.npz', surface_top=layout['surface_top_m'], speed=speed,
                        carry_frames=np.array(report['carry_frames']), **made, **profile)
    retarget.save_config(out / 'retarget.json', config)
    # Collision spheres of the robot for the self-collision bounds of the mink IK (``replay --ik mink``).
    collision.export(out / 'model/collision_spheres.npz')
    # Arm posture preference looked up from the teleoperated robot recordings (posture_prior.py).
    if posture_prior.TABLE.is_file():
        shutil.copyfile(posture_prior.TABLE, out / 'model/teleop_posture.npz')
    # HL-IK elbow prior: torch weights -> numpy arrays the container can load.
    prior = None
    if use_elbow_prior:
        name = 'human' if use_elbow_prior is True else str(use_elbow_prior)
        if name not in elbow_prior.CHECKPOINTS:
            raise ValueError(f'Unknown elbow prior {name!r}; one of {sorted(elbow_prior.CHECKPOINTS)}')
        checkpoint = Path(checkpoint) if checkpoint else elbow_prior.CHECKPOINTS[name]
        if not checkpoint.is_file():
            raise FileNotFoundError(f'FiSTA checkpoint missing: {checkpoint}; copy it or leave --elbow-prior out')
        prior = elbow_prior.export(checkpoint, out / 'model/fista.npz')
    # 4. Compose the scene into the vendor model with the existing, audited tool.
    sys.path.insert(0, str(VENDOR_ROOM))
    import prepare as vendor_prepare
    manifest = vendor_prepare.prepare(out / 'run', scene=out / 'scene/scene.usda', vendor=vendor)
    pads_only_contact(out / 'run/scene.xml', pads)
    manifest['hand_environment_contact'] = dict(geom_names=pads, hand_contype=2, hand_conaffinity=0, room_conaffinity=3,
                                                hand_meshes='non-colliding, as in the vendor model')
    manifest.update(
        stage_room_after_preparation=True, room_reference_base_xyyaw=[0., 0., 0.],
        umi_replay=dict(dataset=episode_module.DATASET, episode=episode_index, speed=speed,
                        targets=str(out / 'targets.npz'), targets_sha256=sha256(out / 'targets.npz'),
                        retarget=config, layout=layout, hand_pose=pose, vendor_model=str(vendor), tracking=cleaning,
                        hand_calibration=calibration, support_clearance=lifted,
                        elbow_prior=dict(weights=str(prior), checkpoint=str(checkpoint),
                                         checkpoint_sha256=sha256(checkpoint)) if prior else None,
                        vendor_model_sha256=sha256(vendor), original_vendor_model=str(VENDOR_MJCF),
                        robot_changes='hand mesh poses, hand friction and one massless contact pad per hand; '
                                      'mass, inertia, joints, actuators unchanged'))
    (out / 'run/manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    summary = dict(output=str(out), episode=episode_index, boxes=report['boxes'], duration_s=report['duration_s'] / speed,
                   box_height_estimate_m=report.get('box_height_m'), support_clearance=lifted,
                   room_bodies=len(manifest['room_bodies']),
                   room_colliders=len(manifest['room_colliders']))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True, help='Local LeRobot dataset directory')
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--speed', type=float, default=1., help='Playback speed; below 1 slows the motion')
    parser.add_argument('--squeeze', type=float, help='Palm travel into each box face [m]')
    parser.add_argument('--toe-free', type=float, nargs=2, metavar=('LOW', 'HIGH'),
                        help='Let the IK choose the toe-in within this range [deg] (preferred 12)')
    parser.add_argument('--pad', choices=['flat', 'capsule'], default='flat', help='Contact surface of the hands')
    parser.add_argument('--toe-in', type=float, default=0., metavar='DEG',
                        help='Turn the hands about their palm contact points so the fingers converge (teleop: 17-26 deg)')
    parser.add_argument('--no-palm-correction', action='store_true', help='Use the recorded palm normals unchanged')
    parser.add_argument('--no-object-centric', action='store_true', help='Replay each hand independently while holding')
    parser.add_argument('--raw-hands', action='store_true',
                        help='Use the recorded hand frames without the left/right calibration (hand_calibration.py)')
    parser.add_argument('--case-top', type=float, help='Height of the case top [m] (default 0.77)')
    parser.add_argument('--support-clearance', type=float,
                        help='Air kept between the hand pads and the case / loading surface [m] (default 0.02; 0 off)')
    parser.add_argument('--raw-tracking', action='store_true',
                        help='Use the recorded poses as they are: keep tracking jumps, stalls and sample-time jitter')
    parser.add_argument('--elbow-prior', nargs='?', const='human', choices=sorted(elbow_prior.CHECKPOINTS),
                        help='Place the elbows with the FiSTA prior: "human" (retargeted human motion; the default when '
                             'no name is given) or "teleop" (trained on this robot\'s teleoperated box handling)')
    parser.add_argument('--checkpoint', type=Path, help='FiSTA checkpoint file instead of the named one')
    parser.add_argument('--carton-height', type=float,
                        help='Height of the cartons under the set-down point [m]; three of them plus the 0.172 m deck '
                             'give the height of both work surfaces')
    parser.add_argument('--cart-setback', type=float, nargs=2, metavar=('DX', 'DY'),
                        help='Roll container offset from the recorded place: along its own -x and +y [m]')
    parser.add_argument('--shift', type=float, nargs=2, metavar=('DX', 'DY'),
                        help='Pelvis position in the recorded base frame [m]; default 0.04 0')
    args = parser.parse_args()
    prepare(args.out, args.dataset, args.episode, speed=args.speed, squeeze=args.squeeze,
            palm_correction=not args.no_palm_correction, object_centric=not args.no_object_centric, shift=args.shift,
            carton_height=args.carton_height, cart_setback=args.cart_setback, use_elbow_prior=args.elbow_prior, checkpoint=args.checkpoint, toe_in_deg=args.toe_in, toe_free_deg=args.toe_free, pad=args.pad,
            clean_tracking=not args.raw_tracking, calibrate_hands=not args.raw_hands, case_top=args.case_top,
            support_clearance=args.support_clearance)


if __name__ == '__main__':
    main()
