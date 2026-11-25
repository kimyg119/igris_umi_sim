"""Fixed RCV-hand posture for the vendor MuJoCo model and the render USD.

The vendor simulator carries each hand as one rigid body whose 38 meshes sit
in the URDF zero pose.  This module moves the meshes that belong to the finger
(MCP), thumb CMC and thumb IP links to the requested joint angles.  The hand
stays a single rigid body with the vendor mass and inertia: nothing is actuated.

``hand_pose.json`` keeps, for every joint, the real-robot encoder target the
user supplied next to the simulated URDF angle.  Their difference is the
encoder zero of the current assembly; no such calibration exists elsewhere.
"""
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .kinematics import _fixed, matrix_quat

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / 'igris/igris_c_description_public/urdf/igris_c_rcv_hand.urdf'
POSE_FILE = Path(__file__).with_name('hand_pose.json')
# Real-robot joint name -> URDF joint suffix.
JOINTS = {'mcp': 'hand_dof_joint_mcp', 'thumb_cmc': 'hand_dof_joint_cmc', 'thumb_ip': 'hand_dof_joint_mcp_thumb'}
SIDES = {'left': 'l', 'right': 'r'}
# Joint angles the vendor baked into its rigid hand meshes (recovered by fitting
# the URDF to the 38 vendor mesh poses; residual below 1e-8 m on both hands).
VENDOR_BAKED = {'mcp': np.deg2rad(10.), 'thumb_cmc': np.deg2rad(-60.), 'thumb_ip': np.deg2rad(-60.)}


def rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _origin(element):
    transform = np.eye(4)
    if element is not None:
        transform[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
        transform[:3, :3] = rpy_matrix(np.fromstring(element.get('rpy', '0 0 0'), sep=' '))
    return transform


def _axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    out = np.eye(4)
    out[:3, :3] = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)
    return out


def _quat_matrix(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class Hand:
    """URDF sub-tree of one hand, rooted at ``<side>_hand_root``."""

    def __init__(self, side, urdf=URDF):
        if side not in SIDES:
            raise ValueError(side)
        self.side = side
        self.prefix = SIDES[side] + '_'
        self.root = self.prefix + 'hand_root'
        tree = ET.parse(urdf).getroot()
        self.visual = {}
        for link in tree.findall('link'):
            name = link.get('name')
            if not name.startswith(self.prefix + 'hand'):
                continue
            visual = link.find('visual')
            if visual is not None and visual.find('geometry/mesh') is not None:
                mesh = visual.find('geometry/mesh').get('filename').rsplit('/', 1)[-1]
                self.visual[name] = (mesh, _origin(visual.find('origin')))
        self.parent = {}
        for joint in tree.findall('joint'):
            child = joint.find('child').get('link')
            if not child.startswith(self.prefix + 'hand') or child == self.root:
                continue
            axis = joint.find('axis')
            self.parent[child] = (joint.find('parent').get('link'), _origin(joint.find('origin')),
                                  joint.get('name') if joint.get('type') != 'fixed' else None,
                                  np.fromstring(axis.get('xyz'), sep=' ') if axis is not None else None)
        self.limits = {}
        for joint in tree.findall('joint'):
            limit = joint.find('limit')
            if joint.get('name', '').startswith(self.prefix + 'hand_dof') and limit is not None:
                self.limits[joint.get('name')] = (float(limit.get('lower')), float(limit.get('upper')))

    def joint_name(self, real_name):
        return self.prefix + JOINTS[real_name]

    def link_poses(self, angles=None):
        """Pose of every hand link in the hand root frame for URDF joint angles [rad]."""
        angles = {self.joint_name(k): float(v) for k, v in (angles or {}).items()}
        for name, value in angles.items():
            lower, upper = self.limits[name]
            if not lower - 1e-9 <= value <= upper + 1e-9:
                raise ValueError(f'{name}={value:.4f} rad is outside the URDF range [{lower}, {upper}]')
        poses = {self.root: np.eye(4)}

        def resolve(link):
            if link not in poses:
                parent, origin, joint, axis = self.parent[link]
                transform = resolve(parent) @ origin
                if joint is not None:
                    transform = transform @ _axis_rotation(axis, angles.get(joint, 0.))
                poses[link] = transform
            return poses[link]

        for link in self.parent:
            resolve(link)
        return poses

    def moved_links(self):
        """Links below a posed joint (their meshes change with the hand pose)."""
        posed = {self.joint_name(k) for k in JOINTS}
        moved = set()
        for link in self.parent:
            walk = link
            while walk in self.parent:
                if self.parent[walk][2] in posed:
                    moved.add(link)
                    break
                walk = self.parent[walk][0]
        return moved

    def visual_poses(self, angles=None):
        poses = self.link_poses(angles)
        return {link: (mesh, poses[link] @ offset) for link, (mesh, offset) in self.visual.items()}


# One flat contact slab per hand, in the hand body frame.  It covers the palm
# pad and the fingertips, which are coplanar at MCP 18 deg, and stands 1 mm
# proud of the meshes.  MuJoCo gives a mesh/box pair a single contact point, so
# the 38 mesh hulls cannot hold a box against rotation; a box/box pair gets a
# full contact patch.  In EE axes the slab spans thumb x +-40 mm, hand z -72..105 mm
# and reaches 31 mm to the palm side.
PAD_SIZE = (.0215, .040, .0885)                 # half sizes along hand-body x, y, z
PAD_CENTRE = {'left': (.00608, -.0002, .1210), 'right': (.00608, .0002, .1210)}
PAD_FACE_IN_EE = {'left': -.031, 'right': .031}  # palm-side face, EE y [m]
# Soft-palm contact: a compliant pad resists twisting about its normal.  MuJoCo
# models that with condim 4 and a torsional coefficient equal to the effective
# patch radius; 2/3 of a 45 mm pad radius is 30 mm.
PAD_TORSION_RADIUS = .03


# Rounded contact for a toe-in angle that changes while the hands hold a box: a capsule across the
# palm whose axis is the line the hand turns about, so it meets the box face the same way at any angle.
PAD_CAPSULE_RADIUS = .020
PAD_CAPSULE_HALF = .040                          # half length of the straight part [m]


def capsule_axis_point(side, contact_z):
    """Point of the capsule axis in the EE frame: behind the palm-side pad face by the capsule radius."""
    face = PAD_FACE_IN_EE[side]
    return np.array([0., face - math.copysign(PAD_CAPSULE_RADIUS, face), float(contact_z)])


def add_pads(root, friction, toe_in=None, capsule_contact_z=None):
    """Add the contact slab to each rigid hand body of a vendor MJCF tree.

    ``toe_in`` maps a side to the EE-frame transform the hand targets are turned by
    (``retarget.toe_in_transform``).  The slab is turned back by the same amount, so it
    meets the box exactly as without toe-in: it stands for finger pads that stay flat on
    the box face while the palm is angled.  Real finger compliance is not modelled.
    ``capsule_contact_z`` replaces the slab by a capsule across the palm at that EE height.
    """
    names = []
    for side, short in SIDES.items():
        body = root.find(f".//body[@name='{short}_hand_rcv']")
        name = f'{short}_hand_pad'
        attributes = {'name': name, 'class': 'visual', 'type': 'box', 'size': ' '.join(map(str, PAD_SIZE)),
                      'pos': ' '.join(map(str, PAD_CENTRE[side])),
                      'friction': f'{friction} {PAD_TORSION_RADIUS} 0.0001', 'condim': '4', 'rgba': '0 0 0 0', 'group': '3'}
        ee = root.find(f".//body[@name='end_effector_{'left' if side == 'left' else 'right'}']")
        ee_in_hand = np.linalg.inv(_fixed(body)) @ _fixed(ee)              # both hang from the wrist connector
        if capsule_contact_z is not None:
            centre = capsule_axis_point(side, capsule_contact_z)
            ends = [(ee_in_hand @ np.r_[centre + [offset, 0., 0.], 1.])[:3] for offset in (-PAD_CAPSULE_HALF, PAD_CAPSULE_HALF)]
            attributes.update(type='capsule', size=str(PAD_CAPSULE_RADIUS), fromto=' '.join(f'{v:.9g}' for v in np.r_[ends[0], ends[1]]))
            del attributes['pos']
        elif toe_in is not None and not np.allclose(toe_in[side], np.eye(4)):
            pad = np.eye(4); pad[:3, 3] = PAD_CENTRE[side]
            pad = ee_in_hand @ np.linalg.inv(toe_in[side]) @ np.linalg.inv(ee_in_hand) @ pad
            attributes['pos'] = ' '.join(f'{v:.9g}' for v in pad[:3, 3])
            attributes['quat'] = ' '.join(f'{v:.9g}' for v in matrix_quat(pad[:3, :3]))
        ET.SubElement(body, 'geom', attributes)
        names.append(name)
    return names


def pad_points_in_ee(root):
    """Outline points of each hand's contact pad in its EE frame: the 8 corners of the slab, or for a capsule
    its two end centres moved out by the radius along every axis."""
    outline = {}
    for side, short in SIDES.items():
        body = root.find(f".//body[@name='{short}_hand_rcv']")
        geom = body.find(f"geom[@name='{short}_hand_pad']")
        if geom is None:
            raise ValueError(f'The {side} hand has no contact pad')
        ee = root.find(f".//body[@name='end_effector_{side}']")
        to_ee = np.linalg.inv(_fixed(ee)) @ _fixed(body)                   # both hang from the wrist connector
        if geom.get('type') == 'capsule':
            ends = np.fromstring(geom.get('fromto'), sep=' ').reshape(2, 3)
            radius = float(geom.get('size'))
            points = np.array([end + radius * axis for end in ends for axis in np.vstack([np.eye(3), -np.eye(3)])])
        else:
            half = np.fromstring(geom.get('size'), sep=' ')
            rotation = _quat_matrix(np.fromstring(geom.get('quat', '1 0 0 0'), sep=' '))
            corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
            points = corners @ rotation.T + np.fromstring(geom.get('pos'), sep=' ')
        outline[side] = points @ to_ee[:3, :3].T + to_ee[:3, 3]
    return outline


def load_pose(path=POSE_FILE):
    """Simulated URDF angles per side: {'left': {'mcp': rad, ...}, 'right': {...}}."""
    data = json.loads(Path(path).read_text())
    return {side: {joint: float(entry['sim_rad']) for joint, entry in joints.items()}
            for side, joints in data['joints'].items()}


def apply_mjcf(root, pose):
    """Re-pose the fixed hand meshes of a vendor MJCF tree in place."""
    report = {}
    for side, short in SIDES.items():
        hand = Hand(side)
        body = root.find(f".//body[@name='{short}_hand_rcv']")
        if body is None or body.findall('body') or body.findall('joint'):
            raise ValueError(f'Expected the rigid vendor {side} hand')
        zero = hand.visual_poses(VENDOR_BAKED)
        target = hand.visual_poses(pose[side])
        moved = hand.moved_links()
        geoms = body.findall('geom')
        if len(geoms) != len(zero):
            raise ValueError(f'{side} hand: {len(geoms)} vendor meshes but {len(zero)} URDF visuals')
        unmatched = dict(zero)
        changed = []
        for geom in geoms:
            position = np.fromstring(geom.get('pos', '0 0 0'), sep=' ')
            rotation = _quat_matrix(np.fromstring(geom.get('quat', '1 0 0 0'), sep=' '))
            # Links reuse mesh files and share frames, so identify by mesh and vendor pose.
            mesh_name = geom.get('mesh', '').removeprefix(f'rcvh_{short}_')
            match = [link for link, (mesh, pose_zero) in unmatched.items()
                     if mesh.rsplit('.', 1)[0].lower() == mesh_name
                     and np.allclose(pose_zero[:3, 3], position, atol=2e-5)
                     and np.allclose(pose_zero[:3, :3], rotation, atol=2e-4)]
            if len(match) != 1:
                raise ValueError(f"{geom.get('name')} does not match exactly one URDF hand link: {match}")
            link = match[0]; del unmatched[link]
            if link in moved:
                new = target[link][1]
                geom.set('pos', ' '.join(format(v, '.17g') for v in new[:3, 3]))
                geom.set('quat', ' '.join(format(v, '.17g') for v in matrix_quat(new[:3, :3])))
                changed.append(geom.get('name'))
        report[side] = dict(angles_rad=pose[side], moved_geoms=changed)
    return report


def usd_name(link):
    """Prim name the URDF importer gave a link (non-ASCII bytes become underscores)."""
    return ''.join(ch if ch.isascii() else '_' * len(ch.encode('utf-8')) for ch in link)


def usd_overrides(pose):
    """Root-frame poses of the links that move, keyed by USD prim name."""
    overrides = {}
    for side in SIDES:
        hand = Hand(side)
        poses = hand.link_poses(pose[side])
        overrides[usd_name(hand.root)] = {usd_name(link): poses[link] for link in hand.moved_links()}
    return overrides


def keypoints(side, pose):
    """Mesh frame origins in the wrist (EE-parallel) frame, for symmetry checks."""
    hand = Hand(side)
    # Vendor attachment of the hand root in the wrist connector frame.
    mount = np.eye(4)
    mount[:3, 3] = [0, .0109 if side == 'left' else -.0109, -.024]
    mount[:3, :3] = _quat_matrix([0, .70710678, -.70710678 if side == 'left' else .70710678, 0])
    return {link.split('hand_', 1)[1]: (mount @ transform)[:3, 3]
            for link, (_, transform) in hand.visual_poses(pose[side]).items()}


def symmetry_error(pose):
    """Largest left/right mismatch [m] after mirroring the right hand across y = 0."""
    left, right = keypoints('left', pose), keypoints('right', pose)
    if left.keys() != right.keys():
        raise ValueError('Left and right hand link sets differ')
    mirror = np.array([1., -1., 1.])
    return float(max(np.linalg.norm(left[name] - mirror * right[name]) for name in left))
