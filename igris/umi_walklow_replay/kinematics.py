"""Pelvis-rooted upper-body kinematics read from the vendor MuJoCo model.

Only numpy is used so the same code runs inside the simulator container.
Angles are SDK joint-space (PJS) radians in the 31-joint SDK order.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VENDOR_MJCF = ROOT / 'igris/sim_local/igris_c_v2_rcv_hand.xml'
VENDOR_LIMITS = ROOT / 'igris/igris_curobo/src/igris_ik/data/vendor/joint_limits.json'

JOINTS = (['waist_yaw', 'waist_roll', 'waist_pitch']
          + [f'{s}_{j}' for s in ('l', 'r') for j in
             ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee_pitch', 'ankle_pitch', 'ankle_roll')]
          + [f'{s}_{j}' for s in ('l', 'r') for j in
             ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow_pitch', 'wrist_yaw', 'wrist_roll', 'wrist_pitch')]
          + ['neck_yaw', 'neck_pitch'])
ACTIVE = [0, 1, 2, *range(15, 29)]          # waist 3 + left arm 7 + right arm 7
END_EFFECTORS = ('end_effector_left', 'end_effector_right')
ELBOWS = ('l_elbow_bracket', 'r_elbow_bracket')
TORSO = 'torso'
# Shoulder joint origins in the torso frame [m]; the HL-IK elbow prior measures from them.
SHOULDERS = np.array([[0., .171, .266], [0., -.171, .266]])


def quat_matrix(q):
    """Rotation matrix of a w,x,y,z quaternion."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def axis_rotation(axis, angle):
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + s * k + (1 - c) * (k @ k)


def matrix_quat(matrix):
    """w,x,y,z quaternion of a rotation matrix."""
    m = np.asarray(matrix, dtype=float)
    trace = np.trace(m)
    if trace > 0:
        scale = 2 * np.sqrt(trace + 1)
        q = [scale / 4, (m[2, 1] - m[1, 2]) / scale, (m[0, 2] - m[2, 0]) / scale, (m[1, 0] - m[0, 1]) / scale]
    else:
        i = int(np.argmax(np.diag(m))); j, k = (i + 1) % 3, (i + 2) % 3
        scale = 2 * np.sqrt(1 + m[i, i] - m[j, j] - m[k, k])
        q = [0., 0., 0., 0.]
        q[0] = (m[k, j] - m[j, k]) / scale; q[1 + i] = scale / 4
        q[1 + j] = (m[j, i] + m[i, j]) / scale; q[1 + k] = (m[k, i] + m[i, k]) / scale
    q = np.array(q)
    return q / np.linalg.norm(q)


def rotation_vector(matrix):
    """Axis-angle vector of a rotation matrix, stable near 0 and pi."""
    cosine = np.clip((np.trace(matrix) - 1) / 2, -1, 1)
    angle = np.arccos(cosine)
    vector = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]])
    if angle < 1e-7:
        return vector / 2
    if np.pi - angle < 1e-5:
        # Near pi the antisymmetric part vanishes; take the axis from the diagonal.
        axis = np.sqrt(np.maximum((np.diag(matrix) + 1) / 2, 0))
        index = int(np.argmax(axis))
        signs = np.sign(matrix[index] + matrix[:, index]); signs[index] = 1
        return angle * axis * np.where(signs == 0, 1, signs)
    return vector * angle / (2 * np.sin(angle))


def _fixed(body):
    transform = np.eye(4)
    transform[:3, 3] = np.fromstring(body.get('pos', '0 0 0'), sep=' ')
    transform[:3, :3] = quat_matrix(np.fromstring(body.get('quat', '1 0 0 0'), sep=' '))
    return transform


def coupled_halfspaces(path=VENDOR_LIMITS):
    """Vendor convex wrist/waist safe sets as (joint pair indices, normals, offsets)."""
    sets = []
    for pair in json.loads(Path(path).read_text())['coupled_limits']:
        vertices = np.array(pair['vertices'], dtype=float)
        centre = vertices.mean(axis=0)
        vertices = centre + .95 * (vertices - centre)   # vendor HARD safe-set ratio
        edge = np.roll(vertices, -1, axis=0) - vertices
        normal = np.column_stack((edge[:, 1], -edge[:, 0]))
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        offset = np.sum(normal * vertices, axis=1)
        if np.any(vertices @ normal.T - offset > 1e-8):
            raise ValueError('Vendor safe polygon must be convex and counter-clockwise')
        sets.append(([JOINTS.index(name) for name in pair['joints']], normal, offset))
    return sets


class UpperBody:
    """Forward kinematics and Jacobian of both hands from the pelvis."""

    def __init__(self, chains, lower, upper):
        self.chains = chains            # per hand: list of (fixed 4x4, sdk joint index or None, axis, body name)
        self.lower = np.asarray(lower, dtype=float)   # 31 values, SDK order
        self.upper = np.asarray(upper, dtype=float)
        self.columns = {joint: column for column, joint in enumerate(ACTIVE)}

    @classmethod
    def from_mjcf(cls, path=VENDOR_MJCF):
        root = ET.parse(path).getroot()
        parents = {child: parent for parent in root.iter() for child in parent}
        lower = np.full(len(JOINTS), -np.inf); upper = np.full(len(JOINTS), np.inf)
        for joint in root.iter('joint'):
            if joint.get('name') in JOINTS and joint.get('range'):
                lower[JOINTS.index(joint.get('name'))], upper[JOINTS.index(joint.get('name'))] = \
                    np.fromstring(joint.get('range'), sep=' ')
        chains = []
        for name in END_EFFECTORS:
            body = root.find(f".//body[@name='{name}']")
            if body is None:
                raise ValueError(f'Missing body {name}')
            path_bodies = []
            while body.get('name') != 'pelvis':
                path_bodies.append(body)
                body = parents[body]
                if body.tag != 'body':
                    raise ValueError(f'{name} is not below the pelvis')
            chain = []
            for body in reversed(path_bodies):
                fixed = _fixed(body)
                hinges = []
                for joint in body.findall('joint'):
                    if joint.get('name', '').endswith('_backlash'):
                        continue    # passive +-0.005..0.01 rad vendor play, not commanded
                    if joint.get('name') not in JOINTS or joint.get('pos') not in (None, '0 0 0'):
                        raise ValueError(f"Unsupported joint in hand chain: {joint.get('name')}")
                    if JOINTS.index(joint.get('name')) not in ACTIVE:
                        raise ValueError(f"Unowned joint in hand chain: {joint.get('name')}")
                    axis = np.fromstring(joint.get('axis'), sep=' ')
                    hinges.append((JOINTS.index(joint.get('name')), axis / np.linalg.norm(axis)))
                if len(hinges) > 1:
                    raise ValueError('Expected at most one commanded joint per body')
                chain.append((fixed, *(hinges[0] if hinges else (None, None)), body.get('name')))
            chains.append(chain)
        if not np.isfinite(lower[ACTIVE]).all():
            raise ValueError('Missing joint range for an owned joint')
        return cls(chains, lower, upper)

    def _chain(self, chain, q, jacobian, marks=None):
        """Walk one hand chain.  ``marks`` collects body name -> (transform, joints passed so far)."""
        transform = np.eye(4)
        axes = []
        for fixed, joint, axis, name in chain:
            transform = transform @ fixed
            if joint is not None:
                if jacobian is not None or marks is not None:
                    axes.append((joint, transform[:3, :3] @ axis, transform[:3, 3].copy()))
                rotation = np.eye(4); rotation[:3, :3] = axis_rotation(axis, q[joint])
                transform = transform @ rotation
            if marks is not None and name in marks:
                marks[name] = (transform.copy(), list(axes))
        if jacobian is not None:
            for joint, world_axis, origin in axes:
                column = self.columns[joint]
                jacobian[:3, column] = np.cross(world_axis, transform[:3, 3] - origin)
                jacobian[3:, column] = world_axis
        return transform

    def fk(self, q):
        """Pelvis-frame 4x4 poses of the left and right end effectors."""
        q = np.asarray(q, dtype=float)
        return tuple(self._chain(chain, q, None) for chain in self.chains)

    def fk_jacobian(self, q):
        """Poses and the 12x17 Jacobian (rows: left v, left w, right v, right w)."""
        q = np.asarray(q, dtype=float)
        jacobian = np.zeros((12, len(ACTIVE)))
        left = self._chain(self.chains[0], q, jacobian[:6])
        right = self._chain(self.chains[1], q, jacobian[6:])
        return left, right, jacobian

    def _until(self, chain, q, last):
        """Pelvis-frame transforms of the chain's bodies up to and including ``last``."""
        transform = np.eye(4); seen = {}
        for fixed, joint, axis, name in chain:
            transform = transform @ fixed
            if joint is not None:
                rotation = np.eye(4); rotation[:3, :3] = axis_rotation(axis, q[joint])
                transform = transform @ rotation
            seen[name] = transform
            if name == last:
                return seen
        raise ValueError(f'No body {last} in the hand chain')

    def torso(self, q):
        """Pelvis-frame pose of the torso body."""
        return self._until(self.chains[0], np.asarray(q, dtype=float), TORSO)[TORSO]

    def upper_arm(self, q, side):
        """Shoulder-joint origin and elbow origin of one arm in the pelvis frame."""
        seen = self._until(self.chains[side], np.asarray(q, dtype=float), ELBOWS[side])
        torso = seen[TORSO]
        return torso[:3, :3] @ SHOULDERS[side] + torso[:3, 3], seen[ELBOWS[side]][:3, 3].copy()

    def arm_frames(self, q):
        """Torso, elbow and hand frames in the pelvis frame, with the elbow-position Jacobian.

        Returns ``(torso 4x4, elbows [2 x 4x4], hands [2 x 4x4], jacobian 6x17)``; the
        Jacobian rows are the left then right elbow origin velocities.
        """
        q = np.asarray(q, dtype=float)
        jacobian = np.zeros((6, len(ACTIVE)))
        elbows, hands, torso = [], [], None
        for side, chain in enumerate(self.chains):
            marks = {TORSO: None, ELBOWS[side]: None}
            hands.append(self._chain(chain, q, None, marks))
            if marks[TORSO] is None or marks[ELBOWS[side]] is None:
                raise ValueError('Hand chain lacks the torso or elbow body')
            torso = marks[TORSO][0]
            elbow, joints = marks[ELBOWS[side]]
            elbows.append(elbow)
            for joint, world_axis, origin in joints:
                jacobian[3 * side:3 * side + 3, self.columns[joint]] = np.cross(world_axis, elbow[:3, 3] - origin)
        return torso, elbows, hands, jacobian
