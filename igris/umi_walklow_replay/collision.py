"""Self-collision clearance of the upper body from the robot's collision spheres, in numpy.

The spheres, the link pairs to check and the pairs to ignore come from the cuRobo robot
description in ``igris/igris_curobo`` (its link frames are the vendor model's body frames;
``l_hand_root``/``r_hand_root`` are the vendor ``l_hand_rcv``/``r_hand_rcv``).  The simulator
container has neither cuRobo nor scipy, so the description is exported once on the host
(``export``) and evaluated here.  Only pairs with a link the replay moves are checked: the
legs belong to the walking controller.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .kinematics import JOINTS, VENDOR_MJCF, _fixed, axis_rotation

ROOT = Path(__file__).resolve().parents[2]
DESCRIPTION = ROOT / 'igris/igris_curobo/src/igris_ik/data/curobo/igris_c_rcv_hand_pelvis.yml'
HAND_LINKS = {'l_hand_root': 'l_hand_rcv', 'r_hand_root': 'r_hand_rcv'}       # description name -> vendor body
MOVED = ('waist_', 'torso', 'neck', 'head', 'l_shoulder', 'r_shoulder', 'l_upper_arm', 'r_upper_arm', 'l_elbow', 'r_elbow',
         'l_forearm', 'r_forearm', 'l_wrist', 'r_wrist', 'l_hand', 'r_hand')
BUFFER_M = .0025                  # added to every sphere radius, as in igris_ik.RobotGeometry
CUTOFF_M = .04                    # gaps are resolved below this; anything wider is reported as this


def export(target, description=DESCRIPTION):
    """Write the sphere model as plain arrays.  Needs PyYAML; run on the host."""
    import yaml
    model = yaml.safe_load(Path(description).read_text())['kinematics']
    names = list(model['collision_link_names']); ignore = model['self_collision_ignore']
    arrays = {}
    for name in names:
        spheres = model['collision_spheres'][name]
        arrays[f'centres/{name}'] = np.array([s['center'] for s in spheres], dtype=float)
        arrays[f'radii/{name}'] = np.array([s['radius'] for s in spheres], dtype=float)
    pairs = [(i, j) for i, a in enumerate(names) for j, b in enumerate(names) if j > i
             and b not in ignore.get(a, []) and a not in ignore.get(b, [])
             and (a.startswith(MOVED) or b.startswith(MOVED))]
    np.savez_compressed(target, links=np.array(names), pairs=np.array(pairs, dtype=int), description=str(description), **arrays)
    return target


class Tree:
    """Pelvis-frame poses of vendor-model bodies for a 31-joint configuration (passive joints at zero)."""

    def __init__(self, bodies, path=VENDOR_MJCF):
        root = ET.parse(path).getroot()
        parents = {child: parent for parent in root.iter() for child in parent}
        self.order = []                               # (name, parent name, fixed 4x4, joint index or None, axis)
        seen = {'pelvis'}

        def visit(name):
            if name in seen:
                return
            body = root.find(f".//body[@name='{name}']")
            if body is None:
                raise ValueError(f'No body {name} in the vendor model')
            parent = parents[body].get('name')
            visit(parent)
            hinges = [(JOINTS.index(j.get('name')), np.fromstring(j.get('axis'), sep=' ')) for j in body.findall('joint')
                      if j.get('name') in JOINTS]
            if len(hinges) > 1:
                raise ValueError(f'More than one commanded joint on {name}')
            joint, axis = hinges[0] if hinges else (None, None)
            self.order.append((name, parent, _fixed(body), joint, None if axis is None else axis / np.linalg.norm(axis)))
            seen.add(name)

        for name in bodies:
            visit(name)

    def poses(self, q):
        out = {'pelvis': np.eye(4)}
        for name, parent, fixed, joint, axis in self.order:
            pose = out[parent] @ fixed
            if joint is not None:
                turn = np.eye(4); turn[:3, :3] = axis_rotation(axis, q[joint])
                pose = pose @ turn
            out[name] = pose
        return out


class SelfCollision:
    """Smallest gap between the sphere sets of the checked link pairs."""

    def __init__(self, path, buffer=BUFFER_M, cutoff=CUTOFF_M):
        data = np.load(path)
        self.links = [str(name) for name in data['links']]
        self.centres = [data[f'centres/{name}'] for name in self.links]
        self.radii = [data[f'radii/{name}'] + buffer for name in self.links]
        self.pairs = [tuple(int(v) for v in pair) for pair in data['pairs']]
        self.bodies = [HAND_LINKS.get(name, name) for name in self.links]
        self.tree = Tree(self.bodies)
        self.cutoff = float(cutoff)

    def near_pairs(self, q, reach=None, per_pair=1):
        """Closest sphere pairs of every checked link pair nearer than ``reach`` (default: the cutoff).

        ``per_pair`` is how many spheres of the first link are reported per link pair, each with its
        nearest sphere of the second link, nearest first.  One is enough to measure the gap; a bound
        on motion needs several, or the unbounded neighbours of the closest pair close in instead.

        Each entry is (gap [m], link a, sphere centre a, link b, sphere centre b), centres in the
        pelvis frame; negative gaps overlap.  Only spheres within ``reach`` of the other link's
        bounding box are compared, which keeps the 3 666-sphere model fast enough for the control loop.
        """
        reach = self.cutoff if reach is None else float(reach)
        poses = self.tree.poses(np.asarray(q, dtype=float))
        points = []; low = []; high = []
        for body, centres, radii in zip(self.bodies, self.centres, self.radii):
            pose = poses[body]
            p = centres @ pose[:3, :3].T + pose[:3, 3]
            points.append(p); low.append((p - radii[:, None]).min(axis=0)); high.append((p + radii[:, None]).max(axis=0))
        found = []
        for a, b in self.pairs:
            if float(np.maximum(low[a] - high[b], low[b] - high[a]).max()) >= reach:      # bounding boxes too far apart
                continue
            near_a = np.flatnonzero(((points[a] + (self.radii[a] + reach)[:, None] >= low[b])
                                     & (points[a] - (self.radii[a] + reach)[:, None] <= high[b])).all(axis=1))
            near_b = np.flatnonzero(((points[b] + (self.radii[b] + reach)[:, None] >= low[a])
                                     & (points[b] - (self.radii[b] + reach)[:, None] <= high[a])).all(axis=1))
            if not len(near_a) or not len(near_b):
                continue
            gap = (np.linalg.norm(points[a][near_a][:, None] - points[b][near_b][None], axis=2)
                   - self.radii[a][near_a][:, None] - self.radii[b][near_b][None])
            nearest = np.argmin(gap, axis=1); smallest = gap[np.arange(len(near_a)), nearest]
            for i in np.argsort(smallest)[:per_pair]:
                if smallest[i] < reach:
                    found.append((float(smallest[i]), self.links[a], points[a][near_a[i]], self.links[b], points[b][near_b[nearest[i]]]))
        return found

    def clearance(self, q):
        """(gap [m], link a, link b) of the closest checked pair, or (cutoff, None, None) when none is that near."""
        found = self.near_pairs(q)
        if not found:
            return self.cutoff, None, None
        gap, a, _, b, _ = min(found, key=lambda entry: entry[0])
        return gap, a, b
