"""Export read-only live box guides from the existing native scene and grasp geometry."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from igris_ik.gui.vr_feedback import pose_dict
from render_results import model_for


def export(out, reference, destination):
    m = model_for(out)
    meta = json.loads(reference.read_text())
    base = Rotation.from_quat(meta['base_quat_xyzw'])
    poses = {}
    for i, side in enumerate(('left', 'right')):
        # The geometry's unit seating rotation aligns palm and proximal MCP plate.
        # Remove the virtual squeeze depth: this is a contact guide, not a force command.
        seat = Rotation.from_rotvec(meta['palm_seat']['rotvecs'][i])
        pivot = np.array(meta['palm_seat']['pivots'][i])
        point = pivot + seat.apply(np.array(meta['closed_ee'][i]) - pivot)
        point = base.apply(point) + meta['base_position']
        point[1] += (1 if i == 0 else -1) * meta['squeeze_m']
        transform = np.eye(4)
        transform[:3, 3] = point - [meta['box_radius'], 0, meta['box_z']]
        transform[:3, :3] = (base * seat * Rotation.from_quat(meta['grasp_quats_xyzw'][i])).as_matrix()
        poses[side] = pose_dict(transform)
    box_body = m.body('eval_box').id
    base_body = m.body('pelvis').id
    joints = [int(m.body_jntadr[body]) for body in (base_body, box_body)]
    assert all(m.jnt_type[j] == 0 for j in joints), 'guide requires free pelvis and box joints'
    guide = dict(physics_path=str((out / 'physics.bin').resolve()), nq_nv=[m.nq, m.nv],
                 base_qposadr=int(m.jnt_qposadr[joints[0]]), box_qposadr=int(m.jnt_qposadr[joints[1]]),
                 size=(2 * m.geom('eval_box_geom').size).tolist(), poses_in_box=poses,
                 reference=str(reference.resolve()),
                 meaning='Geometric palm/MCP contact alignment; no preload or grasp-success guarantee')
    destination.write_text(json.dumps(guide, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sim_directory', type=Path)
    parser.add_argument('grasp_metadata', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    export(args.sim_directory, args.grasp_metadata, args.output)
