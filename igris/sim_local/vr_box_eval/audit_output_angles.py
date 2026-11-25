"""Compare recorded native EE poses with rigid FK at motor/output joint angles."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from igris_ik.constants import URDF_JOINT_NAMES
from igris_ik.sim.mujoco_model import load_fixed_hand_model
from render_results import model_for


def audit(out):
    record = np.load(out / 'physics.npz')
    native = model_for(out)
    measured = mujoco.MjData(native)
    rigid = load_fixed_hand_model()
    fk = mujoco.MjData(rigid.model)
    addresses = [native.jnt_qposadr[native.joint(name).id] for name in URDF_JOINT_NAMES]
    offset_ids = [mujoco.mj_name2id(native, mujoco.mjtObj.mjOBJ_JOINT, name + '_backlash') for name in URDF_JOINT_NAMES]

    def pose(model, data, name):
        base, body = model.body('pelvis').id, model.body(name).id
        rotation = Rotation.from_matrix(data.xmat[base].reshape(3, 3))
        return (rotation.inv().apply(data.xpos[body] - data.xpos[base]),
                rotation.inv() * Rotation.from_matrix(data.xmat[body].reshape(3, 3)))

    errors = {'motor_only': [], 'motor_plus_offset': []}
    for row in record['qpos'][::10]:
        measured.qpos[:] = row
        mujoco.mj_forward(native, measured)
        offsets = np.array([row[native.jnt_qposadr[j]] if j >= 0 else 0. for j in offset_ids])
        for label, q in [('motor_only', row[addresses]), ('motor_plus_offset', row[addresses] + offsets)]:
            fk.qpos[list(rigid.qpos_addresses)] = q
            mujoco.mj_forward(rigid.model, fk)
            for side in ('left', 'right'):
                p, r = pose(native, measured, 'end_effector_' + side)
                target, rotation = pose(rigid.model, fk, 'end_effector_' + side)
                errors[label].append([np.linalg.norm(p - target) * 1000,
                                      np.rad2deg((rotation.inv() * r).magnitude())])
    result = {label: {'samples': len(e), 'position_mm_p95_max': np.percentile(e, [95, 100], axis=0)[:, 0].tolist(),
                      'orientation_deg_p95_max': np.percentile(e, [95, 100], axis=0)[:, 1].tolist()}
              for label, e in errors.items()}
    assert result['motor_plus_offset']['position_mm_p95_max'][1] < .01
    assert result['motor_plus_offset']['orientation_deg_p95_max'][1] < .001
    (out / 'output_angle_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    audit(parser.parse_args().out.resolve())
