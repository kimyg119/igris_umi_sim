"""Bounded, simulation-only arm gain comparison through the existing vendor trial."""
import argparse
import csv
import json
import shutil
import struct
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

import evaluate
from igris_ik.constants import URDF_JOINT_NAMES
from igris_ik.gui.robot_monitor import RobotMonitor
from igris_ik.realtime.robot_control import ARM_GAIN_INIT, ARM_GROUPS
from render_results import model_for, process


PROFILES = {
    'vendor': dict(ARM_GAIN_INIT),
    'wrist45': {**ARM_GAIN_INIT, 'wrist': (45., 3.)},
    'wrist45_damped': {**ARM_GAIN_INIT, 'wrist': (45., 4.5)},
    'wrist60_damped': {**ARM_GAIN_INIT, 'wrist': (60., 4.5)},
    'arm150': {'shoulder': (150., 7.5), 'elbow': (150., 7.5), 'wrist': (45., 4.5)},
    'arm150_wrist60': {'shoulder': (150., 7.5), 'elbow': (150., 7.5), 'wrist': (60., 4.5)},
    'arm180_wrist60': {'shoulder': (180., 9.), 'elbow': (180., 9.), 'wrist': (60., 4.5)},
    'arm180': {'shoulder': (180., 9.), 'elbow': (180., 9.), 'wrist': (45., 4.5)},
}
# Keep historical Kp presets, with the user's maximum arm damping applied explicitly.
PROFILES = {name:{group:(kp,min(kd,5.)) for group,(kp,kd) in gains.items()} for name,gains in PROFILES.items()}


def hand_inertia_audit(out, apply=False):
    """Match only rigid wrist descendants; do not double-count moving wrist linkages."""
    model = model_for(out)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    source = evaluate.ROOT / 'igris_c_description_public/urdf/igris_c_rcv_hand_ctrl.urdf'
    shutil.copy2(source, out/'original_hand_ctrl.urdf')
    tree = ET.parse(source)
    audit = {}
    for side in ('l', 'r'):
        name = side + '_wrist_connector'
        wrist = model.body(name).id
        bodies = [wrist]
        for body in range(wrist + 1, model.nbody):
            if model.body_parentid[body] in bodies and model.body_jntnum[body] == 0:
                bodies.append(body)
        mass = model.body_mass[bodies].sum()
        center = (model.body_mass[bodies, None] * data.xipos[bodies]).sum(0) / mass
        inertia = np.zeros((3, 3))
        for body in bodies:
            rotation = data.ximat[body].reshape(3, 3)
            delta = data.xipos[body] - center
            inertia += rotation @ np.diag(model.body_inertia[body]) @ rotation.T
            inertia += model.body_mass[body] * (delta @ delta * np.eye(3) - np.outer(delta, delta))
        rotation = data.xmat[wrist].reshape(3, 3)
        center = rotation.T @ (center - data.xpos[wrist])
        inertia = rotation.T @ inertia @ rotation
        original = tree.find(f'link[@name="{name}"]/inertial')
        old_center = np.fromstring(original.find('origin').get('xyz'), sep=' ')
        audit[side] = dict(old_mass_kg=float(original.find('mass').get('value')),
                           native_mass_kg=float(mass), old_com_m=old_center.tolist(),
                           native_com_m=center.tolist(), com_difference_mm=float(np.linalg.norm(center-old_center)*1000),
                           native_inertia_kg_m2=inertia.tolist(), bodies=[model.body(i).name for i in bodies])
        original.find('mass').set('value', str(mass))
        original.find('origin').set('xyz', ' '.join(map(str, center)))
        original.find('origin').set('rpy', '0 0 0')
        for key, (i, j) in zip(('ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz'), ((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))):
            original.find('inertia').set(key, str(inertia[i,j]))
    (out / 'hand_inertia_audit.json').write_text(json.dumps(audit, indent=2))
    if apply:
        # Keep the vendor's 33-link / 31-axis RBDL topology and zero inertial RPY.
        dest = out / 'matched_hand_ctrl.urdf'
        tree.write(dest, encoding='unicode')
        config_path = out / 'robot.yaml'
        config = yaml.safe_load(config_path.read_text())
        scene_path = Path(config['igris_c']['description']['mujoco']['robot_model_path'])
        config['igris_c']['description']['urdf']['robot_model_path'] = str(scene_path.with_name(dest.name))
        config_path.write_text(yaml.safe_dump(config))
    return audit


def score(out):
    process(out, metrics_only=True)
    meta = json.loads((out / 'metadata.json').read_text())
    physics = np.load(out / 'physics.npz')
    model = model_for(out)
    result = {'path': str(out), 'metrics': json.loads((out/'metrics.json').read_text()),
              'task': json.loads((out/'contact_task.json').read_text())}
    rows = list(csv.DictReader(next(out.glob('*.csv')).open()))
    time = np.array([float(row['t_wall'])-meta['start_wall'] for row in rows])
    inputs = json.loads((out/'input.json').read_text())
    index = np.clip(np.searchsorted([x['t'] for x in inputs], time, side='right')-1, 0, len(inputs)-1)
    phases = np.array([inputs[i]['stage'] for i in index])
    for phase, mask in [('all', (time>=3)&(time<=meta['duration']-2)),
                        ('contact', np.isin(phases, ['squeeze','seat_palm','probe_lift','lift','carry_left','lower'])),
                        ('carry', phases=='carry_left')]:
        if not mask.any():
            continue
        values = {}
        for group, indices in ARM_GROUPS.items():
            def vector(prefix):
                return np.array([[float(row[f'{prefix}_{i}']) for i in indices] for row in rows])[mask]
            error = np.rad2deg(vector('cmd_q')-vector('q'))
            if phase == 'all':
                expected = meta.get('sim_arm_gains', ARM_GAIN_INIT)[group]
                assert np.allclose(vector('kp'), expected[0]), (group, 'Kp not applied')
                assert np.allclose(vector('kd'), expected[1]), (group, 'Kd not applied')
            values[group] = dict(motor_error_rms_deg=float(np.sqrt(np.mean(error**2))),
                                 motor_error_p95_deg=float(np.percentile(abs(error),95)),
                                 speed_p95_deg_s=float(np.percentile(abs(np.rad2deg(vector('dq'))),95)),
                                 kp_range=[float(vector('kp').min()),float(vector('kp').max())],
                                 kd_range=[float(vector('kd').min()),float(vector('kd').max())])
        result[phase] = values
    actual, commanded = mujoco.MjData(model), mujoco.MjData(model)
    upper = [0,1,2,*range(15,29)]
    addresses = [model.jnt_qposadr[model.joint(URDF_JOINT_NAMES[i]).id] for i in upper]
    tools = [model.body('end_effector_'+side).id for side in ('left','right')]
    ee = []
    for k in range(0,len(physics['t']),3):
        sec = physics['t'][k]
        index = int(np.clip(np.searchsorted(time,sec,side='right')-1,0,len(rows)-1))
        if phases[index] != 'carry_left':
            continue
        actual.qpos[:] = physics['qpos'][k]
        commanded.qpos[:] = actual.qpos
        commanded.qpos[addresses] = [float(rows[index][f'cmd_q_{i}']) for i in upper]
        mujoco.mj_forward(model,actual);mujoco.mj_forward(model,commanded)
        for body in tools:
            difference = actual.xmat[body].reshape(3,3).T @ commanded.xmat[body].reshape(3,3)
            ee.append([1000*np.linalg.norm(actual.xpos[body]-commanded.xpos[body]),
                       np.rad2deg(Rotation.from_matrix(difference).magnitude())])
    if ee:
        result['carry_ee_tracking'] = dict(samples=len(ee),position_p95_mm=float(np.percentile(ee,95,axis=0)[0]),
            orientation_p95_deg=float(np.percentile(ee,95,axis=0)[1]),
            convention='logged motor command FK versus native pose; same measured free base and backlash offsets')
    result['backlash'] = {}
    for group, indices in ARM_GROUPS.items():
        joints = [model.joint(URDF_JOINT_NAMES[i]+'_backlash').id for i in indices]
        angles = physics['qpos'][:,model.jnt_qposadr[joints]]
        limits = model.jnt_range[joints,1]
        result['backlash'][group] = dict(p95_deg=float(np.rad2deg(np.percentile(abs(angles),95))),
            max_deg=float(np.rad2deg(abs(angles).max())),
            outside_nominal_range_fraction=float(np.mean(abs(angles)>limits+.001)))
    path = out/'physics.bin.effort'
    if path.exists():
        raw = path.read_bytes();nu,nv = struct.unpack('ii',raw[:8]);width = 2+2*nu+nv
        assert (len(raw)-8) % (8*width) == 0
        effort = np.frombuffer(raw[8:],np.float64).reshape(-1,width)
        effort = effort[(effort[:,0]>=meta['start_wall']+3)&(effort[:,0]<=meta['start_wall']+meta['duration']-2)]
        result['motor_torque'] = {}
        for group, indices in (ARM_GROUPS.items() if len(effort) else []):
            limits = model.actuator_ctrlrange[indices,1]
            control = effort[:,2+np.array(indices)]
            force = effort[:,2+nu+np.array(indices)]
            # These vendor trials use torque actuators; servo channels must be inactive.
            servo_force = effort[:,2+nu+np.array(indices)+31] + effort[:,2+nu+np.array(indices)+62]
            assert np.max(abs(servo_force)) < 1e-8, 'Active servo force needs separate saturation accounting'
            result['motor_torque'][group] = dict(force_p95_Nm=float(np.percentile(abs(force),95)),
                force_max_Nm=float(abs(force).max()),ctrl_limit_fraction=float(np.mean(abs(control)>=limits*.99)),
                actuator_names=[model.actuator(i).name for i in indices], limits_Nm=limits.tolist())
    result['teleop'] = json.loads((out/'teleop_summary.json').read_text())
    (out/'gain_score.json').write_text(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, default='vendor',help='historical Kp presets; every arm Kd is capped at 5')
    parser.add_argument('--arm400-kd',type=float,help='override profile: Kp 400 on all arm axes, with this Kd (0 < Kd <= 5)')
    parser.add_argument('--mass', type=float, default=.01)
    parser.add_argument('--radius', type=float, default=.4)
    parser.add_argument('--height', type=float, default=.9, help='initial box center height in metres')
    parser.add_argument('--hand-face-rotation', type=float, choices=(0.,45.), default=45., help='finger rotation within the box side face, mirrored left/right')
    parser.add_argument('--match-hand-inertia', action='store_true')
    parser.add_argument('--preload-limit',type=float,default=.05,help='maximum inward EE target offset per hand, metres (up to 0.05)')
    parser.add_argument('--palm-force',type=float,help='minimum palm normal force per hand in N; can only raise the mass/friction threshold')
    parser.add_argument('--tip-force',type=float,help='minimum distal normal force per hand in N before lifting; can only raise the existing threshold')
    parser.add_argument('--seat-angle',type=float,default=12.,help='maximum palm-pivot angular target preload in degrees, up to 20; no in-face rotation')
    parser.add_argument('--compliant-hands', action='store_true', help='30 degree MCP rest, HAND admittance, runtime current limit')
    parser.add_argument('--mcp-pwm400',action='store_true',help='simulation sensitivity: MCP PWM 400, retaining the 180 mA current limit and all hand guards')
    parser.add_argument('--dip-preload-deg',type=float,default=0.,help='simulation sensitivity: additional spring deflection at URDF DIP zero')
    parser.add_argument('--score', type=Path)
    args = parser.parse_args()
    if args.score:
        score(args.score.resolve())
    else:
        gains=PROFILES[args.profile]
        if args.arm400_kd is not None:
            if not np.isfinite(args.arm400_kd) or not 0 < args.arm400_kd <= 5:
                raise ValueError('Kp 400 experiment requires 0 < Kd <= 5')
            gains={group:(400.,args.arm400_kd) for group in ARM_GROUPS}
        if not np.isfinite(args.radius) or not .3 <= args.radius <= .4:
            raise ValueError('evaluation radius must be between .3 and .4 m')
        if not np.isfinite(args.height) or not .75 <= args.height <= 1.:
            raise ValueError('evaluation box center height must be between .75 and 1.0 m')
        evaluate.BOX_RADIUS = args.radius
        evaluate.BOX_Z = args.height
        evaluate.HAND_FACE_ROTATION_DEG = args.hand_face_rotation
        if not np.isfinite(args.seat_angle) or not 0 < args.seat_angle <= 20:
            raise ValueError('palm seating angle must be in (0, 20] degrees')
        evaluate.PALM_SEAT_ANGLE_DEG=args.seat_angle
        hand_control=None
        if args.compliant_hands:
            if args.match_hand_inertia:
                raise ValueError('Fixed-hand inertia matching cannot lump an articulated MCP')
            from compliant_hands import CompliantHands
            hand_control=CompliantHands(dip_preload_deg=args.dip_preload_deg,pwm400=args.mcp_pwm400)
        elif args.dip_preload_deg!=0:
            raise ValueError('DIP preload requires --compliant-hands')
        elif args.mcp_pwm400:
            raise ValueError('MCP PWM experiment requires --compliant-hands')
        original_scene, original_start = evaluate.scene, RobotMonitor.start
        outputs = []
        def scene(mass, out, friction):
            outputs.append(out)
            result = original_scene(mass, out, friction)
            if hand_control is not None:
                hand_control.prepare(out)
                path=out/'compliance.json';metadata=json.loads(path.read_text())
                metadata['hand_roll_deg']=[-args.hand_face_rotation,args.hand_face_rotation]
                path.write_text(json.dumps(metadata,indent=2))
            hand_inertia_audit(out, apply=args.match_hand_inertia)
            shutil.copy2(__file__,out/'tune_gains.py')
            shutil.copy2(evaluate.ROOT/'configs/controller/params.yaml',out/'vendor_params.yaml')
            (out/'gain_experiment.json').write_text(json.dumps(vars(args),indent=2))
            return result
        def start(self, *a, **kw):
            result = original_start(self, *a, **kw)
            self.set_recording(True)
            return result
        evaluate.scene, RobotMonitor.start = scene, start
        try:
            evaluate.trial('off', args.mass, friction=1.2, separate_waist=True,
                           sim_arm_gains=gains,hand_control=hand_control,preload_limit_m=args.preload_limit,palm_force_N=args.palm_force,tip_force_N=args.tip_force)
        finally:
            evaluate.ctl('sim','pause')
            if hand_control is not None:hand_control.close()
        score(outputs[-1])
