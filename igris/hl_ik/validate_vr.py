"""Synthetic VR -> HL-IK -> guarded controller -> fixed-pelvis MuJoCo smoke test.

No sockets, DDS, policy, or hardware. Ideal gravity compensation and fixed pelvis
isolate arm/waist tracking; this is explicitly NOT a WALK_LOW balance evaluation.
"""
import argparse
from pathlib import Path
import time

import mujoco
import numpy as np

from igris_ik.geometry import Pose, UpperBodyTargets
from igris_ik.realtime.retarget import DeltaCalibration, hmd_waist_yaw
from igris_ik.realtime.robot_control import ArmController, ControlConfig
from igris_ik.realtime.robot_io import RobotState
from igris_ik.robot_geometry import RobotGeometry
from igris_ik.sim.mujoco_model import load_fixed_hand_model
from igris_ik.vendor_limits import coupled_halfspaces
from igris_ik.constants import URDF_JOINT_NAMES
from .common import Rotation, save_json
from .vr_adapter import HLIKTeleopIK


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    ik = HLIKTeleopIK()
    q0 = ik.default_configuration().q_sdk.copy()
    q0[:3] = 0.; q0[[15, 22]] = -.5; q0[[16, 23]] = [.4, -.4]; q0[[18, 25]] = -1.7
    ik.reset(q0)
    origin = ik.forward(q0)
    assert ik.check_configuration(q0, origin).success
    mapping = DeltaCalibration(body_relative=True, direct_hmd_yaw=True)
    mapping.calibrate(origin, origin)
    bundle = load_fixed_hand_model(); model = bundle.model; data = mujoco.MjData(model)
    qadr = np.array(bundle.qpos_addresses); dadr = np.array(bundle.dof_addresses)
    actuators = np.array(bundle.actuator_ids)
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    passive_damping = model.dof_damping[dadr].copy()
    data.qpos[qadr] = q0; mujoco.mj_forward(model, data)
    guard = RobotGeometry().command_guard
    controller = ArmController(ControlConfig(command_waist=True, hold_untracked=False,
        slew_rad_s=.5, accel_rad_s2=2., required_robot_mode='ROBOT_STATE_WALK_LOW'),
        *ik.model.full_joint_limits(), joint_halfspaces=coupled_halfspaces(list(URDF_JOINT_NAMES)),
        command_guard=guard)

    def state():
        return RobotState(0, data.time, data.time, 'fixed_pelvis_fixture',
                          data.qpos[qadr], data.qvel[dadr], np.zeros(31), 'ROBOT_STATE_WALK_LOW')

    controller.arm(state()); controller.follow()
    selected = q0.copy(); target = origin; previous_target = origin
    yaw = 0.; command = None; records = []; solve_ms = []; rejected = 0
    started = time.perf_counter()
    steps = 8*2100
    for step in range(steps):
        t = step/2100.
        if step % 70 == 0:  # 30 Hz source / IK, simulated clock, not wall time.
            measured = state()
            ik.observe(measured.q, t)
            angle = np.deg2rad(20)*.5*(1-np.cos(np.pi*min(t/2., 1.)))
            phase = np.clip((t-2.)/4., 0., 1.)
            travel = .025*np.sin(np.pi*phase)**2
            rotation = Rotation.from_euler('z', angle)
            def operator(pose, hand=False):
                p = pose.position + ([travel, 0., .01*np.sin(2*np.pi*phase)] if hand else np.zeros(3))
                return Pose(origin.head.position + rotation.apply(p-origin.head.position),
                    (rotation*Rotation.from_quat(pose.wxyz, scalar_first=True)).as_quat(scalar_first=True))
            sample = UpperBodyTargets(operator(origin.left, True), operator(origin.right, True),
                                      head=operator(origin.head))
            target = mapping.map(sample)
            yaw = hmd_waist_yaw(sample.head, mapping.head_yaw_rad, 0., np.deg2rad([-88, 88]))
            moving = any(np.linalg.norm(getattr(target, s).position-getattr(previous_target, s).position)>1e-9
                         for s in ('left', 'right'))
            if moving:
                saved = ik.warm_start()
                result = ik.solve(measured.q, target)
                solve_ms.append(result.diagnostics.solve_time_s*1000)
                if result.success:
                    previous = selected.copy(); previous[:15] = measured.q[:15]
                    result = ik.check_motion(previous, result.configuration, target)
                if result.success:
                    selected = result.q_sdk.copy(); previous_target = target
                else:
                    rejected += 1; ik.restore_warm_start(saved)
        if step % 42 == 0:  # Same 50 Hz command controller as the live VR monitor.
            command = controller.tick(data.time, state(), selected[15:29], ik_q_waist=np.array([yaw, 0., 0.]))
            assert controller.last_trip is None, (controller.last_trip, t,
                [(URDF_JOINT_NAMES[i], float(data.qpos[qadr][i]), float(controller.q_cmd[i]))
                 for i in np.flatnonzero(abs(data.qpos[qadr]-controller.q_cmd)>.2)])
            assert command is not None and not command.kp[3:15].any() and not command.kp[29:].any()
        # Independent fixture support for legs/neck. Teleop provides no leg/neck gain.
        support_kp = np.full(31, 300.); support_kd = np.full(31, 20.)
        kp = np.where(command.kp > 0., command.kp, support_kp)
        kd = np.where(command.kp > 0., command.kd, support_kd)
        reference = np.where(command.kp > 0., command.q, q0)
        # Ideal model gravity compensation is a fixture assumption, not a tuned real gain.
        # Express -Kd*dq as implicit joint damping. An explicit external torque
        # destabilizes the tiny wrist inertias even at this 2100 Hz fixture step.
        model.dof_damping[dadr] = passive_damping + kd
        torque = kp*(reference-data.qpos[qadr])+data.qfrc_bias[dadr]
        data.ctrl[actuators] = torque
        mujoco.mj_step(model, data)
        if step % 70 == 0:
            actual = ik.forward(data.qpos[qadr])
            error = [np.linalg.norm(getattr(actual, s).position-getattr(target, s).position) for s in ('left', 'right')]
            records.append((data.time, data.qpos[qadr].copy(), command.q.copy(), error))
    arrays = dict(time=np.array([r[0] for r in records]), q=np.array([r[1] for r in records]),
                  command=np.array([r[2] for r in records]), measured_ee_error_m=np.array([r[3] for r in records]))
    np.savez(out/'rollout.npz', **arrays)
    summary = dict(kind='fixed_pelvis_mujoco_dynamics', synthetic_vr=True, physics_steps=steps,
        simulation_seconds=float(data.time), lower_body_policy=False, ideal_gravity_compensation=True,
        hardware_commands_sent=0, execution_ready=False, checkpoint_sha256=ik.solver.prior.sha256,
        ik_solves=len(solve_ms), ik_rejected=rejected, command_trips=controller.last_trip,
        solve_ms_p50_p95=np.percentile(solve_ms, [50, 95]).tolist(),
        measured_ee_error_mm_p50_p95_max=np.percentile(arrays['measured_ee_error_m']*1000, [50, 95, 100]).tolist(),
        waist_requested_rad=yaw, waist_measured_rad=float(arrays['q'][-1, 0]),
        wall_seconds=time.perf_counter()-started)
    save_json(out/'summary.json', summary)
    assert rejected == 0, summary
    assert arrays['measured_ee_error_m'].max() < .02, summary
    assert abs(summary['waist_measured_rad']-yaw) < .02, summary
    render(out)
    print(summary)


def render(out):
    # Render actual simulated states, never the desired q, for inspection.
    from PIL import Image, ImageDraw, ImageFont
    out = Path(out)
    with np.load(out/'rollout.npz') as saved:
        snapshots = saved['q'][[0, 60, 120, 210]]
    bundle = load_fixed_hand_model(); model = bundle.model; data = mujoco.MjData(model)
    model.vis.global_.offwidth = 640; model.vis.global_.offheight = 640
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = .7; model.vis.headlight.diffuse[:] = .8
    model.geom_rgba[:, :3] = np.maximum(model.geom_rgba[:, :3], .45)
    model.mat_rgba[:, :3] = np.maximum(model.mat_rgba[:, :3], .45)
    model.geom_rgba[:, 3] = 1.; model.mat_rgba[:, 3] = 1.
    model.geom_group[model.geom('floor').id] = 1
    camera = mujoco.MjvCamera(); camera.lookat[:] = [0, 0, 1.1]
    camera.distance = 2.; camera.azimuth = 145.; camera.elevation = -12.
    option = mujoco.MjvOption(); option.geomgroup[0] = 0; option.geomgroup[1] = 1
    renderer = mujoco.Renderer(model, height=640, width=640)
    sheet = Image.new('RGB', (1280, 1280), 'white')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    for i, q in enumerate(snapshots):
        data.qpos[np.array(bundle.qpos_addresses)] = q; mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera, option)
        tile = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, 640, 42), fill='white')
        draw.text((12, 10), f'{[0, 2, 4, 7][i]} s | fixed pelvis | synthetic VR', fill='black', font=font)
        sheet.paste(tile, ((i%2)*640, (i//2)*640))
    renderer.close(); sheet.save(out/'preview.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--render-only', action='store_true', help='render saved measured q without repeating physics')
    args = parser.parse_args()
    (render if args.render_only else run)(args.out)
