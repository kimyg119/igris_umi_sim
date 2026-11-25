"""G0 gate — G1-self MuJoCo reproduction of the SONIC obs contract.

Ruling R-pre2 (G1-native): bypass Phi_enc/Phi_dec entirely. G1's own MuJoCo
state IS the proprioception (no reverse affine), and G1's actuators are driven
directly by the SONIC action target (no codec). What G0 validates is the REAL
obs assembly (Task 2, sonic_obs) + the REAL SONIC encoder/decoder (Task 3,
sonic_engine). If G1 holds standing here, the obs contract is correct and every
downstream failure is isolated to codec/dynamics.

Loop (50 Hz control, model timestep 0.002 s -> 10 physics substeps/tick):
  1. read G1 MuJoCo state (hardware order q/dq, base quat wxyz, base gyro, gravity)
  2. build proprio DIRECTLY (hw->IsaacLab via MUJOCO_TO_ISAACLAB, default-relative)
  3. encode the STANDING reference window (build_encoder_window -> delta-heading anchor)
  4. decode -> action(29, IsaacLab)
  5. action_to_qtarget_g1 -> q_g1_hw ; PD (deploy kp/kd) -> actuator torque

Run:  conda run -n gmr python sim_local/tools/sonic/g1_selfcheck.py
Contract authority: docs/superpowers/specs/2026-08-26-s2-sonic-zeroshot-tracker-design.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import mujoco

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]                       # .../igris
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import sonic_const as C
from sonic_obs import (
    StateLog,
    assemble_decoder_obs,
    assemble_encoder_obs,
    action_to_qtarget_g1,
    apply_delta_heading,
    _quat_conjugate,
    _quat_rotate,
)
from sonic_engine import (
    SonicEngine,
    MotionRef,
    build_encoder_window,
    DEFAULT_ANGLES_ISAACLAB,
)

MODEL_XML = _REPO / "GR00T-WholeBodyControl" / "gear_sonic" / "data" / "robots" / "g1" / "scene_29dof_mujoco.xml"

NUM_JOINTS = 29
HISTORY = 10
CONTROL_HZ = 50.0
INIT_Z = 0.757          # feet-just-touching pelvis height for the default pose (measured)
DEFAULT_ANGLES = np.asarray(C.DEFAULT_ANGLES, dtype=np.float64)
MJ2IL = np.asarray(C.MUJOCO_TO_ISAACLAB, dtype=np.intp)


# --- deploy PD gains (policy_parameters.hpp:143-207, copied faithfully) -------
# stiffness = armature*w^2 ; damping = 2*zeta*armature*w (zeta=2). The 2.0x on
# ankles + waist roll/pitch is in the hpp arrays and is NOT implied by the motor
# type, so the arrays are laid out explicitly rather than derived from _MOTOR_TYPE_HW.
def _stiff(mt):
    return C._ARMATURE[mt] * C._OMEGA * C._OMEGA


def _damp(mt):
    return 2.0 * 2.0 * C._ARMATURE[mt] * C._OMEGA


def _gain_arrays():
    s, d = _stiff, _damp
    kp = np.array([
        s("7520_22"), s("7520_22"), s("7520_14"), s("7520_22"), 2 * s("5020"), 2 * s("5020"),
        s("7520_22"), s("7520_22"), s("7520_14"), s("7520_22"), 2 * s("5020"), 2 * s("5020"),
        s("7520_14"), 2 * s("5020"), 2 * s("5020"),
        s("5020"), s("5020"), s("5020"), s("5020"), s("5020"), s("4010"), s("4010"),
        s("5020"), s("5020"), s("5020"), s("5020"), s("5020"), s("4010"), s("4010"),
    ], dtype=np.float64)
    kd = np.array([
        d("7520_22"), d("7520_22"), d("7520_14"), d("7520_22"), 2 * d("5020"), 2 * d("5020"),
        d("7520_22"), d("7520_22"), d("7520_14"), d("7520_22"), 2 * d("5020"), 2 * d("5020"),
        d("7520_14"), 2 * d("5020"), 2 * d("5020"),
        d("5020"), d("5020"), d("5020"), d("5020"), d("5020"), d("4010"), d("4010"),
        d("5020"), d("5020"), d("5020"), d("5020"), d("5020"), d("4010"), d("4010"),
    ], dtype=np.float64)
    return kp, kd


def _standing_motion_ref(init_base_quat):
    """The G0 standing reference: 10 future frames all == the default standing pose."""
    return MotionRef(
        joint_positions=np.tile(DEFAULT_ANGLES_ISAACLAB, (HISTORY, 1)),
        joint_velocities=np.zeros((HISTORY, NUM_JOINTS)),
        root_quat=np.tile(np.asarray(init_base_quat, dtype=np.float64), (HISTORY, 1)),
    )


def run_gate(duration_s=10.0, engine=None, verbose=True):
    """Run the standing G0 gate. Returns a result dict (JSON-serializable)."""
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    n_sub = int(round((1.0 / CONTROL_HZ) / dt))          # 0.02/0.002 = 10
    n_ticks = int(round(duration_s * CONTROL_HZ))
    kp, kd = _gain_arrays()
    tlim = model.actuator_ctrlrange[:, 1].copy()          # effort limits, hardware order

    if engine is None:
        engine = SonicEngine()

    # start AT the standing pose (proprio ~= reference initially), qvel=0.
    data.qpos[:] = 0.0
    data.qpos[2] = INIT_Z
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:36] = DEFAULT_ANGLES
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    init_base_quat = data.qpos[3:7].copy()
    motion_ref = _standing_motion_ref(init_base_quat)
    delta_heading = apply_delta_heading(init_base_quat, motion_ref.root_quat[0])

    statelog = StateLog()
    last_action = np.zeros(NUM_JOINTS, dtype=np.float64)
    z0 = float(data.qpos[2])

    z_traj = []
    target_sq_err = np.zeros(NUM_JOINTS)   # accumulated (q_target_hw - default_hw)^2
    fell = False

    for tick in range(n_ticks):
        # 1. G1 MuJoCo state (hardware order).
        q_hw = data.qpos[7:36].copy()
        dq_hw = data.qvel[6:35].copy()
        base_quat = data.qpos[3:7].copy()                 # wxyz
        base_angvel = data.qvel[3:6].copy()               # free-joint angular vel is base-local
        gravity = _quat_rotate(_quat_conjugate(base_quat), [0.0, 0.0, -1.0])

        # 2. proprio DIRECTLY: hw->IsaacLab, default-relative jpos (no reverse affine).
        jpos_rel = q_hw[MJ2IL] - DEFAULT_ANGLES_ISAACLAB
        jvel_il = dq_hw[MJ2IL]
        statelog.push(jpos_rel, jvel_il, base_angvel, gravity, last_action)

        # 3. encode the delta-heading-adjusted standing window (reuse Task-3 builder).
        window = build_encoder_window(motion_ref, base_quat, delta_heading)
        tokens = engine.encode(assemble_encoder_obs(window))

        # 4. decode -> action (29, IsaacLab).
        action = engine.decode(assemble_decoder_obs(tokens, statelog))

        # 5. action -> hardware joint target; PD -> torque over the physics substeps.
        q_target_hw = action_to_qtarget_g1(action)
        target_sq_err += (q_target_hw - DEFAULT_ANGLES) ** 2
        for _ in range(n_sub):
            q = data.qpos[7:36]
            dq = data.qvel[6:35]
            tau = np.clip(kp * (q_target_hw - q) - kd * dq, -tlim, tlim)
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)

        last_action = np.asarray(action, dtype=np.float64)
        z = float(data.qpos[2])
        z_traj.append(z)
        if z < 0.5:
            fell = True
            if verbose:
                print(f"[G0] FELL at tick {tick} (t={tick/CONTROL_HZ:.2f}s) z={z:.3f}")
            break

    z_arr = np.asarray(z_traj)
    z_min = float(z_arr.min())
    z_max = float(z_arr.max())
    z_dev = float(np.abs(z_arr - z0).max())
    n_done = len(z_traj)
    rms_per_joint = np.sqrt(target_sq_err / max(n_done, 1))       # hardware order
    rms_overall = float(np.sqrt((target_sq_err.sum()) / max(n_done * NUM_JOINTS, 1)))

    # PASS: no fall (z never < 0.5) AND base z held within initial +/- 0.03 m.
    no_fall = (not fell) and (z_min > 0.5)
    z_held = z_dev <= 0.03
    passed = bool(no_fall and z_held and n_done == n_ticks)

    result = {
        "gate": "G0",
        "passed": passed,
        "no_fall": bool(no_fall),
        "z_held_within_0.03": bool(z_held),
        "duration_s": duration_s,
        "control_hz": CONTROL_HZ,
        "physics_hz": 1.0 / dt,
        "substeps_per_tick": n_sub,
        "ticks_run": n_done,
        "ticks_expected": n_ticks,
        "z0": z0,
        "z_min": z_min,
        "z_max": z_max,
        "z_max_dev_from_init": z_dev,
        "provider": engine.providers[0],
        "tracking_rms_overall_rad": rms_overall,
        "tracking_rms_per_joint_hw": {C.G1_HW_NAMES[i]: float(rms_per_joint[i]) for i in range(NUM_JOINTS)},
        "z_traj": [round(z, 5) for z in z_traj],
    }
    if verbose:
        verdict = "PASS" if passed else "FAIL"
        print(f"[G0] {verdict}  z0={z0:.4f}  z_min={z_min:.4f}  z_max={z_max:.4f}  "
              f"max_dev={z_dev:.4f}  ticks={n_done}/{n_ticks}  rms={rms_overall:.4f} rad")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=12.0, help="standing-check seconds")
    ap.add_argument("--json-out", type=str, default=str(_REPO / "logs" / "g0_g1_selfcheck.json"))
    args = ap.parse_args()

    result = run_gate(args.duration)
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[G0] wrote {out}")
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
