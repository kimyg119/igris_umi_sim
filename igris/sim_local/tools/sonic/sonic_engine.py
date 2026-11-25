"""SONIC engine + tracker step (S2 Task 3).

Wires the frozen SONIC encoder/decoder ONNX to the Task-2 obs assembly and the
codec: one .step() maps a G1 motion reference window + IGRIS robot proprio to an
IGRIS joint-position target. Runs offline; no sim needed.

  IGRIS proprio  --Phi_enc-->  G1 proprio  --StateLog-->  decoder obs (994)
  G1 motion ref  --(delta-heading anchor)-->  encoder obs (1762) --enc--> tokens(64)
  decoder(tokens, statelog)  ->  action(29, IsaacLab)
  action  --action_to_qtarget_g1-->  q_g1_hw(29, hardware==G1_MJ29)
  q_g1_hw  --Phi_dec(codec)-->  {igris_joint: target}

Contract authority: docs/superpowers/specs/2026-08-26-s2-sonic-zeroshot-tracker-design.md
Task brief:         .superpowers/sdd/2026-08-26-s2-sonic-zeroshot-tracker/task-3-brief.md

CRITICAL (Task-2 review): the encoder anchor block needs the reference root quat
pre-multiplied by the init-frame delta-heading BEFORE anchor6(). That pre-apply
lives HERE, in SonicStep (the layer that owns episode-start capture) via
build_encoder_window(); Task-2's anchor6() stays the pure 2-arg primitive.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

# --- make the Task-2 sonic modules and the codec importable by bare name -----
_HERE = Path(__file__).resolve().parent               # .../sim_local/tools/sonic
_REPO = _HERE.parents[2]                               # .../igris
_CODEC_DIR = _REPO / "retargeting" / "tools"
for _p in (str(_HERE), str(_CODEC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sonic_const as C  # noqa: E402
from sonic_obs import (  # noqa: E402
    StateLog,
    MotionRefWindow,
    assemble_decoder_obs,
    assemble_encoder_obs,
    anchor6,
    apply_delta_heading,
    action_to_qtarget_g1,
    _quat_mul,
)
from igris_codec import PhiDecoder, closed_form_invert  # noqa: E402,F401  (re-export for callers/tests)

# --- default artifact paths --------------------------------------------------
_RELEASE = _REPO / "GR00T-WholeBodyControl" / "gear_sonic_deploy" / "policy" / "release"
DEFAULT_ENC_ONNX = _RELEASE / "model_encoder.onnx"
DEFAULT_DEC_ONNX = _RELEASE / "model_decoder.onnx"
# P4 sim2sim gate: SONIC_DEC_ONNX 로 학습 ckpt에서 내보낸 디코더를 꽂는다 (인코더는 release 유지).
import os as _os_dec
if _os_dec.environ.get("SONIC_DEC_ONNX"):
    DEFAULT_DEC_ONNX = Path(_os_dec.environ["SONIC_DEC_ONNX"])
    print(f"[sonic_engine] decoder override: {DEFAULT_DEC_ONNX}")
DEFAULT_PHI_ENC = _REPO / "retargeting" / "phi" / "phi_igris_enc.json"
DEFAULT_PHI_DEC = _REPO / "retargeting" / "phi" / "phi_igris_29dof.json"

NUM_JOINTS = 29
HISTORY = 10
ENCODER_DIM = 1762
DECODER_DIM = 994

# IsaacLab-order default pose: default_il[i] = DEFAULT_ANGLES[MUJOCO_TO_ISAACLAB[i]]
# (mirrors g1_deploy_onnx_ref proprio: body_q_il[i] = q_hw[MUJOCO_TO_ISAACLAB[i]] - default[...]).
DEFAULT_ANGLES_ISAACLAB = np.array(
    [C.DEFAULT_ANGLES[C.MUJOCO_TO_ISAACLAB[i]] for i in range(NUM_JOINTS)], dtype=np.float64
)


# --- ONNX engine -------------------------------------------------------------
class SonicEngine:
    """Frozen SONIC encoder + decoder ONNX sessions, single-threaded.

    Threads pinned to 1 (project trap 4: a thread storm starves the sim).
    Providers try CUDA then fall back to CPU; the active one is printed.
    """

    def __init__(self, enc_onnx_path=DEFAULT_ENC_ONNX, dec_onnx_path=DEFAULT_DEC_ONNX, providers=None):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        if providers is None:
            avail = ort.get_available_providers()
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in avail
                else ["CPUExecutionProvider"]
            )
        self.enc = ort.InferenceSession(str(enc_onnx_path), sess_options=so, providers=providers)
        self.dec = ort.InferenceSession(str(dec_onnx_path), sess_options=so, providers=providers)
        self.providers = self.enc.get_providers()
        self._enc_in = self.enc.get_inputs()[0].name
        self._dec_in = self.dec.get_inputs()[0].name
        print(f"[SonicEngine] provider={self.providers[0]} threads=1")

    def encode(self, enc_obs_1762):
        x = np.asarray(enc_obs_1762, dtype=np.float32).reshape(1, ENCODER_DIM)
        return self.enc.run(None, {self._enc_in: x})[0].reshape(-1)  # (64,)

    def decode(self, dec_obs_994):
        x = np.asarray(dec_obs_994, dtype=np.float32).reshape(1, DECODER_DIM)
        return self.dec.run(None, {self._dec_in: x})[0].reshape(-1)  # (29,)


# --- Phi_enc: IGRIS proprio -> G1 proprio ------------------------------------
@dataclass
class RobotProprio:
    """IGRIS-side proprioception for one tick.

    joint_pos, joint_vel : {igris_joint_name: float}, the 29 IGRIS joints.
    base_angvel, gravity : (3,) in the robot base/sensor frame.
    base_quat            : (4,) wxyz robot base orientation (used by the anchor).
    """

    joint_pos: dict
    joint_vel: dict
    base_angvel: np.ndarray
    gravity: np.ndarray
    base_quat: np.ndarray


@dataclass
class G1Proprio:
    jpos_isaaclab: np.ndarray   # (29,) ABSOLUTE G1 joint pos, IsaacLab order
    jvel_isaaclab: np.ndarray   # (29,) G1 joint vel, IsaacLab order
    base_angvel: np.ndarray     # (3,) aligned
    gravity: np.ndarray         # (3,) aligned


class PhiEnc:
    """Phi_enc: IGRIS proprio -> G1 proprio (json 'igris-phi-enc/2').

    Position: waist(3)+shoulder(6) joints marked mode=="closed_form" use the codec's
    exact closed-form inverse (closed_form_invert -> the geometric inverse of Phi_dec),
    so Phi_dec(Phi_enc(q)) is self-consistent on those balance-critical joints instead
    of the low-R2 reverse affine. All other joints (legs/elbows/wrists/ankle-roll) use
    the per-joint reverse affine  q_g1 = a*q_igris + b  (ankle-roll's low R2 is amplitude,
    not axis convention -- closed-form cannot help, so it stays affine).

    Velocity uses the reverse-affine slope for ALL joints (v_g1 = a*v_igris; the constant
    b drops under d/dt). The closed-form map is nonlinear so this is a local linearization
    of the velocity, kept simple because the decoder's velocity obs is not balance-critical.

    Joints are mapped BY NAME into IsaacLab order. base_angvel / gravity are left-multiplied
    by `align_rot` (3x3, default identity) -- the IGRIS-base -> G1-base sensor-frame alignment.
    It is identity until the two base frames are measured to differ; leave it as the
    calibration knob, not code.
    """

    def __init__(self, phi_enc_path=DEFAULT_PHI_ENC, align_rot=None):
        d = json.loads(Path(phi_enc_path).read_text())
        self.entries = [(j["g1"], j["igris"], float(j["a"]), float(j["b"])) for j in d["joints"]]
        self.closed_form = {j["g1"] for j in d["joints"] if j.get("mode") == "closed_form"}
        self.align_rot = np.eye(3) if align_rot is None else np.asarray(align_rot, dtype=np.float64)
        self._il_index = {name: i for i, name in enumerate(C.G1_ISAACLAB_NAMES)}

    def __call__(self, proprio):
        jpos = np.zeros(NUM_JOINTS, dtype=np.float64)
        jvel = np.zeros(NUM_JOINTS, dtype=np.float64)
        cf = closed_form_invert(proprio.joint_pos) if self.closed_form else {}
        for g1, ig, a, b in self.entries:
            i = self._il_index[g1]
            if g1 in self.closed_form:
                jpos[i] = float(np.asarray(cf[g1]).reshape(-1)[0])
            else:
                jpos[i] = a * float(proprio.joint_pos[ig]) + b
            jvel[i] = a * float(proprio.joint_vel[ig])
        return G1Proprio(
            jpos_isaaclab=jpos,
            jvel_isaaclab=jvel,
            base_angvel=self.align_rot @ np.asarray(proprio.base_angvel, dtype=np.float64),
            gravity=self.align_rot @ np.asarray(proprio.gravity, dtype=np.float64),
        )


# --- raw motion reference window + delta-heading-aware builder ----------------
@dataclass
class MotionRef:
    """Raw 10-frame G1 motion reference window (current->future, step5).

    joint_positions  : (10,29) ABSOLUTE G1 joint pos, IsaacLab order (trap 1).
    joint_velocities : (10,29) G1 joint vel, IsaacLab order.
    root_quat        : (10,4)  raw reference root quat wxyz per frame (pre delta-heading).
    """

    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    root_quat: np.ndarray


def build_encoder_window(motion_ref, base_quat, delta_heading_quat):
    """Raw MotionRef -> Task-2 MotionRefWindow with the delta-heading-adjusted anchor.

    Per frame the anchor input is (CRITICAL requirement):
        anchor6(base_now, delta_heading (x) raw_ref_root_quat[f])
    `delta_heading_quat` is computed once per episode via apply_delta_heading();
    pass identity [1,0,0,0] to reproduce the (wrong) no-pre-apply case.
    """
    rq = np.asarray(motion_ref.root_quat, dtype=np.float64)
    a6 = np.stack(
        [anchor6(base_quat, _quat_mul(delta_heading_quat, rq[f])) for f in range(HISTORY)]
    )
    return MotionRefWindow(
        joint_positions=np.asarray(motion_ref.joint_positions, dtype=np.float64),
        joint_velocities=np.asarray(motion_ref.joint_velocities, dtype=np.float64),
        anchor6=a6,
    )


# --- full tracker step -------------------------------------------------------
class SonicStep:
    """One G1-tracking step: motion ref + IGRIS proprio -> IGRIS joint target dict.

    Owns episode-start capture of init base/ref quats and the delta-heading, so the
    anchor pre-apply is done here (not in the caller). last_action feeds the decoder
    obs and is the previous IsaacLab-order action (zeros before the first decode).
    """

    def __init__(self, engine, phi_dec, phi_enc, statelog=None):
        self.engine = engine
        self.phi_dec = phi_dec
        self.phi_enc = phi_enc
        self.statelog = statelog if statelog is not None else StateLog()
        self.last_action = np.zeros(NUM_JOINTS, dtype=np.float64)
        self._delta_heading = None  # set on first step (episode start)

    def reset(self):
        self.statelog = StateLog()
        self.last_action = np.zeros(NUM_JOINTS, dtype=np.float64)
        self._delta_heading = None

    def step(self, motion_ref, robot_proprio):
        base_now = np.asarray(robot_proprio.base_quat, dtype=np.float64)

        # 0. episode-start capture: delta-heading from init base & init ref-root (frame 0).
        if self._delta_heading is None:
            init_ref_root = np.asarray(motion_ref.root_quat, dtype=np.float64)[0]
            self._delta_heading = apply_delta_heading(base_now, init_ref_root)

        # 1. Phi_enc: IGRIS proprio -> G1 proprio (IsaacLab order).
        g1 = self.phi_enc(robot_proprio)

        # 2. default-relative jpos; push the newest proprio frame (StateLog keeps oldest->newest
        #    and left-pads the first ticks by repeating the oldest frame).
        jpos_rel = g1.jpos_isaaclab - DEFAULT_ANGLES_ISAACLAB
        self.statelog.push(jpos_rel, g1.jvel_isaaclab, g1.base_angvel, g1.gravity, self.last_action)

        # 3. encode the delta-heading-adjusted motion window.
        window = build_encoder_window(motion_ref, base_now, self._delta_heading)
        tokens = self.engine.encode(assemble_encoder_obs(window))

        # 4. decode -> action (29, IsaacLab order).
        action = self.engine.decode(assemble_decoder_obs(tokens, self.statelog))

        # 5. action -> G1 hardware target -> IGRIS joints via the codec.
        #    action_to_qtarget_g1 returns hardware order == G1_MJ29, which phi_dec.joints wants.
        q_g1_hw = action_to_qtarget_g1(action)
        q_igris = self.phi_dec.joints(q_g1_hw)
        self.last_action = np.asarray(action, dtype=np.float64)
        return {k: float(np.asarray(v).reshape(-1)[0]) for k, v in q_igris.items()}


if __name__ == "__main__":
    # smoke self-check: one static step yields 29 finite IGRIS targets.
    eng = SonicEngine()
    penc = PhiEnc()
    names = [ig for (_g1, ig, _a, _b) in penc.entries]
    pro = RobotProprio(
        joint_pos={n: 0.0 for n in names}, joint_vel={n: 0.0 for n in names},
        base_angvel=np.zeros(3), gravity=np.array([0.0, 0.0, -1.0]), base_quat=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    ref = MotionRef(
        joint_positions=np.tile(DEFAULT_ANGLES_ISAACLAB, (10, 1)),
        joint_velocities=np.zeros((10, 29)),
        root_quat=np.tile([1.0, 0.0, 0.0, 0.0], (10, 1)),
    )
    step = SonicStep(eng, PhiDecoder(str(DEFAULT_PHI_DEC)), penc)
    out = step.step(ref, pro)
    vals = np.array(list(out.values()))
    assert len(out) == 29 and np.all(np.isfinite(vals)), out
    print("sonic_engine self-check OK:", {k: round(v, 3) for k, v in list(out.items())[:4]})


# --- Matrix codec (element #3 of the 4-part deploy port, 2026-08-31) ----------------------------
# Same 29x29 linear codec the Isaac zero-shot (55% walk completion, ref_v25) runs on:
#   dec: q_ig = A @ q_g1 + b      enc: q_g1 = A_inv @ (q_ig - b)      vel: linear part only.
# Replaces the S2-era per-joint affine (PhiEnc json, low-R2 on ankle-roll/waist) and the closed-form
# inverse. Name bridge: npz `igris_mjcf` (Joint_*) is index-aligned with `igris_joints` (lowercase);
# g1_joints share the *_joint convention with C.G1_HW_NAMES / C.G1_ISAACLAB_NAMES.
NEWCODEC_NPZ = _REPO / "GR00T-WholeBodyControl" / "gear_sonic" / "data" / "igris_codec" / "newcodec_linear.npz"


class _MatrixCodec:
    def __init__(self, npz_path=NEWCODEC_NPZ):
        z = np.load(npz_path, allow_pickle=True)
        self.A = np.asarray(z["A"], dtype=np.float64)
        self.b = np.asarray(z["b"], dtype=np.float64)
        self.Ainv = np.linalg.inv(self.A)
        self.g1_names = [str(x) for x in z["g1_joints"]]
        self.ig_names = [str(x) for x in z["igris_mjcf"]]  # Joint_* style (proprio/build_q31 keys)


class PhiEncMatrix:
    """Drop-in for PhiEnc: IGRIS proprio -> G1Proprio via the linear matrix codec."""

    def __init__(self, npz_path=NEWCODEC_NPZ, align_rot=None):
        self.m = _MatrixCodec(npz_path)
        self.align_rot = np.eye(3) if align_rot is None else np.asarray(align_rot, dtype=np.float64)
        il = {name: i for i, name in enumerate(C.G1_ISAACLAB_NAMES)}
        self._to_il = np.array([il[n] for n in self.m.g1_names], dtype=int)

    def __call__(self, proprio):
        q = np.array([float(proprio.joint_pos[n]) for n in self.m.ig_names], dtype=np.float64)
        v = np.array([float(proprio.joint_vel[n]) for n in self.m.ig_names], dtype=np.float64)
        qg = self.m.Ainv @ (q - self.m.b)
        vg = self.m.Ainv @ v
        jpos = np.zeros(NUM_JOINTS, dtype=np.float64)
        jvel = np.zeros(NUM_JOINTS, dtype=np.float64)
        jpos[self._to_il] = qg
        jvel[self._to_il] = vg
        return G1Proprio(
            jpos_isaaclab=jpos,
            jvel_isaaclab=jvel,
            base_angvel=self.align_rot @ np.asarray(proprio.base_angvel, dtype=np.float64),
            gravity=self.align_rot @ np.asarray(proprio.gravity, dtype=np.float64),
        )


class PhiDecoderMatrix:
    """Drop-in for PhiDecoder.joints: G1 target (29, hardware order) -> {Joint_*: igris target}."""

    def __init__(self, npz_path=NEWCODEC_NPZ):
        self.m = _MatrixCodec(npz_path)
        hw = {name: i for i, name in enumerate(C.G1_HW_NAMES)}
        self._from_hw = np.array([hw[n] for n in self.m.g1_names], dtype=int)

    def joints(self, q_g1_hw):
        qg = np.asarray(q_g1_hw, dtype=np.float64).reshape(-1)[self._from_hw]
        qi = self.m.A @ qg + self.m.b
        return {n: float(qi[i]) for i, n in enumerate(self.m.ig_names)}
