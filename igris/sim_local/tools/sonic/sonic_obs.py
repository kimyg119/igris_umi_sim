"""SONIC obs assembly (decoder 994 / encoder 1762) + action->joint-target.

Contract authority: docs/superpowers/specs/2026-08-26-s2-sonic-zeroshot-tracker-design.md
Reference deploy: GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/

The 6 traps (spec §7), and where each is handled:
  1. proprio joints are default-RELATIVE, motion-ref joints are ABSOLUTE.
     -> the CALLER pushes already-default-relative `joint_pos_rel` into StateLog;
        the encoder window carries absolute joint positions. Two different inputs,
        this module does no default subtraction on either. (StateLog.push / MotionRefWindow)
  2. proprio history = oldest->newest (stride 1); motion ref = current->future (step5).
     -> StateLog.stack() emits oldest->newest (deque left=oldest). The step5 /
        current->future sampling of the motion window is the caller's job; this
        module just row-major flattens the (10, N) window it is handed.
  3. encoder_mode_4 = [mode_id,0,0,0], g1 mode_id=0 => [0,0,0,0]  (assemble_encoder_obs)
  4. encoder 1762 is full-length; g1 fills ONLY offsets 0/4/294/601, rest 0. (assemble_encoder_obs)
  5. orientation obs = rotation matrix first two columns, row-wise:
        [R00,R01,R10,R11,R20,R21]  (anchor6)
  6. all math in float64; cast to float32 only at the ONNX-boundary return
     (assemble_decoder_obs / assemble_encoder_obs).
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

import sonic_const as C

# --- dimensions --------------------------------------------------------------
NUM_JOINTS = 29
HISTORY = 10
DECODER_DIM = 994
ENCODER_DIM = 1762

# decoder 994 block offsets (spec §2.1, confirmed against observation_config.yaml)
#   token_state(64) + angvel*10(30) + jpos*10(290) + jvel*10(290) + lastact*10(290) + gravity*10(30)
_DEC_TOKEN = 0
_DEC_ANGVEL = 64
_DEC_JPOS = 94
_DEC_JVEL = 384
_DEC_LASTACT = 674
_DEC_GRAVITY = 964

# encoder 1762 g1-mode block offsets (spec §2.2; config-order accumulation, trap 4)
_ENC_MODE4 = 0     # encoder_mode_4                          (4)
_ENC_MJPOS = 4     # motion_joint_positions_10frame_step5    (290)
_ENC_MJVEL = 294   # motion_joint_velocities_10frame_step5   (290)
_ENC_ANCHOR = 601  # motion_anchor_orientation_10frame_step5 (60)


# --- proprio history ring buffer (trap 2: oldest->newest) --------------------
class StateLog:
    """10-frame ring buffer of proprio state, oldest->newest.

    push() appends the newest frame (evicting the oldest past capacity).
    stack(field) returns that field for all frames flattened frame-major,
    oldest-first: [frame0_all, frame1_all, ...] as float64.

    During warm-up (fewer than 10 frames pushed) stack() left-pads by
    repeating the oldest available frame so the output always has 10 frames.
    """

    _FIELDS = ("joint_pos_rel", "joint_vel", "base_angvel", "gravity", "last_action")

    def __init__(self, history=HISTORY):
        self.history = history
        self._buf = deque(maxlen=history)  # left = oldest, right = newest

    def push(self, joint_pos_rel, joint_vel, base_angvel, gravity, last_action):
        self._buf.append({
            "joint_pos_rel": np.asarray(joint_pos_rel, dtype=np.float64),
            "joint_vel": np.asarray(joint_vel, dtype=np.float64),
            "base_angvel": np.asarray(base_angvel, dtype=np.float64),
            "gravity": np.asarray(gravity, dtype=np.float64),
            "last_action": np.asarray(last_action, dtype=np.float64),
        })

    def stack(self, field):
        if field not in self._FIELDS:
            raise KeyError(f"unknown field {field!r}; expected one of {self._FIELDS}")
        if not self._buf:
            raise RuntimeError("StateLog is empty; push at least one frame before stack()")
        frames = list(self._buf)
        # ponytail: warm-up pad by repeating oldest frame to reach `history`.
        # Upgrade path: mirror the reference's exact GetLatest padding if the
        # first ~10 ticks turn out to matter for the standing gate.
        if len(frames) < self.history:
            frames = [frames[0]] * (self.history - len(frames)) + frames
        return np.concatenate([f[field] for f in frames]).astype(np.float64)


# --- motion reference window (encoder input, g1 mode) ------------------------
@dataclass
class MotionRefWindow:
    """10-frame motion reference window for the g1 encoder, current->future step5.

    joint_positions  : (10, 29) ABSOLUTE joint pos, IsaacLab order (trap 1).
    joint_velocities : (10, 29) joint vel, IsaacLab order.
    anchor6          : (10, 6)  per-frame 6D orientation (see anchor6()).

    Frame 0 = now, frame 9 = +0.9 s (step5 = 5 ticks at 50 Hz). Building the
    step5 / delta-heading-corrected window is the caller's job (Task 3).
    """
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    anchor6: np.ndarray


# --- decoder / encoder assembly (float64 internal, float32 at return) --------
def assemble_decoder_obs(tokens64, statelog):
    """Assemble the 994-dim decoder obs (spec §2.1). Returns float32[994]."""
    tokens = np.asarray(tokens64, dtype=np.float64).reshape(-1)
    if tokens.shape[0] != 64:
        raise ValueError(f"tokens64 must be length 64, got {tokens.shape[0]}")

    obs = np.zeros(DECODER_DIM, dtype=np.float64)
    obs[_DEC_TOKEN:_DEC_TOKEN + 64] = tokens
    obs[_DEC_ANGVEL:_DEC_ANGVEL + 30] = statelog.stack("base_angvel")
    obs[_DEC_JPOS:_DEC_JPOS + 290] = statelog.stack("joint_pos_rel")
    obs[_DEC_JVEL:_DEC_JVEL + 290] = statelog.stack("joint_vel")
    obs[_DEC_LASTACT:_DEC_LASTACT + 290] = statelog.stack("last_action")
    obs[_DEC_GRAVITY:_DEC_GRAVITY + 30] = statelog.stack("gravity")
    return obs.astype(np.float32)


def assemble_encoder_obs(window):
    """Assemble the 1762-dim encoder obs in g1 mode (spec §2.2). Returns float32[1762].

    Fills ONLY the 4 g1 blocks at their full-layout offsets and leaves the rest
    zero (trap 4). encoder_mode_4 stays [0,0,0,0] since g1 mode_id=0 (trap 3).
    """
    jpos = np.asarray(window.joint_positions, dtype=np.float64)
    jvel = np.asarray(window.joint_velocities, dtype=np.float64)
    a6 = np.asarray(window.anchor6, dtype=np.float64)
    if jpos.shape != (HISTORY, NUM_JOINTS):
        raise ValueError(f"joint_positions must be (10,29), got {jpos.shape}")
    if jvel.shape != (HISTORY, NUM_JOINTS):
        raise ValueError(f"joint_velocities must be (10,29), got {jvel.shape}")
    if a6.shape != (HISTORY, 6):
        raise ValueError(f"anchor6 must be (10,6), got {a6.shape}")

    obs = np.zeros(ENCODER_DIM, dtype=np.float64)
    # obs[0:4] already zero == encoder_mode_4 = [mode_id=0, 0, 0, 0]
    obs[_ENC_MJPOS:_ENC_MJPOS + 290] = jpos.reshape(-1)      # frame-major, absolute
    obs[_ENC_MJVEL:_ENC_MJVEL + 290] = jvel.reshape(-1)
    obs[_ENC_ANCHOR:_ENC_ANCHOR + 60] = a6.reshape(-1)
    return obs.astype(np.float32)


# --- action -> joint target (spec §2.4, trap 1 + trap 7) ---------------------
def action_to_qtarget_g1(action29_isaaclab):
    """G1 action (29, IsaacLab order) -> joint position target (29, hardware order).

        q_target_hw[i] = default[i] + action_isaaclab[ISAACLAB_TO_MUJOCO[i]] * scale[i]

    (g1_deploy_onnx_ref.cpp:3123-3125). default/scale are hardware order.
    Returns float64 (this target feeds the codec/SDK, not ONNX).
    """
    a = np.asarray(action29_isaaclab, dtype=np.float64).reshape(-1)
    if a.shape[0] != NUM_JOINTS:
        raise ValueError(f"action must be length 29, got {a.shape[0]}")
    default = np.asarray(C.DEFAULT_ANGLES, dtype=np.float64)
    scale = np.asarray(C.ACTION_SCALE, dtype=np.float64)
    idx = np.asarray(C.ISAACLAB_TO_MUJOCO, dtype=np.intp)
    return default + a[idx] * scale


# --- quaternion helpers (wxyz, matching math_utils.hpp *_d) ------------------
def _quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_mul(a, b):
    """Hamilton product, wxyz (matches quat_mul_d)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def _quat_rotate(q, v):
    """Rotate vec v by unit quat q (wxyz). Rodrigues form, matches quat_rotate_d."""
    w = q[0]
    u = np.array([q[1], q[2], q[3]], dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return v * (2.0 * w * w - 1.0) + np.cross(u, v) * (2.0 * w) + u * (2.0 * np.dot(u, v))


def _quat_to_matrix(quat):
    """Unit-normalized wxyz quat -> 3x3 rotation matrix (matches quat_to_rotation_matrix_d)."""
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def heading_quat(q):
    """Yaw-only (heading) quaternion of a wxyz quat (matches calc_heading_quat_d).

    heading = atan2(rot_y, rot_x) of q applied to +x; returned as a z-axis quat.
    """
    rot = _quat_rotate(q, [1.0, 0.0, 0.0])
    ang = np.arctan2(rot[1], rot[0])
    return np.array([np.cos(ang / 2.0), 0.0, 0.0, np.sin(ang / 2.0)], dtype=np.float64)


def apply_delta_heading(init_base_quat, init_ref_root_quat):
    """Re-anchoring quat that aligns the reference heading to the robot heading
    at initialization time (matches ComputeApplyDeltaHeading, delta_heading==0):

        apply_delta_heading = heading(init_base) (x) conj(heading(init_ref_root))

    The caller pre-multiplies each future reference root quat by this before
    calling anchor6():  ref_adj = apply_delta_heading (x) raw_ref_root_quat.
    """
    return _quat_mul(heading_quat(init_base_quat), _quat_conjugate(heading_quat(init_ref_root_quat)))


def anchor6(robot_base_quat, ref_root_quat):
    """6D orientation obs for one frame (spec §2.2, trap 5).

        R = quat_to_matrix( conj(robot_base_quat) (x) ref_root_quat )
        return [R00, R01, R10, R11, R20, R21]   # first two columns, row-wise

    Both quats are wxyz. `ref_root_quat` is expected to already carry the
    delta-heading re-anchoring (see apply_delta_heading); this function is the
    clean relative-orientation primitive, so anchor6(q, q) == [1,0,0,1,0,0].
    """
    base = np.asarray(robot_base_quat, dtype=np.float64)
    ref = np.asarray(ref_root_quat, dtype=np.float64)
    rel = _quat_mul(_quat_conjugate(base), ref)
    R = _quat_to_matrix(rel)
    return np.array([R[0, 0], R[0, 1], R[1, 0], R[1, 1], R[2, 0], R[2, 1]], dtype=np.float64)


if __name__ == "__main__":
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(anchor6(q, q), [1, 0, 0, 1, 0, 0])
    # aligned headings -> delta-heading is identity on the heading component
    assert np.allclose(apply_delta_heading(q, q), [1, 0, 0, 0], atol=1e-12)
    print("sonic_obs self-check OK")
