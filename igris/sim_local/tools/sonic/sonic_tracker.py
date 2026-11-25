#!/usr/bin/env python3
"""SONIC tracker node (S2 Task 6) — the 50 Hz in-container loop that replaces student_130.

Wires every prior task into one process:

  /igris_c_001/rt/joint_state (serial 31) ─┐
  /mujoco/imu (quat wxyz + gyro) ──────────┤─ build_proprio ─▶ RobotProprio(IGRIS)
  redis `sonic_g1_reference` (G1 qpos 36) ── build_motion_ref ─▶ MotionRef(G1, 10-frame)
                                                     │
        SonicStep.step(MotionRef, RobotProprio) ─▶ {igris_joint: target} (29, incl. wrists)
                                                     │
        build_q31: 23 policy → SDK 31 (kActionsToSystemJointMapping) + wrist/neck HOLD
                                                     │
        LowCmdPublisher.send(q31, 0, kp31, kd31, PJS) ─▶ igris_c_001/rt/lowcmd (50 Hz)

Runs as ONE process (no fork after DDS init — Task-5 trap 3). ONNX threads pinned to 1.

Contract authority: docs/superpowers/specs/2026-08-26-s2-sonic-zeroshot-tracker-design.md
Task brief:         .superpowers/sdd/2026-08-26-s2-sonic-zeroshot-tracker/task-6-brief.md

STATE-SOURCE NOTE (deviation from brief, deliberate): the brief named
/mujoco/joint_states, but that topic is ACTUATOR/flange space for the parallel
joints (l_waist_flange, l_ankle_flange_out/in, l_wrist_motor_output_front/back),
whereas Phi_enc and the serial SDK lowcmd slots both need SERIAL joint angles
(Joint_Waist_Roll, Joint_Ankle_Pitch, ...). /igris_c_001/rt/joint_state is the
same 31 joints in SDK serial order (name=[], positional == SDK31/JointIndex),
verified to match /mujoco/joint_states on the serial joints (shoulders) and to
differ only on the parallel joints. We subscribe the serial topic via rclpy.

ROOT-FRAME ALIGNMENT: build_proprio replicates student_130's state_handler
(IMU rotated -90 deg about Y into base_link before projected gravity / angvel).
IMU_MOUNT_ROT is that fixed hardware correction. ALIGN_ROT (default identity,
passed into Phi_enc) is THE single gate-calibration knob for any residual
IGRIS-base -> G1-pelvis rotation (spec section 6 open item; tune in Task 8).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as _R

# --- make the sibling sonic modules + codec importable by bare name ----------
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_CODEC_DIR = _REPO / "retargeting" / "tools"
for _p in (str(_HERE), str(_CODEC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sonic_engine import (  # noqa: E402
    SonicEngine, SonicStep, PhiEnc, PhiDecoder, RobotProprio, MotionRef,
    DEFAULT_ENC_ONNX, DEFAULT_DEC_ONNX, DEFAULT_PHI_ENC, DEFAULT_PHI_DEC,
    NUM_JOINTS, HISTORY,
)
import sonic_const as C  # noqa: E402
from sonic_obs import _quat_to_matrix  # noqa: E402  (wxyz -> 3x3, deploy-matched)
from spike_hold import K_ACTIONS_TO_SYSTEM, HOLD_SLOTS, load_gains, YAML  # noqa: E402
from lowcmd_pub import NUM_MOTORS  # noqa: E402
import motion_common as mc  # noqa: E402  (POLICY_JOINT_NAMES, policy order)
from igris_codec import SDK31  # noqa: E402  (31 IGRIS joint names, SDK serial order)

STATE_TOPIC = "/igris_c_001/rt/joint_state"   # sensor_msgs/JointState, serial, SDK31 order
IMU_TOPIC = "/mujoco/imu"                     # sensor_msgs/Imu (orientation xyzw, angvel)
REF_KEY = "sonic_g1_reference"                # redis key (Task 7)
REF_STALE_MS = 500.0                          # older than this -> fall back

# IGRIS IMU -> base_link fixed correction (student_130 state_handler: Rotate Y by -90 deg).
def _rot_y(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


IMU_MOUNT_ROT = _rot_y(-np.pi / 2.0)          # fixed hardware constant (not the tuning target)
ALIGN_ROT = np.eye(3)                         # single gate-calibration knob (Task 8), -> Phi_enc
WORLD_GRAVITY = np.array([0.0, 0.0, -1.0])


def _mat_to_quat_wxyz(rotm):
    x, y, z, w = _R.from_matrix(rotm).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def load_bounds(path=YAML):
    """23-entry joint_position_min/max (policy order) from action_wrist_hold.yaml."""
    import re
    txt = Path(path).read_text()

    def arr(key):
        m = re.search(key + r":\s*\[([^\]]*)\]", txt)
        vals = [float(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]
        assert len(vals) == 23, f"{key} has {len(vals)} entries, want 23"
        return np.array(vals, dtype=np.float64)

    return arr("joint_position_min"), arr("joint_position_max")


def build_proprio(jpos31, jvel31, imu_wxyz, gyro):
    """SDK31-serial joint state + IMU -> RobotProprio (IGRIS), base frame = base_link.

    Replicates state_handler.cpp (ComputeBaseLinkRotm / ProjectedGravity / AngVel):
        R_base = R_imu @ Rotate_Y(-90 deg)
        gravity(base) = R_base^T @ world_gravity
        angvel(base)  = R_base^T @ (R_imu @ gyro)
    ALIGN_ROT (residual base_link->G1 pelvis) is applied downstream inside Phi_enc.
    """
    r_imu = _quat_to_matrix(np.asarray(imu_wxyz, dtype=np.float64))
    r_base = r_imu @ IMU_MOUNT_ROT
    gravity = r_base.T @ WORLD_GRAVITY
    angvel = r_base.T @ (r_imu @ np.asarray(gyro, dtype=np.float64))
    jp = {SDK31[i]: float(jpos31[i]) for i in range(NUM_MOTORS)}
    jv = {SDK31[i]: float(jvel31[i]) for i in range(NUM_MOTORS)}
    return RobotProprio(joint_pos=jp, joint_vel=jv, base_angvel=angvel,
                        gravity=gravity, base_quat=_mat_to_quat_wxyz(r_base))


def build_motion_ref(qpos36):
    """redis G1 qpos [root_pos3, root_quat4 wxyz, q29 G1_MJ29] -> 10-frame MotionRef.

    ponytail: the redis key holds only the CURRENT serving frame, so we build a
    hold-current window (frame0..9 = now, vel 0). Upgrade path for a moving gate:
    buffer frames and sample current->future step5 once Task 7 streams a lookahead.
    """
    q = np.asarray(qpos36, dtype=np.float64).reshape(-1)
    jpos_il = q[7:36][C.MUJOCO_TO_ISAACLAB]   # G1_MJ29 (hardware) -> IsaacLab order
    return MotionRef(
        joint_positions=np.tile(jpos_il, (HISTORY, 1)),
        joint_velocities=np.zeros((HISTORY, NUM_JOINTS), dtype=np.float64),
        root_quat=np.tile(q[3:7], (HISTORY, 1)),
    )


def fallback_ref(g1_jpos_il, base_quat):
    """Static hold-current reference from the robot's own G1 proprio (no redis)."""
    return MotionRef(
        joint_positions=np.tile(np.asarray(g1_jpos_il, dtype=np.float64), (HISTORY, 1)),
        joint_velocities=np.zeros((HISTORY, NUM_JOINTS), dtype=np.float64),
        root_quat=np.tile(np.asarray(base_quat, dtype=np.float64), (HISTORY, 1)),
    )


def build_q31(codec_dict, measured31, jmin, jmax):
    """29-dof IGRIS codec dict -> SDK 31-motor q vector.

    Start from measured (holds wrists 19-21,26-28 and neck 29,30 in place), then
    overwrite the 23 policy slots with the codec targets (clamped to soft limits)
    via kActionsToSystemJointMapping. Neck kp/kd are 0 in the yaml (free); wrists
    kp30/kd3 (hold) — matching the proven action_wrist_hold config.
    """
    q31 = np.array([float(measured31[i]) for i in range(NUM_MOTORS)], dtype=np.float64)
    n_clamp = 0
    for j, slot in enumerate(K_ACTIONS_TO_SYSTEM):
        v = float(codec_dict[mc.POLICY_JOINT_NAMES[j]])
        c = min(max(v, jmin[j]), jmax[j])
        n_clamp += (c != v)
        q31[slot] = c
    return q31, n_clamp


class _RedisGet:
    """Minimal persistent-socket RESP GET (no redis-py; safe at 50 Hz)."""

    def __init__(self, host="127.0.0.1", port=6379):
        self.host, self.port, self.sock, self.buf = host, port, None, b""

    def _readline(self):
        while b"\r\n" not in self.buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("redis closed")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\r\n", 1)
        return line

    def _readn(self, n):
        while len(self.buf) < n + 2:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("redis closed")
            self.buf += chunk
        data, self.buf = self.buf[:n], self.buf[n + 2:]
        return data

    def get(self, key):
        try:
            if self.sock is None:
                self.sock = socket.create_connection((self.host, self.port), timeout=0.3)
                self.buf = b""
            self.sock.sendall(f"*2\r\n$3\r\nGET\r\n${len(key)}\r\n{key}\r\n".encode())
            hdr = self._readline()
            if hdr[:1] != b"$" or hdr == b"$-1":
                return None
            return self._readn(int(hdr[1:])).decode()
        except Exception:
            self.sock = None
            return None


class SonicTracker:
    def __init__(self, align_rot=None, redis_host="127.0.0.1", redis_port=6379,
                 enc=DEFAULT_ENC_ONNX, dec=DEFAULT_DEC_ONNX,
                 phi_enc=DEFAULT_PHI_ENC, phi_dec=DEFAULT_PHI_DEC):
        self.engine = SonicEngine(enc, dec)
        if os.environ.get("SONIC_CODEC_MATRIX", "1") == "1":
            # element #3: same linear matrix codec as the Isaac zero-shot (see sonic_engine)
            from sonic_engine import PhiEncMatrix, PhiDecoderMatrix
            self.phi_enc = PhiEncMatrix(align_rot=ALIGN_ROT if align_rot is None else align_rot)
            self.phi_dec = PhiDecoderMatrix()
            print("[sonic_tracker] codec=MATRIX (newcodec_linear)")
        else:
            self.phi_enc = PhiEnc(phi_enc, align_rot=ALIGN_ROT if align_rot is None else align_rot)
            self.phi_dec = PhiDecoder(str(phi_dec))
            print("[sonic_tracker] codec=affine/closed-form (S2-era)")
        self.step = SonicStep(self.engine, self.phi_dec, self.phi_enc)
        self.kp, self.kd = load_gains()
        self.jmin, self.jmax = load_bounds()
        self.redis = _RedisGet(redis_host, redis_port)
        self._jpos = self._jvel = self._imu = self._gyro = None  # state caches

    # --- reference --------------------------------------------------------
    def read_reference(self):
        raw = self.redis.get(REF_KEY)
        if not raw:
            return None
        try:
            d = json.loads(raw)
            if (time.time() * 1000.0 - float(d["record_meta"]["unix_ms"])) > REF_STALE_MS:
                return None
            return d["qpos"]
        except Exception:
            return None

    def make_ref(self, proprio):
        qpos = self.read_reference()
        if qpos is not None and len(qpos) == 36:
            return build_motion_ref(qpos), True
        return fallback_ref(self.phi_enc(proprio).jpos_isaaclab, proprio.base_quat), False

    # --- one control tick -------------------------------------------------
    def tick(self, jpos31, jvel31, imu_wxyz, gyro):
        proprio = build_proprio(jpos31, jvel31, imu_wxyz, gyro)
        ref, live = self.make_ref(proprio)
        codec = self.step.step(ref, proprio)            # {igris_joint: target}, 29
        q31, n_clamp = build_q31(codec, jpos31, self.jmin, self.jmax)
        return q31, codec, live, n_clamp

    # --- live 50 Hz loop (rclpy state + cyclonedds lowcmd, one process) ---
    def spin(self, duration=None, hz=50.0):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState, Imu
        from lowcmd_pub import LowCmdPublisherROS, PJS  # ROS-bridge route; raw DDS path dead (type-object mismatch, see lowcmd_pub)

        rclpy.init()
        node = Node("sonic_tracker")

        def on_state(m):
            self._jpos = np.asarray(m.position, dtype=np.float64)
            self._jvel = (np.asarray(m.velocity, dtype=np.float64)
                          if len(m.velocity) == NUM_MOTORS else np.zeros(NUM_MOTORS))

        def on_imu(m):
            o = m.orientation
            self._imu = np.array([o.w, o.x, o.y, o.z], dtype=np.float64)
            w = m.angular_velocity
            self._gyro = np.array([w.x, w.y, w.z], dtype=np.float64)

        node.create_subscription(JointState, STATE_TOPIC, on_state, qos_profile_sensor_data)
        node.create_subscription(Imu, IMU_TOPIC, on_imu, qos_profile_sensor_data)

        import threading
        ex = rclpy.executors.SingleThreadedExecutor()
        ex.add_node(node)
        threading.Thread(target=ex.spin, daemon=True).start()

        # wait for first state before creating the DDS publisher (never fork after)
        t_wait = time.monotonic()
        while (self._jpos is None or self._imu is None):
            if time.monotonic() - t_wait > 10.0:
                raise RuntimeError(f"no {STATE_TOPIC}/{IMU_TOPIC} within 10 s")
            time.sleep(0.05)

        pub = LowCmdPublisherROS(node)
        # --- element #4: sonic-x2 deploy tuning (configs/real_deploy_tuning/bigrun.yaml) ---------
        #   max_target_dev: leg 0.7 / waist 0.45 / arm 2.0 rad (측정 q 기준 편차 클램프)
        #   target_lpf_hz : leg 16 Hz 1차 LPF (waist/arm 0 = off), wrist freeze 는 build_q31 이 이미 수행
        deploy_tune = os.environ.get("SONIC_DEPLOY_TUNE", "1") == "1"
        if deploy_tune:
            import re as _re
            def _grp(name):
                if _re.search(r"Hip|Knee|Ankle", name): return "leg"
                if "Waist" in name: return "waist"
                return "arm"
            _dev = {"leg": 0.7, "waist": 0.45, "arm": 2.0}
            _names23 = list(mc.POLICY_JOINT_NAMES)
            _devv = np.array([_dev[_grp(n)] for n in _names23])
            _alpha = 1.0 - np.exp(-2.0 * np.pi * 16.0 / hz)   # leg 16 Hz @ control rate
            _lpf_mask = np.array([1.0 if (_grp(n) == "leg" and os.environ.get("SONIC_TUNE_LPF","1")=="1") else 0.0 for n in _names23])
            _lpf_state = {}
            print(f"[sonic_tracker] deploy-tune ON: dev(leg0.7/waist0.45/arm2.0), leg LPF 16Hz (a={_alpha:.2f})")
        
        zeros = [0.0] * NUM_MOTORS
        dt = 1.0 / hz
        nxt = time.monotonic()
        n, t0, t_log = 0, time.monotonic(), time.monotonic()
        print(f"[sonic_tracker] loop @ {hz:.0f} Hz  state={STATE_TOPIC}  imu={IMU_TOPIC}")
        try:
            while duration is None or time.monotonic() - t0 < duration:
                if self._jpos is None or len(self._jpos) != NUM_MOTORS or self._imu is None:
                    time.sleep(dt); continue
                q31, _codec, live, n_clamp = self.tick(self._jpos, self._jvel,
                                                       self._imu, self._gyro)
                if deploy_tune:
                    q31 = np.asarray(q31, dtype=np.float64).copy()
                    _slots = list(K_ACTIONS_TO_SYSTEM)
                    _tgt = q31[_slots]
                    _meas = np.asarray(self._jpos, dtype=np.float64)[_slots]
                    _tgt = np.clip(_tgt, _meas - _devv, _meas + _devv)      # max_target_dev
                    _prev = _lpf_state.get("t", _tgt)
                    _filt = _prev + _alpha * (_tgt - _prev)                  # leg 16 Hz LPF
                    _tgt = _lpf_mask * _filt + (1.0 - _lpf_mask) * _tgt
                    _lpf_state["t"] = _tgt
                    q31[_slots] = _tgt
                pub.send(list(q31), zeros, self.kp, self.kd, PJS)
                n += 1
                now = time.monotonic()
                if now - t_log >= 2.0:
                    print(f"[sonic_tracker] action_hz={n / (now - t0):.1f}  "
                          f"ref={'live' if live else 'fallback'}  clamped={n_clamp}")
                    t_log = now
                nxt += dt
                time.sleep(max(0.0, nxt - time.monotonic()))
        finally:
            print(f"[sonic_tracker] stopped after {n} ticks "
                  f"({n / max(1e-6, time.monotonic() - t0):.1f} Hz)")
            ex.shutdown(); node.destroy_node(); rclpy.shutdown()

    # --- dry run: dummy state, no DDS, no sim -----------------------------
    def dry_run(self, ticks=55, hz=50.0):
        jpos = np.zeros(NUM_MOTORS); jvel = np.zeros(NUM_MOTORS)
        # standing-ish knees/hips/ankles so the fallback ref is a plausible stance
        for nm, v in (("Joint_Hip_Pitch_Left", -0.2), ("Joint_Hip_Pitch_Right", -0.2),
                      ("Joint_Knee_Pitch_Left", 0.3), ("Joint_Knee_Pitch_Right", 0.3),
                      ("Joint_Ankle_Pitch_Left", -0.1), ("Joint_Ankle_Pitch_Right", -0.1)):
            jpos[SDK31.index(nm)] = v
        imu_wxyz = np.array([1.0, 0.0, 0.0, 0.0]); gyro = np.zeros(3)

        dt = 1.0 / hz
        nxt = time.monotonic(); t0 = time.monotonic()
        for k in range(ticks):
            q31, codec, live, n_clamp = self.tick(jpos, jvel, imu_wxyz, gyro)
            cv = np.array(list(codec.values()))
            assert len(codec) == NUM_JOINTS, f"codec has {len(codec)} joints"
            assert np.all(np.isfinite(cv)), "non-finite codec target"
            assert q31.shape == (NUM_MOTORS,) and np.all(np.isfinite(q31)), "bad q31"
            # policy slots within soft limits (post-clamp, so this must hold)
            for j, slot in enumerate(K_ACTIONS_TO_SYSTEM):
                assert self.jmin[j] - 1e-6 <= q31[slot] <= self.jmax[j] + 1e-6, \
                    f"policy slot {slot} out of bounds"
            # wrist + neck slots must equal measured (HOLD)
            for slot in HOLD_SLOTS:
                assert q31[slot] == jpos[slot], f"hold slot {slot} not held"
            nxt += dt
            time.sleep(max(0.0, nxt - time.monotonic()))
        hz_eff = ticks / (time.monotonic() - t0)
        # in-bounds report on RAW (pre-clamp) policy targets
        raw = np.array([codec[mc.POLICY_JOINT_NAMES[j]] for j in range(23)])
        oob = int(np.sum((raw < self.jmin) | (raw > self.jmax)))
        print(f"[dry-run] OK  ticks={ticks}  action_hz={hz_eff:.1f}  ref={'live' if live else 'fallback'}")
        print(f"[dry-run] codec 29 finite, q31 built, HOLD slots {HOLD_SLOTS} held, "
              f"policy clamped_this_tick={n_clamp}, raw_out_of_bounds={oob}/23")
        print(f"[dry-run] sample q31[policy LSP slot15]={q31[15]:.4f}  "
              f"q31[WY slot0]={q31[0]:.4f}  q31[LHP slot3]={q31[3]:.4f}")
        return hz_eff


def main():
    ap = argparse.ArgumentParser(description="SONIC tracker node (S2 Task 6)")
    ap.add_argument("--dry-run", action="store_true", help="dummy state, no DDS/sim; assert wiring")
    ap.add_argument("--duration", type=float, default=None, help="live loop seconds (default: run until Ctrl-C)")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--redis-host", default="127.0.0.1")
    ap.add_argument("--ticks", type=int, default=55, help="dry-run tick count")
    args = ap.parse_args()

    trk = SonicTracker(redis_host=args.redis_host)
    if args.dry_run:
        trk.dry_run(ticks=args.ticks, hz=args.hz)
        return 0
    trk.spin(duration=args.duration, hz=args.hz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
