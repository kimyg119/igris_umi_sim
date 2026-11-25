#!/usr/bin/env python3
"""S2 Task-8 gate metrics — IGRIS standing zero-shot (G1 gate).

Reads a state_recorder.py recording (t, base[N,7]=pos3+quat_wxyz4, q[N,>=31]
in /igris_c_001/rt/joint_state == SDK31 serial order) plus the tracker/sim logs,
and computes the G1 gate:

  base-z trajectory + in-band hold, survival time, no-fall,
  tracking RMS of the 23 policy joints (measured vs the STATIC reference),
  base lean (deg + azimuth, for ALIGN_ROT diagnosis),
  action_hz (from sonic.log), physics Hz (from sim.log "Control: N Hz").

TRACKING-RMS DEFINITION (honest, self-contained): the redis reference is a
STATIC standing G1 pose, so a perfect tracker holds every policy joint constant.
We therefore report per-joint RMS-about-the-window-mean (= how still the tracked
joints are held). A constant offset between the codec target and the settled
pose is steady-state PD error, not a tracking failure, so it is excluded by
design. Max drift from mean is reported alongside.

G1 PASS = no fall, z within z_target +/- band continuously for >= hold seconds,
action_hz ~50, physics ~300 Hz.

Run:  python3 s2_metrics.py --rec logs/s2_standing.npz \
          --sonic-log logs/sonic.log --sim-log logs/sim.log --out logs/g1_standing.json
Self-check: python3 s2_metrics.py --selftest
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

# 23 policy joints -> SDK31 slot (spike_hold.K_ACTIONS_TO_SYSTEM; recorded q is SDK31 order)
K_ACTIONS_TO_SYSTEM = [15, 22, 0, 16, 23, 1, 17, 24, 2, 18, 25,
                       3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]
POLICY_JOINT_NAMES = (
    "Joint_Shoulder_Pitch_Left", "Joint_Shoulder_Pitch_Right", "Joint_Waist_Yaw",
    "Joint_Shoulder_Roll_Left", "Joint_Shoulder_Roll_Right", "Joint_Waist_Roll",
    "Joint_Shoulder_Yaw_Left", "Joint_Shoulder_Yaw_Right", "Joint_Waist_Pitch",
    "Joint_Elbow_Pitch_Left", "Joint_Elbow_Pitch_Right",
    "Joint_Hip_Pitch_Left", "Joint_Hip_Pitch_Right", "Joint_Hip_Roll_Left", "Joint_Hip_Roll_Right",
    "Joint_Hip_Yaw_Left", "Joint_Hip_Yaw_Right", "Joint_Knee_Pitch_Left", "Joint_Knee_Pitch_Right",
    "Joint_Ankle_Pitch_Left", "Joint_Ankle_Pitch_Right", "Joint_Ankle_Roll_Left", "Joint_Ankle_Roll_Right",
)


def _longest_true_run(mask, t):
    """Longest continuous span (seconds) where mask is True, measured on time axis t."""
    best = cur_start = None
    best_dur = 0.0
    for i, m in enumerate(mask):
        if m and cur_start is None:
            cur_start = t[i]
        if (not m or i == len(mask) - 1) and cur_start is not None:
            end = t[i] if m else t[i - 1]
            best_dur = max(best_dur, end - cur_start)
            cur_start = None
    return best_dur


def _base_lean(quat_wxyz):
    """quat (w,x,y,z) world orientation of base -> (lean_deg, azimuth_deg) of the base z-axis.

    lean_deg = angle between base up-axis and world up. azimuth = compass dir the
    top tilts toward in the world XY plane (0=+x forward, 90=+y left)."""
    w, x, y, z = quat_wxyz.T
    zx = 2 * (x * z + w * y)
    zy = 2 * (y * z - w * x)
    zz = 1 - 2 * (x * x + y * y)
    lean = np.degrees(np.arccos(np.clip(zz, -1.0, 1.0)))
    azim = np.degrees(np.arctan2(zy, zx))
    return lean, azim


def parse_action_hz(path):
    vals = re.findall(r"action_hz=([0-9.]+)", Path(path).read_text()) if Path(path).exists() else []
    return float(vals[-1]) if vals else None


def parse_physics_hz(path):
    vals = re.findall(r"Control:\s*([0-9.]+)\s*Hz", Path(path).read_text()) if Path(path).exists() else []
    return float(vals[-1]) if vals else None


def parse_lowcmd_hz(path):
    """Authoritative action rate = LowCmd Hz the sim controller reports receiving.

    (sonic_tracker's stdout action_hz print is block-buffered to its log, so it is
    unreliable while the process runs; the sim's LowCmd counter is ground truth.)"""
    vals = re.findall(r"LowCmd:\s*([0-9.]+)\s*Hz", Path(path).read_text()) if Path(path).exists() else []
    return float(vals[-1]) if vals else None


def compute(rec, action_hz, physics_hz, settle=3.0, z_target=0.88, band=0.03,
            hold=30.0, fall_z=0.6):
    t = np.asarray(rec["t"], dtype=np.float64)
    base = np.asarray(rec["base"], dtype=np.float64)   # [N,7] pos3 + quat wxyz4
    q = np.asarray(rec["q"], dtype=np.float64)         # [N,>=31] SDK31 order
    assert q.shape[1] >= 31, f"recorded q has {q.shape[1]} cols, need >=31 (SDK31)"
    z = base[:, 2]

    # survival over the WHOLE recording (fall = z ever below fall_z)
    below = np.where(z < fall_z)[0]
    fell = below.size > 0
    survival_time = float(t[below[0]] - t[0]) if fell else float(t[-1] - t[0])

    # evaluation window: skip handoff settle
    ev = t - t[0] >= settle
    if ev.sum() < 2:
        ev = np.ones_like(t, dtype=bool)
    te, ze = t[ev], z[ev]
    in_band = np.abs(ze - z_target) <= band
    longest_hold = _longest_true_run(in_band, te)

    lean, azim = _base_lean(base[ev, 3:7])

    # tracking is measured ONLY over the genuine standing phase (z still up, past the
    # handoff transient) so a collapse does not pollute it -- this is "tracking RMS before
    # fall". 23 policy joints, RMS about the standing-window mean (static-reference stability).
    stand = (z > z_target - 3 * band) & (t - t[0] >= 0.5)
    if stand.sum() < 2:
        stand = z > z_target - 3 * band
    track_window_s = float(t[stand][-1] - t[stand][0]) if stand.sum() >= 2 else 0.0
    pj = q[stand][:, K_ACTIONS_TO_SYSTEM]              # [M,23]
    dev = pj - pj.mean(axis=0, keepdims=True)
    per_joint_rms = np.sqrt((dev ** 2).mean(axis=0))   # [23]
    overall_rms = float(np.sqrt((dev ** 2).mean()))
    max_drift = float(np.abs(dev).max())
    worst = sorted(zip(POLICY_JOINT_NAMES, per_joint_rms.tolist()),
                   key=lambda kv: -kv[1])[:5]

    gates = {
        "no_fall": bool(not fell),
        "z_in_band_hold_ge": bool(longest_hold >= hold),
        "action_hz_ok": bool(action_hz is not None and 45.0 <= action_hz <= 55.0),
        "physics_hz_ok": bool(physics_hz is not None and physics_hz >= 250.0),
    }
    verdict = "PASS" if all(gates.values()) else "FAIL"

    return {
        "verdict": verdict,
        "gates": gates,
        "params": {"z_target": z_target, "band": band, "hold_required_s": hold,
                   "fall_z": fall_z, "settle_s": settle},
        "duration_s": float(t[-1] - t[0]),
        "n_samples": int(t.size),
        "survival_time_s": survival_time,
        "fell": fell,
        "base_z": {"min": float(ze.min()), "max": float(ze.max()),
                   "mean": float(ze.mean()), "std": float(ze.std()),
                   "start": float(z[0]), "end": float(z[-1])},
        "z_in_band_longest_s": float(longest_hold),
        "z_in_band_fraction": float(in_band.mean()),
        "base_lean_deg": {"mean": float(lean.mean()), "max": float(lean.max()),
                          "end": float(lean[-1]),
                          "azimuth_end_deg": float(azim[-1]),
                          "azimuth_mean_deg": float(np.degrees(np.arctan2(
                              np.sin(np.radians(azim)).mean(),
                              np.cos(np.radians(azim)).mean())))},
        "tracking_rms_rad": {"overall": overall_rms, "max_drift": max_drift,
                             "window_s": track_window_s,
                             "worst5": [{"joint": n, "rms": v} for n, v in worst]},
        "action_hz": action_hz,
        "physics_hz": physics_hz,
    }


def _selftest():
    n = 2000
    t = np.arange(n) / 50.0
    base = np.zeros((n, 7)); base[:, 2] = 0.881 + 0.005 * np.sin(t); base[:, 3] = 1.0
    q = np.zeros((n, 31)) + 0.01 * np.sin(t)[:, None]
    r = compute({"t": t, "base": base, "q": q}, action_hz=50.0, physics_hz=299.9)
    assert r["verdict"] == "PASS", r["gates"]
    assert r["z_in_band_longest_s"] >= 30.0
    # inject a fall at t=20 s
    base[1000:, 2] = 0.3
    r2 = compute({"t": t, "base": base, "q": q}, action_hz=50.0, physics_hz=299.9)
    assert r2["verdict"] == "FAIL" and r2["fell"] and r2["survival_time_s"] < 30.0, r2
    print("[selftest] OK  pass-case and fall-case both correct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec"); ap.add_argument("--sonic-log")
    ap.add_argument("--sim-log"); ap.add_argument("--out")
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--z-target", type=float, default=0.88)
    ap.add_argument("--band", type=float, default=0.03)
    ap.add_argument("--hold", type=float, default=30.0)
    ap.add_argument("--fall-z", type=float, default=0.6)
    ap.add_argument("--note", default="")
    ap.add_argument("--align-rot", default="identity", help="label for ALIGN_ROT used this run")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest(); return 0

    rec = np.load(a.rec, allow_pickle=True)
    action_hz = parse_action_hz(a.sonic_log) if a.sonic_log else None
    action_hz_source = "sonic.log"
    if action_hz is None and a.sim_log:                # buffered tracker stdout -> use sim LowCmd counter
        action_hz = parse_lowcmd_hz(a.sim_log)
        action_hz_source = "sim.log LowCmd counter"
    physics_hz = parse_physics_hz(a.sim_log) if a.sim_log else None
    out = compute(rec, action_hz, physics_hz, settle=a.settle, z_target=a.z_target,
                  band=a.band, hold=a.hold, fall_z=a.fall_z)
    out["align_rot"] = a.align_rot
    out["action_hz_source"] = action_hz_source
    if a.note:
        out["note"] = a.note
    out["recording"] = a.rec
    txt = json.dumps(out, indent=2)
    if a.out:
        Path(a.out).write_text(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
