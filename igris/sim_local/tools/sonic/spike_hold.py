#!/usr/bin/env python3
"""S2 command-path spike — prove Python cyclonedds -> lowcmd -> IGRIS controller.

Stands the robot (resume/torque/home/reset), captures the 31-joint hold pose from
/mujoco/joint_states, then hands off to a Python cyclonedds LowCmd publisher
(wrist-hold gains) as the SOLE lowcmd source and switches the controller to LOW.

PASS = COMMAND PATH: the controller receives our lowcmd at ~50 Hz AND the measured
joints track the commanded pose (small |measured-commanded| via lowstate) — i.e.
Python drives the sim. NOTE: a pure open-loop joint hold does NOT balance the
humanoid, so the robot still tips (base z drops) within ~1-2 s; that is a physics
limitation of static holding, not a command-path failure, and does not block the
Task-6 tracker (which publishes active, balancing lowcmd).

Run INSIDE the igris-sim container (needs cyclonedds 0.10.5 + ROS 2 jazzy):
  docker exec igris-sim bash -lc \
    'source /opt/ros/jazzy/setup.bash; \
     source /igris/sim_local/tools_ws/install/setup.bash; \
     export CYCLONEDDS_URI=file:///robros/configs/cyclonedds/cyclonedds_local.xml; \
     python3 /igris/sim_local/tools/sonic/spike_hold.py'

  --check   run only the joint-mapping / gain self-asserts (no sim), then exit.

Note: the SDK controller trips JOINT_HOLD if lowcmd goes stale >~0.5 s, so the
publisher is started and flowing BEFORE `mode low` — the handoff has no stale gap.
"""
import argparse
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lowcmd_pub import NUM_MOTORS, SYS_JOINT_NAMES  # noqa: E402

# SONIC_PARAMS_YAML lets the sonic runner swap in the vendor sim2real gains
# (params_sonic_vendor.yaml) without touching the proven teleop wrist-hold config.
YAML = os.environ.get("SONIC_PARAMS_YAML", "/igris/sim_local/action_wrist_hold.yaml")
CYCLONEDDS_XML = "/robros/configs/cyclonedds/cyclonedds_local.xml"
OUT_JSON = "/igris/sim_local/logs/spike_hold.json"

# 23-entry policy/action -> SDK 31 system-joint mapping (types.hpp
# kActionsToSystemJointMapping, evaluated from the JointIndex enum).
K_ACTIONS_TO_SYSTEM = [15, 22, 0, 16, 23, 1, 17, 24, 2, 18, 25,
                       3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]
# System joints NOT driven by the 23 policy actions: 6 wrists + 2 neck.
HOLD_SLOTS = [19, 20, 21, 26, 27, 28, 29, 30]


# ---------------------------------------------------------------------------
def load_gains(path=YAML):
    """Read the 31-entry kp/kd arrays from action_wrist_hold.yaml (no yaml dep)."""
    txt = open(path).read()

    def arr(key):
        m = re.search(key + r":\s*\[([^\]]*)\]", txt)
        vals = [float(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]
        assert len(vals) == NUM_MOTORS, f"{key} has {len(vals)} entries, want 31"
        return vals

    return arr("kp"), arr("kd")


def validate_mapping(kp, kd):
    """Assert the 23->31 mapping and the wrist/neck hold slots (spec Task-5 step 4)."""
    assert len(K_ACTIONS_TO_SYSTEM) == 23
    assert len(set(K_ACTIONS_TO_SYSTEM)) == 23, "mapping has duplicates"
    assert all(0 <= i < NUM_MOTORS for i in K_ACTIONS_TO_SYSTEM)
    # the un-actioned complement must be exactly wrists (19-21, 26-28) + neck (29,30)
    complement = sorted(set(range(NUM_MOTORS)) - set(K_ACTIONS_TO_SYSTEM))
    assert complement == HOLD_SLOTS, f"un-actioned joints {complement} != {HOLD_SLOTS}"
    # wrist-hold gains: wrists kp=30/kd=3, neck kp=0/kd=0
    for i in (19, 20, 21, 26, 27, 28):
        assert kp[i] == 30.0 and kd[i] == 3.0, f"wrist slot {i} kp/kd={kp[i]}/{kd[i]}"
    for i in (29, 30):
        assert kp[i] == 0.0 and kd[i] == 0.0, f"neck slot {i} kp/kd={kp[i]}/{kd[i]}"
    print("[ok] 23->31 mapping + wrist(kp30)/neck(kp0) hold slots validated")


# ---------------------------------------------------------------------------
def ctl(*args, timeout=15):
    """Fire one igris_ctl command (sim resume / torque on / mode home|low / sim reset)."""
    cmd = ["ros2", "run", "igris_sim_ctl", "igris_ctl", *args,
           "--cyclonedds-xml", CYCLONEDDS_XML]
    subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)


def rt_physics():
    """Promote the busiest MuJoCo thread to SCHED_FIFO 50 (300 Hz loop rides it)."""
    try:
        pid = subprocess.check_output(
            ["bash", "-c",
             "ps -eo pid,stat,comm | awk '$2!~/Z/ && $3==\"igris_c_mujoco\"{print $1}' | head -1"],
            text=True).strip()
        if not pid:
            return
        tasks = f"/proc/{pid}/task"
        def busy(t):
            with open(f"{tasks}/{t}/stat") as f:
                p = f.read().split()
            return int(p[13]) + int(p[14])
        before = {t: busy(t) for t in os.listdir(tasks)}
        time.sleep(1.0)
        best, tid = 0, None
        for t in os.listdir(tasks):
            d = busy(t) - before.get(t, 0)
            if d > best:
                best, tid = d, t
        if tid and best >= 10:
            subprocess.run(["chrt", "-f", "-p", "50", tid], capture_output=True)
            print(f"[ok] MuJoCo physics tid {tid} -> SCHED_FIFO 50 (cpu {best}%)")
    except Exception as e:  # rt is best-effort; sim still runs without it
        print(f"[warn] rt_physics skipped: {e}")


def base_z():
    """Current base height from /mujoco/base_odom (nav_msgs/Odometry)."""
    try:
        out = subprocess.check_output(
            ["ros2", "topic", "echo", "--once", "--field", "pose.pose.position.z",
             "/mujoco/base_odom"], timeout=8, text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            line = line.strip()
            if re.match(r"^-?\d", line):
                return float(line)
    except Exception:
        pass
    return None


def capture_pose():
    """31-vector of joint positions in SDK system order, gathered by name."""
    out = subprocess.check_output(
        ["ros2", "topic", "echo", "--once", "/mujoco/joint_states"],
        timeout=10, text=True, stderr=subprocess.DEVNULL)
    block = out.split("---")[0].splitlines()
    names, positions, mode = [], [], None
    for ln in block:
        s = ln.strip()
        if s.startswith("name:"):
            mode = "name"; continue
        if s.startswith("position:"):
            mode = "pos"; continue
        if s.startswith("velocity:") or s.startswith("effort:") or s.startswith("header:"):
            mode = None; continue
        if s.startswith("- ") and mode == "name":
            names.append(s[2:].strip())
        elif s.startswith("- ") and mode == "pos":
            positions.append(float(s[2:].strip()))
    by_name = dict(zip(names, positions))
    missing = [n for n in SYS_JOINT_NAMES if n not in by_name]
    assert not missing, f"joint_states missing {missing}; got {names}"
    return [by_name[n] for n in SYS_JOINT_NAMES]


def stand(max_reset=8, stand_min=0.7):
    """resume + rt_physics + torque on + mode home + reset-until-standing."""
    print("[1] sim resume"); ctl("sim", "resume"); time.sleep(2)
    rt_physics()
    print("[2] torque on"); ctl("torque", "on"); time.sleep(2)
    print("[3] mode home"); ctl("mode", "home"); time.sleep(4)
    print("[4] sim reset (stand)")
    for i in range(max_reset):
        ctl("sim", "reset"); time.sleep(3)
        z = base_z() or 0.0
        print(f"    reset #{i + 1} -> z={z:.3f}")
        if z > stand_min:
            return z
    raise RuntimeError("failed to stand")


# ---------------------------------------------------------------------------
def _publisher_proc(q31, kp, kd, hz, stop_evt, count, trk_sum, trk_max, trk_n):
    """Run in a SEPARATE process so the (forking) parent holds no DDS participant.

    Forking the parent while a background thread writes to CycloneDDS corrupts
    libddsc locks and silently kills the stream — isolating the writer in its own
    fork-free process is the fix.

    Also subscribes to lowstate to measure joint tracking error (|measured -
    commanded|): small error proves the controller is APPLYING our lowcmd.
    """
    from lowcmd_pub import PJS, LowCmdPublisher, LowStateSub  # DDS init in child only
    pub = LowCmdPublisher(domain=0)
    sub = LowStateSub(domain=0)
    zeros = [0.0] * NUM_MOTORS
    dt = 1.0 / hz
    nxt = time.monotonic()
    last_probe = 0.0
    while not stop_evt.is_set():
        pub.send(q31, zeros, kp, kd, kinematic_modes=PJS)
        with count.get_lock():
            count.value += 1
        now = time.monotonic()
        if now - last_probe > 0.1:  # sample tracking ~10 Hz
            last_probe = now
            meas = sub.joint_q()
            if meas is not None:
                err = sum(abs(meas[i] - q31[i]) for i in range(NUM_MOTORS)) / NUM_MOTORS
                with trk_n.get_lock():
                    trk_sum.value += err
                    trk_n.value += 1
                    if err > trk_max.value:
                        trk_max.value = err
        nxt += dt
        time.sleep(max(0.0, nxt - time.monotonic()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=12.0, help="hold seconds (>=10)")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--check", action="store_true", help="run mapping asserts only")
    args = ap.parse_args()

    kp, kd = load_gains()
    validate_mapping(kp, kd)
    if args.check:
        print("[check-only] OK"); return 0

    z_stand = stand()
    q31 = capture_pose()
    print(f"[5] captured 31-joint hold pose (stand z={z_stand:.3f}); "
          f"q[0:3]={[round(v,3) for v in q31[:3]]}")

    # Handoff: stop the student policy so our publisher is the sole lowcmd source,
    # start publishing BEFORE mode low so there is no stale gap.
    print("[6] stop student_130 (igris_c_gmt_public_node)")
    subprocess.run(["pkill", "-f", "igris_c_gmt_public_nod[e]"], capture_output=True)
    time.sleep(0.3)

    print("[7] start cyclonedds lowcmd publisher @ %.0f Hz (separate process)" % args.hz)
    stop_evt = mp.Event()
    count = mp.Value("i", 0)
    trk_sum, trk_max, trk_n = mp.Value("d", 0.0), mp.Value("d", 0.0), mp.Value("i", 0)
    proc = mp.Process(target=_publisher_proc,
                      args=(q31, kp, kd, args.hz, stop_evt, count, trk_sum, trk_max, trk_n),
                      daemon=True)
    proc.start()
    time.sleep(0.8)  # let discovery match + frames flow before LOW

    print("[8] mode low (handoff into LOW under our publisher)")
    ctl("mode", "low")

    print(f"[9] hold + measure for {args.duration:.0f}s")
    samples, t0 = [], time.monotonic()
    while time.monotonic() - t0 < args.duration:
        z = base_z()
        if z is not None:
            samples.append(z)
            print(f"    t={time.monotonic() - t0:4.1f}s  z={z:.3f}")
    stop_evt.set(); proc.join(timeout=2)

    zmin, zmax = min(samples), max(samples)
    zmean = sum(samples) / len(samples)
    dur = time.monotonic() - t0
    hz_eff = count.value / dur
    trk_mean = (trk_sum.value / trk_n.value) if trk_n.value else None

    # COMMAND-PATH proof (what Task 5 validates): the controller receives our
    # lowcmd at ~50 Hz AND the measured joints track our commanded pose
    # (small tracking error => controller is applying our lowcmd).
    cmd_path_ok = (hz_eff >= 45.0 and trk_n.value > 0 and
                   trk_mean is not None and trk_mean < 0.15)
    # HOLD stability (a proxy): base stays up. An open-loop joint hold cannot
    # balance the humanoid, so this is expected to fail even when the path works.
    held = (zmin >= 0.85) and (zmax - zmin) < 0.06 and zmin >= 0.7

    result = {
        "command_path_ok": cmd_path_ok,
        "hold_standing": held,
        "publish_hz_effective": round(hz_eff, 1), "publish_count": count.value,
        "joint_track_err_mean_rad": round(trk_mean, 4) if trk_mean is not None else None,
        "joint_track_err_max_rad": round(trk_max.value, 4),
        "lowstate_samples": trk_n.value,
        "stand_z": round(z_stand, 4),
        "z_min": round(zmin, 4), "z_max": round(zmax, 4), "z_mean": round(zmean, 4),
        "n_z_samples": len(samples), "hold_duration_s": round(dur, 2),
        "topic": "igris_c_001/rt/lowcmd", "type": "igris_c::msg::dds::LowCmd",
        "qos": "best_effort/volatile/keep_last5", "cyclonedds": "0.10.5",
        "note": ("open-loop joint hold does not balance the humanoid; joints track "
                 "the command (small err) but the robot tips — active balancing "
                 "(HIGH mode / RL policy) is needed to hold standing."),
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(result, open(OUT_JSON, "w"), indent=2)
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    print("COMMAND-PATH: " + ("PASS" if cmd_path_ok else "FAIL") +
          "   |   HOLD-STANDING: " + ("PASS" if held else "FAIL (open-loop cannot balance)"))
    return 0 if cmd_path_ok else 1


if __name__ == "__main__":
    sys.exit(main())
