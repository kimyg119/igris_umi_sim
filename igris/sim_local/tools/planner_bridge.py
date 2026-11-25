#!/usr/bin/env python3
"""S0 플래너 브릿지 (docs/superpowers/specs/2026-08-26-s0-planner-bridge-design.md).

얼린 G1 MotionBricks(planner_sonic.onnx) 를 G1 공간에서 돌리고(컨텍스트 = 자기 서빙 플랜),
50 Hz 서빙 프레임만 Φ_dec → pelvis MJCF FK → torso 규약 GMR 페이로드 → Redis 로 보낸다.

    ./start_planner.sh --script tools/scripts/s0_walk.json          # 스크립트 시퀀스
    ./start_planner.sh --keyboard                                    # 키보드 (상류 바인딩)
    ./start_planner.sh --audit --script ... --out /tmp/audit.npz     # 발행 없이 G-K 통계
"""
from __future__ import annotations

import argparse
import json
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RET = HERE.parent.parent / "retargeting"
for p in (RET / "tools", RET / "GMR"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import motion_common as mc  # noqa: E402
import igris_codec as ic  # noqa: E402

FPS, PLANNER_HZ, LOOK_AHEAD, BLEND = 50.0, 10.0, 2, 8
LIMIT_TOL = 0.01          # rad. 참조는 PD 목표일 뿐 — 배포 액션 빌더가 자체 소프트 한계로 클립한다 (감사 R9)
MODE_SETS = [[0, 1, 2, 3], [4, 5, 6, 7, 8, 14], [9, 10, 11, 12, 13, 15, 16], list(range(17, 27))]
STATIC_MODES = {0, 4, 5, 6, 7, 9}


def replan_interval(mode):
    return 0.1 if mode == 3 else 0.2 if mode in (8, 14) else 1.0


@dataclass
class Intent:
    mode: int = 0
    speed: float | None = None                       # m/s, IGRIS-C 기준; None = 모드 기본
    move: np.ndarray = field(default_factory=lambda: np.zeros(2))
    face: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    height: float | None = None                      # m, IGRIS-C 골반 높이

    def key(self):
        return (self.mode, self.speed, tuple(np.round(self.move, 4)), tuple(np.round(self.face, 4)), self.height)


# ------------------------------------------------------------------ 플래너
class StubPlanner:
    """테스트용: 컨텍스트 마지막 프레임을 48 프레임 복제하고 move 방향으로 0.5 m/s 전진."""

    def plan(self, ctx, intent_g1, seed):
        q = np.repeat(ctx[-1:], 48, axis=0)
        v = np.array(intent_g1["move"] + [0.0]) * (intent_g1["target_vel"] if intent_g1["target_vel"] > 0 else 0.0)
        q[:, :3] += np.arange(48)[:, None] / 30.0 * v
        return q


class OnnxPlanner:
    def __init__(self, path, provider="cuda"):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 1
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider == "cuda" else ["CPUExecutionProvider"]
        self.s = ort.InferenceSession(str(path), so, providers=prov)
        assert len(self.s.get_inputs()) == 11, "V1/V2 플래너(11 입력)만 지원"
        used = self.s.get_providers()
        print(f"[planner] providers={used}", flush=True)
        if provider == "cuda" and "CUDAExecutionProvider" not in used:
            print("[planner] WARNING: CUDA EP 미로드 → CPU 실행(57~77 ms). start_planner.sh 의 LD_LIBRARY_PATH 확인", flush=True)

    def plan(self, ctx, intent_g1, seed):
        feed = {
            "context_mujoco_qpos": ctx[None].astype(np.float32),
            "target_vel": np.array([intent_g1["target_vel"]], np.float32),
            "mode": np.array([intent_g1["mode"]], np.int64),
            "movement_direction": np.array([intent_g1["move"] + [0.0]], np.float32),
            "facing_direction": np.array([intent_g1["face"] + [0.0]], np.float32),
            "height": np.array([intent_g1["height"]], np.float32),
            "random_seed": np.array([seed], np.int64),
            "has_specific_target": np.zeros((1, 1), np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), np.float32),
            "specific_target_headings": np.zeros((1, 4), np.float32),
            "allowed_pred_num_tokens": np.ones((1, 11), np.int64),
        }
        q, n = self.s.run(None, feed)
        return q[0, : int(n[0])].astype(np.float64)


# ------------------------------------------------------------------ 시간축
def _interp(q, t_src, t_dst):
    pos = mc.linear_resample(q[:, :3], t_src, t_dst)
    quat = mc.slerp_resample(q[:, 3:7], t_src, t_dst)
    jnt = mc.linear_resample(q[:, 7:], t_src, t_dst)
    return np.concatenate([pos, quat, jnt], 1)


def resample_30_to_50(q30):
    t_src = np.arange(len(q30)) / 30.0
    return _interp(q30, t_src, mc.resample_times(len(q30), 30.0, FPS))


def _lerp_frame(a, b, w):
    out = (1 - w) * a + w * b
    out[3:7] = mc.slerp_resample(np.stack([a[3:7], b[3:7]]), np.array([0.0, 1.0]), np.array([w]))[0]
    return out


def blend(old, cur, new, offset, nb=BLEND):
    tail = old[cur:] if cur < len(old) else old[-1:]
    L = max(len(tail), offset + len(new))
    out = np.empty((L, 36))
    for i in range(L):
        o = tail[min(i, len(tail) - 1)]
        if i < offset:
            out[i] = o
        else:
            j = i - offset
            out[i] = _lerp_frame(o, new[min(j, len(new) - 1)], min(1.0, (j + 1) / nb))
    return out


class Serving:
    def __init__(self, q50):
        self.lock = threading.Lock()
        self.buf, self.current = q50, 0
        self.seam_steps = []                      # 감사용: 플랜 교체 직후 관절 스텝

    def frame(self):
        with self.lock:
            f = self.buf[min(self.current, len(self.buf) - 1)].copy()
            self.current += 1
        return f

    def context(self, gen_frame):
        with self.lock:
            t_src = np.arange(len(self.buf)) / FPS
            t = np.clip(gen_frame / FPS + np.arange(4) / 30.0, 0.0, t_src[-1])
            return _interp(self.buf, t_src, t)

    def apply(self, new50, gen_frame):
        with self.lock:
            offset = max(0, gen_frame - self.current)
            self.buf = blend(self.buf, self.current, new50, offset)
            self.current = 0
            ramp = self.buf[max(0, offset - 1): offset + BLEND + 1, 7:]   # 옛 플랜 마지막 프레임 ~ 램프 끝
            self.seam_steps.append(float(np.abs(np.diff(ramp, axis=0)).max()) if len(ramp) > 1 else 0.0)


# ------------------------------------------------------------------ 입력
class ScriptInput:
    """JSON 리스트 [{"t":초, "mode":int, "speed":float|null, "move":[x,y]|null, "face":[x,y]|null, "height":float|null}]"""

    def __init__(self, path, start_file=None):
        self.events = sorted(json.loads(Path(path).read_text()), key=lambda e: e["t"])
        self.i = 0
        self.start_file, self.t_off = start_file, 0.0   # --start-file: 파일이 생길 때까지 idle, 그 순간이 스크립트 t=0

    def poll(self, intent, now):
        if self.start_file is not None:
            if not Path(self.start_file).exists():
                return False
            print(f"[planner] start-file seen at t={now:.2f}s → script t=0", flush=True)
            self.t_off, self.start_file = now, None
        now -= self.t_off
        while self.i < len(self.events) and self.events[self.i]["t"] <= now:
            e = self.events[self.i]; self.i += 1
            intent.mode = int(e.get("mode", intent.mode))
            if "speed" in e:
                intent.speed = e["speed"]
            if "move" in e:
                intent.move = np.array(e["move"] or [0.0, 0.0], float)
            if "face" in e:
                intent.face = np.array(e["face"], float)
            if "height" in e:
                intent.height = e["height"]
        return False                                  # 종료 요청 없음


def _rot(v, ang):
    c, s = np.cos(ang), np.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


class KeyboardInput:
    """상류 keyboard.md 바인딩 부분집합. 모멘텀 없음(키 = 즉시 설정, R = 정지)."""
    HELP = "(플래너는 idle 로 이미 가동) 1/2/3/4 = 현재 세트의 1~4번 모드 (기본 세트: idle/slowWalk/walk/run) | N/P 모션세트 | W/S 전후 | A/D 헤딩±15° | ,/. 스트레이프 | Q/E 제자리회전±30° | 9/0 속도 | -/= 높이 | R 정지 | O 종료"

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.set = 0
        print(self.HELP, flush=True)

    def close(self):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll(self, intent, now):
        while select.select([sys.stdin], [], [], 0)[0]:
            k = sys.stdin.read(1).lower()
            if k in ("", "o"):                     # "" = EOF
                return True
            elif k in "12345678":
                modes = MODE_SETS[self.set]
                if int(k) - 1 < len(modes):
                    intent.mode = modes[int(k) - 1]
            elif k == "n":
                self.set = (self.set + 1) % len(MODE_SETS); intent.mode = MODE_SETS[self.set][0]
            elif k == "p":
                self.set = (self.set - 1) % len(MODE_SETS); intent.mode = MODE_SETS[self.set][0]
            elif k == "w":
                intent.move = intent.face.copy()
            elif k == "s":
                intent.move = -intent.face
            elif k == ",":
                intent.move = _rot(intent.face, np.pi / 2)
            elif k == ".":
                intent.move = _rot(intent.face, -np.pi / 2)
            elif k in "ad":
                ang = np.pi / 12 * (1 if k == "a" else -1)
                intent.face = _rot(intent.face, ang); intent.move = _rot(intent.move, ang)
            elif k in "qe":
                intent.face = _rot(intent.face, np.pi / 6 * (1 if k == "q" else -1))
            elif k in "90":
                base = intent.speed if intent.speed else 0.5
                intent.speed = float(np.clip(base + (0.1 if k == "0" else -0.1), 0.1, 3.0))
            elif k in "-=":
                base = intent.height if intent.height else 0.89
                intent.height = float(np.clip(base + (0.05 if k == "=" else -0.05), 0.2, 1.0))
            elif k == "r":
                intent.move = np.zeros(2); intent.speed = None
            print(f"\rmode={intent.mode} speed={intent.speed} move={np.round(intent.move,2)} face={np.round(intent.face,2)} h={intent.height}   ", end="", flush=True)
        return False


# ------------------------------------------------------------------ 출력
class RedisSink:
    def __init__(self, host="127.0.0.1", port=6379):
        from general_motion_retargeting.utils.twist2_redis import Twist2RedisPublisher
        self.pub = Twist2RedisPublisher(host, "robros_igris_c_v2", port)
        self.empty = np.zeros(0)

    def push(self, payload, dof31, igris_qpos):
        qpos = np.concatenate([np.asarray(payload["root_pos"][0]), np.asarray(payload["root_rot"][0]), dof31])
        # 발행 시각 도장. redis_to_ros2_bridge 는 |t_action - record_meta.unix_ms| > stale_warn_ms(150) 면
        # 프레임을 버리고 keybody_pos_local 이 빈 기본 프레임으로 대체한다 → 정책 action_hz=0.
        payload["record_meta"]["unix_ms"] = int(time.time() * 1000)
        self.pub.publish_action(self.empty, self.empty, self.empty, None, qpos=qpos, retarget_frame=payload)


class AuditSink:
    def __init__(self, out=None):
        self.out, self.qpos, self.cmd = out, [], []

    def push(self, payload, dof31, igris_qpos):
        self.qpos.append(igris_qpos)

    def note_intent(self, intent):
        self.cmd.append((intent.mode, "default" if intent.speed is None else str(float(intent.speed))))

    def stats(self, seam_steps=()):
        q = np.asarray(self.qpos)
        model, data = mc.load_model()
        lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
        dof = q[:, 7:]
        exc = np.maximum(lo - dof, dof - hi)                 # 한계 초과량 (음수 = 여유)
        viol = int((exc > LIMIT_TOL).sum())
        h = mc.foot_heights(model, data, mc.foot_support(model), q[:, :3], q[:, 3:7], dof)
        v = np.linalg.norm(np.diff(q[:, :2], axis=0), axis=1) * FPS
        jerk = np.abs(np.diff(dof, n=3, axis=0)) * FPS ** 3
        st = {"frames": int(len(q)), "limit_violations": viol, "foot_min_z_m": float(h.min()),
              "foot_stance_float_m": float(np.percentile(h.min(1), 10)),
              "jerk_p95": float(np.percentile(jerk, 95)) if len(jerk) else 0.0,
              "seam_step_max_rad": float(max(seam_steps)) if len(seam_steps) else 0.0,
              "limit_excursion_max_rad": float(exc.max().clip(0.0)),
              "realized_speed": {}}
        last, prev = {}, None                                # 의도별 마지막 연속 구간 [시작, 끝]
        for i, c in enumerate(self.cmd):
            if c != prev:
                last[c] = [i, i]
            last[c][1] = i
            prev = c
        for (mode, spd), (a, b) in sorted(last.items()):
            seg = v[a + int(FPS) if b - a > int(FPS) else a: b]        # 1 s 정착 구간 제외
            st["realized_speed"][f"mode{mode}_cmd{spd}"] = float(seg.mean()) if len(seg) else 0.0
        if self.out:
            np.savez(self.out, qpos=q, **{k: v for k, v in st.items() if not isinstance(v, dict)})
        return st


class ReferenceSink:
    """S2 SONIC 트래커용: 코덱 디코드 이전의 **원본 G1 qpos**(36 = root7 + 29 MuJoCo/G1_MJ29 관절)를
    redis 키 `sonic_g1_reference` 로 발행. Twist2 페이로드(action_retarget_frame_*) 와 별개 채널이며
    하류(sonic_tracker)가 G1→IGRIS 매핑을 직접 한다. record_meta.unix_ms 로 신선도 스탬프(RedisSink 규약)."""

    KEY = "sonic_g1_reference"

    def __init__(self, host="127.0.0.1", port=6379, client=None):
        if client is None:                                  # 테스트는 fake client 주입(라이브 redis 불필요)
            from general_motion_retargeting.utils.twist2_redis import make_redis_client
            client = make_redis_client(host, port)
        self.client = client

    def push_reference(self, qpos36):
        frame = {
            "qpos": np.asarray(qpos36, dtype=float).tolist(),   # [root_pos3, root_quat4(wxyz), q29(G1_MJ29)]
            "fps": FPS,
            "joint_order": "G1_MJ29",
            "record_meta": {"source": "planner_bridge_reference", "unix_ms": int(time.time() * 1000)},
        }
        self.client.set(self.KEY, json.dumps(frame, separators=(",", ":")))


# ------------------------------------------------------------------ 브릿지
class Bridge:
    def __init__(self, planner, phi, pb, sink, inp, seed=1234):
        self.planner, self.phi, self.pb, self.sink, self.inp, self.seed = planner, phi, pb, sink, inp, seed
        self.intent = Intent()
        self.last_key, self.last_replan_t = None, -1e9
        self.serving = None

    # --- 의도 → G1 입력
    def intent_g1(self):
        it = self.intent
        s, za, zb = (self.phi.s, self.phi.za, self.phi.zb) if self.phi else (1.0, 1.0, 0.0)
        return {"mode": it.mode, "target_vel": -1.0 if it.speed is None else it.speed / s,
                "move": list(map(float, it.move)), "face": list(map(float, it.face)),
                "height": ((it.height - zb) / za) if it.height is not None else -1.0}

    def need_replan(self, now):
        key = self.intent.key()
        if key != self.last_key:
            return True
        if self.intent.mode in STATIC_MODES or self.intent.speed == 0 or not np.any(self.intent.move):
            return False
        return now - self.last_replan_t >= replan_interval(self.intent.mode)

    def mark_replanned(self, now):
        self.last_key, self.last_replan_t = self.intent.key(), now

    def initialize(self):
        ctx = np.tile(np.concatenate([[0, 0, ic.G1_DEFAULT_HEIGHT, 1, 0, 0, 0], ic.G1_DEFAULT29]), (4, 1))
        self.intent = Intent(mode=0)
        self.serving = Serving(resample_30_to_50(self.planner.plan(ctx, self.intent_g1(), self.seed)))
        self.mark_replanned(0.0)

    def step_planner(self, now):
        if not self.need_replan(now):
            return
        gen = self.serving.current + LOOK_AHEAD
        ctx = self.serving.context(gen)
        q30 = self.planner.plan(ctx, self.intent_g1(), self.seed)
        self.serving.apply(resample_30_to_50(q30), gen)
        self.mark_replanned(now)

    def step_serve(self):
        f = self.serving.frame()
        if hasattr(self.sink, "push_reference"):        # --reference: 코덱 디코드 없이 원본 G1 qpos 발행
            self.sink.push_reference(f)
            return
        root_pos, root_quat, joints = self.phi.decode(f)
        payload, dof31, qpos = self.pb.frame(root_pos[0], root_quat[0], {k: v[0] for k, v in joints.items()})
        self.sink.push(payload, dof31, qpos)
        if hasattr(self.sink, "note_intent"):
            self.sink.note_intent(self.intent)

    def run(self, realtime=True, duration=None):
        self.initialize()
        n, per = 0, int(FPS / PLANNER_HZ)
        t0 = time.perf_counter()
        stop = False
        while not stop and (duration is None or n / FPS < duration):
            now = n / FPS
            stop = self.inp.poll(self.intent, now) if self.inp else False
            if n % per == 0:
                self.step_planner(now)
            self.step_serve()
            n += 1
            if realtime:
                time.sleep(max(0.0, t0 + n / FPS - time.perf_counter()))
        return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="planner_sonic.onnx (생략 시 StubPlanner)")
    ap.add_argument("--phi", required=True)
    ap.add_argument("--provider", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--script"); ap.add_argument("--keyboard", action="store_true")
    ap.add_argument("--audit", action="store_true"); ap.add_argument("--out", default=None)
    ap.add_argument("--reference", action="store_true", help="코덱 디코드 없이 원본 G1 qpos(50Hz)를 redis 키 sonic_g1_reference 로 발행 (S2 SONIC 트래커)")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--start-file", default=None, help="--script 의 t=0 을 이 파일이 생기는 순간으로 (그 전엔 idle 발행)")
    ap.add_argument("--redis_host", default="127.0.0.1"); ap.add_argument("--redis_port", type=int, default=6379)
    a = ap.parse_args()
    planner = OnnxPlanner(a.model, a.provider) if a.model else StubPlanner()
    inp = None if a.keyboard else (ScriptInput(a.script, a.start_file) if a.script else None)
    sink = (ReferenceSink(a.redis_host, a.redis_port) if a.reference
            else AuditSink(a.out) if a.audit else RedisSink(a.redis_host, a.redis_port))
    pb = None if a.reference else ic.PayloadBuilder()          # 레퍼런스 모드는 코덱/FK 불필요
    b = Bridge(planner, ic.PhiDecoder(a.phi), pb, sink, inp)
    try:
        if a.keyboard:
            b.inp = inp = KeyboardInput()          # 터미널 raw 모드는 finally 로만 복구된다
        b.run(realtime=not a.audit, duration=a.duration)
    finally:
        if inp is not None and a.keyboard:
            inp.close()
    if a.audit:
        print(json.dumps(sink.stats(b.serving.seam_steps), indent=1))


if __name__ == "__main__":
    main()
