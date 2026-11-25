#!/usr/bin/env python3
"""글러브 OSC -> IGRIS 관절각.

받는다  /{left,right}/joint_angles/get  <타입> <이름> <값> ... (21쌍)
보낸다  /device/status/get "4"          1초 하트비트. 이걸 안 보내면 글러브가
                                        joint_angles 를 안 쏜다.

값은 받은 순서대로 넣는다. 글러브 순서와 모델 입력 순서가 같다:
index, middle, pinky, ring 각 (mcp굴곡, mcp외전, pip, dip), 그 뒤 엄지 5개.

    python igris_osc.py --onnx uni_igris_lr.onnx --tables uni_tables.npz
"""
from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import onnxruntime as ort
from pythonosc import dispatcher as osc_dispatcher
from pythonosc import osc_server
from pythonosc.udp_client import SimpleUDPClient

SIDES = ("left", "right")


class IgrisRetarget:
    """ONNX 하나로 좌우 둘 다."""

    def __init__(self, onnx: str, tables: str, hand: str = "igris",
                 threads: int = 1):
        t = np.load(tables, allow_pickle=False)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx, so,
                                        providers=["CPUExecutionProvider"])
        gin = {i.name for i in self.sess.get_inputs()}

        self.n_in = len(t["human:joints"])
        self.q_in = {s: np.zeros((1, self.n_in), np.float32) for s in SIDES}
        self.names, self.lo, self.hi, self.feed = {}, {}, {}, {}
        for k, side in enumerate(SIDES):
            key = f"{hand}:{side}"
            if f"{key}:lo" not in t:
                continue
            self.names[side] = [str(n) for n in t[f"{key}:joints"]]
            self.lo[side] = t[f"{key}:lo"].astype(np.float32)
            self.hi[side] = t[f"{key}:hi"].astype(np.float32)
            # 넣을 값은 그래프에 물어서 정한다 (baked / 좌우결합 / uni_all)
            f = {}
            for n in gin - {"q_human"}:
                f[n] = (np.array([k], np.int64) if n == "side"
                        else t[f"{key}:{n}"].astype(np.float32))
            self.feed[side] = f
        if not self.names:
            raise SystemExit(f"표에 {hand} 가 없다: {sorted(t.files)[:6]} ...")

    def compute(self, side: str, vals) -> np.ndarray:
        q = self.q_in[side]
        q[:] = 0.0
        q[0, :len(vals)] = vals[:self.n_in]
        out = self.sess.run(None, {"q_human": q, **self.feed[side]})[0][0]
        # 그래프에 클램프가 박혀 있으면 무동작. 안 박힌 판용 안전망이다.
        return np.clip(out, self.lo[side], self.hi[side])


def parse_values(args) -> list:
    """(이름, 값) 짝의 값만 순서대로. 앞의 타입 토큰은 건너뛴다."""
    out, i = [], 0
    while i + 1 < len(args):
        if isinstance(args[i], str) and isinstance(args[i + 1], (int, float)):
            out.append(float(args[i + 1]))
            i += 2
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", default="uni_all.onnx")
    ap.add_argument("--tables", default="uni_tables.npz")
    ap.add_argument("--hand", default="igris")
    ap.add_argument("--port", type=int, default=4040, help="글러브 -> 여기")
    ap.add_argument("--listen", default="0.0.0.0")
    ap.add_argument("--glove-ip", default="127.0.0.1")
    ap.add_argument("--glove-port", type=int, default=4042,
                    help="하트비트 보낼 곳")
    ap.add_argument("--sides", nargs="+", default=list(SIDES))
    ap.add_argument("--print-hz", type=float, default=2.0, help="0 이면 조용히")
    args = ap.parse_args()

    net = IgrisRetarget(args.onnx, args.tables, args.hand)
    lock = threading.Lock()
    last = {s: None for s in SIDES}

    def on_angles(address, *a):
        side = address.strip("/").split("/")[0]
        if side not in net.names:
            return
        vals = parse_values(a)
        if len(vals) != net.n_in:      # 개수가 다르면 순서를 못 믿는다
            return
        q = net.compute(side, vals)
        with lock:
            last[side] = (q, time.monotonic())

    disp = osc_dispatcher.Dispatcher()
    for s in SIDES:
        disp.map(f"/{s}/joint_angles/get", on_angles)
    osc_server.ThreadingOSCUDPServer.allow_reuse_address = True
    srv = osc_server.ThreadingOSCUDPServer((args.listen, args.port), disp)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    hb = SimpleUDPClient(args.glove_ip, args.glove_port)
    try:
        while True:
            hb.send_message("/device/status/get", "4")
            time.sleep(1.0)
            if args.print_hz <= 0:
                continue
            now = time.monotonic()
            with lock:
                snap = {s: last[s] for s in args.sides}
            for s in args.sides:
                if snap[s] is None or now - snap[s][1] > 1.0:
                    print(f"[{s:5s}] 수신 없음", flush=True)
                else:
                    print(f"[{s:5s}] q(도) " + " ".join(
                        f"{v:+7.2f}" for v in np.degrees(snap[s][0])),
                        flush=True)
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
