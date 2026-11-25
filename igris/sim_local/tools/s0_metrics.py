#!/usr/bin/env python3
"""state_recorder.py npz 에서 S0 게이트 지표: 생존 / 구간별 실현 속도 / 낙상 시각.

    python3 tools/s0_metrics.py --npz logs/s0_walk.npz \
        --seg 10:40:slowWalk@0.3 40:70:walk@0.5 70:85:turn 85:100:idle
구간 속도는 각 구간 마지막 2 s 의 xy 변위율(정착 후 값). --t0 는 npz 시각축 대비
스크립트 t=0 의 오프셋.
"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--t0", type=float, default=0.0)
ap.add_argument("--stand", type=float, default=0.7, help="생존 판정 base z 하한")
ap.add_argument("--seg", nargs="*", default=[], help="시작:끝:라벨 (스크립트 시간축)")
ap.add_argument("--tail", type=float, default=2.0, help="구간 속도 측정 구간 길이(s)")
a = ap.parse_args()

d = np.load(a.npz)
t = d["t"] - a.t0
xy, z = d["base"][:, :2], d["base"][:, 2]

below = np.flatnonzero(z < a.stand)
fall = float(t[below[0]]) if len(below) else None
print(f"샘플 {len(t)}개  t={t[0]:.1f}~{t[-1]:.1f}s")
print(f"base z: 최소 {z.min():.3f} m (@t={t[np.argmin(z)]:.1f}s), 최종 {z[-1]:.3f} m")
print(f"생존(z>{a.stand}): {'OK' if fall is None else f'FAIL — t={fall:.1f}s 에서 첫 이탈'}")
print(f"xy 총 변위 {np.linalg.norm(xy[-1] - xy[0]):.3f} m, 경로장 {np.linalg.norm(np.diff(xy, axis=0), axis=1).sum():.3f} m")


def speed(t0, t1):
    m = (t >= max(t0, t1 - a.tail)) & (t <= t1)
    if m.sum() < 2:
        return float("nan")
    tt, p = t[m], xy[m]
    return float(np.linalg.norm(p[-1] - p[0]) / (tt[-1] - tt[0]))


for s in a.seg:
    t0, t1, label = s.split(":")
    t0, t1 = float(t0), float(t1)
    m = (t >= t0) & (t <= t1)
    if not m.any():
        print(f"  [{label}] {t0:.0f}~{t1:.0f}s: 샘플 없음")
        continue
    print(f"  [{label}] {t0:.0f}~{t1:.0f}s: 실현속도 {speed(t0, t1):.3f} m/s "
          f"(마지막 {a.tail:.0f}s), 구간평균 {np.linalg.norm(xy[m][-1]-xy[m][0])/(t[m][-1]-t[m][0]):.3f} m/s, "
          f"z 최소 {z[m].min():.3f} m")
