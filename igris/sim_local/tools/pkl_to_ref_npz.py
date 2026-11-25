#!/usr/bin/env python3
"""텔레옵 녹화 pkl 을 시뮬 녹화(state_recorder.py npz)와 같은 타임라인의 레퍼런스 npz 로 변환.

render_replay.py 로 같은 viz 모델·카메라로 렌더해 시뮬 영상과 hstack 비교하기 위한 것.
- 시각: 시뮬 npz 의 t 를 그대로 사용. t0(재생 시작) 이전은 첫 프레임, 이후는 최근접 프레임,
  끝나면 마지막 프레임 홀드 (gmr_pkl_pub.py --start_file/--no-loop 재생과 동일).
- 정렬: 초기 yaw/xy 를 시뮬 첫 샘플에 맞춤 (정책은 루트 상대 추종이라 절대 방향 무의미). z 는 유지.
- 실제 프레임률은 record_meta 로 복원 (fps 키는 라벨).
"""
import argparse
import pickle

import numpy as np


def yaw_of(q):  # wxyz
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--sim_npz", required=True, help="state_recorder.py 산출물 (타임라인·초기 자세 기준)")
    ap.add_argument("--t0", type=float, required=True, help="녹화 시작 기준 재생 시작 시각(s) — 퍼블리셔 '트리거 감지' 로그 시각 − 녹화 시작 로그 시각")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sim = np.load(args.sim_npz)
    t, names, sb = sim["t"], sim["joint_names"], sim["base"]
    p = pickle.load(open(args.pkl, "rb"))
    m = p["record_meta"]
    n = m["num_frames"]
    fps = (n - 1) / (m["record_end_unix"] - m["record_start_unix"])
    rp, rq, dof = (np.asarray(p[k], dtype=np.float64) for k in ("root_pos", "root_rot", "dof_pos"))

    dyaw = yaw_of(sb[0, 3:7]) - yaw_of(rq[0])
    qz = np.array([np.cos(dyaw / 2), 0, 0, np.sin(dyaw / 2)])
    R = np.array([[np.cos(dyaw), -np.sin(dyaw)], [np.sin(dyaw), np.cos(dyaw)]])
    xy = (rp[:, :2] - rp[0, :2]) @ R.T + sb[0, :2]
    rp_al = np.column_stack([xy, rp[:, 2]])
    rq_al = np.array([qmul(qz, q) for q in rq])

    k = np.clip(np.rint((t - args.t0) * fps).astype(int), 0, n - 1)
    np.savez(args.out, t=t, base=np.column_stack([rp_al[k], rq_al[k]]), q=dof[k], joint_names=names)
    print(f"{args.out}: {len(t)} 샘플, t0={args.t0:.2f}s, ref fps={fps:.2f}, dyaw={np.degrees(dyaw):+.1f}°")


if __name__ == "__main__":
    main()
