import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import planner_bridge as pb  # noqa: E402
import igris_codec as ic  # noqa: E402


def _phi(tmp_path):
    joints = [{"igris": ig, "g1": g1, "a": -1.0 if "Elbow" in ig else 1.0, "b": 0.0, "r2": 1.0} for ig, g1 in ic.G1_OF.items()]
    p = tmp_path / "phi.json"
    p.write_text(json.dumps({"format": "igris-phi/1", "joints": joints, "waist": {"mode": "closed_form"},
                             "root": {"s_xy": 1.1, "z_a": 1.127, "z_b": 0.0}, "intent": {"v_scale": 1 / 1.1}}))
    return p


class FakeRedis:
    """redis.set() 캡처 — 라이브 redis 없이 on-wire 페이로드를 그대로 검사."""

    def __init__(self):
        self.sets = []

    def set(self, key, val):
        self.sets.append((key, val))


def test_reference_publishes_raw_g1_qpos(tmp_path):
    script = tmp_path / "s.json"
    script.write_text(json.dumps([{"t": 0, "mode": 0},
                                  {"t": 0.5, "mode": 2, "speed": 0.5, "move": [1, 0]}]))
    fake = FakeRedis()
    sink = pb.ReferenceSink(client=fake)
    b = pb.Bridge(pb.StubPlanner(), ic.PhiDecoder(_phi(tmp_path)), None, sink, pb.ScriptInput(script))
    b.run(realtime=False, duration=3.0)

    assert len(fake.sets) == 150                              # 50 Hz × 3 s
    assert {k for k, _ in fake.sets} == {"sonic_g1_reference"}  # 전용 키 하나만

    frames = [json.loads(v) for _, v in fake.sets]
    for fr in frames:
        assert len(fr["qpos"]) == 36                          # root7 + 관절29
        assert fr["fps"] == 50.0 and fr["joint_order"] == "G1_MJ29"
        assert fr["record_meta"]["unix_ms"] > 0               # 신선도 스탬프

    q = np.array(frames[-1]["qpos"])
    assert np.allclose(q[3:7], [1, 0, 0, 0])                  # root quat wxyz 항등
    assert abs(q[2] - ic.G1_DEFAULT_HEIGHT) < 1e-9            # 코덱 za/zb 미적용 → 원본 G1 높이
    # 29 관절 = 원본 G1 기본자세(MuJoCo/G1_MJ29 순서). 코덱 디코드였다면 아핀 a*q+b 로 값이 달라진다.
    assert np.allclose(q[7:], ic.G1_DEFAULT29)
    assert np.array(frames[-1]["qpos"])[0] > np.array(frames[0]["qpos"])[0]  # move=[1,0] → +x 전진
