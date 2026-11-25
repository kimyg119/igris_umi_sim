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


def test_resample_and_blend_continuity():
    q30 = np.zeros((48, 36)); q30[:, 3] = 1.0; q30[:, 0] = np.arange(48) / 30 * 0.5     # 0.5 m/s 전진
    q50 = pb.resample_30_to_50(q30)
    assert q50.shape[0] == int(np.floor(47 / 30 * 50 + 1e-9)) + 1 and abs(q50[-1, 0] - 0.5 * 78 / 50) < 1e-9
    old = q50.copy(); new = q50.copy() + np.array([0.3] + [0] * 35)                    # 30 cm 점프하는 새 플랜
    out = pb.blend(old, cur=10, new=new, offset=2, nb=8)
    step = np.abs(np.diff(out[:, 0]))
    assert step.max() < 0.3 / 8 + 0.02 and abs(out[-1, 0] - new[-1, 0]) < 1e-9


def test_replan_rules():
    assert pb.replan_interval(3) == 0.1 and pb.replan_interval(8) == 0.2 and pb.replan_interval(2) == 1.0
    b = pb.Bridge(pb.StubPlanner(), None, None, None, None)
    b.intent = pb.Intent(mode=0); b.mark_replanned(now=0.0)
    assert b.need_replan(now=5.0) is False                       # 정적 모드: 타이머 없음
    b.intent = pb.Intent(mode=2, speed=0.5, move=np.array([1.0, 0.0]))
    assert b.need_replan(now=5.0) is True                        # 의도 변경
    b.mark_replanned(now=5.0)
    assert b.need_replan(now=5.5) is False and b.need_replan(now=6.05) is True   # 1.0 s 주기


def test_audit_run(tmp_path):
    phi = ic.PhiDecoder(_phi(tmp_path))
    script = tmp_path / "s.json"
    script.write_text(json.dumps([{"t": 0, "mode": 0}, {"t": 1.0, "mode": 2, "speed": 0.5, "move": [1, 0]}]))
    sink = pb.AuditSink()
    b = pb.Bridge(pb.StubPlanner(), phi, ic.PayloadBuilder(), sink, pb.ScriptInput(script))
    b.run(realtime=False, duration=3.0)
    st = sink.stats(b.serving.seam_steps)
    assert st["frames"] == 150 and st["limit_violations"] == 0 and st["seam_step_max_rad"] < 0.2
    assert st["foot_min_z_m"] > -0.02 and "realized_speed" in st


def test_seam_metric_measures_the_blend_ramp():
    q = np.zeros((80, 36)); q[:, 3] = 1.0
    s = pb.Serving(q.copy())
    for _ in range(10):
        s.frame()
    new = q.copy(); new[:, 7] = 0.3                                # 0.3 rad 점프하는 새 플랜
    s.apply(new, gen_frame=12)                                     # offset = 12 - 10 = 2
    assert abs(s.seam_steps[-1] - 0.3 / 8) < 1e-6                  # 8 프레임 램프 = 스텝당 0.3/8


def test_default_speed_replans_but_zero_speed_does_not():
    b = pb.Bridge(pb.StubPlanner(), None, None, None, None)
    b.intent = pb.Intent(mode=2, speed=None, move=np.array([1.0, 0.0])); b.mark_replanned(now=0.0)
    assert b.need_replan(now=0.5) is False and b.need_replan(now=1.05) is True    # None = 모드 기본 속도
    b.intent = pb.Intent(mode=2, speed=0.0, move=np.array([1.0, 0.0])); b.mark_replanned(now=0.0)
    assert b.need_replan(now=5.0) is False                                        # 0 = 명시적 정지


def test_intent_g1_speed_and_height_mapping(tmp_path):
    b = pb.Bridge(pb.StubPlanner(), ic.PhiDecoder(_phi(tmp_path)), None, None, None)
    b.intent = pb.Intent(mode=2, speed=None)
    assert b.intent_g1()["target_vel"] == -1.0 and b.intent_g1()["height"] == -1.0
    b.intent = pb.Intent(mode=2, speed=0.0, height=0.9)
    g = b.intent_g1()
    assert g["target_vel"] == 0.0 and abs(g["height"] - (0.9 - b.phi.zb) / b.phi.za) < 1e-12
    b.intent = pb.Intent(mode=2, speed=0.5)
    assert abs(b.intent_g1()["target_vel"] - 0.5 / b.phi.s) < 1e-12


def test_realized_speed_uses_last_contiguous_run(tmp_path):
    script = tmp_path / "s.json"
    script.write_text(json.dumps([{"t": 0, "mode": 0},
                                  {"t": 1.0, "mode": 2, "speed": 0.5, "move": [1, 0]},
                                  {"t": 3.0, "mode": 0, "speed": None, "move": [0, 0]}]))
    sink = pb.AuditSink()
    b = pb.Bridge(pb.StubPlanner(), ic.PhiDecoder(_phi(tmp_path)), ic.PayloadBuilder(), sink, pb.ScriptInput(script))
    b.run(realtime=False, duration=5.0)
    rs = sink.stats(b.serving.seam_steps)["realized_speed"]
    assert rs["mode0_cmddefault"] < 1e-6 and rs["mode2_cmd0.5"] > 0.4    # idle 은 마지막 구간만 (걷기 제외)
