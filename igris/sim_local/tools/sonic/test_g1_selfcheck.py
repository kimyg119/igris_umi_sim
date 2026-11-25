"""Reproducible short G0 gate: 3 s standing check, assert no-fall + z held.

Run: conda run -n gmr python sim_local/tools/sonic/test_g1_selfcheck.py
(kept short so the gate is cheap to re-run; the full 10-12 s gate lives in g1_selfcheck.main).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_selfcheck import run_gate


def test_g0_standing_no_fall():
    r = run_gate(duration_s=3.0, verbose=True)
    assert r["ticks_run"] == r["ticks_expected"], f"stopped early: {r['ticks_run']}/{r['ticks_expected']}"
    assert r["no_fall"], f"G1 fell: z_min={r['z_min']:.3f}"
    assert r["z_held_within_0.03"], f"base z drifted: max_dev={r['z_max_dev_from_init']:.3f}"
    assert r["passed"]


if __name__ == "__main__":
    test_g0_standing_no_fall()
    print("test_g1_selfcheck OK")
