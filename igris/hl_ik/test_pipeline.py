"""Regression contracts at the neural-prior / IK / command boundary."""
import numpy as np
import pytest
from pathlib import Path
from .common import OUTPUT
from .solver import HLIK
from .pipeline import PipelineIK


class ConflictingPrior:
    def predict(self, history, goal):
        elbow = history[-1, :, 7:].copy()
        elbow[:, 0] += .15
        return elbow


def seed():
    with np.load(OUTPUT/'box_plans/full_guarded/ik.npz') as z:
        return z['q'][0].copy()


def target(solver, q):
    solver.reset(q)
    poses = solver.states()[:, :7].copy()
    poses[:, :3] += solver.shoulder
    return poses


def test_conflicting_prior_preserves_stationary_hand_target():
    q = seed()
    old = HLIK(ConflictingPrior(), collision=False)
    goals = target(old, q)
    assert old.solve(goals, q).position_error_m > .01  # Reproduces the old failure.
    solver = PipelineIK(ConflictingPrior(), collision=False)
    target(solver, q)
    result = solver.solve(goals, q)
    assert result.accepted and result.position_error_m < .0005
    np.testing.assert_array_equal(result.q[:15], q[:15])


def test_command_acceleration_and_jerk_on_target_reversal():
    q = seed(); solver = PipelineIK(collision=False,acceleration_limit=6.)
    home = target(solver, q); commands = [q.copy(), q.copy(), q.copy()]
    for i in range(45):
        goal = home.copy(); goal[:, 0] += .07 if i < 20 else -.07
        result = solver.solve(goal, q)
        assert result.accepted, result.reason
        q = result.q; commands.append(q.copy())
    q = np.array(commands)[:, 15:29]
    assert abs(np.diff(q, axis=0)*30).max() <= 3.+1e-6
    assert abs(np.diff(q, n=2, axis=0)*900).max() <= 6.+1e-5
    # 6 rad/s² at 30 Hz implies <=360 rad/s³ discrete jerk; there is no
    # stronger default jerk constraint requiring a full stopping trajectory.
    assert abs(np.diff(q, n=3, axis=0)*27000).max() <= 360.+1e-4


def test_rejected_candidate_does_not_advance_history():
    q=seed(); solver=PipelineIK(collision=False); goals=target(solver,q)
    previous=np.array(solver.history).copy()
    q[15]=solver.limits[0,1]+.1
    result=solver.solve(goals,q)
    assert not result.accepted and result.reason=='invalid_seed_limits'
    np.testing.assert_array_equal(np.array(solver.history),previous)
    np.testing.assert_array_equal(result.q,q)


def test_first_invalid_seed_does_not_initialize_history():
    q=seed();reference=HLIK(collision=False);goals=target(reference,q)
    solver=PipelineIK(collision=False);bad=q.copy();bad[15]=solver.limits[0,1]+.1
    assert not solver.solve(goals,bad).accepted
    assert not solver.history and solver._observed_arm is None
    result=solver.solve(goals,q)
    assert result.accepted and abs(solver.velocity).max()<1e-7


def test_observed_motion_seeds_rate_bounds_without_double_counting():
    q=seed();solver=PipelineIK(collision=False,acceleration_limit=6.);target(solver,q)
    for _ in range(4):q[15]+=.01;solver.observe(q)
    previous_velocity=solver.velocity.copy()
    assert np.isclose(previous_velocity[0],.3)
    goals=solver.states()[:,:7];goals[:,:3]+=solver.shoulder
    result=solver.solve(goals,q,record_history=False)
    assert result.accepted
    velocity=(result.q[15:29]-q[15:29])*30
    acceleration=(velocity-previous_velocity)*30
    assert abs(acceleration).max()<=6.+1e-5
    solver.observe(result.q)
    np.testing.assert_allclose(solver.velocity,velocity,atol=1e-10)
    np.testing.assert_allclose(solver.acceleration,acceleration,atol=1e-9)


def test_failed_prior_is_reported_and_ee_path_remains_available():
    class Invalid:
        def predict(self,history,goal): return np.full((2,7),np.nan)
    q=seed(); solver=PipelineIK(Invalid(),collision=False); goals=target(solver,q)
    result=solver.solve(goals,q)
    assert result.accepted and result.prior_status=='invalid_prediction'
    assert result.position_error_m<1e-7


def test_feasible_rank_deficient_qp_is_not_rejected_for_dual_roundoff():
    # Captured primary EE problem: constraints satisfied, but eps_abs=1e-8
    # exhausted 50 iterations at dual residual1.04e-8 and falsely rejected it.
    z=np.load(Path(__file__).parent/'fixtures/feasible_ee_qp.npz')
    solver=PipelineIK(collision=False)
    x,ok=solver._qp(z['H'],z['g'],z['C'],z['l'],z['u'],0)
    assert ok and solver._qp_fallbacks==1
    assert np.all(z['C']@x>=z['l']-2e-8)
    assert np.all(z['C']@x<=z['u']+2e-8)


def test_alternative_qp_still_requires_original_constraint_gate(monkeypatch):
    z=np.load(Path(__file__).parent/'fixtures/feasible_ee_qp.npz')
    monkeypatch.setattr('igris.hl_ik.pipeline.daqp.solve',
        lambda *args,**kwargs:(np.full(14,100.),0.,1,{}))
    solver=PipelineIK(collision=False)
    _,ok=solver._qp(z['H'],z['g'],z['C'],z['l'],z['u'],0)
    assert not ok


def test_clock_reader_rejects_partial_and_stale_updates(tmp_path):
    import struct
    from .native_clock import read_clock
    path=tmp_path/'clock.bin'
    path.write_bytes(struct.pack('Qddd',1,100.,10.,5.))
    with pytest.raises(RuntimeError,match='incomplete'): read_clock(path,now=100.)
    path.write_bytes(struct.pack('Qddd',2,100.,10.,5.))
    assert read_clock(path,now=100.01)['sim']==5.
    with pytest.raises(RuntimeError,match='stale'): read_clock(path,now=100.3)
    path.write_bytes(struct.pack('Qddd',2,100.,10.,float('nan')))
    with pytest.raises(RuntimeError,match='invalid'): read_clock(path,now=100.01)


def test_clock_uses_monotonic_age_after_payload_read(tmp_path,monkeypatch):
    import struct
    from .native_clock import read_clock
    path=tmp_path/'clock.bin';path.write_bytes(struct.pack('Qddd',2,999999.,10.,5.))
    monkeypatch.setattr('time.monotonic',lambda:10.01)
    monkeypatch.setattr('time.time',lambda:1.)
    assert read_clock(path)['sim']==5.
