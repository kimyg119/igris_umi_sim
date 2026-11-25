"""Opt-in feedback reaches the actual actor while original zero scope is retained."""
from dataclasses import replace
import numpy as np
import pytest

from test_htd_loaded import setup, snapshot


def goal_snapshot(policy, **kwargs):
    s = snapshot(policy, **kwargs)
    return replace(s, torso_pose=s.pelvis_pose.copy(), pose_native_time=s.native_time)


def test_tip_contact_relaxation_requires_explicit_goal_hold_scope():
    c, p, _ = setup(current_grasp_reference=True, goal_hold=True)
    s = replace(goal_snapshot(p), palm_tip_force_N=np.array([0., 2., 0., 2.]))
    c.request_handoff(s)
    assert c.goal_hold.metrics['goal_xy_world'] == [0., 0.]
    old, p, _ = setup(current_grasp_reference=True)
    with pytest.raises(RuntimeError, match='friction-qualified'):
        old.request_handoff(replace(snapshot(p), palm_tip_force_N=s.palm_tip_force_N))


@pytest.mark.parametrize('forces', [[0., .1, 0., 2.], [0., 2., 0., .1],
                                   [-1., 3., 0., 2.], [0., np.nan, 0., 2.]])
def test_combined_contact_still_requires_finite_nonnegative_per_hand_margin(forces):
    c, p, _ = setup(current_grasp_reference=True, goal_hold=True)
    with pytest.raises(RuntimeError, match='friction-qualified'):
        c.request_handoff(replace(goal_snapshot(p), palm_tip_force_N=np.array(forces)))
    assert c.state == 'prepared' and not c.goal_hold.metrics


def test_goal_hold_feedback_is_used_by_real_actor_once_per_actor_period():
    c, p, upper = setup(current_grasp_reference=True, goal_hold=True)
    c.request_handoff(goal_snapshot(p))
    s = goal_snapshot(p, mode=3)
    pose = s.pelvis_pose.copy()
    pose[3:] = [np.cos(.15), 0., 0., np.sin(.15)]
    s = replace(s, pelvis_pose=pose, torso_pose=pose.copy(), pose_native_time=1.0001,
                native_time=1.0001)
    calls = []
    original = p.step
    def record(*args):
        calls.append(np.array(args[4], copy=True))
        return original(*args)
    p.step = record
    assert c.advance_actor(s) is not None
    assert calls[0][2] < 0 and abs(calls[0][2]) <= .010001
    frame = c.compute_pd(s, upper)
    c.ack_published(frame, s)
    assert c.advance_actor(s) is None
    assert len(c.goal_hold.events) == 2 and len(calls) == 1
    assert c.contract['commanded_velocity'] == 'WorldXY goal/heading feedback'


def test_feedback_scope_requires_current_grasp_and_complete_pose():
    with pytest.raises(ValueError, match='current-grasp'):
        setup(goal_hold=True)
    c, p, _ = setup(current_grasp_reference=True, goal_hold=True)
    with pytest.raises((ValueError, TypeError)):
        c.request_handoff(snapshot(p))
    assert c.state == 'prepared'


def test_only_explicit_goal_hold_accepts_15mm_clearance_with_existing_support_and_dwell_guards():
    with pytest.raises(ValueError, match='clearance'):
        setup(current_grasp_reference=True, clearance_minimum_rise_m=.015)
    c, p, _ = setup(current_grasp_reference=True, goal_hold=True, clearance_minimum_rise_m=.015)
    s = replace(goal_snapshot(p), clearance_rise_m=.017)
    for change in (dict(clearance_rise_m=.0149), dict(clearance_ready_s=.49),
                   dict(source_support_z_N=.3), dict(payload_speed_m_s=.021)):
        with pytest.raises(RuntimeError, match='clearance'):
            c.request_handoff(replace(s, **change))
    c.request_handoff(s)
    assert c.current_grasp_evidence['clearance_minimum_rise_m']==.015


def test_50mm_raise_is_only_available_in_explicit_goal_hold_acquisition():
    from htd_loaded_setup import validate_options
    args=dict(live_model='actual',walk_to_rolltainer=True,inplace_transfer=False,force_grasp=True,
              relaxed_grasp=True,payload_mass=.1,mu=4.,carry_clearance_lift=.05,
              simulation_rate_limit=.5,simulation_rate_start_phase=90.)
    with pytest.raises(ValueError):
        validate_options(**args)
    validate_options(**args,goal_hold=True)
    with pytest.raises(ValueError):
        validate_options(**dict(args,carry_clearance_lift=.06),goal_hold=True)


def test_contact_grace_reaches_core_only_with_explicit_goal_scope():
    with pytest.raises(ValueError,match='goal hold'):
        setup(current_grasp_reference=True,contact_grace=True)
    c,p,_=setup(current_grasp_reference=True,goal_hold=True,contact_grace=True)
    def measured(t,force):
        return replace(goal_snapshot(p,sim=t),palm_tip_force_N=np.array(force),force_native_time=t,
            payload_pose=np.array([0.,0.,.9,1.,0.,0.,0.]),
            hand_poses=np.array([[0.,.1,.85,1.,0.,0.,0.],[0.,-.1,.85,1.,0.,0.,0.]]))
    c.request_handoff(measured(1.,[2,0,2,0]))
    c._check(measured(1.01,[0,0,2,0]))
    assert c.grasp_retention.metrics['contact_grace_active']
    c._check(measured(1.09,[2,0,2,0]))
    assert not c.grasp_retention.metrics['contact_grace_active']
    c._check(measured(1.10,[0,0,2,0]))
    with pytest.raises(RuntimeError,match='100ms'):
        c._check(measured(1.201,[0,0,2,0]))
    assert c.epoch==1 and c.policy._step_index==0
