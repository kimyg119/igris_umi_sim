"""Opt-in native105 geometry route, actual pose qualification, and bounded commands."""
from dataclasses import asdict, replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from htd_goal_hold import world_xy
from htd_navigation import navigation_config
from htd_navigation_feedback import NavigationFeedback
from htd_reviewed_approach import ReviewedApproachRoute, ReviewedApproachError, ReviewedApproachController


def route(angle=0.):
    r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    return ReviewedApproachRoute([.2,.3],np.array([.2,.3])+r@[.85,-.04],angle,
                                 native_time=1.,pose_native_time=1.)


def update(r,t,local,heading=None,stamp=None):
    return r.advance(r.start+r.rotation@local,heading=r.heading if heading is None else heading,
                     native_time=t,pose_native_time=t if stamp is None else stamp)


def aligned(r,t=1.08):
    update(r,1.02,[0.,.07]);update(r,1.04,[.65,.085]);update(r,1.06,[.65,-.04])
    for tt in np.arange(t,t+.301,.02):update(r,float(tt),[.65,-.04])
    assert r.final


def test_geometry_review_waypoints_rotate_with_captured_heading():
    r=route(.7)
    np.testing.assert_allclose(r.rotation.T@(r.waypoints[1]-r.start),[.65,.085],atol=1e-12)
    np.testing.assert_allclose(r.rotation.T@(r.waypoints[2]-r.start),[.65,-.04],atol=1e-12)
    assert not r.final and r.index==0
    assert r.evidence()['geometry_review']['dynamic_validation']=='pending'


def test_source_forward_cannot_skip_to_alignment_at_legacy_x04():
    r=route();assert update(r,1.02,[0.,.07]);assert r.index==1
    assert not update(r,1.04,[.4,.085]);assert r.index==1
    assert not update(r,1.06,[.65,.064]);assert r.index==1
    recovery=r.clearance_recovery_goal(r.start+r.rotation@[.65,.064]);assert recovery is not None
    assert update(r,1.08,[.65,.07]);assert r.index==2
    assert r.clearance_recovery_goal(r.start+r.rotation@[.65,-.04])is None


@pytest.mark.parametrize('x',[.629,.671])
def test_source_forward_x_must_enter_reviewed_band(x):
    r=route();update(r,1.02,[0.,.07]);assert not update(r,1.04,[x,.085]);assert r.index==1


def test_alignment_uses_actual_both_components_and_continuous_native_dwell():
    r=route();update(r,1.02,[0.,.07]);update(r,1.04,[.65,.085]);update(r,1.06,[.65,-.04])
    # Passing the lateral target quickly cannot advance a phase.
    assert r.index==2 and r.qualified_duration_s==0
    for t in np.arange(1.08,1.361,.02):update(r,float(t),[.65,-.04])
    assert not r.final
    update(r,1.38,[.65,-.04]);assert r.final


@pytest.mark.parametrize('delta,heading',[(np.array([.001,0.]),0.),(np.array([0.,.001]),0.),(np.zeros(2),.081)])
def test_alignment_motion_or_heading_resets_dwell(delta,heading):
    r=route();update(r,1.02,[0.,.07]);update(r,1.04,[.65,.085]);update(r,1.06,[.65,-.04])
    for t in np.arange(1.08,1.301,.02):update(r,float(t),[.65,-.04])
    update(r,1.32,np.array([.65,-.04])+delta,heading=heading)
    assert r.index==2 and r.qualified_duration_s==0
    for t in np.arange(1.34,1.621,.02):update(r,float(t),[.65,-.04])
    assert not r.final


def test_unobserved_native_gap_cannot_supply_continuous_alignment_dwell():
    r=route();update(r,1.02,[0.,.07]);update(r,1.04,[.65,.085]);update(r,1.06,[.65,-.04])
    update(r,1.08,[.65,-.04]);update(r,1.40,[.65,-.04])
    assert not r.final and r.qualified_duration_s==0


def test_repeated_pose_stamp_keeps_actual_velocity_and_does_not_add_dwell():
    r=route();update(r,1.02,[0.,.07]);update(r,1.04,[.65,.085]);update(r,1.06,[.65,-.04])
    velocity=r.root_velocity_route.copy();update(r,1.08,[.65,-.04],stamp=1.06)
    np.testing.assert_array_equal(r.root_velocity_route,velocity);assert r.qualified_duration_s==0
    update(r,1.10,[.65,-.04]);update(r,1.12,[.65,-.04],stamp=1.10)
    assert r.qualified_duration_s==0 and not r.final


def test_alignment_timeout_fails_latched_without_skip():
    r=route();update(r,1.02,[0.,.07]);update(r,1.04,[.65,.085])
    with pytest.raises(ReviewedApproachError,match='alignment_timeout'):
        update(r,31.05,[.65,.085])
    assert r.index==2 and not r.final
    with pytest.raises(ReviewedApproachError,match='alignment_timeout'):
        update(r,31.07,[.65,-.04])


@pytest.mark.parametrize('local,reason',[([.85,-.055001],'final_lateral_corridor_exit'),
                                        ([.880001,-.04],'final_forward_overshoot'),
                                        ([.629999,-.04],'final_source_backtrack')])
def test_final_corridor_failure_latches_before_changing_goal(local,reason):
    r=route();aligned(r);goal=r.goal
    with pytest.raises(ReviewedApproachError,match=reason):update(r,1.50,local)
    np.testing.assert_array_equal(r.goal,goal)
    assert r.evidence()['failure']['code']==reason
    with pytest.raises(ReviewedApproachError,match=reason):update(r,1.52,[.85,-.04])


def test_final_corridor_inclusive_boundaries_and_goal_are_fixed():
    r=route();aligned(r)
    for i,p in enumerate([[.63,-.055],[.88,-.025],[.85,-.04]]):
        assert not update(r,1.50+i*.02,p)
    np.testing.assert_allclose(r.goal,r.start+[.85,-.04])


@pytest.mark.parametrize('local',[([np.nan,0.]),([.65,np.inf])])
def test_bad_position_does_not_mutate_phase(local):
    r=route()
    with pytest.raises(ValueError):
        r.advance(local,heading=0.,native_time=1.02,pose_native_time=1.02)
    assert r.index==0


def ctl(final=True):
    c=ReviewedApproachController(1,cfg=navigation_config());c.reviewed_final_phase=final
    c.goal[0]=torch.tensor([1.,0.]);return c


def tick(c,xy=(0.,0.),vel=(0.,0.),dt=.02):
    return c.update(torch.tensor([xy],dtype=torch.float32),torch.zeros(1),
                    torch.tensor([vel],dtype=torch.float32),torch.zeros(1),dt).clone()


def test_final_cap_limits_desired_forward_but_preserves_reverse_and_cfg():
    c=ctl();cfg=c.cfg;serialized=asdict(cfg)
    for _ in range(80):out=tick(c)
    assert out[0,0]==pytest.approx(.04,abs=1e-6)
    assert c.cfg is cfg and asdict(c.cfg)==serialized
    c.goal[0]=torch.tensor([-1.,0.])
    for _ in range(100):out=tick(c)
    assert out[0,0]==pytest.approx(-.10,abs=1e-6)


def test_switch_to_final_preserves_acceleration_filter_and_command_history():
    c=ctl(False)
    for _ in range(80):tick(c)
    assert c.command[0,0]==pytest.approx(.10,abs=1e-6)
    previous=c.command.clone();before=c.filtered_xy.clone();c.reviewed_final_phase=True
    out=tick(c,xy=(.01,0.))
    assert out[0,0]>.04 and torch.linalg.vector_norm(out[0,:2]-previous[0,:2])<=.0030001
    expected=before+(.02/.22)*(torch.tensor([[.01,0.]])-before)
    torch.testing.assert_close(c.filtered_xy,expected)


def test_cap_decision_uses_next_filter_and_velocity_damping_not_old_error():
    c=ctl();c.goal[:]=0.;c.filtered_xy[0]=torch.tensor([-.2,0.]);c.filter_fresh[:]=False
    c.command[0,0]=-.08
    out=tick(c,xy=(10.,0.))
    assert out[0,0]==pytest.approx(-.083,abs=1e-6)
    # Goal forward, but actual forward velocity calls for full reverse braking.
    c=ctl();c.command[0,0]=-.08
    out=tick(c,vel=(20.,0.));assert out[0,0]==pytest.approx(-.083,abs=1e-6)


def test_controller_cfg_restored_on_invalid_update():
    c=ctl();cfg=c.cfg
    with pytest.raises(ValueError):tick(c,dt=0.)
    assert c.cfg is cfg


def test_controller_cfg_restored_if_base_update_raises(monkeypatch):
    c=ctl();cfg=c.cfg
    def fail(*args,**kwargs):
        assert c.cfg.max_vx==.04
        raise RuntimeError('injected base failure')
    monkeypatch.setattr(world_xy.WorldXYController,'update',fail)
    with pytest.raises(RuntimeError,match='injected'):tick(c)
    assert c.cfg is cfg


@pytest.mark.parametrize('final,goal,limit',[(False,[1.,.4],.10),
                                           (True,[1.,.4],.04),
                                           (True,[-1.,.4],.10)])
def test_controller_matches_base_state_update_except_reviewed_desired_limit(final,goal,limit):
    c=ctl(final);base=world_xy.WorldXYController(1,cfg=replace(navigation_config(),max_vx=limit))
    for controller in (c,base):
        controller.goal[0]=torch.tensor(goal);controller.filtered_xy[0]=torch.tensor([.02,.03])
        controller.filter_fresh[:]=False;controller.command[0]=torch.tensor([.05,.02,.1])
        controller.heading[0]=.1
    args=(torch.tensor([[0.,.1]]),torch.tensor([.2]),torch.tensor([[.01,-.02]]),torch.tensor([.01]),.02)
    torch.testing.assert_close(c.update(*args),base.update(*args),rtol=0,atol=0)
    for field in ('filtered_xy','filter_fresh','arrived','settled_s','command','goal','heading'):
        torch.testing.assert_close(getattr(c,field),getattr(base,field),rtol=0,atol=0)
    assert c.cfg==navigation_config()


def snap(t,x=0.,y=0.):
    return SimpleNamespace(native_time=t,pose_native_time=t,
        pelvis_pose=np.array([x,y,.87,1.,0.,0.,0.]),
        torso_pose=np.array([x-.05,y,.94,np.cos(-.38),0.,0.,np.sin(-.38)]),
        payload_pose=np.array([x+.22,y-.24,.88,1.,0.,0.,0.]))


def feedback(enabled=True):return NavigationFeedback([1.07,-.28,.758],-.76,reviewed_approach=enabled)


@pytest.mark.parametrize('enabled',[None,1,'true'])
def test_reviewed_approach_requires_explicit_bool(enabled):
    with pytest.raises(ValueError,match='bool'):feedback(enabled)


def test_default_feedback_and_legacy_waypoints_stay_unchanged():
    c=NavigationFeedback([1.07,-.28,.758],-.76);c.capture(snap(0.));c.step(snap(.02,y=.07));c.step(snap(.04,x=.4,y=.085))
    assert c.navigation.route.final and type(c._controller)is world_xy.WorldXYController
    assert 'reviewed_approach'not in c.contract


def test_explicit_feedback_keeps_final_arrival_requirements_and_failure_evidence():
    c=feedback();c.capture(snap(0.));c.step(snap(.02,y=.07));c.step(snap(.04,x=.4,y=.085))
    assert not c.navigation.route.final
    c.step(snap(.06,x=.65,y=.085));c.step(snap(.08,x=.65,y=-.04))
    for i in range(5,21):c.step(snap(i*.02,x=.65,y=-.04))
    assert c.navigation.route.final and not c.success
    c.step(snap(.42,x=.85,y=-.04));assert not c.success
    for i in range(22,100):c.step(snap(i*.02,x=.85,y=-.04))
    assert c.success and c.arrival_signal(30.,snap(1.98,.85,-.04))['feedback_settled']
    before=c._controller.command.clone()
    with pytest.raises(ReviewedApproachError,match='final_lateral_corridor_exit'):
        c.step(snap(2.,x=.85,y=-.06))
    torch.testing.assert_close(before,c._controller.command)
    assert c.events[-1]['event']=='reviewed_approach_rejected'
    assert c.metrics['navigation']['route']['failure']['code']=='final_lateral_corridor_exit'
    assert not c.success and not c.metrics['success']


def test_stale_snapshot_cannot_mutate_reviewed_route():
    c=feedback();c.capture(snap(0.));s=snap(.1,y=.07);s.pose_native_time=0.
    with pytest.raises(ValueError):c.step(s)
    assert c.navigation.route.index==0
