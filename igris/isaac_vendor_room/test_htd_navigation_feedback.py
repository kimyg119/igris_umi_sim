"""Pose freshness, route timing and real measured-velocity arrival feedback."""
from types import SimpleNamespace
import numpy as np
import pytest


def snapshot(t,x=0.,y=0.):
    return SimpleNamespace(native_time=t,pose_native_time=t,
        pelvis_pose=np.array([x,y,.87,1.,0.,0.,0.]),
        torso_pose=np.array([x-.05,y,.94,np.cos(-.38),0.,0.,np.sin(-.38)]),
        payload_pose=np.array([x+.22,y-.24,.88,1.,0.,0.,0.]))


def controller():
    from htd_navigation_feedback import NavigationFeedback
    return NavigationFeedback([1.12,-.24,.758],-.76)


def test_immediate_route_commands_are_bounded_and_waypoints_preserve_command_history():
    c=controller();c.capture(snapshot(0.))
    for i in range(1,21):
        a=c.step(snapshot(i*.02))
        assert a[0]==pytest.approx(0.,abs=1e-7)
        assert 0<a[1]<=.003*i+1e-7
    previous=a.copy()
    a=c.step(snapshot(.42,y=.07))
    assert c.navigation.route.index==1
    assert np.max(abs(a[:2]-previous[:2]))<=.003+1e-7
    assert not c.success
    assert c.metrics['goal_xy_world']==pytest.approx(c.navigation.goal_torso_xy.tolist())


def test_final_arrival_requires_measured_stop_and_reports_fixed_pelvis_goal():
    c=controller();c.capture(snapshot(0.));c.step(snapshot(.02,y=.07))
    c.step(snapshot(.04,x=.38,y=.085));c.step(snapshot(.06,x=.9,y=0.))
    assert c.navigation.route.final and not c.success
    for i in range(4,90):c.step(snapshot(i*.02,x=.9))
    assert c.success
    signal=c.arrival_signal(100.,snapshot(1.78,x=.9))
    assert signal['sample_time_s']==100. and signal['route_final'] and signal['feedback_settled']
    np.testing.assert_allclose(signal['goal_xy_world'],[.9,0.])
    with pytest.raises(ValueError,match='fresh'):
        c.arrival_signal(100.1,snapshot(1.9,x=.9))


def test_stale_pose_rejected_before_advancing_route():
    c=controller();c.capture(snapshot(0.));s=snapshot(.1,y=.07);s.pose_native_time=0.
    with pytest.raises(ValueError):c.step(s)
    assert c.navigation.route.index==0
    assert not c.metrics['success']
    with pytest.raises(RuntimeError):c.capture(snapshot(.1))
