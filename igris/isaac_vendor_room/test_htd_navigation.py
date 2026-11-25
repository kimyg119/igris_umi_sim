"""Frame and route regressions for the future opt-in HTD transport scope."""
import numpy as np
import pytest
import torch

from htd_goal_hold import world_xy
from htd_navigation import NavigationGoals


def pose(x, y, yaw=0., z=.9):
    return np.array([x,y,z,np.cos(yaw/2),0.,0.,np.sin(yaw/2)])


def case(angle=0.):
    c,s=np.cos(angle),np.sin(angle);r=np.array([[c,-s],[s,c]])
    p=np.array([.2,.3]);t=p+r@[-.05,0.];b=p+r@[.25,-.20]
    target=b+r@[.8,-.04]
    return NavigationGoals(pose(*p,angle),pose(*t,angle-.75),pose(*b),
                           [*target,.758],angle-.75),p,r


def test_fixed_torso_offset_rotated_route_and_payload_goal_are_consistent():
    n,p,r=case(.7)
    np.testing.assert_allclose(n.final_pelvis_xy,p+r@[.8,-.04])
    np.testing.assert_allclose(n.final_torso_xy,p+r@[.75,-.04])
    np.testing.assert_allclose(n.goal_torso_xy,p+r@[-.05,.085])
    assert n.heading==pytest.approx(.7)
    assert n.evidence()['heading_frame']=='pelvis yaw'


def test_transit_does_not_latch_at_55mm_before_65mm_departure_plane():
    n,p,r=case();ctl=world_xy.WorldXYController(1)
    n.apply(ctl)
    measured=p+[-.05,.055]
    cmd=ctl.update(torch.tensor(measured[None],dtype=torch.float32),torch.zeros(1),
                   torch.zeros(1,2),torch.zeros(1),.02)
    assert cmd[0,1]>0 and not ctl.arrived[0]
    assert n.update(pose(*(p+[0.,.055]))) is False
    assert n.update(pose(*(p+[0.,.070]))) is True
    np.testing.assert_allclose(n.goal_torso_xy,p+[.35,.085])


def test_waypoint_change_retains_bounded_command_and_filter_history():
    n,p,r=case();ctl=world_xy.WorldXYController(1)
    n.apply(ctl)
    ctl.command[0]=torch.tensor([.01,.02,-.03]);ctl.filtered_xy[0]=torch.tensor([.15,.36])
    ctl.filter_fresh[:]=False;ctl.arrived[:]=True;ctl.settled_s[:]=.8
    before=ctl.command.clone();filtered=ctl.filtered_xy.clone()
    n.update(pose(*(p+[0.,.07])));n.apply(ctl)
    assert torch.equal(ctl.command,before) and torch.equal(ctl.filtered_xy,filtered)
    assert not ctl.filter_fresh[0] and not ctl.arrived[0] and ctl.settled_s[0]==0
    ctl.arrived[:]=True;ctl.settled_s[:]=.2
    n.apply(ctl)  # unchanged goal must not erase qualification each actor tick
    assert ctl.arrived[0] and ctl.settled_s[0]==pytest.approx(.2)


def test_recovery_uses_measured_pelvis_and_fixed_captured_torso_offset():
    n,p,r=case();n.update(pose(*(p+[0.,.07])))
    n.update(pose(*(p+[.20,.04])))
    assert n.recovering
    np.testing.assert_allclose(n.goal_torso_xy,p+[.15,.085])
    n.update(pose(*(p+[.36,.07])))
    assert n.route.final and not n.recovering
    np.testing.assert_allclose(n.goal_torso_xy,n.final_torso_xy)
    ctl=world_xy.WorldXYController(1);n.apply(ctl)
    assert ctl.cfg.arrival_radius==.015 and ctl.cfg.departure_radius==.03


def test_final_drift_outside_release_region_restarts_translation():
    n,p,r=case();n.update(pose(*(p+[0.,.07])))
    n.update(pose(*(p+[.36,.07])))
    ctl=world_xy.WorldXYController(1);n.apply(ctl)
    ctl.arrived[:]=True
    measured=n.final_torso_xy+np.array([.04,0.])
    for _ in range(100):
        cmd=ctl.update(torch.tensor(measured[None],dtype=torch.float32),torch.zeros(1),
                       torch.zeros(1,2),torch.zeros(1),.02)
    assert not ctl.arrived[0] and cmd[0,0]<0
    assert ctl.cfg.position_gain==2.5 and ctl.cfg.heading_gain==4.
    assert ctl.cfg.position_filter_s==.2


def test_heading_mismatch_and_invalid_route_reject_before_navigation():
    with pytest.raises(ValueError,match='heading'):
        NavigationGoals(pose(0,0),pose(-.05,0,-.75),pose(.25,-.2),[1.05,-.24,.758],0.)
    with pytest.raises(ValueError,match='corridor'):
        NavigationGoals(pose(0,0),pose(-.05,0,-.75),pose(.25,-.2),[-1.05,-.24,.758],-.75)
