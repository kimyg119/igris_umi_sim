"""One-shot command arithmetic; no policy inference, physics, or service calls."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from htd_reviewed_approach import ReviewedApproachController, ReviewedApproachRoute, ReviewedApproachError
from htd_navigation import navigation_config
from htd_navigation_feedback import NavigationFeedback
from test_htd_reviewed_approach import snap, tick, route, update


def controller(selected=True):
    c=ReviewedApproachController(1,cfg=navigation_config(),prealignment_coast_seconds=3. if selected else None)
    c.route_context=route();r=c.route_context
    update(r,1.02,[0.,.07]);update(r,1.04,[.65,.085]);update(r,1.06,[.65,-.04]);update(r,1.08,[.65,-.04])
    c.goal[0]=torch.tensor([.03,0.]);c.heading[0]=.1
    return c


def coast_tick(c,t,local=(.65,-.04),heading=0.):
    update(c.route_context,t,local,heading=heading)
    return tick(c,xy=(.01,.005),vel=(.01,0.))


def test_selected_base_advances_once_and_override_is_stored_slew():
    c=controller();c.command[0]=torch.tensor([.004,0.,.01]);c.filter_fresh[:]=False
    baseline=copy.deepcopy(c);baseline.prealignment_coast_seconds=None
    nominal=tick(baseline,xy=(.01,.005),vel=(.01,0.))
    out=tick(c,xy=(.01,.005),vel=(.01,0.))
    assert out[0,0]==pytest.approx(.001,abs=1e-8) and out[0,2]==0
    torch.testing.assert_close(out,c.command,rtol=0,atol=0)
    for key in ('filtered_xy','filter_fresh','arrived','settled_s'):
        torch.testing.assert_close(getattr(c,key),getattr(baseline,key),rtol=0,atol=0)
    e=c.coast_evidence();assert e['state']=='ramping' and e['zero_started_native'] is None
    np.testing.assert_array_equal(e['nominal_command_xyz'],nominal[0].numpy())
    coast_tick(c,1.10,heading=.081)
    assert c.coast_evidence()['state']=='zero' and c.coast_evidence()['zero_started_native']==1.10


@pytest.mark.parametrize('prior',[[.00501,0,0],[0,0,.01001]])
def test_moving_prior_cannot_admit(prior):
    c=controller();c.command[0]=torch.tensor(prior);tick(c)
    assert c.coast_evidence()['state']=='armed'


@pytest.mark.parametrize('local,heading', [([.629,-.04],0),([.65,-.056],0),([.65,-.04],.08001),([.651,-.04],0)])
def test_all_actual_original_predicates_required(local,heading):
    c=controller();coast_tick(c,1.10,local,heading)
    assert c.coast_evidence()['state']=='armed'


def test_repeated_pose_and_insufficient_remaining_time_do_not_admit():
    c=controller();update(c.route_context,1.10,[.65,-.04],stamp=1.08);tick(c)
    assert c.coast_evidence()['state']=='armed'
    c=controller();c.route_context.alignment_started_native=-26.
    tick(c);assert c.coast_evidence()['state']=='armed'


def test_full_three_seconds_then_resume_once_without_resetting_filter():
    c=controller();tick(c);start=c.coast_evidence()['zero_started_native'];assert start==1.08
    for i in range(1,150):
        out=coast_tick(c,start+i*.02,heading=.081)
        assert torch.count_nonzero(out)==0
    before=c.filtered_xy.clone()
    out=coast_tick(c,start+3.,heading=.081)
    assert c.coast_evidence()['state']=='completed' and torch.count_nonzero(out)>0
    assert c.coast_evidence()['zero_elapsed_s']==pytest.approx(3.)
    expected=before+.02/.22*(torch.tensor([[.01,.005]])-before)
    torch.testing.assert_close(c.filtered_xy,expected,rtol=0,atol=0)
    c.command[:]=0;coast_tick(c,start+3.02)
    assert c.coast_evidence()['state']=='completed'


def test_original_actual_dwell_advances_and_censors_before_three_seconds():
    c=controller();tick(c)
    for i in range(1,16):coast_tick(c,1.08+i*.02)
    assert c.route_context.final
    assert c.coast_evidence()['state']=='censored'
    assert c.coast_evidence()['reason']=='route_left_prealignment'
    assert c.coast_evidence()['zero_elapsed_s']==pytest.approx(.3)
    assert torch.count_nonzero(c.command)>0


def test_original_timeout_retained_during_coast():
    c=controller();tick(c)
    with pytest.raises(ReviewedApproachError,match='alignment_timeout'):
        coast_tick(c,31.06,heading=.081)


@pytest.mark.parametrize('phase',[0,1,3])
def test_source_final_commands_and_default_state_are_identical(phase):
    c=controller();c.route_context.index=phase;c.reviewed_final_phase=phase==3
    baseline=copy.deepcopy(c);baseline.prealignment_coast_seconds=None
    for _ in range(8):
        c.route_context._last_native+=.02
        a=tick(c);b=tick(baseline)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        for key in ('filtered_xy','filter_fresh','arrived','settled_s'):
            torch.testing.assert_close(getattr(c,key),getattr(baseline,key),rtol=0,atol=0)
    assert c.coast_evidence()['state']=='armed'


def test_navigation_duplicate_timestamp_keeps_histories_and_events_unchanged():
    c=NavigationFeedback([1.07,-.28,.758],-.76,reviewed_approach=True,prealignment_coast_seconds=3.)
    c.capture(snap(0.));a=c.step(snap(.02,y=.07));before=copy.deepcopy(c)
    b=c.step(snap(.02,y=.07))
    np.testing.assert_array_equal(a,b);assert c.events==before.events
    for key in ('filtered_xy','settled_s','command'):
        torch.testing.assert_close(getattr(c._controller,key),getattr(before._controller,key),rtol=0,atol=0)
    assert c.events[-1]['prealignment_coast']==c.metrics['prealignment_coast']
    c.censor_coast('operator stop')
    assert c.metrics['prealignment_coast']['state']=='not_triggered'
    assert c.metrics['prealignment_coast']['ended_native'] is None


@pytest.mark.parametrize('bad',[True,False,0.,2.99,3.01,'3',float('nan'),float('inf'),10**400])
def test_constructor_rejects_unsupported_value(bad):
    with pytest.raises(ValueError,match='coast'):
        ReviewedApproachController(1,prealignment_coast_seconds=bad)


def test_coast_without_reviewed_route_is_rejected():
    with pytest.raises(ValueError,match='reviewed'):
        NavigationFeedback([1,0,0],0,prealignment_coast_seconds=3.)
