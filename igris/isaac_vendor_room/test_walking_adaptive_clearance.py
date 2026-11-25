"""Measured clearance cap; deterministic state-machine checks, no physics."""
import numpy as np
import pytest

from walking_transfer import WalkingTransfer
from test_walking_transfer import fixture


def setup(**kwargs):
    _,p,t,b,h=fixture()
    options=dict(clearance_lift=.05,clearance_minimum_rise=.02,
                 heading_preserving_route=True,adaptive_clearance=True)
    options.update(kwargs)
    c=WalkingTransfer([1.16,-.26,.758],-np.pi/4,.01,4.,**options)
    c.start(0.,p,t,b,h)
    return c,p,t,b,h


def prime(c,p,t,b,*,rise=.024364,force=(6.,0.,6.,0.),support=0.):
    z=b[2]
    for stamp in np.arange(1,180)/30.:
        b[2]=z+rise*stamp/6.
        state=c.update(float(stamp),p,t,b,force,support,ik_position_error_m=.009)
    b[2]=z+rise
    return state


def test_trial98_error_caps_previous_request_once_then_preserves_full_entry_gates():
    c,p,t,b,h=setup();prior=prime(c,p,t,b)
    state=c.update(6.,p,t,b,[6,0,6,0],0.,ik_position_error_m=.010)
    # The accepted-error trigger sees24.364mm actual rise; it must not issue
    # the next nominal target, or retract to the measured box rise.
    np.testing.assert_array_equal(state['world_hand_targets'],prior['world_hand_targets'])
    cap=state['clearance_requested_cap_fraction']
    assert .8<cap<.9 and .04<state['clearance_effective_ee_rise_m']<.045
    assert state['clearance_lift_m']==.05
    trigger=state['clearance_cap_trigger']
    assert trigger['native_time_s']==6. and trigger['ik_position_error_m']==.010
    assert trigger['measured_rise_m']==pytest.approx(.024364)
    assert trigger['requested_fraction']==cap
    targets=np.array(state['world_hand_targets'])
    for stamp in np.arange(181,254)/30.:
        state=c.update(float(stamp),p,t,b,[6,0,6,0],0.,ik_position_error_m=.001)
        np.testing.assert_array_equal(state['world_hand_targets'],targets)
        assert state['clearance_requested_cap_fraction']==cap
        assert state['clearance_cap_trigger']==trigger
        assert state['phase']=='clearance' and not np.any(state['command_xyz'])
    state=c.update(8.5,p,t,b,[6,0,6,0],0.,ik_position_error_m=.001)
    assert state['phase']=='carry' and state['clearance_ready_s']>=.5
    # The accepted-error metadata is clearance-only, not a new lowering gate.
    c.update(8.6,p,t,b,[6,0,6,0],0.)


@pytest.mark.parametrize('change',[{'rise':.0199},{'force':(0.,6.,6.,0.)},
                                 {'support':.02}])
def test_no_cap_when_actual_rise_palm_or_source_unload_gate_is_missing(change):
    c,p,t,b,_=setup();prior=prime(c,p,t,b)
    if 'rise' in change:b[2]=c.clearance_start_box_z+change['rise']
    state=c.update(6.,p,t,b,change.get('force',[6,0,6,0]),change.get('support',0.),
                   ik_position_error_m=.012)
    assert state['clearance_requested_cap_fraction'] is None
    assert state['clearance_effective_ee_rise_m']>prior['clearance_effective_ee_rise_m']
    assert state['phase']=='clearance' and not np.any(state['command_xyz'])


def test_fast_payload_prevents_cap_even_with_qualified_rise_and_force():
    c,p,t,b,_=setup();prime(c,p,t,b)
    b[2]+=.02
    state=c.update(6.,p,t,b,[6,0,6,0],0.,ik_position_error_m=.012)
    assert state['payload_speed_m_s']>.02
    assert state['clearance_requested_cap_fraction'] is None


@pytest.mark.parametrize('error',[None,np.nan,np.inf,-.001,[],[.01],True,'0.01'])
def test_missing_or_invalid_accepted_error_fails_before_clearance_mutation(error):
    c,p,t,b,_=setup()
    hands=c.local_hands.copy();stamp=c.last_time
    with pytest.raises(ValueError,match='IK.*error|error.*IK'):
        c.update(.02,p,t,b,[6,0,6,0],0.,ik_position_error_m=error)
    assert c.last_time==stamp and c.phase=='clearance'
    np.testing.assert_array_equal(c.local_hands,hands)


@pytest.mark.parametrize('kwargs',[{'adaptive_clearance':1},
    {'heading_preserving_route':False,'clearance_minimum_rise':None},
    {'clearance_lift':0.,'clearance_minimum_rise':None},
    {'clearance_minimum_rise':.019}])
def test_adaptive_requires_explicit_reviewed_route_and_twenty_mm_minimum(kwargs):
    with pytest.raises(ValueError,match='Adaptive|adaptive'):setup(**kwargs)


@pytest.mark.parametrize('gate',['rise','palm','source','speed'])
def test_capped_clearance_cannot_depart_without_original_measured_gates(gate):
    c,p,t,b,_=setup();prime(c,p,t,b)
    c.update(6.,p,t,b,[6,0,6,0],0.,ik_position_error_m=.010)
    for stamp in np.arange(181,260)/30.:
        f=[6,0,6,0];support=0.
        # Invalidate near the end of the minimum8s path, for less than the
        # independent600ms palm-loss deadline when testing a missing palm.
        if stamp>=7.95:
            if gate=='rise':b[2]=c.clearance_start_box_z+.019
            elif gate=='palm':f=[0,6,6,0]
            elif gate=='source':support=.02
            else:b[2]+=.001
        if gate=='palm' and stamp>8.4:break
        state=c.update(float(stamp),p,t,b,f,support,ik_position_error_m=.001)
        assert state['phase']=='clearance' and not np.any(state['command_xyz'])
    assert state['clearance_ready_s']<.5


def test_cap_does_not_turn_deadline_into_success():
    c,p,t,b,_=setup();prime(c,p,t,b)
    c.update(6.,p,t,b,[6,0,6,0],0.,ik_position_error_m=.010)
    with pytest.raises(RuntimeError,match='clearance'):
        c.update(20.01,p,t,b,[6,0,6,0],.02,ik_position_error_m=.001)
    assert c.phase=='clearance' and not np.any(c.command)


def test_default_ignores_optional_error_and_reaches_original_full_fifty_mm_target():
    c,p,t,b,h=setup(adaptive_clearance=False)
    for stamp in np.arange(1,250)/30.:
        state=c.update(float(stamp),p,t,b,[6,0,6,0],0.)
    assert state['phase']=='clearance'  # no actual payload rise
    np.testing.assert_allclose(np.array(state['world_hand_targets'])[:,:3,3],
                               h[:,:3,3]+[0,0,.05],atol=1e-12)
    assert state['clearance_requested_cap_fraction'] is None
