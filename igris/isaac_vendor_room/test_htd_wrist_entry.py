"""Pure transition tests; no ROS, native controller or simulation is started."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def helper():
    path=Path(__file__).with_name('htd_wrist_entry.py')
    assert path.exists(), 'Bounded wrist mode-entry helper is not implemented'
    spec=importlib.util.spec_from_file_location('htd_wrist_entry',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.WristModeEntry()


def command():
    q=np.arange(31,dtype=float)/100
    q[[20,21,27,28]]=[.810673,.163189,-.726243,.267538]
    c=dict(q=q,dq=np.zeros(31),tau=np.arange(31,dtype=float),
           kp=np.full(31,250.),kd=np.full(31,3.),kinematic_modes=np.ones(5,dtype=int),
           metadata=dict(epoch=2,tag=['original']))
    c['kp'][[20,21,27,28]]=400.;c['kd'][[20,21,27,28]]=2.
    c['tau'][[20,21,27,28]]=0.
    return c


def sample(wall=100.,native=10.,mode=6):
    q=command()['q'].copy();q[[20,21,27,28]]=[.592757,.704105,-.489458,-.681106]
    return SimpleNamespace(wall_time=wall,native_time=native,mode=mode,output=dict(q_motor=q))


def assert_equal_command(actual,expected):
    for key in ['q','dq','tau','kp','kd','kinematic_modes']:
        np.testing.assert_array_equal(actual[key],expected[key])
    assert actual['metadata']==expected['metadata']


def prime(h,c=None):
    c=command() if c is None else c
    h.apply(c,sample())
    h.apply(c,sample(100.5,10.05))
    assert h.ready_for_low(sample(100.5,10.05))


def test_only_four_frozen_motor_positions_and_two_modes_change_without_input_mutation():
    h=helper();c=command();original=copy.deepcopy(c);s=sample()
    got=h.apply(c,s)
    expected=copy.deepcopy(c);expected['q'][[20,21,27,28]]=s.output['q_motor'][[20,21,27,28]]
    expected['kinematic_modes']=np.array([1,1,1,0,0])
    assert_equal_command(got,expected);assert_equal_command(c,original)
    got['q'][0]=999;got['metadata']['tag'].append('changed')
    assert_equal_command(c,original)


def test_seeding_never_rebases_and_other_live_fields_keep_following_input():
    h=helper();c=command();first=h.apply(c,sample())
    changed=copy.deepcopy(c);changed['q'][0]=1.23;changed['tau'][3]=4.56
    s=sample(100.1,10.01);s.output['q_motor'][[20,21,27,28]]+=.005
    got=h.apply(changed,s)
    np.testing.assert_array_equal(got['q'][[20,21,27,28]],first['q'][[20,21,27,28]])
    assert got['q'][0]==1.23 and got['tau'][3]==4.56
    assert len([e for e in h.events if e['event']=='seeded_motor_wrist_hold'])==1


def test_priming_requires_native_dwell_and_latches_before_pending_low():
    h=helper();c=command()
    assert not h.ready_for_low(sample())
    h.apply(c,sample())
    assert not h.ready_for_low(sample(100.49,10.049))
    assert h.ready_for_low(sample(100.5,10.05))
    assert h.ready_for_low(sample(102.1,10.06))  # Pending LOW timeout belongs to coordinator.


def test_unprimed_native_clock_stall_times_out_in_wall_time():
    h=helper();h.apply(command(),sample())
    with pytest.raises(RuntimeError,match='priming'):
        h.ready_for_low(sample(102.001,10.001))


def test_low_cannot_be_entered_before_native_priming():
    h=helper();h.apply(command(),sample())
    with pytest.raises(RuntimeError,match='primed'):
        h.apply(command(),sample(100.1,10.01,3))


def test_restoration_requires_low_wall_time_and_returns_current_original_pjs():
    h=helper();c=command();prime(h,c)
    h.apply(c,sample(100.6,10.06,3))
    for t in np.arange(100.65,103.66,.05):
        got=h.apply(c,sample(float(t),10.+(t-100.)*.1,3))
        assert list(got['kinematic_modes'])==[1,1,1,0,0]
    current=copy.deepcopy(c);current['q'][21]+=.007
    got=h.apply(current,sample(103.71,10.371,3))
    assert_equal_command(got,current)
    assert h.state=='released'
    assert h.events[-1]['event']=='restored_live_pjs'
    assert h.events[-1]['deactivation_observed'] is False


def test_error_in_any_other_upper_joint_resets_continuous_expiration_condition():
    h=helper();c=command();prime(h,c);h.apply(c,sample(100.6,10.06,3))
    for t in np.arange(100.65,103.66,.05):
        h.apply(c,sample(float(t),10.+(t-100.)*.1,3))
    # A shoulder disturbance at the otherwise eligible release time must discard
    # the preceding three seconds of good measurements, not merely wrist checks.
    bad=sample(103.70,10.370,3);bad.output['q_motor'][15]+=.19;h.apply(c,bad)
    assert h.state=='low_hold'
    h.apply(c,sample(103.75,10.375,3))
    assert h.state=='low_hold'
    h.apply(c,sample(103.80,10.380,3))
    assert h.state=='low_hold'
    h.apply(c,sample(103.851,10.3851,3))
    assert h.state=='released'


def test_sparse_good_samples_do_not_prove_continuous_error_bound():
    h=helper();prime(h);c=command();h.apply(c,sample(100.6,10.06,3))
    h.apply(c,sample(103.8,10.38,3))
    assert h.state=='low_hold'
    h.apply(c,sample(103.85,10.385,3));h.apply(c,sample(103.901,10.3901,3))
    assert h.state=='released'


def test_unmet_low_condition_times_out_instead_of_releasing():
    h=helper();prime(h);c=command();h.apply(c,sample(100.6,10.06,3))
    bad=sample(105.101,10.5101,3);bad.output['q_motor'][22]+=.2
    with pytest.raises(RuntimeError,match='LOW'):
        h.apply(c,bad)
    assert h.state!='released'


@pytest.mark.parametrize('field,value',[('kp',399.),('kd',1.),('dq',.01),('tau',.01)])
def test_rejects_unreviewed_wrist_gains_velocity_or_host_feedforward(field,value):
    h=helper();c=command();c[field][21]=value
    with pytest.raises(ValueError,match='wrist'):
        h.apply(c,sample())


def test_rejects_transformed_command_feedback_instead_of_using_motor_q_as_live_pjs():
    h=helper();got=h.apply(command(),sample())
    with pytest.raises(ValueError,match='PJS'):
        h.apply(got,sample(100.1,10.01))


@pytest.mark.parametrize('change',['wall_backward','native_backward','bad_mode','motor_nan','command_nan'])
def test_invalid_clock_mode_or_nonfinite_input_fails_closed(change):
    h=helper();c=command();h.apply(c,sample());s=sample(100.1,10.01)
    if change=='wall_backward':s.wall_time=99.
    if change=='native_backward':s.native_time=9.
    if change=='bad_mode':s.mode=2
    if change=='motor_nan':s.output['q_motor'][20]=np.nan
    if change=='command_nan':c['tau'][3]=np.nan
    with pytest.raises((ValueError,RuntimeError)):
        h.apply(c,s)


def test_mode_regression_after_low_is_rejected():
    h=helper();prime(h);h.apply(command(),sample(100.6,10.06,3))
    with pytest.raises(RuntimeError,match='mode'):
        h.apply(command(),sample(100.7,10.07,6))
