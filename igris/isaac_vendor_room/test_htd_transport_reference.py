import numpy as np
import pytest
from htd_transport_reference import applied_transport_reference
from reference_spline import evaluate


def case(phase='lower', planned='lower'):
    segment=dict(start_time=10.,end_time=10.+1/30,accepted=True,prior_status='used',
                 walking_phase=phase,reference_control_time_s=9.9,
                 control_points=np.arange(56,dtype=float).reshape(4,14).tolist())
    walking=dict(phase=planned,target_payload_xyz=[1.,-.2,.758],done=planned=='done')
    q=np.zeros(31);q[15:29]=evaluate(segment['control_points'],.5)
    return walking,segment,10.+1/60,q


def test_fresh_accepted_applied_segment_authorizes_only_its_phase():
    args=case('carry','lower')
    result=applied_transport_reference(*args)
    assert result['transport_phase']=='carry' and result['phase_target_valid']
    assert not result['placement_done']
    assert applied_transport_reference(*case('done','done'))['placement_done']


def test_different_interpolated_target_is_not_authorized():
    args=case();args[-1][18]+=.001
    with pytest.raises(ValueError,match='applied'):applied_transport_reference(*args)


def test_unaccepted_missing_stale_or_future_phase_rejected():
    for key,value in [('accepted',False),('walking_phase',None),('walking_phase','open'),
                      ('reference_control_time_s',11.)]:
        args=case();args[1][key]=value
        with pytest.raises(ValueError):applied_transport_reference(*args)
    args=list(case());args[2]=10.5
    with pytest.raises(ValueError,match='timeline'):applied_transport_reference(*args)
