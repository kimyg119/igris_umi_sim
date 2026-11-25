import pytest
import run


def metadata():
    return dict(wall=10.,reference_control_time_s=5.,end_time=5.2,
                accepted=True,prior_status='used',position_error_m=.010)


def test_fresh_previously_accepted_response_can_cap_clearance():
    assert run.adaptive_clearance_error(metadata(),{'wall':10.01},10.05,5.03)==.010


@pytest.mark.parametrize('change',[
    {'wall':9.5},{'wall':11.},{'reference_control_time_s':4.8},
    {'reference_control_time_s':5.1},{'accepted':False},
    {'prior_status':'unused'},{'position_error_m':float('nan')},
    {'end_time':4.8},
])
def test_invalid_retained_metadata_rejected_before_adaptive_update(change):
    response=metadata();response.update(change)
    with pytest.raises(RuntimeError):
        run.adaptive_clearance_error(response,{'wall':10.01},10.05,5.03)


@pytest.mark.parametrize('heartbeat',[{'wall':9.5},{'wall':11.},{}])
def test_stale_or_missing_heartbeat_rejected(heartbeat):
    with pytest.raises(RuntimeError):
        run.adaptive_clearance_error(metadata(),heartbeat,10.05,5.03)
