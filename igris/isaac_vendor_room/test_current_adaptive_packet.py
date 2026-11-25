import json
import pytest
import run


def packet():
    return dict(wall=10.,reference_control_time_s=5.,end_time=5.2,
                accepted=True,prior_status='used',position_error_m=.012)


def test_current_published_planning_error_is_read_before_validation(tmp_path,monkeypatch):
    (tmp_path/'live_ik_response.json').write_text(json.dumps(packet()))
    (tmp_path/'live_ik_heartbeat.json').write_text(json.dumps({'wall':10.01}))
    monkeypatch.setattr(run.time,'time',lambda:10.05)
    assert run.current_adaptive_clearance_error(tmp_path,5.03)==.012


@pytest.mark.parametrize('bad',['stale','unaccepted','missing','malformed'])
def test_current_packet_must_qualify_without_a_cached_fallback(tmp_path,monkeypatch,bad):
    response=packet()
    if bad=='stale':response['reference_control_time_s']=4.8
    if bad=='unaccepted':response['accepted']=False
    if bad!='missing':
        (tmp_path/'live_ik_response.json').write_text('{broken' if bad=='malformed' else json.dumps(response))
    (tmp_path/'live_ik_heartbeat.json').write_text(json.dumps({'wall':10.01}))
    monkeypatch.setattr(run.time,'time',lambda:10.05)
    with pytest.raises((RuntimeError,ValueError,FileNotFoundError)):
        run.current_adaptive_clearance_error(tmp_path,5.03)
