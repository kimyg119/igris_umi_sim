import copy
import numpy as np
import pytest
from walking_transfer import WalkingTransfer
from test_walking_transfer import fixture


def test_loaded_carry_stays_at_zero_after_clearance_and_refreshes_targets():
    _, p, t, b, h = fixture()
    ctl = WalkingTransfer([1.16, -.26, .758], -np.pi/4, .1, 4., clearance_lift=.03,
                          heading_preserving_route=True)
    ctl.start(0., p, t, b, h)
    for stamp in np.arange(.02, 9., .02):
        b[2] = .85 + .03*min(stamp/8, 1.)
        s = ctl.update(float(stamp), p, t, b, [6,0,6,0], 0., hold_carry=True)
        assert not np.any(s['command_xyz'])
    assert s['phase'] == 'carry' and s['clearance_ready_s'] >= .5
    before = np.asarray(s['world_hand_targets'])
    t[0] += .01
    s = ctl.update(9.1, p, t, b, [6,0,6,0], 0., hold_carry=True)
    assert s['phase'] == 'carry' and s['navigation_route']['waypoint_index'] == 0
    np.testing.assert_allclose(np.asarray(s['world_hand_targets'])[:,:3,3], before[:,:3,3]+[.01,0,0])
    with pytest.raises(RuntimeError, match='palm'):
        ctl.update(10., p, t, b, [0,0,0,0], 0., hold_carry=True)


def request():
    q=np.zeros(31);q[:3]=[.71,.01,.02]
    return dict(nominal=q.tolist(), measured=q.tolist(),
                htd_loaded_waist=dict(schema='htd_loaded_actual_waist/v1',epoch=1,
                    native_sample_sim=4.,native_time_s=4.001, sample_monotonic=10.,
                    request_monotonic=10.01,waist=q[:3].tolist()))


def test_same_actual_waist_in_publisher_and_solver_without_mutating_archive():
    from htd_loaded_reference import apply_loaded_waist
    q=np.arange(31)/100;old=q.copy();r=request();saved=copy.deepcopy(r)
    out=apply_loaded_waist(q,r,enabled=True)
    np.testing.assert_array_equal(out[:3],r['measured'][:3])
    np.testing.assert_array_equal(out[3:],old[3:])
    np.testing.assert_array_equal(q,old)
    assert r==saved
    np.testing.assert_array_equal(apply_loaded_waist(q,{},enabled=False),q)
    np.testing.assert_array_equal(apply_loaded_waist(q,{},enabled=True),q)


@pytest.mark.parametrize('field,value',[('epoch',0),('epoch',1.5),('native_sample_sim',3.9),
    ('native_time_s',3.9),('request_monotonic',10.2),('waist',[float('nan'),0,0]),
    ('schema','wrong')])
def test_bad_epoch_clock_or_waist_rejected(field,value):
    from htd_loaded_reference import apply_loaded_waist
    r=request();r['htd_loaded_waist'][field]=value
    with pytest.raises(ValueError):apply_loaded_waist(np.zeros(31),r,enabled=True)


def test_waist_requires_opt_in_and_agrees_with_both_nominal_and_measured():
    from htd_loaded_reference import apply_loaded_waist
    r=request()
    with pytest.raises(ValueError):apply_loaded_waist(np.zeros(31),r,enabled=False)
    for key in ('measured','nominal'):
        r=request();r[key][0]+=.01
        with pytest.raises(ValueError):apply_loaded_waist(np.zeros(31),r,enabled=True)
