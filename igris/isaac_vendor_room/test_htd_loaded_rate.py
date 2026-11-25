"""Selected SIM pacing, with actual policy/config and fake ROS; no physics."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pytest

import htd_loaded_setup as host
from htd_loaded import LoadedZero
from htd_loaded_runtime import LoadedRuntime
from test_htd_loaded import setup, snapshot
from test_htd_loaded_transport import coordinator, sample
from test_htd_loaded_runtime import Node, Service


OPTIONS = dict(enabled=True, current_grasp_reference=True, command_observer=True,
               wrist_mode_entry=True, goal_hold=True, contact_grace=True, transport=True)


@pytest.mark.parametrize('rate', [True, False, None, '.05', float('nan'), float('inf'), 0., .2, .0500001])
def test_bad_rates_rejected_before_host_io_and_coordinator_use(rate):
    with pytest.raises(ValueError, match='rate|pacing'):
        host.prepare('/does/not/exist', loaded_rate=rate)
    with pytest.raises(ValueError, match='rate|pacing'):
        LoadedZero(None, None, None, mass_kg=.1, friction=4., clearance_lift_m=.04,
                   loaded_rate=rate)


@pytest.mark.parametrize('options', [{}, dict(enabled=True),
    dict(enabled=True,current_grasp_reference=True),
    {**OPTIONS,'transport':False}])
def test_slow_rate_requires_transport_before_io(options):
    with pytest.raises(ValueError, match='transport|Transport'):
        host.prepare('/does/not/exist', loaded_rate=.05, **options)


def test_default_remains_noop_and_original_handoff():
    assert host.prepare('/does/not/exist') is None
    c,p,_=setup()
    assert c.loaded_rate == .1
    c.check_handoff_ready(snapshot(p))
    with pytest.raises(RuntimeError,match='pacing'):
        c.check_handoff_ready(replace(snapshot(p),pacing_ack={**snapshot(p).pacing_ack,'rate':.05}))


def test_slow_coordinator_requires_selected_ack_without_changing_native_contract():
    c,p,_=coordinator(loaded_rate=.05)
    assert c.loaded_rate==.05 and p.dt==.02 and c.serial_pd.dt==.005
    with pytest.raises(RuntimeError,match='pacing'):
        c.check_handoff_ready(sample(p,mode=6))
    s=sample(p,mode=6)
    s=replace(s,pacing_ack={**s.pacing_ack,'rate':.05})
    c.check_handoff_ready(s)
    assert c.state=='prepared' and p._step_index==0
    # Pacing cannot relax the fresh output requirement.
    stale=replace(s,native_time=s.native_time+.0021)
    with pytest.raises(RuntimeError): c.check_handoff_ready(stale)


def test_slow_coordinator_rejects_nontransport():
    with pytest.raises(ValueError,match='transport'):
        LoadedZero(None,None,None,mass_kg=.1,friction=4.,clearance_lift_m=.04,
                   current_grasp_reference=True,loaded_rate=.05)


@pytest.fixture(scope='module')
def selected_trial():
    with tempfile.TemporaryDirectory(prefix='htd_rate_test_',dir=host.ROOT/'outputs') as directory:
        out=Path(directory)
        for name in ('manifest.json','robot.yaml','scene.xml','vendor.xml'):
            shutil.copyfile(host.P/'transfer_trial76/run'/name,out/name)
        path=out/'manifest.json';manifest=json.loads(path.read_text())
        manifest['walking_transfer'].update(heading_preserving_route=True,xy_first_lowering=True,
            world_position_lowering=True,torso_relative_lowering=True)
        manifest['evaluation_profile']='strict'
        path.write_text(json.dumps(manifest))
        before={p.name:p.read_bytes() for p in out.iterdir()}
        with pytest.raises(ValueError,match='relaxed'):
            host.prepare(out,loaded_rate=.05,**OPTIONS)
        assert before=={p.name:p.read_bytes() for p in out.iterdir()}
        manifest['evaluation_profile']='relaxed_transport'
        path.write_text(json.dumps(manifest))
        evidence=host.prepare(out,loaded_rate=.05,**OPTIONS)
        assert evidence['manifest_scope']['loaded_rate']==.05
        assert evidence['native_start_environment_required']['IGRIS_ROOM_SLOW_REALTIME_FACTOR']=='.05'
        (out/'runtime.json').write_text(json.dumps(dict(network='none',ipc='private',
            hardware_executed=False,simulation_rate_limit=.5)))
        yield out


def make_runtime(out):
    node=Node()
    obj=LoadedRuntime(out,node,services={'mode':Service,'sim':Service},spin=lambda:None,
        monotonic=lambda:1000.,clock=lambda:dict(sim=100.))
    assert not node.requests
    return obj


def test_host_worker_selected_rate_and_exact_ack(selected_trial):
    obj=make_runtime(selected_trial)
    assert obj.loaded_rate==obj.coordinator.loaded_rate==.05
    assert obj.coordinator.policy._step_index==0
    obj._pacing_requested_at=999.
    ack=dict(rate=.1,previous_rate=.5,monotonic_s=999.5,native_time_s=99.)
    try:
        obj.pacing_ack_path.write_text(json.dumps(ack))
        with pytest.raises(RuntimeError,match='pacing'): obj.poll_pacing()
        assert obj._pacing_ack is None and not obj.events
        ack['rate']=.05
        obj.pacing_ack_path.write_text(json.dumps(ack))
        assert obj.poll_pacing()==ack
        assert obj.events[-1]['acknowledgement']==ack
    finally: obj.pacing_ack_path.unlink(missing_ok=True)


@pytest.mark.parametrize('change', ['rate_bool','rate_missing','rate_invalid','unrelaxed','noncurrent','nontransport'])
def test_worker_independently_rejects_rate_and_scope_before_services(selected_trial,change):
    path=selected_trial/'manifest.json';original=path.read_bytes();m=json.loads(original)
    s=m['htd_loaded_zero']
    if change=='rate_bool': s['loaded_rate']=True
    if change=='rate_missing': s.pop('loaded_rate')
    if change=='rate_invalid': s['loaded_rate']=.2
    if change=='unrelaxed': m['evaluation_profile']='strict'
    if change=='noncurrent': s['current_grasp_reference']=False
    if change=='nontransport':
        s.pop('transport');s['scope']='loaded_goal_hold_only';s['duration_s']=10.
    path.write_text(json.dumps(m))
    class NoServices:
        def get_service_names_and_types(self): raise AssertionError('Validation must precede ROS')
    try:
        with pytest.raises(ValueError):
            LoadedRuntime(selected_trial,NoServices(),services={'mode':Service,'sim':Service},spin=lambda:None)
    finally:path.write_bytes(original)
