"""Offline adapter tests: fake ROS boundary, actual frozen actor/profile files."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

import htd_loaded_runtime as runtime
import htd_loaded_setup as setup
from test_htd_candidate_package import package


class Service:
    class Request:
        pass


class Future:
    def __init__(self, success=True): self.success = success
    def done(self): return True
    def result(self): return SimpleNamespace(success=self.success, message='offline acknowledgement')


class Node:
    def __init__(self): self.requests = []
    def get_service_names_and_types(self):
        return [('/igris_c_001/mode', ['igris_c_sdk/srv/ControlModeCommandRequest']),
                ('/igris_c_001/sim', ['igris_c_sdk/srv/MujocoSimCmd'])]
    def create_client(self, kind, name):
        node = self
        class Client:
            def service_is_ready(self): return True
            def call_async(self, req):
                node.requests.append((name, req))
                return Future()
        return Client()


@pytest.fixture(scope='module')
def prepared_trial():
    source = setup.P/'transfer_trial76/run'
    if not source.exists(): pytest.skip('Explicit profile76 local inputs required')
    with tempfile.TemporaryDirectory(prefix='htd_loaded_runtime_test_', dir=setup.ROOT/'outputs') as temporary:
        trial = Path(temporary)
        for name in ('manifest.json', 'robot.yaml', 'scene.xml', 'vendor.xml'):
            shutil.copyfile(source/name, trial/name)
        setup.prepare(trial, enabled=True, environment={})
        (trial/'runtime.json').write_text(json.dumps(dict(network='none', ipc='private',
            hardware_executed=False, simulation_rate_limit=.5)))
        yield trial


@pytest.fixture
def adapter(prepared_trial):
    for name in ('htd_pacing.request', 'htd_pacing.request.ack'):
        (prepared_trial/name).unlink(missing_ok=True)
    clocks = dict(native=100., monotonic=1000., wall=2000.)
    output = {}
    node = Node()
    obj = runtime.LoadedRuntime(prepared_trial, node, services={'mode': Service, 'sim': Service},
        clock=lambda: {'sim': clocks['native']}, output=lambda: output.copy(),
        monotonic=lambda: clocks['monotonic'], wall=lambda: clocks['wall'], spin=lambda: None)
    output.update(sim=100., monotonic=1000., sequence=2, q_output=obj.coordinator.hold_q.copy(),
        dq_output=np.zeros(31), q_encoder=obj.coordinator.hold_q.copy(), dq_encoder=np.zeros(31))
    return obj, clocks, output, node


def arguments(obj):
    q = obj.coordinator.hold_q.copy()
    return dict(state=dict(q=q.copy(), dq=np.zeros(31), status=np.zeros(31),
        imu_quat=obj.mount.copy(), imu_gyro=[0., 0., 0.]), received=999.99,
        mode=6, environment=1, mode_at=999.98, stamp=[1999.97, 100.],
        poses=np.array([[0., 0., .9, 1., 0., 0., 0.]]), pelvis_index=0,
        forces=dict(wall=1999.99, palm_tip_force_N=[1., 0., 1., 0.],
                    source_support_force_world_z_N=0., support_force_groups_configured=True),
        hand_wall=1999.99, hl_response=dict(wall=1999.50, heartbeat_wall=1999.99,
            accepted=True, prior_status='used', grasp_verified=True),
        upper_output_target=q.copy(), walking_state=dict(clearance_measured_rise_m=.04,
                    clearance_ready_s=.5, payload_speed_m_s=.001))


def test_actual_package_constructor_warms_without_low_or_pacing(adapter):
    obj, _, _, node = adapter
    assert obj.coordinator.policy._step_index == 0
    assert obj.coordinator.required_palm_force == pytest.approx(.1839375)
    assert node.requests == []
    assert not obj.pacing_request_path.exists()


def test_runtime_routes_candidate_and_independently_rejects_descriptor_tamper(package):
    directory, descriptor = package
    with tempfile.TemporaryDirectory(prefix='htd_loaded_candidate_runtime_', dir=setup.ROOT/'outputs') as temporary:
        trial = Path(temporary)
        for name in ('manifest.json', 'robot.yaml', 'scene.xml', 'vendor.xml'):
            shutil.copyfile(setup.P/'transfer_trial76/run'/name, trial/name)
        setup.prepare(trial, enabled=True, current_grasp_reference=True, command_observer=True,
            wrist_mode_entry=True, goal_hold=True, policy_package=directory)
        (trial/'runtime.json').write_text(json.dumps(dict(network='none', ipc='private',
            hardware_executed=False, simulation_rate_limit=.5)))
        node = Node()
        obj = runtime.LoadedRuntime(trial, node, services={'mode': Service, 'sim': Service}, spin=lambda: None)
        assert obj.coordinator.policy.policy_path == directory/'policy.ts'
        assert obj.coordinator.policy._step_index == 0 and node.requests == []
        assert obj.preflight['policy_selection']['descriptor_sha256'] == setup.sha(directory/'loaded_candidate.json')
        # Re-pin a changed descriptor: worker must still reject its host-time hash.
        descriptor['historically_trained_on'] += '; changed after setup'
        (directory/'loaded_candidate.json').write_text(json.dumps(descriptor))
        class NoServices:
            def get_service_names_and_types(self): raise AssertionError('No service discovery on invalid selection')
        with pytest.raises(ValueError, match='selection|descriptor'):
            runtime.LoadedRuntime(trial, NoServices(), services={'mode': Service, 'sim': Service})


def test_runtime_cannot_select_other_package_via_paths_without_explicit_selection(prepared_trial, package):
    directory, _ = package
    path = prepared_trial/'manifest.json'; original = path.read_bytes()
    manifest = json.loads(original)
    manifest['htd_loaded_zero'].pop('policy_selection', None)
    manifest['htd_loaded_zero']['policy_path'] = str(directory/'policy.ts')
    path.write_text(json.dumps(manifest))
    try:
        with pytest.raises(ValueError, match='Unexpected loaded runtime path'):
            runtime.LoadedRuntime(prepared_trial, Node(), services={'mode': Service, 'sim': Service})
    finally:
        path.write_bytes(original)


def test_command_observer_arms_once_before_pacing_and_never_changes_policy(adapter):
    import struct
    from test_htd_command_scope import scope
    old, clocks, output, _ = adapter
    path=old.out/'manifest.json';original=path.read_bytes()
    manifest=json.loads(original)
    observed=scope(old.out)
    manifest['htd_loaded_zero']['command_observer']=observed
    path.write_text(json.dumps(manifest))
    try:
        node=Node()
        obj=runtime.LoadedRuntime(old.out,node,services={'mode':Service,'sim':Service},
            clock=lambda:{'sim':clocks['native']},output=lambda:output.copy(),
            monotonic=lambda:clocks['monotonic'],wall=lambda:clocks['wall'],spin=lambda:None)
        arm=Path(observed['arm_path'])
        assert not arm.exists()
        assert obj.request_pacing() is False
        assert arm.exists() and not obj.pacing_request_path.exists()
        before=arm.stat().st_mtime_ns
        Path(observed['trace_path']).write_bytes(struct.pack('<8sIIIIQQQQQ',b'HTDCMD01',1,64,8064,4096,0,0,0,1000_000_000_000,123))
        assert obj.request_pacing() is False and arm.stat().st_mtime_ns==before
        assert obj.pacing_request_path.exists()
        assert [e['event'] for e in obj.events][:3]==['command_observer_armed','command_observer_ready','pacing_requested']
        assert node.requests==[] and obj.coordinator.policy._step_index==0
    finally:
        path.write_bytes(original)
        for key in ('library','arm_path','trace_path'):Path(observed[key]).unlink(missing_ok=True)


@pytest.mark.parametrize('goal,grace',[(False,False),(True,False),(True,True)])
def test_observed_wrist_entry_primes_before_low_without_rebasing_motor_hold(adapter,goal,grace):
    import struct
    from htd_wrist_entry import WristModeEntry
    from test_htd_command_scope import scope
    old,clocks,output,_=adapter
    path=old.out/'manifest.json';original=path.read_bytes();manifest=json.loads(original)
    observed=scope(old.out)
    manifest['htd_loaded_zero'].update(command_observer=observed,current_grasp_reference=True,
        wrist_mode_entry=WristModeEntry().contract)
    if goal:
        from htd_goal_hold import GoalHold
        manifest['htd_loaded_zero'].update(scope='loaded_goal_hold_only',goal_hold=GoalHold().contract,
            grasp_contact_rule='per-hand palm + tip')
    if grace:
        from htd_grasp_retention import GraspRetention
        manifest['htd_loaded_zero'].update(grasp_retention=GraspRetention(.1,4.).contract,
            grasp_contact_rule='per-hand palm+tip, with explicit force/pose guarded100ms grace')
    path.write_text(json.dumps(manifest))
    try:
        node=Node()
        obj=runtime.LoadedRuntime(old.out,node,services={'mode':Service,'sim':Service},
            clock=lambda:{'sim':clocks['native']},output=lambda:output.copy(),
            monotonic=lambda:clocks['monotonic'],wall=lambda:clocks['wall'],spin=lambda:None)
        raw=obj.coordinator.hold_q.copy();raw[[20,21,27,28]]=[.1,-.8,-.1,-.7]
        output.update(q_motor=raw.copy(),dq_motor=np.zeros(31))
        obj.request_pacing()
        Path(observed['trace_path']).write_bytes(struct.pack('<8sIIIIQQQQQ',b'HTDCMD01',1,64,8064,4096,0,0,0,1000_000_000_000,123))
        obj.request_pacing()
        obj.pacing_ack_path.write_text(json.dumps(dict(previous_rate=.5,rate=.1,native_time_s=100.,monotonic_s=1000.)))
        obj.poll_pacing()
        args=arguments(obj);args["torso_index"]=1
        args['poses']=np.array([[.2,.3,.9,1.,0.,0.,0.],
                               [.4,.5,1.,np.cos(.2),0.,0.,np.sin(.2)],
                               [.65,.3,.9,1.,0.,0.,0.],
                               [.65,.4,.85,1.,0.,0.,0.],[.65,.2,.85,1.,0.,0.,0.]])
        args.update(payload_index=2,hand_indices=[3,4]);args['forces']['sim']=100.
        s=obj.snapshot(**args)
        command=dict(q=obj.coordinator.hold_q.copy(),dq=np.zeros(31),tau=np.zeros(31),
            kp=np.full(31,400.),kd=np.full(31,2.))
        held=obj.entry_command(s,command)
        np.testing.assert_array_equal(held['q'][[20,21,27,28]],raw[[20,21,27,28]])
        np.testing.assert_array_equal(held['kinematic_modes'],[1,1,1,0,0])
        with pytest.raises(RuntimeError,match='prim'):
            obj.request_low(s)
        assert node.requests==[] and obj.coordinator.state=='prepared'
        clocks.update(native=100.06,monotonic=1000.6,wall=2000.6)
        output.update(sim=100.06,monotonic=1000.6,q_motor=raw+.001)
        s2=replace(s,native_time=100.06,pose_native_time=100.06,force_native_time=100.06,wall_time=1000.6,output=output.copy())
        held2=obj.entry_command(s2,command)
        np.testing.assert_array_equal(held2['q'][[20,21,27,28]],raw[[20,21,27,28]])
        obj.request_low(s2)
        assert len(node.requests)==1 and node.requests[0][1].command_type==5
        assert obj.coordinator.policy._step_index==0
        if goal:
            assert obj.coordinator.goal_hold.metrics['pose_native_time']==100.06
            np.testing.assert_array_equal(s.torso_pose,args['poses'][1])
            assert obj.coordinator.goal_hold.metrics['heading_goal_rad']==0.
            active=replace(s2,mode=3)
            actor=obj.coordinator.advance_actor(active)
            obj.record_actor(actor,active)
            assert obj.actor_records[0]['goal_hold_pose_native_time']==100.06
            result=obj.pause_cleanup('offline goal hold boundary',held2,lambda c:None)
            assert result['goal_hold_metrics']['goal_xy_world']==[.4,.5]
            assert not result['bounded_zero_completed'] and result['pause_acknowledged']
            if grace:
                np.testing.assert_array_equal(s.payload_pose,args['poses'][2])
                np.testing.assert_array_equal(s.hand_poses,args['poses'][[3,4]])
                assert result['grasp_retention_events'][0]['event']=='captured_actual_grasp'
                assert result['grasp_retention_metrics']['both_hands_qualified']
    finally:
        path.write_bytes(original)
        for key in ('library','arm_path','trace_path'):Path(observed[key]).unlink(missing_ok=True)


def test_manifest_current_grasp_scope_reaches_low_with_original_weights_and_saves_difference(adapter):
    old, clocks, output, _ = adapter
    path = old.out/'manifest.json'
    original = path.read_bytes()
    manifest = json.loads(original)
    manifest['walking_transfer']['clearance_minimum_rise_m'] = .02
    manifest['htd_loaded_zero'].update(current_grasp_reference=True,
        clearance_minimum_rise_m=.02, runtime_domain_validated=False)
    path.write_text(json.dumps(manifest))
    try:
        node = Node()
        obj = runtime.LoadedRuntime(old.out, node, services={'mode': Service, 'sim': Service},
            clock=lambda: {'sim': clocks['native']}, output=lambda: output.copy(),
            monotonic=lambda: clocks['monotonic'], wall=lambda: clocks['wall'], spin=lambda: None)
        args = arguments(obj)
        args['upper_output_target'][[15, 17]] += [.62, .79]
        output['q_output'][15:29] = args['upper_output_target'][15:29]
        output['q_encoder'][15:29] = args['upper_output_target'][15:29]
        args['walking_state']['clearance_measured_rise_m'] = .021
        obj.request_pacing()
        obj.pacing_ack_path.write_text(json.dumps(dict(previous_rate=.5, rate=.1,
            native_time_s=100., monotonic_s=1000.)))
        obj.poll_pacing()
        obj.request_low(obj.snapshot(**args))
        assert [req.command_type for name, req in node.requests if name.endswith('/mode')] == [5]
        assert obj.preflight['current_grasp_reference'] is True
        assert obj.coordinator.current_grasp_evidence['max_abs_delta_from_frozen_profile_rad'] == pytest.approx(.79)
        assert obj.preflight['artifacts'] == old.preflight['artifacts']
        command = {k: np.zeros(31) for k in ('q','dq','kp','kd','tau')}
        obj.pause_cleanup('offline boundary only', command, lambda x: None)
        saved = json.loads((obj.out/'htd_loaded_result.json').read_text())
        assert saved['coordinator_contract']['runtime_domain_validated'] is False
        assert saved['current_grasp_reference_evidence']['max_abs_delta_from_frozen_profile_rad'] == pytest.approx(.79)
    finally:
        path.write_bytes(original)


def test_snapshot_uses_measured_native_output_and_documented_mounted_imu(adapter):
    obj, _, output, _ = adapter
    args = arguments(obj)
    args['state']['q'] = np.zeros(31)
    s = obj.snapshot(**args)
    np.testing.assert_array_equal(s.output['q_output'], output['q_output'])
    np.testing.assert_allclose(s.torso_quat_wxyz, [1., 0., 0., 0.], atol=1e-7)
    assert s.state_age_s == pytest.approx(.01)
    assert s.pose_age_s == pytest.approx(.03)
    assert s.hl_age_s == pytest.approx(.01)  # Native30Hz response may be0.5wallsec old.
    assert s.pacing_ack is None
    del args['state']['status']
    with pytest.raises(ValueError, match='status'):
        obj.snapshot(**args)


def test_policy_deadline_crossing_between_actor_and_pd_reads_runs_due_actor(adapter):
    obj,clocks,output,_=adapter
    obj.request_pacing()
    obj.pacing_ack_path.write_text(json.dumps(dict(previous_rate=.5,rate=.1,native_time_s=100.,monotonic_s=1000.)))
    obj.poll_pacing()
    start=obj.snapshot(**arguments(obj));obj.request_low(start)
    sequence=[0]
    def sample(dt):
        sequence[0]+=2
        clocks.update(native=100.+dt,monotonic=1000.+10*dt,wall=2000.+10*dt)
        output.update(sim=clocks['native'],monotonic=clocks['monotonic'],sequence=sequence[0])
        return replace(start,mode=3,native_time=clocks['native'],wall_time=clocks['monotonic'],output=output.copy())
    command={key:np.zeros(31) for key in ('q','dq','kp','kd','tau')}
    for k in range(4):
        s=sample(.005*k)
        obj.coordinator.advance_actor(s)
        frame=obj.coordinator.compute_pd(s,command)
        obj.coordinator.ack_published(frame,s)
    assert obj.coordinator.actor_count==1
    before=sample(1/60.)  # Last300Hz output before the20ms actor deadline.
    assert obj.coordinator.advance_actor(before) is None
    after=sample(.020)    # New output arrives between the two independent reads.
    with pytest.raises(RuntimeError,match='Due actor'):
        obj.coordinator.compute_pd(after,command)
    ready=obj.prepare_pd_snapshot(after)
    assert obj.coordinator.actor_count==2 and len(obj.actor_records)==1
    frame=obj.coordinator.compute_pd(ready,command)
    obj.coordinator.check_publication(frame,ready)
    obj.coordinator.ack_published(frame,ready)
    assert obj.coordinator.published_count==5


def test_snapshot_requires_measured_speed_and_validated_hl_support(adapter):
    obj, _, _, _ = adapter
    args = arguments(obj)
    del args['walking_state']['payload_speed_m_s']
    with pytest.raises(ValueError, match='payload_speed'):
        obj.snapshot(**args)
    args = arguments(obj)
    args['hl_response']['accepted'] = False
    with pytest.raises(ValueError, match='HL'):
        obj.snapshot(**args)
    args = arguments(obj)
    del args['hl_response']['heartbeat_wall']
    with pytest.raises(ValueError, match='heartbeat'):
        obj.snapshot(**args)


def test_fresh_updates_all_ages_after_clock_io_without_reading_json_or_outputs(adapter, monkeypatch):
    obj, clocks, _, _ = adapter
    s = obj.snapshot(**arguments(obj))
    def clock_after_io():
        clocks['monotonic'] += .01
        return {'sim': 100.001}
    obj._clock = clock_after_io
    obj._output = lambda: pytest.fail('Fresh must not reread the output after PD')
    monkeypatch.setattr(json, 'loads', lambda *a, **k: pytest.fail('No JSON after PD'))
    new = obj.fresh(s)
    assert new.wall_time == 1000.01 and new.native_time == 100.001
    assert new.output is s.output
    for field in ('state_age_s','mode_age_s','pose_age_s','force_age_s','hand_age_s','hl_age_s'):
        assert getattr(new, field) == pytest.approx(getattr(s, field)+.01)


def test_only_native_ack_enables_low_and_only_low_mode_service_is_requested(adapter):
    obj, clocks, _, node = adapter
    assert obj.request_pacing() is False
    before = obj.pacing_request_path.stat().st_mtime_ns
    assert obj.request_pacing() is False
    assert obj.pacing_request_path.stat().st_mtime_ns == before
    assert obj.poll_pacing() is None
    with pytest.raises(RuntimeError, match='pacing'):
        obj.request_low(obj.snapshot(**arguments(obj)))
    obj.pacing_ack_path.write_text(json.dumps(dict(previous_rate=.5, rate=.1,
        native_time_s=100., monotonic_s=1000.)))
    assert obj.poll_pacing()['rate'] == .1
    assert obj.request_pacing() is True
    obj.request_low(obj.snapshot(**arguments(obj)))
    assert obj.low_ready()
    assert len(node.requests) == 1
    name, request = node.requests[0]
    assert name.endswith('/mode') and request.command_type == 5
    assert request.preset_id == '' and request.is_cyclic is False
    with pytest.raises(RuntimeError): obj.request_low(obj.snapshot(**arguments(obj)))


def test_actual_waist_request_matches_same_sample_for_nominal_and_measured(adapter):
    from htd_loaded_reference import apply_loaded_waist
    obj, _, output, _ = adapter
    obj.coordinator.epoch = 1
    spec, q = obj.waist_request()
    assert spec['waist'] == q[:3].tolist()
    np.testing.assert_array_equal(q, output['q_output'])
    request = dict(htd_loaded_waist=spec, nominal=q.tolist(), measured=q.tolist())
    np.testing.assert_array_equal(apply_loaded_waist(np.zeros(31), request, enabled=True)[:3], q[:3])


def test_cleanup_zeroes_only_owned_fields_then_pauses_without_reset_or_grip_release(adapter):
    obj, _, _, node = adapter
    q = obj.coordinator.hold_q.copy()
    command = dict(q=q, dq=np.ones(31), kp=np.ones(31)*20, kd=np.ones(31), tau=np.ones(31))
    publications = []
    obj.note_environment(1)
    result = obj.pause_cleanup('offline failure test', command, publications.append)
    assert result['pause_acknowledged'] and result['hold_grip'] and not result['bounded_zero_completed']
    assert len(publications) == 1
    for field in ('dq','kp','kd','tau'):
        np.testing.assert_array_equal(publications[0][field][:15], 0.)
        np.testing.assert_array_equal(publications[0][field][15:], command[field][15:])
    assert [req.command_type for _, req in node.requests] == [0]
    assert (obj.out/'htd_loaded_result.json').exists()


def test_pending_service_future_timeout_is_guarded_even_before_low_mode_ack(adapter):
    obj, clocks, _, _ = adapter
    obj.request_pacing()
    obj.pacing_ack_path.write_text(json.dumps(dict(previous_rate=.5, rate=.1, native_time_s=100., monotonic_s=1000.)))
    assert obj.request_pacing()
    obj.request_low(obj.snapshot(**arguments(obj)))
    obj._low_future = SimpleNamespace(done=lambda: False)
    clocks['monotonic'] += .51
    with pytest.raises(RuntimeError, match='transition'):
        obj.low_ready()


def test_runtime_recording_matches_root_single_writer_call_signature(adapter):
    obj, _, _, _ = adapter
    obj.request_pacing()
    obj.pacing_ack_path.write_text(json.dumps(dict(previous_rate=.5, rate=.1, native_time_s=100., monotonic_s=1000.)))
    obj.request_pacing()
    s = obj.snapshot(**arguments(obj))
    obj.request_low(s)
    s = replace(s, mode=3)
    actor = obj.coordinator.advance_actor(s)
    obj.record_actor(actor, s)
    q = obj.coordinator.hold_q
    command = dict(q=q, dq=np.zeros(31), kp=np.ones(31), kd=np.ones(31), tau=np.zeros(31))
    frame = obj.coordinator.compute_pd(s, command)
    obj.coordinator.ack_published(frame, s)
    obj.record_pd(frame, s, s)
    assert len(obj.actor_records) == len(obj.pd_records) == 1
    result = obj.pause_cleanup('offline recorded segment', command, lambda c: None)
    assert result['pause_acknowledged']
    with np.load(obj.out/'htd_loaded_pd.npz') as data:
        assert data['tau'].shape == (1,31)
        np.testing.assert_array_equal(data['published_tau'][0], frame.command['tau'])


@pytest.mark.parametrize('problem', ['publication', 'pause_rejected'])
def test_cleanup_failure_is_saved_and_raised_even_if_other_cleanup_step_succeeds(adapter, problem):
    obj, _, _, node = adapter
    obj.note_environment(1)
    command = dict(q=obj.coordinator.hold_q, dq=np.zeros(31), kp=np.ones(31), kd=np.ones(31), tau=np.zeros(31))
    def publish(command):
        if problem == 'publication': raise RuntimeError('offline publisher failure')
    if problem == 'pause_rejected':
        obj._clients['sim'] = SimpleNamespace(call_async=lambda request: Future(False))
    with pytest.raises(RuntimeError, match='cleanup'):
        obj.pause_cleanup('test cleanup failure', command, publish)
    saved = json.loads((obj.out/'htd_loaded_result.json').read_text())
    assert saved['cleanup_errors']
    assert saved['pause_acknowledged'] is (problem == 'publication')


@pytest.mark.parametrize('graph', ['delayed', 'not_ready', 'missing', 'duplicate'])
def test_constructor_polls_service_graph_boundedly_without_requesting_motion(prepared_trial, graph):
    for name in ('htd_pacing.request','htd_pacing.request.ack'):
        (prepared_trial/name).unlink(missing_ok=True)
    now = [1000.]
    class GraphNode(Node):
        def get_service_names_and_types(self):
            available = super().get_service_names_and_types()
            if graph == 'duplicate': return available+[('/igris_c_001/second_mode', available[0][1])]
            if graph == 'missing' or now[0] < 1000.2: return []
            return available
        def create_client(self, kind, name):
            client = super().create_client(kind, name)
            if graph == 'not_ready': client.service_is_ready = lambda: now[0] >= 1000.4
            return client
    node = GraphNode()
    def spin(): now[0] += .05
    def construct():
        return runtime.LoadedRuntime(prepared_trial, node, services={'mode':Service,'sim':Service},
            clock=lambda: {'sim':100.}, output=lambda: {}, monotonic=lambda:now[0], wall=lambda:2000., spin=spin)
    if graph in ('delayed','not_ready'):
        construct()
        assert .2 <= now[0]-1000. < 2.
    else:
        with pytest.raises(RuntimeError, match='service'):
            construct()
        assert now[0]-1000. <= 2.1
        if graph == 'missing': assert now[0]-1000. >= 2.
    assert node.requests == []


def test_complete_public_runtime_loop_at100hz_wall_preserves_upper_and_publishes2000_pd_ticks(adapter):
    """Functional protocol replay, not physics: real actor and caps, fake ROS IO."""
    obj, clocks, output, node = adapter
    obj.note_environment(1)
    obj.warm_observer()
    assert not obj.request_pacing()
    obj.pacing_ack_path.write_text(json.dumps(dict(previous_rate=.5, rate=.1,
                                                  native_time_s=100., monotonic_s=1000.)))
    assert obj.request_pacing()
    args = arguments(obj)
    q = obj.coordinator.hold_q.copy()
    upper = dict(q=q.copy(), dq=np.zeros(31), kp=np.ones(31)*20., kd=np.ones(31), tau=np.zeros(31))
    published_count = 0
    for i in range(10001):
        clocks.update(native=100.+i/1000., monotonic=1000.+i/100., wall=2000.+i/100.)
        observer_tick = int(np.floor((i+1e-8)*.3))
        sample_sim = 100.+observer_tick/300.
        output.update(sim=sample_sim, monotonic=1000.+observer_tick/30., sequence=2*(observer_tick+1))
        args.update(received=clocks['monotonic']-.001, mode_at=clocks['monotonic']-.001,
                    mode=6 if i == 0 else 3, stamp=[clocks['wall']-.01, clocks['native']],
                    hand_wall=clocks['wall']-.01)
        args['forces']['wall'] = args['hl_response']['heartbeat_wall'] = clocks['wall']-.01
        upper['dq'][15:] = .01*np.sin(i*.01)
        upper['tau'][15:] = .05*np.cos(i*.01)
        s = obj.snapshot(**args)
        if obj.coordinator.state == 'prepared':
            obj.request_low(obj.refresh_output(s))
        assert obj.low_ready()
        actor_snapshot = obj.refresh_output(s)
        actor = obj.coordinator.advance_actor(actor_snapshot)
        if actor is not None: obj.record_actor(actor, actor_snapshot)
        if obj.coordinator.state == 'complete': break
        if obj.coordinator.state != 'active': continue
        pd_snapshot = obj.refresh_output(s)
        frame = obj.coordinator.compute_pd(pd_snapshot, upper)
        if frame is not None:
            obj.coordinator.check_publication(frame, obj.fresh(pd_snapshot))
            for key in upper:
                np.testing.assert_array_equal(frame.command[key][15:], upper[key][15:])
            published_count += 1  # sole fake publisher boundary
            after = obj.fresh(pd_snapshot)
            obj.coordinator.ack_published(frame, after)
            obj.record_pd(frame, pd_snapshot, after)
    assert obj.coordinator.state == 'complete'
    assert len(obj.actor_records) == 500
    assert published_count == len(obj.pd_records) == 2000
    assert obj.coordinator.policy._gait_clock.item() == 0.
    cleanup = []
    result = obj.pause_cleanup('complete functional protocol replay', upper, cleanup.append)
    assert result['bounded_zero_completed'] and result['pause_acknowledged']
    assert [request.command_type for _,request in node.requests] == [5, 0]
    assert len(cleanup) == 1 and not np.any(cleanup[0]['tau'][:15])


def test_postpublish_clock_io_failure_records_sent_tick_without_inventing_timestamps(adapter):
    obj, _, _, _ = adapter
    obj.request_pacing()
    obj.pacing_ack_path.write_text(json.dumps(dict(previous_rate=.5, rate=.1, native_time_s=100., monotonic_s=1000.)))
    obj.request_pacing()
    s = obj.snapshot(**arguments(obj))
    obj.request_low(s)
    s = replace(s, mode=3)
    obj.coordinator.advance_actor(s)
    upper = dict(q=obj.coordinator.hold_q.copy(), dq=np.zeros(31), kp=np.ones(31), kd=np.ones(31), tau=np.zeros(31))
    sent = {k: v.copy() for k,v in upper.items()}
    sent['q'][18] += .1
    frame = obj.coordinator.compute_pd(s, sent)
    obj.coordinator.remember_published(frame)
    obj._clock = lambda: (_ for _ in ()).throw(RuntimeError('offline incomplete native clock'))
    published = None
    with pytest.raises(RuntimeError, match='incomplete'):
        try:
            published = obj.fresh(s)
            obj.coordinator.ack_published(frame, published)
        finally:
            obj.record_pd(frame, s, published)
    cleanup = []
    obj.pause_cleanup('postpublish clock read failure', upper, cleanup.append)
    np.testing.assert_array_equal(cleanup[0]['q'][15:], sent['q'][15:])
    with np.load(obj.out/'htd_loaded_pd.npz') as data:
        assert data['publication_clock_available'].tolist() == [False]
        assert np.isnan(data['publication_native_time'][0])
        np.testing.assert_array_equal(data['published_q'][0], frame.command['q'])
