"""Offline loaded arbitration tests: real actor/serial PD, no ROS or physics."""
from dataclasses import replace
import copy

import numpy as np
import pytest

from htd_loaded import LoadedSnapshot, LoadedZero, scoped_gc_configuration
from htd_native_probe import TargetLimits, load_gains
from htd_policy import HTDPolicy
from htd_serial_pd import SerialPD


def setup(**scope):
    policy = HTDPolicy()
    kp, kd, _ = load_gains()
    pd = SerialPD(policy.default_q, kp, kd, TargetLimits(), delay_steps=(0, 0, 0, 0))
    coordinator = LoadedZero(policy, pd, policy.default_q, mass_kg=.1, friction=.8,
                             clearance_lift_m=.06, **scope)
    command = dict(q=policy.default_q.copy(), dq=np.arange(31)*.001,
                   kp=kp.copy(), kd=kd.copy(), tau=np.arange(31)*.01)
    return coordinator, policy, command


def snapshot(policy, sim=1., mode=6, sequence=2):
    output = dict(sim=sim, monotonic=10.+10*(sim-1.), sequence=sequence,
                  q_output=policy.default_q.copy(), dq_output=np.zeros(31),
                  q_encoder=policy.default_q.copy(), dq_encoder=np.zeros(31))
    return LoadedSnapshot(native_time=sim, wall_time=output['monotonic'], environment=1,
        mode=mode, state_age_s=.001, mode_age_s=.001, joint_status=np.zeros(31),
        output=output, torso_quat_wxyz=np.array([1., 0., 0., 0.]), torso_gyro=np.zeros(3),
        pelvis_pose=np.array([0., 0., .9, 1., 0., 0., 0.]), pose_age_s=.01,
        force_age_s=.01, hand_age_s=.01, hl_age_s=.01, grasp_verified=True,
        clearance_rise_m=.06, clearance_ready_s=.5, payload_speed_m_s=.005,
        source_support_z_N=0., palm_tip_force_N=np.full(4, 2.),
        upper_output_target=policy.default_q.copy(),
        pacing_ack=dict(previous_rate=.5, rate=.1, native_time_s=1., monotonic_s=10.))


def activate(coordinator, policy):
    request = coordinator.request_handoff(snapshot(policy))
    assert request == dict(mode_command_type=5, preset_id='', is_cyclic=False)
    assert coordinator.advance_actor(snapshot(policy)) is None
    return coordinator.advance_actor(snapshot(policy, mode=3))


def test_gc_change_is_trial_local_exact_lower15_mask_and_preserves_upper16():
    params = {'igris_c_param_server': {'ros__parameters': {
        'gravity_comp_gain': np.arange(31).tolist(), 'gravity_comp_offset': np.arange(31).tolist(),
        'torque_max': [12]*31}}}
    robot = {'igris_c': {'controller': {'GravityCompensationModule': {'enabled': True}}}}
    before = copy.deepcopy((params, robot))
    p, r = scoped_gc_configuration(params, robot)
    assert (params, robot) == before
    values = p['igris_c_param_server']['ros__parameters']
    for field in ('gravity_comp_gain', 'gravity_comp_offset'):
        assert values[field][:15] == [0.]*15
        assert values[field][15:] == before[0]['igris_c_param_server']['ros__parameters'][field][15:]
    assert values['torque_max'] == [12]*31
    assert r == robot
    robot['igris_c']['controller']['GravityCompensationModule']['enabled'] = False
    with pytest.raises(ValueError, match='upper.*GC'):
        scoped_gc_configuration(params, robot)


@pytest.mark.parametrize('change', [dict(environment=0), dict(grasp_verified=False),
    dict(clearance_rise_m=.02), dict(clearance_ready_s=.49), dict(payload_speed_m_s=.021),
    dict(source_support_z_N=.25), dict(palm_tip_force_N=np.zeros(4)), dict(hand_age_s=.25),
    dict(hl_age_s=.25), dict(force_age_s=.25), dict(state_age_s=.1)])
def test_no_low_request_without_measured_live_loaded_readiness(change):
    c, p, _ = setup()
    with pytest.raises((ValueError, RuntimeError)):
        c.request_handoff(replace(snapshot(p), **change))
    assert c.state == 'prepared'


def test_only_owned15_are_merged_and_actual_waist_goes_to_hl():
    c, p, upper = setup()
    first = activate(c, p)
    assert first is not None
    np.testing.assert_allclose(first.observation[12:26], [0., 0., 0., .94, 0., 0., 0.]*2)
    np.testing.assert_array_equal(first.observation[86:120], 0.)
    s = snapshot(p, mode=3)
    s.output['q_output'][0] = s.output['q_encoder'][0] = .012
    before = copy.deepcopy(upper)
    frame = c.compute_pd(s, upper)
    for field in upper:
        np.testing.assert_array_equal(frame.command[field][15:], upper[field][15:])
        np.testing.assert_array_equal(upper[field], before[field])
    for field in ('kp', 'kd', 'dq'):
        np.testing.assert_array_equal(frame.command[field][:15], 0.)
    np.testing.assert_array_equal(frame.command['tau'][:15], frame.pd.tau[:15])
    np.testing.assert_array_equal(frame.measured_waist, s.output['q_output'][:3])
    assert frame.command['tau'].flags.writeable is False
    c.check_publication(frame, s)
    c.ack_published(frame, s)
    with pytest.raises(RuntimeError, match='published'):
        c.ack_published(frame, s)


def test_mixed_wrist_modes_survive_owned_pd_and_abort_unchanged():
    c,p,upper=setup()
    upper['kinematic_modes']=np.array([1,1,1,0,0])
    activate(c,p)
    s=snapshot(p,mode=3)
    frame=c.compute_pd(s,upper)
    np.testing.assert_array_equal(frame.command['kinematic_modes'],[1,1,1,0,0])
    c.remember_published(frame)
    cleanup=c.abort('test-only pause',upper)
    np.testing.assert_array_equal(cleanup.command['kinematic_modes'],[1,1,1,0,0])
    np.testing.assert_array_equal(cleanup.command['q'][15:],frame.command['q'][15:])
    for key in ('kp','kd','dq','tau'):np.testing.assert_array_equal(cleanup.command[key][:15],0.)


def test_loaded_modes_cannot_change_owned_parallel_groups():
    c,p,upper=setup();activate(c,p)
    upper['kinematic_modes']=np.array([0,1,1,0,0])
    with pytest.raises(ValueError,match='kinematic'):
        c.compute_pd(snapshot(p,mode=3),upper)


def test_native300hz_sampling_runs_actor50_pd200_and_exact_ten_seconds():
    c, p, upper = setup()
    activate(c, p)
    actors = 1
    records = []
    for n in range(3001):
        s = snapshot(p, 1.+n/300., mode=3, sequence=2*(n+1))
        if n:
            actors += c.advance_actor(s) is not None
        frame = c.compute_pd(s, upper)
        if frame is not None:
            c.check_publication(frame, s)
            c.ack_published(frame, s)
            records.append(frame.pd)
    assert c.state == 'complete'
    assert actors == 500
    assert len(records) == 2000
    np.testing.assert_allclose([r.scheduled_time for r in records], 1.+.005*np.arange(2000))
    assert max(r.lag_s for r in records) < .001667
    assert c.policy._previous_raw_action.shape == (15,)


@pytest.mark.parametrize('failure', ['stale', 'missed', 'bias', 'fell', 'tilt', 'upper', 'lost_palm', 'mode'])
def test_failure_cleanup_preserves_upper_and_requires_pause_without_grip_release(failure):
    c, p, upper = setup()
    activate(c, p)
    s = snapshot(p, mode=3)
    if failure == 'stale': s = replace(s, state_age_s=.1)
    if failure == 'missed': s = replace(s, native_time=1.005)
    if failure == 'bias': s.output['q_output'][0] += .101
    if failure == 'fell': s.pelvis_pose[2] = .69
    if failure == 'tilt': s.pelvis_pose[3:] = [.9689124217, .2474039593, 0., 0.]
    if failure == 'upper': s.output['q_output'][18] += .201
    if failure == 'lost_palm': s.palm_tip_force_N[0] = 0.
    if failure == 'mode': s = replace(s, mode=6)
    with pytest.raises((ValueError, RuntimeError)) as error:
        c.compute_pd(s, upper)
    cleanup = c.abort(str(error.value), upper)
    assert c.state == 'paused'
    assert cleanup.pause_required and cleanup.hold_grip
    for field in upper:
        np.testing.assert_array_equal(cleanup.command[field][15:], upper[field][15:])
    for field in ('dq', 'kp', 'kd', 'tau'):
        np.testing.assert_array_equal(cleanup.command[field][:15], 0.)
    with pytest.raises(RuntimeError, match='paused'):
        c.advance_actor(snapshot(p, mode=3))


def test_final_publish_guard_rejects_actor_or_file_io_delay_and_cannot_replay_frame():
    c, p, upper = setup()
    activate(c, p)
    s = snapshot(p, mode=3)
    frame = c.compute_pd(s, upper)
    with pytest.raises(RuntimeError, match='publication|native output'):
        c.check_publication(frame, replace(s, native_time=1.0021))
    with pytest.raises(RuntimeError, match='unpublished'):
        c.compute_pd(snapshot(p, sim=1.005, mode=3, sequence=4), upper)


def test_pending_low_timeout_cannot_be_used_as_a_slowstart_wait():
    c, p, _ = setup()
    c.request_handoff(snapshot(p))
    with pytest.raises(RuntimeError, match='transition'):
        c.advance_actor(snapshot(p, sim=1.06, mode=6, sequence=4))


@pytest.mark.parametrize('ack', [None, {}, dict(previous_rate=.5, rate=.5, native_time_s=1., monotonic_s=10.),
    dict(previous_rate=.5, rate=.1, native_time_s=2., monotonic_s=10.),
    dict(previous_rate=.5, rate=.1, native_time_s=1., monotonic_s=11.)])
def test_low_request_requires_native_ack_of_one_way_point_one_pacing(ack):
    c, p, _ = setup()
    with pytest.raises(RuntimeError, match='pacing'):
        c.request_handoff(replace(snapshot(p), pacing_ack=ack))


def test_poll_older300hz_sample_waits_without_consuming_actor_or_pd_deadline():
    c, p, upper = setup()
    activate(c, p)
    first = c.compute_pd(snapshot(p, mode=3), upper)
    c.ack_published(first, snapshot(p, mode=3))
    s = replace(snapshot(p, mode=3), native_time=1.0025, wall_time=10.025)
    assert c.advance_actor(s) is None
    assert c.compute_pd(s, upper) is None
    assert c.actor_count == 1 and c.serial_pd.step_count == 1


def test_pd_cannot_silently_reuse_actor_reference_after_missed_actor_call():
    c, p, upper = setup()
    activate(c, p)
    for i in range(4):
        s = snapshot(p, sim=1.+i*.005, mode=3, sequence=2*(i+1))
        frame = c.compute_pd(s, upper)
        c.ack_published(frame, s)
    with pytest.raises(RuntimeError, match='actor'):
        c.compute_pd(snapshot(p, sim=1.020, mode=3, sequence=10), upper)


def test_runtime_verification_requires_exact_trial_gc_override_and_private_paced_sim():
    from htd_loaded import validate_loaded_runtime
    params = {'igris_c_param_server': {'ros__parameters': {
        'gravity_comp_gain': [1.]*31, 'gravity_comp_offset': [0.]*31,
        'torque_max': [60.]*31}}}
    robot = {'igris_c': {'controller': {'GravityCompensationModule': {'enabled': True}}}}
    scoped, _ = scoped_gc_configuration(params, robot)
    runtime = dict(network='none', ipc='private', hardware_executed=False, simulation_rate_limit=.5)
    manifest = dict(htd_backlash_limit_timeconst_s=.005, simulation_rate_limit=.5,
                    simulation_rate_start_phase_s=90., htd_loaded_zero=dict(enabled=True))
    validate_loaded_runtime(runtime, manifest, robot, scoped, params)
    scoped['igris_c_param_server']['ros__parameters']['torque_max'][0] = 61.
    with pytest.raises(ValueError, match='parameters'):
        validate_loaded_runtime(runtime, manifest, robot, scoped, params)
    scoped, _ = scoped_gc_configuration(params, robot)
    for bad in [dict(network='host'), dict(ipc='host'), dict(hardware_executed=True), dict(simulation_rate_limit=.1)]:
        with pytest.raises(ValueError):
            validate_loaded_runtime({**runtime, **bad}, manifest, robot, scoped, params)
    with pytest.raises(ValueError, match='initializer'):
        validate_loaded_runtime(runtime, dict(manifest, htd_initial_state={}), robot, scoped, params)


@pytest.mark.parametrize('problem', ['bias', 'owned_limit', 'profile', 'imu'])
def test_first_handoff_checks_owned_and_upper_contract_before_low_request(problem):
    c, p, _ = setup()
    s = snapshot(p)
    if problem == 'bias': s.output['q_output'][0] += .101
    if problem == 'owned_limit': s.output['q_output'][6] = s.output['q_encoder'][6] = -1.
    if problem == 'profile': s.upper_output_target[15] += .101
    if problem == 'imu': s.torso_gyro[0] = np.nan
    with pytest.raises((ValueError, RuntimeError)):
        c.request_handoff(s)
    assert c.state == 'prepared'


def test_cleanup_uses_last_published_upper_not_a_later_unsent_candidate():
    c, p, upper = setup()
    activate(c, p)
    frame = c.compute_pd(snapshot(p, mode=3), upper)
    c.ack_published(frame, snapshot(p, mode=3))
    rejected = {k: v.copy() for k, v in upper.items()}
    rejected['q'][18] += .15
    cleanup = c.abort('publication deadline rejected a new upper target', rejected)
    np.testing.assert_array_equal(cleanup.command['q'][15:], upper['q'][15:])


def test_authorized_point_one_kg_mu_four_payload_preserves_force_formula():
    c, p, _ = setup()
    loaded = LoadedZero(p, c.serial_pd, p.default_q, mass_kg=.1, friction=4., clearance_lift_m=.04)
    assert loaded.required_palm_force == pytest.approx(.1839375)
    loaded.request_handoff(snapshot(p))


def test_postpublish_late_check_retains_the_command_that_was_actually_sent():
    c, p, upper = setup()
    activate(c, p)
    sent = {k: v.copy() for k, v in upper.items()}
    sent['q'][18] += .1
    frame = c.compute_pd(snapshot(p, mode=3), sent)
    c.check_publication(frame, snapshot(p, mode=3))
    with pytest.raises(RuntimeError):
        c.ack_published(frame, replace(snapshot(p, mode=3), native_time=1.0021))
    cleanup = c.abort('postpublication latency', upper)
    np.testing.assert_array_equal(cleanup.command['q'][15:], sent['q'][15:])


def test_remember_publication_before_postpublish_clock_read_preserves_upper():
    c, p, upper = setup()
    activate(c, p)
    sent = {k: v.copy() for k, v in upper.items()}
    sent['q'][18] += .1
    frame = c.compute_pd(snapshot(p, mode=3), sent)
    c.remember_published(frame)
    cleanup = c.abort('postpublish clock read failed before acknowledgement', upper)
    np.testing.assert_array_equal(cleanup.command['q'][15:], sent['q'][15:])


def current_grasp_snapshot(policy, **kwargs):
    s = snapshot(policy, **kwargs)
    s.upper_output_target[[15, 17]] += [.62, .79]
    for key in ('q_output', 'q_encoder'):
        s.output[key][15:29] = s.upper_output_target[15:29]
    return replace(s, clearance_rise_m=.021)


def test_current_grasp_reference_is_captured_only_after_qualification_and_never_rebased():
    c, p, _ = setup(current_grasp_reference=True, clearance_minimum_rise_m=.02)
    s = current_grasp_snapshot(p)
    with pytest.raises(RuntimeError, match='clearance'):
        c.request_handoff(replace(s, clearance_ready_s=.49))
    assert c.current_grasp_evidence is None
    c.request_handoff(s)
    evidence = copy.deepcopy(c.current_grasp_evidence)
    assert evidence['max_abs_delta_from_frozen_profile_rad'] == pytest.approx(.79)
    assert c.contract['runtime_domain_validated'] is False
    assert c.contract['current_grasp_reference'] is True
    assert c.contract['clearance_minimum_rise_m'] == .02
    active = current_grasp_snapshot(p, mode=3)
    assert c.advance_actor(active) is not None
    active.upper_output_target[15] += .09
    active.output['q_output'][15] += .09
    c._check(active)
    active.upper_output_target[15] += .02
    with pytest.raises(RuntimeError, match='captured current grasp'):
        c._check(active)
    assert c.current_grasp_evidence == evidence
    np.testing.assert_array_equal(c.hold_q, p.default_q)
    np.testing.assert_array_equal(c.serial_pd.contract['feasible_command_q'], p.default_q)


@pytest.mark.parametrize('change', [dict(clearance_rise_m=.019), dict(clearance_ready_s=.49),
    dict(payload_speed_m_s=.02), dict(source_support_z_N=.2), dict(grasp_verified=False),
    dict(palm_tip_force_N=np.zeros(4)), dict(environment=0)])
def test_current_grasp_keeps_measured_handoff_guards(change):
    c, p, _ = setup(current_grasp_reference=True, clearance_minimum_rise_m=.02)
    with pytest.raises((RuntimeError, ValueError)):
        c.request_handoff(replace(current_grasp_snapshot(p), **change))
    assert c.current_grasp_evidence is None
    assert c.state == 'prepared'


def test_current_grasp_keeps_physical_upper_error_guard_and_default_profile_guard():
    c, p, _ = setup(current_grasp_reference=True, clearance_minimum_rise_m=.02)
    s = current_grasp_snapshot(p)
    s.output['q_output'][18] += .201
    with pytest.raises(RuntimeError, match='Physical upper'):
        c.request_handoff(s)
    default, p, _ = setup()
    with pytest.raises(RuntimeError, match='frozen profile'):
        default.request_handoff(current_grasp_snapshot(p))
    assert default.contract['clearance_minimum_rise_m'] == pytest.approx(.048)


@pytest.mark.parametrize('scope', [dict(clearance_minimum_rise_m=.02),
    dict(current_grasp_reference='true'), dict(current_grasp_reference=True, clearance_minimum_rise_m=.019),
    dict(current_grasp_reference=True, clearance_minimum_rise_m=float('nan'))])
def test_current_grasp_scope_rejects_implicit_or_unreviewed_clearance_override(scope):
    with pytest.raises(ValueError, match='scope|clearance'):
        setup(**scope)
