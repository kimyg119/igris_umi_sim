"""Offline contract regressions; never start ROS, physics or publish commands."""
import importlib

import numpy as np
import pytest


def setup(delay_steps=(0, 0, 0, 0)):
    module = importlib.import_module('htd_serial_pd')
    from htd_native_probe import TargetLimits, load_gains
    from htd_policy import HTDPolicy
    policy = HTDPolicy()
    kp, kd, _ = load_gains()
    adapter = module.SerialPD(policy.default_q, kp, kd, TargetLimits(), delay_steps=delay_steps)
    adapter.reset(1.)
    return module, adapter, policy


def sample(policy, sim=1., sequence=2):
    return dict(q_output=policy.default_q.copy(), dq_output=np.zeros(31),
                q_encoder=policy.default_q.copy(), dq_encoder=np.zeros(31),
                sim=sim, sequence=sequence, monotonic=10.)


def test_raw_outside_target_preserves_training_effort_and_held_axes_without_mutation():
    _, adapter, policy = setup()
    raw = policy.default_q.copy()
    raw[8] = -.7007119655609131
    state = sample(policy)
    state['q_output'][8] = -.04
    state['q_encoder'][8] = -.04
    state['dq_output'][8] = -3.
    original = {k: v.copy() for k, v in state.items() if isinstance(v, np.ndarray)}
    before = raw.copy()
    r = adapter.step(raw, state, native_time=1., wall_time=10.)
    assert r.tau[8] == pytest.approx(77.634366*(float(raw[8])+.04)+12.355893*3)
    assert r.tau[8] < 0  # Clipping the reference would give the opposite sign.
    np.testing.assert_array_equal(r.raw_target, raw)
    np.testing.assert_array_equal(r.q, policy.default_q)
    np.testing.assert_array_equal(r.kp[:15], 0.)
    np.testing.assert_array_equal(r.kd[:15], 0.)
    np.testing.assert_array_equal(r.tau[15:], 0.)
    np.testing.assert_array_equal(r.dq, 0.)
    from htd_native_probe import load_gains
    baseline_kp, baseline_kd, _ = load_gains()
    np.testing.assert_array_equal(r.kp[15:], baseline_kp[15:])
    np.testing.assert_array_equal(r.kd[15:], baseline_kd[15:])
    np.testing.assert_array_equal(raw, before)
    for key, value in original.items():
        np.testing.assert_array_equal(state[key], value)


def test_training_and_native_effort_caps_are_separate_and_signed():
    _, adapter, policy = setup()
    raw = policy.default_q.copy()
    raw[1], raw[8], raw[9] = 9., -9., 9.
    r = adapter.step(raw, sample(policy), native_time=1., wall_time=10.)
    np.testing.assert_allclose(r.tau_training[[1, 8, 9]], [217.0253, -148.7251, 150.])
    np.testing.assert_allclose(r.tau[[1, 8, 9]], [60., -90., 150.])
    assert r.training_clipped[1] and r.native_clipped[1]
    assert r.training_clipped[9] and not r.native_clipped[9]


def test_group_delay_uses_pd_ticks_and_repeats_first_reference_not_zero():
    _, adapter, policy = setup((0, 1, 2, 2))
    raw = policy.default_q.copy()
    raw[:15] += .01
    first = adapter.step(raw, sample(policy), native_time=1., wall_time=10.)
    np.testing.assert_array_equal(first.delayed_target, raw)
    second_raw = raw.copy()
    second_raw[:15] += .02
    second = adapter.step(second_raw, sample(policy, 1.005, 4), native_time=1.005, wall_time=10.)
    assert second.delayed_target[3] == second_raw[3]  # X12: zero delay
    assert second.delayed_target[4] == raw[4]  # X8: one step
    assert second.delayed_target[5] == raw[5]  # X6: two steps
    assert second.delayed_target[8] == raw[8]  # PLA: two steps
    third = adapter.step(second_raw, sample(policy, 1.010, 6), native_time=1.010, wall_time=10.)
    assert third.delayed_target[4] == second_raw[4]
    assert third.delayed_target[8] == raw[8]
    fourth = adapter.step(second_raw, sample(policy, 1.015, 8), native_time=1.015, wall_time=10.)
    assert fourth.delayed_target[8] == second_raw[8]


def test_deadlines_keep_200hz_native_cadence_on_2100hz_samples_no_duplicates():
    _, adapter, policy = setup()
    records = []
    for step in range(105):
        t = 1.+step/2100.
        state = sample(policy, t, 2*(step+1))
        if adapter.due(state, native_time=t, wall_time=10.):
            records.append(adapter.step(policy.default_q, state, native_time=t, wall_time=10.))
            assert not adapter.due(state, native_time=t, wall_time=10.)
    assert len(records) == 10
    np.testing.assert_allclose([r.scheduled_time for r in records], 1.+np.arange(10)*.005, atol=1e-12)
    assert max(r.lag_s for r in records) <= 1/2100.+1e-9


@pytest.mark.parametrize('problem', ['missed', 'backwards', 'stale_native', 'stale_wall', 'same_sequence', 'nonfinite', 'bias', 'physical'])
def test_bad_state_or_deadline_aborts_without_consuming_pd_tick(problem):
    _, adapter, policy = setup()
    adapter.step(policy.default_q, sample(policy), native_time=1., wall_time=10.)
    s = sample(policy, 1.005, 4)
    now = 1.005
    wall = 10.
    if problem == 'missed':
        s['sim'] = now = 1.010
    elif problem == 'backwards':
        s['sim'] = now = .999
    elif problem == 'stale_native':
        now = 1.008
    elif problem == 'stale_wall':
        wall = 10.101
    elif problem == 'same_sequence':
        s['sequence'] = 2
    elif problem == 'nonfinite':
        s['dq_output'][4] = np.nan
    elif problem == 'bias':
        s['q_output'][3] -= .11
    elif problem == 'physical':
        s['q_output'][6] = s['q_encoder'][6] = -.01
    with pytest.raises((ValueError, RuntimeError)):
        adapter.step(policy.default_q, s, native_time=now, wall_time=wall)
    assert adapter.step_count == 1


def test_publication_rejects_late_candidate_and_preserves_record_copies():
    _, adapter, policy = setup()
    s = sample(policy)
    r = adapter.step(policy.default_q, s, native_time=1., wall_time=10.)
    adapter.check_publication(r, native_time=1.0019, wall_time=10.0019)
    with pytest.raises(RuntimeError, match='publication'):
        adapter.check_publication(r, native_time=1.005, wall_time=10.003)
    with pytest.raises(ValueError):
        r.tau[0] = 9.


def test_clock_reversal_between_not_due_polls_is_rejected():
    _, adapter, policy = setup()
    adapter.step(policy.default_q, sample(policy), native_time=1., wall_time=10.)
    assert not adapter.due(sample(policy, 1.004, 4), native_time=1.004, wall_time=10.)
    with pytest.raises(RuntimeError, match='backwards'):
        adapter.due(sample(policy, 1.003, 6), native_time=1.003, wall_time=10.)


def test_pre_activation_actor_warmup_does_not_advance_raw_history_or_phase():
    module, _, policy = setup()
    policy.reset(sim_time=1.)
    module.warm_policy_actor(policy)
    r = policy.step(policy.default_q, np.zeros(31), [1., 0., 0., 0.], np.zeros(3), np.zeros(3), 1.)
    assert r.policy_time == 1.
    np.testing.assert_array_equal(r.observation[86:116], 0.)
    np.testing.assert_array_equal(r.observation[-4:], 0.)


def test_300hz_observer_waits_for_first_sample_after_200hz_deadline_without_stale_poll_abort():
    _, adapter, policy = setup()
    records = []
    for physics_tick in range(210):
        native = 1.+physics_tick/2100.
        observer_tick = physics_tick//7
        state = sample(policy, 1.+observer_tick/300., 2*(observer_tick+1))
        if adapter.due(state, native_time=native, wall_time=10.):
            r = adapter.step(policy.default_q, state, native_time=native, wall_time=10.)
            adapter.check_publication(r, native_time=native, wall_time=10.)
            records.append(r)
    assert len(records) == 20
    np.testing.assert_allclose([r.lag_s for r in records], [0., 1/600.]*10, atol=1e-12)
    np.testing.assert_allclose(np.diff([r.observation_sim_time for r in records]), [1/150., 1/300.]*9+[1/150.], atol=1e-12)


def test_publication_has_stricter_sample_freshness_than_full_deadline_guard():
    _, adapter, policy = setup()
    r = adapter.step(policy.default_q, sample(policy), native_time=1., wall_time=10.)
    with pytest.raises(RuntimeError, match='publication'):
        adapter.check_publication(r, native_time=1.0021, wall_time=10.0021)


def test_stand12_stale_after_actor_waits_for_next_300hz_sample_without_consuming_tick():
    _, adapter, policy = setup()
    old = sample(policy, 1., 2)
    assert not adapter.due(old, native_time=1.+6/2100., wall_time=10.)
    assert adapter.step_count == 0
    with pytest.raises(RuntimeError):
        adapter.step(policy.default_q, old, native_time=1.+6/2100., wall_time=10.)
    fresh = sample(policy, 1.+7/2100., 4)
    record = adapter.step(policy.default_q, fresh, native_time=fresh['sim'], wall_time=10.)
    assert record.scheduled_time == 1.
    assert record.lag_s == pytest.approx(1/300.)
    assert adapter.step_count == 1


def test_waiting_for_stale_sample_never_escapes_full_missed_pd_deadline():
    _, adapter, policy = setup()
    assert not adapter.due(sample(policy), native_time=1.003, wall_time=10.)
    with pytest.raises(RuntimeError, match='Missed PD deadline'):
        adapter.due(sample(policy), native_time=1.005, wall_time=10.)
    assert adapter.step_count == 0


def test_pacing_request_is_explicit_one_shot_and_records_clock_bracket(tmp_path):
    import htd_native_probe as probe
    clocks = iter([dict(sim=1., monotonic=10.), dict(sim=1.00047619, monotonic=10.0005)])
    assert probe.request_pd_pacing(tmp_path, None, lambda: next(clocks)) is None
    assert not (tmp_path/'pacing.request').exists()
    event = probe.request_pd_pacing(tmp_path, .5, lambda: next(clocks))
    assert event['rate'] == .5
    assert event['clock_before']['sim'] == 1.
    assert event['clock_after']['sim'] == 1.00047619
    assert event['native_activation_acknowledged'] is False
    assert event['request_file_mtime_ns'] == (tmp_path/'pacing.request').stat().st_mtime_ns
    with pytest.raises(FileExistsError):
        probe.request_pd_pacing(tmp_path, .5, lambda: dict(sim=2.))


@pytest.mark.parametrize('rate', [.1, .25, 1., 0., -.1, 1.001, float('nan'), float('inf')])
def test_pacing_rate_domain_validated_before_request_or_worker(rate, tmp_path, monkeypatch):
    import htd_native_probe as probe
    calls = []
    monkeypatch.setattr(probe, 'worker', lambda out, args: calls.append(args.simulation_rate_limit) or 0)
    monkeypatch.setattr('sys.argv', ['htd_native_probe', 'worker', '--out', str(tmp_path),
        '--serial-pd-torque', '--output-coordinates', '--low-warmup-seconds', '3.2',
        '--serial-pd-delay-steps', '0', '0', '0', '0', '--simulation-rate-limit', str(rate)])
    if np.isfinite(rate) and 0 < rate <= 1:
        assert probe.main() == 0
        assert calls == [rate]
        event = probe.request_pd_pacing(tmp_path, rate, lambda: dict(sim=1.))
        assert event['rate'] == rate
        assert (tmp_path/'pacing.request').exists()
    else:
        with pytest.raises(SystemExit) as error:
            probe.main()
        assert error.value.code == 2
        assert not calls
        with pytest.raises(ValueError):
            probe.request_pd_pacing(tmp_path, rate, lambda: pytest.fail('Invalid rate read clock'))
        assert not (tmp_path/'pacing.request').exists()


@pytest.mark.parametrize('failure', ['hardware', 'ownership', 'status', 'stale_sdk'])
def test_no_spin_publication_still_refuses_bad_ownership_state_before_command_fields(failure):
    from types import SimpleNamespace
    import time
    import htd_native_probe as probe
    _, adapter, policy = setup()
    now = time.monotonic()
    s = sample(policy)
    s['monotonic'] = now
    record = adapter.step(policy.default_q, s, native_time=1., wall_time=now)
    message = SimpleNamespace(motors=[SimpleNamespace(q=99., tau=99.) for _ in range(31)])
    sdk = dict(q=policy.default_q, dq=np.zeros(31), status=[0]*31)
    environment, mode, state_at = 1, 3, now
    if failure == 'hardware': environment = 0
    if failure == 'ownership': mode = 5
    if failure == 'status': sdk['status'][6] = 1
    if failure == 'stale_sdk': state_at -= .101
    def clock():
        raise AssertionError('Invalid SDK ownership/state must be rejected before preparing publication')
    with pytest.raises(RuntimeError):
        probe.prepare_serial_pd_message(message, record, adapter, environment, mode, sdk,
                                        state_at, now, clock, {})
    assert all(m.q == 99. and m.tau == 99. for m in message.motors)


@pytest.mark.parametrize('native_age', [.0019, .0021])
def test_no_spin_publication_checks_coordinate_freshness_and_preserves_message_contract(native_age):
    from types import SimpleNamespace
    import time
    import htd_native_probe as probe
    _, adapter, policy = setup()
    now = time.monotonic()
    s = sample(policy)
    s['monotonic'] = now
    record = adapter.step(policy.default_q, s, native_time=1., wall_time=now)
    message = SimpleNamespace(motors=[SimpleNamespace() for _ in range(31)])
    sdk = dict(q=policy.default_q, dq=np.zeros(31), status=[0]*31)
    stage = {}
    invoke = lambda: probe.prepare_serial_pd_message(
        message, record, adapter, 1, 3, sdk, now, now, lambda: dict(sim=1.+native_age), stage)
    if native_age > .002:
        with pytest.raises(RuntimeError, match='PD publication'):
            invoke()
    else:
        clock, publication_time = invoke()
        assert clock['sim'] == 1.+native_age
        assert now <= stage['message_start_monotonic'] <= stage['message_done_monotonic'] <= publication_time
        for name in ('q', 'dq', 'kp', 'kd', 'tau'):
            np.testing.assert_array_equal([getattr(m, name) for m in message.motors], getattr(record, name))


def test_delay_config_is_explicit_and_cannot_claim_nominal_zero_delay():
    module, adapter, _ = setup()
    contract = adapter.contract
    assert contract['training_delay_range_steps'] == [0, 2]
    assert contract['nominal_disables_delay'] is False
    assert contract['configured_delay_steps'] == [0, 0, 0, 0]
    assert contract['exact_native_actuator_equivalence'] is False
    assert module.DELAY_GROUPS == ('X12-150', 'X8-120', 'X6-60', 'PLA')
    for delay in ((0, 0, 0), (0, 0, 0, 3), (0, 0, 0, .5)):
        with pytest.raises(ValueError):
            setup(delay)


def test_ros_field_boundary_and_cleanup_never_replay_stale_effort():
    from types import SimpleNamespace
    module, adapter, policy = setup()
    r = adapter.step(policy.default_q, sample(policy), native_time=1., wall_time=10.)
    message = SimpleNamespace(motors=[SimpleNamespace() for _ in range(31)])
    module.write_motor_fields(message, r)
    assert all(m.kp == 0 and m.kd == 0 for m in message.motors[:15])
    assert all(m.tau == 0 for m in message.motors[15:])
    for m in message.motors[:15]:
        m.tau = 99.  # An acknowledged previous torque must not leak during pause.
    module.zero_owned_effort(message)
    assert all(m.tau == 0 and m.kp == 0 and m.kd == 0 for m in message.motors[:15])
    assert message.motors[19].kp == 400.


def test_probe_default_stays_position_control_and_opt_in_requires_explicit_contract(monkeypatch, tmp_path):
    import htd_native_probe as probe
    captured = []
    monkeypatch.setattr(probe, 'worker', lambda out, args: captured.append(args) or 0)
    monkeypatch.setattr('sys.argv', ['htd_native_probe', 'worker', '--out', str(tmp_path)])
    assert probe.main() == 0
    assert captured[-1].serial_pd_torque is False
    assert captured[-1].serial_pd_delay_steps is None
    assert captured[-1].simulation_rate_limit is None
    valid = ['htd_native_probe', 'worker', '--out', str(tmp_path), '--serial-pd-torque',
             '--output-coordinates', '--low-warmup-seconds', '3.2', '--serial-pd-delay-steps', '0', '1', '2', '0']
    monkeypatch.setattr('sys.argv', valid)
    assert probe.main() == 0
    assert captured[-1].serial_pd_torque is True
    assert captured[-1].serial_pd_delay_steps == [0, 1, 2, 0]
    monkeypatch.setattr('sys.argv', valid+['--simulation-rate-limit', '0.5'])
    assert probe.main() == 0
    assert captured[-1].simulation_rate_limit == .5
    for bad in (valid+['--project-targets'], valid+['--map-waist-gains'],
                valid[:-5], [v for v in valid if v != '--output-coordinates']):
        monkeypatch.setattr('sys.argv', bad)
        with pytest.raises(SystemExit) as error:
            probe.main()
        assert error.value.code == 2
