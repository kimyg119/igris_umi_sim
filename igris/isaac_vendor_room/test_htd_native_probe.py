"""Offline boundaries for the empty-hand native HTD probe; no ROS/sim startup."""
import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def probe():
    assert importlib.util.find_spec('htd_native_probe') is not None, 'Native HTD probe is not implemented'
    return importlib.import_module('htd_native_probe')


@pytest.mark.parametrize('change', [dict(bms_init_state=0), dict(bms_init_state=1),
                                   dict(bms_init_state=2), dict(estop=1),
                                   dict(body_power=0), dict(legs_power=0)])
def test_torque_ready_requires_both_initialized_power_and_no_estop(change):
    status=dict(bms_init_state=3,body_power=1,legs_power=1,estop=0)
    assert probe().torque_ready(status,received_at=10.02,torque_ack_at=10.,now=10.03)
    status.update(change)
    assert not probe().torque_ready(status,received_at=10.02,torque_ack_at=10.,now=10.03)


@pytest.mark.parametrize('received,ack,now', [(9.99,10.,10.01),(10.,10.,10.01),
                                            (10.01,10.,10.12),(10.04,10.,10.03),
                                            (float('nan'),10.,10.03)])
def test_torque_ready_rejects_old_stale_future_or_nonfinite_bms(received,ack,now):
    status=dict(bms_init_state=3,body_power=1,legs_power=1,estop=0)
    assert not probe().torque_ready(status,received_at=received,torque_ack_at=ack,now=now)
    assert not probe().torque_ready(None,received_at=received,torque_ack_at=ack,now=now)


def test_gains_use_training_for_23_axes_and_saved_wrists_with_native_ankle_mapping():
    module = probe()
    kp, kd, evidence = module.load_gains()
    np.testing.assert_allclose(kp[:9], [121.514570, 660.634418, 230.579719,
                                     118.435253, 122.136354, 121.514570, 118.435253, 56.0487, 56.0572])
    np.testing.assert_allclose(kd[:9], [6.077864, 34.67, 12.1,
                                     15.079645, 10.367256, 6.077864, 15.079645, 8.92108, 8.92244])
    np.testing.assert_allclose(kp[[15, 16, 17, 18, 22, 23, 24, 25]], 121.514570)
    np.testing.assert_allclose(kp[[19, 20, 21, 26, 27, 28]], 400.)
    np.testing.assert_allclose(kd[[19, 20, 21, 26, 27, 28]], 2.)
    np.testing.assert_array_equal(kp[29:], [30., 30.])
    np.testing.assert_array_equal(kd[29:], [3., 3.])
    assert evidence['wrists_source'].endswith('replay_gains.toml')


def test_opt_in_waist_mapping_matches_vendor_audit_and_preserves_other_29_axes():
    module = probe()
    baseline_kp, baseline_kd, _ = module.load_gains()
    kp, kd, evidence = module.load_gains(map_waist=True)
    np.testing.assert_array_equal(kp[1:3], [107.38057936091046]*2)
    np.testing.assert_array_equal(kd[1:3], [5.6351217576586912]*2)
    other = np.r_[0, 3:31]
    np.testing.assert_array_equal(kp[other], baseline_kp[other])
    np.testing.assert_array_equal(kd[other], baseline_kd[other])
    assert evidence['waist_mapping']['enabled'] is True
    assert evidence['waist_mapping']['exact_serial_pd_equivalence'] is False
    # Independent local mechanics check from the supplied vendor IK derivative.
    # The actual vendor audit constant differs slightly from finite differences.
    jac = np.array([[-1.8091306491606534, -1.], [1.8091306491606534, -1.]])
    inverse = jac.T @ np.linalg.inv(jac @ jac.T + .1**2*np.eye(2))
    expected_motor = inverse.T @ np.diag(baseline_kp[1:3]) @ inverse
    np.testing.assert_allclose(kp[1:3], np.diag(expected_motor), rtol=0, atol=1e-5)
    effective = jac.T @ np.diag(kp[1:3]) @ jac
    np.testing.assert_allclose(effective, np.diag([702.903330285978, 214.761158721821]), atol=1e-7)
    assert np.max(np.abs(np.diag(effective)/baseline_kp[1:3]-1)) < .07


@pytest.mark.parametrize('axis,kind', [(1, 'kp'), (2, 'kd')])
def test_waist_mapping_refuses_unknown_serial_profile_without_mutating_inputs(axis, kind):
    module = probe()
    kp, kd, _ = module.load_gains()
    (kp if kind == 'kp' else kd)[axis] += .01
    before_kp, before_kd = kp.copy(), kd.copy()
    with pytest.raises(ValueError, match='Waist gains differ'):
        module.map_vendor_waist_gains(kp, kd)
    np.testing.assert_array_equal(kp, before_kp)
    np.testing.assert_array_equal(kd, before_kd)


def test_waist_mapping_does_not_mutate_valid_serial_inputs():
    module = probe()
    kp, kd, _ = module.load_gains()
    before_kp, before_kd = kp.copy(), kd.copy()
    module.map_vendor_waist_gains(kp, kd)
    np.testing.assert_array_equal(kp, before_kp)
    np.testing.assert_array_equal(kd, before_kd)


def test_target_gate_preserves_policy_output_but_rejects_urdf_and_coupled_limits():
    module = probe()
    from htd_policy import HTDPolicy
    gate = module.TargetLimits()
    q = HTDPolicy().default_q.copy()
    gate.check(q)
    q[3] = 99.
    with pytest.raises(ValueError, match='URDF'):
        gate.check(q)
    q = HTDPolicy().default_q.copy()
    q[1], q[2] = .3, .25
    with pytest.raises(ValueError, match='coupled'):
        gate.check(q)
    with pytest.raises(ValueError):
        gate.check(np.full(31, np.nan))


def test_imu_mount_is_read_from_torso_site_and_transforms_sensor_axes():
    module = probe()
    mount = module.imu_mount(module.ROOT/'igris/sim_local/igris_c_v2_rcv_hand.xml')
    quat, gyro = module.torso_imu(mount, [1., 2., 3.], mount)
    np.testing.assert_allclose(quat, [1., 0., 0., 0.], atol=1e-7)
    np.testing.assert_allclose(gyro, [3., 2., -1.], atol=1e-7)


def test_probe_velocity_schedule_stands_then_acceleration_limited_motion_and_stop():
    module = probe()
    command = np.zeros(3)
    for index in range(750):
        previous = command.copy()
        command = module.velocity_command(index*.02, command, .02, 10., 5., [.03, 0., 0.])
        assert np.max(np.abs(command-previous)) <= .00300000001
        if index < 500:
            np.testing.assert_array_equal(command, [0, 0, 0])
    np.testing.assert_allclose(command, [.03, 0, 0])
    for index in range(750, 1000):
        command = module.velocity_command(index*.02, command, .02, 10., 5., [.03, 0., 0.])
    np.testing.assert_array_equal(command, [0, 0, 0])


@pytest.mark.parametrize('environment,age,mode_age', [(0, .001, .01), (1, .11, .01), (1, .001, .51)])
def test_state_guard_rejects_real_environment_and_stale_sdk_state(environment, age, mode_age):
    module = probe()
    with pytest.raises(RuntimeError):
        module.require_state(environment, age, mode_age)
    module.require_state(1, .099, .49)


def test_startup_waits_for_shm_recovery_after_latest_degraded_event():
    module = probe()
    degraded = 'Runtime SHM degraded; forcing safe stop (hold, or torque-off if unpowered)'
    recovered = 'Runtime SHM recovered; holding in JOINT_HOLD (run HOME preset to resume).'
    assert not module.runtime_recovered(degraded)
    assert module.runtime_recovered(degraded+'\n'+recovered)
    assert not module.runtime_recovered(degraded+'\n'+recovered+'\n'+degraded)


def test_projection_clips_small_knee_violation_preserves_other_targets_and_input():
    module = probe()
    from htd_policy import HTDPolicy
    gate = module.TargetLimits()
    q = HTDPolicy().default_q.copy()
    q[6] = -.04
    before = q.copy()
    actual = gate.project(q)
    assert actual[6] == 0.
    np.testing.assert_array_equal(actual[np.arange(31) != 6], q[np.arange(31) != 6])
    np.testing.assert_array_equal(q, before)
    gate.check(actual)


def test_projection_enforces_coupled_polygon_with_euclidean_nearest_point():
    module = probe()
    from htd_policy import HTDPolicy
    gate = module.TargetLimits()
    q = HTDPolicy().default_q.copy()
    q[1], q[2] = .10, .25
    actual = gate.project(q)
    # Hand-derived perpendicular projection to the upper-right waist polygon
    # edge, after the vendor's 0.95 hard-boundary contraction.
    a = np.array([.389785, -.155453])
    b = np.array([0., .270622])
    point = a+np.dot(np.array([.10, .25])-a, b-a)/np.dot(b-a, b-a)*(b-a)
    np.testing.assert_allclose(actual[[1, 2]], point, atol=1e-8)
    gate.check(actual)


def test_projection_aborts_corrections_above_point_one_radian():
    module = probe()
    from htd_policy import HTDPolicy
    q = HTDPolicy().default_q.copy()
    q[6] = -.1001
    with pytest.raises(ValueError, match='0.10'):
        module.TargetLimits().project(q)


def test_slow_start_log_evidence_does_not_invent_an_inactive_transition():
    module = probe()
    active = 'SlowStartModule elapsedSec: 2.007344, slowStartActive: true, maxQDelta: 0.014069, maxQDeltaJoint: r_shoulder_roll, errorJoints: []'
    evidence = module.slow_start_records('unrelated\n'+active+'\nother log')
    assert evidence == [dict(elapsed_s=2.007344, active=True, max_q_delta_rad=.014069,
                             max_q_delta_joint='r_shoulder_roll', error_joints='')]
    assert module.slow_start_records('no SlowStart log') == []
    assert module.slow_start_records(active.replace('true', 'false'))[0]['active'] is False


def test_low_hold_requires_fresh_finite_all31_state_and_confirmed_low():
    module = probe()
    good = dict(q=np.zeros(31), dq=np.zeros(31), status=[0]*31)
    module.require_low_hold(1, 3, good, .01, .02)
    for state in (None, dict(good, q=np.zeros(30)), dict(good, dq=np.full(31, np.nan)),
                  dict(good, status=[1]+[0]*30)):
        with pytest.raises(RuntimeError):
            module.require_low_hold(1, 3, state, .01, .02)
    with pytest.raises(RuntimeError, match='LOW'):
        module.require_low_hold(1, 5, good, .01, .02)


def test_cleanup_restores_only_the_last_published_command_not_an_unsent_candidate():
    module = probe()
    from htd_policy import HTDPolicy
    original = HTDPolicy().default_q.copy()
    memory = module.PublishedCommand(original, np.zeros(31))
    candidate = original.copy()
    candidate[3] -= .1
    # Candidate preparation/guard failure does not acknowledge publication.
    q, dq = memory.restore()
    np.testing.assert_array_equal(q, original)
    np.testing.assert_array_equal(dq, np.zeros(31))
    memory.remember(candidate, np.full(31, .1))
    candidate[:] = 99.
    q, dq = memory.restore()
    assert q[3] == original[3]-.1
    np.testing.assert_array_equal(dq, np.full(31, .1))
    q[:] = -99.
    assert memory.restore()[0][3] == original[3]-.1
