"""Native coordinate contracts; no simulator is launched or stepped."""
import importlib
import struct
from pathlib import Path

import numpy as np
import pytest


def module():
    return importlib.import_module('htd_output_coordinates')


def snapshot():
    from htd_policy import HTDPolicy
    q = HTDPolicy().default_q.copy()
    return dict(q_output=q, dq_output=np.linspace(-.3, .3, 31),
                q_encoder=q-np.linspace(-.005, .005, 31),
                dq_encoder=np.linspace(-.1, .1, 31))


def test_two_sided_compensation_matches_scalar_output_pd_and_keeps_inputs():
    m = module()
    from htd_native_probe import TargetLimits, load_gains
    s = snapshot()
    reference = s['q_output'].copy()
    before = {k: v.copy() for k, v in s.items()}
    q, dq, bias, dbias = m.compensate_output(reference, reference, s, TargetLimits())
    kp, kd, _ = load_gains()
    actual = kp*(q-s['q_encoder'])+kd*(dq-s['dq_encoder'])
    expected = kp*(reference-s['q_output'])-kd*s['dq_output']
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(bias, s['q_output']-s['q_encoder'])
    np.testing.assert_allclose(dbias, s['dq_output']-s['dq_encoder'])
    for k in s:
        np.testing.assert_array_equal(s[k], before[k])


def test_compensation_checks_physical_and_virtual_targets_and_total_correction():
    m = module()
    from htd_native_probe import TargetLimits
    s = snapshot()
    q = s['q_output'].copy()
    s['q_encoder'] = s['q_output'].copy()
    s['q_encoder'][6] -= .5
    with pytest.raises(ValueError):
        m.compensate_output(q, q, s, TargetLimits())
    s = snapshot()
    s['q_output'][1:3] = [0., .2]
    s['q_encoder'][1:3] = [.08, .25]
    q = s['q_output'].copy()
    with pytest.raises(ValueError, match='coupled'):
        m.compensate_output(q, q, s, TargetLimits())
    s = snapshot()
    with pytest.raises(ValueError, match='0.10'):
        m.compensate_output(q-np.r_[np.zeros(3), .101, np.zeros(27)], q, s, TargetLimits())


@pytest.mark.parametrize('bad', [np.full(31, np.nan), np.zeros(30)])
def test_compensation_rejects_invalid_native_vectors(bad):
    m = module()
    from htd_native_probe import TargetLimits
    s = snapshot()
    q = s['q_output'].copy()
    s['dq_output'] = bad
    with pytest.raises(ValueError):
        m.compensate_output(q, q, s, TargetLimits())


def test_compensation_rejects_excess_velocity_bias_instead_of_clipping():
    m = module()
    from htd_native_probe import TargetLimits
    s = snapshot()
    s['dq_output'][3] = 3.
    with pytest.raises(ValueError, match='velocity'):
        m.compensate_output(s['q_output'], s['q_output'], s, TargetLimits())


def test_opt_in_velocity_saturation_bounds_command_and_preserves_raw_bias_evidence():
    m = module()
    from htd_native_probe import TargetLimits, load_gains
    s = snapshot()
    s['dq_output'][3] = 5.
    reference = s['q_output'].copy()
    q, dq, bias, dbias = m.compensate_output(reference, reference, s, TargetLimits(), saturate_velocity=True)
    assert dbias[3] > 5. and dq[3] == -2.
    np.testing.assert_allclose(dq, np.clip(-dbias, -2., 2.))
    kp, kd, _ = load_gains()
    actual = kp*(q-s['q_encoder'])+kd*(dq-s['dq_encoder'])
    desired = kp*(reference-s['q_output'])-kd*s['dq_output']
    # Precisely characterize the lost damping identity; do not call it parity.
    np.testing.assert_allclose(actual-desired, kd*(dq+dbias), atol=1e-12)
    assert abs(actual[3]-desired[3]) > 40.


def test_policy_joint_scope_keeps_all16_nonowned_targets_and_velocities_exact_baseline():
    m = module()
    from htd_native_probe import TargetLimits
    from htd_policy import POLICY_SDK_INDICES
    s = snapshot()
    reference = s['q_output'].copy()
    s['q_encoder'][15:] += 2.
    s['dq_encoder'][15:] += 100.
    q, dq, bias, dbias = m.compensate_output(reference, reference, s, TargetLimits(),
                                            saturate_velocity=True, owned_indices=POLICY_SDK_INDICES)
    np.testing.assert_array_equal(q[15:], reference[15:])
    np.testing.assert_array_equal(dq[15:], np.zeros(16))
    np.testing.assert_allclose(q[:15], reference[:15]-bias[:15])
    np.testing.assert_allclose(dq[:15], np.clip(-dbias[:15], -2., 2.))
    assert np.max(abs(bias[15:])) >= 1.99  # Full observer evidence remains visible.
    s['q_encoder'][6] -= .2
    with pytest.raises(ValueError, match='0.10'):
        m.compensate_output(reference, reference, s, TargetLimits(),
                            saturate_velocity=True, owned_indices=POLICY_SDK_INDICES)


def test_encoder_projection_keeps_positive_output_knee_target_feasible_after_bias():
    m = module()
    from htd_native_probe import TargetLimits
    from htd_policy import POLICY_SDK_INDICES
    s = snapshot()
    reference = s['q_output'].copy()
    reference[12] = .00747072696685791
    s['q_encoder'][12] = s['q_output'][12]-.02463108780008455
    q, dq, bias, dbias = m.compensate_output(reference, reference, s, TargetLimits(),
        saturate_velocity=True, owned_indices=POLICY_SDK_INDICES, project_encoder=True)
    assert q[12] == 0.
    assert q[12]+bias[12]-reference[12] == pytest.approx(.01716036083322664)
    np.testing.assert_array_equal(q[15:], reference[15:])
    np.testing.assert_array_equal(dq[15:], np.zeros(16))
    TargetLimits().check(q)


def test_encoder_projection_rejects_combined_budget_even_when_each_projection_is_small():
    m = module()
    from htd_native_probe import TargetLimits
    from htd_policy import POLICY_SDK_INDICES
    s = snapshot()
    raw = s['q_output'].copy()
    raw[6] = -.08
    s['q_encoder'][6] = s['q_output'][6]-.04
    gate = TargetLimits()
    projected = gate.project(raw)
    # First projection=.08, second=.04. Both individually pass .10, but their
    # aggregate must fail, even though final command0 differs from raw by .08.
    with pytest.raises(ValueError, match='Combined projection'):
        m.compensate_output(raw, projected, s, gate, saturate_velocity=True,
                            owned_indices=POLICY_SDK_INDICES, project_encoder=True)


def test_explicit_projection_budget_accepts_stand08_target_without_relaxing_physical_limits():
    m = module()
    from htd_native_probe import TargetLimits
    from htd_policy import POLICY_SDK_INDICES
    s = snapshot()
    raw = s['q_output'].copy()
    raw[12] = -.17610791325569153
    s['q_encoder'][12] = s['q_output'][12]-.03
    gate = TargetLimits()
    with pytest.raises(ValueError, match='0.10'):
        gate.project(raw)
    projected = gate.project(raw, max_correction_rad=.30)
    q, dq, bias, _ = m.compensate_output(raw, projected, s, gate,
        saturate_velocity=True, owned_indices=POLICY_SDK_INDICES, project_encoder=True,
        projection_budget_rad=.30, raw_command_budget_rad=.40)
    gate.check(q)
    assert q[12] == 0.
    assert np.max(abs(q-raw)) < .40
    np.testing.assert_array_equal(q[15:], raw[15:])
    np.testing.assert_array_equal(dq[15:], np.zeros(16))


def test_larger_projection_budget_still_rejects_aggregate_and_position_bias_excess():
    m = module()
    from htd_native_probe import TargetLimits
    from htd_policy import POLICY_SDK_INDICES
    s = snapshot()
    raw = s['q_output'].copy()
    raw[6] = -.28
    s['q_encoder'][6] = s['q_output'][6]-.04
    gate = TargetLimits()
    projected = gate.project(raw, max_correction_rad=.30)
    options = dict(saturate_velocity=True, owned_indices=POLICY_SDK_INDICES,
                   project_encoder=True, projection_budget_rad=.30, raw_command_budget_rad=.40)
    with pytest.raises(ValueError, match='Combined projection'):
        m.compensate_output(raw, projected, s, gate, **options)
    s['q_encoder'][6] = s['q_output'][6]-.1001
    with pytest.raises(ValueError, match='position bias'):
        m.compensate_output(raw, projected, s, gate, **options)


def test_raw_to_command_budget_is_separate_from_projection_budget():
    m = module()
    from htd_native_probe import TargetLimits
    from htd_policy import POLICY_SDK_INDICES
    s = snapshot()
    raw = s['q_output'].copy()
    raw[6] = -.20
    gate = TargetLimits()
    projected = gate.project(raw, max_correction_rad=.30)
    with pytest.raises(ValueError, match='Total projection'):
        m.compensate_output(raw, projected, s, gate, saturate_velocity=True,
            owned_indices=POLICY_SDK_INDICES, project_encoder=True,
            projection_budget_rad=.30, raw_command_budget_rad=.10)


def test_reader_rejects_unfinished_stale_future_and_pre_reset_records(tmp_path):
    m = module()
    path = tmp_path/'output_coordinates.bin'
    data = np.zeros(313, dtype=np.float64)
    data[:3] = [100., 50., 2.]
    path.write_bytes(struct.pack('=QQ', 2, 1)+data.tobytes())
    s = m.read_output(path, now=50.01, native_time=2.003)
    assert s['sequence'] == 2 and s['sim'] == 2.
    for now, native_time in [(50.11, 2.003), (49., 2.003), (50.01, .03), (50.01, 2.02)]:
        with pytest.raises(RuntimeError):
            m.read_output(path, now=now, native_time=native_time)
    path.write_bytes(struct.pack('=QQ', 3, 1)+data.tobytes())
    with pytest.raises(RuntimeError, match='complete'):
        m.read_output(path, now=50.01, native_time=2.003)
    data[10] = np.nan
    path.write_bytes(struct.pack('=QQ', 4, 1)+data.tobytes())
    with pytest.raises(RuntimeError, match='finite'):
        m.read_output(path, now=50.01, native_time=2.003)


def test_native_mapping_compiles_and_checks_actual_model_without_stepping(tmp_path):
    m = module()
    from htd_native_probe import ROOT
    xml = ROOT/'outputs/hlik_transport_20260918_2005/htd_native_probe04/run/vendor.xml'
    result = m.build_observer(tmp_path, xml)
    assert result['self_test']['joints'] == 31
    assert result['self_test']['coaxial_pairs'] == 29
    assert result['self_test']['perturbation_checks'] == 31
    assert result['self_test']['physics_steps'] == 0
    assert Path(result['library']).is_file()
