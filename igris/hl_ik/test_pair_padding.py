"""Pair padding changes clearance, never raw spheres or configured exclusions."""
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from . import geometry as geometry_module
from .common import ARM_NAMES, SDK_NAMES, sha256
from .geometry import SphereGeometry
from .retarget_quality import Distances, fk

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'outputs/hlik_transport_20260918_2005'
PAIR = ('pelvis', 'r_upper_leg')


@pytest.fixture
def toy(monkeypatch):
    names = ['pelvis', 'r_upper_leg', 'l_upper_leg']
    cfg = {'kinematics': {
        'collision_link_names': names,
        'collision_spheres': {n: [{'center': [0., 0., 0.], 'radius': .0625}] for n in names},
        'self_collision_ignore': {},
    }}
    monkeypatch.setattr(geometry_module.yaml, 'safe_load', lambda _: cfg)
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="pelvis"><freejoint/><geom size=".01"/>
        <body name="r_upper_leg" pos=".128 0 0"><joint type="slide" axis="1 0 0"/><geom size=".01"/></body>
        <body name="l_upper_leg" pos="0 .129 0"><joint type="slide" axis="0 1 0"/><geom size=".01"/></body>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    fk(model, data, model.qpos0)
    return model, data, cfg


def assert_contacts_equal(actual, expected, atol=0.):
    assert [r[2:] for r in actual] == [r[2:] for r in expected]
    np.testing.assert_allclose([r[0] for r in actual], [r[0] for r in expected], rtol=0, atol=atol)
    for a, b in zip(actual, expected):
        if a[1] is None:
            assert b[1] is None
        else:
            np.testing.assert_allclose(a[1], b[1], rtol=0, atol=atol)


@pytest.mark.parametrize('cls', [SphereGeometry, Distances])
@pytest.mark.parametrize('bad', [
    [], True, 'pelvis/r_upper_leg',
    {'pelvis': 0.}, {('pelvis',): 0.}, {(1, 'r_upper_leg'): 0.},
    {('pelvis', 'missing'): 0.}, {('pelvis', 'pelvis'): 0.},
    {PAIR: -1e-9}, {PAIR: .00250001}, {PAIR: float('nan')},
    {PAIR: float('inf')}, {PAIR: True}, {PAIR: '0'}, {PAIR: [0.]},
    {PAIR: 0., PAIR[::-1]: .001},
])
def test_invalid_override_is_rejected(toy, cls, bad):
    with pytest.raises(ValueError):
        cls(toy[0], pair_buffer_overrides=bad)


def test_configured_ignored_pair_cannot_be_restored(toy):
    model, _, cfg = toy
    cfg['kinematics']['self_collision_ignore'] = {'pelvis': ['r_upper_leg']}
    with pytest.raises(ValueError):
        Distances(model, pair_buffer_overrides={PAIR: 0.})


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -.001, True, 1e308, 10**400])
def test_nonfinite_or_overflowing_global_offset_rejected(toy, value):
    with pytest.raises(ValueError):
        Distances(toy[0], buffer=value, pair_buffer_overrides={PAIR: 0.})


def test_arbitrarily_large_override_rejected(toy):
    with pytest.raises(ValueError):
        Distances(toy[0], pair_buffer_overrides={PAIR: 10**400})


def test_copy_canonical_provenance_and_immutable_padding(toy):
    supplied = {PAIR[::-1]: .001}
    geometry = Distances(toy[0], pair_buffer_overrides=supplied)
    supplied[PAIR[::-1]] = 0.
    assert dict(geometry.pair_buffer_overrides) == {PAIR: .001}
    assert geometry.provenance()['pair_buffer_overrides'] == [
        {'pair': list(PAIR), 'buffer_per_sphere_m': .001, 'gap_offset_m': .003}]
    with pytest.raises(TypeError):
        geometry.pair_buffer_overrides[PAIR] = 0.
    with pytest.raises(ValueError):
        geometry.pair_gap_offsets[0] = 0.
    published = geometry.provenance()
    published['pair_buffer_overrides'][0]['pair'][0] = 'tampered'
    assert geometry.provenance()['pair_buffer_overrides'][0]['pair'] == list(PAIR)


@pytest.mark.parametrize('native', [False, True])
def test_exact_pair_offset_only_and_gradients_unchanged(toy, native):
    model, data, _ = toy
    base = Distances(model)
    adjusted = Distances(model, pair_buffer_overrides={PAIR: 0.})
    if native:
        base.enable_native_contacts(); adjusted.enable_native_contacts()
    before = base.contacts(data, threshold=.3, per_pair=4)
    after = adjusted.contacts(data, threshold=.3, per_pair=4)
    expected = [(gap + (.005 if pair == PAIR else 0.), gradient, pair, ids)
                for gap, gradient, pair, ids in before]
    assert_contacts_equal(after, expected)
    for a, b in zip(base.radius, adjusted.radius):
        np.testing.assert_array_equal(a, b)
    assert base.pairs == adjusted.pairs
    assert base.collision(data)['pair'] == list(PAIR)
    assert adjusted.collision(data)['pair'] == ['pelvis', 'l_upper_leg']


@pytest.mark.parametrize('native', [False, True])
def test_thresholds_are_shifted_before_stop_first(toy, native):
    model, data, _ = toy
    geometry = Distances(model, pair_buffer_overrides={PAIR: 0.})
    if native:
        geometry.enable_native_contacts()
    # The first pair has buffered overlap but positive effective gap; the
    # native kernel must skip it before its stop-first decision.
    hits = geometry.contacts(data, threshold=0., derivatives=False, per_pair=1, stop_at_first=True)
    assert len(hits) == 1 and hits[0][2] == ('pelvis', 'l_upper_leg')
    horizons = np.full(len(geometry.pairs), .1)
    horizons[0] = .002
    assert all(h[2] != PAIR for h in geometry.contacts(data, .1, pair_thresholds=horizons))
    horizons[0] = .004
    assert any(h[2] == PAIR for h in geometry.contacts(data, .1, pair_thresholds=horizons))
    assert all(h[2] != PAIR for h in geometry.contacts(data, .002, pair_thresholds=horizons))


@pytest.mark.parametrize('cls,native', [(SphereGeometry, False), (Distances, False), (Distances, True)])
def test_raw_overlap_and_exact_zero_boundary(toy, cls, native):
    model, data, _ = toy
    # Binary-exact radii/buffer make a strict zero boundary unambiguous.
    geometry = cls(model, buffer=.03125, pair_buffer_overrides={PAIR: 0.})
    if native:
        geometry.enable_native_contacts()
    body = model.body('r_upper_leg').id
    data.xpos[model.body('l_upper_leg').id] = [10., 10., 10.]
    data.xpos[body] = [.125, 0., 0.]
    assert geometry.collision(data) is None
    data.xpos[body, 0] = .125 - 1e-9
    hit = geometry.collision(data)
    assert hit['pair'] == list(PAIR) and hit['margin_m'] == pytest.approx(-1e-9, abs=1e-16)
    data.xpos[body, 0] = .125 + 1e-9
    assert geometry.collision(data) is None


def test_native_exact_ties_use_same_numpy_order_after_padding(toy):
    model, data, cfg = toy
    for name in cfg['kinematics']['collision_link_names']:
        cfg['kinematics']['collision_spheres'][name] *= 2
    numpy = Distances(model, pair_buffer_overrides={PAIR: 0.})
    native = Distances(model, pair_buffer_overrides={PAIR: 0.})
    native.enable_native_contacts()
    for keep in (1, 3, 4):
        for first in (False, True):
            expected = numpy.contacts(data, .02, per_pair=keep, stop_at_first=first)
            actual = native.contacts(data, .02, per_pair=keep, stop_at_first=first)
            assert_contacts_equal(actual, expected)
    assert native._native_contacts.tie_fallbacks == 6


def test_native_negative_threshold_randomized_against_numpy(toy):
    model, data, _ = toy
    numpy = Distances(model, pair_buffer_overrides={PAIR: 0.})
    native = Distances(model, pair_buffer_overrides={PAIR: 0.})
    native.enable_native_contacts()
    rng = np.random.default_rng(107)
    for _ in range(60):
        q = model.qpos0.copy(); q[7:] = rng.uniform(-.04, .04, 2)
        fk(model, data, q)
        horizons = rng.uniform(0., .01, len(numpy.pairs))
        for threshold in (0., .003, .02):
            assert_contacts_equal(native.contacts(data, threshold, pair_thresholds=horizons),
                                  numpy.contacts(data, threshold, pair_thresholds=horizons), atol=1e-16)


@pytest.fixture(scope='module')
def actual107():
    evidence = json.loads((EVIDENCE / 'native107_collision_seed01/seed_probe.json').read_text())
    model_path = Path(evidence['model'])
    assert sha256(model_path) == evidence['model_sha256']
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    q = model.qpos0.copy()
    for name, value in zip(SDK_NAMES[:29], evidence['failed_tick']['q']):
        q[model.joint(name).qposadr[0]] = value
    fk(model, data, q)
    return model, data, q, evidence['geometry_provenance']['hand_envelope']


def test_actual107_seed_corridor_cache_and_active_arm_mask(actual107):
    model, data, q, hand = actual107
    base = Distances(model, hand_envelope=hand)
    adjusted = Distances(model, hand_envelope=hand, pair_buffer_overrides={PAIR: 0.})
    reference = SphereGeometry(model, hand_envelope=hand, pair_buffer_overrides={PAIR: 0.})
    arm_dofs = [model.joint(name).dofadr[0] for name in ARM_NAMES]
    assert base.collision(data)['margin_m'] == pytest.approx(-.001605707, abs=1e-9)
    assert base.corridor(data, q, q) is not None
    for native in (False, True):
        if native:
            adjusted.enable_native_contacts(); adjusted._collision_cache.clear()
        assert adjusted.collision(data) is None
        assert adjusted.collision(data) is None  # exact transform cache
        assert reference.collision(data) is None
        assert adjusted.corridor(data, q, q) is None
        np.testing.assert_array_equal(data.qpos, q)
        contacts = adjusted.contacts(data, .01, active_dofs=arm_dofs)
        assert_contacts_equal(contacts, base.contacts(data, .01, active_dofs=arm_dofs), atol=1e-12)
        assert all(c[2] != PAIR for c in contacts)
        target = [c for c in adjusted.contacts(data, .01) if c[2] == PAIR][0]
        assert target[0] == pytest.approx(.003394293, abs=1e-9)
        assert np.linalg.norm(target[1][np.asarray(arm_dofs)-6]) == 0.
    np.testing.assert_array_equal(adjusted._active_pair_masks[tuple(arm_dofs)],
                                  base._active_pair_masks[tuple(arm_dofs)])


def test_default_and_explicit_full_buffer_match_frozen_source(actual107):
    model, data, q, hand = actual107
    frozen = EVIDENCE / 'pair_padding_validation01/source_before/retarget_quality.py'
    spec = importlib.util.spec_from_file_location('igris.hl_ik._pair_padding_reference', frozen)
    module = importlib.util.module_from_spec(spec)
    # dataclasses consult sys.modules during class creation.
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    reference = module.Distances(model, hand_envelope=hand)
    defaults = [Distances(model, hand_envelope=hand),
                Distances(model, hand_envelope=hand, pair_buffer_overrides={}),
                Distances(model, hand_envelope=hand, pair_buffer_overrides={PAIR: .0025})]
    rng = np.random.default_rng(180)
    for _ in range(15):
        pose = q.copy(); pose[7:] += rng.normal(0., .01, len(q)-7)
        fk(model, data, pose)
        expected = reference.contacts(data, .03, per_pair=2)
        for geometry in defaults:
            assert_contacts_equal(geometry.contacts(data, .03, per_pair=2), expected)
            assert geometry.collision(data) == reference.collision(data)
    fk(model, data, q)


def test_cached_collision_returns_copy_and_changes_with_pose(toy):
    model, data, _ = toy
    geometry = Distances(model, pair_buffer_overrides={PAIR: 0.})
    hit = geometry.collision(data)
    hit['pair'][0] = 'changed'; hit['margin_m'] = 42.
    assert geometry.collision(data)['pair'] == ['pelvis', 'l_upper_leg']
    data.xpos[model.body('l_upper_leg').id] = [10., 0., 0.]
    assert geometry.collision(data) is None
