"""Two startup variants compose only through their verified original-scene hash."""
import copy
import hashlib
import json

import pytest

from test_dip_damping import trial


def prepared(trial):
    import dip_damping
    import htd_backlash_limits as limits
    dip = dip_damping.configure(trial, .002)
    backlash = limits.configure(trial, .005, upper_timeconst=.005)
    manifest = json.loads((trial/'manifest.json').read_text())
    manifest['htd_upper_backlash_limit'] = backlash['upper_backlash_limit']
    manifest['htd_loaded_zero'] = dict(
        enabled=True, scope='loaded_transport', current_grasp_reference=True,
        transport={}, goal_hold={}, command_observer={}, wrist_mode_entry={},
        grasp_retention={}, dip_damping=dip,
        upper_backlash_limit=backlash['upper_backlash_limit'])
    return dip, manifest


def test_upper_chain_accepts_prospective_manifest_without_writing_it(trial):
    from dip_damping import validate
    dip, manifest = prepared(trial)
    before = (trial/'manifest.json').read_bytes()
    assert validate(trial, manifest=manifest) == dip
    assert (trial/'manifest.json').read_bytes() == before
    # Independently reloaded startup/worker validation requires persisted intent.
    with pytest.raises(ValueError):
        validate(trial)
    (trial/'manifest.json').write_text(json.dumps(manifest))
    assert validate(trial) == dip


@pytest.mark.parametrize('change', ['undeclared', 'mismatch', 'wrong_scope', 'original', 'audit'])
def test_upper_chain_rejects_selection_or_native_hash_tampering(trial, change):
    from dip_damping import validate
    _, manifest = prepared(trial)
    manifest = copy.deepcopy(manifest)
    if change == 'undeclared':
        del manifest['htd_upper_backlash_limit']
        del manifest['htd_loaded_zero']['upper_backlash_limit']
    elif change == 'mismatch':
        manifest['htd_loaded_zero']['upper_backlash_limit']['timeconst_s'] = .02
    elif change == 'wrong_scope':
        manifest['htd_loaded_zero']['scope'] = 'loaded_goal_hold_only'
    elif change == 'original':
        original = trial/'htd_backlash_original_scene.xml'
        original.write_bytes(original.read_bytes()+b'\n')
        path = trial/'htd_backlash_limit_audit.json'
        audit = json.loads(path.read_text())
        audit['original_scene_sha256'] = hashlib.sha256(original.read_bytes()).hexdigest()
        path.write_text(json.dumps(audit))
    else:
        path = trial/'htd_backlash_limit_audit.json'
        audit = json.loads(path.read_text())
        audit['native_audit']['changed_joints'] = 15
        path.write_text(json.dumps(audit))
    with pytest.raises(ValueError):
        validate(trial, manifest=manifest)
