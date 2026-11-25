import pytest
from pathlib import Path
import transport_trial as trial
import run
import sys
from types import SimpleNamespace


def test_default_damping_option_is_inert_for_all_scopes():
    assert trial.validate_dip_damping_option(None,False,False) is None


def test_reviewed_damping_requires_explicit_relaxed_loaded_transport():
    assert trial.validate_dip_damping_option(.002,True,True)==.002


@pytest.mark.parametrize('value,relaxed,transport',[
    (.002,False,True),(.002,True,False),(.0001,True,True),(.005,True,True),
    (float('nan'),True,True),(float('inf'),True,True),(-.002,True,True),
    (True,True,True),('0.002',True,True),
])
def test_unreviewed_or_wrong_scope_rejected(value,relaxed,transport):
    with pytest.raises(ValueError,match='DIP damping'):
        trial.validate_dip_damping_option(value,relaxed,transport)


def test_wrong_scope_rejected_before_creating_trial(tmp_path):
    out=tmp_path/'fresh'
    with pytest.raises(ValueError,match='DIP damping'):
        trial.run(out,Path('/missing-motion'),dip_damping=.002)
    assert not out.exists()


@pytest.mark.parametrize('manifest',[
    {'dip_damping_requested':.002},
    {'htd_loaded_zero':{'scope':'loaded_goal_hold_only'}},
])
def test_native_start_rejects_dip_variant_outside_loaded_transport(tmp_path,monkeypatch,manifest):
    monkeypatch.setitem(sys.modules,'dip_damping',SimpleNamespace(validate=lambda out:{'value':.002}))
    with pytest.raises(ValueError,match='DIP damping'):
        run.loaded_start_environment(tmp_path,manifest)


@pytest.mark.parametrize('name',['dip_damping.json','dip_damping_audit','dip_damping_candidate_scene.xml'])
def test_orphan_dip_artifact_cannot_hide_in_normal_start(tmp_path,monkeypatch,name):
    (tmp_path/name).write_text('{}')
    calls=[]
    monkeypatch.setitem(sys.modules,'dip_damping',SimpleNamespace(validate=lambda out:calls.append(out) or {'value':.002}))
    with pytest.raises(ValueError,match='DIP damping'):
        run.loaded_start_environment(tmp_path,{})
    assert calls==[tmp_path]
