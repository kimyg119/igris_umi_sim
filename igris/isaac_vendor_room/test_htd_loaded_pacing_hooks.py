"""Explicit slower loaded-SIM pacing leaves native deadlines and defaults intact."""
import ast
import copy
import inspect
import json
import subprocess
import sys
from pathlib import Path
import pytest
import transport_trial as host
import run as native
from test_htd_loaded_hooks import manifest


def selected_manifest(tmp_path,monkeypatch,rate=.05):
    m=manifest(tmp_path,monkeypatch)
    m.update(hardware_executed=False,physics_backend='vendor MuJoCo 3.2.7',evaluation_profile='relaxed_transport')
    m['htd_loaded_zero'].update(loaded_rate=rate,scope='loaded_transport',current_grasp_reference=True)
    (tmp_path/'manifest.json').write_text(json.dumps(m))
    return m


def test_default_rate_keeps_prior_native_environment_exactly(tmp_path,monkeypatch):
    assert inspect.signature(host.run).parameters['htd_loaded_rate'].default==.1
    m=manifest(tmp_path,monkeypatch)
    assert native.loaded_start_environment(tmp_path,m)==(
        'export IGRIS_ROOM_PACER_SLOWDOWN='+str(tmp_path/'htd_pacing.request')+' IGRIS_ROOM_SLOW_REALTIME_FACTOR=.1')


def test_scoped_005_exports_only_selected_native_slowdown(tmp_path,monkeypatch):
    m=selected_manifest(tmp_path,monkeypatch);before=copy.deepcopy(m)
    env=native.loaded_start_environment(tmp_path,m)
    assert env==('export IGRIS_ROOM_PACER_SLOWDOWN='+str(tmp_path/'htd_pacing.request')+' IGRIS_ROOM_SLOW_REALTIME_FACTOR=.05')
    assert m==before and not (tmp_path/'htd_pacing.request').exists()


@pytest.mark.parametrize('field,bad',[('scope','loaded_goal_hold_only'),('current_grasp_reference',False),('current_grasp_reference',1),('enabled',False)])
def test_005_rejects_incomplete_loaded_scope(tmp_path,monkeypatch,field,bad):
    m=selected_manifest(tmp_path,monkeypatch);m['htd_loaded_zero'][field]=bad
    with pytest.raises(ValueError):native.loaded_start_environment(tmp_path,m)


@pytest.mark.parametrize('field,bad',[('hardware_executed',True),('hardware_executed',0),('physics_backend','other'),('evaluation_profile','original_grasp_requirements')])
def test_005_rejects_nonrelaxed_or_non_sim_manifest(tmp_path,monkeypatch,field,bad):
    m=selected_manifest(tmp_path,monkeypatch);m[field]=bad
    with pytest.raises(ValueError):native.loaded_start_environment(tmp_path,m)


@pytest.mark.parametrize('bad',[True,False,None,'0.05',.025,.2,float('nan'),float('inf')])
def test_invalid_rate_rejected_before_asset_or_output_creation(tmp_path,bad):
    out=tmp_path/'never-created'
    with pytest.raises(ValueError,match='loaded rate'):
        host.run(out,tmp_path/'missing-motion',htd_loaded_rate=bad)
    assert not out.exists()


@pytest.mark.parametrize('bad',[True,False,None,'0.05',.025,.2,float('nan'),float('inf')])
def test_invalid_manifest_rate_rejected(tmp_path,monkeypatch,bad):
    m=selected_manifest(tmp_path,monkeypatch);m['htd_loaded_zero']['loaded_rate']=bad
    with pytest.raises(ValueError):native.loaded_start_environment(tmp_path,m)


@pytest.mark.parametrize('flag',['loaded_transport','current_grasp_reference','relaxed_grasp'])
def test_005_option_needs_explicit_true_scope(flag):
    args=dict(loaded_transport=True,current_grasp_reference=True,relaxed_grasp=True)
    for bad in (False,1):
        args[flag]=bad
        with pytest.raises(ValueError,match='loaded rate'):
            host.validate_htd_loaded_rate(.05,**args)


@pytest.mark.parametrize('rate',[.1,.05])
def test_valid_option_exact_selection(rate):
    assert host.validate_htd_loaded_rate(rate,loaded_transport=True,current_grasp_reference=True,relaxed_grasp=True)==rate
    assert host.validate_htd_loaded_rate(.1,loaded_transport=False,current_grasp_reference=False,relaxed_grasp=False)==.1


@pytest.mark.parametrize('cli,expected',[([],.1),(['--htd-loaded-rate','.05'],.05)])
def test_real_cli_forwards_selected_rate_to_run(monkeypatch,tmp_path,cli,expected):
    # Execute the checked-in parser/entrypoint with only the trial launcher replaced.
    source=ast.parse(Path(host.__file__).read_text());entry=source.body[-1]
    assert isinstance(entry,ast.If)
    calls=[]
    def capture(*args,**kwargs):calls.append(inspect.signature(host.run).bind(*args,**kwargs).arguments)
    monkeypatch.setattr(sys,'argv',[host.__file__,'--out',str(tmp_path/'unused'),'--motion',str(tmp_path/'missing'),*cli])
    namespace=dict(vars(host),run=capture)
    exec(compile(ast.Module(body=entry.body,type_ignores=[]),host.__file__,'exec'),namespace)
    assert len(calls)==1 and calls[0]['htd_loaded_rate']==expected


def test_cli_005_wrong_scope_fails_before_inputs(tmp_path):
    out=tmp_path/'unused'
    result=subprocess.run([sys.executable,host.__file__,'--out',str(out),'--motion',str(tmp_path/'missing'),'--htd-loaded-rate','.05'],capture_output=True,text=True)
    assert result.returncode!=0 and 'unrecognized arguments' not in result.stderr
    assert 'loaded rate' in result.stderr and not out.exists()
