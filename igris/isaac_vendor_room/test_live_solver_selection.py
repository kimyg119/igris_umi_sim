import hashlib
import pytest
import run


def selection(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'ROOT',tmp_path)
    source=tmp_path/'igris/hl_ik/pipeline.py'
    source.parent.mkdir(parents=True);source.write_text('original solver source')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    manifest={'htd_loaded_zero':dict(enabled=True,scope='loaded_transport',
        transport={'primary_qp_start':'cold','primary_qp_iteration_limits':{'outer':10,'inner':10}},
        setup_sha256={str(source):digest})}
    ready=dict(primary_qp_start='cold',pipeline_source_sha256=digest,
               primary_qp_iteration_limits={'outer':10,'inner':10})
    return manifest,ready,source


def test_loaded_transport_checks_actual_solver_and_worker_selection(tmp_path,monkeypatch):
    manifest,ready,source=selection(tmp_path,monkeypatch)
    assert run.validate_live_solver_selection(manifest,ready)==ready


@pytest.mark.parametrize('bad',['warm_worker','warm_scope','stale_worker','stale_setup','changed_source',
                              'missing_budget','wrong_worker_budget','wrong_scope_budget'])
def test_wrong_or_stale_selection_rejected_before_acquisition(tmp_path,monkeypatch,bad):
    manifest,ready,source=selection(tmp_path,monkeypatch)
    if bad=='warm_worker':ready['primary_qp_start']='warm'
    if bad=='warm_scope':manifest['htd_loaded_zero']['transport']['primary_qp_start']='warm'
    if bad=='stale_worker':ready['pipeline_source_sha256']='old'
    if bad=='stale_setup':manifest['htd_loaded_zero']['setup_sha256'][str(source)]='old'
    if bad=='changed_source':source.write_text('changed solver source')
    if bad=='missing_budget':ready.pop('primary_qp_iteration_limits')
    if bad=='wrong_worker_budget':ready['primary_qp_iteration_limits']['outer']=50
    if bad=='wrong_scope_budget':manifest['htd_loaded_zero']['transport']['primary_qp_iteration_limits']['inner']=50
    with pytest.raises(RuntimeError,match='solver selection'):
        run.validate_live_solver_selection(manifest,ready)


@pytest.mark.parametrize('scope',[None,'loaded_zero_only','loaded_goal_hold_only'])
def test_existing_modes_do_not_require_new_solver_mode(tmp_path,monkeypatch,scope):
    monkeypatch.setattr(run,'ROOT',tmp_path/'missing')
    manifest={} if scope is None else {'htd_loaded_zero':dict(enabled=True,scope=scope)}
    assert run.validate_live_solver_selection(manifest,{}) is None
