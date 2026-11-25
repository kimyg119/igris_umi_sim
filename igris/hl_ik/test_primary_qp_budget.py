"""Primary iteration budgets leave secondary and feasibility contracts intact."""
from pathlib import Path
import numpy as np
import pytest
import proxsuite
from .pipeline import PipelineIK
from .test_primary_qp_start import LIVE_MODEL, ROOT, problem


@pytest.mark.parametrize('name',['primary_max_iter','primary_max_iter_in'])
@pytest.mark.parametrize('value',[None,True,False,0,-1,10.,'10',np.int64(10)])
def test_budget_requires_positive_python_integer_before_loading_assets(name,value):
    with pytest.raises(ValueError,match=name):
        PipelineIK(model_path='/missing/model.xml',**{name:value})


def test_default_budget_and_secondary_remain_fifty():
    solver=PipelineIK(collision=False,model_path=LIVE_MODEL)
    assert solver.primary_max_iter==solver.primary_max_iter_in==50
    for stage in (0,1):
        _,ok=solver._qp(*problem(),stage)
        qp=solver._qps[(stage,2,2)]
        assert ok and qp.settings.max_iter==qp.settings.max_iter_in==50


def test_short_primary_only_keeps_warm_secondary_and_original_tolerances():
    solver=PipelineIK(collision=False,model_path=LIVE_MODEL,warm_start_primary=False,
                      primary_max_iter=10,primary_max_iter_in=10)
    for stage in (0,1):
        _,ok=solver._qp(*problem(),stage)
        assert ok
    primary=solver._qps[(0,2,2)];secondary=solver._qps[(1,2,2)]
    assert primary.settings.max_iter==primary.settings.max_iter_in==10
    assert secondary.settings.max_iter==secondary.settings.max_iter_in==50
    assert primary.settings.eps_abs==secondary.settings.eps_abs==1e-8
    assert primary.settings.eps_rel==secondary.settings.eps_rel==0
    solver._qp(*problem(),0);solver._qp(*problem(),1)
    assert solver._qps[(0,2,2)] is not primary
    assert solver._qps[(1,2,2)] is secondary
    assert secondary.settings.initial_guess==proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT


def test_cold_retry_keeps_short_budget_and_daqp_original_unit_feasibility(monkeypatch):
    actual=proxsuite.proxqp.dense.QP;created=[]
    def capture(*args,**kwargs):
        backend=actual(*args,**kwargs);created.append(backend);return backend
    monkeypatch.setattr(proxsuite.proxqp.dense,'QP',capture)
    solver=PipelineIK(collision=False,model_path=LIVE_MODEL,warm_start_primary=False,
                      primary_max_iter=10,primary_max_iter_in=10)
    z=np.load(Path(__file__).parent/'fixtures/feasible_ee_qp.npz')
    x,ok=solver._qp(z['H'],z['g'],z['C'],z['l'],z['u'],0)
    assert len(created)==2 and solver._qp_fallbacks==1 and ok
    assert all(q.settings.max_iter==q.settings.max_iter_in==10 for q in created)
    assert all(q.results.info.status!=proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED for q in created)
    assert np.all(z['C']@x>=z['l']-2e-8) and np.all(z['C']@x<=z['u']+2e-8)


def test_recorded104_primary_matrices_keep_original_constraints():
    capture=ROOT/'outputs/hlik_transport_20260918_2005/native104_QP_diagnosis01/primary_cold_last_matrices.npz'
    z=np.load(capture)
    solver=PipelineIK(collision=False,model_path=LIVE_MODEL,warm_start_primary=False,
                      primary_max_iter=10,primary_max_iter_in=10)
    names=[k[:-2]for k in z.files if k.endswith('_H') and z[k].shape==(14,14)]
    assert names
    for name in names:
        h,g,c,l,u=[z[f'{name}_{key}']for key in ('H','g','C','l','u')]
        x,ok=solver._qp(h,g,c,l,u,0)
        assert ok and np.isfinite(x).all()
        assert np.all(c@x>=l-2e-8) and np.all(c@x<=u+2e-8)
