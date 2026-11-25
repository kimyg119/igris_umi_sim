"""Explicit primary cold starts preserve QP gates and the default warm cache."""
from pathlib import Path
import numpy as np
import pytest
from .pipeline import PipelineIK


ROOT=Path(__file__).resolve().parents[2]
LIVE_MODEL=ROOT/'outputs/hlik_transport_20260918_2005/ik_model.xml'


@pytest.mark.parametrize('value',[None,0,1,'cold',np.bool_(False)])
def test_primary_warm_option_requires_actual_bool_before_assets(value):
    with pytest.raises(ValueError,match='warm_start_primary'):
        PipelineIK(warm_start_primary=value,model_path='/missing/model.xml')


def problem():
    return np.eye(2),np.array([.3,-.2]),np.eye(2),-np.ones(2),np.ones(2)


def test_default_primary_reuses_real_backend_and_original_solver_settings():
    solver=PipelineIK(collision=False,model_path=LIVE_MODEL)
    assert solver.warm_start_primary is True
    x,ok=solver._qp(*problem(),0);first=solver._qps[(0,2,2)]
    again,accepted=solver._qp(*problem(),0)
    assert first is solver._qps[(0,2,2)] and ok and accepted
    np.testing.assert_allclose(x,[-.3,.2],atol=1e-7)
    np.testing.assert_allclose(again,x,atol=1e-7)
    assert first.settings.eps_abs==1e-8 and first.settings.eps_rel==0
    assert first.settings.max_iter==50 and first.settings.max_iter_in==50


def test_cold_primary_constructs_fresh_backend_but_secondary_cache_is_unchanged():
    solver=PipelineIK(collision=False,model_path=LIVE_MODEL,warm_start_primary=False)
    primary,ok=solver._qp(*problem(),0);first_primary=solver._qps[(0,2,2)]
    secondary,ok2=solver._qp(*problem(),1);first_secondary=solver._qps[(1,2,2)]
    primary_again,ok3=solver._qp(*problem(),0)
    secondary_again,ok4=solver._qp(*problem(),1)
    assert all((ok,ok2,ok3,ok4))
    assert first_primary is not solver._qps[(0,2,2)]
    assert first_secondary is solver._qps[(1,2,2)]
    np.testing.assert_allclose(primary_again,primary,atol=1e-7)
    np.testing.assert_allclose(secondary_again,secondary,atol=1e-7)
    qp=solver._qps[(0,2,2)]
    assert qp.settings.eps_abs==1e-8 and qp.settings.eps_rel==0
    assert qp.settings.max_iter==50 and qp.settings.max_iter_in==50


def test_recorded103_primary_matrices_keep_original_unit_feasibility_and_solution():
    capture=ROOT/'outputs/hlik_transport_20260918_2005/native103_heartbeat_diagnosis01/last_tick_qp_matrices.npz'
    z=np.load(capture);solver=PipelineIK(collision=False,model_path=LIVE_MODEL,warm_start_primary=False)
    names=sorted(k[:-2]for k in z.files if k.endswith('_H') and z[k].shape==(14,14))
    assert len(names)>=5
    for name in names:
        h,g,c,l,u=[z[f'{name}_{key}']for key in ('H','g','C','l','u')]
        x,ok=solver._qp(h,g,c,l,u,0)
        assert ok and np.isfinite(x).all()
        assert np.all(c@x>=l-2e-8) and np.all(c@x<=u+2e-8)
        np.testing.assert_allclose(x,z[f'{name}_x'],atol=1e-8,rtol=0)


def test_cold_mode_keeps_existing_feasible_active_set_fallback():
    z=np.load(Path(__file__).parent/'fixtures/feasible_ee_qp.npz')
    solver=PipelineIK(collision=False,model_path=LIVE_MODEL,warm_start_primary=False)
    x,ok=solver._qp(z['H'],z['g'],z['C'],z['l'],z['u'],0)
    assert ok and solver._qp_fallbacks==1
    assert np.all(z['C']@x>=z['l']-2e-8) and np.all(z['C']@x<=z['u']+2e-8)
