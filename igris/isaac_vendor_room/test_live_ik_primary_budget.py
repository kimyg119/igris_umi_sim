"""Only the explicitly pinned loaded transport selects the short primary budget."""
import copy
import pytest
import live_ik


@pytest.mark.parametrize('manifest',[{}, {'htd_loaded_zero':None},
    {'htd_loaded_zero':{'enabled':True,'scope':'loaded_goal_hold_only',
                        'transport':{'primary_qp_iteration_limits':{'outer':10,'inner':10}}}}])
def test_default_scope_keeps_original_budget(manifest):
    assert live_ik.primary_qp_iteration_limits(manifest)=={'outer':50,'inner':50}


def manifest(limits):
    return {'htd_loaded_zero':{'enabled':True,'scope':'loaded_transport',
        'transport':{'primary_qp_start':'cold','primary_qp_iteration_limits':limits}}}


def test_transport_selects_exact_budget_without_mutating_manifest():
    source=manifest({'outer':10,'inner':10});before=copy.deepcopy(source)
    limits=live_ik.primary_qp_iteration_limits(source)
    assert limits=={'outer':10,'inner':10} and source==before
    limits['outer']=50
    assert source==before


@pytest.mark.parametrize('limits',[None,{},[10,10],{'outer':10},
    {'outer':50,'inner':50},{'outer':10.,'inner':10},
    {'outer':True,'inner':10},{'outer':10,'inner':10,'secondary':10}])
def test_transport_rejects_missing_or_unreviewed_budget(limits):
    with pytest.raises(ValueError,match='primary QP'):
        live_ik.primary_qp_iteration_limits(manifest(limits))


def test_transport_budget_requires_enabled_cold_selection():
    source=manifest({'outer':10,'inner':10});source['htd_loaded_zero']['enabled']=False
    with pytest.raises(ValueError,match='primary QP'):
        live_ik.primary_qp_iteration_limits(source)
