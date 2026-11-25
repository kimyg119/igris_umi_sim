"""Scope selection does not silently change original/research IK callers."""
import copy
import pytest
import live_ik


@pytest.mark.parametrize('manifest',[{}, {'htd_loaded_zero':None},
    {'htd_loaded_zero':{'enabled':True,'scope':'loaded_zero_only'}},
    {'htd_loaded_zero':{'enabled':True,'scope':'loaded_goal_hold_only'}}])
def test_nontransport_keeps_primary_warm_default(manifest):
    assert live_ik.primary_qp_start(manifest)=='warm'


def test_transport_requires_pinned_cold_selection_without_mutating_scope():
    manifest={'htd_loaded_zero':{'enabled':True,'scope':'loaded_transport',
        'transport':{'primary_qp_start':'cold'}}}
    before=copy.deepcopy(manifest)
    assert live_ik.primary_qp_start(manifest)=='cold' and manifest==before


@pytest.mark.parametrize('transport',[None,{}, {'primary_qp_start':'warm'},
                                     {'primary_qp_start':False}, {'primary_qp_start':'COLD'}])
def test_loaded_transport_cannot_silently_use_unreviewed_selection(transport):
    manifest={'htd_loaded_zero':{'enabled':True,'scope':'loaded_transport','transport':transport}}
    with pytest.raises(ValueError,match='primary QP'):live_ik.primary_qp_start(manifest)


def test_disabled_transport_descriptor_is_not_a_valid_cold_optin():
    with pytest.raises(ValueError,match='primary QP'):
        live_ik.primary_qp_start({'htd_loaded_zero':{'enabled':False,'scope':'loaded_transport',
                                                   'transport':{'primary_qp_start':'cold'}}})
