import copy
import numpy as np
import pytest


def state():
    return dict(caller_reference_schema='explicit_inplace_waist/v1',caller_sdk_indices=[0,1,2],
                waist_target_rad=[-.62,0.,.018],nominal_waist_target_rad=[-.62,0.,.018],
                command_xyz=[0.,0.,0.])


def test_publisher_and_solver_share_explicit_waist_without_rewriting_archive():
    from inplace_reference import apply_waist_reference, reference_error
    archive=np.arange(31,dtype=float)/100;archive[:3]=[.755,0.,.018]
    saved=archive.copy();s=state();before=copy.deepcopy(s)
    updated=apply_waist_reference(archive,s)
    np.testing.assert_array_equal(archive,saved)
    np.testing.assert_array_equal(updated[:3],s['waist_target_rad'])
    np.testing.assert_array_equal(updated[3:],archive[3:])
    assert reference_error(updated,archive,s)==0
    assert reference_error(archive,archive,s)>1.
    updated[7]+=.01
    assert reference_error(updated,archive,s)==pytest.approx(.01)
    assert s==before


@pytest.mark.parametrize('key,value',[
    ('caller_sdk_indices',[2,1,0]),('waist_target_rad',[2.,0.,0.]),
    ('nominal_waist_target_rad',[-.6,0.,.018]),('command_xyz',[.01,0.,0.]),
    ('waist_target_rad',[float('nan'),0.,0.]),('caller_reference_schema','unknown')])
def test_inconsistent_or_moving_inplace_reference_rejected(key,value):
    from inplace_reference import apply_waist_reference
    s=state();s[key]=value
    with pytest.raises(ValueError):apply_waist_reference(np.zeros(31),s)


def test_ordinary_archive_path_unchanged():
    from inplace_reference import apply_waist_reference
    q=np.arange(31,dtype=float)
    np.testing.assert_array_equal(apply_waist_reference(q,None),q)
