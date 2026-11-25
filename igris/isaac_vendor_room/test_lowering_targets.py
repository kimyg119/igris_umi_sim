import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lowering_targets import WorldPositionLowering, phase_acceleration_limit


@pytest.mark.parametrize('phase',[None,'pickup','clearance','carry','lower','open','done'])
def test_acceleration_override_only_applies_after_walking_stops(phase):
    state=None if phase is None else {'phase':phase}
    assert phase_acceleration_limit(1.,None,state)==1.
    assert phase_acceleration_limit(1.,3.,state)==(3. if phase in ('lower','open','done') else 1.)


@pytest.mark.parametrize('bad',[0.,-1.,3.01,float('nan'),float('inf')])
def test_lowering_acceleration_override_remains_bounded(bad):
    with pytest.raises(ValueError):phase_acceleration_limit(1.,bad,{'phase':'lower'})


def fixture():
    hands=np.repeat(np.eye(4)[None],2,axis=0)
    hands[:,:3,3]=[[.4,.095,-.1],[.4,-.095,-.1]]
    torso=np.eye(4);torso[:3,3]=[0.,1.,1.]
    preload=np.array([[0.,-.003,0.],[0.,.004,0.]])
    return hands,torso,preload


def test_capture_keeps_accepted_goals_without_double_preload():
    hands,torso,preload=fixture()
    path=WorldPositionLowering(hands,torso,[.03,-.02,-.12],preload)
    np.testing.assert_allclose(path.targets(torso,0.,0.,preload),hands,atol=1e-14)
    extra=np.array([[0.,-.001,0.],[0.,.002,0.]])
    result=path.targets(torso,0.,0.,preload+extra)
    np.testing.assert_allclose(result[:,:3,3],hands[:,:3,3]+extra,atol=1e-14)


def test_world_positions_hold_when_torso_translates_and_rotates():
    hands,torso,preload=fixture();delta=np.array([.03,-.02,-.12])
    path=WorldPositionLowering(hands,torso,delta,preload)
    moved=torso.copy();moved[:3,3]+=[.04,-.015,.01]
    moved[:3,:3]=Rotation.from_euler('xyz',[.02,-.03,.04]).as_matrix()
    result=path.targets(moved,.5,0.,preload)
    world=moved[None]@result
    np.testing.assert_allclose(world[:,:3,3],(torso[None]@hands)[:,:3,3]+.5*delta,atol=1e-14)
    np.testing.assert_allclose(result[:,:3,:3],hands[:,:3,:3],atol=1e-14)


def test_opening_removes_incremental_force_correction_and_preserves_center():
    hands,torso,preload=fixture();delta=np.array([.03,-.02,-.12])
    path=WorldPositionLowering(hands,torso,delta,preload)
    opened=path.targets(torso,1.,1.,np.zeros((2,3)))
    expected=hands[:,:3,3]+delta+np.array([[0.,.055,0.],[0.,-.055,0.]])
    np.testing.assert_allclose(opened[:,:3,3],expected,atol=1e-14)
    # Intermediate release includes only the fraction of force correction
    # that the existing bounded force controller actually retains.
    middle=path.targets(torso,1.,.5,.5*preload)
    np.testing.assert_allclose(middle[:,:3,3],hands[:,:3,3]+delta+np.array([[0.,.0275,0.],[0.,-.0275,0.]]),atol=1e-14)
