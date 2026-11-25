from types import SimpleNamespace
import numpy as np
import pytest

from htd_grasp_retention import GraspRetention


def snapshot(t=1.,forces=(1.,0.,1.,0.)):
    return SimpleNamespace(native_time=t,pose_native_time=t,force_native_time=t,
        palm_tip_force_N=np.array(forces),payload_pose=np.array([0.,0.,.9,1.,0.,0.,0.]),
        hand_poses=np.array([[0.,.1,.85,1.,0.,0.,0.],[0.,-.1,.85,1.,0.,0.,0.]]))


def test_capture_still_requires_both_hands_and_only_happens_once():
    g=GraspRetention(.1,4.)
    with pytest.raises(RuntimeError,match='both hands'):
        g.capture(snapshot(forces=(0,0,1,0)))
    g.capture(snapshot())
    with pytest.raises(RuntimeError,match='once'):
        g.capture(snapshot())


def test_brief_one_hand_zero_recovers_without_changing_reference():
    g=GraspRetention(.1,4.);g.capture(snapshot())
    g.check(snapshot(1.01,(0,0,.57,0)))
    g.check(snapshot(1.08,(0,0,.57,0)))
    assert g.metrics['contact_grace_active']
    g.check(snapshot(1.09))
    assert not g.metrics['contact_grace_active']
    assert g.events[-1]['event']=='both_hands_recovered'
    assert g.events[-1]['duration_s']==pytest.approx(.08)


def test_alternating_sides_does_not_restart_a_continuous_bad_interval():
    g=GraspRetention(.1,4.);g.capture(snapshot())
    g.check(snapshot(1.01,(0,0,.57,0)))
    g.check(snapshot(1.08,(.57,0,0,0)))
    with pytest.raises(RuntimeError,match='100ms'):
        g.check(snapshot(1.111,(.57,0,0,0)))


def test_a_late_recovery_after_an_unobserved_gap_does_not_erase_expired_grace():
    g=GraspRetention(.1,4.);g.capture(snapshot())
    g.check(snapshot(1.01,(0,0,.57,0)))
    with pytest.raises(RuntimeError,match='100ms'):
        g.check(snapshot(1.111))


@pytest.mark.parametrize('forces',[(0,0,0,0),(0,0,.2,0),(0,np.nan,1,0),(-1,0,2,0)])
def test_grace_never_bypasses_total_load_support_or_force_validity(forces):
    g=GraspRetention(.1,4.);g.capture(snapshot())
    with pytest.raises((RuntimeError,ValueError)):
        g.check(snapshot(1.01,forces))


@pytest.mark.parametrize('bad_contact',[False,True])
def test_relative_box_translation_and_rotation_are_guarded_even_with_good_forces(bad_contact):
    g=GraspRetention(.1,4.);g.capture(snapshot())
    s=snapshot(1.01,(0,0,.57,0) if bad_contact else (1,0,1,0))
    s.payload_pose[0]=.0151
    with pytest.raises(RuntimeError,match='relative translation'):
        g.check(s)
    s.payload_pose[:3]=[0,0,.9]
    s.payload_pose[3:]=[np.cos(.14),0.,0.,np.sin(.14)]
    with pytest.raises(RuntimeError,match='relative rotation'):
        g.check(s)


def test_world_rigid_motion_preserves_grasp_and_stale_or_future_samples_fail():
    g=GraspRetention(.1,4.);g.capture(snapshot());s=snapshot(1.01)
    s.payload_pose[:3]+=[.8,.2,.03];s.hand_poses[:,:3]+=[.8,.2,.03]
    g.check(s)
    assert max(g.metrics['relative_translation_m'])<1e-12
    s.pose_native_time=.9
    with pytest.raises(ValueError,match='pose'):
        g.check(s)
    s.pose_native_time=s.native_time;s.force_native_time=2.
    with pytest.raises(ValueError,match='force'):
        g.check(s)


def test_explicit25mm_only_applies_while_both_hands_are_qualified():
    g=GraspRetention(.1,4.,qualified_translation_limit_m=.025);g.capture(snapshot())
    s=snapshot(1.01);s.payload_pose[0]=.0249
    g.check(s)
    assert g.metrics['active_translation_limit_m']==.025
    s=snapshot(1.02,(0,0,.57,0));s.payload_pose[0]=.016
    with pytest.raises(RuntimeError,match='relative translation'):g.check(s)
    s=snapshot(1.03);s.payload_pose[0]=.0251
    with pytest.raises(RuntimeError,match='relative translation'):g.check(s)
    s=snapshot(1.04);s.payload_pose[3:]=[np.cos(.14),0,0,np.sin(.14)]
    with pytest.raises(RuntimeError,match='relative rotation'):g.check(s)
