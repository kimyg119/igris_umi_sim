import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from walking_transfer import pose_matrix


def setup():
    from inplace_transfer import InplaceTransfer
    pelvis=np.array([0.,0.,.87,1.,0.,0.,0.])
    torso=np.array([0.,0.,.94,1.,0.,0.,0.])
    box=np.array([.3,-.2,.85,1.,0.,0.,0.])
    local=np.repeat(np.eye(4)[None],2,axis=0);local[:,:3,3]=[[.3,-.1,-.13],[.3,-.3,-.13]]
    waist=np.array([.7556667619552057,0.,.01830421765232494])
    ctl=InplaceTransfer([.33,.2,.7561],.1,4.)
    return ctl,pelvis,torso,box,local,waist


def start(ctl,p,t,b,h,w):
    ctl.start(0.,p,t,b,h,w,[6.,0.,6.,0.],pickup_qualified=True)


def advance(ctl,p,t,b,w,end,*,source=0.,destination=0.,force=None):
    state=None
    for stamp in np.arange(ctl.last_time+.02,end+1e-7,.02):
        w[:]=ctl.waist_target
        state=ctl.update(float(stamp),p,t,b,w,[6,0,6,0] if force is None else force,source,destination)
        assert state['command_xyz']==[0.,0.,0.]
        assert state['waist_target_rad']==state['nominal_waist_target_rad']
    return state


def test_qualified_pickup_and_measured_clearance_required_before_waist_arc():
    ctl,p,t,b,h,w=setup()
    with pytest.raises(ValueError,match='qualified'):
        ctl.start(0.,p,t,b,h,w,[6,0,6,0],pickup_qualified=False)
    start(ctl,p,t,b,h,w)
    state=advance(ctl,p,t,b,w,9.)
    assert state['phase']=='clearance'
    np.testing.assert_allclose(np.array(state['torso_relative_hand_targets'])[:,:3,3],h[:,:3,3]+[0,0,.04])
    b[2]+=.04
    assert advance(ctl,p,t,b,w,10.,source=1.)['phase']=='clearance'
    state=advance(ctl,p,t,b,w,12.)
    assert state['phase']=='waist_arc'
    assert state['release_fraction']==0.


def test_arc_is_smooth_zero_walk_and_waits_for_actual_waist_and_body_settling():
    ctl,p,t,b,h,w=setup();start(ctl,p,t,b,h,w);b[2]+=.04
    advance(ctl,p,t,b,w,10.)
    began=ctl.phase_started
    initial=ctl.local_hands.copy()
    state=advance(ctl,p,t,b,w,began+10.)
    assert state['waist_target_rad'][0]==pytest.approx((.7556667619552057-.6200024167299453)/2,abs=1e-6)
    np.testing.assert_allclose(state['torso_relative_hand_targets'],initial)
    np.testing.assert_array_equal(state['waist_measured_rad'],w)
    np.testing.assert_allclose(state['waist_tracking_error_rad'],w-ctl.waist_target)
    # Supplying a constant old measured waist cannot qualify the final target.
    for stamp in np.arange(ctl.last_time+.02,began+21.+1e-7,.02):
        state=ctl.update(float(stamp),p,t,b,[.7556667619552057,0,.01830421765232494],[6,0,6,0],0,0)
    assert state['phase']=='waist_arc'
    state=advance(ctl,p,t,b,w,began+24.)
    assert state['phase']=='lower'
    assert state['waist_target_rad'][0]==pytest.approx(-.6200024167299453)
    assert state['world_position_lowering'] and state['lowering_duration_s']==12.


def lowered():
    ctl,p,t,b,h,w=setup();start(ctl,p,t,b,h,w);b[2]+=.04
    advance(ctl,p,t,b,w,34.)
    assert ctl.phase=='lower'
    return ctl,p,t,b,h,w


def test_time_and_source_force_cannot_release_without_destination_xyz_speed_support():
    ctl,p,t,b,h,w=lowered();lower_start=ctl.phase_started
    advance(ctl,p,t,b,w,lower_start+12.5,source=100.)
    assert ctl.phase=='lower'
    b[:3]=ctl.target+[.04,0,0]
    assert advance(ctl,p,t,b,w,lower_start+14.,destination=1.)['phase']=='lower'
    b[:3]=ctl.target
    state=advance(ctl,p,t,b,w,lower_start+16.,destination=1.)
    assert state['phase']=='open' and state['release_fraction']<1
    opened=ctl.phase_started
    state=advance(ctl,p,t,b,w,opened+6.9,destination=1.,force=[0,0,0,0])
    assert not state['done'] and state['release_fraction']==1.
    state=advance(ctl,p,t,b,w,opened+7.1,destination=1.,force=[0,0,0,0])
    assert state['done']
    pair=np.array(state['torso_relative_hand_targets'])
    assert np.linalg.norm(pair[0,:3,3]-pair[1,:3,3])==pytest.approx(.31)


def test_hybrid_metadata_preserves_exact_local_handoff_and_world_position_rule():
    from lowering_targets import WorldPositionLowering
    ctl,p,t,b,h,w=lowered()
    local=ctl.lower_local_start.copy();delta=ctl.lower_delta.copy()
    captured_torso=ctl.lower_torso.copy();helper=WorldPositionLowering(local,captured_torso,delta,np.zeros((2,3)))
    np.testing.assert_allclose(helper.targets(captured_torso,0,0,np.zeros((2,3))),local)
    moved=t.copy();moved[:3]+=[.01,.02,.003]
    moved[3:]=Rotation.from_euler('z',.02).as_quat(scalar_first=True)
    result=helper.targets(pose_matrix(moved),.5,0,np.zeros((2,3)))
    actual=pose_matrix(moved)[None]@result
    expected=captured_torso[None]@local
    np.testing.assert_allclose(actual[:,:3,3],expected[:,:3,3]+.5*delta)
    np.testing.assert_allclose(result[:,:3,:3],local[:,:3,:3])


def test_palm_loss_or_clearance_timeout_stops_without_opening():
    ctl,p,t,b,h,w=setup();start(ctl,p,t,b,h,w)
    with pytest.raises(RuntimeError,match='palm'):
        advance(ctl,p,t,b,w,1.,force=[0,0,6,0])
    assert ctl.phase=='clearance'
    ctl,p,t,b,h,w=setup();start(ctl,p,t,b,h,w)
    with pytest.raises(RuntimeError,match='clearance'):
        advance(ctl,p,t,b,w,21.)
    assert ctl.phase=='clearance'


def test_invalid_time_and_waist_limits_fail_without_input_mutation():
    ctl,p,t,b,h,w=setup();snap=[x.copy()for x in(p,t,b,h,w)]
    bad=w.copy();bad[0]=2.
    with pytest.raises(ValueError,match='waist'):
        start(ctl,p,t,b,h,bad)
    start(ctl,p,t,b,h,w)
    with pytest.raises(ValueError,match='time'):
        ctl.update(-1.,p,t,b,w,[6,0,6,0],0.,0.)
    for before,after in zip(snap,(p,t,b,h,w)):np.testing.assert_array_equal(before,after)
