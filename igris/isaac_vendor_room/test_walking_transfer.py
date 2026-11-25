import numpy as np
import pytest

from walking_transfer import WalkingTransfer, pose_matrix


def test_stationary_loaded_probe_ignores_unused_cart_heading_but_requires_clearance():
    _,p,t,b,h=fixture()
    ctl=WalkingTransfer([1.16,-.26,.758],-.49,.1,4.,clearance_lift=.04,
                        heading_preserving_route=True,stationary_only=True)
    ctl.start(0.,p,t,b,h)
    for stamp in np.arange(.02,8.02,.02):
        state=ctl.update(float(stamp),p,t,b,[6,0,6,0],0.)
        assert state['phase']=='clearance'
        assert not np.any(state['command_xyz'])
    b[2]+=.04
    for stamp in np.arange(8.02,20.,.02):
        state=ctl.update(float(stamp),p,t,b,[6,0,6,0],0.)
        assert not np.any(state['command_xyz'])
        assert state['navigation_route'] is None
        assert state['release_fraction']==0.
    assert state['phase']=='carry' and state['clearance_ready_s']>=.5
    np.testing.assert_allclose(state['goal_xy'],p[:2])


def test_heading_route_requires_actual_departure_before_forward_and_final_lowering():
    _,p,t,b,h=fixture()
    ctl=WalkingTransfer([1.16,-.26,.758],-np.pi/4,.1,4.,heading_preserving_route=True)
    ctl.start(0.,p,t,b,h)
    assert ctl.goal_yaw==pytest.approx(0.)
    original_p,original_t,original_b=p.copy(),t.copy(),b.copy()
    stamp=0.
    def hold(offset,seconds):
        nonlocal stamp
        for pose,original in ((p,original_p),(t,original_t),(b,original_b)):
            pose[:2]=original[:2]+offset
        for _ in range(round(seconds/.02)):
            stamp+=.02
            state=ctl.update(stamp,p,t,b,[6,0,6,0],0.)
        return state
    state=hold([0.,0.],4.)
    assert state['phase']=='carry' and state['navigation_route']['waypoint_index']==0
    assert abs(state['command_xyz'][0])<1e-12 and state['command_xyz'][1]>.4
    state=hold([0.,.04],4.)
    assert state['navigation_route']['waypoint_index']==0  # cannot cut through source
    state=hold([0.,.065],4.)
    assert state['phase']=='carry' and state['navigation_route']['waypoint_index']==1
    state=hold([.38,.085],4.)
    assert state['phase']=='carry' and state['navigation_route']['final_waypoint']
    np.testing.assert_allclose(ctl.goal_xy,[.9,0.],atol=1e-12)
    state=hold([.9,0.],4.)
    assert state['phase']=='lower' and not np.any(state['command_xyz'])
    assert state['release_fraction']==0.  # navigating does not establish support


def test_heading_route_restarts_from_measured_post_clearance_state():
    _,p,t,b,h=fixture()
    ctl=WalkingTransfer([1.16,-.26,.758],-np.pi/4,.1,4.,clearance_lift=.04,heading_preserving_route=True)
    ctl.start(0.,p,t,b,h)
    for stamp in np.arange(.02,9.,.02):ctl.update(float(stamp),p,t,b,[6,0,6,0],0.)
    p[:2]+=[.01,-.01];t[:2]+=[.01,-.01];b[:3]+=[.02,-.015,.04]
    for stamp in np.arange(9.,12.,.02):
        state=ctl.update(float(stamp),p,t,b,[6,0,6,0],0.)
        if state['phase']=='carry':break
    assert state['phase']=='carry'
    np.testing.assert_allclose(ctl.route.start,p[:2])
    np.testing.assert_allclose(ctl.route.waypoints[-1],ctl.target[:2]-(b[:2]-p[:2]))
    assert ctl.route.index==0


def test_intermediate_plane_crossing_while_moving_does_not_require_precise_stop():
    _,p,t,b,h=fixture()
    ctl=WalkingTransfer([1.16,-.26,.758],-np.pi/4,.1,4.,heading_preserving_route=True)
    ctl.start(0.,p,t,b,h)
    for pose in (p,t,b):pose[1]+=.07
    state=ctl.update(.1,p,t,b,[6,0,6,0],0.)
    assert np.linalg.norm(ctl.velocity)>.03
    assert state['navigation_route']['waypoint_index']==1
    assert state['phase']=='carry' and state['settled_s']==0.
    # A later lateral regression must request lateral clearance before forward.
    for pose in (p,t,b):pose[1]-=.02
    state=ctl.update(.12,p,t,b,[6,0,6,0],0.)
    assert state['command_xyz'][0]==pytest.approx(0.)
    assert state['command_xyz'][1]>0.
    for pose in (p,t,b):pose[0]+=.36;pose[1]+=.04
    state=ctl.update(.2,p,t,b,[6,0,6,0],0.)
    assert state['navigation_route']['final_waypoint']
    assert state['phase']=='carry'  # final landing still requires measured stop


def test_heading_route_cannot_lower_outside_audited_35mm_arrival_samples():
    _,p,t,b,h=fixture()
    ctl=WalkingTransfer([1.16,-.26,.758],-np.pi/4,.1,4.,heading_preserving_route=True)
    ctl.start(0.,p,t,b,h)
    ctl.route.index=2;ctl.goal_xy=ctl.route.goal
    p[0]=.86;t[0]+=.86;b[0]+=.86
    for stamp in np.arange(.02,5.,.02):
        state=ctl.update(float(stamp),p,t,b,[6,0,6,0],0.)
    assert state['phase']=='carry'
    assert np.linalg.norm(state['command_xyz'][:2])>.2
    p[0]=.87;t[0]+=.01;b[0]+=.01
    for stamp in np.arange(5.,10.,.02):
        state=ctl.update(float(stamp),p,t,b,[6,0,6,0],0.)
    assert state['phase']=='lower'


def test_final_braking_waits_for_gait_to_stop_before_restarting_outside_arrival_radius():
    _,p,t,b,h=fixture()
    ctl=WalkingTransfer([1.16,-.26,.758],-np.pi/4,.1,4.,heading_preserving_route=True)
    ctl.start(0.,p,t,b,h);ctl.route.index=2;ctl.goal_xy=ctl.route.goal
    p[0]=.855;t[0]+=.855;b[0]+=.855
    ctl.last_xy=p[:2].copy();ctl.velocity[:]=[.1,0.]
    state=ctl.update(.02,p,t,b,[6,0,6,0],0.)
    assert state['phase']=='carry' and not np.any(state['command_xyz'])
    assert state['arrival_coasting']
    state=ctl.update(.5,p,t,b,[6,0,6,0],0.)
    assert not np.any(state['command_xyz'])  # 45mm outside35mm, still braking
    assert state['phase']=='carry'
    state=ctl.update(2.6,p,t,b,[6,0,6,0],0.)
    assert state['command_xyz'][0]>0.
    assert not state['arrival_coasting']


def fixture():
    pelvis=np.array([0.,0.,.82,1.,0.,0.,0.])
    a=-np.pi/4
    torso=np.array([0.,0.,.94,np.cos(a/2),0.,0.,np.sin(a/2)])
    box=np.array([.26,-.26,.85,1.,0.,0.,0.])
    hands=np.repeat(np.eye(4)[None],2,axis=0)
    hands[:,:3,3]=[[.31,-.21,.85],[.21,-.31,.85]]
    ctl=WalkingTransfer([.028,1.396,.758],np.pi/4,.25,1.2)
    ctl.start(0.,pelvis,torso,box,hands)
    return ctl,pelvis,torso,box,hands


def test_carry_is_rigid_in_torso_and_goal_accounts_for_waist_heading():
    ctl,pelvis,torso,box,hands=fixture()
    assert ctl.goal_yaw==pytest.approx(np.pi/2)
    np.testing.assert_allclose(ctl.goal_xy,[-.232,1.136],atol=1e-12)
    moved=torso.copy();moved[:3]+=[.1,.2,.01]
    state=ctl.update(.02,pelvis,moved,box,[3,0,3,0],0.)
    np.testing.assert_allclose(np.array(state['world_hand_targets'])[:,:3,3],hands[:,:3,3]+[.1,.2,.01])
    np.testing.assert_allclose(pose_matrix(moved)[None]@np.array(state['torso_relative_hand_targets']),state['world_hand_targets'])
    assert np.linalg.norm(state['command_xyz'][:2])<=.10*.02+1e-12
    assert state['phase']=='carry'


def test_lost_palm_stops_without_releasing_box():
    ctl,p,t,b,h=fixture()
    ctl.update(.1,p,t,b,[0,0,3,0],0)
    with pytest.raises(RuntimeError,match='palm'):
        ctl.update(.8,p,t,b,[0,0,3,0],0)
    assert ctl.phase=='carry'


def test_explicit_held_goal_scope_counts_tip_support_without_releasing():
    ctl,p,t,b,h=fixture()
    for time in np.arange(.02,1.,.02):
        state=ctl.update(float(time),p,t,b,[0,3,0,3],0.,hold_carry=True,combined_carry_contact=True)
        assert state['phase']=='carry' and state['release_fraction']==0.
        assert not np.any(state['command_xyz'])
    with pytest.raises(ValueError,match='held carry'):
        ctl.update(1.,p,t,b,[0,3,0,3],0.,combined_carry_contact=True)
    with pytest.raises(RuntimeError,match='palm'):
        ctl.update(1.7,p,t,b,[0,0,0,3],0.,hold_carry=True,combined_carry_contact=True)


def test_release_requires_measured_support_at_destination():
    ctl,p,t,b,h=fixture()
    p[:2]=ctl.goal_xy;p[3:]=[np.cos(ctl.goal_yaw/2),0,0,np.sin(ctl.goal_yaw/2)]
    t[:2]=p[:2];t[3:]=[np.cos(np.pi/8),0,0,np.sin(np.pi/8)]
    for time in np.arange(.02,3.1,.02):
        state=ctl.update(float(time),p,t,b,[3,0,3,0],0)
    assert state['phase']=='lower'
    b[:3]=ctl.target
    for time in np.arange(3.1,13,.02):
        state=ctl.update(float(time),p,t,b,[3,0,3,0],0)
    assert state['phase']=='lower'
    for time in np.arange(13.,14.1,.02):
        state=ctl.update(float(time),p,t,b,[3,0,3,0],2.5)
    assert state['phase']=='open'
    for time in np.arange(14.1,21,.02):
        state=ctl.update(float(time),p,t,b,[0,0,0,0],2.5)
    assert state['done']
    assert state['release_fraction']==1.
    assert not np.any(state['command_xyz'])


def test_bad_clock_or_unqualified_start_rejected():
    ctl,p,t,b,h=fixture()
    with pytest.raises(ValueError,match='time'):
        ctl.update(-1,p,t,b,[3,0,3,0],0)
    with pytest.raises(ValueError):pose_matrix([0]*7)


def test_native_readonly_pose_buffer_is_supported_without_mutation():
    values=np.array([.1,.2,.9,1.,0.,0.,0.])
    payload=values.tobytes();readonly=np.frombuffer(payload,dtype=np.float64)
    assert not readonly.flags.writeable
    np.testing.assert_allclose(pose_matrix(readonly)[:3,3],values[:3])
    assert readonly.tobytes()==payload


def test_carry_handoff_preserves_validated_local_targets_despite_torso_filter_lag():
    _,p,t,b,h=fixture()
    filtered=t.copy();filtered[0]+=.004
    validated_local=np.linalg.inv(pose_matrix(filtered))[None]@h
    ctl=WalkingTransfer([.028,1.396,.758],np.pi/4,.25,1.2)
    ctl.start(0.,p,t,b,h,torso_relative_hands=validated_local)
    state=ctl.update(.02,p,t,b,[3,0,3,0],0.)
    np.testing.assert_allclose(state['torso_relative_hand_targets'],validated_local,atol=1e-12)
    np.testing.assert_allclose(state['world_hand_targets'],pose_matrix(t)[None]@validated_local)
    assert not np.allclose(state['world_hand_targets'],h)


def test_vendor_v5_deadband_scaling_and_stop_hysteresis():
    ctl,p,t,b,h=fixture()
    for time in np.arange(.02,3.,.02):
        state=ctl.update(float(time),p,t,b,[3,0,3,0],0.)
    assert abs(state['command_xyz'][2])>.2
    assert np.linalg.norm(state['command_xyz'][:2])==0. # turn before translation
    p[3:]=[np.cos(ctl.goal_yaw/2),0,0,np.sin(ctl.goal_yaw/2)]
    for time in np.arange(3.,7.,.02):
        state=ctl.update(float(time),p,t,b,[3,0,3,0],0.)
    raw=np.asarray(state['command_xyz']); effective=raw*np.array([1.,.5,1.])
    assert .2<np.linalg.norm(effective[:2])<=.23+1e-12
    assert effective[2]==0.
    p[:2]=ctl.goal_xy+[.025,0.]
    for time in np.arange(7.,7.5,.02):
        state=ctl.update(float(time),p,t,b,[3,0,3,0],0.)
    assert not np.any(state['command_xyz'])
    p[0]+=.01 # still inside the restart radius
    state=ctl.update(7.5,p,t,b,[3,0,3,0],0.)
    assert not np.any(state['command_xyz'])


def test_clearance_keeps_zero_twist_until_measured_lift_and_stability():
    _,p,t,b,h=fixture()
    ctl=WalkingTransfer([.028,1.396,.758],np.pi/4,.25,1.2,clearance_lift=.04)
    ctl.start(0.,p,t,b,h)
    original=ctl.local_hands.copy();initial_xy=ctl.goal_xy.copy()
    for time in np.arange(.02,9.,.02):
        state=ctl.update(float(time),p,t,b,[3,0,3,0],0.)
        assert state['phase']=='clearance' and not np.any(state['command_xyz'])
    np.testing.assert_allclose(np.array(state['world_hand_targets'])[:,:3,3],h[:,:3,3]+[0,0,.04])
    b=b.copy();b[:3]+=[.02,0,.04]
    for time in np.arange(9.,10.,.02):
        state=ctl.update(float(time),p,t,b,[3,0,3,0],2.5)
        assert state['phase']=='clearance' and not np.any(state['command_xyz'])
    for time in np.arange(10.,12.,.02):
        state=ctl.update(float(time),p,t,b,[3,0,3,0],0.)
        if state['phase']=='carry':break
    assert state['phase']=='carry'
    # Exact goal continuity, recomputed body destination from the lifted box.
    np.testing.assert_allclose(np.array(state['torso_relative_hand_targets'])[:,:3,3],original[:,:3,3]+[0,0,.04])
    np.testing.assert_allclose(ctl.goal_xy,initial_xy+[0,-.02],atol=1e-12)


def test_clearance_retains_contact_guard_and_cannot_time_out_into_walking():
    _,p,t,b,h=fixture()
    for missing_contact in [True,False]:
        ctl=WalkingTransfer([.028,1.396,.758],np.pi/4,.25,1.2,clearance_lift=.04)
        ctl.start(0.,p,t,b,h)
        with pytest.raises(RuntimeError,match='palm'if missing_contact else 'clearance'):
            ctl.update(.8 if missing_contact else 21.,p,t,b,[0,0,3,0]if missing_contact else[3,0,3,0],0.)
        assert ctl.phase=='clearance' and not np.any(ctl.command)


def test_forward_ramp_survives_gait_yaw_oscillation_after_turn_settles():
    ctl,p,t,b,h=fixture()
    def heading(angle):p[3:]=[np.cos(angle/2),0,0,np.sin(angle/2)]
    heading(ctl.goal_yaw)
    state=ctl.update(.02,p,t,b,[3,0,3,0],0.)
    assert abs(ctl.yaw_rate)>.06
    assert not np.any(state['command_xyz'][:2]) # first residual turn must settle
    for stamp in np.arange(.04,5.,.02):
        state=ctl.update(float(stamp),p,t,b,[3,0,3,0],0.)
    assert np.linalg.norm(state['vendor_command_before_filter'][:2])==pytest.approx(.23)
    heading(ctl.goal_yaw+.025) # small heading error, ordinary gait angular motion
    state=ctl.update(5.,p,t,b,[3,0,3,0],0.)
    assert abs(ctl.yaw_rate)>.06
    assert .22<np.linalg.norm(state['vendor_command_before_filter'][:2])<=.23
    heading(ctl.goal_yaw+.10) # meaningful heading error must stop forward/re-turn
    state=ctl.update(5.02,p,t,b,[3,0,3,0],0.)
    assert not np.any(state['command_xyz'][:2])
    heading(ctl.goal_yaw-.025)
    state=ctl.update(5.04,p,t,b,[3,0,3,0],0.)
    assert abs(ctl.yaw_rate)>.06
    assert not np.any(state['command_xyz'][:2]) # previous qualification was cleared


def test_xy_stop_clears_heading_qualification_before_forward_restart():
    ctl,p,t,b,h=fixture()
    p[3:]=[np.cos(ctl.goal_yaw/2),0,0,np.sin(ctl.goal_yaw/2)]
    for stamp in np.arange(.02,5.,.02):
        state=ctl.update(float(stamp),p,t,b,[3,0,3,0],0.)
    assert np.linalg.norm(state['vendor_command_before_filter'][:2])==pytest.approx(.23)
    p[:2]=ctl.goal_xy
    state=ctl.update(5.,p,t,b,[3,0,3,0],0.)
    assert not np.any(state['command_xyz'][:2])
    p[0]+=.2
    angle=ctl.goal_yaw+.025
    p[3:]=[np.cos(angle/2),0,0,np.sin(angle/2)]
    state=ctl.update(5.02,p,t,b,[3,0,3,0],0.)
    assert abs(ctl.yaw_rate)>.06
    assert not np.any(state['command_xyz'][:2])


def test_fast_heading_overshoot_stops_forward_during_predictive_turn_braking():
    ctl,p,t,b,h=fixture()
    angle=ctl.goal_yaw-.07
    # First finish the turn at the exact heading, then keep a small residual
    # heading offset long enough for angular speed to settle.
    p[3:]=[np.cos(ctl.goal_yaw/2),0,0,np.sin(ctl.goal_yaw/2)]
    for stamp in np.arange(.02,4.,.02):
        ctl.update(float(stamp),p,t,b,[3,0,3,0],0.)
    p[3:]=[np.cos(angle/2),0,0,np.sin(angle/2)]
    for stamp in np.arange(4.,6.,.02):
        state=ctl.update(float(stamp),p,t,b,[3,0,3,0],0.)
    assert np.linalg.norm(state['vendor_command_before_filter'][:2])>.22
    angle=ctl.goal_yaw+.10
    p[3:]=[np.cos(angle/2),0,0,np.sin(angle/2)]
    state=ctl.update(6.,p,t,b,[3,0,3,0],0.)
    assert abs(ctl.yaw_rate)>.7
    assert not ctl.turning # predictive angular braking can remain appropriate
    assert not np.any(state['command_xyz'][:2])


@pytest.mark.parametrize("hybrid",[False,True])
def test_optional_local_lowering_preserves_local_goals_through_torso_motion(hybrid):
    _,p,t,b,h=fixture()
    ctl=WalkingTransfer([.028,1.396,.758],np.pi/4,.1,4.,torso_relative_lowering=True,world_position_lowering=hybrid)
    ctl.start(0.,p,t,b,h)
    local=ctl.local_hands.copy()
    p[:2]=ctl.goal_xy;p[3:]=[np.cos(ctl.goal_yaw/2),0,0,np.sin(ctl.goal_yaw/2)]
    for stamp in np.arange(.02,4.,.02):
        state=ctl.update(float(stamp),p,t,b,[6,0,6,0],0.)
        if state['phase']=='lower':break
    assert state['phase']=='lower'
    start=ctl.phase_started
    assert state['world_position_lowering']==hybrid and state['lowering_fraction']==0.
    np.testing.assert_allclose(state['lowering_delta_world_m'],ctl.target-b[:3])
    np.testing.assert_allclose(state['torso_relative_hand_targets'],local,atol=1e-12)
    delta_local=pose_matrix(t)[:3,:3].T@(ctl.target-b[:3])
    moved=t.copy();moved[:3]+=[.01,-.005,.002]
    state=ctl.update(start+6.,p,moved,b,[6,0,6,0],0.)
    expected=local.copy();expected[:,:3,3]+=.5*delta_local
    np.testing.assert_allclose(state['torso_relative_hand_targets'],expected,atol=1e-12)
    np.testing.assert_allclose(state['world_hand_targets'],pose_matrix(moved)[None]@expected,atol=1e-12)
    assert state['phase']=='lower' and state['release_fraction']==0.
    # A completed geometric path alone still cannot release the payload.
    state=ctl.update(start+12.,p,moved,b,[6,0,6,0],0.)
    assert state['phase']=='lower' and state['release_fraction']==0.
    b[:3]=ctl.target
    for stamp in np.arange(start+12.02,start+14.01,.02):
        state=ctl.update(float(stamp),p,moved,b,[6,0,6,0],1.)
    assert state['phase']=='open'
    for stamp in np.arange(start+14.02,start+22.,.02):
        state=ctl.update(float(stamp),p,moved,b,[0,0,0,0],1.)
    assert state['done'] and state['release_fraction']==1.
    opened=np.asarray(state['torso_relative_hand_targets'])
    assert np.linalg.norm(opened[0,:3,3]-opened[1,:3,3])==pytest.approx(np.linalg.norm(local[0,:3,3]-local[1,:3,3])+.110)
    np.testing.assert_allclose(state['world_hand_targets'],pose_matrix(moved)[None]@opened,atol=1e-12)


def external_fixture():
    _,p,t,b,h=fixture()
    ctl=WalkingTransfer([1.16,-.26,.758],-np.pi/4,.1,4.,heading_preserving_route=True,
                        torso_relative_lowering=True,world_position_lowering=True,xy_first_lowering=True)
    ctl.start(0.,p,t,b,h)
    return ctl,p,t,b


def arrival_signal(stamp, *, final=True, settled=True):
    return dict(sample_time_s=stamp,goal_xy_world=[.9,0.],heading_goal_rad=0.,
                route_final=final,feedback_settled=settled)


def test_external_navigation_never_uses_vendor_twist_and_requires_actual_final_arrival():
    ctl,p,t,b=external_fixture();stamp=0.
    for pose in (p,t,b):pose[0]+=.9
    def hold(seconds,**kw):
        nonlocal stamp
        for _ in range(round(seconds/.02)):
            stamp+=.02
            state=ctl.update(stamp,p,t,b,[0,6,0,6],0.,combined_carry_contact=True,
                             external_navigation=arrival_signal(stamp,**kw))
            assert not np.any(state['command_xyz'])
        return state
    assert hold(4.,final=False)['phase']=='carry'
    assert hold(2.,settled=False)['phase']=='carry'
    p[0]+=.04
    assert hold(3.)['phase']=='carry'  # feedback cannot waive35mm pelvis region
    p[0]-=.04
    state=hold(4.)
    assert state['phase']=='lower' and state['release_fraction']==0
    assert state['navigation_owner']=='external_htd'
    np.testing.assert_allclose(state['lowering_delta_world_m'],[0.,0.,-.092],atol=1e-12)
    # No support must still stop without a release, despite external arrival.
    with pytest.raises(RuntimeError,match='No stable destination support'):
        ctl.update(stamp+26,p,t,b,[6,0,6,0],0.,external_navigation=arrival_signal(stamp+26))


@pytest.mark.parametrize('fault',['stale','changed_goal','fallback','hold'])
def test_external_navigation_rejects_stale_retargeting_or_vendor_fallback(fault):
    ctl,p,t,b=external_fixture()
    ctl.update(.02,p,t,b,[6,0,6,0],0.,external_navigation=arrival_signal(.02))
    signal=arrival_signal(.2);kwargs={}
    if fault=='stale':signal['sample_time_s']=.02
    if fault=='changed_goal':signal['goal_xy_world']=[.91,0.]
    if fault=='fallback':signal=None
    if fault=='hold':kwargs['hold_carry']=True
    with pytest.raises(ValueError):
        ctl.update(.2,p,t,b,[6,0,6,0],0.,external_navigation=signal,**kwargs)
