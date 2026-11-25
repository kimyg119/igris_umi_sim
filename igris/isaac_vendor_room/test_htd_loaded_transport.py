"""Transport arbitration with real actor/PD and synthetic measured state; no physics."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from test_htd_loaded import setup, snapshot
from htd_transport_phase import TransportPlacement


TARGET = np.array([1.12, -.24, .758])


def sample(policy, t=1., *, phase='carry', supported=False, opened=0., done=False,
           x=0., y=0., mode=3):
    s = snapshot(policy, t, mode=mode, sequence=2+int(round((t-1)*600)))
    box = np.array([x+.22,y-.24,.758 if supported else .9,1.,0.,0.,0.])
    hands = np.stack([box.copy(),box.copy()]); hands[:,1] += [.15+opened,-.15-opened]
    return replace(s, pelvis_pose=np.array([x,y,.9,1.,0.,0.,0.]),
        torso_pose=np.array([x-.05,y,1.,1.,0.,0.,0.]), payload_pose=box, hand_poses=hands,
        pose_native_time=t,force_native_time=t,transport_phase=phase,
        phase_target_valid=phase!='carry',placement_target_xyz=TARGET.copy() if phase!='carry' else None,
        placement_done=done,destination_support_z_N=.981 if supported else 0.,
        palm_tip_force_N=np.zeros(4) if supported else np.full(4,2.))


def coordinator(**kwargs):
    return setup(current_grasp_reference=True,goal_hold=True,contact_grace=True,
        transport_target_xyz=TARGET,transport_heading=0.,**kwargs)


@pytest.mark.parametrize('kwargs', [dict(transport_target_xyz=TARGET),dict(transport_heading=0.),
    dict(transport_target_xyz=TARGET,transport_heading=0.),
    dict(current_grasp_reference=True,transport_target_xyz=TARGET,transport_heading=0.)])
def test_transport_requires_complete_explicit_scope(kwargs):
    with pytest.raises(ValueError,match='transport|Transport'): setup(**kwargs)


def test_transport_has_explicit180_timeout_and_immediate_route_first_actor():
    c,p,_=coordinator();assert c.duration_s==180.
    c.request_handoff(sample(p,mode=6))
    record=c.advance_actor(sample(p))
    assert record is not None and c.goal_hold.navigation is not None
    assert c.goal_hold.metrics['command_xyz'][1]>0
    assert c.policy._step_index==1 and c.serial_pd.step_count==0
    assert c.transport is True and c.placement.phase=='carry'
    with pytest.raises(RuntimeError,match='transport.*timeout'):
        c.advance_actor(sample(p,181.))
    assert c.state!='complete'


def placement(policy):
    obj=TransportPlacement(TARGET,.1)
    obj.capture(sample(policy))
    return obj


def enter_lower(obj,p,t=1.02):
    obj.check(sample(p,t,phase='lower',x=.9),navigation_ready=True)


def qualify_support(obj,p,start=1.04):
    for i in range(27):
        obj.check(sample(p,start+i*.02,phase='lower',x=.9,supported=True),navigation_ready=True)
    return start+.52


def enter_open(obj,p):
    enter_lower(obj,p); t=qualify_support(obj,p)+.02
    obj.check(sample(p,t,phase='open',x=.9,supported=True),navigation_ready=True)
    return t


def test_phase_skip_regression_and_unaccepted_lower_targets_rejected():
    _,p,_=coordinator();obj=placement(p)
    with pytest.raises(RuntimeError,match='transition'):
        obj.check(sample(p,1.02,phase='open',x=.9,supported=True),navigation_ready=True)
    with pytest.raises(RuntimeError,match='navigation'):
        obj.check(sample(p,1.02,phase='lower',x=.9),navigation_ready=False)
    with pytest.raises(RuntimeError,match='accepted'):
        obj.check(replace(sample(p,1.02,phase='lower',x=.9),phase_target_valid=False),navigation_ready=True)
    enter_lower(obj,p)
    with pytest.raises(RuntimeError,match='transition'):
        obj.check(sample(p,1.04),navigation_ready=True)
    with pytest.raises(ValueError,match='destination'):
        obj.check(replace(sample(p,1.04,phase='lower'),placement_target_xyz=TARGET+.01),navigation_ready=True)


def test_support_dwell_uses_fresh_continuous_samples_and_cannot_be_phase_flagged():
    _,p,_=coordinator();obj=placement(p);enter_lower(obj,p)
    obj.check(sample(p,1.04,phase='lower',x=.9,supported=True),navigation_ready=True)
    with pytest.raises(RuntimeError,match='support.*dwell'):
        obj.check(sample(p,1.06,phase='open',x=.9,supported=True),navigation_ready=True)
    with pytest.raises(ValueError,match='fresh|Fresh'):
        obj.check(replace(sample(p,1.6,phase='open',x=.9,supported=True),force_native_time=1.04),navigation_ready=True)
    # A large unobserved gap resets the dwell even with a new fresh sample.
    obj.check(sample(p,1.7,phase='lower',x=.9,supported=True),navigation_ready=True)
    with pytest.raises(RuntimeError,match='support.*dwell'):
        obj.check(sample(p,1.72,phase='open',x=.9,supported=True),navigation_ready=True)


@pytest.mark.parametrize('change',[dict(destination_support_z_N=0.),dict(payload_speed_m_s=.0201),
    dict(payload_pose=np.array([1.151,-.24,.758,1.,0.,0.,0.])),dict(pose_native_time=1.)])
def test_open_rejects_support_loss_speed_pose_or_freshness(change):
    _,p,_=coordinator();obj=placement(p);t=enter_open(obj,p)
    with pytest.raises((ValueError,RuntimeError)):
        obj.check(replace(sample(p,t+.02,phase='open',x=.9,supported=True),**change),navigation_ready=True)


def test_measured_open_requires_seven_seconds_each_relative_ee_and_low_forces():
    _,p,_=coordinator();obj=placement(p);t=enter_open(obj,p)
    for i in range(1,351):
        obj.check(sample(p,t+i*.02,phase='open',x=.9,supported=True,opened=.055),navigation_ready=True)
    s=sample(p,t+7.02,phase='done',x=.9,supported=True,opened=0.,done=True)
    # Common translation of BOTH hands cannot fake two outward openings.
    s.hand_poses[:,:3]+=[0.,.06,0.]
    obj.check(s,navigation_ready=True);assert not obj.completion_ready
    obj.check(replace(sample(p,t+7.04,phase='done',x=.9,supported=True,opened=.055,done=True),
        palm_tip_force_N=np.full(4,.1)),navigation_ready=True)
    assert not obj.completion_ready
    obj.check(sample(p,t+7.06,phase='done',x=.9,supported=True,opened=.055,done=True),navigation_ready=True)
    assert obj.completion_ready
    assert min(obj.metrics['outward_open_m'])>=.045


def test_done_wait_is_bounded_and_lower_timeout_never_completes():
    _,p,_=coordinator();obj=placement(p);t=enter_open(obj,p)
    obj.check(sample(p,t+.02,phase='done',x=.9,supported=True,done=True),navigation_ready=True)
    for i in range(1,151):
        obj.check(sample(p,t+.02+i*.02,phase='done',x=.9,supported=True,done=True),navigation_ready=True)
    with pytest.raises(RuntimeError,match='verification.*timeout'):
        obj.check(sample(p,t+3.04,phase='done',x=.9,supported=True,done=True),navigation_ready=True)
    obj=placement(p);enter_lower(obj,p)
    with pytest.raises(RuntimeError,match='lower.*timeout'):
        obj.check(sample(p,26.04,phase='lower',x=.9),navigation_ready=True)


def test_full_loop_preserves_actor_pd_ownership_through_lower_open_completion():
    c,p,upper=coordinator();c.request_handoff(sample(p,mode=6))
    records=[];last_phase='carry';lower_at=open_at=None; resets=[]
    original_reset=p.reset
    def reset(*args,**kwargs):resets.append(kwargs);return original_reset(*args,**kwargs)
    p.reset=reset
    for n in range(6000):
        t=1+n/300.
        x,y=(0.,0.) if n<6 else ((0.,.07) if n<12 else ((.38,.085) if n<18 else (.9,0.)))
        if c.goal_hold.success and lower_at is None:lower_at=t
        if lower_at is not None:
            phase='lower';supported=t-lower_at>.1
            if t-lower_at>3.:
                phase='open';open_at=t if open_at is None else open_at
            if open_at is not None and t-open_at>=7.:
                phase='done'
        else:phase='carry';supported=False
        s=sample(p,t,phase=phase,x=x,y=y,supported=supported,
            opened=0. if open_at is None else min(.055,(t-open_at)*.01),done=phase=='done')
        if phase!='carry':
            s.upper_output_target[15]+=.12
            for key in ('q_output','q_encoder'):s.output[key][15]+=.12
            upper['q'][15]=s.upper_output_target[15]
        c.advance_actor(s)
        if c.state=='complete':break
        frame=c.compute_pd(s,upper)
        if frame is not None:
            for key in upper:np.testing.assert_array_equal(frame.command[key][15:],upper[key][15:])
            c.check_publication(frame,s);c.ack_published(frame,s);records.append(frame.pd.scheduled_time)
        last_phase=phase
    assert c.state=='complete' and c.placement.phase=='done' and c._pending is None
    assert len(resets)==1 and c.actor_count>500  # continued beyond original10s diagnostic
    np.testing.assert_allclose(records,1+.005*np.arange(len(records)))
    assert c.published_count==int(np.ceil((t-c.origin)/.005-1e-9))
    cleanup=c.abort('supported placement complete',upper)
    for key in upper:np.testing.assert_array_equal(cleanup.command[key][15:],upper[key][15:])
    for key in ('tau','kp','kd','dq'):np.testing.assert_array_equal(cleanup.command[key][:15],0.)
    assert not cleanup.hold_grip and cleanup.pause_required


def test_carry_drift_guard_remains_and_lower_needs_valid_hl_with_physical_tracking():
    c,p,upper=coordinator();c.request_handoff(sample(p,mode=6));c.advance_actor(sample(p))
    s=sample(p);s.upper_output_target[15]+=.101;s.output['q_output'][15]+=.101
    with pytest.raises(RuntimeError,match='held8'):c._check(s)
    # Isolate the already-settled navigation gate; upper/HL/contact gates remain real.
    c.goal_hold=SimpleNamespace(success=True)
    s=sample(p,1.02,phase='lower',x=.9);s.upper_output_target[15]+=.15
    s.output['q_output'][15]+=.15
    c._check(s)
    with pytest.raises(RuntimeError,match='hl_age'):
        c._check(replace(s,hl_age_s=.25))
    s.output['q_output'][15]+=.201
    with pytest.raises(RuntimeError,match='Physical upper'):
        c._check(s)


def test_lower_contact_loss_without_support_and_supported_geometry_slip_are_rejected():
    c,p,_=coordinator();c.request_handoff(sample(p,mode=6));c.advance_actor(sample(p))
    c.goal_hold=SimpleNamespace(success=True)
    with pytest.raises(RuntimeError,match='friction'):
        c._check(replace(sample(p,1.02,phase='lower',x=.9),palm_tip_force_N=np.zeros(4)))
    s=sample(p,1.04,phase='lower',x=.9,supported=True)
    c._check(s)  # Measured receiver support permits unloading, preserving geometry.
    bad=sample(p,1.06,phase='lower',x=.9,supported=True)
    bad.hand_poses[0,0]+=.0251
    with pytest.raises(RuntimeError,match='relative translation'):c._check(bad)


def test_transport25mm_does_not_widen_weak_hand_grace_or_supported_placement_region():
    c,p,_=coordinator();c.request_handoff(sample(p,mode=6));c.advance_actor(sample(p))
    s=sample(p,1.02);s.payload_pose[0]+=.02
    c._check(s)
    assert c.placement.metrics['active_translation_limit_m']==.025
    assert c.grasp_retention.metrics['active_translation_limit_m']==.025
    s=sample(p,1.04);s.payload_pose[0]+=.02;s.palm_tip_force_N[:2]=0.
    with pytest.raises(RuntimeError,match='15mm'):c._check(s)
    c.goal_hold=SimpleNamespace(success=True)
    s=sample(p,1.06,phase='lower',x=.9,supported=True);s.hand_poses[:,0]+=.02
    c._check(s)
    assert c.placement.metrics['active_translation_limit_m']==.025
    # The box cannot drift outside the same30mm placement region and keep the unloading exception.
    s=sample(p,1.08,phase='lower',x=.931,supported=True);s.hand_poses[:,0]+=.02
    with pytest.raises(RuntimeError,match='15mm'):c._check(s)


@pytest.mark.parametrize('phase',['carry','lower','open','done'])
def test_cleanup_in_every_phase_preserves_last_published_upper_without_regrip(phase):
    c,p,upper=coordinator();c.request_handoff(sample(p,mode=6));c.advance_actor(sample(p))
    frame=c.compute_pd(sample(p),upper);c.ack_published(frame,sample(p))
    # Placement helper separately tests all transition prerequisites; isolate
    # cleanup semantics from navigation and intentionally keep a published frame.
    c.placement.phase=phase
    newer={k:v.copy() for k,v in upper.items()};newer['q'][18]+=.3
    cleanup=c.abort('test phase guard stop',newer)
    for key in upper:np.testing.assert_array_equal(cleanup.command[key][15:],frame.command[key][15:])
    assert cleanup.hold_grip is (phase in ('carry','lower'))
    assert cleanup.pause_required and c.state=='paused'


def test_pending_frame_and_missing_publication_prevent_transport_completion():
    c,p,upper=coordinator();c.request_handoff(sample(p,mode=6));c.advance_actor(sample(p))
    s=sample(p)
    frame=c.compute_pd(s,upper)
    # Isolate coverage at a verified placement boundary; the helper's measured
    # release gate is exercised independently and in the full actual actor loop.
    c.placement.check=lambda *args,**kwargs:None
    c.placement.completion_ready=True
    assert c.advance_actor(s) is None and c.state=='active'
    c.ack_published(frame,s)
    s=replace(s,native_time=1.001,wall_time=10.01)
    assert c.advance_actor(s) is None and c.state=='complete'
    assert c.published_count==1 and c._pending is None


def test_diagnostic_scope_cannot_enter_placement_or_extend_timeout():
    c,p,_=setup(current_grasp_reference=True,goal_hold=True)
    assert c.duration_s==10. and c.transport is False and c.placement is None
    with pytest.raises(RuntimeError,match='transport scope'):
        c._check(sample(p,phase='lower'))
    with pytest.raises(ValueError,match='Bounded'):
        setup(duration_s=180.)
