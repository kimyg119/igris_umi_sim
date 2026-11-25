"""Observer publication before outer clock publication; no physics or ROS."""
from types import SimpleNamespace

import numpy as np
import pytest
from dataclasses import replace

import htd_loaded_runtime as runtime
from test_htd_navigation_feedback import controller, snapshot


STEP=1/2100.


class Clocks:
    def __init__(self,values):
        self.values=list(values);self.reads=0;self.now=1000.;self.sleeps=[]
        self.returned=[]

    def read(self):
        value=self.values[min(self.reads,len(self.values)-1)];self.reads+=1
        if isinstance(value,Exception):raise value
        result=dict(sim=value,monotonic=self.now,sequence=2*self.reads,wall=self.wall())
        self.returned.append(result)
        return result

    def wall(self):return 2000.+self.now-1000.

    def sleep(self,seconds):
        assert 0<seconds<=.00005
        self.sleeps.append(seconds);self.now+=seconds


def adapter(monkeypatch,values):
    clock=Clocks(values)
    obj=runtime.LoadedRuntime.__new__(runtime.LoadedRuntime)
    obj._clock=clock.read;obj._monotonic=lambda:clock.now;obj._wall=clock.wall
    obj.events=[];obj._pacing_ack=None;obj.mount=np.array([1.,0.,0.,0.])
    c=controller();c.capture(snapshot(99.98));c.step(snapshot(100.))
    obj.coordinator=SimpleNamespace(transport=True,goal_hold=c)
    monkeypatch.setattr(runtime.time,'sleep',clock.sleep)
    return obj,clock


def test_one_step_intermediate_publication_waits_for_actual_clock_and_records_evidence(monkeypatch):
    pose=100.+STEP
    obj,clock=adapter(monkeypatch,[100.,100.,pose+.0001])
    result=obj._clock_after_observations({'pose':pose})
    assert result is clock.returned[-1]  # Real clock packet, never max(clock,pose).
    assert result['sim']==pose+.0001 and clock.reads==3 and len(clock.sleeps)==2
    event=obj.events[-1]
    assert event['event']=='pose_clock_synchronized'
    assert event['original_clock_sim']==100. and event['pose_sim']==pose
    assert event['accepted_clock_sim']==pose+.0001 and event['reads']==3
    assert event['wall_wait_s']==pytest.approx(.0001)


def test_normal_path_uses_one_actual_read_without_wait_or_event(monkeypatch):
    obj,clock=adapter(monkeypatch,[100.001])
    assert obj._clock_after_observations({'pose':100.}) is clock.returned[0]
    assert clock.reads==1 and clock.sleeps==[] and obj.events==[]


def test_future_beyond_one_native_step_is_rejected_without_retry(monkeypatch):
    obj,clock=adapter(monkeypatch,[100.])
    with pytest.raises(RuntimeError,match='pose.*clock.*ahead'):
        obj._clock_after_observations({'pose':100.+STEP+.00001})
    assert clock.reads==1 and clock.sleeps==[]


def test_stalled_producer_times_out_without_changing_timestamp(monkeypatch):
    obj,clock=adapter(monkeypatch,[100.])
    with pytest.raises(RuntimeError,match='timeout.*pose.*clock.*ahead.*wall_wait.*reads'):
        obj._clock_after_observations({'pose':100.+STEP})
    assert 1000.+.005<=clock.now<1000.+.00500001
    assert len(clock.sleeps)>1 and all(x['sim']==100. for x in clock.returned)
    assert not any(e['event']=='pose_clock_synchronized' for e in obj.events)


def test_catchup_after_wall_deadline_is_rejected(monkeypatch):
    obj,clock=adapter(monkeypatch,[100.,100.+STEP])
    original=clock.read
    def slow_read():
        result=original()
        if clock.reads==2:clock.now+=.005
        return result
    obj._clock=slow_read
    with pytest.raises(RuntimeError,match='timeout'):obj._clock_after_observations({'pose':100.+STEP})


def test_reader_stale_clock_error_is_not_treated_as_a_retriable_publication_gap(monkeypatch):
    obj,clock=adapter(monkeypatch,[100.,RuntimeError('stale native physics clock')])
    with pytest.raises(RuntimeError,match='stale native physics clock'):
        obj._clock_after_observations({'pose':100.+STEP})
    assert clock.reads==2


def test_native_clock_regression_during_wait_is_rejected(monkeypatch):
    obj,_=adapter(monkeypatch,[100.,99.99999,100.+STEP])
    with pytest.raises(RuntimeError,match='backward'):obj._clock_after_observations({'pose':100.+STEP})


def test_navigation_uses_synchronized_real_clocks_without_advancing_actor_or_route(monkeypatch):
    pose=100.+STEP;obj,clock=adapter(monkeypatch,[100.,pose])
    c=obj.coordinator.goal_hold;prior=c.metrics.copy();events=len(c.events)
    s=snapshot(pose);poses=np.stack([s.pelvis_pose,s.torso_pose])
    signal=obj.navigation_signal(106.,[2000.,pose],poses,0,1)
    assert signal['sample_time_s']==106. and not signal['feedback_settled']
    assert c.metrics==prior and len(c.events)==events and c.navigation.route.index==0
    assert clock.reads==2


def test_stale_pose_still_fails_original_fifty_ms_guard_without_wait(monkeypatch):
    obj,clock=adapter(monkeypatch,[100.02]);s=snapshot(99.96)
    with pytest.raises(ValueError,match='older than .05'):
        obj.navigation_signal(106.,[2000.,99.96],np.stack([s.pelvis_pose,s.torso_pose]),0,1)
    assert clock.reads==1 and clock.sleeps==[]


@pytest.mark.parametrize('newer',['pose','force','output'])
def test_snapshot_reads_ages_after_wait_and_preserves_pose_targets_and_output(monkeypatch,newer):
    stamps={name:100.+STEP if name==newer else 100. for name in ('pose','force','output')}
    pose=stamps['pose'];obj,clock=adapter(monkeypatch,[100.,100.+STEP])
    obj.coordinator.transport=False
    q=np.arange(31)*.001
    output=dict(sim=stamps['output'],monotonic=1000.,sequence=2,q_output=q.copy(),q_encoder=q.copy(),
                dq_output=np.zeros(31),dq_encoder=np.zeros(31))
    obj._output=lambda:output
    poses=np.array([[0.,0.,.9,1.,0.,0.,0.]])
    s=obj.snapshot(dict(q=q,dq=np.zeros(31),status=np.zeros(31),imu_quat=[1.,0.,0.,0.],
        imu_gyro=[0.,0.,0.]),999.99,6,1,999.98,[1999.99,pose],poses,0,
        dict(wall=1999.99,sim=stamps['force'],palm_tip_force_N=[1,0,1,0],source_support_force_world_z_N=0.,
             support_force_groups_configured=True),1999.99,
        dict(accepted=True,prior_status='used',heartbeat_wall=1999.99,grasp_verified=True),q,
        dict(clearance_measured_rise_m=.04,clearance_ready_s=.5,payload_speed_m_s=.001))
    assert s.native_time==100.+STEP and s.pose_native_time==pose and s.output is output
    assert s.force_native_time==stamps['force']
    assert s.wall_time==clock.now and s.state_age_s==pytest.approx(.01005)
    assert s.pose_age_s==pytest.approx(.01005)
    np.testing.assert_array_equal(s.upper_output_target,q)
    np.testing.assert_array_equal(s.pelvis_pose,poses[0])
    assert output['sim']==stamps['output'] and len(obj.events)==1
    assert obj.events[-1]['observation_native_times_s']==stamps


@pytest.mark.parametrize('source',['force','output'])
def test_later_force_or_output_timestamp_is_waited_for_without_restamping_pose(monkeypatch,source):
    obj,clock=adapter(monkeypatch,[100.,100.+STEP])
    stamps=dict(pose=100.,force=100.,output=100.);stamps[source]=100.+STEP
    result=obj._clock_after_observations(stamps)
    assert result is clock.returned[-1] and clock.reads==2
    assert obj.events[-1]['observation_native_times_s']==stamps
    assert obj.events[-1]['pose_sim']==100. and stamps['pose']==100.


def test_nine_decimal_force_roundoff_needs_no_next_physics_tick(monkeypatch):
    obj,clock=adapter(monkeypatch,[100.0000000006])
    stamps=dict(pose=100.,force=100.000000001,output=100.)
    result=obj._clock_after_observations(stamps)
    assert result['sim']==100.0000000006 and stamps['force']==100.000000001
    assert clock.reads==1 and clock.sleeps==[] and obj.events==[]


@pytest.mark.parametrize('source',['pose','force','output'])
@pytest.mark.parametrize('bad',[np.nan,np.inf,-.1,None,True,[100.]])
def test_invalid_observation_times_fail_before_reading_clock(monkeypatch,source,bad):
    obj,clock=adapter(monkeypatch,[100.]);stamps=dict(pose=100.,force=100.,output=100.)
    stamps[source]=bad
    with pytest.raises(ValueError,match='observation'):obj._clock_after_observations(stamps)
    assert clock.reads==0


def stored_snapshot():
    from test_htd_loaded import snapshot as loaded_snapshot
    # Only actual saved fields consumed by fresh/refresh_output are needed;
    # keep the full immutable LoadedSnapshot to exercise dataclass replacement.
    policy=SimpleNamespace(default_q=np.zeros(31))
    return replace(loaded_snapshot(policy,100.),native_time=100.,wall_time=1000.,
                   pose_native_time=100.,force_native_time=100.)


def test_fresh_waits_for_new_force_but_keeps_all_observations_and_age_guards(monkeypatch):
    obj,clock=adapter(monkeypatch,[100.,100.+STEP])
    s=replace(stored_snapshot(),force_native_time=100.+STEP)
    result=obj.fresh(s)
    assert result.native_time==100.+STEP and result.pose_native_time==100.
    assert result.force_native_time==s.force_native_time and result.output is s.output
    for key in runtime.AGES:
        assert getattr(result,key)==pytest.approx(getattr(s,key)+.00005)
    assert clock.reads==2


def test_refresh_output_synchronizes_new_sample_not_old_snapshot_before_replacement(monkeypatch):
    obj,clock=adapter(monkeypatch,[100.,100.+STEP])
    s=stored_snapshot();new=dict(s.output,sim=100.+STEP,sequence=4)
    obj._output=lambda:new
    result=obj.refresh_output(s)
    assert result.output is new and result.native_time==new['sim']
    assert result.pose_native_time==s.pose_native_time and result.force_native_time==s.force_native_time
    assert clock.reads==2 and obj.events[-1]['observation_native_times_s']['output']==new['sim']
