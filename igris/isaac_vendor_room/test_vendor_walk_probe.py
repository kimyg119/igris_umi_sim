"""Offline safety and schedule checks; no ROS or simulator startup."""
import importlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest


def probe():
    assert importlib.util.find_spec('vendor_walk_probe') is not None
    return importlib.import_module('vendor_walk_probe')


def test_schedule_uses_native_seconds_and_bounded_yaw_ramp_then_exact_zero():
    module=probe()
    assert module.twist(2.999)==[0.,0.,0.]
    assert module.twist(3.)==[0.,0.,0.]
    assert module.twist(3.5)==[0.,0.,.1]
    assert module.twist(4.15)==[0.,0.,.23]
    assert module.twist(10.999)==[0.,0.,.23]
    assert module.twist(11.)==[0.,0.,0.]
    assert module.twist(12.99)==[0.,0.,0.]


def test_approach_spreads_before_folding_preserves_unowned_legs_and_speed():
    module=probe();initial=np.zeros(31);initial[3:15]=np.linspace(-.1,.1,12)
    goal=initial.copy();goal[0]=.755667;goal[18]=goal[25]=-1.43;goal[16]=.15;goal[23]=-.21
    points,times=module.approach(initial,goal)
    assert points[1,16]==.9 and points[1,23]==-.9
    assert points[1,18]==points[1,25]==0.
    assert points[2,18]==points[2,25]==-1.43
    for q in points:np.testing.assert_array_equal(q[3:15],initial[3:15])
    samples=np.array([module.interpolate(points,times,t)for t in np.arange(0,times[-1]+.01,.01)])
    assert abs(np.diff(samples,axis=0)/.01).max()<=.300001
    np.testing.assert_array_equal(module.interpolate(points,times,times[-1]+1),goal)


def test_state_requires_sim_walklow_finite_errorfree_31_and_fresh_timestamps():
    module=probe();state=dict(q=[0.]*31,dq=[0.]*31,error_code=[0]*31)
    module.require_state(1,6,state,.01,.01)
    for environment,mode,low_age,mode_age in [(0,6,.01,.01),(1,3,.01,.01),(1,6,.251,.01),(1,6,.01,.501)]:
        with pytest.raises(RuntimeError):module.require_state(environment,mode,state,low_age,mode_age)
    for key,value in [('q',[0.]*30),('dq',[float('nan')]*31),('error_code',[1]+[0]*30)]:
        with pytest.raises(RuntimeError):module.require_state(1,6,{**state,key:value},.01,.01)


def test_pose_guards_use_start_frame_lateral_displacement_and_tilt():
    module=probe();base=np.array([1.,2.,.86,np.sqrt(.5),0,0,np.sqrt(.5)])
    module.require_pose(base,base)
    displaced=base.copy();displaced[0]+=.401
    with pytest.raises(RuntimeError,match='[Ll]ateral'):module.require_pose(displaced,base)
    low=base.copy();low[2]=.69
    with pytest.raises(RuntimeError,match='height'):module.require_pose(low,base)
    tilted=base.copy();tilted[3:]=[np.cos(.31/2),np.sin(.31/2),0,0]
    with pytest.raises(RuntimeError,match='tilt'):module.require_pose(tilted,base)


def test_upper_gate_rejects_hard_and_coupled_limits_while_legs_are_unowned():
    module=probe();gate=module.UpperLimits();q=np.zeros(31)
    gate.check(q);q[3]=100.;gate.check(q)
    q[0]=100.
    with pytest.raises(ValueError,match='hard'):gate.check(q)
    q[0]=0.;q[1:3]=[.3,.25]
    with pytest.raises(ValueError,match='coupled'):gate.check(q)
    kp,kd=module.gains()
    np.testing.assert_array_equal(kp[3:15],np.zeros(12));np.testing.assert_array_equal(kd[3:15],np.zeros(12))


def test_prepare_copies_scene_preserves_physics_and_never_activates_room(tmp_path):
    module=probe();source=tmp_path/'source';source.mkdir()
    for name in module.INPUT_FILES:
        if name=='robot.yaml':text='igris_c:\n  description:\n    mujoco:\n      robot_model_path: /old/scene.xml\n  controller:\n    GravityCompensationModule:\n      enabled: true\n'
        elif name=='manifest.json':text=json.dumps(dict(stage_room_after_preparation=True,compliant_hands=True,payload=dict(mass_kg=.25)))
        else:text='original '+name
        (source/name).write_text(text)
    out=tmp_path/'new';trial=module.prepare(source,out)
    assert (trial/'scene.xml').read_bytes()==(source/'scene.xml').read_bytes()
    import yaml
    cfg=yaml.safe_load((trial/'robot.yaml').read_text())
    assert cfg['igris_c']['controller']['GravityCompensationModule']['enabled'] is True
    assert cfg['igris_c']['description']['mujoco']['robot_model_path']==str(trial/'scene.xml')
    assert not (trial/'room_layout.request').exists()
    assert json.loads((trial/'manifest.json').read_text())['stage_room_after_preparation'] is True
    with pytest.raises(FileExistsError):module.prepare(source,out)


def test_prepare_explicit_scheduler_and_rate_scope(tmp_path):
    module=probe();source=tmp_path/'source';source.mkdir()
    for name in module.INPUT_FILES:
        if name=='robot.yaml':text='igris_c:\n  description:\n    mujoco:\n      robot_model_path: /old/scene.xml\n'
        elif name=='manifest.json':text=json.dumps(dict(stage_room_after_preparation=True))
        else:text='unchanged '+name
        (source/name).write_text(text)
    out=tmp_path/'native'
    trial=module.prepare(source,out,native_policy_clock=True,simulation_rate_limit=.4)
    manifest=json.loads((trial/'manifest.json').read_text())
    assert manifest['vendor_policy_native_clock'] is True
    assert manifest['simulation_rate_limit']==.4
    assert (trial/'scene.xml').read_bytes()==(source/'scene.xml').read_bytes()
    for rate in (0.,-1.,1.01,float('nan')):
        with pytest.raises(ValueError):module.prepare(source,tmp_path/str(rate),simulation_rate_limit=rate)
