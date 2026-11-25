import hashlib
import json
import pytest
import run
from types import SimpleNamespace


def test_normal_start_has_no_loaded_environment(tmp_path):
    assert run.loaded_start_environment(tmp_path,{}) == ''


def manifest(tmp_path,monkeypatch):
    # These startup-hook tests isolate paths/environment/GC; model equivalence
    # is exercised with real audited scenes in test_htd_upper_backlash_wiring.
    import htd_backlash_limits
    monkeypatch.setattr(htd_backlash_limits,'validate',
        lambda directory,expected_upper_timeconst=None: {})
    library=tmp_path/'output_coordinates.so';library.write_bytes(b'observer')
    params=tmp_path/'htd_gc_params.yaml';params.write_text('test: true\n')
    robot=tmp_path/'robot.yaml'
    robot.write_text('igris_c:\n  controller:\n    gain_parameter_path: '+str(params)+'\n    GravityCompensationModule:\n      enabled: true\n')
    monkeypatch.setenv('IGRIS_ROOM_PROFILE_LIBRARY',str(library))
    monkeypatch.setattr(run,'HTD_LOADED_ZERO_RUNTIME_CONTRACT','htd_loaded_zero/v1',raising=False)
    result=dict(simulation_rate_limit=.5,simulation_rate_start_phase_s=90.,
        htd_loaded_zero=dict(enabled=True,schema='htd_loaded_zero/v1',
            observer_library=str(library),pacing_request_path=str(tmp_path/'htd_pacing.request'),
            pacing_ack_path=str(tmp_path/'htd_pacing.request.ack'),
            acquisition_rate=.5,loaded_rate=.1,params_path=str(params),
            setup_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in (library,robot,params)}))
    (tmp_path/'manifest.json').write_text(json.dumps(result))
    return result


def test_loaded_native_slowdown_is_explicit_and_one_shot(tmp_path,monkeypatch):
    m=manifest(tmp_path,monkeypatch)
    text=run.loaded_start_environment(tmp_path,m)
    assert 'IGRIS_ROOM_PACER_SLOWDOWN=' in text
    assert 'IGRIS_ROOM_SLOW_REALTIME_FACTOR=.1' in text
    assert str(tmp_path/'htd_pacing.request') in text
    (tmp_path/'htd_pacing.request.ack').touch()
    with pytest.raises(ValueError):run.loaded_start_environment(tmp_path,m)


def test_optional_command_observer_exports_scoped_variables_without_replacing_preloads(tmp_path,monkeypatch):
    from test_htd_command_scope import scope
    m=manifest(tmp_path,monkeypatch)
    m['htd_loaded_zero']['command_observer']=scope(tmp_path)
    text=run.loaded_start_environment(tmp_path,m)
    assert 'IGRIS_HTD_COMMAND_OBSERVER_EXE=/igris/ws/install/igris_c_controller/lib/igris_c_controller/sim' in text
    assert 'LD_PRELOAD=' not in text
    assert not (tmp_path/'htd_command_observer.arm').exists()
    (tmp_path/'htd_command_observer.arm').touch()
    with pytest.raises(FileExistsError):run.loaded_start_environment(tmp_path,m)


@pytest.mark.parametrize('bad',['contract','library','initializer','rate','path','hash','params','stale_initializer'])
def test_loaded_start_fails_closed(tmp_path,monkeypatch,bad):
    m=manifest(tmp_path,monkeypatch)
    if bad=='contract':monkeypatch.setattr(run,'HTD_LOADED_ZERO_RUNTIME_CONTRACT','unfinished')
    if bad=='library':monkeypatch.delenv('IGRIS_ROOM_PROFILE_LIBRARY')
    if bad=='initializer':m['htd_initial_state']={}
    if bad=='rate':m['simulation_rate_limit']=.2
    if bad=='path':m['htd_loaded_zero']['pacing_request_path']=str(tmp_path/'different')
    if bad=='hash':(tmp_path/'output_coordinates.so').write_bytes(b'changed')
    if bad=='params':(tmp_path/'htd_gc_params.yaml').write_text('changed: true\n')
    if bad=='stale_initializer':(tmp_path/'htd_initial_state.request').touch()
    with pytest.raises((ValueError,RuntimeError)):run.loaded_start_environment(tmp_path,m)


def test_slow_message_fill_is_rejected_before_any_publication():
    elapsed=[0.]
    class Motor:
        def __setattr__(self,name,value):
            elapsed[0]+=.00002
            object.__setattr__(self,name,value)
    published=[]
    def message():return SimpleNamespace(header=SimpleNamespace(),motors=[Motor()for _ in range(31)])
    command={k:[0.]*31 for k in ('q','dq','kp','kd','tau')}
    def check():
        if elapsed[0]>=.002:raise RuntimeError('sample aged during message fill')
    # The command is fresh before construction but stale once31 motors are filled.
    check()
    with pytest.raises(RuntimeError,match='aged'):
        run.publish_lowcmd(SimpleNamespace(publish=published.append),message,command,4,before_publish=check)
    assert not published


def test_sole_publisher_keeps_all_live_fields_and_checks_filled_message():
    published=[];messages=[];checked=[]
    def message():
        m=SimpleNamespace(header=SimpleNamespace(),motors=[SimpleNamespace()for _ in range(31)])
        messages.append(m);return m
    command={k:[i+j/10 for i in range(31)]for j,k in enumerate(('q','dq','kp','kd','tau'))}
    def check():
        assert messages[0].motors[30].tau==command['tau'][30]
        assert not published
        checked.append(True)
    run.publish_lowcmd(SimpleNamespace(publish=published.append),message,command,9,before_publish=check)
    assert checked==[True] and len(published)==1
    assert published[0].header.seq==9 and published[0].kinematic_modes==[1]*5
    for i,motor in enumerate(published[0].motors):
        for key,values in command.items():assert getattr(motor,key)==values[i]


def test_sole_publisher_preserves_explicit_wrist_modes_and_rejects_bad_modes():
    published=[]
    def message():return SimpleNamespace(header=SimpleNamespace(),motors=[SimpleNamespace()for _ in range(31)])
    command={k:[0.]*31 for k in ('q','dq','kp','kd','tau')}
    command['kinematic_modes']=[1,1,1,0,0]
    run.publish_lowcmd(SimpleNamespace(publish=published.append),message,command,2)
    assert published[0].kinematic_modes==[1,1,1,0,0]
    command['kinematic_modes']=[1,1,1,0,2]
    with pytest.raises(ValueError,match='kinematic'):
        run.publish_lowcmd(SimpleNamespace(publish=published.append),message,command,3)
    assert len(published)==1
