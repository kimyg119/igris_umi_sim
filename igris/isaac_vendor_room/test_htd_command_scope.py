"""The diagnostic cannot arm another process or replace simulation preloads."""
import hashlib
from pathlib import Path

import pytest


def scope(out):
    lib=out/'htd_command_observer.so'
    lib.write_bytes(b'test-only library')
    arm=out/'htd_command_observer.arm';trace=out/'htd_command_observer.bin'
    return dict(library=str(lib),arm_path=str(arm),trace_path=str(trace),
        sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),
        contract=dict(schema='htd_command_observer/v1',logging_only=True,max_records=4096),
        environment=dict(LD_PRELOAD=str(lib),IGRIS_HTD_COMMAND_OBSERVER_ARM=str(arm),
            IGRIS_HTD_COMMAND_OBSERVER_TRACE=str(trace),IGRIS_HTD_COMMAND_OBSERVER_MAX_RECORDS='4096',
            IGRIS_HTD_COMMAND_OBSERVER_EXE='/igris/ws/install/igris_c_controller/lib/igris_c_controller/sim'))


def test_valid_scope_exports_only_observer_controls_and_never_arms(tmp_path):
    from htd_command_scope import validate
    s=scope(tmp_path)
    env=validate(tmp_path,s)
    assert 'LD_PRELOAD' not in env
    assert len(env)==4
    assert env['IGRIS_HTD_COMMAND_OBSERVER_EXE'].endswith('/sim')
    assert not Path(s['arm_path']).exists() and not Path(s['trace_path']).exists()


@pytest.mark.parametrize('bad',['escape','exe','hash','arm','trace','bound','contract','extra_env'])
def test_corrupted_or_prearmed_scope_rejected(tmp_path,bad):
    from htd_command_scope import validate
    s=scope(tmp_path)
    if bad=='escape':s['arm_path']=str(tmp_path.parent/'foreign.arm')
    if bad=='exe':s['environment']['IGRIS_HTD_COMMAND_OBSERVER_EXE']='/another/sim'
    if bad=='hash':Path(s['library']).write_bytes(b'changed')
    if bad in ('arm','trace'):Path(s[bad+'_path']).touch()
    if bad=='bound':s['environment']['IGRIS_HTD_COMMAND_OBSERVER_MAX_RECORDS']='9000'
    if bad=='contract':s['contract']['logging_only']=False
    if bad=='extra_env':s['environment']['UNRELATED']='not authorized'
    with pytest.raises((ValueError,FileExistsError)):validate(tmp_path,s)
