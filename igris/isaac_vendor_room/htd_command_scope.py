"""Fresh, trial-local SIM-only command diagnostic configuration."""
import hashlib
from pathlib import Path

SIM_EXE='/igris/ws/install/igris_c_controller/lib/igris_c_controller/sim'


def validate(out, scope):
    """Return four scoped variables; caller merges the library into existing preloads."""
    out=Path(out).resolve()
    contract=scope.get('contract',{})
    count=contract.get('max_records')
    if (contract.get('schema')!='htd_command_observer/v1' or contract.get('logging_only') is not True
            or type(count) is not int or not 1<=count<=8192):
        raise ValueError('Invalid logging-only command observer contract')
    for key,name in [('library','htd_command_observer.so'),('arm_path','htd_command_observer.arm'),
                     ('trace_path','htd_command_observer.bin')]:
        if scope.get(key)!=str(out/name) or Path(scope[key]).resolve()!=out/name:
            raise ValueError('Command observer path escapes this trial: '+key)
    if hashlib.sha256(Path(scope['library']).read_bytes()).hexdigest()!=scope.get('sha256'):
        raise ValueError('Command observer library hash changed')
    if Path(scope['arm_path']).exists() or Path(scope['trace_path']).exists():
        raise FileExistsError('Command observer requires fresh unarmed paths')
    env=dict(IGRIS_HTD_COMMAND_OBSERVER_ARM=scope['arm_path'],
             IGRIS_HTD_COMMAND_OBSERVER_TRACE=scope['trace_path'],
             IGRIS_HTD_COMMAND_OBSERVER_EXE=SIM_EXE,IGRIS_HTD_COMMAND_OBSERVER_MAX_RECORDS=str(count))
    if scope.get('environment')!={**env,'LD_PRELOAD':scope['library']}:
        raise ValueError('Command observer environment differs from scoped SIM contract')
    return env
