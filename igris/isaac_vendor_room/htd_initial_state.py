"""Opt-in fresh empty-hand native initial data, never a loaded reset path."""
from pathlib import Path
import hashlib
import json
import os
import subprocess

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(__file__).with_suffix('.cpp')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out,state_path):
    """Configure a fresh probe's complete robot data; state_path=None is inert."""
    if state_path is None:return None
    out=Path(out);state_path=Path(state_path).resolve()
    manifest=json.loads((out/'manifest.json').read_text())
    if any((out/n).exists()for n in ('runtime.json','poses.bin','htd_initial_state.request','htd_initial_state_applied.json','htd_initial_state.txt')):
        raise ValueError('Initial state requires a fresh never-started simulation directory')
    if not (manifest.get('htd_probe') is True and manifest.get('empty_hand_only') is True
            and manifest.get('room_must_remain_parked') is True):
        raise ValueError('Initial state is restricted to explicit empty-hand HTD probes')
    target=out/'htd_initial_state.txt';target.write_bytes(state_path.read_bytes())
    inc=ROOT/'igris/ws/install/igris_c_mujoco/include';lib=ROOT/'igris/ws/install/igris_c_mujoco/lib/igris_c_mujoco'
    common=['g++','-O2','-std=c++17',f'-I{inc}',str(SOURCE)]
    library=out/'htd_initial_state.so';audit=out/'htd_initial_state_audit'
    subprocess.run([*common,'-shared','-fPIC','-ldl','-o',str(library)],check=True)
    subprocess.run([*common,'-DIGRIS_INITIAL_STATE_AUDIT',f'-L{lib}',f'-Wl,-rpath,{lib}','-l:libmujoco.so.3.2.7','-ldl','-o',str(audit)],check=True)
    env=os.environ.copy();env.pop('IGRIS_HTD_INITIAL_STATE_DIR',None)
    result=json.loads(subprocess.check_output([str(audit),str(out/'scene.xml'),str(target)],text=True,env=env))
    if result['mujoco']!='3.2.7' or result['robot_joints']!=97 or result['maximum_equality_residual']>.001:
        raise ValueError('Native initial state model/closure audit failed')
    config=dict(schema='native_empty_initial_state/v1',state_path=str(target.resolve()),state_sha256=sha(target),
        source_state_path=str(state_path),source_state_sha256=sha(state_path),library=str(library.resolve()),library_sha256=sha(library),
        cpp_sha256=sha(SOURCE),scene_sha256=sha(out/'scene.xml'),audit=result,
        scope='One observed fresh empty-hand reset; all robot mechanism coordinates; no room/model modifications; no grasp claim')
    manifest['htd_initial_state']=config
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return config


def validate(out,manifest=None):
    out=Path(out);manifest=manifest or json.loads((out/'manifest.json').read_text());c=manifest.get('htd_initial_state')
    if c is None:return None
    if not(manifest.get('htd_probe') is True and manifest.get('empty_hand_only') is True and manifest.get('room_must_remain_parked') is True):
        raise ValueError('Initialization is not an empty-hand probe')
    if (c.get('schema')!='native_empty_initial_state/v1' or Path(c['state_path']).resolve()!=out.resolve()/'htd_initial_state.txt'
            or Path(c['library']).resolve()!=out.resolve()/'htd_initial_state.so'
            or sha(c['state_path'])!=c['state_sha256'] or sha(c['library'])!=c['library_sha256']
            or sha(out/'scene.xml')!=c['scene_sha256'] or sha(SOURCE)!=c['cpp_sha256']):
        raise ValueError('Initial-state contract/hash mismatch')
    return c


def request(out):
    out=Path(out)
    if validate(out) is None:raise ValueError('No explicit initial-state contract')
    if any((out/n).exists()for n in ('htd_initial_state.request','htd_initial_state_applied.json','room_layout.request','room_layout_applied.json')):
        raise ValueError('Initial-state request cannot be reused or applied to staged objects')
    temporary=out/'htd_initial_state.request.tmp';temporary.write_text('initialize-on-empty-reset\n')
    temporary.replace(out/'htd_initial_state.request')


def acknowledgement(out):
    out=Path(out);c=validate(out)
    if c is None:raise ValueError('No explicit initial-state contract')
    result=json.loads((out/'htd_initial_state_applied.json').read_text())
    if (result.get('schema')!=c['schema'] or result.get('apply_count')!=1 or result.get('robot_joints')!=97
            or not 0<=result.get('sim',-1)<.01 or result.get('prior_sim',0)<1
            or result.get('robot_velocity_zero') is not True or result.get('room_qpos_unchanged') is not True
            or result.get('model_parameters_unchanged') is not True):
        raise ValueError('Invalid native initial-state acknowledgement')
    return result
