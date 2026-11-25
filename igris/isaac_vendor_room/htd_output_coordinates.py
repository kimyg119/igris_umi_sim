"""SIM-only synchronized physical output coordinates and bounded PJS correction."""
from pathlib import Path
import hashlib
import json
import struct
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ('q_hinge', 'dq_hinge', 'q_backlash', 'dq_backlash', 'q_output', 'dq_output',
          'q_encoder', 'dq_encoder', 'q_motor', 'dq_motor')
RECORD_BYTES = 2520
MAX_BIAS_VELOCITY = 2.  # Conservative empty-hand experiment guard, not a vendor limit.


def build_observer(out, xml):
    out = Path(out)
    source = Path(__file__).with_suffix('.cpp')
    inc = ROOT/'igris/ws/install'
    lib = inc/'igris_c_mujoco/lib/igris_c_mujoco'
    common = ['g++', '-O2', '-std=c++17', '-I/usr/include/eigen3',
              '-I'+str(inc/'igris_c_mujoco/include'), '-I'+str(inc/'igris_c_shm/include'),
              '-I'+str(inc/'igris_c/include'), str(source)]
    library, audit = out/'htd_output_coordinates.so', out/'htd_output_audit'
    subprocess.run([*common, '-shared', '-fPIC', '-ldl', '-o', str(library)], check=True)
    subprocess.run([*common, '-L'+str(lib), '-Wl,-rpath,'+str(lib), '-l:libmujoco.so.3.2.7',
                    '-ldl', '-o', str(audit)], check=True)
    result = json.loads(subprocess.check_output([str(audit), str(xml)], text=True))
    if result['mujoco_version'] != '3.2.7':
        raise RuntimeError('Native observer requires provided MuJoCo3.2.7')
    evidence = dict(library=str(library), self_test=result,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                    capture='Same native physics callback as vendor motor2Joint; original arguments/return preserved',
                    maximum_bias_velocity_rad_s=MAX_BIAS_VELOCITY)
    (out/'htd_output_observer.json').write_text(json.dumps(evidence, indent=2)+'\n')
    return evidence


def read_output(path, *, now=None, native_time):
    with Path(path).open('rb', buffering=0) as stream:
        for _ in range(5):
            stream.seek(0)
            raw = stream.read(RECORD_BYTES)
            if len(raw) != RECORD_BYTES:
                continue
            sequence, version = struct.unpack_from('=QQ', raw)
            stream.seek(0)
            after = stream.read(8)
            if sequence and not sequence % 2 and after == raw[:8]:
                break
        else:
            raise RuntimeError('No complete synchronized native coordinate sample')
    if version != 1:
        raise RuntimeError('Unknown native coordinate record version')
    values = np.frombuffer(raw, dtype=np.float64, offset=16).copy()
    if not np.isfinite(values).all():
        raise RuntimeError('Nonfinite native coordinate sample')
    wall, monotonic, sim = values[:3]
    now = time.monotonic() if now is None else now
    native_time = native_time() if callable(native_time) else native_time
    if not 0 <= now-monotonic < .1 or not -1e-9 <= native_time-sim < .01:
        raise RuntimeError('Stale, future or pre-reset native coordinate sample')
    result = dict(zip(FIELDS, values[3:].reshape(10, 31)))
    result.update(sequence=sequence, wall=wall, monotonic=monotonic, sim=sim,
                  wall_age=now-monotonic, native_lag=native_time-sim)
    return result


def correction_evidence(raw, projected, command, bias, owned):
    requested = np.where(owned, projected-bias, raw)
    correction = command-requested
    return dict(encoder_target_unprojected=requested.copy(), encoder_projection=correction.copy(),
                projection_budget_used_rad=float(np.max(abs(projected[owned]-raw[owned]))+np.max(abs(correction[owned]))),
                expected_output_target_residual=np.where(owned, command+bias-projected, 0.))


def compensate_output(raw_target, projected_target, sample, gate, *, saturate_velocity=False, owned_indices=None,
                      project_encoder=False, projection_budget_rad=.10, raw_command_budget_rad=.10):
    if not np.isfinite([projection_budget_rad, raw_command_budget_rad]).all() or not 0 < projection_budget_rad <= .30 or not 0 < raw_command_budget_rad <= .40:
        raise ValueError('Expected finite projection budget<=0.30 and raw-command budget<=0.40 rad')
    arrays = [np.asarray(x, dtype=float) for x in (raw_target, projected_target,
              sample['q_output'], sample['dq_output'], sample['q_encoder'], sample['dq_encoder'])]
    if any(x.shape != (31,) or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Expected finite all31 output-coordinate vectors')
    raw, projected, q_out, dq_out, q_encoder, dq_encoder = arrays
    gate.check(projected)
    bias, dbias = q_out-q_encoder, dq_out-dq_encoder
    owned = np.ones(31, dtype=bool) if owned_indices is None else np.zeros(31, dtype=bool)
    if owned_indices is not None:
        owned[list(owned_indices)] = True
    command, velocity = raw.copy(), np.zeros(31)
    command[owned] = projected[owned]-bias[owned]
    velocity[owned] = -dbias[owned]
    if np.max(abs(bias[owned])) > .10+1e-12:
        raise ValueError('Owned output position bias exceeds 0.10 rad')
    if project_encoder:
        command = gate.project(command)
        command[~owned] = raw[~owned]
        evidence = correction_evidence(raw, projected, command, bias, owned)
        if evidence['projection_budget_used_rad'] > projection_budget_rad+1e-12:
            raise ValueError(f'Combined projection exceeds {projection_budget_rad:.2f} rad budget')
    if saturate_velocity:
        velocity = np.clip(velocity, -MAX_BIAS_VELOCITY, MAX_BIAS_VELOCITY)
    elif np.max(np.abs(velocity)) > MAX_BIAS_VELOCITY:
        joint = int(np.argmax(np.abs(velocity)))
        raise ValueError(f'Output velocity compensation exceeds conservative 2 rad/s guard: SDK index {joint}, {velocity[joint]:.6f} rad/s')
    if np.max(np.abs(command-raw)) > raw_command_budget_rad+1e-12:
        raise ValueError(f'Total projection plus coordinate correction exceeds {raw_command_budget_rad:.2f} rad')
    gate.check(command)
    return command, velocity, bias, dbias
