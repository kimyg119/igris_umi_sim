"""Build a simulation-only clock shim for the exact audited vendor binary.

The shim changes no physics state or vendor files. Pass its path through the
existing private launcher's IGRIS_ROOM_PROFILE_LIBRARY, and set the returned
binary environment value inside that private launch process.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BINARY=ROOT/'igris/ws/install/igris_c_controller/lib/igris_c_controller/sim'
BINARY_SHA256='5003375d46f80d7e1064b50a1933e0a1fb70f7566724ac682379ef909beb8b9b'
RETURN_OFFSETS=[0x5ab8a,0x5ac64,0x5ace9,0x5ad00]


def binary_manifest(binary=BINARY):
    binary=Path(binary).resolve()
    digest=hashlib.sha256(binary.read_bytes()).hexdigest()
    if digest!=BINARY_SHA256:
        raise ValueError('Vendor policy clock binary fingerprint differs from audited simulation executable')
    disassembly=subprocess.check_output(['objdump','-dC','--start-address=0x5ab42',
                                        '--stop-address=0x5bb40',str(binary)],text=True)
    # Each x86-64 rel32 CALL is five bytes; return addresses are the allowlist.
    calls=[int(x,16)+5 for x in re.findall(
        r'^\s*([0-9a-f]+):[^\n]*call\s+[^\n]*<std::chrono::_V2::steady_clock::now\(\)@plt>',
        disassembly,re.M)]
    if calls!=RETURN_OFFSETS:
        raise ValueError('Audited inference clock call sites changed')
    return dict(binary=str(binary),sha256=digest,return_offsets=calls,
                scope='RobotHandler simulation inference scheduler only; unrelated clocks remain wall time',
                vendor_files_modified=False,physics_state_modified=False,hardware_executed=False)


def build(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    manifest=binary_manifest();source=HERE/'vendor_policy_clock.cpp'
    target=out/'vendor_policy_clock.so'
    if target.exists():raise FileExistsError(target)
    subprocess.run(['g++','-O2','-std=c++17','-shared','-fPIC',str(source),'-ldl','-o',str(target)],check=True)
    manifest.update(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    library_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (out/'vendor_policy_clock_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return target
