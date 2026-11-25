"""Explicit fresh-scene HTD numerical variant; original vendor XML is retained."""
from pathlib import Path
import hashlib
import json
import math
from numbers import Real
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OWNED_BACKLASH = tuple(name+'_backlash' for name in
    ['waist_yaw', 'waist_roll', 'waist_pitch']+
    [side+'_'+joint for side in ('l', 'r') for joint in
     ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee_pitch', 'ankle_pitch', 'ankle_roll')])

UPPER_BACKLASH = tuple(side+'_'+joint+'_backlash' for side in ('l', 'r') for joint in
    ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow_pitch','wrist_yaw','wrist_roll','wrist_pitch'))


def upper_descriptor(timeconst):
    """Canonical opt-in selection; never relabel upper14 as policy-owned."""
    if timeconst is None:
        return None
    if isinstance(timeconst, bool) or not isinstance(timeconst, Real) or timeconst != .005:
        raise ValueError('Only explicit numeric0.005s upper backlash timeconstant is supported')
    return dict(schema='htd_upper_backlash_limit/v1',simulation_only=True,startup_only=True,
        original_timeconst_s=.02,timeconst_s=.005,damping_ratio=1.,sdk_indices=list(range(15,29)),
        joint_names=list(UPPER_BACKLASH),unchanged=['ranges','solimp','gains','equalities','mass',
            'friction','all_other_serialized_model_fields'])


def audit_model(executable, original, candidate, timeconst, *, upper_timeconst=None):
    descriptor=upper_descriptor(upper_timeconst)
    args=[str(executable), str(original), str(candidate), str(timeconst)]
    if descriptor is not None:
        args.append(str(descriptor['timeconst_s']))
    result = subprocess.run(args,
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('Native backlash variant audit failed: '+result.stderr.strip())
    return json.loads(result.stdout)


def configure(directory, timeconst, *, upper_timeconst=None):
    """Change owned15, plus explicitly selected upper14, after native equivalence.

    None has no side effects. This is simulation configuration, not a measured
    mechanism correction or a change to range, torque or observed-bias guards.
    """
    descriptor=upper_descriptor(upper_timeconst)
    if descriptor is not None and timeconst != .005:
        raise ValueError('Upper backlash variant requires the owned0.005s profile')
    if timeconst is None:
        return None
    if not math.isfinite(timeconst) or timeconst != .005:
        raise ValueError('Only the explicitly reviewed0.005s backlash limit profile is supported')
    directory = Path(directory)
    scene = directory/'scene.xml'
    original = directory/'htd_backlash_original_scene.xml'
    if original.exists():
        raise FileExistsError('Backlash variant requires a fresh unmodified probe scene')
    before = scene.read_bytes()
    tree = ET.fromstring(before)
    joints = list(tree.iter('joint'))
    selected=OWNED_BACKLASH+(UPPER_BACKLASH if descriptor is not None else ())
    for name in selected:
        matches = [joint for joint in joints if joint.get('name') == name]
        if len(matches) != 1:
            raise ValueError('Expected exactly one owned/selected backlash joint: '+name)
        matches[0].set('solreflimit', '0.005 1')
    original.write_bytes(before)
    candidate = directory/'htd_backlash_candidate.xml'
    ET.ElementTree(tree).write(candidate)
    source = Path(__file__).with_suffix('.cpp')
    installation = ROOT/'igris/ws/install/igris_c_mujoco'
    library = installation/'lib/igris_c_mujoco'
    executable = directory/'htd_backlash_audit'
    subprocess.run(['g++', '-O2', '-std=c++17', '-I'+str(installation/'include'), str(source),
                    '-L'+str(library), '-Wl,-rpath,'+str(library), '-l:libmujoco.so.3.2.7',
                    '-o', str(executable)], check=True)
    native = audit_model(executable, original, candidate, timeconst, upper_timeconst=upper_timeconst)
    report = dict(pass_=True, simulation_only=True, timeconst_s=timeconst,
                  damping_ratio=1., owned_sdk_indices=list(range(15)), native_audit=native,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  original_scene_sha256=hashlib.sha256(before).hexdigest(),
                  variant_scene_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                  scope='Only owned15 backlash limit solref time constants0.02->0.005s; range/solimp/mass/friction unchanged',
                  limitations=['Explicit numerical simulation variant; original vendor model unchanged.',
                               'Finite limit penetration remains; no rigid-actuation or torque-delay equivalence claim.'])
    report['pass'] = report.pop('pass_')
    if descriptor is not None:
        report['upper_backlash_limit']=descriptor
        report['scope']='Owned15 plus explicit upper14 backlash limit solref time constants0.02->0.005s; all other model fields unchanged'
    candidate.replace(scene)
    (directory/'htd_backlash_limit_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def validate(directory, expected_upper_timeconst=None):
    """Read-only exact selection/XML/hash proof; host pins all audit artifacts.

    None selects the legacy15 audit, not an arbitrary configured upper variant.
    Native serialized equivalence was proved before publication of scene.xml;
    this checks that precisely those audited inputs/output/source remain selected.
    """
    descriptor=upper_descriptor(expected_upper_timeconst)
    directory=Path(directory)
    report=json.loads((directory/'htd_backlash_limit_audit.json').read_text())
    selected=OWNED_BACKLASH+(UPPER_BACKLASH if descriptor is not None else ())
    native=dict(mujoco_version='3.2.7',changed_joints=len(selected),
        all_other_serialized_model_fields_equal=True,changed_joint_names=list(selected))
    expected=dict(pass_=True,simulation_only=True,timeconst_s=.005,damping_ratio=1.,
        owned_sdk_indices=list(range(15)),native_audit=native,upper_backlash_limit=descriptor)
    expected['pass']=expected.pop('pass_')
    # JSON equality preserves bool/int/float distinctions that Python dict == loses.
    canonical=lambda value:json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':'))
    if canonical({key:report.get(key)for key in expected})!=canonical(expected):
        raise ValueError('Backlash audit differs from explicit selected15/29 profile')
    source=Path(__file__).with_suffix('.cpp')
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    original=directory/'htd_backlash_original_scene.xml';scene=directory/'scene.xml'
    if (report.get('source_sha256')!=digest(source)
            or report.get('original_scene_sha256')!=digest(original)
            or report.get('variant_scene_sha256')!=digest(scene)):
        raise ValueError('Backlash source or original/variant scene hash changed')
    before=ET.parse(original).getroot();after=ET.parse(scene).getroot()
    for name in selected:
        a=[j for j in before.iter('joint')if j.get('name')==name]
        b=[j for j in after.iter('joint')if j.get('name')==name]
        if len(a)!=1 or len(b)!=1 or b[0].get('solreflimit')!='0.005 1':
            raise ValueError('Backlash selected joint/XML timeconstant differs: '+name)
        if 'solreflimit' in a[0].attrib:
            b[0].set('solreflimit',a[0].get('solreflimit'))
        else:
            b[0].attrib.pop('solreflimit')
    if ET.tostring(before)!=ET.tostring(after):
        raise ValueError('Backlash XML changes fields outside selected timeconstants')
    return report
