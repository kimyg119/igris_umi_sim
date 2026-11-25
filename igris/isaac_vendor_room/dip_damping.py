"""Explicit fresh-trial SIM DIP damping; installed/source models stay unchanged."""
import copy
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='igris_dip_damping/v1'
JOINTS=('l_hand_dip_spring','r_hand_dip_spring')
ORIGINALS=('vendor.xml','scene.xml','compliance.json','manifest.json')
INSTALL=ROOT/'igris/ws/install/igris_c_mujoco'
LIBRARY=INSTALL/'lib/igris_c_mujoco/libmujoco.so.3.2.7'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scope(manifest):
    if (manifest.get('hardware_executed') is not False
            or manifest.get('physics_backend')!='vendor MuJoCo 3.2.7'
            or manifest.get('evaluation_profile')!='relaxed_transport'
            or manifest.get('dip_damping_requested')!=.002
            or manifest.get('dip_damping_scope')!='loaded_transport'):
        raise ValueError('DIP damping requires explicit native SIM loaded_transport intent')
    walking=manifest.get('walking_transfer',{})
    if not all(walking.get(k) is True for k in ('heading_preserving_route','xy_first_lowering',
                                               'torso_relative_lowering','world_position_lowering')):
        raise ValueError('DIP damping requires the reviewed loaded transport route')
    actual=manifest.get('htd_loaded_zero')
    if actual is not None and (actual.get('enabled') is not True or actual.get('scope')!='loaded_transport'
            or actual.get('current_grasp_reference') is not True or actual.get('transport') is None):
        raise ValueError('DIP damping requires actual current-grasp HTD loaded_transport scope')


def _dip_joints(tree,damping):
    for name in JOINTS:
        matches=[j for j in tree.iter('joint') if j.get('name')==name]
        if (len(matches)!=1 or matches[0].get('type')!='hinge'
                or matches[0].get('limited')!='false'
                or float(matches[0].get('damping','nan'))!=damping):
            raise ValueError('Unexpected passive DIP joint/profile: '+name)
        yield matches[0]


def _audit(executable,original,candidate):
    result=subprocess.run([str(executable),str(original),str(candidate)],capture_output=True,text=True)
    if result.returncode:raise RuntimeError('Native DIP damping audit failed: '+result.stderr.strip())
    return json.loads(result.stdout)


def configure(trial,damping):
    """Before HTD setup/start, change exactly two DIP fields in both trial models."""
    if damping is None:return None
    if isinstance(damping,bool) or not isinstance(damping,Real) or not math.isfinite(damping) or damping!=.002:
        raise ValueError('Only explicit reviewed DIP damping0.002 is supported')
    trial=Path(trial).resolve()
    if not trial.is_relative_to(ROOT):raise ValueError('DIP variant must remain inside the project')
    forbidden=('runtime.json','poses.bin','physics_stats.json','htd_loaded_preflight.json',
               'htd_backlash_original_scene.xml','htd_gc_params.yaml','dip_damping.json','dip_damping_audit')
    if any((trial/name).exists() for name in forbidden) or any(trial.glob('dip_damping_original_*')):
        raise FileExistsError('DIP variant requires a fresh unconfigured trial before HTD setup')
    before={name:(trial/name).read_bytes() for name in ORIGINALS}
    manifest=json.loads(before['manifest.json']);_scope(manifest)
    if manifest.get('htd_loaded_zero') is not None or 'dip_damping' in manifest:
        raise FileExistsError('DIP variant must precede HTD configuration')
    compliance=json.loads(before['compliance.json'])
    if (manifest.get('compliant_hands')!=compliance
            or compliance.get('dip_spring',{}).get('damping_Nms_rad')!=.0001):
        raise ValueError('Original compliant-hand metadata/profile differs')
    trees={name:ET.fromstring(before[name]) for name in ('vendor.xml','scene.xml')}
    for tree in trees.values():
        for joint in _dip_joints(tree,.0001):joint.set('damping','.002')
    # Candidate/audit artifacts may be retained on failure; live models are only
    # replaced once both native serialized-model comparisons have passed.
    for name,data in before.items():(trial/('dip_damping_original_'+name)).write_bytes(data)
    source=Path(__file__).with_suffix('.cpp');executable=trial/'dip_damping_audit'
    subprocess.run(['g++','-O2','-std=c++17','-I'+str(INSTALL/'include'),str(source),
        '-L'+str(LIBRARY.parent),'-Wl,-rpath,'+str(LIBRARY.parent),'-l:libmujoco.so.3.2.7',
        '-o',str(executable)],check=True)
    audits={};candidates={}
    for name,tree in trees.items():
        candidate=trial/('dip_damping_candidate_'+name)
        ET.ElementTree(tree).write(candidate);candidates[name]=candidate
        audits[name]=_audit(executable,trial/('dip_damping_original_'+name),candidate)
    compliance=copy.deepcopy(compliance)
    compliance['dip_spring']['damping_Nms_rad']=.002
    configured_compliance=(json.dumps(compliance,indent=2)+'\n').encode()
    report=dict(schema=SCHEMA,scope='loaded_transport',simulation_only=True,physics_steps=0,
        joint_names=list(JOINTS),original_damping_Nms_rad=.0001,requested_damping_Nms_rad=.002,
        native_audits=audits,original_sha256={n:hashlib.sha256(b).hexdigest() for n,b in before.items()},
        configured_sha256={**{n:sha(p) for n,p in candidates.items()},
                           'compliance.json':hashlib.sha256(configured_compliance).hexdigest()},
        source_sha256={str(p.resolve()):sha(p) for p in (Path(__file__),source,LIBRARY)},
        audit_executable_sha256=sha(executable),
        passive_log_note='hand_motor dip_spring_torque_Nm is qfrc_passive, including spring and DIP viscous damping',
        limitation='Explicit uncalibrated SIM damping variant; no hardware or successful transport claim')
    descriptor=(json.dumps(report,indent=2)+'\n').encode()
    manifest.update(compliant_hands=compliance,dip_damping=report,
        dip_damping_descriptor_sha256=hashlib.sha256(descriptor).hexdigest())
    for name,path in candidates.items():path.replace(trial/name)
    (trial/'compliance.json').write_bytes(configured_compliance)
    (trial/'dip_damping.json').write_bytes(descriptor)
    (trial/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    validate(trial)
    return report


def validate(trial, *, manifest=None):
    """Read-only validation before setup/start and independently in the worker.

    Explicit intent exists before setup; if actual HTD scope is present it must
    agree. Setup may supply its prospective manifest before publishing it;
    prestart and worker independently reload the on-disk manifest.
    """
    trial=Path(trial).resolve()
    if manifest is None:
        manifest=json.loads((trial/'manifest.json').read_text())
    report=manifest.get('dip_damping')
    if report is None:
        if (any(k in manifest for k in ('dip_damping_requested','dip_damping_scope','dip_damping_descriptor_sha256'))
                or (trial/'dip_damping.json').exists() or any(trial.glob('dip_damping_*'))
                or manifest.get('compliant_hands',{}).get('dip_spring',{}).get('damping_Nms_rad',.0001)!=.0001
                or manifest.get('htd_loaded_zero',{}).get('dip_damping') is not None):
            raise ValueError('Missing explicit DIP damping descriptor')
        return None
    if not trial.is_relative_to(ROOT):raise ValueError('DIP variant must remain inside the project')
    _scope(manifest)
    if (not isinstance(report,dict) or report.get('schema')!=SCHEMA or report.get('scope')!='loaded_transport'
            or report.get('simulation_only') is not True or report.get('physics_steps')!=0
            or report.get('joint_names')!=list(JOINTS) or report.get('original_damping_Nms_rad')!=.0001
            or report.get('requested_damping_Nms_rad')!=.002):
        raise ValueError('Unknown DIP damping descriptor')
    if (sha(trial/'dip_damping.json')!=manifest.get('dip_damping_descriptor_sha256')
            or json.loads((trial/'dip_damping.json').read_text())!=report):
        raise ValueError('DIP damping descriptor hash/content changed')
    actual=manifest.get('htd_loaded_zero')
    if actual is not None and actual.get('dip_damping')!=report:
        raise ValueError('Actual HTD DIP descriptor differs')
    expected_sources={str(p.resolve()) for p in (Path(__file__),Path(__file__).with_suffix('.cpp'),LIBRARY)}
    if set(report.get('source_sha256',{}))!=expected_sources:
        raise ValueError('Unexpected DIP audit source paths')
    for path,expected in report['source_sha256'].items():
        if sha(path)!=expected:raise ValueError('DIP audit source hash changed: '+path)
    if sha(trial/'dip_damping_audit')!=report.get('audit_executable_sha256'):
        raise ValueError('DIP audit executable changed')
    if set(report.get('original_sha256',{}))!=set(ORIGINALS):
        raise ValueError('DIP original artifact set differs')
    for name,expected in report['original_sha256'].items():
        if sha(trial/('dip_damping_original_'+name))!=expected:
            raise ValueError('DIP original artifact hash changed: '+name)
    for name in ('vendor.xml','scene.xml'):
        audit=report.get('native_audits',{}).get(name,{})
        if (audit.get('mujoco_version')!='3.2.7' or audit.get('physics_steps')!=0
                or audit.get('changed_dofs')!=2 or audit.get('changed_joint_names')!=list(JOINTS)
                or audit.get('original_damping_Nms_rad')!=.0001 or audit.get('requested_damping_Nms_rad')!=.002
                or audit.get('all_other_serialized_model_fields_equal') is not True):
            raise ValueError('DIP native audit does not prove exactly two damping changes')
        list(_dip_joints(ET.parse(trial/name).getroot(),.002))
    if sha(trial/'vendor.xml')!=report['configured_sha256'].get('vendor.xml'):
        raise ValueError('Configured DIP vendor model changed')
    if sha(trial/'scene.xml')!=report['configured_sha256'].get('scene.xml'):
        # Startup backlash configuration follows DIP preparation. Its pristine
        # input must be our exact configured scene. Independently verify the
        # declared owned15 or owned15+upper14 selection and compiled model.
        from htd_backlash_limits import validate as validate_backlash
        from htd_loaded import upper_backlash_selection
        audit_path=trial/'htd_backlash_limit_audit.json';original=trial/'htd_backlash_original_scene.xml'
        if not audit_path.exists() or not original.exists():raise ValueError('Configured DIP scene changed')
        if sha(original)!=report['configured_sha256']['scene.xml']:
            raise ValueError('DIP -> backlash original scene hash chain differs')
        validate_backlash(trial, expected_upper_timeconst=upper_backlash_selection(manifest))
    compliance=json.loads((trial/'compliance.json').read_text())
    original=json.loads((trial/'dip_damping_original_compliance.json').read_text())
    original['dip_spring']['damping_Nms_rad']=.002
    if (sha(trial/'compliance.json')!=report['configured_sha256'].get('compliance.json')
            or compliance!=original or manifest.get('compliant_hands')!=compliance):
        raise ValueError('DIP compliance metadata differs from the two-joint variant')
    return report
