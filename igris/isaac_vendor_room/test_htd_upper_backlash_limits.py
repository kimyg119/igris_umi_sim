"""Native3.2.7 serialized models only; no mjData, stepping, ROS or controller."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import pytest
import htd_backlash_limits as limits
from test_htd_backlash_limits import SCENE


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.mark.parametrize('upper', [True,False,'.005',.02,.002,0.,float('nan'),float('inf')])
def test_upper_value_refused_before_io(tmp_path,upper):
    with pytest.raises(ValueError):limits.configure(tmp_path,.005,upper_timeconst=upper)
    assert not list(tmp_path.iterdir())


def test_upper_requires_owned_profile_and_default_none_remains_noop(tmp_path):
    with pytest.raises(ValueError):limits.configure(tmp_path,None,upper_timeconst=.005)
    assert limits.configure(tmp_path,None,upper_timeconst=None) is None
    assert not list(tmp_path.iterdir())


@pytest.fixture(scope='module')
def variants(tmp_path_factory):
    root=tmp_path_factory.mktemp('upper_limit_native')
    results={}
    for key,upper in [('default',None),('upper',.005)]:
        out=root/key;out.mkdir();shutil.copyfile(SCENE,out/'scene.xml')
        before=(out/'scene.xml').read_bytes()
        report=limits.configure(out,.005,upper_timeconst=upper)
        assert (out/'htd_backlash_original_scene.xml').read_bytes()==before
        results[key]=(out,report)
    return results


def test_compiled29_only_added14_and_legacy15_report_unchanged(variants):
    default,old=variants['default'];out,report=variants['upper']
    assert 'upper_backlash_limit' not in old and old['native_audit']['changed_joints']==15
    descriptor=limits.upper_descriptor(.005)
    assert report['upper_backlash_limit']==descriptor
    assert descriptor['sdk_indices']==list(range(15,29)) and len(descriptor['joint_names'])==14
    assert report['owned_sdk_indices']==list(range(15))
    assert report['native_audit']['changed_joints']==29
    assert report['native_audit']['changed_joint_names']==list(limits.OWNED_BACKLASH+limits.UPPER_BACKLASH)
    assert report['native_audit']['all_other_serialized_model_fields_equal'] is True
    a=ET.parse(default/'scene.xml');b=ET.parse(out/'scene.xml')
    for n in limits.UPPER_BACKLASH:
        x=a.find(f".//joint[@name='{n}']");y=b.find(f".//joint[@name='{n}']")
        assert y.attrib==dict(x.attrib,solreflimit='0.005 1')
        y.attrib.clear();y.attrib.update(x.attrib)
    assert ET.tostring(a.getroot())==ET.tostring(b.getroot())
    assert limits.validate(default)==old
    assert limits.validate(out,expected_upper_timeconst=.005)==report
    with pytest.raises(ValueError):limits.validate(out)
    with pytest.raises(ValueError):limits.validate(default,expected_upper_timeconst=.005)
    with pytest.raises(FileExistsError):limits.configure(out,.005,upper_timeconst=.005)


@pytest.mark.parametrize('kind', ['upper_range','upper_solimp','upper_damping','upper_ratio','unselected','equality','actuator','room_mass'])
def test_native_rejects_every_change_beyond_selected_timeconstants(variants,tmp_path,kind):
    out,_=variants['upper'];tree=ET.parse(out/'scene.xml')
    j=tree.find(".//joint[@name='l_shoulder_yaw_backlash']")
    if kind=='upper_range':j.set('range','-.006 .005')
    if kind=='upper_solimp':j.set('solimplimit','.8 .95 .001 .5 2')
    if kind=='upper_damping':tree.find(".//joint[@name='l_wrist_pitch']").set('damping','3')
    if kind=='upper_ratio':j.set('solreflimit','.005 .9')
    if kind=='unselected':tree.find(".//joint[@name='l_hand_dip_spring']").set('damping','.003')
    if kind=='equality':tree.find('.//equality/connect').set('solref','.003 1')
    if kind=='actuator':tree.find('.//actuator/motor').set('gear','2')
    if kind=='room_mass':
        body=tree.find(".//body[@name='room_Boxes__Small_RightFront']")
        g=body.find('geom');g.set('mass','0.2')
    candidate=tmp_path/'bad.xml';tree.write(candidate)
    with pytest.raises(RuntimeError):limits.audit_model(out/'htd_backlash_audit',out/'htd_backlash_original_scene.xml',candidate,.005,upper_timeconst=.005)


@pytest.mark.parametrize('kind',['name','count','source','descriptor','scene','original','undeclared'])
def test_readonly_validator_rejects_metadata_or_rehashed_extra_xml_change(variants,tmp_path,kind):
    out,report=variants['upper'];copydir=tmp_path/'case';shutil.copytree(out,copydir)
    r=copy.deepcopy(report)
    if kind=='name':r['native_audit']['changed_joint_names'][-1]='neck_yaw_backlash'
    if kind=='count':r['native_audit']['changed_joints']=15
    if kind=='source':r['source_sha256']='0'*64
    if kind=='descriptor':r['upper_backlash_limit']['sdk_indices'][-1]=29
    if kind=='scene':(copydir/'scene.xml').write_bytes(b'changed')
    if kind=='original':(copydir/'htd_backlash_original_scene.xml').write_bytes(b'changed')
    if kind=='undeclared':
        path=copydir/'scene.xml';tree=ET.parse(path);tree.find('.//option').set('gravity','0 0 0');tree.write(path)
        r['variant_scene_sha256']=sha(path)
    (copydir/'htd_backlash_limit_audit.json').write_text(json.dumps(r))
    with pytest.raises(ValueError):limits.validate(copydir,expected_upper_timeconst=.005)


@pytest.mark.parametrize('kind',['missing','duplicate','original_profile'])
def test_invalid_upper_joint_or_source_profile_keeps_scene_unchanged(tmp_path,kind):
    path=tmp_path/'scene.xml';shutil.copyfile(SCENE,path);tree=ET.parse(path)
    j=tree.find(".//joint[@name='l_shoulder_yaw_backlash']")
    if kind=='missing':j.set('name','not_the_reviewed_joint')
    if kind=='duplicate':tree.find('.//worldbody/body').append(copy.deepcopy(j))
    if kind=='original_profile':j.set('solreflimit','.03 1')
    tree.write(path);before=path.read_bytes()
    with pytest.raises((ValueError,RuntimeError)):limits.configure(tmp_path,.005,upper_timeconst=.005)
    assert path.read_bytes()==before


@pytest.mark.parametrize('kind',['descriptor_bool','descriptor_ratio','native_equal','native_count_float'])
def test_validator_exact_metadata_preserves_types(variants,tmp_path,kind):
    out,report=variants['upper'];case=tmp_path/'case';shutil.copytree(out,case);r=copy.deepcopy(report)
    if kind=='descriptor_bool':r['upper_backlash_limit']['simulation_only']=1
    if kind=='descriptor_ratio':r['upper_backlash_limit']['damping_ratio']=True
    if kind=='native_equal':r['native_audit']['all_other_serialized_model_fields_equal']=1
    if kind=='native_count_float':r['native_audit']['changed_joints']=29.0
    (case/'htd_backlash_limit_audit.json').write_text(json.dumps(r))
    with pytest.raises(ValueError):limits.validate(case,expected_upper_timeconst=.005)
