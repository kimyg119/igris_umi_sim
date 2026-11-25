"""Fresh-scene numerical variant checks; no ROS, controller or full probe launch."""
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import pytest

import htd_backlash_limits as limits

ROOT=Path(__file__).resolve().parents[2]
SCENE=ROOT/'outputs/hlik_transport_20260918_2005/htd_native_probe15/run/scene.xml'


def test_default_does_not_read_or_change_scene(tmp_path):
    scene=tmp_path/'scene.xml';scene.write_text('deliberately not XML')
    assert limits.configure(tmp_path,None) is None
    assert scene.read_text()=='deliberately not XML'
    assert list(tmp_path.iterdir())==[scene]


@pytest.mark.parametrize('tc',[0.,-.005,.002,.02,float('nan'),float('inf')])
def test_unreviewed_profile_refused_before_scene_io(tmp_path,tc):
    with pytest.raises(ValueError):limits.configure(tmp_path,tc)
    assert not list(tmp_path.iterdir())


def test_missing_owned_joint_refused_without_changing_original(tmp_path):
    scene=tmp_path/'scene.xml';scene.write_text('<mujoco><worldbody/></mujoco>');before=scene.read_bytes()
    with pytest.raises(ValueError,match='owned'):limits.configure(tmp_path,.005)
    assert scene.read_bytes()==before


def test_probe_variant_requires_explicit_serial_pd_profile(monkeypatch,tmp_path):
    import htd_native_probe as probe
    calls=[]
    monkeypatch.setattr(probe,'worker',lambda out,args:calls.append(args) or 0)
    base=['htd_native_probe','worker','--out',str(tmp_path)]
    monkeypatch.setattr('sys.argv',base)
    assert probe.main()==0 and calls[-1].backlash_limit_timeconst is None
    monkeypatch.setattr('sys.argv',base+['--backlash-limit-timeconst','.005'])
    with pytest.raises(SystemExit):probe.main()
    serial=base+['--serial-pd-torque','--output-coordinates','--low-warmup-seconds','3.2',
                 '--serial-pd-delay-steps','0','0','0','0']
    monkeypatch.setattr('sys.argv',serial+['--backlash-limit-timeconst','.005'])
    assert probe.main()==0 and calls[-1].backlash_limit_timeconst==.005
    monkeypatch.setattr('sys.argv',serial+['--backlash-limit-timeconst','.002'])
    with pytest.raises(SystemExit):probe.main()


def test_native_load_accepts_exact15_changes_and_rejects_nonowned_or_range_change(tmp_path):
    scene=tmp_path/'scene.xml';shutil.copyfile(SCENE,scene);before=scene.read_bytes()
    report=limits.configure(tmp_path,.005)
    assert report['pass'] is True
    assert set(report['native_audit']['changed_joint_names'])==set(limits.OWNED_BACKLASH)
    assert report['native_audit']['changed_joints']==15
    assert report['native_audit']['all_other_serialized_model_fields_equal'] is True
    assert (tmp_path/'htd_backlash_original_scene.xml').read_bytes()==before
    tree=ET.parse(scene)
    assert all(tree.find(f".//joint[@name='{n}']").get('solreflimit')=='0.005 1' for n in limits.OWNED_BACKLASH)
    with pytest.raises(FileExistsError):limits.configure(tmp_path,.005)
    for key,value in [('nonowned','0.005 1'),('range','-0.006 0.005')]:
        tree=ET.parse(scene)
        if key=='nonowned':tree.find(".//joint[@name='l_shoulder_pitch_backlash']").set('solreflimit',value)
        else:tree.find(".//joint[@name='l_hip_yaw_backlash']").set('range',value)
        candidate=tmp_path/(key+'.xml');tree.write(candidate)
        with pytest.raises(RuntimeError):
            limits.audit_model(tmp_path/'htd_backlash_audit',tmp_path/'htd_backlash_original_scene.xml',candidate,.005)
