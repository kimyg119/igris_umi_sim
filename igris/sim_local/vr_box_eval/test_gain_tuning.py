"""The GC comparison may change only terminal inertia, never vendor joint topology."""
import xml.etree.ElementTree as ET
import numpy as np
import evaluate
from tune_gains import hand_inertia_audit


def test_every_gain_profile_respects_user_arm_damping_limit():
    from tune_gains import PROFILES
    assert all(0<=kd<=5. for gains in PROFILES.values() for kp,kd in gains.values())


def test_matching_hand_inertia_preserves_controller_structure(tmp_path):
    evaluate.scene(.01, tmp_path, 1.2)
    audit = hand_inertia_audit(tmp_path, apply=True)
    source = ET.parse(evaluate.ROOT/'igris_c_description_public/urdf/igris_c_rcv_hand_ctrl.urdf').getroot()
    result = ET.parse(tmp_path/'matched_hand_ctrl.urdf').getroot()
    assert [ET.tostring(j) for j in source.findall('joint')] == [ET.tostring(j) for j in result.findall('joint')]
    assert len(source.findall('link')) == len(result.findall('link')) == 33
    for side in ('l','r'):
        assert np.isclose(audit[side]['native_mass_kg'], .6331102687755983)
        assert all('universal' not in name for name in audit[side]['bodies'])
        inertia = np.array(audit[side]['native_inertia_kg_m2'])
        assert np.linalg.eigvalsh(inertia).min() > 0
        name = side+'_wrist_connector'
        source.find(f'link[@name="{name}"]').remove(source.find(f'link[@name="{name}"]/inertial'))
        result.find(f'link[@name="{name}"]').remove(result.find(f'link[@name="{name}"]/inertial'))
    assert ET.tostring(source) == ET.tostring(result)
