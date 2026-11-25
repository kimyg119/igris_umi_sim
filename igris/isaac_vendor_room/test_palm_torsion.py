"""Two-palm-only mutation and actual native3.2.7 contact parameter checks."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest
import contact_solver

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'outputs/hlik_transport_20260918_2005/task2_layout_narrow01/run'
PALMS={'l_hand_본체1_5_geom','r_hand_본체1_5_geom'}


@pytest.fixture
def fresh(tmp_path):
    for name in ('scene.xml','manifest.json'):
        shutil.copyfile(SOURCE/name,tmp_path/name)
    return tmp_path


def test_only_two_palm_spin_components_change_and_provenance_is_recorded(fresh):
    before=(fresh/'scene.xml').read_bytes();old=ET.fromstring(before)
    manifest=json.loads((fresh/'manifest.json').read_text())
    report=contact_solver.apply_palm_torsion(fresh,.02)
    after=ET.parse(fresh/'scene.xml').getroot();changed=[]
    for a,b in zip(old.iter('pair'),after.iter('pair'),strict=True):
        if a.attrib==b.attrib:continue
        changed.append(b.get('geom1'))
        assert b.get('geom1')in PALMS and b.get('geom2')==manifest['payload']['geom']
        x,y=a.get('friction').split(),b.get('friction').split()
        assert float(y[2])==.02 and x[:2]+x[3:]==y[:2]+y[3:]
        assert b.get('condim')=='4'
        b.set('friction',a.get('friction'))
    assert set(changed)==PALMS and len(changed)==2
    assert ET.canonicalize(ET.tostring(old,encoding='unicode'))==ET.canonicalize(ET.tostring(after,encoding='unicode'))
    updated=json.loads((fresh/'manifest.json').read_text())
    assert updated.pop('palm_torsion')==report
    assert updated==manifest
    assert report['coefficient_m']==.02 and report['pair_count']==2
    assert report['original_scene_sha256']==hashlib.sha256(before).hexdigest()
    assert report['scene_sha256']==hashlib.sha256((fresh/'scene.xml').read_bytes()).hexdigest()
    assert json.loads((fresh/'palm_torsion_audit.json').read_text())==report
    recorded=(fresh/'scene.xml').read_bytes()
    with pytest.raises(ValueError):contact_solver.apply_palm_torsion(fresh,.04)
    assert (fresh/'scene.xml').read_bytes()==recorded


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),0,-.001,.05001])
def test_invalid_coefficient_cannot_mutate_scene(fresh,bad):
    before={p.name:p.read_bytes()for p in fresh.iterdir()}
    with pytest.raises(ValueError):contact_solver.apply_palm_torsion(fresh,bad)
    assert {p.name:p.read_bytes()for p in fresh.iterdir()}==before


@pytest.mark.parametrize('marker',['runtime.json','poses.bin'])
def test_started_scene_cannot_be_modified(fresh,marker):
    (fresh/marker).write_text('existing')
    before={p.name:p.read_bytes()for p in fresh.iterdir()}
    with pytest.raises(ValueError):contact_solver.apply_palm_torsion(fresh,.02)
    assert {p.name:p.read_bytes()for p in fresh.iterdir()}==before


@pytest.mark.parametrize('fault',['missing_palm','duplicate_pair','wrong_payload','wrong_dimension','bad_friction'])
def test_ambiguous_material_contract_is_rejected_before_mutation(fresh,fault):
    tree=ET.parse(fresh/'scene.xml');contact=tree.find('contact');pair=next(p for p in contact if p.get('geom1')in PALMS)
    if fault=='missing_palm':contact.remove(pair)
    elif fault=='duplicate_pair':ET.SubElement(contact,'pair',dict(pair.attrib,name='unexpected_duplicate'))
    elif fault=='wrong_payload':pair.set('geom2','floor')
    elif fault=='wrong_dimension':pair.set('condim','3')
    else:pair.set('friction','2 2 nan .0001 .0001')
    tree.write(fresh/'scene.xml',encoding='unicode')
    before={p.name:p.read_bytes()for p in fresh.iterdir()}
    with pytest.raises(ValueError):contact_solver.apply_palm_torsion(fresh,.02)
    assert {p.name:p.read_bytes()for p in fresh.iterdir()}==before


@pytest.fixture(scope='module')
def native_audit(tmp_path_factory):
    out=tmp_path_factory.mktemp('native_palm_audit')/'audit_material'
    lib=ROOT/'igris/ws/install/igris_c_mujoco/lib/igris_c_mujoco'
    subprocess.run(['g++','-O2','-std=c++17',str(ROOT/'igris/isaac_vendor_room/audit_material.cpp'),
                    '-I'+str(ROOT/'igris/ws/install/igris_c_mujoco/include'),'-L'+str(lib),'-Wl,-rpath,'+str(lib),
                    '-l:libmujoco.so.3.2.7','-o',str(out)],check=True)
    return out


def test_native_contacts_check_two_overrides_and_74_original_pairs(fresh,native_audit):
    baseline=fresh/'baseline_audit.json'
    subprocess.run([str(native_audit),str(fresh/'scene.xml'),str(baseline),'2'],check=True)
    assert json.loads(baseline.read_text())['pass']
    contact_solver.apply_palm_torsion(fresh,.02)
    output=fresh/'audit.json'
    args=[str(native_audit),str(fresh/'scene.xml'),str(output),'2','0','without-feet','0','66','.02']
    subprocess.run(args,check=True)
    result=json.loads(output.read_text())
    assert result['pass']and result['mujoco_version']=='3.2.7'and result['physics_steps']==0
    assert result['palm_torsion_pairs']==2 and result['original_torsion_pairs']==74
    assert result['contact_probes']==76 and len(result['torsion_probes'])==76
    for probe in result['torsion_probes']:
        assert probe['contact_friction'][2]==(.02 if probe['hand_geom']in PALMS else .005)
        assert probe['contact_friction'][:2]==[2,2]
    # A mutation to one of the untouched 74 pairs must be detected natively.
    tree=ET.parse(fresh/'scene.xml');pair=next(p for p in tree.iter('pair')if p.get('name','').startswith('paper_silicone_')and p.get('geom1')not in PALMS)
    pair.set('friction','2 2 .03 .0001 .0001');tree.write(fresh/'scene.xml',encoding='unicode')
    assert subprocess.run(args).returncode==1
    assert not json.loads(output.read_text())['pass']
    for bad in ('nan','0','.05001','garbage'):
        assert subprocess.run(args[:-1]+[bad]).returncode==2
