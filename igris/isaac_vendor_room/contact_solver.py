"""Explicit experimental friction refinement, separate from the vendor baseline."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def apply_palm_torsion(out,coefficient_m):
    """Override only the two palm/payload spin coefficients in a fresh scene."""
    out=Path(out);coefficient_m=float(coefficient_m)
    if not math.isfinite(coefficient_m) or not 0<coefficient_m<=.05:
        raise ValueError('Palm torsion coefficient must be finite, positive and at most .05 m')
    if any((out/name).exists() for name in ('runtime.json','poses.bin')):
        raise ValueError('Palm torsion requires a fresh, unstarted scene')
    manifest=json.loads((out/'manifest.json').read_text())
    if 'palm_torsion' in manifest or (out/'palm_torsion_audit.json').exists():
        raise ValueError('Palm torsion already configured')
    path=out/'scene.xml';before=hashlib.sha256(path.read_bytes()).hexdigest();tree=ET.parse(path)
    payload=manifest['payload']['geom'];hands=manifest['hand_environment_contact']['geom_names']
    palms={'l_hand_본체1_5_geom','r_hand_본체1_5_geom'}
    if len(hands)!=76 or len(set(hands))!=76 or not palms<=set(hands):
        raise ValueError('Expected all 76 hand geoms including both palms')
    pairs=[p for p in tree.iter('pair') if p.get('name','').startswith('paper_silicone_')]
    expected={frozenset((h,payload)) for h in hands}
    actual=[frozenset((p.get('geom1'),p.get('geom2'))) for p in pairs]
    if len(pairs)!=76 or set(actual)!=expected:
        raise ValueError('Expected exactly 76 unique hand/payload material pairs')
    all_pairs=[frozenset((p.get('geom1'),p.get('geom2'))) for p in tree.iter('pair')]
    if any(all_pairs.count(pair)!=1 for pair in expected):
        raise ValueError('Duplicate hand/payload contact pair')
    selected=[]
    for pair in pairs:
        hand=next(g for g in (pair.get('geom1'),pair.get('geom2')) if g!=payload)
        values=pair.get('friction','').split()
        try:valid=len(values)==5 and all(math.isfinite(float(v)) and float(v)>0 for v in values)
        except ValueError:valid=False
        if not valid or pair.get('condim')!='4':
            raise ValueError('Expected condim4 hand pairs with five positive finite friction values')
        if hand in palms:selected.append((pair,hand,values))
    records=[]
    for pair,hand,values in selected:
        original=values.copy();values[2]=str(coefficient_m);pair.set('friction',' '.join(values))
        records.append(dict(pair=pair.get('name'),hand_geom=hand,payload_geom=payload,
                            original_friction=list(map(float,original)),friction=list(map(float,values))))
    tree.write(path,encoding='unicode')
    report=dict(coefficient_m=coefficient_m,pair_count=len(records),unchanged_hand_pair_count=74,
                scope='Native palm/payload spin friction only; two explicit condim4 pairs',
                tangential_normal_rolling_robot_geometry_masses_unchanged=True,
                usd_scalar_friction_unchanged=True,measured_on_this_hand=False,
                physical_interpretation='Experimental soft-contact torsion length; not a calibrated silicone parameter',
                physics_steps=0,native_contact_probe_pending=True,pairs=records,
                original_scene_sha256=before,scene_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source='https://mujoco.readthedocs.io/en/3.2.7/computation/#contact')
    manifest['palm_torsion']=report
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'palm_torsion_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def apply_pair_friction(out,time_constant,include_feet=False,hand_time_constant=None):
    """Refine selected tangential contacts, with explicit optional foot pairs."""
    import itertools
    import numpy as np
    import mujoco
    out=Path(out)
    if (out/'runtime.json').exists():raise ValueError('Only fresh scenes may change')
    manifest=json.loads((out/'manifest.json').read_text())
    if 'pair_friction_solver' in manifest:raise ValueError('Pair friction already configured')
    path=out/'scene.xml';before=hashlib.sha256(path.read_bytes()).hexdigest();tree=ET.parse(path)
    compatible=ET.fromstring(ET.tostring(tree.getroot()))
    for element in compatible.iter():
        if 'dampratio' in element.attrib and element.get('kv')=='0':element.attrib.pop('kv')
    model=mujoco.MjModel.from_xml_string(ET.tostring(compatible,encoding='unicode'))
    if not np.isfinite(time_constant) or not 2*model.opt.timestep<=time_constant<=.02:
        raise ValueError('Friction time constant must be at least two physics steps and at most .02 s')
    hand_time_constant=time_constant if hand_time_constant is None else hand_time_constant
    if not np.isfinite(hand_time_constant) or not 2*model.opt.timestep<=hand_time_constant<=.02:
        raise ValueError('Hand friction time constant must be two physics steps to .02 s')
    contact=tree.find('contact');hands=[]
    for pair in contact.findall('pair'):
        if pair.get('name','').startswith('paper_silicone_'):
            pair.set('solreffriction',f'{hand_time_constant} 1');hands.append(pair.get('name'))
    if len(hands)!=76:raise ValueError('Expected all 76 material hand pairs')
    source_bodies=manifest.get('source_support_bodies',[
        'room_Boxes__'+x for x in ('Pallet_C1_L1','Pallet_C1_L2','Pallet_C2_L1','Pallet_C1_L3')])
    bodies=set(manifest['destination_bodies'])|{manifest['payload']['body']}|set(source_bodies)
    geoms=[i for i in range(model.ngeom) if model.body(int(model.geom_bodyid[i])).name in bodies]
    geoms.append(model.geom('floor').id);records=[]
    for i,j in itertools.combinations(geoms,2):
        if model.geom_priority[i]!=model.geom_priority[j] or not np.array_equal(model.geom_solref[i],model.geom_solref[j]) or not np.array_equal(model.geom_solimp[i],model.geom_solimp[j]):
            raise ValueError('Normal-contact mixing requires a separate audit')
        if not(model.geom_contype[i]&model.geom_conaffinity[j] or model.geom_contype[j]&model.geom_conaffinity[i]):
            raise ValueError('Pair would enable a disabled contact')
        if np.any(model.geom_margin[[i,j]]) or np.any(model.geom_gap[[i,j]]):raise ValueError('Unexpected contact margin/gap')
        friction=np.maximum(model.geom_friction[i],model.geom_friction[j])
        values=dict(name=f'box_friction_{i}_{j}',geom1=model.geom(i).name,geom2=model.geom(j).name,
                    condim=str(max(model.geom_condim[i],model.geom_condim[j])),
                    friction=' '.join(map(str,[friction[0],friction[0],friction[1],friction[2],friction[2]])),
                    solref=' '.join(map(str,model.geom_solref[i])),solimp=' '.join(map(str,model.geom_solimp[i])),
                    solreffriction=f'{time_constant} 1',margin='0',gap='0')
        ET.SubElement(contact,'pair',values);records.append(values)
    foot_records=[]
    if include_feet:
        floor=model.geom('floor').id
        for side in ('l','r'):
            foot=model.geom(side+'_foot_original_collision').id
            # The existing foot has higher priority than the floor: retain its
            # complete winning normal/friction parameters, changing only tau.
            if model.geom_priority[foot]<=model.geom_priority[floor]:
                raise ValueError('Expected foot priority to determine contact parameters')
            if not(model.geom_contype[foot]&model.geom_conaffinity[floor] or model.geom_contype[floor]&model.geom_conaffinity[foot]):
                raise ValueError('Foot pair would enable a disabled contact')
            if np.any(model.geom_margin[[foot,floor]]) or np.any(model.geom_gap[[foot,floor]]):
                raise ValueError('Unexpected foot contact margin/gap')
            friction=model.geom_friction[foot]
            values=dict(name=f'foot_friction_{side}',geom1=model.geom(foot).name,geom2='floor',
                        condim=str(model.geom_condim[foot]),
                        friction=' '.join(map(str,[friction[0],friction[0],friction[1],friction[2],friction[2]])),
                        solref=' '.join(map(str,model.geom_solref[foot])),
                        solimp=' '.join(map(str,model.geom_solimp[foot])),
                        solreffriction=f'{time_constant} 1',margin='0',gap='0')
            ET.SubElement(contact,'pair',values);foot_records.append(values)
    tree.write(path,encoding='unicode')
    report=dict(friction_time_constant_s=time_constant,hand_pair_count=len(hands),support_pair_count=len(records),
                hand_friction_time_constant_s=hand_time_constant,
                foot_pair_count=len(foot_records),
                scope=('Payload/hand, source/receiving boxes and floor, plus two foot/floor pairs' if include_feet
                       else 'Payload/hand plus source/receiving boxes and floor; robot/foot pairs unchanged'),
                global_solver_unchanged=True,normal_contact_and_friction_coefficients_unchanged=True,
                original_scene_sha256=before,scene_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                physics_steps=0,pairs=records+foot_records)
    manifest['pair_friction_solver']={k:v for k,v in report.items() if k!='pairs'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'pair_friction_solver_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return manifest['pair_friction_solver']


def apply_impratio(out,ratio):
    """Declare global friction/normal impedance ratio; retain Coulomb coefficients."""
    out=Path(out);ratio=float(ratio)
    if not math.isfinite(ratio) or not 1<=ratio<=1000:
        raise ValueError('Expected finite friction impedance ratio from 1 to 1000')
    if (out/'runtime.json').exists():raise ValueError('Only fresh scenes may change')
    manifest=json.loads((out/'manifest.json').read_text())
    if 'friction_impedance' in manifest:raise ValueError('Friction impedance already configured')
    path=out/'scene.xml';before=hashlib.sha256(path.read_bytes()).hexdigest()
    tree=ET.parse(path);option=tree.find('option')
    previous=float(option.get('impratio','1'));option.set('impratio',str(ratio))
    tree.write(path,encoding='unicode')
    report=dict(impratio=ratio,previous_impratio=previous,
                scope='Global friction/normal constraint impedance; changes numerical slip, not Coulomb friction',
                mass_friction_timestep_actuators_unchanged=True,physics_steps=0,
                original_scene_sha256=before,scene_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source='https://mujoco.readthedocs.io/en/3.2.7/modeling.html#preventing-slip')
    manifest['friction_impedance']=report
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'friction_impedance_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def apply(out,iterations):
    out=Path(out)
    if not isinstance(iterations,int) or not 1<=iterations<=10:
        raise ValueError('Expected 1 to 10 friction refinement iterations')
    if (out/'runtime.json').exists():raise ValueError('Only fresh, unstarted scenes may change')
    manifest=json.loads((out/'manifest.json').read_text())
    if 'contact_solver' in manifest:raise ValueError('Contact solver already configured')
    path=out/'scene.xml';before=hashlib.sha256(path.read_bytes()).hexdigest()
    tree=ET.parse(path);option=tree.find('option')
    previous=int(option.get('noslip_iterations','0'))
    option.set('noslip_iterations',str(iterations));tree.write(path,encoding='unicode')
    report=dict(noslip_iterations=iterations,previous_noslip_iterations=previous,
                scope='Global MuJoCo tangential-contact postprocessing; vendor default is not retained',
                mass_friction_timestep_actuators_unchanged=True,physics_steps=0,
                original_scene_sha256=before,scene_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source='https://mujoco.readthedocs.io/en/3.2.7/modeling.html#preventing-slip')
    manifest['contact_solver']=report
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'contact_solver_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--iterations',type=int)
    choice.add_argument('--pair-friction-time',type=float)
    choice.add_argument('--impratio',type=float)
    choice.add_argument('--palm-torsion-m',type=float)
    parser.add_argument('--include-feet',action='store_true')
    parser.add_argument('--hand-friction-time',type=float)
    args=parser.parse_args()
    if args.include_feet and args.pair_friction_time is None:parser.error('--include-feet requires --pair-friction-time')
    if args.hand_friction_time is not None and args.pair_friction_time is None:parser.error('--hand-friction-time requires --pair-friction-time')
    print(json.dumps(apply_palm_torsion(args.out,args.palm_torsion_m) if args.palm_torsion_m is not None else apply_impratio(args.out,args.impratio) if args.impratio is not None else apply(args.out,args.iterations) if args.iterations is not None else apply_pair_friction(args.out,args.pair_friction_time,args.include_feet,args.hand_friction_time),indent=2))
