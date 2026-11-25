"""Reuse the previously simulated MCP motor and passive DIP spring, without changing the vendor body."""
import copy
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'igris/sim_local/vr_box_eval/results/20260914_synthetic_motion100_1kg'
CONTROL=ROOT/'igris/sim_local/vr_box_eval/compliant_hands.py'
MOTOR=CONTROL.with_name('hand_motor.cpp')
PYTHON=Path(os.environ.get('IGRIS_HAND_PYTHON',str(ROOT.parent/'curobo/.venv/bin/python'))).expanduser()
URDF=ROOT/'outputs/hl_ik_20260915/model/igris_c_rcv_hand_torso.urdf'


def install(root,out):
    source=ET.parse(SOURCE/'scene.xml').getroot()
    original=copy.deepcopy(root)
    def joints(r):return [j for j in r.find('worldbody').iter() if j.tag in ('joint','freejoint')]
    def width(j):return 7 if j.tag=='freejoint' or j.get('type')=='free' else 4 if j.get('type')=='ball' else 1
    old=joints(root);keys=[]
    for key in root.findall('./keyframe/key'):
        values=np.fromstring(key.get('qpos'),sep=' ');at=0;mapping={}
        for j in old:
            n=width(j);mapping[j.get('name')]=values[at:at+n];at+=n
        if at!=len(values):raise ValueError('Invalid manufacturer keyframe')
        keys.append((key,mapping))
    hand_names=[]
    for side in ('l','r'):
        name=side+'_hand_rcv';target=root.find(f".//body[@name='{name}']")
        hand=copy.deepcopy(source.find(f".//body[@name='{name}']"))
        if hand is None or target is None:raise ValueError('Missing hand model')
        if hand.get('pos')!=target.get('pos') or hand.get('quat')!=target.get('quat'):
            raise ValueError('Hand attachment frame differs')
        total=sum(float(b.find('inertial').get('mass')) for b in hand.iter('body'))
        if not np.isclose(total,.6):raise ValueError('Expected original 0.6 kg per hand')
        for geom in hand.iter('geom'):
            geom.set('contype','0');geom.set('conaffinity','0');hand_names.append(geom.get('name'))
        parent=next(p for p in root.iter() if target in list(p));parent[list(parent).index(target)]=hand
    custom=root.find('custom')
    if custom is None:custom=ET.SubElement(root,'custom')
    custom.append(copy.deepcopy(source.find("./custom/numeric[@name='hand_motor_config']")))
    added={s+'_hand_mcp_compliant':np.deg2rad(30.) for s in ('l','r')}
    added.update({s+'_hand_dip_spring':0. for s in ('l','r')})
    for key,mapping in keys:
        mapping.update({n:[v] for n,v in added.items()})
        key.set('qpos',' '.join(str(v) for j in joints(root) for v in mapping[j.get('name')]))
        if key.get('qvel'):raise ValueError('Explicit keyframe velocities need hand insertion')
    # Strip precisely the authorized additions and compare every other XML field.
    restored=copy.deepcopy(root)
    for side in ('l','r'):
        name=side+'_hand_rcv';hand=restored.find(f".//body[@name='{name}']")
        parent=next(p for p in restored.iter() if hand in list(p))
        parent[list(parent).index(hand)]=copy.deepcopy(original.find(f".//body[@name='{name}']"))
    c=restored.find('custom');c.remove(c.find("numeric[@name='hand_motor_config']"))
    if original.find('custom') is None:restored.remove(c)
    for a,b in zip(restored.findall('./keyframe/key'),original.findall('./keyframe/key')):a.set('qpos',b.get('qpos'))
    canonical=lambda r:ET.canonicalize(ET.tostring(r,encoding='unicode'),strip_text=True)
    if canonical(restored)!=canonical(original):raise ValueError('Unexpected non-hand change')
    metadata=json.loads((SOURCE/'compliance.json').read_text())
    metadata.update(non_hand_manufacturer_xml_unchanged=True,source_scene=str(SOURCE/'scene.xml'),
                    sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE/'scene.xml',SOURCE/'compliance.json',CONTROL,MOTOR]},
                    hand_geoms=hand_names,global_solver_options_changed=False)
    (out/'compliance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata
