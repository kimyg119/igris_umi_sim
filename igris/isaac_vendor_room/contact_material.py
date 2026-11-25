"""Apply a documented silicone/cardboard approximation to a fresh experiment.

Explicit pairs prevent the hand's default friction from overriding the lower
paper/silicone coefficient. Only tangential friction changes; normal contact,
torsion, rolling, collision eligibility and vendor dynamics are preserved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
from pxr import Usd, UsdPhysics, UsdShade


def apply(out, mu=.60):
    out = Path(out).resolve()
    if not np.isfinite(mu) or mu <= 0:
        raise ValueError('Positive finite friction required')
    if (out/'runtime.json').exists() or (out/'material_audit.json').exists():
        raise ValueError('Material changes require a fresh, unstarted scene')
    manifest = json.loads((out/'manifest.json').read_text())
    scene = out/'scene.xml'
    tree = ET.parse(scene)
    root = tree.getroot()
    # Host MuJoCo is used only to read compiled material defaults, never to step.
    # The actual vendor 3.2.7 model is audited separately before execution.
    compatible = ET.fromstring(ET.tostring(root))
    for element in compatible.iter():
        if 'dampratio' in element.attrib and element.get('kv') == '0':
            element.attrib.pop('kv')
    model = mujoco.MjModel.from_xml_string(ET.tostring(compatible, encoding='unicode'))
    payload = manifest['payload']['geom']
    pid = model.geom(payload).id
    hands = manifest['hand_environment_contact']['geom_names']
    contact = root.find('contact')
    if contact is None:
        contact = ET.SubElement(root, 'contact')
    records = []
    for name in hands:
        hid = model.geom(name).id
        if model.geom_priority[hid] != model.geom_priority[pid]:
            raise ValueError('Unexpected material priority; audit before adapting')
        if not (model.geom_contype[hid] & model.geom_conaffinity[pid]
                or model.geom_contype[pid] & model.geom_conaffinity[hid]):
            raise ValueError('Do not enable new collisions through material pairs')
        if not np.array_equal(model.geom_solref[hid], model.geom_solref[pid]) or not np.array_equal(model.geom_solimp[hid], model.geom_solimp[pid]):
            raise ValueError('Different normal-contact parameters need explicit mixing audit')
        if np.any(model.geom_margin[[hid,pid]]) or np.any(model.geom_gap[[hid,pid]]):
            raise ValueError('Unexpected contact margin/gap')
        friction = np.maximum(model.geom_friction[hid], model.geom_friction[pid])
        pair_friction = [mu, mu, friction[1], friction[2], friction[2]]
        dim = int(max(model.geom_condim[hid], model.geom_condim[pid]))
        attrs = dict(name='paper_silicone_'+name, geom1=name, geom2=payload,
                     condim=str(dim), friction=' '.join(map(str,pair_friction)),
                     solref=' '.join(map(str,model.geom_solref[hid])),
                     solimp=' '.join(map(str,model.geom_solimp[hid])), margin='0',gap='0')
        ET.SubElement(contact,'pair',attrs)
        records.append(attrs)
    original_hash = hashlib.sha256(scene.read_bytes()).hexdigest()
    geom = root.find(f".//geom[@name='{payload}']")
    values = np.fromstring(geom.get('friction'),sep=' ')
    values[0] = mu
    geom.set('friction',' '.join(map(str,values)))
    tree.write(scene,encoding='unicode')

    usd = Path(manifest['source_usd'])
    stage = Usd.Stage.Open(str(usd))
    material = UsdShade.Material.Define(stage,'/World/Materials/PaperSiliconeContact')
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateDynamicFrictionAttr(mu)
    physics.CreateStaticFrictionAttr(mu)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(manifest['payload']['usd_path']+'/Body_Collider')).Bind(material,materialPurpose='physics')
    stage.GetRootLayer().Save()
    manifest['payload']['friction'] = mu
    manifest['contact_material'] = dict(mu=mu, scope='payload/hand tangential contact',
        pair_count=len(records), static_equals_dynamic_approximation=True,
        measured_on_this_hand=False, coefficient_role='baseline silicone/paper approximation' if mu==.6 else 'experimental friction override',
        source='https://doi.org/10.1007/s11012-023-01719-5' if mu==.6 else None)
    manifest['sha256'][str(usd)] = hashlib.sha256(usd.read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    layout_path=out.parent/'layout.json'
    if layout_path.exists():
        layout=json.loads(layout_path.read_text())
        layout['payload']['friction']=mu
        layout['contact_material']=manifest['contact_material']
        layout_path.write_text(json.dumps(layout,indent=2)+'\n')
    report = dict(**manifest['contact_material'], original_scene_sha256=original_hash,
        scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
        pairs=records, normal_contact_unchanged=True, vendor_geoms_unchanged=True,
        physics_steps=0, native_contact_probe_pending=True)
    (out/'material_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mu',type=float,default=.6)
    a=p.parse_args()
    r=apply(a.out,a.mu)
    print(json.dumps({k:v for k,v in r.items() if k!='pairs'},indent=2))
