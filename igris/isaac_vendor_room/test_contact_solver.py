"""Friction refinement must preserve physical parameters and active scenes."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET
from contact_solver import apply_impratio


def check():
    original='<mujoco><option timestep=".001"/><worldbody><body><freejoint/><geom type="box" size=".1 .1 .1" mass="1" friction=".6 .005 .0001"/></body></worldbody></mujoco>'
    with TemporaryDirectory() as directory:
        out=Path(directory);scene=out/'scene.xml';manifest=out/'manifest.json'
        scene.write_text(original);manifest.write_text('{}')
        for bad in [float('nan'),float('inf'),0,-1,1001]:
            try:apply_impratio(out,bad)
            except ValueError:pass
            else:raise AssertionError('Invalid solver parameter accepted')
            assert scene.read_text()==original and manifest.read_text()=='{}'
        (out/'runtime.json').write_text('{}')
        try:apply_impratio(out,100)
        except ValueError:pass
        else:raise AssertionError('A running scene was modified')
        assert scene.read_text()==original
        (out/'runtime.json').unlink()
        report=apply_impratio(out,100)
        changed=ET.parse(scene);assert changed.find('option').get('impratio')=='100.0'
        changed.find('option').attrib.pop('impratio')
        assert ET.canonicalize(ET.tostring(changed.getroot(),encoding='unicode'))==ET.canonicalize(original)
        assert report['previous_impratio']==1 and report['impratio']==100
        recorded=scene.read_bytes()
        try:apply_impratio(out,10)
        except ValueError:pass
        else:raise AssertionError('An existing setting was overwritten')
        assert scene.read_bytes()==recorded
    print('PASS: only impedance changed; invalid, active and duplicate scenes rejected')

if __name__=='__main__':check()
