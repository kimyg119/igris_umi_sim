import copy
import json
from pathlib import Path
import pytest

P=Path(__file__).resolve().parents[2]/'outputs/hlik_transport_20260918_2005/task1_near_layout03/run'


def fixture():
    return json.loads((P/'manifest.json').read_text()),json.loads((P/'material_audit.json').read_text())


def test_prepared_contact_chain_accepts_exact_native_audited_profile():
    from transport_trial import validate_prepared_inplace_contacts
    import hashlib
    m,a=fixture();before=copy.deepcopy(m)
    validate_prepared_inplace_contacts(m,a,hashlib.sha256((P/'scene.xml').read_bytes()).hexdigest(),4.,.001,True,.005,.02)
    assert m==before


@pytest.mark.parametrize('mutation',['scene','chain','mu','hand_tau','feet','spin'])
def test_prepared_chain_or_requested_parameter_mismatch_rejected(mutation):
    from transport_trial import validate_prepared_inplace_contacts
    m,a=fixture();sha=m['palm_torsion']['scene_sha256'];args=[4.,.001,True,.005,.02]
    if mutation=='scene':sha='wrong'
    elif mutation=='chain':m['palm_torsion']['original_scene_sha256']='wrong'
    else:args[{'mu':0,'hand_tau':3,'feet':2,'spin':4}[mutation]]={ 'mu':2.,'hand_tau':.002,'feet':False,'spin':.01}[mutation]
    with pytest.raises(ValueError):validate_prepared_inplace_contacts(m,a,sha,*args)
