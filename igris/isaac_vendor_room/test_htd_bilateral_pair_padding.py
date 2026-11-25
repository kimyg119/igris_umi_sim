"""Bilateral padding remains a separate, explicitly scoped SIM diagnostic."""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import live_ik as live
import transport_trial as host
from igris.hl_ik.pipeline import PipelineIK

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'outputs/hlik_transport_20260918_2005'
PAIRS = [('pelvis', 'l_upper_leg'), ('pelvis', 'r_upper_leg')]
OVERRIDES = {pair: 0. for pair in PAIRS}


def metadata():
    return host.htd_pair_padding_metadata(False, bilateral=True, loaded_transport=True,
        current_grasp_reference=True, relaxed_grasp=True)


def manifest():
    return dict(hardware_executed=False, physics_backend='vendor MuJoCo 3.2.7',
        evaluation_profile='relaxed_transport', hl_ik_pair_padding=metadata(),
        htd_loaded_zero=dict(enabled=True, scope='loaded_transport', current_grasp_reference=True))


def test_default_and_right_v1_are_preserved():
    assert inspect.signature(host.run).parameters['htd_pelvis_thighs_unbuffered'].default is False
    assert host.htd_pair_padding_metadata(False, bilateral=False, loaded_transport=False,
        current_grasp_reference=False, relaxed_grasp=False) is None
    frozen=json.loads((P/'transfer_trial108/run/manifest.json').read_text())['hl_ik_pair_padding']
    actual=host.htd_pair_padding_metadata(True, loaded_transport=True,
        current_grasp_reference=True, relaxed_grasp=True)
    assert actual == frozen and actual['schema']=='htd_pelvis_right_thigh_unbuffered/v1'


@pytest.mark.parametrize('value', [None, 0, 1, 'true', np.bool_(True)])
def test_bilateral_strict_bool_before_mutation(tmp_path, value):
    out=tmp_path/'never'
    with pytest.raises(ValueError, match='pair padding.*bool'):
        host.run(out, tmp_path/'missing', htd_pelvis_thighs_unbuffered=value)
    assert not out.exists()


def test_flags_conflict_before_assets_or_mutations(tmp_path):
    out=tmp_path/'never'
    with pytest.raises(ValueError, match='pair padding.*mutually exclusive'):
        host.run(out, tmp_path/'missing', htd_pelvis_thighs_unbuffered=True,
                 htd_pelvis_right_thigh_unbuffered=True)
    assert not out.exists()


@pytest.mark.parametrize('field', ['loaded_transport','current_grasp_reference','relaxed_grasp'])
@pytest.mark.parametrize('value', [False, 1])
def test_bilateral_requires_exact_scope(field, value):
    scope=dict(loaded_transport=True,current_grasp_reference=True,relaxed_grasp=True)
    scope[field]=value
    with pytest.raises(ValueError, match='pair padding'):
        host.htd_pair_padding_metadata(False,bilateral=True,**scope)


def test_host_worker_roundtrip_exact_bilateral_and_frozen_evidence(tmp_path):
    m=manifest();m.pop('hl_ik_pair_padding');original=copy.deepcopy(m)
    path=tmp_path/'manifest.json';path.write_text(json.dumps(m))
    result=host.configure_htd_pair_padding(tmp_path,metadata())
    written=json.loads(path.read_text())
    assert {k:v for k,v in written.items() if k!='hl_ik_pair_padding'}==original
    assert result==written['hl_ik_pair_padding']
    assert live.htd_pair_buffer_overrides(written)==OVERRIDES
    assert result['pairs']==[list(pair) for pair in PAIRS]
    assert result['schema']=='htd_pelvis_thighs_unbuffered/v2'
    assert result['buffer_per_sphere_m']==0. and result['default_buffer_per_sphere_m']==.0025
    assert result['interior_margin_m']==.0005 and result['physics_modified'] is False
    assert len(result['evidence'])==2
    for record in result['evidence']:
        assert hashlib.sha256((ROOT/record['path']).read_bytes()).hexdigest()==record['sha256']
    assert result['evidence'][1]['sha256']=='ef19a1b3c1257a8279a5eb416dbb263c40c4a1c5b43ea4264250606be53e60ef'


@pytest.mark.parametrize('bad', ['missing_left','missing_right','extra_pair','duplicate','reverse',
    'pair_order','legacy_schema','legacy_pair_key','buffer_bool','padding','margin','evidence',
    'missing_evidence','extra_key','hardware','not_current','disabled','not_relaxed','wrong_scope'])
def test_worker_rejects_malformed_bilateral_before_assets(tmp_path,bad):
    m=manifest();d=m['hl_ik_pair_padding']
    if bad=='missing_left':d['pairs']=d['pairs'][1:]
    if bad=='missing_right':d['pairs']=d['pairs'][:1]
    if bad=='extra_pair':d['pairs'].append(['pelvis','l_lower_leg'])
    if bad=='duplicate':d['pairs'].append(d['pairs'][0])
    if bad=='reverse':d['pairs'][0]=d['pairs'][0][::-1]
    if bad=='pair_order':d['pairs'].reverse()
    if bad=='legacy_schema':d['schema']='htd_pelvis_right_thigh_unbuffered/v1'
    if bad=='legacy_pair_key':d['pair']=d['pairs'][0]
    if bad=='buffer_bool':d['buffer_per_sphere_m']=False
    if bad=='padding':d['default_buffer_per_sphere_m']=0.
    if bad=='margin':d['interior_margin_m']=0.
    if bad=='evidence':d['evidence'][1]['sha256']='0'*64
    if bad=='missing_evidence':d['evidence'].pop()
    if bad=='extra_key':d['disable_collisions']=True
    if bad=='hardware':m['hardware_executed']=True
    if bad=='not_current':m['htd_loaded_zero']['current_grasp_reference']=False
    if bad=='disabled':m['htd_loaded_zero']['enabled']=False
    if bad=='not_relaxed':m['evaluation_profile']='original_grasp_requirements'
    if bad=='wrong_scope':m['htd_loaded_zero']['scope']='loaded_goal_hold_only'
    (tmp_path/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='pair padding'):
        live.run(tmp_path,tmp_path/'missing-plan',tmp_path/'missing-model')
    assert not (tmp_path/'live_ik_ready.json').exists()


@pytest.mark.parametrize('flags', [['--htd-pelvis-thighs-unbuffered'],
    ['--htd-pelvis-thighs-unbuffered','--htd-pelvis-right-thigh-unbuffered']])
def test_cli_accepts_option_but_rejects_scope_or_conflict(tmp_path,flags):
    out=tmp_path/'never'
    r=subprocess.run([sys.executable,str(host.HERE/'transport_trial.py'),'--out',str(out),
        '--motion',str(tmp_path/'missing'),*flags],capture_output=True,text=True)
    assert r.returncode!=0 and 'unrecognized arguments' not in r.stderr
    assert 'pair padding' in r.stderr and not out.exists()


def test_actual108_seed_bilateral_raw_gap_and_unchanged_other_pairs():
    row=json.loads((P/'native108_geometry_audit01/seed_probe.json').read_text())['failed_tick']
    args=dict(model_path=P/'ik_model.xml',hand_envelope=ROOT/'igris/sim_local/vr_box_eval/results/20260914_synthetic_motion100_1kg/compliance.json',
        acceleration_limit=3.,joint_limit_cost=1000.,soft_joint_margin=.3,collision_braking=True,
        warm_start_primary=False,primary_max_iter=10,primary_max_iter_in=10)
    right=PipelineIK(**args,pair_buffer_overrides={PAIRS[1]:0.})
    both=PipelineIK(**args,pair_buffer_overrides=live.htd_pair_buffer_overrides(manifest()))
    for solver in (right,both):solver.set_q(row['q'])
    hit=right.geometry.collision(right.configuration.data)
    assert hit['pair']==list(PAIRS[0]) and hit['margin_m']==pytest.approx(-.000248053749,abs=1e-12)
    before=right.geometry.contacts(right.configuration.data,.02,derivatives=False,per_pair=1)
    expected={pair:gap+(.005 if pair==PAIRS[0] else 0.) for gap,_,pair,_ in before}
    for native in (False,True):
        if native:both.geometry.enable_native_contacts();both.geometry._collision_cache.clear()
        assert both.geometry.collision(both.configuration.data) is None
        after=both.geometry.contacts(both.configuration.data,.02,derivatives=False,per_pair=1)
        for gap,_,pair,_ in after:
            assert gap==pytest.approx(expected[pair],abs=1e-12)
        assert next(h[0] for h in after if h[2]==PAIRS[0])==pytest.approx(.004751946251,abs=1e-12)
    assert both.collision_margin==right.collision_margin==.0005
    assert both.joint_margin==right.joint_margin==.0001
    assert both.geometry.pairs==right.geometry.pairs
    for a,b in zip(both.geometry.radius,right.geometry.radius):np.testing.assert_array_equal(a,b)


def test_bilateral_worker_startup_records_actual_geometry_with_zero_ticks(tmp_path):
    out=tmp_path/'run';out.mkdir()
    m=json.loads((P/'transfer_trial108/run/manifest.json').read_text());m['hl_ik_pair_padding']=metadata()
    (out/'manifest.json').write_text(json.dumps(m))
    (tmp_path/'layout.json').write_bytes((P/'transfer_trial108/layout.json').read_bytes())
    (out/'live_ik_request.json').write_text(json.dumps(dict(active=True,wall=0.)))
    (out/'stop.request').write_text('Offline zero-tick startup: no physics or solve.')
    r=subprocess.run([sys.executable,str(host.HERE/'live_ik.py'),'--out',str(out),
        '--plan',str(P/'ready_plan43'),'--model',str(P/'ik_model.xml'),'--native-collision-contacts'],
        capture_output=True,text=True,timeout=45)
    assert r.returncode==0,r.stderr
    ready=json.loads((out/'live_ik_ready.json').read_text());result=json.loads((out/'live_ik_result.json').read_text())
    for record in (ready,result):
        assert record['collision_geometry']['pair_buffer_overrides']==[
            dict(pair=list(pair),buffer_per_sphere_m=0.,gap_offset_m=.005) for pair in PAIRS]
        assert record['collision_geometry']['buffer_per_sphere_m']==.0025
        assert record['collision_geometry']['pair_count']==340
        assert record['hardware_executed'] is False
    assert result['ticks']==0 and result['error'] is None
