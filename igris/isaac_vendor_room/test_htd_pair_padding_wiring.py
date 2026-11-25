"""Explicit pair-only SIM padding selection; historical acquisition stays default."""
from pathlib import Path
import copy,inspect,json,subprocess,sys
import numpy as np
import pytest
import transport_trial as host
import live_ik as live
from igris.hl_ik.pipeline import PipelineIK

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'outputs/hlik_transport_20260918_2005'
OVERRIDE={('pelvis','r_upper_leg'):0.0}


def metadata():
    return host.htd_pair_padding_metadata(True,loaded_transport=True,
        current_grasp_reference=True,relaxed_grasp=True)


def manifest():
    return dict(hardware_executed=False,physics_backend='vendor MuJoCo 3.2.7',
        evaluation_profile='relaxed_transport',hl_ik_pair_padding=metadata(),
        htd_loaded_zero=dict(enabled=True,scope='loaded_transport',current_grasp_reference=True))


def test_default_preserves_manifest_bytes_and_no_override(tmp_path):
    assert inspect.signature(host.run).parameters['htd_pelvis_right_thigh_unbuffered'].default is False
    assert host.htd_pair_padding_metadata(False,loaded_transport=False,current_grasp_reference=False,relaxed_grasp=False) is None
    path=tmp_path/'manifest.json';original=b'{"unrelated": true}\n';path.write_bytes(original)
    assert host.configure_htd_pair_padding(tmp_path,None) is None
    assert path.read_bytes()==original and live.htd_pair_buffer_overrides(json.loads(original)) is None


@pytest.mark.parametrize('value',[None,0,1,'true',np.bool_(True)])
def test_option_requires_strict_bool_before_asset_or_output_creation(tmp_path,value):
    out=tmp_path/'never-created'
    with pytest.raises(ValueError,match='pair padding.*bool'):
        host.run(out,tmp_path/'missing-motion',htd_pelvis_right_thigh_unbuffered=value)
    assert not out.exists()


@pytest.mark.parametrize('flag',['loaded_transport','current_grasp_reference','relaxed_grasp'])
def test_enabled_option_rejects_incomplete_or_truthy_scope(flag):
    args=dict(loaded_transport=True,current_grasp_reference=True,relaxed_grasp=True)
    for bad in (False,1):
        args[flag]=bad
        with pytest.raises(ValueError,match='pair padding'):
            host.htd_pair_padding_metadata(True,**args)


def test_host_manifest_roundtrip_matches_worker_exact_pair(tmp_path):
    m=manifest();m.pop('hl_ik_pair_padding');(tmp_path/'manifest.json').write_text(json.dumps(m))
    original={k:v for k,v in m.items()};result=host.configure_htd_pair_padding(tmp_path,metadata())
    written=json.loads((tmp_path/'manifest.json').read_text())
    assert {k:v for k,v in written.items()if k!='hl_ik_pair_padding'}==original
    assert result==written['hl_ik_pair_padding']
    assert live.htd_pair_buffer_overrides(written)==OVERRIDE
    assert written['hl_ik_pair_padding']['physics_modified'] is False
    assert written['hl_ik_pair_padding']['interior_margin_m']==.0005


@pytest.mark.parametrize('bad',['hardware','backend','legacy','not_current','disabled','not_relaxed',
                                'other_pair','negative','global_padding','margin','missing_key','extra_key','bool_buffer','false_metadata'])
def test_worker_rejects_malformed_metadata_before_loading_plan_or_prior(tmp_path,bad):
    m=manifest()
    if bad=='hardware':m['hardware_executed']=True
    if bad=='backend':m['physics_backend']='other'
    if bad=='legacy':m['htd_loaded_zero']['scope']='loaded_goal_hold_only'
    if bad=='not_current':m['htd_loaded_zero']['current_grasp_reference']=False
    if bad=='disabled':m['htd_loaded_zero']['enabled']=False
    if bad=='not_relaxed':m['evaluation_profile']='original_grasp_requirements'
    if bad=='other_pair':m['hl_ik_pair_padding']['pair']=['pelvis','l_upper_leg']
    if bad=='negative':m['hl_ik_pair_padding']['buffer_per_sphere_m']=-.001
    if bad=='global_padding':m['hl_ik_pair_padding']['default_buffer_per_sphere_m']=0.
    if bad=='margin':m['hl_ik_pair_padding']['interior_margin_m']=0.
    if bad=='missing_key':m['hl_ik_pair_padding'].pop('pair')
    if bad=='extra_key':m['hl_ik_pair_padding']['ignore_all_collisions']=True
    if bad=='bool_buffer':m['hl_ik_pair_padding']['buffer_per_sphere_m']=False
    if bad=='false_metadata':m['hl_ik_pair_padding']=False
    (tmp_path/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='pair padding'):
        live.run(tmp_path,tmp_path/'missing-plan',tmp_path/'missing-model')
    assert not (tmp_path/'live_ik_ready.json').exists()


def test_cli_parses_optin_but_refuses_unscoped_execution(tmp_path):
    out=tmp_path/'never-created'
    result=subprocess.run([sys.executable,str(host.HERE/'transport_trial.py'),'--out',str(out),
        '--motion',str(tmp_path/'missing-motion'),'--htd-pelvis-right-thigh-unbuffered'],capture_output=True,text=True)
    assert result.returncode!=0 and 'unrecognized arguments' not in result.stderr
    assert 'pair padding' in result.stderr and not out.exists()


def test_pipeline_override_reaches_replacement_geometry_and_retains_interior_margin():
    row=json.loads((P/'native107_collision_seed01/seed_probe.json').read_text())['failed_tick']
    kwargs=dict(model_path=P/'ik_model.xml',hand_envelope=ROOT/'igris/sim_local/vr_box_eval/results/20260914_synthetic_motion100_1kg/compliance.json',
                acceleration_limit=3.,joint_limit_cost=1000.,soft_joint_margin=.3,collision_braking=True,
                warm_start_primary=False,primary_max_iter=10,primary_max_iter_in=10)
    original=PipelineIK(**kwargs);changed=PipelineIK(**kwargs,pair_buffer_overrides=OVERRIDE)
    for obj in (original,changed):obj.set_q(row['q'])
    assert original.geometry.collision(original.configuration.data)['pair']==['pelvis','r_upper_leg']
    assert changed.geometry.collision(changed.configuration.data) is None
    hits=changed.geometry.contacts(changed.configuration.data,threshold=.01,derivatives=False,per_pair=1)
    selected=next(x for x in hits if x[2]==('pelvis','r_upper_leg'))
    assert selected[0]==pytest.approx(.0033942932856285,abs=1e-10)
    assert changed.collision_margin==original.collision_margin==.0005
    assert changed.joint_margin==original.joint_margin==.0001
    assert changed.geometry.provenance()!=original.geometry.provenance()


@pytest.mark.parametrize('enabled',[False,True])
def test_live_worker_logs_actual_geometry_before_any_control_tick(tmp_path,enabled):
    out=tmp_path/'run';out.mkdir()
    m=json.loads((P/'transfer_trial107/run/manifest.json').read_text())
    if enabled:m['hl_ik_pair_padding']=metadata()
    (out/'manifest.json').write_text(json.dumps(m))
    (tmp_path/'layout.json').write_bytes((P/'transfer_trial107/layout.json').read_bytes())
    (out/'live_ik_request.json').write_text(json.dumps(dict(active=True,wall=0.)))
    (out/'stop.request').write_text('Offline startup audit: exit before any solve/control tick.')
    process=subprocess.run([sys.executable,str(host.HERE/'live_ik.py'),'--out',str(out),
        '--plan',str(P/'ready_plan43'),'--model',str(P/'ik_model.xml'),
        '--native-collision-contacts'],capture_output=True,text=True,timeout=45)
    assert process.returncode==0,process.stderr
    ready=json.loads((out/'live_ik_ready.json').read_text());result=json.loads((out/'live_ik_result.json').read_text())
    expected=[dict(pair=['pelvis','r_upper_leg'],buffer_per_sphere_m=0.,gap_offset_m=.005)]if enabled else []
    for record in (ready,result):
        assert record['collision_geometry']['pair_buffer_overrides']==expected
        assert record['collision_geometry']['buffer_per_sphere_m']==.0025
        assert record['hardware_executed'] is False
    assert result['ticks']==0 and result['error'] is None
