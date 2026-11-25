"""Explicit evaluated-posture routing; synthetic metadata is not evaluation evidence."""
import copy,json,shutil,sys
from types import SimpleNamespace as NS
import numpy as np
import pytest
import yaml
import htd_candidate_package as candidate
from test_htd_candidate_package import package,write_descriptor
sys.path.insert(0,str(candidate.ROOT/'igris/igris-c-wbc-base/src'))


def repin(directory,descriptor):
    descriptor['artifact_sha256']={name:candidate.sha(directory/name)for name in candidate.EXPECTED}
    write_descriptor(directory,descriptor)


@pytest.fixture
def posture_package(package):
    directory,descriptor=package;profile=directory/'profile';profile.mkdir()
    for name in descriptor['profile_sha256']:shutil.copyfile(candidate.PROFILE/name,profile/name)
    report=json.loads((profile/'report.json').read_text());targets=json.loads((profile/'carry_joint_targets.json').read_text())
    name='Joint_Shoulder_Pitch_Left';targets['joint_targets_rad'][name]+=.02
    (profile/'carry_joint_targets.json').write_text(json.dumps(targets))
    report['proxy_asset']=str(profile/'carry_load_proxy.usda')
    (profile/'report.json').write_text(json.dumps(report))
    spec=json.loads((directory/'deployment.json').read_text())
    from wbc_base.carry_profile import configure_carry_profile
    cfg=NS(phase_gait=True,observation_space={'policy':120,'critic':131},action_space=15,
        robot_cfg=NS(spawn=NS(usd_path=report['source_asset']),init_state=NS(joint_pos={
            **spec['carry_profile']['baseline_policy_joint_offsets'],**spec['carry_profile']['held_joint_targets_rad']})))
    metadata=configure_carry_profile(cfg,profile)
    spec['carry_profile']=metadata;spec['contract']['action_offset_profile_sha256']=metadata['profile_sha256']
    spec['default_joint_pos'][spec['all_joint_names'].index(name)]=float(np.float32(targets['joint_targets_rad'][name]))
    (directory/'deployment.json').write_text(json.dumps(spec))
    training=yaml.load((directory/'runtime_env.yaml').read_text(),Loader=yaml.BaseLoader)
    training['robot_cfg']['init_state']['joint_pos'][name]=str(spec['default_joint_pos'][spec['all_joint_names'].index(name)])
    (directory/'runtime_env.yaml').write_text(yaml.safe_dump(training))
    descriptor.update(schema='htd_loaded_candidate/v2',profile_directory=str(profile),
        profile_sha256=metadata['profile_sha256'])
    repin(directory,descriptor)
    return directory,descriptor


def test_posture_selection_pins_new_profile_and_validates_real_actor(posture_package):
    directory,descriptor=posture_package;selection=candidate.resolve_package(directory)
    assert selection['selection']=='evaluated_posture_continuation'
    assert selection['profile_directory']==str(directory/'profile')
    assert selection['profile_sha256']==descriptor['profile_sha256']
    assert selection['profile_handoff_max_error_rad']==.1


@pytest.mark.parametrize('change',['offset','gain','cap','delay','dt','order','controller','source','mass','inertia','default'])
def test_posture_cannot_repin_numeric_or_unaudited_profile_drift(posture_package,change):
    directory,descriptor=posture_package;spec=json.loads((directory/'deployment.json').read_text())
    if change=='offset':spec['carry_profile']['policy_joint_offsets']['Joint_Waist_Yaw']+=.001
    elif change=='order':spec['policy_joint_names'].reverse()
    elif change=='controller':spec['world_xy_controller']['position_gain']+=.1
    elif change=='default':spec['default_joint_pos'][0]+=.001
    elif change in ('gain','cap','delay','dt'):
        path=directory/'runtime_env.yaml';text=path.read_text()
        old,new={'gain':('122.136354','123.136354'),'cap':('217.0253','216.0253'),
                 'delay':("max_delay: '2'","max_delay: '1'"),'dt':("dt: '0.005'","dt: '0.006'")}[change]
        assert old in text;text=text.replace(old,new,1);path.write_text(text)
    else:
        report_path=directory/'profile/report.json';report=json.loads(report_path.read_text())
        if change=='source':report['source_asset']=str(directory/'policy.ts')
        if change=='mass':report['payload']['mass_kg']=.01
        if change=='inertia':report['mass_property_changes'][0]['new_inertia_at_com_body_kg_m2'][0][0]=-1
        report_path.write_text(json.dumps(report))
        descriptor['profile_sha256']['report.json']=candidate.sha(report_path)
        spec['carry_profile']['profile_sha256']=descriptor['profile_sha256'].copy()
        spec['contract']['action_offset_profile_sha256']=descriptor['profile_sha256'].copy()
    (directory/'deployment.json').write_text(json.dumps(spec));repin(directory,descriptor)
    with pytest.raises(ValueError):candidate.resolve_package(directory)


def test_posture_descriptor_cannot_be_used_as_legacy_candidate(posture_package):
    directory,descriptor=posture_package;descriptor['schema']='htd_loaded_candidate/v1';write_descriptor(directory,descriptor)
    with pytest.raises(ValueError,match='profile76'):candidate.resolve_package(directory)


def test_worker_revalidates_posture_scope_and_profile_tamper(posture_package):
    directory,_=posture_package;selection=candidate.resolve_package(directory)
    scope=dict(policy_selection=selection,current_grasp_reference=True,scope='loaded_goal_hold_only',
        goal_hold={},command_observer={},wrist_mode_entry={},historically_trained_on=selection['historically_trained_on'])
    assert candidate.validate_runtime_selection(scope)==selection
    bad=copy.deepcopy(scope);bad['current_grasp_reference']=False
    with pytest.raises(ValueError):candidate.validate_runtime_selection(bad)
    (directory/'profile/carry_load_proxy.usda').write_text('changed')
    with pytest.raises(ValueError,match='hash'):candidate.validate_runtime_selection(scope)


@pytest.mark.parametrize('delta,allowed',[(.099,True),(.101,False)])
def test_posture_handoff_requires_current_held8_near_selected_profile(delta,allowed):
    hold=np.zeros(31);upper=hold.copy();upper[17]=delta
    selection={'selection':'evaluated_posture_continuation','profile_handoff_max_error_rad':.1}
    if allowed:
        result=candidate.check_posture_handoff(selection,hold,upper)
        assert result['max_abs_error_rad']==pytest.approx(delta)
    else:
        with pytest.raises(ValueError,match='posture'):candidate.check_posture_handoff(selection,hold,upper)
    assert candidate.check_posture_handoff({'selection':'evaluated_continuation'},hold,upper) is None


def test_setup_and_worker_route_selected_profile_before_any_service(posture_package,fresh_trial):
    import htd_loaded_setup as setup
    import htd_loaded_runtime as runtime
    from test_htd_loaded_runtime import Node,Service
    directory,_=posture_package
    setup.prepare(fresh_trial,enabled=True,current_grasp_reference=True,command_observer=True,
                  wrist_mode_entry=True,goal_hold=True,policy_package=directory)
    scope=json.loads((fresh_trial/'manifest.json').read_text())['htd_loaded_zero']
    assert scope['profile_directory']==str(directory/'profile')
    assert scope['native_hold_source']==str(directory/'profile/carry_joint_targets.json')
    assert scope['profile_handoff_max_error_rad']==.1
    (fresh_trial/'runtime.json').write_text(json.dumps(dict(network='none',ipc='private',
        hardware_executed=False,simulation_rate_limit=.5)))
    node=Node();obj=runtime.LoadedRuntime(fresh_trial,node,services={'mode':Service,'sim':Service},spin=lambda:None)
    assert obj.preflight['policy_selection']['profile_directory']==str(directory/'profile')
    assert node.requests==[]


def test_posture_mismatch_rejected_before_coordinator_or_low_service():
    from htd_loaded_runtime import LoadedRuntime
    obj=LoadedRuntime.__new__(LoadedRuntime)
    obj.policy_selection={'selection':'evaluated_posture_continuation','profile_handoff_max_error_rad':.1}
    obj.events=[];obj.entry_ready=lambda s:True
    def forbidden(s):raise AssertionError('Handoff must not begin for an out-of-profile posture')
    obj.coordinator=NS(hold_q=np.zeros(31),request_handoff=forbidden)
    upper=np.zeros(31);upper[15]=.101
    with pytest.raises(ValueError,match='posture'):
        obj.request_low(NS(upper_output_target=upper,native_time=100.))
    assert obj.events[-1]['event']=='posture_handoff_rejected'


from test_htd_loaded_setup import fresh_trial


@pytest.mark.parametrize('change',['wrist','proxy_mass','proxy_mass_nan','proxy_geometry','profile_path'])
def test_repin_cannot_hide_unreviewed_wrist_or_proxy_edits(posture_package,change,tmp_path):
    directory,descriptor=posture_package;profile=directory/'profile'
    spec=json.loads((directory/'deployment.json').read_text())
    if change=='profile_path':descriptor['profile_directory']=str(tmp_path)
    else:
        report=json.loads((profile/'report.json').read_text())
        if change=='wrist':
            path=profile/'carry_joint_targets.json';hold=json.loads(path.read_text())
            hold['native_wrist_command_rad']['left'][0]=100.;path.write_text(json.dumps(hold))
        else:
            path=profile/'carry_load_proxy.usda';text=path.read_text()
            if change.startswith('proxy_mass'):
                value='nan' if change=='proxy_mass_nan' else '4.3551102'
                text=text.replace('float physics:mass = 3.3551102','float physics:mass = '+value,1)
            else:text+='\ndef Xform "ExtraRobotBody" {}\n'
            path.write_text(text);report['proxy_asset_sha256']=candidate.sha(path)
            (profile/'report.json').write_text(json.dumps(report))
        hashes={name:candidate.sha(profile/name)for name in descriptor['profile_sha256']}
        descriptor['profile_sha256']=hashes;spec['carry_profile']['profile_sha256']=hashes
        spec['carry_profile']['verified_input_sha256'][str(profile/'carry_load_proxy.usda')]=report['proxy_asset_sha256']
        spec['contract']['action_offset_profile_sha256']=hashes
        (directory/'deployment.json').write_text(json.dumps(spec))
    repin(directory,descriptor)
    with pytest.raises(ValueError):candidate.resolve_package(directory)
