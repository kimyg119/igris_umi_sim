"""Candidate acceptance requires exactly the declared proxy-only filter pair set."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from pxr import Sdf, Usd, UsdPhysics
import htd_candidate_package as candidate
from test_htd_posture_candidate import posture_package,repin
from test_htd_candidate_package import package
from wbc_base.carry_profile import configure_carry_profile,fixed_wrist_contact_filter_spec


def refresh_hashes(directory,descriptor):
    profile=directory/'profile';report=json.loads((profile/'report.json').read_text())
    report['proxy_asset_sha256']=candidate.sha(profile/'carry_load_proxy.usda')
    (profile/'report.json').write_text(json.dumps(report))
    hashes={name:candidate.sha(profile/name)for name in descriptor['profile_sha256']}
    descriptor['profile_sha256']=hashes
    spec=json.loads((directory/'deployment.json').read_text())
    spec['carry_profile']['profile_sha256']=hashes
    spec['carry_profile']['verified_input_sha256'][str(profile/'carry_load_proxy.usda')]=report['proxy_asset_sha256']
    spec['contract']['action_offset_profile_sha256']=hashes
    (directory/'deployment.json').write_text(json.dumps(spec));repin(directory,descriptor)


@pytest.fixture
def filtered_posture_package(posture_package):
    directory,descriptor=posture_package;profile=directory/'profile';contract=fixed_wrist_contact_filter_spec()
    stage=Usd.Stage.Open(str(profile/'carry_load_proxy.usda'))
    UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(contract['body_path'])).CreateFilteredPairsRel().SetTargets(contract['targets'])
    stage.GetRootLayer().Save()
    report=json.loads((profile/'report.json').read_text())
    report.update(fixed_wrist_contact_filter=contract,proxy_asset_sha256=candidate.sha(profile/'carry_load_proxy.usda'))
    (profile/'report.json').write_text(json.dumps(report))
    spec=json.loads((directory/'deployment.json').read_text());old=spec['carry_profile']
    cfg=NS(phase_gait=True,observation_space={'policy':120,'critic':131},action_space=15,
        robot_cfg=NS(spawn=NS(usd_path=report['source_asset']),init_state=NS(joint_pos={
            **old['baseline_policy_joint_offsets'],**old['held_joint_targets_rad']})))
    spec['carry_profile']=configure_carry_profile(cfg,profile)
    (directory/'deployment.json').write_text(json.dumps(spec))
    descriptor['fixed_wrist_contact_filter']=contract
    refresh_hashes(directory,descriptor)
    return directory,descriptor


def test_exact_filter_requires_explicit_descriptor_and_roundtrips_worker_selection(filtered_posture_package):
    directory,descriptor=filtered_posture_package
    selection=candidate.resolve_package(directory)
    assert selection['fixed_wrist_contact_filter']==descriptor['fixed_wrist_contact_filter']
    scope=dict(policy_selection=selection,current_grasp_reference=True,scope='loaded_goal_hold_only',
        goal_hold={},command_observer={},wrist_mode_entry={},historically_trained_on=selection['historically_trained_on'])
    assert candidate.validate_runtime_selection(scope)==selection


@pytest.mark.parametrize('change',['descriptor_missing','descriptor_wrong','extra_target','other_body_api',
    'global_self_collision','disabled_collider','extra_api','wrong_target_order','target_list_op','relationship_metadata'])
def test_repin_cannot_expand_filtered_proxy_scope(filtered_posture_package,change):
    directory,descriptor=filtered_posture_package
    if change=='descriptor_missing':descriptor.pop('fixed_wrist_contact_filter')
    elif change=='descriptor_wrong':descriptor['fixed_wrist_contact_filter']['scope']='native'
    else:
        contract=fixed_wrist_contact_filter_spec()
        layer=Sdf.Layer.FindOrOpen(str(directory/'profile/carry_load_proxy.usda'))
        prim=layer.GetPrimAtPath(contract['body_path'])
        rel=layer.GetRelationshipAtPath(contract['body_path']+'.physics:filteredPairs')
        if change=='extra_target':rel.targetPathList.explicitItems=contract['targets']+['/IGRIS_C_custom_hand_torso/Link_Pelvis']
        elif change=='wrong_target_order':rel.targetPathList.explicitItems=contract['targets'][::-1]
        elif change=='target_list_op':rel.targetPathList.ClearEdits();rel.targetPathList.prependedItems=contract['targets']
        elif change=='relationship_metadata':rel.custom=True
        elif change=='extra_api':prim.SetInfo('apiSchemas',Sdf.TokenListOp.Create(prependedItems=['PhysicsMassAPI','PhysicsFilteredPairsAPI','PhysicsCollisionAPI']))
        elif change=='other_body_api':
            other=layer.GetPrimAtPath(contract['targets'][0]);other.SetInfo('apiSchemas',Sdf.TokenListOp.Create(prependedItems=['PhysicsMassAPI','PhysicsFilteredPairsAPI']))
        elif change=='global_self_collision':
            root=layer.GetPrimAtPath('/IGRIS_C_custom_hand_torso')
            Sdf.AttributeSpec(root,'physxArticulation:enabledSelfCollisions',Sdf.ValueTypeNames.Bool).default=False
        elif change=='disabled_collider':
            collider=Sdf.CreatePrimInLayer(layer,contract['body_path']+'/collisions/cube')
            Sdf.AttributeSpec(collider,'physics:collisionEnabled',Sdf.ValueTypeNames.Bool).default=False
        layer.Save()
    refresh_hashes(directory,descriptor)
    with pytest.raises(ValueError):candidate.resolve_package(directory)


def test_unfiltered_posture_cannot_declare_a_filter_without_authored_audit(posture_package):
    directory,descriptor=posture_package;descriptor['fixed_wrist_contact_filter']=fixed_wrist_contact_filter_spec()
    repin(directory,descriptor)
    with pytest.raises(ValueError,match='filter|Filter'):candidate.resolve_package(directory)


def test_legacy_candidate_cannot_claim_filter_variant(package):
    directory,descriptor=package;descriptor['fixed_wrist_contact_filter']=fixed_wrist_contact_filter_spec()
    repin(directory,descriptor)
    with pytest.raises(ValueError,match='filter|Filter'):candidate.resolve_package(directory)
