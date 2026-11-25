"""Source alignment must never drag the fixed-cart destination or its goals."""
import copy
import numpy as np
import pytest


def fixture_manifest():
    source = ['room_source_' + str(i) for i in range(4)]
    return dict(source_support_bodies=source, payload={'body': 'room_payload'},
                destination_bodies=['room_destination'],
                room_bodies=[{'body': name} for name in source + ['room_payload', 'room_destination', 'room_extra', 'room_pallet']])


def test_only_source_and_payload_are_selected():
    from rolltainer_staging import staging_contract
    manifest = fixture_manifest()
    before = copy.deepcopy(manifest)
    contract = staging_contract(manifest)
    assert set(contract['transform_bodies']) == set(manifest['source_support_bodies'] + ['room_payload'])
    assert set(contract['fixed_bodies']) == {'room_destination', 'room_extra', 'room_pallet'}
    assert not contract['transforms_destination']
    assert manifest == before
    for wrong in ('room_missing', 'room_destination'):
        broken = copy.deepcopy(manifest)
        broken['source_support_bodies'][0] = wrong
        with pytest.raises(ValueError):
            staging_contract(broken)


def test_shifted_rotated_start_preserves_fixed_goals_and_maps_staged_grasp():
    from rolltainer_staging import adapt_goal
    anchor = np.eye(4)
    anchor[:3, 3] = [.3, -.1, .8]
    placement = np.eye(4)
    placement[:3, 3] = [1., 2., .76]
    source_to_place = placement @ np.linalg.inv(anchor)
    goal = dict(robot_goal_xy_world_m=[1., 2.], robot_goal_yaw_world_rad=.4,
                payload_center_world_m=[1.1, 2.1, .76], placement_ee_world=[placement.tolist()] * 2,
                carry_front_ee_world=[placement.tolist()] * 2,
                source_to_placement_transform=source_to_place.tolist())
    before = copy.deepcopy(goal)
    result = adapt_goal(goal, [0., 0., 0.], [.2, -.3, np.pi / 2])
    np.testing.assert_allclose(result['robot_goal_xy_initial_base_m'], [2.3, -.8], atol=1e-12)
    assert result['robot_goal_yaw_initial_base_rad'] == pytest.approx(.4 - np.pi / 2)
    for key in ('robot_goal_xy_world_m', 'payload_center_world_m', 'placement_ee_world', 'carry_front_ee_world'):
        assert result[key] == goal[key]
    transformed_anchor = np.array(result['source_alignment_transform']) @ anchor
    np.testing.assert_allclose(np.array(result['source_to_placement_transform']) @ transformed_anchor, placement, atol=1e-12)
    assert goal == before


def test_invalid_startup_pose_fails_before_generating_metadata():
    from rolltainer_staging import alignment_transform
    for pose in ([0., 0.], [0., float('nan'), 0.], [0., 0., float('inf')]):
        with pytest.raises(ValueError):
            alignment_transform([0., 0., 0.], pose)
