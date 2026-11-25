"""Pure metadata helpers for source-only startup alignment beside a fixed cart.

These functions neither write simulator requests nor change physics state.
The native staging hook must explicitly implement the returned body selector.
"""
import copy
import numpy as np


def _pose(xyyaw):
    value = np.asarray(xyyaw, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError('Finite [x, y, yaw] required')
    x, y, yaw = value
    c, s = np.cos(yaw), np.sin(yaw)
    result = np.eye(4)
    result[:2, :2] = [[c, -s], [s, c]]
    result[:2, 3] = [x, y]
    return result


def alignment_transform(reference_xyyaw, actual_xyyaw):
    """Planar transform from authored source poses to actual startup poses."""
    return _pose(actual_xyyaw) @ np.linalg.inv(_pose(reference_xyyaw))


def staging_contract(manifest, model=None):
    """Select pickup supports/payload; all destination and extra poses stay fixed.

    If provided, a compiled MuJoCo model contributes diagnostic indices. Native
    code should resolve names against its own model instead of hardcoding them.
    """
    room = [body['body'] for body in manifest['room_bodies']]
    selected = list(manifest['source_support_bodies']) + [manifest['payload']['body']]
    destinations = set(manifest['destination_bodies'])
    if len(set(room)) != len(room) or len(set(selected)) != len(selected):
        raise ValueError('Duplicate room or source body')
    if not set(selected) <= set(room) or not destinations <= set(room):
        raise ValueError('Staging body is absent from the room manifest')
    if destinations & set(selected):
        raise ValueError('Fixed destination cannot be included in source alignment')
    result = dict(scope='source_only', transform_bodies=selected,
                  fixed_bodies=[body for body in room if body not in selected],
                  transforms_destination=False, park_before_ready='all_room_free_bodies',
                  ready_pose_for_unselected_bodies='model_qpos0',
                  reset_velocity_for='all_room_free_bodies',
                  static_cart_transform='unchanged',
                  native_selector_required=True)
    if model is not None:
        import mujoco
        indices = []
        for body in room:
            index = model.body(body).id
            joint = int(model.body_jntadr[index])
            if model.body_jntnum[index] != 1 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError('Expected one free joint for room body: ' + body)
            indices.append(dict(body=body, body_id=int(index), joint_id=joint,
                                qpos_address=int(model.jnt_qposadr[joint]),
                                dof_address=int(model.jnt_dofadr[joint]),
                                transform_on_ready=body in selected))
        result['compiled_body_indices'] = indices
    return result


def adapt_goal(goal, reference_xyyaw, actual_xyyaw):
    """Keep fixed-cart world goals; re-express them from actual initial base.

    If A aligns the source and T maps the authored grasp to placement, the
    runtime mapping is T @ inv(A), because the actual source grasp is A @ G.
    World placement and carry-at-goal EE poses are therefore unchanged.
    """
    alignment = alignment_transform(reference_xyyaw, actual_xyyaw)
    actual_to_world = _pose(actual_xyyaw)
    result = copy.deepcopy(goal)
    root_goal = np.r_[goal['robot_goal_xy_world_m'], 0., 1.]
    result['robot_goal_xy_initial_base_m'] = (np.linalg.inv(actual_to_world) @ root_goal)[:2].tolist()
    difference = goal['robot_goal_yaw_world_rad'] - float(actual_xyyaw[2])
    result['robot_goal_yaw_initial_base_rad'] = float(np.arctan2(np.sin(difference), np.cos(difference)))
    result['initial_base_to_world'] = actual_to_world.tolist()
    result['initial_base_frame_convention'] = 'Yaw-only navigation frame at actual initial base XY and floor Z; not the full pelvis pose.'
    result['source_alignment_transform'] = alignment.tolist()
    result['source_to_placement_transform'] = (np.array(goal['source_to_placement_transform']) @ np.linalg.inv(alignment)).tolist()
    result['runtime_frame_rule'] = 'Source-only startup alignment: transform pickup source poses; keep cart, destination, world goalXY/yaw, placement EE and carry-at-goal EE fixed. Re-express navigation goals in the actual initial base frame.'
    return result
