"""Scene geometry and physics structure.  Needs pxr (env_isaaclab) and the UR5e desk scene."""
import json

import numpy as np
import pytest

pxr = pytest.importorskip('pxr')
from pxr import Usd, UsdGeom, UsdPhysics   # noqa: E402

from igris.umi_walklow_replay import scene   # noqa: E402

REPORT = dict(pick_xy=[.444, -.303], place_xy=[.142, .415], pick_axis_yaw_rad=.878, place_axis_yaw_rad=2.914,
              pick_axis_yaws_rad=[.816, .907, .917])


def test_layout_applies_the_robot_shift():
    layout = scene.layout_from_episode(REPORT, shift=(.07, -.03))
    assert np.allclose(layout['parcels'][0]['xy'], [.374, -.273])
    assert np.allclose(layout['place']['xy'], [.072, .445])
    assert [p['bottom'] for p in layout['parcels']] == pytest.approx([scene.CASE_TOP + k * .15 for k in range(3)])
    # The top box is taken first, so it stands at the first carry's heading.
    assert layout['parcels'][2]['yaw'] == .816 and layout['parcels'][0]['yaw'] == .917
    json.dumps(layout)
    taller = scene.layout_from_episode(REPORT, carton_height=.225)
    assert taller['surface_top_m'] == pytest.approx(.172 + 3 * .225) and taller['case']['size'][2] == taller['surface_top_m']


def test_measured_surface_height_and_container_offset():
    """Measured on the real setup: stacked cartons 0.80 m, black case 0.77 m above the floor."""
    layout = scene.layout_from_episode(REPORT)
    assert layout['surface_top_m'] == pytest.approx(.80)
    assert layout['case_top_m'] == pytest.approx(.77) and layout['case']['size'][2] == pytest.approx(.77)
    assert layout['parcels'][0]['bottom'] == pytest.approx(.77)
    assert scene.layout_from_episode(REPORT, case_top=.80)['parcels'][0]['bottom'] == pytest.approx(.80)
    assert scene.DECK_TOP + scene.FRONT_LEVELS * layout['carton_size_m'][2] == pytest.approx(.80)
    # The offset moves the container, not the place where the boxes are set down.
    moved = scene.layout_from_episode(REPORT, cart_setback=(0., 0.))
    assert moved['place'] == layout['place'] and moved['parcels'] == layout['parcels']
    delta = scene.rotation(layout['cart']['yaw']).T @ (np.array(layout['cart']['xy']) - np.array(moved['cart']['xy']))
    assert delta == pytest.approx([-scene.CART_SETBACK[0], scene.CART_SETBACK[1]])


@pytest.fixture(scope='module')
def stage(tmp_path_factory):
    try:
        scene.room_usd()
    except FileNotFoundError:
        pytest.skip('UR5e desk scene is not available on this machine')
    out = tmp_path_factory.mktemp('scene') / 'built'
    path = scene.build(out, scene.layout_from_episode(REPORT))
    return Usd.Stage.Open(str(path))


def test_stage_units_and_physical_bodies(stage):
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1 and UsdGeom.GetStageUpAxis(stage) == 'Z'
    bodies = sorted(str(p.GetPath()) for p in stage.Traverse()
                    if p.HasAPI(UsdPhysics.RigidBodyAPI) and not str(p.GetPath()).startswith('/World/IGRIS_C'))
    assert bodies == ['/World/Case', '/World/Parcel_0', '/World/Parcel_1', '/World/Parcel_2', '/World/Rolltainer']
    for path in bodies:
        assert UsdPhysics.MassAPI(stage.GetPrimAtPath(path)).GetMassAttr().Get() > 0
    colliders = [p for p in stage.Traverse()
                 if p.HasAPI(UsdPhysics.CollisionAPI) and not str(p.GetPath()).startswith('/World/IGRIS_C')]
    assert colliders and all(p.IsA(UsdGeom.Cube) for p in colliders)        # what the vendor composition accepts
    assert not any(str(p.GetPath()).startswith('/World/Backdrop') for p in colliders)


def test_parcel_and_surface_dimensions(stage):
    cache = UsdGeom.XformCache()

    def box(path):
        prim = stage.GetPrimAtPath(path)
        matrix = np.array(cache.GetLocalToWorldTransform(prim)).T
        size = np.array(prim.GetAttribute('xformOp:scale').Get())
        return matrix[:3, 3], size

    for level in range(3):
        centre, size = box(f'/World/Parcel_{level}/Collider')
        assert np.allclose(size, [.27, .18, .15])
        assert centre[2] == pytest.approx(scene.CASE_TOP + level * .15 + .075, abs=1e-6)
    centre, size = box('/World/Case/Collider')
    assert centre[2] + size[2] / 2 == pytest.approx(scene.CASE_TOP, abs=1e-3)
    centre, size = box('/World/Rolltainer/Front_Collider')
    assert centre[2] + size[2] / 2 == pytest.approx(scene.SURFACE_TOP, abs=1e-3)
    # Roll container: 0.92 m wide outside (measured), the side panels at its edges.
    _, size = box('/World/Rolltainer/Base_Collider')
    assert size[0] == pytest.approx(.92)
    left, thick = box('/World/Rolltainer/Side0_Collider'); right, _ = box('/World/Rolltainer/Side1_Collider')
    assert np.linalg.norm(right[:2] - left[:2]) + thick[0] == pytest.approx(.92, abs=1e-3)


def test_robot_model_and_hand_pose_present(stage):
    robot = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI) and str(p.GetPath()).startswith('/World/IGRIS_C')]
    assert len(robot) == 143
    assert stage.GetPrimAtPath('/World/Cameras/UmiOverview').IsA(UsdGeom.Camera)
