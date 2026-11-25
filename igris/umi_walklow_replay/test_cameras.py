"""Robot cameras of the replay (cameras.py).  The OpenGL part runs only where MuJoCo can open a context."""
import json
import os

import numpy as np
import pytest

from igris.umi_walklow_replay import cameras
from igris.umi_walklow_replay.kinematics import axis_rotation, matrix_quat


def colour(direction):
    """A smooth colour of a viewing direction (N x 3 unit vectors, fisheye axes) in 0..255."""
    return 127.5 + 100. * np.asarray(direction, dtype=float)


def faces_of(size):
    """What each cube face would see of ``colour``: 5 x size x size x 3."""
    centre = (np.arange(size) + .5) / size * 2. - 1.                 # tangent of the pixel centres
    column, row = np.meshgrid(centre, centre)
    out = []
    for forward, right, down in cameras.FACES:
        ray = forward + column[..., None] * right + row[..., None] * down
        out.append(colour(ray / np.linalg.norm(ray, axis=-1, keepdims=True)))
    return np.array(out)


def test_fisheye_is_equidistant_about_the_image_centre():
    mapping = cameras.FisheyeMap(640, 480, focal=188., face=400)
    image = mapping.warp(faces_of(400))
    assert image.shape == (480, 640, 3)
    for angle, around in ((0., 0.), (.5, 0.), (1.2, 2.), (1.5, -1.), (1.1, np.pi / 2), (1.9, .3)):
        # A ray ``angle`` off the axis lands ``focal * angle`` pixels from the centre, whatever its direction around it.
        x = 320. + 188. * angle * np.cos(around); y = 240. + 188. * angle * np.sin(around)
        direction = np.array([np.sin(angle) * np.cos(around), np.sin(angle) * np.sin(around), np.cos(angle)])
        if 0 <= int(y) < 480 and 0 <= int(x) < 640:
            assert np.allclose(image[int(y), int(x)], colour(direction), atol=3.), (angle, around)


def test_fisheye_is_black_outside_its_field_of_view():
    mapping = cameras.FisheyeMap(640, 480, focal=188., face=400, field_deg=180.)
    image = mapping.warp(faces_of(400))
    assert np.all(image[240, 2] == 0) and np.all(image[240, 637] == 0)          # 318 px > 188 * pi / 2 = 295 px
    assert np.any(image[240, 40] != 0) and np.any(image[5, 320] != 0)
    assert image.dtype == np.uint8


def test_fisheye_through_opencv_is_the_same_picture():
    pytest.importorskip('cv2')
    mapping = cameras.FisheyeMap(320, 240, focal=94., face=200, field_deg=190.)
    faces = np.clip(np.rint(faces_of(200)), 0, 255)
    assert np.abs(mapping.warp(faces.astype(np.uint8)).astype(int) - mapping.warp(faces).astype(int)).max() <= 2


def test_every_fisheye_pixel_inside_the_field_reads_one_face_inside_its_bounds():
    mapping = cameras.FisheyeMap(320, 240, focal=94., face=200, field_deg=200.)
    inside = mapping.valid
    assert inside.any() and not inside.all()
    assert mapping.index.min() >= 0 and mapping.index.max() < 5 * 200 * 200
    assert np.allclose(mapping.weight[inside].sum(axis=-1), 1.)


def box_scene(tmp_path):
    """A stand-in for a prepared run: a torso with two wrist-camera bodies on a slide, and a box in the room."""
    xml = '''<mujoco><worldbody>
      <geom name="floor" type="plane" size="5 5 .1"/>
      <body name="pelvis" pos="0 0 1"><freejoint/><geom type="sphere" size=".05"/>
        <body name="torso" pos="0 0 .1"><joint name="waist_yaw" axis="0 0 1"/><geom name="torso_visual" type="box" size=".1 .12 .18" pos="0 0 .18"/>
          <body name="l_wrist_cam" pos=".3 .2 0"><joint name="l" type="slide" axis="1 0 0"/><geom type="box" size=".016 .016 .016"/></body>
          <body name="r_wrist_cam" pos=".3 -.2 0"><geom type="box" size=".016 .016 .016"/></body>
        </body></body>
      <body name="room_Parcel_0" pos=".8 0 1.1"><freejoint/><geom name="room_geom_7" type="box" size=".135 .09 .075"/></body>
    </worldbody></mujoco>'''
    run = tmp_path / 'run'; run.mkdir()
    (run / 'scene.xml').write_text(xml)
    names = ['world', 'pelvis', 'torso', 'l_wrist_cam', 'r_wrist_cam', 'room_Parcel_0']
    (run / 'model.json').write_text(json.dumps(dict(bodies=[dict(name=name) for name in names])))
    return run, names


def test_render_model_adds_the_cameras_to_their_links_and_keeps_the_bodies(tmp_path):
    mujoco = pytest.importorskip('mujoco')
    run, names = box_scene(tmp_path)
    model = mujoco.MjModel.from_xml_string(cameras.render_xml(run / 'scene.xml'))
    for name in names[1:]:
        assert model.body(name).id > 0
    assert model.camera('left_cam').bodyid[0] == model.body('l_wrist_cam').id
    assert model.camera('right_cam').bodyid[0] == model.body('r_wrist_cam').id
    for face in range(5):
        assert model.camera(f'head_cam_{face}').bodyid[0] == model.body('torso').id
        assert np.isclose(model.camera(f'head_cam_{face}').fovy[0], 90.)
    assert model.nq == 0 and model.neq == 0                       # nothing is simulated in it
    watcher = model.camera('overview')                            # for people, fixed in the room, looking at the robot
    assert watcher.bodyid[0] == 0
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    looking = -data.cam_xmat[watcher.id].reshape(3, 3)[:, 2]
    wanted = cameras.OVERVIEW_AT - cameras.OVERVIEW_FROM
    assert np.allclose(looking, wanted / np.linalg.norm(wanted), atol=1e-6)
    assert data.cam_xmat[watcher.id].reshape(3, 3)[2, 1] > 0      # up in the picture is up in the room


def test_prepared_textures_put_the_print_of_each_side_on_a_parcel(tmp_path):
    mujoco = pytest.importorskip('mujoco')
    image = pytest.importorskip('PIL.Image')
    run, names = box_scene(tmp_path)
    folder = tmp_path / 'scene/textures'; folder.mkdir(parents=True)
    shades = dict(parcel_short=(200, 0, 0), parcel_long=(0, 200, 0), parcel_top=(0, 0, 200), parcel_bottom=(90, 90, 90))
    for name, shade in shades.items():
        image.new('RGB', (300, 120), shade).save(folder / f'{name}.png')
    cache = run / 'camera_textures'
    pictures = cameras.square_pictures(folder, cache)
    assert set(pictures) == set(shades) and image.open(pictures['parcel_long']).size == (cameras.PICTURE_SIZE,) * 2
    model = mujoco.MjModel.from_xml_string(cameras.render_xml(run / 'scene.xml', pictures=pictures, cache=cache))
    geom = model.geom('room_geom_7')
    assert geom.type[0] == mujoco.mjtGeom.mjGEOM_MESH and model.material(geom.matid[0]).name == 'parcel_print'
    sheet = np.asarray(image.open(cache / 'parcel_sheet.png'))
    size = cameras.PICTURE_SIZE
    for index, side in enumerate(cameras.PARCEL_SIDES):               # tile k of the sheet is the picture of side k
        assert tuple(sheet[(index // 3) * size + size // 2, (index % 3) * size + size // 2]) == shades[side]


def test_box_mesh_has_each_side_on_its_own_tile_facing_outwards():
    half = np.array([.135, .09, .075])
    rows = [line.split() for line in cameras.box_obj(half).splitlines()]
    corners = np.array([[float(value) for value in row[1:]] for row in rows if row[0] == 'v'])
    places = np.array([[float(value) for value in row[1:]] for row in rows if row[0] == 'vt'])
    assert corners.shape == (24, 3) and places.shape == (24, 2) and sum(row[0] == 'f' for row in rows) == 12
    for index, axis in enumerate((0, 0, 1, 1, 2, 2)):
        side = corners[4 * index:4 * index + 4]; sign = 1. if index % 2 == 0 else -1.
        assert np.allclose(side[:, axis], sign * half[axis])                    # the four corners of one face
        outward = np.cross(side[1] - side[0], side[3] - side[0])
        assert outward[axis] * sign > 0                                          # counter-clockwise seen from outside
        tile = places[4 * index:4 * index + 4]
        assert np.all((tile[:, 0] > (index % 3) / 3.) & (tile[:, 0] < (index % 3 + 1) / 3.))
        assert np.all((tile[:, 1] > (1 - index // 3) / 2.) & (tile[:, 1] < (2 - index // 3) / 2.))
    # Side pictures stand upright: the picture's lower edge is the box's lower edge.
    for index in range(4):
        assert np.allclose(corners[4 * index:4 * index + 2, 2], -half[2])


def test_recorded_body_poses_move_the_geoms_and_the_cameras(tmp_path):
    """The scene is drawn from recorded body poses alone: what hangs on a body goes where the record puts it."""
    mujoco = pytest.importorskip('mujoco')
    run, names = box_scene(tmp_path)
    placed = cameras.Placement(mujoco.MjModel.from_xml_string(cameras.render_xml(run / 'scene.xml')), names)
    rows = np.zeros((len(names), 7)); rows[:, 3] = 1.
    turn = axis_rotation(np.array([0., 0., 1.]), .7)
    rows[names.index('torso')] = [.2, -.1, 1.3, *matrix_quat(turn)]
    rows[names.index('l_wrist_cam')] = [1., 2., 3., 1., 0., 0., 0.]
    rows[names.index('room_Parcel_0')] = [.5, .5, .5, *matrix_quat(axis_rotation(np.array([1., 0., 0.]), .3))]
    data = placed.place(rows)
    model = placed.model
    assert np.allclose(data.geom_xpos[model.geom('room_geom_7').id], [.5, .5, .5])
    assert np.allclose(data.geom_xmat[model.geom('room_geom_7').id].reshape(3, 3), axis_rotation(np.array([1., 0., 0.]), .3))
    # A geom off its body's origin: the torso box sits 0.18 m up the turned torso.
    assert np.allclose(data.geom_xpos[model.geom('torso_visual').id], [.2, -.1, 1.48])
    assert np.allclose(data.cam_xpos[model.camera('left_cam').id], np.array([1., 2., 3.]) + cameras.WRIST_OFFSET)
    head = model.camera('head_cam_0').id
    forward = -data.cam_xmat[head].reshape(3, 3)[:, 2]              # a MuJoCo camera looks along its -z
    tilt = np.radians(cameras.HEAD_TILT_DEG)
    assert np.allclose(forward, turn @ [np.cos(tilt), 0., -np.sin(tilt)], atol=1e-9)
    assert np.allclose(data.cam_xpos[head], np.array([.2, -.1, 1.3]) + turn @ cameras.HEAD_POSITION)


@pytest.mark.skipif(os.environ.get('MUJOCO_GL') not in ('egl', 'osmesa'), reason='needs MUJOCO_GL=egl or osmesa')
def test_cameras_see_the_box_in_front_of_them(tmp_path):
    pytest.importorskip('mujoco')
    run, names = box_scene(tmp_path)
    view = cameras.Cameras(run, width=320, height=240)
    rows = np.zeros((len(names), 7)); rows[:, 3] = 1.
    rows[names.index('pelvis'), :3] = [0., 0., 1.]; rows[names.index('torso'), :3] = [0., 0., 1.1]
    rows[names.index('l_wrist_cam'), :3] = [.3, .2, 1.1]; rows[names.index('r_wrist_cam'), :3] = [.3, -.2, 1.1]
    rows[names.index('room_Parcel_0'), :3] = [.8, 0., 1.1]
    images = view.render(rows)
    assert set(images) == {'head_cam', 'left_cam', 'right_cam'}
    for image in images.values():
        assert image.shape == (240, 320, 3) and image.dtype == np.uint8 and image.std() > 5.
    far = rows.copy(); far[names.index('room_Parcel_0'), :3] = [.8, 3., 1.1]
    moved = view.render(far)
    assert np.abs(images['head_cam'].astype(float) - moved['head_cam']).mean() > .5        # the box left its place in the view
