"""The robot's own cameras, drawn with MuJoCo from the body poses the vendor simulator recorded.

Three views, the ones the UMI recordings have (``observation.images.*``):

  head_cam    a fisheye on the middle of the chest (torso link), in place of the operator's head camera:
              equidistant projection, the image circle a little wider than the picture is high, as in the recordings
  left_cam    the camera of the URDF link ``l_wrist_cam`` (igris_c_description/urdf/igris_c_rcv_hand_pelvis.urdf)
  right_cam   the same on ``r_wrist_cam``

Nothing is simulated here.  The scene file of a prepared run is stripped down to what can be seen, the cameras
are hung on their links, and every body is put where the simulator's observer wrote it (``poses.bin`` for a
finished run, ``latest.bin`` while it runs).  MuJoCo has no fisheye camera: the chest view is five 90 degree
views (ahead, left, right, up, down) resampled into one equidistant image (``FisheyeMap``).

Rendering needs an OpenGL context: ``MUJOCO_GL=egl`` on a GPU, ``osmesa`` without one.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

# Chest camera, in the torso frame (x ahead, y left, z up; the torso mesh reaches x = 0.106 m, z = 0.37 m and the
# shoulder joints sit at z = 0.266 m): on the front of the chest between the shoulders, looking ahead and down.
HEAD_POSITION = np.array([.115, 0., .21])
HEAD_TILT_DEG = 25.
HEAD_CIRCLE = .92                 # width of the 180 degree circle as a share of the image width (the recordings: about 0.92)
HEAD_FIELD_DEG = 190.
# Wrist cameras, in the frame of the URDF camera link.  The housing is a 32 x 32 mm block from z = -6.6 to
# +26.3 mm of that frame: the lens looks along +z from its front face.  ``WRIST_UP`` is the link axis that is up
# in the picture.
WRIST_OFFSET = np.array([0., 0., .027])
WRIST_UP = np.array([0., -1., 0.])
WRIST_FOVY_DEG = 75.              # vertical; 91 degrees across at 4:3
# Cube faces of the chest view as (ahead, right, down) in fisheye axes (x right, y down, z ahead).
FACES = [tuple(np.array(axis, dtype=float) for axis in face) for face in (
    ((0, 0, 1), (1, 0, 0), (0, 1, 0)),          # ahead
    ((1, 0, 0), (0, 0, -1), (0, 1, 0)),         # right
    ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),         # left
    ((0, -1, 0), (1, 0, 0), (0, 0, 1)),         # up
    ((0, 1, 0), (1, 0, 0), (0, 0, -1)))]        # down
NAMES = ('head_cam', 'left_cam', 'right_cam')
# A view from outside for people to watch a run with (not an input of any policy): from in front of the robot and
# high above, looking down into the roll container, in the frame the scene is staged in (the robot stands near
# the origin facing +x, the case to its right, the roll container to its left; from lower down its panels hide the robot).
OVERVIEW_FROM = np.array([1.9, -.25, 2.8])
OVERVIEW_AT = np.array([.25, .02, .95])
# What the scene bodies look like (the physics model has no colours).  Without the textures of a prepared scene:
# cube textures with drawn edges, so that a box shows its outline against another box of the same colour.
LOOKS = {
    'room_Parcel': dict(rgb=(.80, .66, .47), mark=(.45, .34, .22)),
    'room_Case': dict(rgb=(.07, .07, .08), mark=(.45, .45, .48)),
    'cart_load': dict(rgb=(.70, .55, .37), mark=(.40, .30, .20)),
    'cart_frame': dict(rgb=(.62, .66, .72), mark=(.30, .32, .36)),
}
# With them (``<output>/scene/textures``, textures.py): one picture all over a body, and on a parcel the print
# of each side.  MuJoCo wraps a cube texture around a box by direction from its centre, which smears the
# pictures over the edges of a box that is not a cube; a parcel is therefore drawn as a mesh with its own
# texture coordinates, the six pictures side by side in one image (``parcel_mesh``).
PICTURES = {'room_Case': 'case_laminate', 'cart_load': 'kraft', 'cart_frame': 'zinc'}
PARCEL_SIDES = ('parcel_short', 'parcel_short', 'parcel_long', 'parcel_long', 'parcel_top', 'parcel_bottom')   # +x -x +y -y +z -z
FLOOR_RGB = (.46, .47, .46)
PICTURE_SIZE = 256


def square_pictures(textures, cache):
    """Square copies of the scene textures (MuJoCo wants the six sides of a cube texture square and alike).
    Returns name -> file for the pictures found; nothing when the folder or Pillow is missing."""
    textures, cache = Path(textures), Path(cache)
    if not textures.is_dir():
        return {}
    try:
        from PIL import Image
    except ImportError:
        return {}
    out = {}
    for name in sorted({*PICTURES.values(), *PARCEL_SIDES}):
        source = textures / f'{name}.png'
        if not source.is_file():
            continue
        target = cache / f'{name}.png'
        if not target.is_file():
            cache.mkdir(parents=True, exist_ok=True)
            Image.open(source).convert('RGB').resize((PICTURE_SIZE, PICTURE_SIZE), Image.BICUBIC).save(target)
        out[name] = target
    return out


# Corners of each box side, counter-clockwise seen from outside, starting at the lower left of its picture:
# signs of (x, y, z).  Side pictures stand upright (+z up); the top reads from the -y side.
BOX_SIDES = (
    ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)),             # +x
    ((-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1)),         # -x
    ((1, 1, -1), (-1, 1, -1), (-1, 1, 1), (1, 1, 1)),             # +y
    ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)),         # -y
    ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)),             # +z
    ((-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)))         # -z


def box_obj(half):
    """Wavefront text of a box with half sizes ``half``: four corners of its own per side, and texture
    coordinates that put side ``k`` on tile ``k`` of a 3 x 2 sheet (tiles left to right, top row first)."""
    lines = []; inset = .5 / PICTURE_SIZE
    for index, corners in enumerate(BOX_SIDES):
        column, row = index % 3, index // 3
        normal = np.cross(np.subtract(corners[1], corners[0]), np.subtract(corners[3], corners[0])) / 4.
        for corner, (u, v) in zip(corners, ((inset, inset), (1 - inset, inset), (1 - inset, 1 - inset), (inset, 1 - inset))):
            lines.append('v ' + _numbers(np.multiply(corner, half)))
            lines.append('vt ' + _numbers([(column + u) / 3., (1 - row + v) / 2.]))
        lines.append('vn ' + _numbers(normal))
    for index in range(6):
        a = 4 * index + 1
        for triangle in ((a, a + 1, a + 2), (a, a + 2, a + 3)):
            lines.append('f ' + ' '.join(f'{vertex}/{vertex}/{index + 1}' for vertex in triangle))
    return '\n'.join(lines) + '\n'


def parcel_mesh(pictures, cache, half):
    """The mesh and the picture sheet of a parcel (files in ``cache``); None when a side picture is missing."""
    if any(side not in pictures for side in PARCEL_SIDES):
        return None
    from PIL import Image
    cache = Path(cache)
    sheet = cache / 'parcel_sheet.png'; mesh = cache / 'parcel_box.obj'
    if not sheet.is_file():
        image = Image.new('RGB', (3 * PICTURE_SIZE, 2 * PICTURE_SIZE))
        for index, side in enumerate(PARCEL_SIDES):
            image.paste(Image.open(pictures[side]), ((index % 3) * PICTURE_SIZE, (index // 3) * PICTURE_SIZE))
        image.save(sheet)
    mesh.write_text(box_obj(half))
    return dict(mesh=mesh, sheet=sheet)


class FisheyeMap:
    """Equidistant fisheye image from five cube faces: a ray ``angle`` off the axis lands ``focal * angle`` pixels
    from the image centre.  Pixels further off the axis than half of ``field_deg`` are black."""

    def __init__(self, width, height, focal, face, field_deg=180.):
        self.width, self.height, self.face = int(width), int(height), int(face)
        x, y = np.meshgrid(np.arange(self.width) + .5 - self.width / 2., np.arange(self.height) + .5 - self.height / 2.)
        radius = np.hypot(x, y)
        angle = radius / float(focal)
        scale = np.where(radius > 0, np.sin(angle) / np.maximum(radius, 1e-12), 0.)
        ray = np.stack([scale * x, scale * y, np.cos(angle)], axis=-1)                    # H x W x 3, unit
        ahead = np.stack([ray @ face_axes[0] for face_axes in FACES], axis=-1)            # share of the ray along each face
        chosen = np.argmax(ahead, axis=-1)
        depth = np.take_along_axis(ahead, chosen[..., None], axis=-1)[..., 0]
        right = np.array([face_axes[1] for face_axes in FACES])[chosen]
        down = np.array([face_axes[2] for face_axes in FACES])[chosen]
        safe = np.maximum(depth, 1e-9)
        across = np.sum(ray * right, axis=-1) / safe; along = np.sum(ray * down, axis=-1) / safe
        self.valid = (angle <= np.radians(field_deg) / 2.) & (depth > 0) & (np.abs(across) <= 1.) & (np.abs(along) <= 1.)
        column = np.clip((across + 1.) / 2. * self.face - .5, 0., self.face - 1.)
        row = np.clip((along + 1.) / 2. * self.face - .5, 0., self.face - 1.)
        column0 = np.minimum(np.floor(column).astype(int), self.face - 2); row0 = np.minimum(np.floor(row).astype(int), self.face - 2)
        share_x = column - column0; share_y = row - row0
        base = chosen * self.face * self.face
        self.index = np.stack([base + row0 * self.face + column0, base + row0 * self.face + column0 + 1,
                               base + (row0 + 1) * self.face + column0, base + (row0 + 1) * self.face + column0 + 1], axis=-1)
        self.weight = np.stack([(1 - share_x) * (1 - share_y), share_x * (1 - share_y),
                                (1 - share_x) * share_y, share_x * share_y], axis=-1).astype(np.float32)
        self.index[~self.valid] = 0

        # The same reading as one lookup in the five faces laid side by side (for OpenCV's remap).
        self.atlas_x = (chosen * self.face + column).astype(np.float32); self.atlas_y = row.astype(np.float32)

    def warp(self, faces):
        """``faces``: 5 x face x face x 3 in the order of ``FACES``.  Returns the height x width x 3 uint8 image."""
        faces = np.asarray(faces)
        flat = faces.reshape(-1, 3)
        if len(flat) != 5 * self.face * self.face:
            raise ValueError('Expected five square cube faces of the size this map was made for')
        if faces.dtype == np.uint8:
            try:
                import cv2
            except ImportError:
                cv2 = None
            if cv2 is not None:
                # A column never passes the last pixel of its face, so no pixel of the next face is mixed in.
                atlas = np.ascontiguousarray(faces.reshape(5, self.face, self.face, 3).transpose(1, 0, 2, 3).reshape(self.face, 5 * self.face, 3))
                out = cv2.remap(atlas, self.atlas_x, self.atlas_y, cv2.INTER_LINEAR)
                out[~self.valid] = 0
                return out
        out = np.zeros((self.height, self.width, 3), dtype=np.float32)
        for tap in range(4):
            out += flat[self.index[..., tap]] * self.weight[..., tap, None]
        out[~self.valid] = 0.
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def head_axes(tilt_deg=HEAD_TILT_DEG):
    """Fisheye axes (columns: right, down, ahead) in the torso frame."""
    tilt = np.radians(tilt_deg)
    ahead = np.array([np.cos(tilt), 0., -np.sin(tilt)]); right = np.array([0., -1., 0.])
    return np.column_stack([right, np.cross(ahead, right), ahead])


def _numbers(values):
    return ' '.join(f'{value:.9g}' for value in np.asarray(values, dtype=float).ravel())


def render_xml(scene_xml, wrist_fovy=WRIST_FOVY_DEG, buffer=(960, 1280), pictures=None, cache=None):
    """MJCF for drawing a prepared run: the bodies and what is seen of them, the three cameras, nothing to simulate.
    ``pictures`` (name -> square image file, ``square_pictures``) puts the prepared scene's textures on the bodies;
    with a ``cache`` folder for the mesh the parcels get the print of each side."""
    pictures = pictures or {}
    root = ET.parse(scene_xml).getroot()
    for tag in ('contact', 'equality', 'actuator', 'sensor', 'keyframe', 'tendon'):
        for element in root.findall(tag):
            root.remove(element)
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag in ('joint', 'freejoint', 'site', 'inertial', 'light'):
                parent.remove(child)
            elif child.tag == 'geom' and (child.get('class') == 'collision' or child.get('group') in ('2', '3', '4', '5')):
                parent.remove(child)                              # collision shapes and contact pads are not seen
    asset = root.find('asset')
    if asset is None:
        asset = ET.SubElement(root, 'asset')
    for material in asset.findall('material'):
        if material.get('rgba'):                                  # the vendor's dark material is 70 % opaque
            material.set('rgba', ' '.join(material.get('rgba').split()[:3] + ['1']))
        material.attrib.pop('reflectance', None)                  # a mirror floor costs a second pass per view
    for name, look in LOOKS.items():
        if PICTURES.get(name) in pictures:
            ET.SubElement(asset, 'texture', name=f'look_{name}', type='cube', file=str(pictures[PICTURES[name]]))
        else:
            ET.SubElement(asset, 'texture', name=f'look_{name}', type='cube', builtin='flat', mark='edge', width='128', height='128',
                          rgb1=_numbers(look['rgb']), rgb2=_numbers(look['rgb']), markrgb=_numbers(look['mark']))
        ET.SubElement(asset, 'material', name=f'look_{name}', texture=f'look_{name}', specular='.05', shininess='.05')
    world = root.find('worldbody')
    for geom in world.findall('geom'):
        if geom.get('type') == 'plane':
            geom.attrib.pop('material', None); geom.set('rgba', _numbers([*FLOOR_RGB, 1.]))
    printed = None                                                # mesh and sheet of the printed parcel, once a parcel is met
    for body in root.iter('body'):
        name = body.get('name', '')
        if not name.startswith('room_'):
            continue
        for order, geom in enumerate(body.findall('geom')):
            if name == 'room_Rolltainer':
                # Geoms in the order scene.py writes them: deck, the cartons it is loaded with (two blocks), three panels.
                look = 'cart_frame' if order == 0 or order >= 3 else 'cart_load'
            else:
                look = next((key for key in LOOKS if name.startswith(key)), None)
            if look is None:
                continue
            geom.set('material', f'look_{look}'); geom.attrib.pop('rgba', None)
            if look == 'room_Parcel' and cache is not None and geom.get('type') == 'box':
                if printed is None:
                    printed = parcel_mesh(pictures, cache, np.fromstring(geom.get('size'), sep=' ')) or {}
                    if printed:
                        ET.SubElement(asset, 'mesh', name='parcel_box', file=str(printed['mesh']))
                        ET.SubElement(asset, 'texture', name='parcel_sheet', type='2d', file=str(printed['sheet']))
                        ET.SubElement(asset, 'material', name='parcel_print', texture='parcel_sheet', specular='.05', shininess='.05')
                if printed:
                    geom.set('type', 'mesh'); geom.set('mesh', 'parcel_box'); geom.set('material', 'parcel_print')
                    geom.attrib.pop('size', None)
    # Light: a ceiling light that casts the shadows and a weaker one from the side; no light rides on the camera.
    ET.SubElement(world, 'light', name='ceiling', directional='true', pos='0 0 4', dir='.25 .15 -1', diffuse='.55 .55 .55',
                  specular='.1 .1 .1', castshadow='true')
    ET.SubElement(world, 'light', name='side', directional='true', pos='0 0 4', dir='-.6 -.4 -.5', diffuse='.22 .22 .22',
                  specular='0 0 0', castshadow='false')
    visual = root.find('visual')
    if visual is None:
        visual = ET.SubElement(root, 'visual')
    for child in list(visual):
        visual.remove(child)
    ET.SubElement(visual, 'global', offheight=str(buffer[0]), offwidth=str(buffer[1]))
    ET.SubElement(visual, 'map', znear='.005', zfar='30')         # times the extent below: 1 cm .. 60 m
    ET.SubElement(visual, 'headlight', ambient='.42 .42 .42', diffuse='0 0 0', specular='0 0 0')
    ET.SubElement(visual, 'quality', shadowsize='2048', offsamples='4')
    statistic = root.find('statistic')
    if statistic is None:
        statistic = ET.SubElement(root, 'statistic')
    statistic.set('extent', '2'); statistic.set('center', '0 0 1')
    bodies = {body.get('name'): body for body in root.iter('body')}
    for needed in ('torso', 'l_wrist_cam', 'r_wrist_cam'):
        if needed not in bodies:
            raise ValueError(f'The scene has no body {needed} to carry a camera')
    axes = head_axes()
    for index, (ahead, right, down) in enumerate(FACES):
        ET.SubElement(bodies['torso'], 'camera', name=f'head_cam_{index}', pos=_numbers(HEAD_POSITION), fovy='90',
                      xyaxes=_numbers(np.r_[axes @ right, -(axes @ down)]))
    ahead = (OVERVIEW_AT - OVERVIEW_FROM) / np.linalg.norm(OVERVIEW_AT - OVERVIEW_FROM)
    right = np.cross(ahead, [0., 0., 1.]); right /= np.linalg.norm(right)
    ET.SubElement(world, 'camera', name='overview', pos=_numbers(OVERVIEW_FROM), fovy='40',
                  xyaxes=_numbers(np.r_[right, np.cross(right, ahead)]))
    side = np.cross(WRIST_UP, [0., 0., -1.])                      # camera x = up x (camera z), camera z = -lens axis
    for name, link in (('left_cam', 'l_wrist_cam'), ('right_cam', 'r_wrist_cam')):
        ET.SubElement(bodies[link], 'camera', name=name, pos=_numbers(WRIST_OFFSET), fovy=str(wrist_fovy),
                      xyaxes=_numbers(np.r_[side, WRIST_UP]))
    return ET.tostring(root, encoding='unicode')


def _matrices(quaternions):
    """Rotation matrices (N x 3 x 3) of w,x,y,z quaternions (N x 4)."""
    q = np.asarray(quaternions, dtype=float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1)], axis=1)


class Placement:
    """Puts the bodies of a render model where a recorded row of body poses (N x 7: position, w,x,y,z) has them,
    and everything that hangs on them - geoms, cameras - with them.  ``names`` is the order of the record."""

    def __init__(self, model, names):
        import mujoco
        self.model, self.data = model, mujoco.MjData(model)
        mujoco.mj_forward(model, self.data)                       # lights and anything the record does not name
        rows = {name: row for row, name in enumerate(names)}
        self.bodies = np.array([body for body in range(model.nbody) if model.body(body).name in rows], dtype=int)
        self.rows = np.array([rows[model.body(body).name] for body in self.bodies], dtype=int)
        self.geom_local = _matrices(model.geom_quat) if model.ngeom else np.zeros((0, 3, 3))
        self.camera_local = _matrices(model.cam_quat) if model.ncam else np.zeros((0, 3, 3))

    def place(self, rows):
        rows = np.asarray(rows, dtype=float)
        model, data = self.model, self.data
        turned = _matrices(rows[self.rows, 3:])
        data.xpos[self.bodies] = rows[self.rows, :3]
        data.xquat[self.bodies] = rows[self.rows, 3:]
        data.xmat[self.bodies] = turned.reshape(-1, 9)
        frames = data.xmat.reshape(-1, 3, 3)
        if model.ngeom:
            owner = model.geom_bodyid
            data.geom_xpos[:] = data.xpos[owner] + np.einsum('nij,nj->ni', frames[owner], model.geom_pos)
            data.geom_xmat[:] = (frames[owner] @ self.geom_local).reshape(-1, 9)
        if model.ncam:
            owner = model.cam_bodyid
            data.cam_xpos[:] = data.xpos[owner] + np.einsum('nij,nj->ni', frames[owner], model.cam_pos)
            data.cam_xmat[:] = (frames[owner] @ self.camera_local).reshape(-1, 9)
        return data


class Cameras:
    """The three views of a prepared run.  ``render(rows)`` takes one record of body poses."""

    def __init__(self, run, width=640, height=480, face=None, textures=True):
        import mujoco
        run = Path(run).resolve()                 # MuJoCo reads relative asset files against its mesh folder
        self.width, self.height = int(width), int(height)
        self.face = int(face or 2 * round(.32 * self.width))          # as fine as the fisheye is at its centre
        cache = run / 'camera_textures'
        pictures = square_pictures(run.parent / 'scene/textures', cache) if textures else {}
        self.model = mujoco.MjModel.from_xml_string(render_xml(
            run / 'scene.xml', buffer=(max(self.height, self.face), max(self.width, self.face)), pictures=pictures,
            cache=cache if pictures else None))
        names = [body['name'] for body in json.loads((run / 'model.json').read_text())['bodies']]
        self.placement = Placement(self.model, names)
        self.fisheye = FisheyeMap(self.width, self.height, focal=HEAD_CIRCLE * self.width / np.pi, face=self.face,
                                  field_deg=HEAD_FIELD_DEG)
        self.flat = mujoco.Renderer(self.model, self.height, self.width)
        self.cube = mujoco.Renderer(self.model, self.face, self.face)
        self.count = len(names)

    def render(self, rows):
        """``rows``: body poses in the order of the run's ``model.json`` (N x 7).  Returns name -> H x W x 3 uint8."""
        data = self.placement.place(np.asarray(rows, dtype=float).reshape(self.count, 7))
        faces = np.empty((5, self.face, self.face, 3), dtype=np.uint8)
        for index in range(5):
            self.cube.update_scene(data, camera=f'head_cam_{index}')
            faces[index] = self.cube.render()
        out = {'head_cam': self.fisheye.warp(faces)}
        for name in NAMES[1:]:
            self.flat.update_scene(data, camera=name)
            out[name] = self.flat.render().copy()
        return out

    def overview(self, rows):
        """The view from outside (H x W x 3 uint8) for one record of body poses."""
        data = self.placement.place(np.asarray(rows, dtype=float).reshape(self.count, 7))
        self.flat.update_scene(data, camera='overview')
        return self.flat.render().copy()

    def close(self):
        self.flat.close(); self.cube.close()


def read_poses(run):
    """Recorded body poses of a run: (wall time, simulation time, T x bodies x 7)."""
    run = Path(run)
    count = len(json.loads((run / 'model.json').read_text())['bodies'])
    table = np.fromfile(run / 'poses.bin', dtype=np.float64)
    table = table[:len(table) // (2 + 7 * count) * (2 + 7 * count)].reshape(-1, 2 + 7 * count)
    return table[:, 0], table[:, 1], table[:, 2:].reshape(len(table), count, 7)


def rows_at(times, sim, poses):
    """The newest recorded row at or before each of ``times``: what a camera read at that moment would have seen."""
    index = np.clip(np.searchsorted(sim, np.asarray(times, dtype=float), side='right') - 1, 0, len(sim) - 1)
    return poses[index], index
