"""Author the Isaac Sim scene for the UMI box-palletizing replay.

Robot frame: pelvis above the origin, facing +X, metres, Z up.  The scene holds
- the UR5e desk room (walls, desk, floor) as a collision-free backdrop,
- a black flight case carrying three Korea Post no. 2 parcels (pick side),
- a roll container loaded with kraft cartons (place side),
- the IGRIS-C render model with the requested RCV hand posture.
Every physical object is one rigid body with box colliders, which is what the
vendor MuJoCo composition (``isaac_vendor_room/prepare.py``) accepts.

Run with a Python that has pxr and Pillow (``env_isaaclab/bin/python``).
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from . import hands, textures

ROOT = Path(__file__).resolve().parents[2]
ROBOT_USD = ROOT / 'assets/room_real2sim/room_corner_igris.usd'
ROOM_CANDIDATES = [ROOT / 'real2sim/ur5e_desk/ur5e_desk_scene.usda',
                   ROOT / '.local/real2sim_ur5e_desk_f976624/real2sim/ur5e_desk/ur5e_desk_scene.usda']

PARCEL = (.27, .18, .15)            # Korea Post no. 2, from the recorded carry geometry
PARCEL_MASS = .30                   # empty-carton assumption
FRONT_LEVELS, BACK_LEVELS = 3, 5
DECK_TOP = .172                     # roll-container deck above the floor
# Loading surface of the roll container (top of the stacked cartons).  Measured on the real setup (user,
# 2026-11-23): the stacked boxes are 80 cm above the floor, the roll container is 92 cm wide and the black case is
# 77 cm high.  Until then both surfaces were 0.847 m, a value picked to make the lowest box easier to reach.
# The recording sees the case 0.8 cm lower than the cartons on average (std 1.9 cm over 34 episodes).
SURFACE_TOP = .80
CASE_TOP = .77                      # top of the black case, measured on the real setup (user, 2026-11-23)
# Kraft "DELIVERY LAB" cartons seen in the roll container: footprint estimated from the video, height so that
# three of them on the deck reach SURFACE_TOP (20.9 cm).
CARTON = (.41, .31, (SURFACE_TOP - DECK_TOP) / FRONT_LEVELS)
# Pelvis position in the recorded base frame.  The robot's arms are shorter than the operator's (the hands work
# at 0.94 of the arm length), and any trunk lean makes the vendor walking controller pull the pelvis back, which
# puts the hands further out still.  The robot therefore stands 7 cm closer and 3 cm to the right.
# Checked again on 2026-11-23 with the measured heights and the calibrated hands, episode 7 in simulation:
#   (0.07, -0.03)  4 of 6 runs stack all three boxes
#   (0.03, -0.03)  0 of 3: more lean at the lowest box, the pelvis retreats 20-90 cm
#   (0.10, -0.03)  1 of 3,  (0.10, -0.06)  1 of 3: arm joints at their limits 6-7 % (1.7 %), right foot 2-4 cm
#                  from the case instead of 6 cm
# Offline over twelve episodes nothing within 3 cm is better either: back costs lean, forward costs joint limits.
DEFAULT_SHIFT = (.07, -.03)
# Where the supports stand relative to the recorded pick and place points [m].  The recording fixes where the
# boxes are; where the case and the 0.92 m wide roll container are around them is estimated, and the robot is
# thicker than the operator, so there is no placement with room everywhere (README, "장면 치수 확인").
# Roll container, along its own -x (sideways) and +y (away from the robot):
#   (0, 0)      left forearm 6 cm inside the left side panel after every set-down (184 N in simulation)
#   (.10, .04)  forearm free, but the third box then passes the right side panel with 2 cm to spare and
#               scraped it in 10 of 12 simulated runs of episode 7
#   (.06, .06)  box 3 cm clear of the right panel over twelve episodes; the forearm still reaches the left
#               panel in empty-handed motion (3 of 6 runs, 23-152 N).  Kept: the box path is free.
# Not solved by any of them: the feet spread and the base drifts 20-50 cm while the robot stands in the walking
# mode, so a foot reaches the case or the container base in most runs (40-130 N).
CART_SETBACK = (.06, .06)
CASE_SETBACK = .04                  # the case moves this far away from the robot; the stack stands 10 cm from its edge
CASE = (1.10, .60)                  # flight-case top, estimated from the head camera
FRICTION = .6                       # silicone palm on paper, as in the earlier transport trials

_spec = importlib.util.spec_from_file_location('room_real2sim_assets', ROOT / 'assets/room_real2sim/build_scene.py')
assets = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(assets)


def room_usd():
    for path in ROOM_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError('UR5e desk scene not found; expected one of: ' + ', '.join(map(str, ROOM_CANDIDATES)))


def rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def layout_from_episode(report, shift=DEFAULT_SHIFT, carton_height=None, cart_setback=None, case_top=None):
    """Object poses in the robot frame from the episode's pick/place geometry.

    ``shift`` is the pelvis position in the recorded base frame; every recorded
    point moves by its negative.  ``carton_height`` sets the height of the kraft
    cartons in the roll container and with it the height of both work surfaces.
    ``cart_setback`` replaces ``CART_SETBACK``.  ``case_top`` replaces ``CASE_TOP``; with a ``carton_height``
    and no ``case_top`` the case follows the loading surface, as it did before the case was measured.
    """
    shift = np.asarray(shift, dtype=float)
    setback = CART_SETBACK if cart_setback is None else tuple(float(v) for v in cart_setback)
    carton = (CARTON[0], CARTON[1], float(carton_height) if carton_height else CARTON[2])
    surface = DECK_TOP + FRONT_LEVELS * carton[2]
    case_top = float(case_top) if case_top else (surface if carton_height else CASE_TOP)
    pick = np.array(report['pick_xy']) - shift; place = np.array(report['place_xy']) - shift
    pick_yaw = float(report['pick_axis_yaw_rad']); place_yaw = float(report['place_axis_yaw_rad'])
    yaws = report.get('pick_axis_yaws_rad') or [pick_yaw]
    # Case: its long edge follows the parcel's long axis; the stack stands near
    # the corner that faces the robot, as in the recording.
    column_in_case = np.array([CASE[0] / 2 - .20, CASE[1] / 2 - .14 + CASE_SETBACK])
    case_xy = pick - rotation(pick_yaw) @ column_in_case
    # Roll container: local +Y is its back.  The operator faces the back while
    # placing, i.e. along the parcel axis turned by -90 deg.
    cart_yaw = place_yaw - math.pi
    column_in_cart = np.array([-carton[0] / 2 - .005 + setback[0], -carton[1] / 2 - .01 - setback[1]])
    cart_xy = place - rotation(cart_yaw) @ column_in_cart
    return dict(surface_top_m=surface, case_top_m=case_top, carton_size_m=list(carton), parcel_size_m=list(PARCEL), parcel_mass_kg=PARCEL_MASS,
                # Boxes leave top first, so stack level k is taken by carry 2 - k; each box
                # stands at the heading the hands had when they took it.
                parcels=[dict(xy=pick.tolist(), yaw=float(yaws[min(len(yaws) - 1, 2 - level)]),
                              bottom=case_top + level * PARCEL[2]) for level in range(3)],
                case=dict(xy=case_xy.tolist(), yaw=pick_yaw, size=[*CASE, case_top]),
                cart=dict(xy=cart_xy.tolist(), yaw=cart_yaw, carton=list(carton)),
                place=dict(xy=place.tolist(), yaw=place_yaw),
                room=dict(origin_xy=[3.2, -.2], yaw=-math.pi / 2), robot_shift_xy=shift.tolist(),
                source=dict(pick_xy=pick.tolist(), place_xy=place.tolist(), pick_axis_yaw_rad=pick_yaw,
                            place_axis_yaw_rad=place_yaw))


# ---- small authoring helpers -------------------------------------------------

def body(stage, path, xy, yaw, mass, z=0.):
    prim = assets.xform(stage, path, (xy[0], xy[1], z), math.degrees(yaw))
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    return prim


def contact_material(stage, path, friction=FRICTION):
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(friction); api.CreateDynamicFrictionAttr(friction); api.CreateRestitutionAttr(0.)
    return material


def collider(stage, path, centre, size, contact):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*centre)); xf.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdGeom.Imageable(cube).MakeInvisible()
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(contact, materialPurpose='physics')
    return cube.GetPrim()


def textured(stage, path, texture, rough=.7, metal=0., tint=(1, 1, 1)):
    """UsdPreviewSurface material with one colour map addressed by ``st``."""
    material = assets.material(stage, path, tint, rough=rough, metal=metal, tex=texture)
    return material


def quad(stage, path, corners, material, uv=((0, 0), (1, 0), (1, 1), (0, 1))):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in corners])
    mesh.CreateFaceVertexCountsAttr([4]); mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr('none'); mesh.CreateDoubleSidedAttr(True)
    points = np.array(corners, dtype=float)
    mesh.CreateExtentAttr([Gf.Vec3f(*points.min(0)), Gf.Vec3f(*points.max(0))])
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
                                            UsdGeom.Tokens.faceVarying).Set([Gf.Vec2f(*c) for c in uv])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh.GetPrim()


def faced_box(stage, path, size, faces, centre=(0, 0, 0)):
    """Box of six quads; ``faces`` maps +x,-x,+y,-y,+z,-z to materials (upright textures)."""
    x, y, z = np.array(size) / 2
    cx, cy, cz = centre
    corners = {
        '+y': [(x, y, -z), (-x, y, -z), (-x, y, z), (x, y, z)],
        '-y': [(-x, -y, -z), (x, -y, -z), (x, -y, z), (-x, -y, z)],
        '+x': [(x, -y, -z), (x, y, -z), (x, y, z), (x, -y, z)],
        '-x': [(-x, y, -z), (-x, -y, -z), (-x, -y, z), (-x, y, z)],
        '+z': [(-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)],
        '-z': [(-x, y, -z), (x, y, -z), (x, -y, -z), (-x, -y, -z)],
    }
    UsdGeom.Xform.Define(stage, path)
    for name, points in corners.items():
        label = {'+': 'P', '-': 'N'}[name[0]] + name[1].upper()
        quad(stage, f'{path}/{label}', [(px + cx, py + cy, pz + cz) for px, py, pz in points], faces[name])


# ---- objects ------------------------------------------------------------------

def parcel(stage, path, pose, looks, contact):
    x, y, z = PARCEL
    prim = body(stage, path, pose['xy'], pose['yaw'], PARCEL_MASS, pose['bottom'] + z / 2)
    prim.SetCustomData({'class': 'korea_post_no2_parcel', 'dimensions_m': list(PARCEL), 'mass_is_estimate': True})
    faced_box(stage, path + '/Shell', PARCEL, {'+y': looks['long'], '-y': looks['long'], '+x': looks['short'],
                                               '-x': looks['short'], '+z': looks['top'], '-z': looks['bottom']})
    collider(stage, path + '/Collider', (0, 0, 0), PARCEL, contact)
    return prim


def carton(stage, path, centre, looks, size, labels=True):
    x, y, z = size
    faced_box(stage, path, size, {k: looks['kraft'] for k in ('+x', '-x', '+y', '-y', '+z', '-z')}, centre)
    cx, cy, cz = centre
    # Tape along the top seam and the two stickers on the face that looks out of the cart.
    assets.block(stage, path + '/Tape', (cx, cy, cz + z / 2 + .0002), (x - .004, .05, .0003), looks['tape'])
    if labels:
        for i, offset in enumerate((-.10, .10)):
            w, h = .075, .047
            quad(stage, f'{path}/Label{i}', [(cx + offset - w / 2, cy - y / 2 - .0006, cz + .06 - h / 2),
                                             (cx + offset + w / 2, cy - y / 2 - .0006, cz + .06 - h / 2),
                                             (cx + offset + w / 2, cy - y / 2 - .0006, cz + .06 + h / 2),
                                             (cx + offset - w / 2, cy - y / 2 - .0006, cz + .06 + h / 2)], looks['label'])


def roll_container(stage, path, pose, looks, contact):
    """Zinc roll container (0.92 x 0.70 x 1.70 m) loaded with kraft cartons; one rigid body."""
    prim = body(stage, path, pose['xy'], pose['yaw'], 32. + 2 * (FRONT_LEVELS + BACK_LEVELS) * .7)
    prim.SetCustomData({'class': 'rolltainer_loaded', 'dimensions_m': [.92, .70, 1.7], 'mass_is_estimate': True})
    metal, rubber, green, strap = looks['zinc'], looks['rubber'], looks['green'], looks['strap']
    a = assets
    a.block(stage, path + '/Deck', (0, 0, .157), (.92, .70, .030), metal, .004)
    for i, x in enumerate((-.37, .37)):
        a.block(stage, f'{path}/UnderRail{i}', (x, 0, .14), (.05, .65, .014), metal, .002)
    for side, x in enumerate((-.444, .444)):
        points = [(x, -.334, .174), (x, -.334, 1.609)]
        points += [(x, -.259 - .075 * math.cos(t), 1.609 + .075 * math.sin(t)) for t in np.linspace(0, math.pi / 2, 9)[1:]]
        points += [(x, .259, 1.684)]
        points += [(x, .259 + .075 * math.sin(t), 1.609 + .075 * math.cos(t)) for t in np.linspace(0, math.pi / 2, 9)[1:]]
        points += [(x, .334, .174)]
        a.bent_tube(stage, f'{path}/Side{side}_Frame', points, .016, metal)
        for i, y in enumerate((-.19, -.055, .085, .22)):
            a.rod(stage, f'{path}/Side{side}_Bar{i}', (x, y, .18), (x, y, 1.663), .007, metal)
        for i, z in enumerate((.68, 1.02)):
            a.block(stage, f'{path}/Side{side}_Cross{i}', (x, 0, z), (.013, .654, .028), metal, .001)
        a.block(stage, f'{path}/Marker{side}', (x + (.002 if not side else -.002), -.329, 1.42), (.035, .036, .061), green, .001)
        for i, z in enumerate((.53, 1.13)):
            a.block(stage, f'{path}/Strap{side}_{i}', (x + (.010 if side else -.010), 0, z), (.003, .67, .035), strap)
    a.rod(stage, path + '/BackTop', (-.428, .333, 1.68), (.428, .333, 1.68), .012, metal)
    a.block(stage, path + '/BackPlate', (0, .342, 1.095), (.85, .006, .78), looks['plate'], .001)
    for i, x in enumerate(np.linspace(-.32, .32, 6)):
        a.rod(stage, f'{path}/BackBar{i}', (float(x), .343, .18), (float(x), .343, 1.67), .006, metal)
    for i, (x, y) in enumerate(((-.365, -.245), (-.365, .245), (.365, -.245), (.365, .245))):
        a.rod(stage, f'{path}/Caster{i}_Bearing', (x, y, .103), (x, y, .13), .025, metal)
        for k, side in enumerate((-1, 1)):
            a.block(stage, f'{path}/Caster{i}_Cheek{k}', (x + side * .024, y - .01, .086), (.006, .052, .049), metal, .002)
        a.rod(stage, f'{path}/Caster{i}_Tire', (x - .0175, y - .02, .063), (x + .0175, y - .02, .063), .063, rubber)
    # Fiducial board taped to the left side frame, as in the recording.
    quad(stage, path + '/Tag', [(-.462, -.02, 1.25), (-.462, -.26, 1.25), (-.462, -.26, 1.49), (-.462, -.02, 1.49)], looks['tag'])
    quad(stage, path + '/TagInside', [(-.4595, -.26, 1.25), (-.4595, -.02, 1.25), (-.4595, -.02, 1.49), (-.4595, -.26, 1.49)], looks['tag'])
    deck = DECK_TOP
    size = tuple(pose.get('carton', CARTON))
    x, y, z = size
    for column, cx in enumerate((-x / 2 - .005, x / 2 + .005)):
        for level in range(FRONT_LEVELS):       # front row: the loading surface
            carton(stage, f'{path}/Front_C{column}_L{level}', (cx, -y / 2 - .01, deck + z / 2 + level * z), looks, size)
        for level in range(BACK_LEVELS):        # back row, two levels above the loading surface
            carton(stage, f'{path}/Back_C{column}_L{level}', (cx, y / 2 + .01, deck + z / 2 + level * z), looks, size)
    # Colliders: wheel base, the two carton stacks and the three cage walls.
    collider(stage, path + '/Base_Collider', (0, 0, deck / 2), (.92, .70, deck), contact)
    collider(stage, path + '/Front_Collider', (0, -y / 2 - .01, deck + FRONT_LEVELS * z / 2), (2 * x + .01, y, FRONT_LEVELS * z), contact)
    collider(stage, path + '/Back_Collider', (0, y / 2 + .01, deck + BACK_LEVELS * z / 2), (2 * x + .01, y, BACK_LEVELS * z), contact)
    for side, sx in enumerate((-.444, .444)):
        collider(stage, f'{path}/Side{side}_Collider', (sx, 0, deck + .756), (.032, .70, 1.512), contact)
    collider(stage, path + '/BackWall_Collider', (0, .343, deck + .756), (.92, .024, 1.512), contact)
    return prim


def flight_case(stage, path, pose, looks, contact):
    """Black laminate transport case with aluminium edging; top at ``size[2]``."""
    length, width, top = pose['size']
    wheel = .105
    prim = body(stage, path, pose['xy'], pose['yaw'], 45.)
    prim.SetCustomData({'class': 'igris_c_flight_case', 'dimensions_m': [length, width, top], 'dimensions_are_estimates': True})
    height = top - wheel
    centre_z = wheel + height / 2
    faced_box(stage, path + '/Panels', (length - .004, width - .004, height - .004),
              {k: looks['laminate'] for k in ('+x', '-x', '+y', '-y', '+z', '-z')}, (0, 0, centre_z))
    edge = .030
    alu = looks['aluminium']
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):            # vertical edges
        assets.block(stage, f'{path}/EdgeZ{i}', (sx * (length - edge) / 2, sy * (width - edge) / 2, centre_z),
                     (edge, edge, height), alu, .003)
    for i, (sy, z) in enumerate(((-1, wheel + edge / 2), (1, wheel + edge / 2), (-1, top - edge / 2), (1, top - edge / 2))):
        assets.block(stage, f'{path}/EdgeX{i}', (0, sy * (width - edge) / 2, z), (length - 2 * edge, edge, edge), alu, .003)
    for i, (sx, z) in enumerate(((-1, wheel + edge / 2), (1, wheel + edge / 2), (-1, top - edge / 2), (1, top - edge / 2))):
        assets.block(stage, f'{path}/EdgeY{i}', (sx * (length - edge) / 2, 0, z), (edge, width - 2 * edge, edge), alu, .003)
    seam = top - .16                                                                     # lid joint extrusion
    for i, sy in enumerate((-1, 1)):
        assets.block(stage, f'{path}/LidX{i}', (0, sy * (width / 2 + .0005), seam), (length - 2 * edge, .003, .034), alu, .001)
        for k, x in enumerate((-length * .28, length * .28)):                            # butterfly latches
            assets.block(stage, f'{path}/Latch{i}{k}', (x, sy * (width / 2 + .002), seam), (.105, .006, .125), looks['steel'], .002)
            assets.block(stage, f'{path}/LatchWing{i}{k}', (x, sy * (width / 2 + .006), seam + .01), (.05, .006, .03), looks['steel'], .002)
    for i, sx in enumerate((-1, 1)):
        assets.block(stage, f'{path}/LidY{i}', (sx * (length / 2 + .0005), 0, seam), (.003, width - 2 * edge, .034), alu, .001)
        assets.block(stage, f'{path}/Handle{i}', (sx * (length / 2 + .002), 0, wheel + height * .42), (.006, .17, .11), looks['steel'], .002)
        assets.block(stage, f'{path}/Grip{i}', (sx * (length / 2 + .007), 0, wheel + height * .42), (.008, .12, .022), looks['rubber'], .002)
    for i, (sx, sy, sz) in enumerate([(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (wheel + .012, top - .012)]):
        ball = UsdGeom.Sphere.Define(stage, f'{path}/Corner{i}')
        ball.CreateRadiusAttr(.024)
        UsdGeom.Xformable(ball).AddTranslateOp().Set(Gf.Vec3d(sx * (length / 2 - .012), sy * (width / 2 - .012), sz))
        UsdShade.MaterialBindingAPI.Apply(ball.GetPrim()).Bind(looks['steel'])
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        x, y = sx * (length / 2 - .10), sy * (width / 2 - .09)
        assets.block(stage, f'{path}/CasterPlate{i}', (x, y, wheel - .004), (.095, .075, .006), looks['steel'], .001)
        assets.rod(stage, f'{path}/CasterTire{i}', (x - .016, y, .05), (x + .016, y, .05), .05, looks['rubber'])
    # "IGRIS-C" lettering on the lid, readable from the robot side.
    lx, ly = -length / 2 + .08, -width / 2 + .07
    quad(stage, path + '/Label', [(lx, ly, top + .0006), (lx + .42, ly, top + .0006), (lx + .42, ly + .11, top + .0006),
                                  (lx, ly + .11, top + .0006)], looks['case_label'], uv=((1, 1), (0, 1), (0, 0), (1, 0)))
    collider(stage, path + '/Collider', (0, 0, top / 2), (length, width, top), contact)
    return prim


def backdrop(out, room):
    """Flatten the UR5e desk room without simulation schemas."""
    source = Usd.Stage.Open(str(room))
    if UsdGeom.GetStageMetersPerUnit(source) != 1 or UsdGeom.GetStageUpAxis(source) != 'Z':
        raise ValueError('UR5e desk scene must use metres and Z up')
    flat = Usd.Stage.Open(source.Flatten())
    for prim in list(flat.Traverse()):
        if prim.IsA(UsdPhysics.Joint) or prim.IsA(UsdPhysics.Scene):
            prim.SetActive(False)
            continue
        for name in list(prim.GetAppliedSchemas()):
            if name.startswith('Physics') or name.startswith('Physx'):
                prim.RemoveAppliedSchema(name)
    target = Path(out) / 'backdrop.usdc'
    flat.GetRootLayer().Export(str(target))
    check = Usd.Stage.Open(str(target))
    if any(p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.CollisionAPI) for p in check.Traverse()):
        raise RuntimeError('Backdrop still carries physics schemas')
    return target


def camera(stage, name, eye, target, focal=28.):
    cam = UsdGeom.Camera.Define(stage, '/World/Cameras/' + name)
    matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
    UsdGeom.Xformable(cam).MakeMatrixXform().Set(matrix)
    cam.CreateFocalLengthAttr(focal); cam.CreateHorizontalApertureAttr(36.)
    cam.CreateClippingRangeAttr(Gf.Vec2f(.01, 100.))
    return cam


def pose_hands(stage, pose):
    """Give the render model the same finger and thumb posture as the physics model."""
    model = '/World/IGRIS_C/Model/'
    cache = UsdGeom.XformCache()
    count = 0
    for root_name, links in hands.usd_overrides(pose).items():
        root = stage.GetPrimAtPath(model + root_name)
        if not root:
            raise ValueError(f'Render model has no {root_name}')
        root_world = np.array(cache.GetLocalToWorldTransform(root)).T       # column-vector 4x4
        for name, in_root in links.items():
            prim = stage.GetPrimAtPath(model + name)
            if not prim:
                raise ValueError(f'Render model has no hand link {name}')
            parent_world = np.array(cache.GetLocalToWorldTransform(prim.GetParent())).T
            local = np.linalg.inv(parent_world) @ root_world @ in_root
            UsdGeom.Xformable(prim).MakeMatrixXform().Set(Gf.Matrix4d(*local.T.reshape(-1).tolist()))
            count += 1
    return count


def build(out, layout, hand_pose=None):
    """Write ``scene.usda`` (plus textures and backdrop) into a new directory."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    textures.build(out)
    stage = Usd.Stage.CreateNew(str(out / 'scene.usda'))
    UsdGeom.SetStageMetersPerUnit(stage, 1); UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, '/World').GetPrim()
    stage.SetDefaultPrim(world)
    m = '/World/Looks/'
    looks = dict(
        long=textured(stage, m + 'ParcelLong', 'parcel_long'), short=textured(stage, m + 'ParcelShort', 'parcel_short'),
        top=textured(stage, m + 'ParcelTop', 'parcel_top', rough=.6), bottom=textured(stage, m + 'ParcelBottom', 'parcel_bottom'),
        kraft=assets.material(stage, m + 'Kraft', (.4, .25, .12), tex='kraft', normal=True),
        tape=assets.material(stage, m + 'Tape', (.45, .3, .13), rough=.36, tex='tape'),
        label=textured(stage, m + 'DeliveryLabel', 'delivery_label', rough=.5),
        zinc=assets.material(stage, m + 'Zinc', (.55, .59, .59), rough=.46, metal=.85, tex='zinc'),
        plate=assets.material(stage, m + 'GreyPlate', (.36, .40, .42), rough=.62, metal=.3),
        rubber=assets.material(stage, m + 'Rubber', (.018, .020, .019), rough=.85),
        green=assets.material(stage, m + 'GreenClip', (.005, .44, .26), rough=.39),
        strap=assets.material(stage, m + 'Strap', (.012, .014, .012), rough=.9),
        tag=textured(stage, m + 'Tag', 'tag', rough=.7),
        laminate=textured(stage, m + 'CaseLaminate', 'case_laminate', rough=.42),
        aluminium=textured(stage, m + 'Aluminium', 'aluminium', rough=.32, metal=.95),
        steel=assets.material(stage, m + 'Steel', (.62, .63, .64), rough=.3, metal=.95),
        case_label=textured(stage, m + 'CaseLabel', 'case_label', rough=.5),
    )
    contact = contact_material(stage, m + 'Contact')

    # Backdrop: UR5e desk room turned so that its wall is ahead of the robot.
    room = layout['room']
    reference = backdrop(out, room_usd())
    holder = assets.xform(stage, '/World/Backdrop', (*room['origin_xy'], 0.), math.degrees(room['yaw']))
    holder.GetReferences().AddReference('./' + reference.name, '/World')
    # The source floor ends 0.7 m behind the robot and the room has one wall;
    # repeat both so that cameras behind the robot still see a closed room.
    def repeat(name, source, material, translate, turn=0., stretch=1.):
        holder_prim = UsdGeom.Xform.Define(stage, '/World/Backdrop/' + name)
        ops = UsdGeom.Xformable(holder_prim)
        ops.AddTranslateOp().Set(Gf.Vec3d(*translate)); ops.AddRotateZOp().Set(turn); ops.AddScaleOp().Set(Gf.Vec3f(stretch, 1, 1))
        copy = stage.DefinePrim(f'/World/Backdrop/{name}/Mesh')
        copy.GetReferences().AddReference('./' + reference.name, source)
        # The source binding points outside the referenced prim; bind the composed material again.
        UsdShade.MaterialBindingAPI.Apply(copy).Bind(UsdShade.Material(stage.GetPrimAtPath('/World/Backdrop/Looks/' + material)))
    repeat('FloorBehind', '/World/Room/Floor', 'FloorVinyl', (0, -4.4, 0))
    repeat('WallBehind', '/World/Room/Wall', 'WallPaint', (0, -8.75, 0))
    repeat('WallRight', '/World/Room/Wall', 'WallPaint', (3.925, -4.0, 0), 90., 1.4)
    repeat('WallLeft', '/World/Room/Wall', 'WallPaint', (-3.175, -4.0, 0), 90., 1.4)

    flight_case(stage, '/World/Case', layout['case'], looks, contact)
    roll_container(stage, '/World/Rolltainer', layout['cart'], looks, contact)
    for index, pose in enumerate(layout['parcels']):
        parcel(stage, f'/World/Parcel_{index}', pose, looks, contact)

    robot = UsdGeom.Xform.Define(stage, '/World/IGRIS_C').GetPrim()
    robot.GetReferences().AddReference(str(ROBOT_USD), '/World/IGRIS_C')
    posed = pose_hands(stage, hand_pose or hands.load_pose())

    # The source room lights were exposed for its own render settings; the vendor
    # room renderer uses a fixed tonemap, so bring them to the same level here.
    for name, value in (('Panel0', 2600.), ('Panel1', 2600.), ('Panel2', 2600.), ('Panel3', 2600.),
                        ('RoomFill', 90.), ('RoomBounce', 45.)):
        prim = stage.GetPrimAtPath('/World/Backdrop/Lights/' + name)
        if not prim:
            raise ValueError(f'Backdrop light {name} is missing')
        prim.GetAttribute('inputs:intensity').Set(value)
    lights = '/World/Lights/'
    for i, (x, y) in enumerate(((.6, -.9), (.6, .9), (-1.4, 0.))):
        light = UsdLux.RectLight.Define(stage, f'{lights}Work{i}')
        light.CreateIntensityAttr(2600); light.CreateWidthAttr(1.2); light.CreateHeightAttr(.3)
        light.CreateColorAttr(Gf.Vec3f(1, .974, .932))
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(x, y, 2.6))
    camera(stage, 'UmiOverview', (-2.3, -1.9, 2.05), (.35, .05, .85), 28)
    camera(stage, 'UmiFront', (2.25, -1.55, 1.75), (.2, .05, .9), 30)
    camera(stage, 'UmiCart', (-.55, 2.35, 2.2), (.3, .2, .85), 30)
    camera(stage, 'UmiTop', (.25, .001, 3.9), (.25, 0., 0.), 24)
    stage.GetRootLayer().customLayerData = {'cameraSettings': {'Perspective': {'position': Gf.Vec3d(-2.3, -1.9, 2.05),
                                                                               'target': Gf.Vec3d(.35, .05, .85)}},
                                            'renderSettings': {'renderer': 'RayTracedLighting'}}
    stage.GetRootLayer().Save()
    report = dict(scene=str(out / 'scene.usda'), layout=layout, posed_hand_links=posed, backdrop=str(reference),
                  robot_usd=str(ROBOT_USD), room_usd=str(room_usd()))
    (out / 'scene.json').write_text(json.dumps(report, indent=2) + '\n')
    return out / 'scene.usda'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode-json', type=Path, required=True, help='episode.json written by episode.py')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    layout = layout_from_episode(json.loads(args.episode_json.read_text()))
    print(build(args.out, layout))


if __name__ == '__main__':
    main()
