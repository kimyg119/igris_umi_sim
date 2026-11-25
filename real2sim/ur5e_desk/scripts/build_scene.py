"""Assemble the UR5e pick-and-place cell (Isaac Sim 6.0, Z-up, metres).

    python scripts/build_scene.py        # any Python with usd-core (pxr); writes ur5e_desk_scene.usda

World frame: origin on the floor under the desk centre. X along the desk (and the wall), +Y toward the wall,
desk-top surface at z = 0.73. The robot stands in front of the desk facing the wall; where it stands and where the
pallet and basket sit relative to it are taken from the real data (see LAYOUT below), not chosen freely.
"""
import math
from pathlib import Path
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ur5e_desk_scene.usda"
TEX = "./assets/textures/"
ROBOT_USD = "./assets/robots/UniversalRobots/ur5e/ur5e.usd"
BASKET_USD = "./assets/basket/basket.usd"
PALLET_USD = "./assets/pallet/pallet.usd"
BLOCK_USD = "./assets/blocks/block.usd"
MOUNT_USD = "./assets/d455_mount/d455_mount.usd"

# --- measured by the user
DESK_L, DESK_D, DESK_TOP_Z = 1.400, 0.600, 0.730
# --- assumed (not visible in the inputs)
TOP_T = 0.025                 # desk-top board thickness
LEG = 0.040                   # black steel square tube
WALL_GAP = 0.025              # desk back edge to wall (the grey shadow strip in the video)
WALL_W, WALL_H, CEIL_Z = 7.0, 2.6, 2.6
WALL_TEX_M = 3.0               # the wall albedo texture covers 3 x 3 m (tiles beyond that)
# --- LAYOUT, recovered from episode 0 of sonnyjin/ur5e_cube_auto_200_success_5826 (wrist-camera frames + joint angles
#     + the nominal camera pose in tool0), all in the robot base frame (base +X points away from the wall):
BASE_TO_DESK_FRONT = 0.300    # desk front edge is this far ahead of the base axis (back edge / wall junction at 0.90)
BASE_ABOVE_DESK = 0.060       # base mounting plane above the desk top (pallet apparent size 57 mm, cube depth 63 mm)
PALLET_IN_BASE = (-0.5752, -0.3506)   # pallet centre; its far-left stack is the cube the episode picks
BASKET_IN_BASE = (-0.6070, 0.0152)    # basket centre (rim corners + floor plate in the home frame)
PALLET_YAW = BASKET_YAW = 0.0         # both square to the desk within ~1.5 deg; basket's low front faces the robot
# --- free choice: where the cell sits along the desk ("pallet a little left of the middle")
PALLET_X = -0.15
# --- robot
ROBOT_YAW = -90.0                                                                   # base -X toward the wall
ROBOT_Q = {"shoulder_pan_joint": 192.99, "shoulder_lift_joint": -99.35, "elbow_joint": 108.10,
           "wrist_1_joint": -111.26, "wrist_2_joint": -95.34, "wrist_3_joint": 11.91}   # deg: the episode's home / camera pose
BRACKET_T = 0.010             # the printed camera bracket sits between flange and gripper (its flange disc is 10 mm)
FINGER_PAD_FRICTION = 0.8     # with 1.2 the pads cannot slide along the block and it stays skewed in the grip
GRIP_TORQUE = 6.0             # N*m limit of the finger drive = ~55 N at the pads (9.2 N per N*m). The asset ships 26 N*m, the
                              # 2F-85's 235 N maximum; PhysX cannot hold a 10 g block against that and the pads sink 11 mm into it.
                              # The force setting used on the real gripper is not in the data.
SOLVER_POSITION_ITERATIONS = 32
UVW_UNIT_M = 1.0              # OmniPBR project_uvw: metres of object space per texture repeat at texture_scale 1
GRAIN_VARIANTS = 8            # grain variants of assets/blocks/block.usd (build_props.py)
# Robotiq 2F-85: asset material name -> finish. The asset's finger links and fingertips are light grey; in the wrist-camera
# footage they are black, with bright pivot pins. Mean roughness of a finish = lerp(constant, 0.5, influence).
GRIPPER_PARTS = {"Cod_Gray": "anodized", "Pumice": "anodized", "Nobel": "anodized", "Gravel": "anodized",
                 "Periwinkle_Gray": "steel", "Stack": "steel", "Celeste": "steel", "Silver_Rust": "steel", "Mystic": "steel",
                 "Rangoon_Green": "rubber", "Dawn": "aluminium", "Gray": "aluminium", "White": "print"}
_UVW = dict(project_uvw=True, reflectionroughness_texture="smudge_roughness.png")       # object-space projection: the CAD meshes have no usable UVs
GRIPPER_FINISH = {
    # black anodised aluminium, blasted: matt. In the wrist-camera footage the fingers stay near black next to the bright desk.
    "anodized": dict(_UVW, diffuse_color_constant=(0.012, 0.012, 0.013), metallic_constant=0.0, specular_level=0.5, reflection_roughness_constant=0.74,
                     reflection_roughness_texture_influence=0.5, texture_scale=(UVW_UNIT_M / 0.08,) * 2),
    "steel": dict(diffuse_color_constant=(0.60, 0.60, 0.61), metallic_constant=1.0, reflection_roughness_constant=0.30),          # pivot pins, screws
    "aluminium": dict(_UVW, diffuse_color_constant=(0.72, 0.73, 0.74), metallic_constant=1.0, reflection_roughness_constant=0.40,
                      reflection_roughness_texture_influence=0.3, texture_scale=(UVW_UNIT_M / 0.08,) * 2),
    "rubber": dict(_UVW, detail_normalmap_texture="surface_grain_normal.png", diffuse_color_constant=(0.022, 0.022, 0.022), metallic_constant=0.0, specular_level=0.35, reflection_roughness_constant=1.0,
                   reflection_roughness_texture_influence=0.3, texture_scale=(UVW_UNIT_M / 0.05,) * 2, detail_texture_scale=(UVW_UNIT_M / 0.006,) * 2, detail_bump_factor=0.8),
    "print": dict(diffuse_color_constant=(0.82, 0.82, 0.80), metallic_constant=0.0, reflection_roughness_constant=0.6)}               # logo
STAND = dict(left=0.47, right=0.13, depth=0.60, plate=0.015, profile=0.040)         # aluminium-profile pedestal (assumed design)
# --- blocks: 9 stacks x 4 layers on the pallet's 3 x 3 squares (77.5 mm pitch, deck at 10 mm); colours bottom -> top
#     as far as the home frame of the episode shows them (top layer exact, lower layers partly guessed)
STACK_PITCH, DECK_Z, BLOCK_EDGE = 0.0775, 0.010, 0.025
STACKS = {(-1, 1): "blue yellow orange green", (0, 1): "red yellow orange blue", (1, 1): "green orange orange blue",
          (-1, 0): "yellow orange red blue", (0, 0): "blue green yellow red", (1, 0): "red green yellow green",
          (-1, -1): "blue yellow red green", (0, -1): "orange red yellow blue", (1, -1): "red yellow yellow blue"}
# --- lighting, solved against the video (see render_check.py): scene-linear desk/wall radiance ratio 0.32 (green) in the
#     phone view, white wall at ~0.8 linear with the default exposure (filmIso 100) so nothing clips.
PANEL_SIZE, PANEL_INTENSITY = (1.2, 0.3), 51500.0                    # ceiling LED panels, neutral white
PANEL_XY = [(-1.0, -2.1), (1.0, -2.1), (-1.0, -4.1), (1.0, -4.1)]
FILL_SIZE, FILL_POS, FILL_INTENSITY = (4.0, 2.4), (0.0, -3.4, 1.35), 1560.0   # bright far side of the room, facing the wall
DOME_INTENSITY, DOME_COLOR = 150.0, (1.0, 1.0, 1.0)                  # residual bounce of the unmodelled white room


def base_to_world(bx, by):
    """Robot-base XY -> world XY."""
    c, s_ = math.cos(math.radians(ROBOT_YAW)), math.sin(math.radians(ROBOT_YAW))
    return ROBOT_XY[0] + c * bx - s_ * by, ROBOT_XY[1] + s_ * bx + c * by


ROBOT_XY = (0.0, -DESK_D / 2 - BASE_TO_DESK_FRONT)
ROBOT_XY = (PALLET_X - base_to_world(*PALLET_IN_BASE)[0], ROBOT_XY[1])
ROBOT_Z = DESK_TOP_Z + BASE_ABOVE_DESK

def chamfer_box(size, c):
    """Box centred at the origin with every edge chamfered by c. Returns (points, counts, indices, face normals)."""
    hx, hy, hz = (s / 2 for s in size)
    pts, idx = [], {}
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                for ax in range(3):       # the corner vertex that lies on the face normal to axis `ax`
                    p = [sx * (hx - c), sy * (hy - c), sz * (hz - c)]
                    p[ax] = (sx, sy, sz)[ax] * (hx, hy, hz)[ax]
                    idx[(sx, sy, sz, ax)] = len(pts); pts.append(Gf.Vec3f(*p))
    faces = []
    for ax in range(3):                   # 6 main faces
        a, b = [i for i in range(3) if i != ax]
        for s in (-1, 1):
            quad = []
            for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                k = [0, 0, 0]; k[ax], k[a], k[b] = s, sa, sb
                quad.append(idx[(*k, ax)])
            faces.append(quad)
    for ax in range(3):                   # 12 edge chamfers (edges parallel to axis `ax`)
        a, b = [i for i in range(3) if i != ax]
        for sa in (-1, 1):
            for sb in (-1, 1):
                def v(s_ax, face_ax):
                    k = [0, 0, 0]; k[ax], k[a], k[b] = s_ax, sa, sb
                    return idx[(*k, face_ax)]
                faces.append([v(-1, a), v(1, a), v(1, b), v(-1, b)])
    for sx in (-1, 1):                    # 8 corner triangles
        for sy in (-1, 1):
            for sz in (-1, 1):
                faces.append([idx[(sx, sy, sz, 0)], idx[(sx, sy, sz, 1)], idx[(sx, sy, sz, 2)]])
    normals, out = [], []
    for f in faces:                       # orient every face outward
        n = Gf.Cross(pts[f[1]] - pts[f[0]], pts[f[2]] - pts[f[0]]).GetNormalized()
        centre = sum((pts[i] for i in f), Gf.Vec3f(0)) / len(f)
        if Gf.Dot(n, centre) < 0:
            f, n = f[::-1], -n
        out.append(f); normals.append(n)
    return pts, [len(f) for f in out], [i for f in out for i in f], normals


def add_box(stage, path, size, centre, chamfer=0.001, uv=None, collide=True):
    pts, counts, indices, normals = chamfer_box(size, chamfer)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(pts); mesh.CreateFaceVertexCountsAttr(counts); mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateNormalsAttr([n for n, c in zip(normals, counts) for _ in range(c)]); mesh.SetNormalsInterpolation("faceVarying")
    mesh.CreateExtentAttr([Gf.Vec3f(*(-s / 2 for s in size)), Gf.Vec3f(*(s / 2 for s in size))])
    UsdGeom.XformCommonAPI(mesh).SetTranslate(Gf.Vec3d(*centre))
    if uv:
        st, k = [], 0
        for n, c in zip(normals, counts):
            for i in indices[k:k + c]:
                st.append(Gf.Vec2f(*uv(pts[i] + Gf.Vec3f(*centre), n)))
            k += c
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying").Set(st)
    if collide:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    return mesh


def gripper_textures():
    """Tileable micro-relief for the gripper finishes: blasted-surface grain (normal map) and handling smudges (roughness)."""
    from PIL import Image
    rng = np.random.default_rng(3); N = 512; f = np.hypot(*np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N)))

    def field(lo, hi, power):                                    # periodic noise, amplitude ~ f^-power between lo and hi cycles/px
        n = np.real(np.fft.ifft2(np.where((f > lo) & (f < hi), np.maximum(f, 1e-6) ** -power, 0) * np.exp(2j * np.pi * rng.random((N, N)))))
        return n / n.std()

    g = field(0.06, 0.45, 0.5); gx, gy = (np.roll(g, -1, 1) - np.roll(g, 1, 1)) / 2, (np.roll(g, -1, 0) - np.roll(g, 1, 0)) / 2
    k = 0.12 / np.hypot(gx, gy).std(); n = np.dstack([-gx * k, gy * k, np.ones_like(g)]); n /= np.linalg.norm(n, axis=2, keepdims=True)
    Image.fromarray(((n * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8)).save(ROOT / TEX / "surface_grain_normal.png")
    Image.fromarray((np.clip(0.5 + 0.12 * field(1.5 / N, 0.05, 1.3), 0.15, 0.85) * 255 + 0.5).astype(np.uint8)).save(ROOT / TEX / "smudge_roughness.png")


def add_material(stage, name, textures=None, preview_color=(0.5, 0.5, 0.5), **inputs):
    """OmniPBR (MDL) material with a UsdPreviewSurface fallback. `textures`: input name -> (file, colour space)."""
    mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    sh = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Shader")
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl"); sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    types = {float: Sdf.ValueTypeNames.Float, bool: Sdf.ValueTypeNames.Bool, tuple: None}
    for k, v in inputs.items():
        if isinstance(v, tuple):
            t = Sdf.ValueTypeNames.Color3f if len(v) == 3 else Sdf.ValueTypeNames.Float2
            sh.CreateInput(k, t).Set(Gf.Vec3f(*v) if len(v) == 3 else Gf.Vec2f(*v))
        else:
            sh.CreateInput(k, types[type(v)]).Set(v)
    for k, (file, space) in (textures or {}).items():
        inp = sh.CreateInput(k, Sdf.ValueTypeNames.Asset); inp.Set(Sdf.AssetPath(TEX + file)); inp.GetAttr().SetColorSpace(space)
    for out in ("surface", "volume", "displacement"):
        getattr(mat, f"Create{out.capitalize()}Output")("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    pv = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Preview"); pv.CreateIdAttr("UsdPreviewSurface")
    pv.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(inputs.get("reflection_roughness_constant", 0.5)))
    pv.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(inputs.get("metallic_constant", 0.0)))
    col = pv.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    if textures and "diffuse_texture" in textures:
        rd = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/PreviewST"); rd.CreateIdAttr("UsdPrimvarReader_float2")
        rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tx = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/PreviewTex"); tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(TEX + textures["diffuse_texture"][0]))
        tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat"); tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
        col.ConnectToSource(tx.ConnectableAPI(), "rgb")
    else:
        col.Set(Gf.Vec3f(*preview_color))
    mat.CreateSurfaceOutput().ConnectToSource(pv.ConnectableAPI(), "surface")
    return mat


def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)


def robot_pose_overrides(q_deg):
    """Forward kinematics on the robot USD's own joints -> {body path relative to the robot root: (translate, orient)}."""
    st = Usd.Stage.Open(str(ROOT / ROBOT_USD))
    root = st.GetDefaultPrim(); root.GetVariantSet("Gripper").SetVariantSelection("Robotiq_2f_85")
    xc = UsdGeom.XformCache()
    bodies = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    old = {p.GetPath(): xc.GetLocalToWorldTransform(p) for p in bodies}
    new = dict(old)

    def frame(j, i):
        m = Gf.Matrix4d().SetRotate(Gf.Quatd(j.GetAttribute(f"physics:localRot{i}").Get()))
        return m.SetTranslateOnly(Gf.Vec3d(j.GetAttribute(f"physics:localPos{i}").Get()))

    for name, q in q_deg.items():                               # dict order = chain order, base to wrist
        j = st.GetPrimAtPath(root.GetPath().AppendPath(f"joints/{name}"))
        b0, b1 = (UsdPhysics.Joint(j).GetBody0Rel().GetTargets()[0], UsdPhysics.Joint(j).GetBody1Rel().GetTargets()[0])
        assert j.GetAttribute("physics:axis").Get() == "Z"
        zero = frame(j, 1).GetInverse() * frame(j, 0) * old[b0]
        assert Gf.IsClose(zero.ExtractTranslation(), old[b1].ExtractTranslation(), 1e-5), name   # USD is assembled at q = 0
        rot = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), q))
        new[b1] = frame(j, 1).GetInverse() * rot * frame(j, 0) * new[b0]
    wrist = root.GetPath().AppendChild("wrist_3_link")
    spacer = Gf.Matrix4d().SetTranslate(old[wrist].TransformDir(Gf.Vec3d(0, 0, BRACKET_T)))   # bracket thickness along the tool axis
    delta = old[wrist].GetInverse() * new[wrist]
    for p in bodies:                                             # the gripper rides rigidly on wrist_3, behind the bracket
        if "/Gripper/" in p.GetPath().pathString:
            new[p.GetPath()] = old[p.GetPath()] * spacer * delta
    out = {}
    for p in bodies:
        local = (new[p.GetPath()] * xc.GetParentToWorldTransform(p).GetInverse()).GetOrthonormalized()
        out[p.GetPath().MakeRelativePath(root.GetPath())] = (local.ExtractTranslation(), local.ExtractRotationQuat())
    # Asset bug (Isaac Sim 6.0 ur5e.usd, Gripper variant): robot_gripper_joint mounts the gripper on the "flange" frame
    # (X out) although the gripper is drawn on tool0 (Z out), so on Play it snaps 90 deg sideways. Re-derive the joint
    # frame on wrist_3 from the drawn pose (localPos1/localRot1 are identity).
    gbase = root.GetPath().AppendPath("Gripper/Robotiq_2F_85/base_link")
    mount = (old[gbase] * spacer * old[wrist].GetInverse()).GetOrthonormalized()
    return out, (Gf.Vec3f(mount.ExtractTranslation()), Gf.Quatf(mount.ExtractRotationQuat()))


def build():
    stage = Usd.Stage.CreateNew(str(OUT))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World"); stage.SetDefaultPrim(world.GetPrim())
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene").GetPrim()
    scene.AddAppliedSchema("PhysxSceneAPI")                       # 240 Hz: small light blocks squeezed by a strong gripper
    scene.CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(240)
    scene.CreateAttribute("physxScene:solverType", Sdf.ValueTypeNames.Token).Set("TGS")

    # ---- materials
    wood = add_material(stage, "DeskWood", {
        "diffuse_texture": ("desk_wood_albedo.png", "sRGB"), "normalmap_texture": ("desk_wood_normal.png", "raw"),
        "reflectionroughness_texture": ("desk_wood_roughness.png", "raw")},
        reflection_roughness_texture_influence=1.0, reflection_roughness_constant=0.47, specular_level=0.5, bump_factor=0.5)
    paint = add_material(stage, "WallPaint", {
        "diffuse_texture": ("wall_paint_albedo.png", "sRGB"), "detail_normalmap_texture": ("wall_stipple_normal.png", "raw")},
        reflection_roughness_constant=0.82, specular_level=0.35, detail_bump_factor=1.0, detail_texture_scale=(WALL_TEX_M / 0.6144,) * 2)
    floor_m = add_material(stage, "FloorVinyl", {"diffuse_texture": ("floor_vinyl_albedo.png", "sRGB")},
                           reflection_roughness_constant=0.42, specular_level=0.5)
    steel = add_material(stage, "BlackSteel", preview_color=(0.02, 0.02, 0.02),
                         diffuse_color_constant=(0.018, 0.018, 0.019), reflection_roughness_constant=0.48, specular_level=0.5)

    # ---- room
    UsdGeom.Xform.Define(stage, "/World/Room")
    wall_y = DESK_D / 2 + WALL_GAP
    wall = add_box(stage, "/World/Room/Wall", (WALL_W, 0.10, WALL_H), (0, wall_y + 0.05, WALL_H / 2), chamfer=0.0005,
                   uv=lambda p, n: ((p[0] + WALL_TEX_M / 2) / WALL_TEX_M, p[2] / WALL_TEX_M))
    bind(wall, paint)
    fl = add_box(stage, "/World/Room/Floor", (WALL_W, 4.4, 0.5), (0, wall_y + 0.10 - 2.2, -0.25), chamfer=0.0005,   # thick slab: thin ones fail GPU convex cooking
                 uv=lambda p, n: (p[0] / 0.6, p[1] / 0.6))
    bind(fl, floor_m)

    # ---- desk
    UsdGeom.Xform.Define(stage, "/World/Desk")
    zb = DESK_TOP_Z - TOP_T

    def top_uv(p, n):                     # u runs along the grain (the face's long horizontal axis), same texel density everywhere
        if abs(n[2]) >= max(abs(n[0]), abs(n[1])):
            return ((p[0] + DESK_L / 2) / DESK_L, (p[1] + DESK_D / 2) / DESK_D)
        if abs(n[1]) >= abs(n[0]):
            return ((p[0] + DESK_L / 2) / DESK_L, (p[2] - zb) / DESK_D)
        return ((p[1] + DESK_D / 2) / DESK_L + 0.2, (p[2] - zb) / DESK_D)

    top = add_box(stage, "/World/Desk/Top", (DESK_L, DESK_D, TOP_T), (0, 0, DESK_TOP_Z - TOP_T / 2), chamfer=0.0012, uv=top_uv)
    bind(top, wood)
    lx, ly = DESK_L / 2 - 0.02 - LEG / 2, DESK_D / 2 - 0.02 - LEG / 2
    for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
        bind(add_box(stage, f"/World/Desk/Leg{i}", (LEG, LEG, zb), (sx * lx, sy * ly, zb / 2), chamfer=0.0015), steel)
    for i, sy in enumerate((-1, 1)):
        bind(add_box(stage, f"/World/Desk/RailLong{i}", (2 * lx - LEG, 0.02, LEG), (0, sy * ly, zb - LEG / 2), chamfer=0.001, collide=False), steel)
    for i, sx in enumerate((-1, 1)):
        bind(add_box(stage, f"/World/Desk/RailShort{i}", (0.02, 2 * ly - LEG, LEG), (sx * lx, 0, zb - LEG / 2), chamfer=0.001, collide=False), steel)

    # ---- robot stand (assumed: only its front beam shows in the wrist camera) and robot
    UsdGeom.Xform.Define(stage, "/World/RobotStand")
    alu = add_material(stage, "Aluminium", preview_color=(0.7, 0.7, 0.72), diffuse_color_constant=(0.62, 0.63, 0.65),
                       metallic_constant=1.0, reflection_roughness_constant=0.38, specular_level=0.5)
    pf, pl = STAND["profile"], STAND["plate"]
    sx0, sx1 = ROBOT_XY[0] - STAND["left"], ROBOT_XY[0] + STAND["right"]
    sy1 = -DESK_D / 2 - 0.005; sy0 = sy1 - STAND["depth"]; beam_z = ROBOT_Z - pl - pf / 2
    bind(add_box(stage, "/World/RobotStand/Plate", (sx1 - sx0, sy1 - sy0 - pf, pl), ((sx0 + sx1) / 2, (sy0 + sy1 - pf) / 2, ROBOT_Z - pl / 2), chamfer=0.001), alu)
    for i, y in enumerate((sy0 + pf / 2, sy1 - pf / 2)):
        bind(add_box(stage, f"/World/RobotStand/BeamX{i}", (sx1 - sx0, pf, pf), ((sx0 + sx1) / 2, y, beam_z), chamfer=0.002, collide=False), alu)
    for i, x in enumerate((sx0 + pf / 2, sx1 - pf / 2)):
        bind(add_box(stage, f"/World/RobotStand/BeamY{i}", (pf, sy1 - sy0 - 2 * pf, pf), (x, (sy0 + sy1) / 2, beam_z), chamfer=0.002, collide=False), alu)
        for k, y in enumerate((sy0 + pf / 2, sy1 - pf / 2)):
            bind(add_box(stage, f"/World/RobotStand/Leg{i}{k}", (pf, pf, beam_z - pf / 2), (x, y, (beam_z - pf / 2) / 2), chamfer=0.002, collide=False), alu)

    # official Isaac Sim 6.0 UR5e with its Robotiq 2F-85 variant, posed by FK so it loads in the episode's home pose
    robot = stage.DefinePrim("/World/UR5e"); robot.GetReferences().AddReference(ROBOT_USD)
    robot.GetVariantSets().AddVariantSet("Gripper").SetVariantSelection("Robotiq_2f_85")
    x = UsdGeom.Xformable(robot); x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*ROBOT_XY, ROBOT_Z)); x.AddRotateZOp().Set(ROBOT_YAW)
    poses, (mount_pos, mount_rot) = robot_pose_overrides(ROBOT_Q)
    gj = stage.OverridePrim(robot.GetPath().AppendPath("joints/robot_gripper_joint"))
    gj.CreateAttribute("physics:localPos0", Sdf.ValueTypeNames.Point3f).Set(mount_pos)
    gj.CreateAttribute("physics:localRot0", Sdf.ValueTypeNames.Quatf).Set(mount_rot)
    for rel, (t, q) in poses.items():
        o = stage.OverridePrim(robot.GetPath().AppendPath(rel))
        o.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(t)
        o.CreateAttribute("xformOp:orient", Sdf.ValueTypeNames.Quatd).Set(q)
    for name, q in ROBOT_Q.items():
        j = stage.OverridePrim(robot.GetPath().AppendPath(f"joints/{name}"))
        j.CreateAttribute("drive:angular:physics:targetPosition", Sdf.ValueTypeNames.Float).Set(float(q))
        j.CreateAttribute("state:angular:physics:position", Sdf.ValueTypeNames.Float).Set(float(q))
    fj = stage.OverridePrim(robot.GetPath().AppendPath("Gripper/Robotiq_2F_85/Joints/finger_joint"))
    fj.CreateAttribute("drive:angular:physics:maxForce", Sdf.ValueTypeNames.Float).Set(GRIP_TORQUE)
    for rel in ("root_joint", "Gripper/Robotiq_2F_85"):          # both carry the articulation root API in the asset
        stage.OverridePrim(robot.GetPath().AppendPath(rel)).CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(SOLVER_POSITION_ITERATIONS)
    pad = UsdShade.Material.Define(stage, "/World/Looks/FingerPad"); pm = UsdPhysics.MaterialAPI.Apply(pad.GetPrim())
    pm.CreateStaticFrictionAttr(FINGER_PAD_FRICTION); pm.CreateDynamicFrictionAttr(FINGER_PAD_FRICTION); pm.CreateRestitutionAttr(0.0)
    for side in ("left", "right"):                               # rubber fingertip pads (the asset ships no contact material)
        f = stage.OverridePrim(robot.GetPath().AppendPath(f"Gripper/Robotiq_2F_85/{side}_inner_finger"))
        UsdShade.MaterialBindingAPI.Apply(f).Bind(pad, UsdShade.Tokens.weakerThanDescendants, "physics")
    # The asset paints every gripper part with a flat OmniPBR colour. Give each part the finish of the real 2F-85.
    src = Usd.Stage.Open(str(ROOT / ROBOT_USD)); src.GetDefaultPrim().GetVariantSet("Gripper").SetVariantSelection("Robotiq_2f_85")
    gripper_textures()
    types = {bool: Sdf.ValueTypeNames.Bool, float: Sdf.ValueTypeNames.Float, str: Sdf.ValueTypeNames.Asset}
    for inst in Usd.PrimRange(src.GetPrimAtPath("/ur5e/Gripper")):
        if not inst.IsInstance():
            continue
        rel = inst.GetPath().MakeRelativePath("/ur5e")
        stage.OverridePrim(robot.GetPath().AppendPath(rel)).SetInstanceable(False)           # instances cannot be overridden
        for m in Usd.PrimRange(inst.GetPrototype()):
            finish = GRIPPER_FINISH.get(GRIPPER_PARTS.get(m.GetParent().GetName())) if m.IsA(UsdShade.Shader) else None
            if finish:
                sh = stage.OverridePrim(robot.GetPath().AppendPath(rel).AppendPath(m.GetPath().MakeRelativePath(inst.GetPrototype().GetPath())))
                for k, v in finish.items():
                    if isinstance(v, tuple):
                        sh.CreateAttribute(f"inputs:{k}", Sdf.ValueTypeNames.Color3f if len(v) == 3 else Sdf.ValueTypeNames.Float2).Set(Gf.Vec3f(*v) if len(v) == 3 else Gf.Vec2f(*v))
                    else:
                        at = sh.CreateAttribute(f"inputs:{k}", types[type(v)]); at.Set(Sdf.AssetPath(TEX + v) if isinstance(v, str) else v)
                        if isinstance(v, str):
                            at.SetColorSpace("raw")
    # printed bracket + D455 on the flange; wrist_3_link's frame is tool0, which is the frame the camera pose is given in
    mount = stage.DefinePrim(robot.GetPath().AppendPath("wrist_3_link/d455_mount")); mount.GetReferences().AddReference(MOUNT_USD)

    # ---- pallet, blocks, basket
    px, py = base_to_world(*PALLET_IN_BASE)
    pallet = stage.DefinePrim("/World/Pallet"); pallet.GetReferences().AddReference(PALLET_USD)
    x = UsdGeom.Xformable(pallet); x.ClearXformOpOrder(); x.AddTranslateOp().Set(Gf.Vec3d(px, py, DESK_TOP_Z)); x.AddRotateZOp().Set(PALLET_YAW)
    UsdGeom.Xform.Define(stage, "/World/Blocks")
    rng = np.random.default_rng(11); cy, sy = math.cos(math.radians(PALLET_YAW)), math.sin(math.radians(PALLET_YAW))
    grain_rng = np.random.default_rng(5)                         # own stream: the stack layout stays as validated
    for (ix, iy), colours in STACKS.items():
        for layer, colour in enumerate(colours.split()):
            lx, ly = np.array([ix, iy]) * STACK_PITCH + rng.uniform(-0.0005, 0.0005, 2)       # hand-stacked: not perfectly square
            blk = stage.DefinePrim(f"/World/Blocks/block_{ix + 1}{iy + 1}_{layer}"); blk.GetReferences().AddReference(BLOCK_USD)
            blk.GetVariantSets().AddVariantSet("color").SetVariantSelection(colour)
            blk.GetVariantSets().AddVariantSet("grain").SetVariantSelection(f"g{grain_rng.integers(GRAIN_VARIANTS)}")
            x = UsdGeom.Xformable(blk); x.ClearXformOpOrder()
            x.AddTranslateOp().Set(Gf.Vec3d(px + cy * lx - sy * ly, py + sy * lx + cy * ly, DESK_TOP_Z + DECK_Z + (layer + 0.5) * BLOCK_EDGE + 0.0002 * (layer + 1)))
            x.AddRotateZOp().Set(float(PALLET_YAW + rng.uniform(-1.5, 1.5) + 90 * rng.integers(4)))
    bx, by = base_to_world(*BASKET_IN_BASE)
    basket = stage.DefinePrim("/World/Basket"); basket.GetReferences().AddReference(BASKET_USD)
    x = UsdGeom.Xformable(basket); x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(bx, by, DESK_TOP_Z + 0.0005)); x.AddRotateZOp().Set(BASKET_YAW)

    # ---- lights (RectLights are not visible to camera rays by default, only in reflections)
    UsdGeom.Xform.Define(stage, "/World/Lights")
    for i, (px, py) in enumerate(PANEL_XY):
        lt = UsdLux.RectLight.Define(stage, f"/World/Lights/Panel{i}")
        lt.CreateWidthAttr(PANEL_SIZE[0]); lt.CreateHeightAttr(PANEL_SIZE[1]); lt.CreateIntensityAttr(PANEL_INTENSITY)
        UsdGeom.XformCommonAPI(lt).SetTranslate(Gf.Vec3d(px, py, CEIL_Z))          # RectLight emits along -Z: facing down
    fill = UsdLux.RectLight.Define(stage, "/World/Lights/RoomFill")
    fill.CreateWidthAttr(FILL_SIZE[0]); fill.CreateHeightAttr(FILL_SIZE[1]); fill.CreateIntensityAttr(FILL_INTENSITY)
    x = UsdGeom.Xformable(fill); x.AddTranslateOp().Set(Gf.Vec3d(*FILL_POS)); x.AddRotateXOp().Set(90.0)   # -Z -> +Y
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/RoomBounce")
    dome.CreateIntensityAttr(DOME_INTENSITY); dome.CreateColorAttr(Gf.Vec3f(*DOME_COLOR))
    stage.GetRootLayer().Save()
    print("wrote", OUT)


if __name__ == "__main__":
    build()
