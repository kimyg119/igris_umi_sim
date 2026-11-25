"""Pallet, D455 wrist mount (bracket + camera) and colour blocks -> assets/{pallet,d455_mount,blocks}/.

    python scripts/build_props.py        # needs usd-core, trimesh, numpy, Pillow, opencv

Inputs: inputs/pallet_lattice_210mm.stl, inputs/D455_only_55mm_PROTOTYPE.stl (mm, black-filament FDM prints),
inputs/camera_color_640x480.json + inputs/cad_d455_rgb.json (D455 colour intrinsics / pose in UR tool0),
inputs/ycb/070-b textured.glb.orig (ai-habitat/ycb "colored wood blocks": one scanned 1-inch cube; its wood grain).
"""
import json, struct, io
from pathlib import Path
import numpy as np, trimesh, cv2
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

ROOT = Path(__file__).resolve().parents[1]
BLOCK_EDGE = 0.025            # m; the dataset labels say edge_m = 0.025 (the YCB scan measures 25.6 mm)
BLOCK_MASS = 0.010            # kg; 15.6 cm^3 of painted hardwood
BLOCK_FRICTION = 0.7
# linear-RGB paint albedo per colour, hue from the YCB 070-a atlas, brightness set against the wrist-camera frames
PAINT = {"blue": (0.030, 0.105, 0.520), "green": (0.020, 0.230, 0.130), "red": (0.640, 0.060, 0.045),
         "orange": (0.800, 0.200, 0.040), "yellow": (0.830, 0.560, 0.030)}
GRAIN_PX, GRAIN_VARIANTS = 256, 8      # texture px per block face; grain variant g0 = the scanned faces, g1.. = procedural with the scan's statistics
GRAIN_RELIEF = 0.00008                 # m of surface height per unit of dye uptake (stain raises the grain by ~10-20 um)
BLOCK_ROUGHNESS = 0.72                 # stained, unlacquered wood: matt
FDM_LAYER, FDM_LINE = 0.0002, 0.00042        # m: layer height (side walls), extrusion width (top/bottom skins)

def new_stage(path, root_name):
    path.parent.mkdir(parents=True, exist_ok=True)
    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st, 1.0)
    st.SetDefaultPrim(UsdGeom.Xform.Define(st, f"/{root_name}").GetPrim())
    return st


def omnipbr(st, path, textures=None, **inputs):
    mat = UsdShade.Material.Define(st, path); sh = UsdShade.Shader.Define(st, f"{path}/Shader")
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl"); sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    for k, v in inputs.items():
        if isinstance(v, tuple):
            sh.CreateInput(k, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*v))
        else:
            sh.CreateInput(k, Sdf.ValueTypeNames.Float).Set(float(v))
    for k, (file, space) in (textures or {}).items():
        i = sh.CreateInput(k, Sdf.ValueTypeNames.Asset); i.Set(Sdf.AssetPath(file)); i.GetAttr().SetColorSpace(space)
    for out in ("Surface", "Volume", "Displacement"):
        getattr(mat, f"Create{out}Output")("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    pv = UsdShade.Shader.Define(st, f"{path}/Preview"); pv.CreateIdAttr("UsdPreviewSurface")
    pv.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*inputs.get("diffuse_color_constant", (0.5, 0.5, 0.5))))
    pv.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(inputs.get("reflection_roughness_constant", 0.5)))
    mat.CreateSurfaceOutput().ConnectToSource(pv.ConnectableAPI(), "surface")
    return mat


def corner_normals(m, limit_deg=35.0):
    """Per-face-corner normals: average the adjacent faces that are within `limit_deg` of the face (keeps print edges crisp)."""
    fn, cosl = m.face_normals, np.cos(np.radians(limit_deg))
    out = np.zeros((len(m.faces), 3, 3))
    vf = m.vertex_faces
    for f, tri in enumerate(m.faces):
        for c, v in enumerate(tri):
            adj = vf[v][vf[v] >= 0]; keep = adj[fn[adj] @ fn[f] > cosl]
            n = (fn[keep] * m.area_faces[keep, None]).sum(0); out[f, c] = n / (np.linalg.norm(n) + 1e-12)
    return out.reshape(-1, 3)


def fdm_mesh(st, path, stl, mat):
    """STL (mm) -> Mesh (m) with UVs that run the FDM line normal map along the layers (walls) or the 45 deg raster (skins)."""
    m = trimesh.load(str(stl), force="mesh"); m.apply_scale(0.001); m.merge_vertices()
    tri = m.vertices[m.faces]; fn = m.face_normals
    skin = np.abs(fn[:, 2]) > 0.7
    p = tri.reshape(-1, 3); sk = np.repeat(skin, 3)
    uv = np.where(sk[:, None], np.stack([(p[:, 0] + p[:, 1]) / np.sqrt(2) / FDM_LINE, (p[:, 0] - p[:, 1]) / np.sqrt(2) / FDM_LINE], 1),
                  np.stack([(p[:, 0] - p[:, 1]) / FDM_LAYER, p[:, 2] / FDM_LAYER], 1))
    mesh = UsdGeom.Mesh.Define(st, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in m.vertices.tolist()])
    mesh.CreateFaceVertexCountsAttr([3] * len(m.faces)); mesh.CreateFaceVertexIndicesAttr(m.faces.ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateNormalsAttr([Gf.Vec3f(*n) for n in corner_normals(m).tolist()]); mesh.SetNormalsInterpolation("faceVarying")
    mesh.CreateExtentAttr([Gf.Vec3f(*m.bounds[0].tolist()), Gf.Vec3f(*m.bounds[1].tolist())])
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying").Set([Gf.Vec2f(*t) for t in uv.tolist()])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    return mesh, m


def black_pla(st, root):
    """Black PLA/PETG FDM print: near-black, satin, with bead ridges from a 1-D normal map (file shared by both prints)."""
    tex = ROOT / "assets/textures/fdm_lines_normal.png"
    v = (np.arange(64) + 0.5) / 64; h = 0.22 * np.sqrt(np.clip(1 - (2 * v - 1) ** 2, 0, 1))      # bead cross-section, height in pitches
    dh = np.gradient(h, 1 / 64); n = np.stack([np.zeros(64), dh, np.ones(64)], 1); n /= np.linalg.norm(n, axis=1, keepdims=True)
    Image.fromarray(np.repeat(((n * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8)[::-1, None], 64, 1)).save(tex)   # row 0 = top = v 1
    rel = "../textures/fdm_lines_normal.png"
    return omnipbr(st, f"{root}/Looks/BlackPLA", {"normalmap_texture": (rel, "raw")}, diffuse_color_constant=(0.016, 0.016, 0.017),
                   reflection_roughness_constant=0.36, specular_level=0.5, bump_factor=0.7)


def guide_collider(st, path, size=None, centre=(0, 0, 0), hull_points=None):
    if hull_points is None:
        g = UsdGeom.Cube.Define(st, path); g.CreateSizeAttr(1.0)
        x = UsdGeom.Xformable(g); x.AddTranslateOp().Set(Gf.Vec3d(*centre)); x.AddScaleOp().Set(Gf.Vec3f(*size))
        g.CreateExtentAttr([Gf.Vec3f(-0.5), Gf.Vec3f(0.5)])
    else:
        hull = trimesh.convex.convex_hull(hull_points)
        g = UsdGeom.Mesh.Define(st, path); g.CreatePointsAttr([Gf.Vec3f(*v) for v in hull.vertices.tolist()])
        g.CreateFaceVertexCountsAttr([3] * len(hull.faces)); g.CreateFaceVertexIndicesAttr(hull.faces.ravel().tolist())
        UsdPhysics.MeshCollisionAPI.Apply(g.GetPrim()).CreateApproximationAttr("convexHull")
    g.CreatePurposeAttr("guide"); UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    return g


def build_pallet():
    st = new_stage(ROOT / "assets/pallet/pallet.usd", "Pallet")
    UsdGeom.Scope.Define(st, "/Pallet/Looks"); mat = black_pla(st, "/Pallet")
    UsdGeom.Xform.Define(st, "/Pallet/visuals"); _, m = fdm_mesh(st, "/Pallet/visuals/plate", ROOT / "inputs/pallet_lattice_210mm.stl", mat)
    UsdGeom.Xform.Define(st, "/Pallet/collisions")
    guide_collider(st, "/Pallet/collisions/plate", (0.210, 0.210, 0.010), (0, 0, 0.005))     # blocks stand on the z = 10 mm deck
    st.GetRootLayer().Save(); print("pallet:", len(m.faces), "tris, extent", m.extents.round(4))


def forward_distortion(d, K, W, H):
    """RealSense 'inverse Brown-Conrady' coefficients undistort (pixel -> ray). Renderers and OpenCV want the forward
    model (ray -> pixel), so fit [k1, k2, p1, p2, k3] for it over the image by linear least squares."""
    u, v = np.meshgrid(np.linspace(0, W - 1, 33), np.linspace(0, H - 1, 25)); x, y = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
    r2 = x * x + y * y; f = 1 + d[0] * r2 + d[1] * r2 ** 2 + d[4] * r2 ** 3
    xu = x * f + 2 * d[2] * x * y + d[3] * (r2 + 2 * x * x); yu = y * f + 2 * d[3] * x * y + d[2] * (r2 + 2 * y * y)   # librealsense deprojection
    xu, yu, x, y = (a.ravel() for a in (xu, yu, x, y)); q = xu * xu + yu * yu
    A = np.concatenate([np.stack([xu * q, xu * q ** 2, 2 * xu * yu, q + 2 * xu * xu, xu * q ** 3], 1),
                        np.stack([yu * q, yu * q ** 2, q + 2 * yu * yu, 2 * xu * yu, yu * q ** 3], 1)])
    c = np.linalg.lstsq(A, np.concatenate([x - xu, y - yu]), rcond=None)[0]
    return c, float(np.abs(A @ c - np.concatenate([x - xu, y - yu])).max() * K[0, 0])


def build_mount():
    """Frame = UR tool0 (z out of the flange). Bracket STL is already in that frame; the gripper goes on its z = 10 mm face.
    The camera is Isaac Sim's ready-made RealSense D455 asset (body + OV9782 colour / IR / pseudo-depth camera prims);
    only its colour camera is re-parameterised, from the factory calibration of the real unit."""
    st = new_stage(ROOT / "assets/d455_mount/d455_mount.usd", "D455Mount")
    UsdGeom.Scope.Define(st, "/D455Mount/Looks"); mat = black_pla(st, "/D455Mount")
    _, m = fdm_mesh(st, "/D455Mount/bracket", ROOT / "inputs/D455_only_55mm_PROTOTYPE.stl", mat)
    guide_collider(st, "/D455Mount/bracket_collision", hull_points=m.vertices[m.vertices[:, 1] < -0.036])   # camera arm only
    T = np.array(json.load(open(ROOT / "inputs/cad_d455_rgb.json"))["T"])                    # p_tool0 = T @ p_colour_optical
    # rsd455.usd root frame: x forward, y left, z up, origin on the front face; its colour camera sits at y = -11.5 mm.
    # Place the asset so that this camera prim lands exactly on the calibrated colour optical frame.
    T_cam_root = np.eye(4); T_cam_root[:3, :3] = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]); T_cam_root[:3, 3] = (-0.0115, 0, 0)
    body = st.DefinePrim("/D455Mount/D455"); body.GetReferences().AddReference("../sensors/RealSense/D455/rsd455.usd")
    UsdGeom.Xformable(body).ClearXformOpOrder(); UsdGeom.Xformable(body).AddTransformOp().Set(Gf.Matrix4d((T @ T_cam_root).T.tolist()))
    rs = st.OverridePrim("/D455Mount/D455/RSD455"); rs.RemoveAPI(UsdPhysics.RigidBodyAPI); rs.RemoveAPI(UsdPhysics.MassAPI)   # rides on the wrist link
    st.OverridePrim("/D455Mount/D455/RSD455/Imu_Sensor").SetActive(False)                   # needs its own rigid body; not used here
    cal = json.load(open(ROOT / "inputs/camera_color_640x480.json")); K = np.array(cal["K"]); W, H = cal["width"], cal["height"]
    cam = st.OverridePrim("/D455Mount/D455/RSD455/Camera_OmniVision_OV9782_Color")
    f = 1.93                                                                                 # mm, asset and datasheet value; kept
    ha, va = W * f / K[0, 0], H * f / K[1, 1]                                                # sensor area the 640x480 stream covers
    F = Sdf.ValueTypeNames.Float
    for name, val in (("horizontalAperture", ha), ("verticalAperture", va), ("horizontalApertureOffset", (W / 2 - K[0, 2]) / W * ha),
                      ("verticalApertureOffset", (K[1, 2] - H / 2) / H * va), ("omni:sensor:tickRate", 30.0)):   # stream is 30 fps
        cam.CreateAttribute(name, F).Set(float(val))
    dist, err = forward_distortion(cal["dist"], K, W, H)
    cam.AddAppliedSchema("OmniLensDistortionOpenCvPinholeAPI")
    cam.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.Token).Set("opencvPinhole")
    for name, val in zip(("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"), (K[0, 0], K[1, 1], K[0, 2], K[1, 2], *dist)):
        cam.CreateAttribute(f"omni:lensdistortion:opencvPinhole:{name}", F).Set(float(val))
    cam.CreateAttribute("omni:lensdistortion:opencvPinhole:imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(W, H))
    cam.SetCustomDataByKey("r2s:resolution", Gf.Vec2i(W, H)); cam.SetCustomDataByKey("r2s:K", Vt.DoubleArray([float(x) for x in K.ravel()]))
    cam.SetCustomDataByKey("r2s:dist_opencv", Vt.DoubleArray([float(x) for x in dist]))
    st.GetRootLayer().Save()
    print(f"mount: bracket {len(m.faces)} tris; colour camera f={f} mm, aperture {ha:.4f} x {va:.4f} mm, forward distortion {np.round(dist, 5)} (fit error {err:.3f} px)")


def dye_gain(rgb):
    """Per-channel gain on the dye-uptake field. Stain darkens the channels it absorbs most: on the scanned block the
    log-variation of a channel is (1 + 0.67 ln(A_max / A_c)) times that of the brightest one."""
    rgb = np.asarray(rgb, np.float64); return 1 + 0.67 * np.log(rgb.max() / rgb)


def scan_grain():
    """Dye-uptake field s (log brightness of the least absorbed channel about its local mean) of the YCB 070-b scan.
    Only the three cleanly unwrapped faces (the atlas column that reaches the top) are used; the other three are rotations of them."""
    d = open(ROOT / "inputs/ycb/070-b_colored_wood_blocks/google_16k/textured.glb.orig", "rb").read()
    n = struct.unpack("<I", d[12:16])[0]; j = json.loads(d[20:20 + n]); bv = j["bufferViews"][j["images"][0]["bufferView"]]
    b0 = 20 + n + 8 + bv.get("byteOffset", 0); atlas = np.asarray(Image.open(io.BytesIO(d[b0:b0 + bv["byteLength"]])).convert("RGB")).astype(np.float32) / 255
    mask = atlas.max(2) > 30 / 255; S = atlas.shape[0]
    xs = np.flatnonzero(mask[:S // 3].mean(0) > 0.5); x0, x1 = xs.min(), xs.max()
    ys = np.flatnonzero(mask[:, x0 + (x1 - x0) // 4: x1 - (x1 - x0) // 4].mean(1) > 0.9); y0, y1 = ys.min(), ys.max()
    tiles = []
    for k in range(3):
        a, b = y0 + (y1 - y0) * k // 3, y0 + (y1 - y0) * (k + 1) // 3; m = int(0.07 * (x1 - x0))
        t = atlas[a + m: b - m, x0 + m: x1 - m]; lin = np.where(t <= 0.04045, t / 12.92, ((t + 0.055) / 1.055) ** 2.4)
        lg = np.log(np.maximum(lin, 1e-4)); lg -= cv2.GaussianBlur(lg, (0, 0), lg.shape[0] / 4)          # scanner shading
        s = (lg / dye_gain(lin.mean((0, 1)))).mean(2)
        tiles.append(np.clip(cv2.resize(s, (GRAIN_PX, GRAIN_PX), interpolation=cv2.INTER_AREA), -0.35, 0.35))
        print(f"blocks: scanned face {k}: dye-uptake std {tiles[-1].std():.3f}")
    return tiles + [np.rot90(tiles[0]).copy(), np.rot90(tiles[1], 2).copy(), tiles[2][:, ::-1].copy()]


def wood_grain(rng):
    """Dye-uptake field on the six faces of one block cut from a virtual board, consistent in 3D: growth rings with the
    scan's period (2-3 mm), contrast and narrow-dark-line profile, cloudy stain, fibre streaks, now and then a dark blotch."""
    F = GRAIN_PX; u = ((np.arange(F) + 0.5) / F - 0.5) * 25.0                                           # mm
    fibre = int(rng.integers(3)); c1, c2 = [i for i in range(3) if i != fibre]
    ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(16, 60); pith = dist * np.array([np.cos(ang), np.sin(ang)])
    period, amp, tilt = rng.uniform(1.8, 3.2), rng.uniform(0.03, 0.11), rng.uniform(-0.06, 0.06, 2)
    waves = [(rng.uniform(0.1, 0.3), rng.normal(size=2) * 2 * np.pi / rng.uniform(14, 40), 2 * np.pi / rng.uniform(60, 200), rng.uniform(0, 6.28)) for _ in range(3)]
    blotch = (int(rng.integers(6)), rng.uniform(-6, 6, 2), rng.uniform(2.0, 3.5)) if rng.random() < 0.35 else None

    def noise(sx, sy):                                                                                   # unit-std noise, correlation lengths in mm
        n = cv2.GaussianBlur(rng.standard_normal((F, F)).astype(np.float32), (0, 0), sigmaX=sx * F / 25, sigmaY=sy * F / 25, borderType=cv2.BORDER_REFLECT)
        return n / n.std()

    faces = []
    for tile in range(6):                                                                                # face order of the UVs: tile = axis * 2 + (normal > 0)
        ax = tile // 2; a, b = [i for i in range(3) if i != ax]
        P = np.zeros((F, F, 3)); P[..., ax] = 12.5 * (2 * (tile % 2) - 1); P[..., a] = u[None, :]; P[..., b] = u[:, None]
        z = P[..., fibre]; x, y = P[..., c1] - pith[0] + tilt[0] * z, P[..., c2] - pith[1] + tilt[1] * z
        r = np.hypot(x, y) + sum(A * np.sin(k[0] * x + k[1] * y + kz * z + ph) for A, k, kz, ph in waves)
        t = (r / period + 0.25 * np.sin(r / period + waves[0][3])) % 1.0                                 # ring width varies ring to ring
        ring = np.where(t < 0.8, np.exp(-((t - 0.8) / 0.34) ** 2), np.exp(-((t - 0.8) / 0.11) ** 2))     # dark line: slow rise, sharp fall
        s = -amp * np.clip(1 + 0.45 * noise(4, 4), 0.3, None) * (ring - 0.42) / 0.33 + 0.028 * noise(0.12, 0.12)   # rings fade in and out
        if ax == fibre:                                                                                  # end grain
            s += 0.04 * noise(0.9, 0.9)
        else:                                                                                            # fibres run along image x if a is the fibre axis
            lo, hi = ((2.5, 0.7), (3.0, 0.12)) if a == fibre else ((0.7, 2.5), (0.12, 3.0))
            s += 0.04 * noise(*lo) + 0.025 * noise(*hi)
        if blotch and blotch[0] == tile:
            s -= 0.12 * np.exp(-((u[None, :] - blotch[1][0]) ** 2 + (u[:, None] - blotch[1][1]) ** 2) / (2 * blotch[2] ** 2))
        faces.append(s.astype(np.float32))
    mean = np.mean(faces); return [f - mean for f in faces]


def build_blocks():
    out = ROOT / "assets/blocks"; out.mkdir(parents=True, exist_ok=True)
    F, rng = GRAIN_PX, np.random.default_rng(7)
    grain = np.zeros((GRAIN_VARIANTS * F, 6 * F), np.float32)                 # variants 2 across, each one 3 x 2 faces
    for g in range(GRAIN_VARIANTS):
        for k, t in enumerate(scan_grain() if g == 0 else wood_grain(rng)):
            r, c = (g // 2 * 2 + k // 3) * F, (g % 2 * 3 + k % 3) * F; grain[r:r + F, c:c + F] = t
    srgb = lambda x: np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.clip(x, 0, 1) ** (1 / 2.4) - 0.055)
    for name, rgb in PAINT.items():
        alb = np.exp(grain[..., None] * dye_gain(rgb)); cell = alb.reshape(GRAIN_VARIANTS // 2, 2 * F, 2, 3 * F, 3)
        alb = (cell / cell.mean((1, 3), keepdims=True)).reshape(alb.shape) * np.array(rgb)          # every block keeps the paint's mean colour
        Image.fromarray((srgb(np.clip(alb, 0, 1)) * 255 + 0.5).astype(np.uint8)).save(out / f"block_{name}.png")
    h = cv2.GaussianBlur(grain, (0, 0), 0.6) * GRAIN_RELIEF / (0.025 / F)      # raised grain, in pixel units
    gx, gy = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3) / 8, cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3) / 8
    nm = np.dstack([-gx, gy, np.ones_like(gx)]); nm /= np.linalg.norm(nm, axis=2, keepdims=True)
    Image.fromarray(((nm * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8)).save(out / "block_normal.png")
    Image.fromarray((np.clip(BLOCK_ROUGHNESS - 0.9 * grain, 0.55, 0.92) * 255 + 0.5).astype(np.uint8)).save(out / "block_roughness.png")   # stain-soaked lines are duller

    st = new_stage(out / "block.usd", "Block"); root = st.GetDefaultPrim()
    UsdPhysics.RigidBodyAPI.Apply(root); UsdPhysics.MassAPI.Apply(root).CreateMassAttr(BLOCK_MASS)
    root.AddAppliedSchema("PhysxRigidBodyAPI")                    # a 10 g block squeezed between driven fingers needs more solver passes
    root.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(32)
    # eased-edge cube: every corner split in three, chamfer c
    e, c = BLOCK_EDGE / 2, 0.0006
    pts, idx = [], {}
    for s in np.ndindex(2, 2, 2):
        sg = np.array(s) * 2 - 1
        for ax in range(3):
            p = sg * (e - c); p[ax] = sg[ax] * e; idx[(*sg, ax)] = len(pts); pts.append(p)
    pts = np.array(pts); faces = []
    for ax in range(3):
        a, b = [i for i in range(3) if i != ax]
        for s in (-1, 1):
            q = []
            for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                k = [0, 0, 0]; k[ax], k[a], k[b] = s, sa, sb; q.append(idx[(*k, ax)])
            faces.append(q)
        for sa in (-1, 1):
            for sb in (-1, 1):
                def v(s_ax, face_ax):
                    k = [0, 0, 0]; k[ax], k[a], k[b] = s_ax, sa, sb; return idx[(*k, face_ax)]
                faces.append([v(-1, a), v(1, a), v(1, b), v(-1, b)])
    for s in np.ndindex(2, 2, 2):
        sg = np.array(s) * 2 - 1; faces.append([idx[(*sg, 0)], idx[(*sg, 1)], idx[(*sg, 2)]])
    normals, st_uv, counts, indices = [], [], [], []
    for f in faces:
        n = np.cross(pts[f[1]] - pts[f[0]], pts[f[2]] - pts[f[0]]); n /= np.linalg.norm(n)
        if n @ pts[f].mean(0) < 0:
            f = f[::-1]; n = -n
        ax = int(np.argmax(np.abs(n))); tile = ax * 2 + (n[ax] > 0); a, b = [i for i in range(3) if i != ax]   # tile of the nearest main face
        for i in f:
            u, v = np.clip((pts[i][[a, b]] / e + 1) / 2, 0.02, 0.98)
            st_uv.append((tile, u, v)); normals.append(n)
        counts.append(len(f)); indices += list(f)
    UsdGeom.Xform.Define(st, "/Block/visuals"); mesh = UsdGeom.Mesh.Define(st, "/Block/visuals/cube")
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts.tolist()]); mesh.CreateFaceVertexCountsAttr(counts); mesh.CreateFaceVertexIndicesAttr([int(i) for i in indices])
    mesh.CreateSubdivisionSchemeAttr("none"); mesh.CreateNormalsAttr([Gf.Vec3f(*n) for n in np.array(normals).tolist()]); mesh.SetNormalsInterpolation("faceVarying")
    mesh.CreateExtentAttr([Gf.Vec3f(-e), Gf.Vec3f(e)])
    UsdGeom.Scope.Define(st, "/Block/Looks")
    mat = omnipbr(st, "/Block/Looks/Paint", {"normalmap_texture": ("./block_normal.png", "raw"),      # colour texture: per variant, below
                                             "reflectionroughness_texture": ("./block_roughness.png", "raw")},
                  reflection_roughness_texture_influence=1.0, reflection_roughness_constant=BLOCK_ROUGHNESS, specular_level=0.35, bump_factor=1.0)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    vs = root.GetVariantSets().AddVariantSet("color")
    for name in PAINT:
        vs.AddVariant(name); vs.SetVariantSelection(name)
        with vs.GetVariantEditContext():
            a = st.GetPrimAtPath("/Block/Looks/Paint/Shader").CreateAttribute("inputs:diffuse_texture", Sdf.ValueTypeNames.Asset)
            a.Set(Sdf.AssetPath(f"./block_{name}.png")); a.SetColorSpace("sRGB")
    vs.SetVariantSelection("blue")
    gs = root.GetVariantSets().AddVariantSet("grain")              # which block of the grain atlas the faces map to
    for g in range(GRAIN_VARIANTS):
        gs.AddVariant(f"g{g}"); gs.SetVariantSelection(f"g{g}")
        with gs.GetVariantEditContext():
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying").Set(
                [Gf.Vec2f((g % 2 * 3 + t % 3 + u) / 6, 1 - (g // 2 * 2 + t // 3 + v) / GRAIN_VARIANTS) for t, u, v in st_uv])
    gs.SetVariantSelection("g0")
    UsdGeom.Xform.Define(st, "/Block/collisions"); col = guide_collider(st, "/Block/collisions/box", (BLOCK_EDGE,) * 3)
    col.GetPrim().AddAppliedSchema("PhysxCollisionAPI")           # PhysX's default contact offset (~2 cm) is the size of the block itself
    col.GetPrim().CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.001)
    col.GetPrim().CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    pm = UsdShade.Material.Define(st, "/Block/Looks/PaintedWood"); api = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
    api.CreateStaticFrictionAttr(BLOCK_FRICTION); api.CreateDynamicFrictionAttr(BLOCK_FRICTION); api.CreateRestitutionAttr(0.1)
    UsdShade.MaterialBindingAPI.Apply(col.GetPrim()).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")
    st.GetRootLayer().Save(); print("blocks:", list(PAINT), "edge", BLOCK_EDGE)


if __name__ == "__main__":
    build_pallet(); build_mount(); build_blocks()
