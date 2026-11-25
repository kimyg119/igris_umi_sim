#!/usr/bin/env python3
"""Ivory folding crate ("basket") for the ur5e_desk real-to-sim scene.

    <python with shapely, trimesh, manifold3d, mapbox-earcut, usd-core> build_basket.py

geometry (shapely -> bevelled extruded solids -> manifold union) -> weighted normals -> basket.usd (pxr)
-> self-check -> Cycles preview renders of the USD. Only the render step needs Blender: it re-runs this file
under BLENDER_PY (a python with bpy 4.x). Modelling units are mm, the USD is in metres.
Frame: X = width, Y = length (+Y = tall back wall, -Y = low front), Z up, origin = footprint centre on the table.
"""
import os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_USD = os.path.join(HERE, '..', 'assets', 'basket', 'basket.usd')
RENDER_DIR = os.path.join(HERE, '..', 'renders')
BLENDER_PY = os.environ.get('BLENDER_PY', '/tmp/claude-1000/-home-rcv-workspace-Project-Humanoid/'
                            '6eed135c-4592-4867-83ac-e55d3657331e/scratchpad/bvenv/bin/python')

# ---------------- parameters (mm) ----------------
LX, LY, H, H_FRONT = 210.0, 300.0, 120.0, 50.0      # hard dimensions: footprint, back height, front height
MASS_KG = 0.33
BEVEL, SMOOTH_DEG = 0.5, 60.0                       # edge bevel width (2 segments), normal smoothing limit
# name -> (material name, sRGB sampled from the photos, roughness)
MESHES = {'body': ('Ivory', (226, 219, 196), 0.45), 'base_tray': ('GreyIvory', (211, 208, 190), 0.45)}

# base tray: band is taller along the long sides than at the two ends (the end walls hinge lower)
BAND_SIDE, BAND_END = 30.0, 16.0
FLOOR_W, FLOOR_L, FLOOR_Z = 192.0, 265.0, (2.5, 5.0)   # flat slatted floor, (bottom, top) height
PITCH, SLOT, NX, NY = 10.0, 5.0, 18, 26             # floor slot pitch / width, slot columns across X and Y
RING_END = ((6, 42), (52, 90))                      # two slot rings, distances from the front/back floor edge
RING_SIDE = ((7, 33), (43, 68))                     # ... and from the side floor edges (rest = solid centre plate)
DIAG_RIB = 6.0                                      # corner-to-centre-plate diagonal ribs
LEDGE_HOLES = (58.0, 14.0, 4.0)                     # back hinge-ledge through holes: |x| centre, width, depth

# walls (all wall faces sit INSET behind the tray footprint; only the handles reach it)
INSET, T_SIDE, T_END, FRONT_D = 1.5, 11.0, 10.5, 15.5   # side / back wall thickness, front ledge depth
PLATE, RECESS, GROOVE = 2.5, 2.0, 4.6               # slotted plate thickness, its recess behind the frame, mullion groove
RIM, RIM_SLOPE, KNEE_Y, KNEE_R = 7.0, 5.5, -102.0, 8.0  # rim height (top / along the slope), where the slope starts
SIDE_PITCH, SIDE_SLOT, SIDE_ROWS = 10.4, 5.3, ((41, 72), (82, 111.5))
MULLIONS = (102.3, 23.1, -56.1)                     # side-wall mullion centres (y); 7 slot columns between them
HANDLE = (MULLIONS[1] - 54, 86.5, MULLIONS[1] + 54, 110.5)   # handle frame outline (y0, z0, y1, z1), wall 4 mm
END_MULL, END_SLOT = 38.5, 5.0                      # back/front wall mullion |x|
BACK_COLS = [0, 10, 20, 30] + [46.75 + 10 * i for i in range(5)]   # |x| of slot columns (5 + 7 + 5)
BACK_ROWS = ((27.5, 66), (77.5, 115))
FRONT_COLS, FRONT_ROWS = [28, 49, 59, 69, 79], ((23, 44.5),)

# preview cameras: (position m, quaternion wxyz) recovered from the photos by PnP, f = 5280 px at 4284 px width
CAMS = {'like_3037': ((-0.3371, -0.1669, 0.3883), (0.7949, 0.3372, -0.2129, -0.4574)),
        'like_3036': ((0.0004, -0.0972, 0.4171), (0.9885, 0.1508, 0.0031, 0.0039)),
        'like_3039': ((0.1859, 0.3688, 0.3562), (0.1855, 0.1222, 0.4373, 0.8715))}

HX, HY = LX / 2, LY / 2
XO, XI, YO = HX - INSET, HX - INSET - T_SIDE, HY - INSET   # side wall outer / inner face, end wall outer face
BW = XI - 0.75                                             # half width of the back / front walls

STAGE = sys.argv[1] if len(sys.argv) > 1 else 'build'
if STAGE == 'build':
    import trimesh
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
    from shapely import affinity, distance, points, set_precision
    from shapely.geometry import LineString, Polygon, box
    from shapely.ops import unary_union
else:
    import bpy


# ---------------- geometry ----------------
# ponytail: hinge barrels, latch tabs, ejector marks and the stepped front/back tray lips are not modelled, and the
# tray band is a solid block; add them as extra solid() parts in body()/tray() if close-up shots ever need them.
def rrect(a, b, c, d, r):
    return box(a + r, b + r, c - r, d - r).buffer(r, quad_segs=6)


def polys(g):
    return [p for p in getattr(g, 'geoms', [g]) if p.geom_type == 'Polygon' and not p.is_empty]


def clean(s, r=0.8):
    """Opening: drops slivers and rounds the slot corners."""
    return unary_union([g for g in polys(s.buffer(-r).buffer(r, quad_segs=4)) if g.area > 5])


def slots(cols, rows, w, clip=None):
    s = unary_union([box(c - w / 2, a, c + w / 2, b) for c in cols for a, b in rows])
    return clean(s if clip is None else s & clip)


def solid(poly, lo, hi, axis='z'):
    """Extrude a 2D shape from lo to hi along axis ('z': shape in x,y; 'x': in y,z; 'y': in x,z), every edge bevelled:
    the corners are filleted in 2D and both caps get a 2-segment bevel (outline rings inset by 0.29 and 1 x BEVEL)."""
    b, mid = BEVEL, 0.293 * BEVEL
    snap = lambda g: set_precision(g, 1e-3)          # boolean leftovers otherwise break the triangulation
    out = []
    for g0 in polys(snap(poly.buffer(-b).buffer(2 * b, quad_segs=2).buffer(-b, quad_segs=2))):
        g1, g2 = snap(g0.buffer(-mid, join_style='mitre')), snap(g0.buffer(-b, join_style='mitre'))
        V, F, n = [], [], 0
        for part in (g2, g1 - g2, g0 - g1):                          # flat cap, then the two bevel rings
            for g in polys(part):
                xy, f = trimesh.creation.triangulate_polygon(g, engine='earcut')
                d = distance(points(xy), g0.boundary)
                drop = np.where(d < 0.15 * b, b, np.where(d < 0.65 * b, mid, 0))     # bevel profile
                V += [np.c_[xy, hi - drop], np.c_[xy, lo + drop]]
                F += [f + n, f[:, ::-1] + n + len(xy)]
                n += 2 * len(xy)
        for ring in (g0.exterior, *g0.interiors):                    # side walls between the two bevels
            p = np.array(ring.coords)[:-1]
            k = len(p)
            i, j = np.arange(k), (np.arange(k) + 1) % k
            V += [np.c_[p, np.full(k, lo + b)], np.c_[p, np.full(k, hi - b)]]
            F += [np.c_[i, j, j + k] + n, np.c_[i, j + k, i + k] + n]
            n += 2 * k
        m = trimesh.Trimesh(np.vstack(V), np.vstack(F))               # merges the shared ring vertices
        trimesh.repair.fix_normals(m)
        m.vertices = m.vertices[:, {'z': [0, 1, 2], 'x': [2, 0, 1], 'y': [0, 2, 1]}[axis]]
        if axis == 'y':
            m.invert()
        out.append(m)
    return out


def mirror(parts, axis):
    out = []
    for m in parts:
        m = m.copy()
        s = np.ones(3)
        s[axis] = -1
        m.vertices = m.vertices * s
        m.invert()
        out.append(m)
    return out


def flip(g, sx, sy):
    return g | affinity.scale(g, sx, sy, origin=(0, 0))


def floor():
    """Picture-frame slot pattern: slots run perpendicular to the nearest edge, two rings, diagonal corner ribs."""
    hw, hl = FLOOR_W / 2, FLOOR_L / 2
    cx, cy = hw - RING_SIDE[1][1], hl - RING_END[1][1]          # solid centre plate half size
    end_sec = flip(Polygon([(-hw, hl), (hw, hl), (cx, cy), (-cx, cy)]), 1, -1)
    side_sec = flip(Polygon([(hw, hl), (hw, -hl), (cx, -cy), (cx, cy)]), -1, 1)
    end_ring = flip(unary_union([box(-hw, hl - b, hw, hl - a) for a, b in RING_END]), 1, -1)
    side_ring = flip(unary_union([box(hw - b, -hl, hw - a, hl) for a, b in RING_SIDE]), -1, 1)
    v = unary_union([box(x - SLOT / 2, -hl, x + SLOT / 2, hl) for x in (np.arange(NX) - (NX - 1) / 2) * PITCH])
    h = unary_union([box(-hw, y - SLOT / 2, hw, y + SLOT / 2) for y in (np.arange(NY) - (NY - 1) / 2) * PITCH])
    rib = LineString([(hw + (hw - cx) * 0.2, hl + (hl - cy) * 0.2), (cx, cy)]).buffer(DIAG_RIB / 2, cap_style='flat')
    ribs = flip(flip(rib, -1, 1), 1, -1) - box(-cx, -hl, cx, hl) - box(-hw, -cy, hw, cy)
    s = (v & end_ring & end_sec | h & side_ring & side_sec) - ribs
    return solid(box(-hw - 1, -hl - 1, hw + 1, hl + 1) - clean(s), *FLOOR_Z)


def side_wall():
    """+X wall, drawn in (y, z): rim along the top and the slope, slotted plate, handle tube, mullions, posts."""
    z0, xp = BAND_SIDE - 0.5, XO - RECESS
    P = Polygon([(YO, z0), (YO, H), (KNEE_Y, H), (-YO, H_FRONT), (-YO, z0)])
    P = P.buffer(-KNEE_R).buffer(KNEE_R, quad_segs=8) | (P - box(KNEE_Y - 20, H - 20, KNEE_Y + 20, H + 1))  # round knee
    k, e = np.array([KNEE_Y, H]), np.array([-YO, H_FRONT])
    t = (e - k) / np.linalg.norm(e - k)
    n = np.array([-t[1], t[0]])                                  # inward normal of the sloped edge

    def inside(top, slope):       # part of P at least `top` below the top edge and `slope` inside the sloped edge
        a, b = k + slope * n - 400 * t, k + slope * n + 400 * t
        return P & box(-HY, 0, HY, H - top) & Polygon([a, b, b + 800 * n, a + 800 * n])

    under_handle = lambda c: HANDLE[0] < c < HANDLE[2]
    handle = rrect(*HANDLE, r=8)
    hole = rrect(HANDLE[0] + 4, HANDLE[1] + 4, HANDLE[2] - 4, HANDLE[3] - 4, r=4)
    cols = [MULLIONS[0] + 8.5 + SIDE_PITCH * i for i in range(3)]
    cols += [m - 8.5 - SIDE_PITCH * i for m in MULLIONS for i in range(7)]
    keep = inside(RIM + 1.5, RIM_SLOPE + 1.5) - box(HANDLE[0] - 1.2, HANDLE[1] - 1.2, HANDLE[2] + 1.2, H)
    keep -= box(YO - 21, 102, HY, H)                              # the latch pocket shortens the last upper slot
    gaps = unary_union([box(c - GROOVE / 2, z0, c + GROOVE / 2, HANDLE[1] if under_handle(c) else H) for c in MULLIONS])
    posts = box(YO - 12, z0, YO, H) | P & box(-HY, 0, -YO + 11.5, H)
    pockets = box(YO - 10, 92, YO - 1.5, H - RIM) | box(-YO + 1.5, 33, -YO + 9, 44)   # latch pockets in the post faces

    parts = solid(P - inside(RIM, RIM_SLOPE), XI, XO, 'x')                        # rim
    parts += solid(posts, XI, XO - 7, 'x') + solid(posts - pockets, XO - 8.5, XO, 'x')
    parts += solid(P - slots(cols, SIDE_ROWS, SIDE_SLOT, keep) - gaps - handle.buffer(-1.5), xp - PLATE, xp, 'x')
    parts += solid(box(-YO, z0, YO, 36), FLOOR_W / 2 + 0.5, XO - 1, 'x')          # bottom (hinge) rail
    parts += solid(handle - hole, XI, HX, 'x')                                    # handle tube, proud of the wall
    for c in MULLIONS:                                                            # U-channel pilasters, drawn in (x, y)
        u = box(XI, c - 4.8, xp, c + 4.8) - box(xp - 4, c - GROOVE / 2, HX, c + GROOVE / 2)
        parts += solid(u, z0, HANDLE[1] + 1 if under_handle(c) else H - RIM + 1)
    return parts


def end_wall(top, depth, caps, cols, rows):
    """Back wall (also the low front wall, mirrored), drawn in (x, z). caps = (y_lo, y_hi, z_lo) top rim / ledge boxes."""
    z0, yp = BAND_END - 0.5, YO - 1.5
    cols = [s * c for c in cols for s in ((1, -1) if c else (1,))]
    gaps = box(END_MULL - GROOVE / 2, z0, END_MULL + GROOVE / 2, top)
    parts = [m for y0, y1, zc in caps for m in solid(box(-BW, zc, BW, top), y0, y1, 'y')]
    parts += solid(box(-BW, 20, BW, top - 1) - slots(cols, rows, END_SLOT) - flip(gaps, -1, 1), yp - PLATE, yp, 'y')
    parts += solid(box(-BW, z0, BW, 22), YO - 9, YO, 'y')                         # bottom (hinge) rail
    parts += solid(flip(box(BW - 1.5, z0, BW, top - 1), -1, 1), YO - depth, YO, 'y')   # end stiles
    u = box(END_MULL - 5, YO - depth, END_MULL + 5, yp) - box(END_MULL - GROOVE / 2, yp - 4, END_MULL + GROOVE / 2, HY)
    return parts + solid(flip(u, -1, 1), z0, top - 1)


def body():
    side = side_wall()
    back = end_wall(H, T_END, [(YO - T_END, YO, H - 3.5)], BACK_COLS, BACK_ROWS)
    ledge = [(YO - FRONT_D, YO, H_FRONT - 2.5), (YO - 1.5, YO, H_FRONT - 4.5), (YO - FRONT_D, YO - FRONT_D + 2, H_FRONT - 6)]
    front = mirror(end_wall(H_FRONT, FRONT_D, ledge, FRONT_COLS, FRONT_ROWS), 1)
    return side + mirror(side, 0) + back + front + floor()


def tray():
    hw, hl = FLOOR_W / 2, FLOOR_L / 2
    foot = rrect(-HX, -HY, HX, HY, r=2.5)
    ends = foot - box(-HX - 1, -hl, HX + 1, hl)
    px, w, d = LEDGE_HOLES
    holes = flip(box(px - w / 2, hl + 1, px + w / 2, hl + 1 + d), -1, 1)
    sides = foot - box(-hw, -HY - 1, hw, HY + 1) | ends - box(-BW - 0.25, -HY - 1, BW + 0.25, HY + 1)   # bands + corner blocks
    return solid(sides, 0, BAND_SIDE) + solid(ends - holes, 0, BAND_END)


# ---------------- mesh -> USD ----------------
def corner_normals(v, f):
    """Area-weighted normals, smoothed only across faces closer than SMOOTH_DEG: flats stay flat, bevels blend."""
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])          # length = 2 x area
    unit = fn / np.linalg.norm(fn, axis=1, keepdims=True)
    vid, fid = f.ravel(), np.repeat(np.arange(len(f)), 3)
    order = np.argsort(vid, kind='stable')
    rank = np.arange(len(vid)) - np.searchsorted(vid[order], vid[order])    # index of a face within its vertex fan
    fan = np.full((len(v), rank.max() + 1), -1)
    fan[vid[order], rank] = fid[order]
    n = np.zeros((len(vid), 3))
    for nb in fan[vid].T:                                                   # faces around each corner's vertex
        n += fn[nb] * ((nb >= 0) & ((unit[nb] * unit[fid]).sum(1) > np.cos(np.radians(SMOOTH_DEG))))[:, None]
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def material(stage, path, srgb, rough):
    lin = Gf.Vec3f(*[((c / 255 + 0.055) / 1.055) ** 2.4 for c in srgb])       # OmniPBR / PreviewSurface want linear
    mat = UsdShade.Material.Define(stage, path)
    mdl = UsdShade.Shader.Define(stage, path + '/Shader')
    mdl.SetSourceAsset(Sdf.AssetPath('OmniPBR.mdl'), 'mdl')
    mdl.SetSourceAssetSubIdentifier('OmniPBR', 'mdl')
    mdl.CreateInput('diffuse_color_constant', Sdf.ValueTypeNames.Color3f).Set(lin)
    mdl.CreateInput('reflection_roughness_constant', Sdf.ValueTypeNames.Float).Set(rough)
    mdl.CreateInput('metallic_constant', Sdf.ValueTypeNames.Float).Set(0.0)
    out = mdl.CreateOutput('out', Sdf.ValueTypeNames.Token)
    for o in ('surface', 'volume', 'displacement'):
        mat.CreateOutput('mdl:' + o, Sdf.ValueTypeNames.Token).ConnectToSource(out)
    pv = UsdShade.Shader.Define(stage, path + '/PreviewSurface')
    pv.CreateIdAttr('UsdPreviewSurface')
    pv.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(lin)
    pv.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(rough)
    pv.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(pv.CreateOutput('surface', Sdf.ValueTypeNames.Token))
    return mat


def write_usd(meshes):
    v3 = lambda a: Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(a, np.float32))
    ints = lambda a: Vt.IntArray.FromNumpy(np.ascontiguousarray(a, np.int32))
    os.makedirs(os.path.dirname(OUT_USD), exist_ok=True)
    if os.path.exists(OUT_USD):
        os.remove(OUT_USD)
    stage = Usd.Stage.CreateNew(OUT_USD)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, '/Basket').GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(MASS_KG)
    UsdGeom.Xform.Define(stage, '/Basket/visuals')
    UsdGeom.Scope.Define(stage, '/Basket/Looks')
    for name, (mat, srgb, rough) in MESHES.items():
        v, f = meshes[name]
        f = f[np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1) > 1e-6]   # zero-area tris
        normals, v = corner_normals(v, f), v * 1e-3
        mesh = UsdGeom.Mesh.Define(stage, f'/Basket/visuals/{name}')
        mesh.CreatePointsAttr(v3(v))
        mesh.CreateFaceVertexCountsAttr(ints(np.full(len(f), 3)))
        mesh.CreateFaceVertexIndicesAttr(ints(f.ravel()))
        mesh.CreateNormalsAttr(v3(normals))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateExtentAttr(v3([v.min(0), v.max(0)]))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material(stage, f'/Basket/Looks/{mat}', srgb, rough))

    # simple invisible colliders (the visual meshes carry no collision)
    UsdGeom.Xform.Define(stage, '/Basket/collisions').CreatePurposeAttr(UsdGeom.Tokens.guide)
    zf = FLOOR_Z[1]

    def collider(geom):
        geom.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        return geom.GetPrim()

    for name, lo, hi in (('floor', (-HX, -HY, 0), (HX, HY, zf)), ('back_wall', (-XI, YO - T_END, zf), (XI, HY, H)),
                         ('front_wall', (-XI, -HY, zf), (XI, -HY + INSET + FRONT_D, H_FRONT))):
        cube = UsdGeom.Cube.Define(stage, f'/Basket/collisions/{name}')
        lo, hi = np.array(lo) * 1e-3, np.array(hi) * 1e-3
        cube.CreateSizeAttr(1.0)
        cube.CreateExtentAttr([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
        cube.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2).tolist()))
        cube.AddScaleOp().Set(Gf.Vec3f(*(hi - lo).tolist()))
        collider(cube)
    prof = [(HY, zf), (HY, H), (KNEE_Y, H), (-HY, H_FRONT), (-HY, zf)]           # convex side profile (y, z)
    for name, xa, xb in (('side_wall_left', -HX, -XI), ('side_wall_right', XI, HX)):
        pts = np.array([(x, y, z) for x in (xa, xb) for y, z in prof]) * 1e-3
        prism = UsdGeom.Mesh.Define(stage, f'/Basket/collisions/{name}')
        prism.CreatePointsAttr(v3(pts))
        prism.CreateFaceVertexCountsAttr([5, 5] + [4] * 5)
        prism.CreateFaceVertexIndicesAttr([4, 3, 2, 1, 0, 5, 6, 7, 8, 9]
                                          + [j for i in range(5) for j in (i, (i + 1) % 5, (i + 1) % 5 + 5, i + 5)])
        prism.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        prism.CreateExtentAttr(v3([pts.min(0), pts.max(0)]))
        UsdPhysics.MeshCollisionAPI.Apply(collider(prism)).CreateApproximationAttr('convexHull')
    stage.Save()


def check():
    """Re-open the USD and assert the contract the scene relies on."""
    stage = Usd.Stage.Open(OUT_USD)
    root = stage.GetDefaultPrim()
    assert root.GetPath() == '/Basket' and UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdGeom.GetStageUpAxis(stage) == 'Z'
    box3 = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render']).ComputeWorldBound(root).ComputeAlignedRange()
    size, zmin = np.array(box3.GetSize()), box3.GetMin()[2]
    assert np.allclose(size, np.array([LX, LY, H]) * 1e-3, atol=5e-4) and abs(zmin) < 1e-6, (size, zmin)
    assert root.HasAPI(UsdPhysics.RigidBodyAPI) and abs(UsdPhysics.MassAPI(root).GetMassAttr().Get() - MASS_KG) < 1e-6
    tris = 0
    for name, (mat, _, _) in MESHES.items():
        prim = stage.GetPrimAtPath(f'/Basket/visuals/{name}')
        assert not prim.HasAPI(UsdPhysics.CollisionAPI)
        bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        assert bound.GetPath() == f'/Basket/Looks/{mat}' and bound.GetOutput('mdl:surface').HasConnectedSource()
        assert bound.GetSurfaceOutput().HasConnectedSource()
        tris += len(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get())
        assert np.allclose(np.linalg.norm(np.array(UsdGeom.Mesh(prim).GetNormalsAttr().Get()), axis=1), 1, atol=1e-3), name
    p = np.array(UsdGeom.Mesh(stage.GetPrimAtPath('/Basket/visuals/body')).GetPointsAttr().Get())
    front = p[(p[:, 1] < -0.14) & (abs(p[:, 0]) < BW * 1e-3 + 1e-5), 2].max()      # top of the low front wall
    assert abs(front - H_FRONT * 1e-3) < 5e-4, front
    cols = stage.GetPrimAtPath('/Basket/collisions').GetChildren()
    assert len(cols) == 5 and all(c.HasAPI(UsdPhysics.CollisionAPI) for c in cols) and tris < 400_000
    print(f'OK {OUT_USD}\n   bbox {size.round(4)} m, z-min {zmin:.4f}, front {front:.4f} m, {tris} triangles, '
          f'{len(cols)} colliders, mass {MASS_KG} kg')


# ---------------- Blender: preview renders of the USD ----------------
def render(usd):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=usd)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'                      # CPU on purpose: the GPU is busy with Isaac Sim
    scene.cycles.samples, scene.cycles.use_denoising = 64, True
    scene.render.resolution_x, scene.render.resolution_y = 1071, 1428
    scene.view_settings.view_transform = 'Standard'
    scene.world = bpy.data.worlds.new('world')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs[1].default_value = 0.3

    def add(name, data, location=(0, 0, 0)):
        ob = bpy.data.objects.new(name, data)
        ob.location = location
        scene.collection.objects.link(ob)
        return ob

    table = bpy.data.meshes.new('table')
    table.from_pydata([(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)], [], [(0, 1, 2, 3)])
    table.materials.append(bpy.data.materials.new('table'))
    table.materials[0].use_nodes = True
    table.materials[0].node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.03, 0.03, 0.035, 1)
    add('table', table)
    lamp = bpy.data.lights.new('lamp', 'AREA')
    lamp.energy, lamp.size = 45, 1.2
    add('lamp', lamp, (0.2, -0.2, 1.5))
    cam = bpy.data.cameras.new('cam')
    cam.sensor_fit, cam.sensor_width, cam.lens, cam.clip_start = 'HORIZONTAL', 36, 36 * 5280 / 4284, 0.01
    scene.camera = add('cam', cam)
    scene.camera.rotation_mode = 'QUATERNION'
    for name, (pos, quat) in CAMS.items():
        scene.camera.location, scene.camera.rotation_quaternion = pos, quat
        scene.render.filepath = os.path.join(RENDER_DIR, f'basket_preview_{name}.png')
        bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    if STAGE == 'render':
        render(sys.argv[2])
    else:
        union = lambda parts: trimesh.boolean.union(parts, engine='manifold')
        write_usd({name: (np.asarray(m.vertices), np.asarray(m.faces)) for name, m in
                   (('body', union(body())), ('base_tray', union(tray())))})
        check()
        subprocess.run([BLENDER_PY, __file__, 'render', os.path.abspath(OUT_USD)], check=True)
