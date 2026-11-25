"""Build metric, portable OpenUSD assets from the four room photographs.

Run with Python containing pxr, numpy and Pillow. No Kit startup needed.
Geometry is reconstructed, not photogrammetrically recovered.
"""
import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

LARGE = (0.345, 0.230, 0.170)
SMALL = (0.180, 0.180, 0.100)


def attr(prim, name, value, kind=Sdf.ValueTypeNames.Float):
    prim.CreateAttribute(name, kind, custom=False).Set(value)


def physx(prim, api, values):
    prim.AddAppliedSchema(api)
    for name, value in values.items():
        kind = (Sdf.ValueTypeNames.Bool if isinstance(value, bool) else
                Sdf.ValueTypeNames.Int if isinstance(value, int) else
                Sdf.ValueTypeNames.Token if isinstance(value, str) else Sdf.ValueTypeNames.Float)
        attr(prim, name, value, kind)


def xform(stage, path, xyz=(0, 0, 0), yaw=0):
    p = UsdGeom.Xform.Define(stage, path).GetPrim()
    xf = UsdGeom.Xformable(p)
    xf.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    xf.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw).GetQuat()))
    return p


def rigid(p, mass, size):
    rb = UsdPhysics.RigidBodyAPI.Apply(p)
    rb.CreateRigidBodyEnabledAttr(True)
    rb.CreateKinematicEnabledAttr(False)
    m = UsdPhysics.MassAPI.Apply(p)
    m.CreateMassAttr(mass)
    m.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
    x, y, z = size
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(mass*(y*y+z*z)/12,
                                        mass*(x*x+z*z)/12, mass*(x*x+y*y)/12))
    physx(p, "PhysxRigidBodyAPI", {
        "physxRigidBody:solverPositionIterationCount": 32,
        "physxRigidBody:solverVelocityIterationCount": 8,
        "physxRigidBody:enableCCD": True,
        "physxRigidBody:linearDamping": 0.02,
        "physxRigidBody:angularDamping": 0.04,
    })


def collide(p, material=None, mesh=False):
    UsdPhysics.CollisionAPI.Apply(p)
    if mesh:
        UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexHull")
    physx(p, "PhysxCollisionAPI", {"physxCollision:contactOffset": 0.001,
                                 "physxCollision:restOffset": 0.0})
    if material:
        UsdShade.MaterialBindingAPI.Apply(p).Bind(material, materialPurpose="physics")


def textures(out):
    """Deterministic physical-material maps; no baked photographic lighting."""
    rng = np.random.default_rng(2417)
    n = 1024
    fine = rng.normal(0, 1, (n, n))
    broad = (np.array(Image.fromarray(np.uint8(np.clip(127+rng.normal(0,30,(32,32)),0,255))).resize((n,n),Image.Resampling.BILINEAR)).astype(float)-127)/30
    noise_im = Image.fromarray(np.uint8(np.clip(127+fine*30,0,255)))
    wave = (np.array(noise_im.filter(ImageFilter.GaussianBlur(3))).astype(float)-127)/30
    configs = {"kraft": (178, 131, 85), "small_kraft": (204, 178, 126),
               "green": (14, 87, 71), "zinc": (157, 166, 165),
               "tape": (158, 111, 66), "small_tape": (185, 157, 106)}
    for name, base in configs.items():
        amp = 9 if name == "zinc" else 2.8
        noise = fine*1.9 + broad*amp + wave*12
        if name == "zinc":
            noise = wave*10 + np.round(broad*3)*3
        arr = np.clip(np.array(base) + noise[:, :, None], 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(out / "textures" / f"{name}.png")
    # Fine paper fibres / orange-peel plastic texture, encoded as a tangent normal.
    height = (np.array(noise_im.filter(ImageFilter.GaussianBlur(.65))).astype(float)-127)/30
    dy, dx = np.gradient(height)
    normals = np.stack((-dx*.065, -dy*.065, np.ones_like(dx)), axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    Image.fromarray(np.uint8(np.clip(normals*.5+.5, 0, 1)*255)).save(out/"textures/paper_normal.png")
    # Black rubber tiles with sparse beige and white chips, matching the room floor.
    arr = np.clip(24 + fine*3, 0, 255).astype(np.uint8)
    im = Image.fromarray(np.repeat(arr[:, :, None], 3, axis=2))
    draw = ImageDraw.Draw(im)
    for _ in range(1700):
        x, y = rng.integers(0, n, 2)
        r = float(rng.uniform(.9, 3.3))
        color = (180, 172, 147) if rng.random() < .5 else (206, 204, 195)
        draw.polygon([(x-r,y), (x,y-r*.6), (x+r*.8,y+r*.5), (x-r*.4,y+r)], fill=color)
    im.save(out/"textures/rubber_floor.png")


def material(stage, path, color, rough=.65, metal=0, tex=None, normal=False):
    m = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path+"/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    if tex:
        uv = UsdShade.Shader.Define(stage, path+"/UV")
        uv.CreateIdAttr("UsdPrimvarReader_float2")
        uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        for label, filename, dest, raw in [("Color", tex, "diffuseColor", False)] + (
                [("Normal", "paper_normal", "normal", True)] if normal else []):
            t = UsdShade.Shader.Define(stage, path+"/"+label)
            t.CreateIdAttr("UsdUVTexture")
            t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("textures/"+filename+".png"))
            t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw" if raw else "sRGB")
            t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
            if raw:
                t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2,2,2,1))
                t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1,-1,-1,0))
            sh.CreateInput(dest, Sdf.ValueTypeNames.Normal3f if raw else Sdf.ValueTypeNames.Color3f).ConnectToSource(t.ConnectableAPI(), "rgb")
    return m


def mesh(stage, path, points, faces, mat, uv=None):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    m.CreateFaceVertexCountsAttr([len(f) for f in faces])
    m.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateExtentAttr([Gf.Vec3f(*np.min(points, axis=0)),Gf.Vec3f(*np.max(points, axis=0))])
    if uv is None:
        uv = []
        pts = np.array(points)
        for face in faces:
            verts = pts[face]
            normal = np.cross(verts[1]-verts[0], verts[2]-verts[0])
            axes = [i for i in range(3) if i != np.argmax(abs(normal))]
            uv.extend([tuple(p[axes]*3.5) for p in verts])
    UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                      UsdGeom.Tokens.faceVarying).Set(uv)
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    return m.GetPrim()


def block(stage, path, xyz, size, mat, bevel=0, collision=False):
    half = np.array(size)/2
    if bevel:
        pts = []
        for axis in range(3):
            for signs in itertools.product((-1,1), repeat=3):
                p = (half-bevel)*signs
                p[axis] = half[axis]*signs[axis]
                pts.append(p)
        pts = np.array(pts)
        faces = []
        for direction in itertools.product((-1,0,1), repeat=3):
            normal = np.array(direction,dtype=float)
            if not np.any(normal):
                continue
            dots = pts@normal
            ids = np.flatnonzero(np.isclose(dots,dots.max(),atol=1e-8,rtol=0))
            if len(ids) < 3:
                continue
            normal /= np.linalg.norm(normal)
            u = pts[ids[0]]-pts[ids].mean(axis=0)
            u /= np.linalg.norm(u)
            v = np.cross(normal,u)
            centered = pts[ids]-pts[ids].mean(axis=0)
            faces.append(ids[np.argsort(np.arctan2(centered@v,centered@u))].tolist())
    else:
        pts = np.array(list(itertools.product((-1,1),repeat=3)))*half
        faces = [[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6],[0,2,6,4],[1,5,7,3]]
    p = mesh(stage,path,pts+np.array(xyz),faces,mat)
    if collision:
        proxy=UsdGeom.Cube.Define(stage,path+"_Collider")
        proxy.CreateSizeAttr(1)
        xf=UsdGeom.Xformable(proxy)
        xf.AddTranslateOp().Set(Gf.Vec3d(*xyz))
        xf.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdGeom.Imageable(proxy).MakeInvisible()
        collide(proxy.GetPrim(),mat)
    return p


def rod(stage, path, a, b, radius, mat, collision=False):
    c = UsdGeom.Cylinder.Define(stage, path)
    a,b = Gf.Vec3d(*a),Gf.Vec3d(*b)
    c.CreateRadiusAttr(radius)
    c.CreateHeightAttr((b-a).GetLength())
    xf = UsdGeom.Xformable(c)
    xf.AddTranslateOp().Set((a+b)/2)
    xf.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0,0,1),(b-a).GetNormalized()).GetQuat()))
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
    if collision:
        collide(c.GetPrim())
    return c.GetPrim()


def bent_tube(stage,path,centers,radius,mat):
    centers=np.array(centers)
    points=[];normals=[];faces=[]
    for i,center in enumerate(centers):
        t=centers[min(i+1,len(centers)-1)]-centers[max(i-1,0)]
        t/=np.linalg.norm(t)
        u=np.array([1.,0,0]);v=np.cross(t,u)
        for theta in np.linspace(0,2*np.pi,16,endpoint=False):
            normal=u*np.cos(theta)+v*np.sin(theta)
            points.append(center+radius*normal);normals.append(normal)
    for i in range(len(centers)-1):
        for j in range(16):
            k=(j+1)%16
            faces.append([i*16+j,i*16+k,(i+1)*16+k,(i+1)*16+j])
    p=mesh(stage,path,points,faces,mat)
    m=UsdGeom.Mesh(p)
    m.CreateNormalsAttr([Gf.Vec3f(*n) for n in normals])
    m.SetNormalsInterpolation("vertex")


def init_stage(path, root):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage,1)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage,root).GetPrim())
    return stage


def box_asset(out, name, size, mass, small=False):
    stage = init_stage(out/f"{name}.usd", "/Box")
    root = stage.GetDefaultPrim()
    rigid(root,mass,size)
    root.SetCustomData({"class":"small_box" if small else "large_box", "dimensions_m":list(size),
                        "mass_is_estimate":True, "source":"user_dimensions"})
    mat = material(stage,"/Box/Looks/Kraft",(.4,.25,.12),tex="small_kraft" if small else "kraft",normal=True)
    tape = material(stage,"/Box/Looks/Tape",(.45,.3,.13),rough=.36,tex="small_tape" if small else "tape")
    seam = material(stage,"/Box/Looks/Seam",(.10,.055,.02),rough=.95)
    contact = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    contact.CreateStaticFrictionAttr(.58)
    contact.CreateDynamicFrictionAttr(.44)
    contact.CreateRestitutionAttr(.015)
    block(stage,"/Box/Body",(0,0,0),size,mat,bevel=.0011,collision=True)
    x,y,z = size
    # Sealed flaps and packing tape wrap over both short ends; no extra collision shells.
    block(stage,"/Box/TopSeam",(0,0,z/2+.00003),(x-.003,.0006,.00008),seam)
    block(stage,"/Box/TapeTop",(0,0,z/2+.00010),(x-.002,.047,.00012),tape)
    for side in (-1,1):
        block(stage,f"/Box/TapeEnd_{side+1}",(side*(x/2+.00008),0,z*.25),(.0001,.047,z*.5-.002),tape)
        block(stage,f"/Box/LidLip_{side+1}",(0,side*(y/2+.00002),z/2-.002),(x-.004,.00008,.00045),seam)
    stage.GetRootLayer().Save()


def pallet_asset(out):
    stage = init_stage(out/"green_pallet.usd","/Pallet")
    rigid(stage.GetDefaultPrim(),12,(1.1,1.1,.15))
    UsdPhysics.MassAPI(stage.GetDefaultPrim()).GetCenterOfMassAttr().Set(Gf.Vec3f(0,0,.070))
    green = material(stage,"/Pallet/Looks/Green",(.012,.11,.075),rough=.47,tex="green",normal=True)
    contact = UsdPhysics.MaterialAPI.Apply(green.GetPrim())
    contact.CreateStaticFrictionAttr(.56)
    contact.CreateDynamicFrictionAttr(.42)
    contact.CreateRestitutionAttr(.01)
    black = material(stage,"/Pallet/Looks/Grip",(.019,.024,.021),rough=.9)
    # Three bottom runners, nine hollow feet, and a genuinely open lattice deck.
    for i,x in enumerate((-.462,0,.462)):
        block(stage,f"/Pallet/Runner{i}",(x,0,.012),(.176,1.10,.024),green,.006,True)
        for j,y in enumerate((-.462,0,.462)):
            for k,(dx,dy,sx,sy) in enumerate(((0,-.066,.166,.034),(0,.066,.166,.034),(-.066,0,.034,.1),(.066,0,.034,.1))):
                block(stage,f"/Pallet/Foot{i}{j}_{k}",(x+dx,y+dy,.078),(sx,sy,.108),green,.004,True)
    for side in (-1,1):
        block(stage,f"/Pallet/RimX{side+1}",(0,side*.531,.1375),(1.10,.038,.025),green,.006,True)
        block(stage,f"/Pallet/RimY{side+1}",(side*.531,0,.1375),(.038,1.024,.025),green,.006,True)
    for i,t in enumerate(np.linspace(-.49,.49,29)):
        block(stage,f"/Pallet/GridX{i}",(0,float(t),.135),(1.024,.004,.030),green,collision=True)
        block(stage,f"/Pallet/GridY{i}",(float(t),0,.135),(.004,1.024,.030),green,collision=True)
    for i,(x,y) in enumerate(itertools.product((-.462,0,.462),repeat=2)):
        block(stage,f"/Pallet/Pad{i}",(x,y,.139),(.083,.083,.022),green,.002,True)
        rod(stage,f"/Pallet/Grip{i}",(x,y,.149),(x,y,.15),.014,black)
    stage.GetDefaultPrim().SetCustomData({"dimensions_m":[1.1,1.1,.15],"mass_is_estimate":True,"class":"pallet"})
    stage.GetRootLayer().Save()


def joint(stage, path, a, b, pa, pb, axis, damping):
    j = UsdPhysics.RevoluteJoint.Define(stage,path)
    j.CreateBody0Rel().SetTargets([a])
    j.CreateBody1Rel().SetTargets([b])
    j.CreateLocalPos0Attr(Gf.Vec3f(*pa))
    j.CreateLocalPos1Attr(Gf.Vec3f(*pb))
    j.CreateAxisAttr(axis)
    j.CreateCollisionEnabledAttr(False)
    d = UsdPhysics.DriveAPI.Apply(j.GetPrim(),"angular")
    d.CreateStiffnessAttr(0)
    d.CreateDampingAttr(damping)
    return j


def rolltainer_asset(out):
    stage = init_stage(out/"silver_rolltainer.usd","/Rolltainer")
    metal = material(stage,"/Rolltainer/Looks/Zinc",(.55,.59,.59),rough=.46,metal=.85,tex="zinc")
    rubber = material(stage,"/Rolltainer/Looks/Rubber",(.018,.020,.019),rough=.85)
    tape = material(stage,"/Rolltainer/Looks/GreenTape",(.005,.44,.26),rough=.39)
    strap = material(stage,"/Rolltainer/Looks/Strap",(.012,.014,.012),rough=.9)
    for mat,fr in ((metal,.42),(rubber,.8)):
        pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pm.CreateStaticFrictionAttr(fr)
        pm.CreateDynamicFrictionAttr(fr*.8)
        pm.CreateRestitutionAttr(.01)
    chassis = xform(stage,"/Rolltainer/Chassis")
    rigid(chassis,32,(.92,.70,1.55))
    UsdPhysics.MassAPI(chassis).GetCenterOfMassAttr().Set(Gf.Vec3f(0,0,.65))
    block(stage,"/Rolltainer/Chassis/Deck",(0,0,.157),(.92,.70,.030),metal,.004,True)
    for i,x in enumerate((-.37,.37)):
        block(stage,f"/Rolltainer/Chassis/UnderRail{i}",(x,0,.14),(.05,.65,.014),metal,.002,True)
    for side,x in enumerate((-.444,.444)):
        points = [(x,-.334,.174),(x,-.334,1.609)]
        points += [(x,-.259-.075*math.cos(t),1.609+.075*math.sin(t)) for t in np.linspace(0,math.pi/2,9)[1:]]
        points += [(x,.259,1.684)]
        points += [(x,.259+.075*math.sin(t),1.609+.075*math.cos(t)) for t in np.linspace(0,math.pi/2,9)[1:]]
        points += [(x,.334,.174)]
        for i,(a,b) in enumerate(zip(points,points[1:])):
            p=rod(stage,f"/Rolltainer/Chassis/Side{side}_Tube{i}",a,b,.016,metal,True)
            UsdGeom.Imageable(p).MakeInvisible()
        bent_tube(stage,f"/Rolltainer/Chassis/Side{side}_BentTube",points,.016,metal)
        for i,y in enumerate((-.19,-.055,.085,.22)):
            rod(stage,f"/Rolltainer/Chassis/Side{side}_Bar{i}",(x,y,.18),(x,y,1.663),.007,metal,True)
        for i,z in enumerate((.68,1.02)):
            block(stage,f"/Rolltainer/Chassis/Side{side}_Cross{i}",(x,0,z),(.013,.654,.028),metal,.001,True)
        # The photographed green tape marks and black retaining webbing.
        block(stage,f"/Rolltainer/Chassis/Marker{side}",(x+(.002 if not side else -.002),-.329,1.42),(.035,.036,.061),tape,.001)
        for i,z in enumerate((.53,1.13)):
            block(stage,f"/Rolltainer/Chassis/Strap{side}_{i}",(x+(.010 if side else -.010),0,z),(.003,.67,.035),strap)
        sx=x+(.005 if not side else -.005)
        rod(stage,f"/Rolltainer/Chassis/Sleeve{side}",(sx,-.329,.88),(sx,-.329,.945),.021,metal)
    rod(stage,"/Rolltainer/Chassis/BackTop",(-.428,.333,1.68),(.428,.333,1.68),.012,metal,True)
    block(stage,"/Rolltainer/Chassis/BackPlate",(0,.342,1.095),(.85,.006,.39),metal,.001,True)
    for i,x in enumerate(np.linspace(-.32,.32,6)):
        rod(stage,f"/Rolltainer/Chassis/BackBar{i}",(float(x),.343,.18),(float(x),.343,1.67),.006,metal,True)
    for i,(x,y) in enumerate(itertools.product((-.365,.365),(-.245,.245))):
        forkpath=f"/Rolltainer/Caster{i}"
        fork=xform(stage,forkpath,(x,y,.108))
        rigid(fork,.6,(.07,.065,.065))
        rod(stage,forkpath+"/Bearing",(0,0,-.005),(0,0,.022),.025,metal)
        for k,side in enumerate((-1,1)):
            block(stage,forkpath+f"/Cheek{k}",(side*.024,-.01,-.022),(.006,.052,.049),metal,.002)
        wheelpath=f"/Rolltainer/Wheel{i}"
        wheel=xform(stage,wheelpath,(x,y-.020,.063))
        rigid(wheel,.7,(.035,.126,.126))
        rim=rod(stage,wheelpath+"/Tire",(-.0175,0,0),(.0175,0,0),.063,rubber,True)
        for side in (-1,1):
            rod(stage,wheelpath+f"/Hub{side+1}",(side*.0176,0,0),(side*.019,0,0),.032,metal)
        joint(stage,f"/Rolltainer/Swivel{i}",str(chassis.GetPath()),forkpath,(x,y,.12),(0,0,.012),"Z",.06)
        joint(stage,f"/Rolltainer/Axle{i}",forkpath,wheelpath,(0,-.020,-.045),(0,0,0),"X",.004)
    stage.GetDefaultPrim().SetCustomData({"class":"rolltainer","dimensions_m":[.92,.70,1.7],
        "caster_construction_is_estimate":True,"mass_is_estimate":True})
    stage.GetRootLayer().Save()


def placement_layout():
    boxes=[]
    def add(name,asset,xyz,yaw,support,evidence):
        boxes.append(dict(name=name,asset=asset,position_m=list(xyz),yaw_deg=yaw,
                          support=support,evidence=evidence))
    # X = photograph-right, Y = toward wall, Z = up. Pallet front is Y=-0.55.
    for col,height in enumerate((4,4,3,1)):
        for level in range(height):
            add(f"Pallet_C{col+1}_L{level+1}","large_box",(.67-.402+col*.233,-.349,.15+.085+level*.1704),90,"pallet","visible")
    add("Pallet_Cap","large_box",(.67-.286,-.349,.15+4*.1704+.085),0,"pallet","visible_rotated_cap")
    # Rolltainer front row is visible. Its occluded back row is a declared completion.
    for row in range(2):
        for col in range(3):
            for level in range(3):
                add(f"Roll_C{col+1}_R{row+1}_L{level+1}","large_box",(-.43-.233+col*.233,-.184+row*.346,.172+.085+level*.1704),90,"rolltainer","visible" if row==0 else "inferred_occluded")
    z=.172+3*.1704+.05
    for level,(dx,dy,yaw) in enumerate(((0,0,-7),(-.006,.003,3),(.008,.006,-5))):
        add(f"Small_Center_L{level+1}","small_box",(-.45+dx,-.149+dy,z+level*.1004),yaw,"rolltainer","visible")
    add("Small_RightFront","small_box",(-.195,-.084,z),-7,"rolltainer","visible")
    for level in range(2):
        add(f"Small_RightBack_L{level+1}","small_box",(-.205,.156,z+level*.1004),2,"rolltainer","visible_partially_occluded")
    add("Small_LeftBack","small_box",(-.70,.111,z),-4,"rolltainer","visible_partially_occluded")
    return {"units":"metres", "axes":"X photo-right, Y toward wall, Z up", "boxes":boxes,
            "pallet_position_m":[.67,0,0],"rolltainer_position_m":[-.43,0,0],
            "hidden_layout_confirmed":False,"large_box_mass_kg":.55,"small_box_mass_kg":.16}


def camera(stage,name,eye,target,focal=28):
    cam=UsdGeom.Camera.Define(stage,"/World/Cameras/"+name)
    m=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse()
    UsdGeom.Xformable(cam).AddTransformOp().Set(m)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateClippingRangeAttr(Gf.Vec2f(.01,100))
    cam.CreateHorizontalApertureAttr(36)
    return str(cam.GetPath())


def loaded_asset(out,layout,support):
    stage=init_stage(out/(support+"_loaded.usd"),"/Assembly")
    name="green_pallet" if support=="pallet" else "silver_rolltainer"
    stage.DefinePrim("/Assembly/Support","Xform").GetReferences().AddReference(name+".usd")
    origin=np.array(layout[support+"_position_m"])
    for b in layout["boxes"]:
        if b["support"]==support:
            p=xform(stage,"/Assembly/Boxes/"+b["name"],np.array(b["position_m"])-origin,b["yaw_deg"])
            p.GetReferences().AddReference(b["asset"]+".usd")
            p.SetCustomDataByKey("layout_evidence",b["evidence"])
            box_appearance(stage,p,b)
    stage.GetRootLayer().Save()


def box_appearance(stage,p,box):
    root=str(p.GetPath())
    # Small exposure-independent albedo variation; every carton keeps its own identity.
    seed=sum((i+1)*ord(c) for i,c in enumerate(box["name"]))
    shade=.95+(seed%11)*.01
    shader=UsdShade.Shader(stage.GetPrimAtPath(root+"/Looks/Kraft/Color"))
    shader.CreateInput("scale",Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(shade,shade,shade,1))
    if box["name"]=="Pallet_C3_L1":
        paper=material(stage,root+"/Looks/LabelRemnant",(.68,.66,.59),rough=.95)
        points=[(-.17265,-.049,-.025),(-.17265,-.070,-.029),(-.17265,-.068,-.038),
                (-.17265,-.054,-.035),(-.17265,-.050,-.030)]
        patch=mesh(stage,root+"/TornLabel",points,[[0,1,2,3,4]],paper)
        UsdGeom.Mesh(patch).CreateDoubleSidedAttr(True)


def scene(out,layout):
    stage=init_stage(out/"room_interactive.usd","/World")
    stage.SetTimeCodesPerSecond(60)
    ps=UsdPhysics.Scene.Define(stage,"/World/PhysicsScene")
    ps.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1))
    ps.CreateGravityMagnitudeAttr(9.81)
    physx(ps.GetPrim(),"PhysxSceneAPI",{"physxScene:timeStepsPerSecond":240,
        "physxScene:enableCCD":True,"physxScene:enableStabilization":True,
        "physxScene:solverType":"TGS","physxScene:enableGPUDynamics":False,
        "physxScene:broadphaseType":"MBP"})
    for name,filename,pos in (("Pallet","green_pallet",layout["pallet_position_m"]),
                              ("Rolltainer","silver_rolltainer",layout["rolltainer_position_m"])):
        p=xform(stage,"/World/"+name,pos)
        p.GetReferences().AddReference(filename+".usd")
    for b in layout["boxes"]:
        p=xform(stage,"/World/Boxes/"+b["name"],b["position_m"],b["yaw_deg"])
        p.GetReferences().AddReference(b["asset"]+".usd")
        p.SetCustomDataByKey("layout_evidence",b["evidence"])
        box_appearance(stage,p,b)
    floor=material(stage,"/World/Looks/Floor",(.02,.02,.02),rough=.83,tex="rubber_floor")
    wall=material(stage,"/World/Looks/Wall",(.73,.715,.675),rough=.85)
    black=material(stage,"/World/Looks/Skirting",(.014,.016,.015),rough=.8)
    for ix in range(-8,9):
        for iy in range(-8,2):
            p=block(stage,f"/World/Room/Tile_{ix+8}_{iy+8}",(ix*.5,iy*.5,-.019),(.498,.498,.038),floor)
            st=UsdGeom.PrimvarsAPI(p).GetPrimvar("st")
            st.Set([v*(2/3.5) for v in st.Get()])
    ground=block(stage,"/World/Room/GroundCollider",(0,-1.75,-.040),(8.5,5,.08),floor,collision=True)
    UsdGeom.Imageable(ground).MakeInvisible()
    block(stage,"/World/Room/Wall",(0,.735,1.5),(8,.06,3.0),wall,collision=True)
    block(stage,"/World/Room/Skirting",(0,.700,.043),(8,.011,.086),black)
    for i,x in enumerate((-1.5,0,1.5)):
        block(stage,f"/World/Room/WallJoint{i}",(x,.699,1.5),(.001,.001,3),wall)
    dome=UsdLux.DomeLight.Define(stage,"/World/Lights/Ambient")
    dome.CreateIntensityAttr(600)
    sun=UsdLux.DistantLight.Define(stage,"/World/Lights/SoftFill")
    sun.CreateIntensityAttr(650)
    sun.CreateAngleAttr(20)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-25,12,-35))
    for i,x in enumerate((-1,1)):
        light=UsdLux.RectLight.Define(stage,f"/World/Lights/Ceiling{i}")
        light.CreateIntensityAttr(1700)
        light.CreateWidthAttr(.30)
        light.CreateHeightAttr(1.15)
        light.CreateColorAttr(Gf.Vec3f(1,.974,.932))
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(x,-.3,2.85))
    fill=UsdLux.RectLight.Define(stage,"/World/Lights/RoomBounce")
    fill.CreateIntensityAttr(2800)
    fill.CreateWidthAttr(2.0)
    fill.CreateHeightAttr(1.5)
    fill.CreateColorAttr(Gf.Vec3f(1,.98,.95))
    UsdGeom.Xformable(fill).AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(-.3,-2.1,2.0),Gf.Vec3d(0,0,.65),Gf.Vec3d(0,0,1)).GetInverse())
    camera(stage,"Overview",(-1.65,-2.90,2.25),(.12,0,.8),32)
    camera(stage,"Pallet",(1.28,-2.5,1.85),(.67,-.21,.48),43)
    camera(stage,"Rolltainer",(-1.85,-2.30,2.05),(-.43,0,.83),39)
    camera(stage,"Overhead",(.12,-.001,3.3),(.12,0,0),29)
    stage.GetRootLayer().customLayerData={"cameraSettings":{"Perspective":{"position":Gf.Vec3d(-1.65,-2.90,2.25),
        "target":Gf.Vec3d(.12,0,.8)}},"renderSettings":{"renderer":"RayTracedLighting"},
        "physicsSettings":{"/physics/mouseInteractionEnabled":True,"/physics/mouseGrab":True,
                           "/physics/mouseGrabIgnoreInvisible":False}}
    stage.GetDefaultPrim().SetCustomData({"reconstruction":"photo_based_metric_reconstruction",
        "hidden_layout_confirmed":layout["hidden_layout_confirmed"],"mass_friction_measured":False})
    stage.GetRootLayer().Save()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--layout",type=Path,help="Explicit per-box poses; defaults to saved layout.json")
    args=parser.parse_args()
    out=args.out.resolve()
    (out/"textures").mkdir(parents=True,exist_ok=True)
    lp=args.layout or out/"layout.json"
    layout=json.loads(lp.read_text()) if lp.exists() else placement_layout()
    (out/"layout.json").write_text(json.dumps(layout,indent=2)+"\n")
    textures(out)
    box_asset(out,"large_box",LARGE,layout["large_box_mass_kg"])
    box_asset(out,"small_box",SMALL,layout["small_box_mass_kg"],True)
    pallet_asset(out)
    rolltainer_asset(out)
    for support in ("pallet","rolltainer"):
        loaded_asset(out,layout,support)
    scene(out,layout)
    print(json.dumps({"scene":str(out/"room_interactive.usd"),"boxes":len(layout["boxes"])}))


if __name__ == "__main__":
    main()
