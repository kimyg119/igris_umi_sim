"""Small offline contract check for the delivered scene (no Isaac Sim required)."""
import itertools
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdUtils


def check(root):
    dimensions={"large_box":(.345,.23,.17),"small_box":(.18,.18,.10),
                "green_pallet":(1.1,1.1,.15),"silver_rolltainer":(.92,.70,1.70)}
    measured={}
    for name,want in dimensions.items():
        s=Usd.Stage.Open(str(root/(name+".usd")))
        assert s and s.GetDefaultPrim(),name
        assert UsdGeom.GetStageMetersPerUnit(s)==1
        assert UsdGeom.GetStageUpAxis(s)=="Z"
        bb=UsdGeom.BBoxCache(Usd.TimeCode.Default(),["default","render"]).ComputeWorldBound(s.GetDefaultPrim()).ComputeAlignedRange()
        got=np.array(bb.GetSize())
        assert np.max(abs(got-want)) < .001,(name,got,want)
        measured[name]=got.tolist()
    path=root/"room_interactive.usd"
    s=Usd.Stage.Open(str(path))
    layers,assets,unresolved=UsdUtils.ComputeAllDependencies(str(path))
    assert not unresolved,unresolved
    bodies=[p for p in s.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    for p in bodies:
        assert not UsdPhysics.RigidBodyAPI(p).GetKinematicEnabledAttr().Get(),str(p.GetPath())
        assert UsdPhysics.MassAPI(p).GetMassAttr().Get()>0
        parent=p.GetParent()
        while parent and not parent.IsPseudoRoot():
            assert not parent.HasAPI(UsdPhysics.RigidBodyAPI),str(p.GetPath())
            parent=parent.GetParent()
    layout=json.loads((root/"layout.json").read_text())
    boxes=layout["boxes"]
    assert len(bodies)==len(boxes)+10,(len(bodies),len(boxes))
    assert sum(p.IsA(UsdPhysics.RevoluteJoint) for p in s.Traverse())==8
    # Separating axis check uses each carton dimension and yaw, not rendered pictures.
    overlaps=[]
    for a,b in itertools.combinations(boxes,2):
        da=np.array(dimensions[a["asset"]]);db=np.array(dimensions[b["asset"]])
        pa=np.array(a["position_m"]);pb=np.array(b["position_m"])
        if (da[2]+db[2])/2-abs(pa[2]-pb[2]) < .0005:
            continue
        ca,sa=np.cos(np.deg2rad(a["yaw_deg"])),np.sin(np.deg2rad(a["yaw_deg"]))
        cb,sb=np.cos(np.deg2rad(b["yaw_deg"])),np.sin(np.deg2rad(b["yaw_deg"]))
        aa=np.array([[ca,sa],[-sa,ca]]);ab=np.array([[cb,sb],[-sb,cb]])
        axes=np.concatenate((aa,ab))
        if all((abs(aa@axis)@da[:2]+abs(ab@axis)@db[:2])/2-abs((pa[:2]-pb[:2])@axis)>.0005 for axis in axes):
            overlaps.append((a["name"],b["name"]))
    assert not overlaps,overlaps
    report={"pass":True,"dimensions_m":measured,"box_count":len(boxes),
            "dynamic_body_count":len(bodies),"caster_joints":8,"penetrating_box_pairs":overlaps,
            "unresolved_assets":len(unresolved),"resolved_layer_count":len(layers),
            "hidden_layout_confirmed":layout["hidden_layout_confirmed"]}
    (root/"offline_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


def check_corner(root):
    original=Usd.Stage.Open(str(root/'room_interactive.usd'))
    stage=Usd.Stage.Open(str(root/'room_corner_igris.usd'))
    layout=json.loads((root/'layout.json').read_text())
    spec=json.loads((root/'corner_layout.json').read_text())
    old,new=UsdGeom.XformCache(),UsdGeom.XformCache()
    distances={}
    relative_error=0.
    corners={}
    for support,path in [('pallet','/World/Pallet'),('rolltainer','/World/Rolltainer')]:
        a=old.GetLocalToWorldTransform(original.GetPrimAtPath(path))
        b=new.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
        names=[p.GetPath() for p in Usd.PrimRange(original.GetPrimAtPath(path)) if p.IsA(UsdGeom.Xformable)]
        names += ['/World/Boxes/'+box['name'] for box in layout['boxes'] if box['support']==support]
        for name in names:
            before=old.GetLocalToWorldTransform(original.GetPrimAtPath(name))*a.GetInverse()
            after=new.GetLocalToWorldTransform(stage.GetPrimAtPath(name))*b.GetInverse()
            relative_error=max(relative_error,float(np.max(abs(np.array(before)-np.array(after)))))
        sx,sy=np.array(spec['supports'][support]['size_m'])/2
        edge=np.array([b.Transform(Gf.Vec3d(x,-sy,0)) for x in (-sx,sx)])[:,:2]
        direction=edge[1]-edge[0]
        t=-edge[0]@direction/(direction@direction)
        assert 0<t<1,(support,t)
        distances[support]=float(np.linalg.norm(edge[0]+t*direction))
        assert abs(distances[support]-.2)<1e-8
        corners[support]=np.array([b.Transform(Gf.Vec3d(x,y,0)) for x,y in [(-sx,-sy),(sx,-sy),(sx,sy),(-sx,sy)]])[:,:2]
    assert relative_error<1e-7,relative_error
    # The two footprints lie in opposite Y half-planes; each has exactly one Y=0 vertex.
    assert corners['pallet'][:,1].max()<1e-8 and corners['rolltainer'][:,1].min()>-1e-8
    contacts=[]
    for vertices in corners.values():
        touching=vertices[abs(vertices[:,1])<1e-8]
        assert len(touching)==1
        contacts.append(touching[0])
    assert np.linalg.norm(contacts[0]-contacts[1])<1e-8
    for box in layout['boxes']:
        body=stage.GetPrimAtPath('/World/Boxes/'+box['name'])
        assert body.HasAPI(UsdPhysics.RigidBodyAPI)
        assert not UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get()
    cart=stage.GetPrimAtPath('/World/Rolltainer')
    assert not any(p.HasAPI(UsdPhysics.RigidBodyAPI) or p.IsA(UsdPhysics.Joint) for p in Usd.PrimRange(cart))
    assert sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(cart))>0
    robot=stage.GetPrimAtPath('/World/IGRIS_C')
    assert np.linalg.norm(new.GetLocalToWorldTransform(robot).ExtractTranslation())<1e-8
    roots=[p for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert len(roots)==1
    bodies=[p for p in Usd.PrimRange(robot) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    graph={p.GetPath():set() for p in bodies}
    joints=[p for p in Usd.PrimRange(robot) if p.IsA(UsdPhysics.Joint)]
    for p in bodies:
        assert not UsdPhysics.RigidBodyAPI(p).GetKinematicEnabledAttr().Get()
    for p in joints:
        joint=UsdPhysics.Joint(p)
        a,b=joint.GetBody0Rel().GetTargets(),joint.GetBody1Rel().GetTargets()
        assert len(a)==len(b)==1,(p.GetPath(),'world-fixed or broken joint')
        assert a[0] in graph and b[0] in graph
        graph[a[0]].add(b[0]);graph[b[0]].add(a[0])
    visited=set();todo=[bodies[0].GetPath()]
    while todo:
        p=todo.pop()
        if p not in visited:
            visited.add(p);todo.extend(graph[p]-visited)
    assert len(visited)==len(bodies)==len(joints)+1
    # Whole robot mesh stays behind both infinite support front planes: a conservative clearance.
    max_planes=np.array([-np.inf,-np.inf])
    for p in Usd.PrimRange(robot):
        if p.IsA(UsdGeom.Mesh):
            points=np.array(UsdGeom.Mesh(p).GetPointsAttr().Get())
            points=np.c_[points,np.ones(len(points))]@np.array(new.GetLocalToWorldTransform(p))
            max_planes=np.maximum(max_planes,[max(points[:,0]+points[:,1]),max(points[:,0]-points[:,1])])
    clearance=(np.sqrt(2)*.2-max_planes)/np.sqrt(2)
    assert clearance.min()>0
    _,_,missing=UsdUtils.ComputeAllDependencies(str(root/'room_corner_igris.usd'))
    assert not missing,missing
    report={'pass':True,'preserved_boxes':len(layout['boxes']),'max_support_relative_matrix_error':relative_error,
        'perpendicular_distances_m':distances,'contact_point_xy_m':contacts[0].tolist(),'footprint_contact':'one corner only',
        'robot_floor_origin_m':[0,0,0],'robot_mesh_front_plane_clearance_m':clearance.tolist(),
        'robot_links':len(bodies),'robot_joints':len(joints),'free_robot_base':True,'fixed_rolltainer':True,'unresolved_assets':missing}
    (root/'corner_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    root=Path(__file__).resolve().parent
    check(root)
    if (root/'room_corner_igris.usd').exists():
        check_corner(root)
