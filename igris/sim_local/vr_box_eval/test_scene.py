"""Runnable preflight: free base/box, composed ready pose, and support clearance."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import mujoco
import numpy as np
from types import SimpleNamespace
from scipy.spatial.transform import Rotation
from evaluate import ROOT, READY_JOINTS, scene, virtual, WorldInput, SQUEEZE_M, grasp_targets, BOX_RADIUS, BOX_Z, palm_pad_alignment, WRIST_OFFSETS, smooth
from render_results import model_for
from igris_ik.constants import URDF_JOINT_NAMES


def hand_vertices(model,data,side,frame,link=None):
    points=[];body=model.body(side+'_'+frame).id
    for g in range(model.ngeom):
        if not model.body(model.geom_bodyid[g]).name.startswith(side+'_hand') or model.body(model.geom_bodyid[g]).name.endswith('_cam') or model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
        if link is not None and model.body(model.geom_bodyid[g]).name!=side+'_hand_'+link:continue
        mesh=model.geom_dataid[g]
        v=model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
        points.append((v@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]-data.xpos[body])@data.xmat[body].reshape(3,3))
    return np.concatenate(points)


def check():
    standby = json.loads((ROOT / 'configs/motions/WALK_STANDBY.json').read_text())['motions'][0]['position']
    expected = {**dict(zip(URDF_JOINT_NAMES[3:15], standby[3:15])), **READY_JOINTS}
    with TemporaryDirectory() as tmp:
        out = Path(tmp)
        for mass in (.01, .5, 1., 2.):
            scene(mass, out,friction=1.2)
            m = model_for(out)
            assert m.opt.noslip_iterations==10
            d = mujoco.MjData(m)
            mujoco.mj_resetDataKeyframe(m, d, 0)
            mujoco.mj_forward(m, d)
            assert m.joint('floating_base').type == mujoco.mjtJoint.mjJNT_FREE
            assert m.joint('eval_box_free').type == mujoco.mjtJoint.mjJNT_FREE
            assert abs(float(m.body('eval_box').mass[0]) - mass) < 1e-9
            for side in ('l','r'):
                hand = m.geom_bodyid == m.body(side+'_hand_rcv').id
                assert np.allclose(m.geom_friction[hand,0],1.2)
            home = json.loads((out / 'motions/HOME.json').read_text())['motions'][0]['position']
            for name, value in expected.items():
                assert abs(d.qpos[m.jnt_qposadr[m.joint(name).id]] - value) < 1e-9
                assert home[URDF_JOINT_NAMES.index(name)] == value
            for contact in d.contact:
                names = [m.geom(int(i)).name for i in contact.geom]
                assert not (('eval_box_geom' in names or any('eval_table' in n for n in names)) and any(m.geom_bodyid[int(i)]>0 and int(i)!=m.geom('eval_box_geom').id for i in contact.geom)), names
            robot = [g for g in range(m.ngeom) if m.geom_bodyid[g] > 0 and g != m.geom('eval_box_geom').id]
            for name in ('eval_box_geom', 'eval_table_-1', 'eval_table_1'):
                assert min(mujoco.mj_geomDistance(m, d, m.geom(name).id, g, .2, None) for g in robot) >= .03
    # Vendor mesh bake, local geometry and IK collision approximation agree.
    from igris_ik.sim.mujoco_model import load_fixed_hand_model
    from igris_ik.model import model_yaml
    from scipy.spatial import cKDTree
    import yaml
    bundle=load_fixed_hand_model();fixed=bundle.model;fd=mujoco.MjData(fixed)
    fd.qpos[list(bundle.qpos_addresses)]=[d.qpos[m.jnt_qposadr[m.joint(n).id]] for n in URDF_JOINT_NAMES]
    mujoco.mj_forward(fixed,fd)
    spheres=yaml.safe_load(model_yaml().read_text())['kinematics']['collision_spheres']
    from igris_ik.assets import HAND_TOOL_FRAMES, hand_tool_transforms
    for side, tool, expected in zip(('l','r'),HAND_TOOL_FRAMES,hand_tool_transforms()):
        wrist=m.body(side+'_wrist_connector').id;ee=m.body(tool).id
        rotation=d.xmat[wrist].reshape(3,3)
        np.testing.assert_allclose(rotation.T@(d.xpos[ee]-d.xpos[wrist]),expected[:3,3],atol=1e-9)
        np.testing.assert_allclose(rotation.T@d.xmat[ee].reshape(3,3),expected[:3,:3],atol=1e-9)
    for side in ('l','r'):
        a=hand_vertices(m,d,side,'wrist_connector');b=hand_vertices(fixed,fd,side,'wrist_connector')
        assert max(cKDTree(a).query(b)[0].max(),cKDTree(b).query(a)[0].max())<1e-5
        vertices=hand_vertices(fixed,fd,side,'hand_root')
        balls=spheres[side+'_hand_root'];centers=np.array([v['center'] for v in balls]);radii=np.array([v['radius'] for v in balls])
        assert np.max(np.min(np.linalg.norm(vertices[:,None,:]-centers[None,:,:],axis=2)-radii,axis=1))<1e-5
    # World pickup/release geometry stays correct with a translated, tilted base.
    landmarks = np.array([[[0,.15,.3],[.1,.2,.1],[.2,.15,.3]],
                          [[0,-.15,.3],[.1,-.2,.1],[.2,-.15,.3]]])
    zero = SimpleNamespace(**{side: SimpleNamespace(position=landmarks[i,2], wxyz=np.array([1.,0,0,0])) for i,side in enumerate(('left','right'))})
    base = (np.array([.04,-.02,.87]), Rotation.from_euler('xyz',[.02,.06,.08]))
    closed = np.array([[.3,.1,.15],[.3,-.1,.15]])
    pickup,_ = virtual(23,zero,landmarks,closed,base,[[0,0,0,1]]*2)
    release,_ = virtual(43,zero,landmarks,closed,base,[[0,0,0,1]]*2)
    for side in ('left','right'):
        a = base[0] + base[1].apply(pickup[side]['pos'])
        b = base[0] + base[1].apply(release[side]['pos'])
        assert np.allclose(b, Rotation.from_euler('z',np.pi/2).apply(a))
    for t in np.linspace(0,48,97):
        packet,_ = virtual(t,zero,landmarks,closed,base,[[0,0,0,1]]*2)
        for i,side in enumerate(('left','right')):
            points=np.array([packet['arms'][side][k] for k in ('shoulder','elbow','wrist')])
            np.testing.assert_allclose(np.linalg.norm(np.diff(points,axis=0),axis=1),
                                       np.linalg.norm(np.diff(landmarks[i],axis=0),axis=1),atol=1e-9)
    # Strong squeeze is staged: approach stays open, then moves inward 5.5 cm per side.
    opened,_ = virtual(19,zero,landmarks,closed,base,[[0,0,0,1]]*2)
    pressed,_ = virtual(23,zero,landmarks,closed,base,[[0,0,0,1]]*2)
    for side in ('left','right'):
        assert abs(np.linalg.norm(np.array(opened[side]['pos'])-pressed[side]['pos'])-(SQUEEZE_M+.02))<1e-9
    assert abs(2*(.075-SQUEEZE_M)-.08)<1e-9
    # Orientation changes follow a straight wrist path, avoiding a tool-offset arc
    # outside the arm's reach even when both endpoint wrist poses are reachable.
    orientations=Rotation.from_euler('xyz',[[.8,.6,1.2],[-.7,.9,-1.1]]).as_quat()
    turn=base[1].inv()*Rotation.from_euler('z',-np.pi/4)*base[1]
    origin=-base[1].inv().apply(base[0])
    for t in np.linspace(11,15,17):
        packet,_=virtual(t,zero,landmarks,closed,base,orientations)
        for i,side in enumerate(('left','right')):
            sign=1 if i==0 else -1
            end=closed[i]+base[1].inv().apply([.05,sign*(SQUEEZE_M+.02+.08),.06])
            start_wrist=getattr(zero,side).position+WRIST_OFFSETS[i]
            end_wrist=end+Rotation.from_quat(orientations[i]).apply(WRIST_OFFSETS[i])
            wrist=(1-smooth(t,11,15))*start_wrist+smooth(t,11,15)*end_wrist
            actual=np.array(packet[side]['pos'])+Rotation.from_quat(packet[side]['quat']).apply(WRIST_OFFSETS[i])
            np.testing.assert_allclose(actual,origin+turn.apply(wrist-origin),atol=1e-9)
    from igris_ik.assets import load_default_pose
    targets,quats,seat=grasp_targets(load_default_pose().q_sdk,base)
    assert np.isfinite(targets).all() and len(quats)==2
    fd.qpos[list(bundle.qpos_addresses)]=load_default_pose().q_sdk;mujoco.mj_forward(fixed,fd)
    shared_center=None;shared_centroids=None
    for i,side in enumerate(('l','r')):
        palm=hand_vertices(fixed,fd,side,'hand_root','본체1_5')
        pad=hand_vertices(fixed,fd,side,'hand_root','본체1_6')
        tip=hand_vertices(fixed,fd,side,'hand_root','본체1_4')
        initial,tilt,center,pivot,patches=palm_pad_alignment(palm,pad,tip,roll_deg=-45. if i==0 else 45.,center_yz=None if shared_center is None else [-shared_center[1],shared_center[2]])
        if i==0:shared_center=center.copy()
        assert 2.6 < np.rad2deg((initial.inv()*tilt).magnitude()) < 2.7
        assert abs((Rotation.from_euler('x',45 if i==0 else -45,degrees=True)*tilt).as_euler('xyz',degrees=True)[1]-8.91595)<.01
        v=initial.apply(palm[palm[:,2]>.05]);w=initial.apply(pad)
        assert abs(v[:,0].max()-w[:,0].max()-.005)<1e-9
        combined=np.concatenate(patches)-center
        assert np.max(np.abs(combined[:,1:]))<=.075+1e-9
        assert abs(combined[:,0].max())<1e-9
        from shapely.geometry import MultiPoint
        v=tilt.apply(palm[palm[:,2]>.05]);full_palm=MultiPoint(v[v[:,0]>v[:,0].max()-.001,1:]).convex_hull
        palm_patch=MultiPoint(patches[0][:,1:]).convex_hull
        assert abs(palm_patch.area-full_palm.area)<1e-10
        assert np.max(np.abs(patches[0][:,1:]-center[1:]))<=.065+1e-9
        assert MultiPoint(patches[1][:,1:]).convex_hull.area>.5e-4
        centroids=np.array([[MultiPoint(p[:,1:]).convex_hull.centroid.y-center[2],
                             (1 if i else -1)*(MultiPoint(p[:,1:]).convex_hull.centroid.x-center[1])] for p in patches])
        if i==0:shared_centroids=centroids
        else:np.testing.assert_allclose(centroids,shared_centroids,atol=.002)
        # Check hull/box-face intersections, including triangles crossing an edge.
        from scipy.spatial import ConvexHull
        from scipy.optimize import linprog
        from contact_carry import PALM_SEAT_LIMIT, PALM_GEOMETRY_ARC
        for fraction in np.linspace(0,PALM_GEOMETRY_ARC,41):
            r=initial*Rotation.from_rotvec(fraction*(initial.inv()*tilt).as_rotvec())
            points=r.apply(tip)+tilt.apply(pivot)-r.apply(pivot)-center
            hull=ConvexHull(points).equations
            solution=linprog([-1,0,0],A_ub=hull[:,:3],b_ub=-hull[:,3],bounds=[(None,None),(-.075,.075),(-.075,.075)],method='highs')
            assert solution.status==2 or (solution.success and solution.x[0]<-.001)
        # The wrist follows an arc while its palm contact point stays fixed.
        packets=[virtual(19,zero,landmarks,targets,base,quats,task_value=np.array([19.,.007,0.,-.785398,s,s]),palm_seat=seat)[0] for s in (0.,PALM_SEAT_LIMIT)]
        hand='left' if side=='l' else 'right'
        local_pivot=Rotation.from_quat(quats[i]).inv().apply(np.array(seat['pivots'][i])-targets[i])
        points=[np.array(p[hand]['pos'])+Rotation.from_quat(p[hand]['quat']).apply(local_pivot) for p in packets]
        np.testing.assert_allclose(points[0],points[1],atol=1e-9)

    # Inverting the actual DeltaCalibration keeps a world goal fixed as base moves.
    from igris_ik.geometry import Pose, UpperBodyTargets
    from igris_ik.realtime.retarget import DeltaCalibration
    import copy
    bp=d.xpos[m.body('pelvis').id].copy();br=Rotation.from_quat(d.xquat[m.body('pelvis').id][[1,2,3,0]])
    hands={}
    for side,short in [('left','l'),('right','r')]:
        bid=m.body('end_effector_'+side).id
        hands[side]=Pose(br.inv().apply(d.xpos[bid]-bp),(br.inv()*Rotation.from_quat(d.xquat[bid][[1,2,3,0]])).as_quat()[[3,0,1,2]])
    target0=UpperBodyTargets(**hands,head=Pose([0,0,.6],[1,0,0,0]))
    p0=dict(seq=1,t_capture=1.,ee_frame='robot_base',head=dict(pos=[0,0,.6],quat=[0,0,0,1]),arms={})
    for side in hands:
        p0[side]=dict(pos=hands[side].position.tolist(),quat=hands[side].wxyz[[1,2,3,0]].tolist())
        p0['arms'][side]=dict(shoulder=[0,0,.3],elbow=[0,0,.1],wrist=p0[side]['pos'])
    def goals(packet):
        return UpperBodyTargets(**{s:Pose(packet[s]['pos'],np.array(packet[s]['quat'])[[3,0,1,2]]) for s in ('left','right','head')})
    registration=WorldInput((bp,br));packet=copy.deepcopy(p0);registration.map(packet,2,d,m)
    cal=DeltaCalibration(body_relative=False);cal.calibrate(goals(packet),target0)
    d.qpos[:3]+=[.03,-.02,0];newbr=Rotation.from_euler('z',.2)*br;d.qpos[3:7]=newbr.as_quat()[[3,0,1,2]];mujoco.mj_forward(m,d)
    packet=copy.deepcopy(p0);info=registration.map(packet,11,d,m);mapped=cal.map(goals(packet))
    for side in ('left','right'):
        np.testing.assert_allclose(d.xpos[m.body('pelvis').id]+newbr.apply(getattr(mapped,side).position.copy()),info['world_hand_goals'][side],atol=1e-9)
        expected=br*Rotation.from_quat(p0[side]['quat'])
        actual=newbr*Rotation.from_quat(getattr(mapped,side).wxyz[[1,2,3,0]])
        assert (expected.inv()*actual).magnitude()<1e-9
    print('PASS: all masses, free joints, WALK_STANDBY legs, forward arms, initial support clearance')


if __name__ == '__main__':
    check()
