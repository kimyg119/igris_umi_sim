"""A biased physical hand pose must not consume wrist range before the waist turn."""
import copy
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as R

from evaluate import WorldInput
from igris_ik.assets import HAND_TOOL_FRAMES
from igris_ik.geometry import Pose, UpperBodyTargets
from igris_ik.realtime.retarget import DeltaCalibration


def test_home_bias_follows_tool_rotation_and_fades_only_during_approach():
    names=['pelvis',*HAND_TOOL_FRAMES,'torso']
    model=SimpleNamespace(body=lambda name:SimpleNamespace(id=names.index(name)))
    base=np.array([0,0,.87]);points=np.array([[.4,.2,.25],[.4,-.2,.25]])
    bias=np.array([.01,0,-.03]);spin=R.from_euler('x',.06)
    data=SimpleNamespace(xpos=np.vstack([base,base+points+bias,base]),
                         xquat=np.vstack([[1,0,0,0],*[spin.as_quat()[[3,0,1,2]]]*2,[1,0,0,0]]))
    packet=dict(head=dict(pos=[0,0,.6],quat=[0,0,0,1]),arms={})
    for i,s in enumerate(['left','right']):
        packet[s]=dict(pos=points[i].tolist(),quat=[0,0,0,1])
        packet['arms'][s]={k:points[i].tolist() for k in ['shoulder','elbow','wrist']}
    def poses(p):
        return UpperBodyTargets(**{s:Pose(p[s]['pos'],np.array(p[s]['quat'])[[3,0,1,2]]) for s in ['left','right','head']})
    robot=poses(packet);mapping=WorldInput((base,R.identity()));p0=copy.deepcopy(packet)
    mapping.map(p0,2,data,model);cal=DeltaCalibration(body_relative=False);cal.calibrate(poses(p0),robot)
    turn=R.from_euler('z',-.5);rotated=copy.deepcopy(packet)
    for i,s in enumerate(['left','right']):
        rotated[s]=dict(pos=turn.apply(points[i]).tolist(),quat=turn.as_quat().tolist())
    rotated['head']['quat']=turn.as_quat().tolist()
    p=copy.deepcopy(rotated);mapping.map(p,8,data,model);target=cal.map(poses(p))
    for i,s in enumerate(['left','right']):
        np.testing.assert_allclose(getattr(target,s).position,turn.apply(points[i]),atol=1e-9)
        actual=R.from_quat(getattr(target,s).wxyz[[1,2,3,0]])
        assert (actual.inv()*turn).magnitude()<1e-9
    p=copy.deepcopy(rotated);info=mapping.map(p,19,data,model)
    for i,s in enumerate(['left','right']):
        np.testing.assert_allclose(info['world_hand_goals'][s],base+turn.apply(points[i]),atol=1e-9)


def test_hmd_and_box_share_world_heading_after_a_rotated_free_base_clutch():
    names=['pelvis',*HAND_TOOL_FRAMES,'torso']
    model=SimpleNamespace(body=lambda name:SimpleNamespace(id=names.index(name)))
    base=np.array([0,0,.87]);initial=R.from_euler('z',-.27)
    data=SimpleNamespace(xpos=np.tile(base,(4,1)),xquat=np.tile(initial.as_quat()[[3,0,1,2]],(4,1)))
    packet=dict(head=dict(pos=[0,0,.6],quat=[0,0,0,1]),arms={})
    for s in ['left','right']:
        packet[s]=dict(pos=[.4,0,.2],quat=[0,0,0,1])
        packet['arms'][s]={k:[.4,0,.2] for k in ['shoulder','elbow','wrist']}
    mapping=WorldInput((base,R.identity()));zero=copy.deepcopy(packet)
    mapping.map(zero,2,data,model)
    h0=R.from_quat(zero['head']['quat']).as_euler('xyz')[2]
    for yaw in [-.3,.1]:
        current=R.from_euler('z',yaw);data.xquat[0]=current.as_quat()[[3,0,1,2]]
        p=copy.deepcopy(packet);p['head']['quat']=R.from_euler('z',-np.pi/4).as_quat().tolist()
        mapping.map(p,19,data,model)
        h=R.from_quat(p['head']['quat']).as_euler('xyz')[2]
        assert abs(yaw+h-h0+np.pi/4)<1e-9


def test_recorded_target_axes_recover_clutch_spin_and_moving_base():
    from evaluate import smooth
    from render_results import recorded_world_rotations
    names=['pelvis',*HAND_TOOL_FRAMES,'torso']
    model=SimpleNamespace(body=lambda name:SimpleNamespace(id=names.index(name)))
    base=np.array([0,0,.87]);base_rot=R.from_euler('xyz',[.02,-.03,.1])
    data=SimpleNamespace(xpos=np.tile(base,(4,1)),
                         xquat=np.tile(R.from_euler('xyz',[.05,.07,.2]).as_quat()[[3,0,1,2]],(4,1)))
    grasp=[R.from_euler('xyz',[.1,.2,.3]),R.from_euler('xyz',[-.2,.3,-.1])]
    vectors=np.array([[.1,.02,-.04],[-.1,.01,.04]])
    meta=dict(world_registered=True,contact_driven=True,motion='box',
              base_quat_xyzw=base_rot.as_quat().tolist(),palm_seat=dict(rotvecs=vectors.tolist()),
              grasp_quats_xyzw=[q.as_quat().tolist() for q in grasp])
    mapping=WorldInput((base,base_rot));frames=[];expected=[]
    for t in [0.,2.,8.,15.,19.,23.,30.]:
        yaw=.5*smooth(t,5,20);seat=[.4*smooth(t,19,23),.3*smooth(t,19,23)]
        data.xquat[0]=R.from_euler('xyz',[.02,.03,t*.01]).as_quat()[[3,0,1,2]]
        packet=dict(head=dict(pos=[0,0,.6],quat=[0,0,0,1]),arms={});world=[]
        for i,s in enumerate(('left','right')):
            goal=R.from_euler('z',yaw)*base_rot*R.from_rotvec(vectors[i]*seat[i])*grasp[i]
            packet[s]=dict(pos=[.4,.2*(-1)**i,.1],quat=(base_rot.inv()*goal).as_quat().tolist())
            packet['arms'][s]={k:packet[s]['pos'] for k in ('shoulder','elbow','wrist')}
            world.append(goal)
        info=mapping.map(packet,t,data,model)
        if mapping.zero is not None:
            world=[q*R.from_rotvec((1-smooth(t,11,19))*mapping.zero[s][-1].as_rotvec()) for q,s in zip(world,('left','right'))]
        expected.append([q.as_matrix() for q in world])
        frames.append(dict(payload=packet,task_value=[t,0,0,yaw,*seat],yaw=yaw,**info))
    np.testing.assert_allclose(recorded_world_rotations(meta,frames),expected,atol=1e-12)
