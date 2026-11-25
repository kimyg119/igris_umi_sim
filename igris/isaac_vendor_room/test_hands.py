"""FK-only cross-check against the torso URDF; native dynamics audited by observer.cpp."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict
import numpy as np
from yourdfpy import URDF
import mujoco
from hands import ROOT,URDF as TORSO_URDF
sys.path.insert(0,str(ROOT/'igris/sim_local/vr_box_eval'))
from render_results import model_for
from compliant_hands import MCPCompliance,DIP_STIFFNESS,FIXED_HAND_JOINT_VALUES


def check(out):
    m=model_for(out);d=mujoco.MjData(m);u=URDF.load(str(TORSO_URDF),load_meshes=False,load_collision_meshes=False)
    metadata=json.loads((out/'compliance.json').read_text());controller=MCPCompliance()
    assert controller.profile.goal_pwm_raw==metadata['goal_pwm_raw']
    assert controller.profile.current_limit_A==metadata['current_limit_A']
    assert controller.profile.external_torque_limit_Nm==metadata['external_torque_limit_Nm']
    assert asdict(controller.config.admittance)==metadata['admittance']
    assert asdict(controller.config.safety)==metadata['safety']
    assert metadata['non_hand_manufacturer_xml_unchanged']
    r=ET.parse(out/'scene.xml').getroot();source=ET.parse(metadata['source_scene']).getroot()
    assert r.find('option').get('noslip_iterations')==ET.parse(out/'manufacturer.xml').find('option').get('noslip_iterations')
    for s in ('l','r'):
        for part in ('mcp_compliant','dip_spring'):
            assert r.find(f".//joint[@name='{s}_hand_{part}']").attrib==source.find(f".//joint[@name='{s}_hand_{part}']").attrib
    maximum=0.
    for mcp in np.deg2rad([20,30,40]):
        for dip in np.deg2rad([-10,0,27,45]):
            values={**FIXED_HAND_JOINT_VALUES,**{f'{s}_hand_dof_joint_mcp':mcp for s in ('l','r')},**{f'{s}_hand_dof_joint_dip':dip for s in ('l','r')}};u.update_cfg(values)
            for s in ('l','r'):
                for part,angle in [('mcp_compliant',mcp),('dip_spring',dip)]:d.qpos[m.jnt_qposadr[m.joint(s+'_hand_'+part).id]]=angle
            mujoco.mj_forward(m,d)
            for s in ('l','r'):
                root=m.body(s+'_hand_rcv').id;rotation=d.xmat[root].reshape(3,3)
                for part,link in [('mcp','본체1_7'),('dip','본체1_4')]:
                    bid=m.body(s+'_hand_'+part).id;tf=np.eye(4);tf[:3,:3]=rotation.T@d.xmat[bid].reshape(3,3);tf[:3,3]=rotation.T@(d.xpos[bid]-d.xpos[root])
                    error=float(np.max(abs(tf-u.get_transform(s+'_hand_'+link,s+'_hand_root'))));maximum=max(maximum,error);assert error<1e-8
                j=m.joint(s+'_hand_dip_spring').id
                assert abs(d.qfrc_passive[m.jnt_dofadr[j]]+DIP_STIFFNESS*dip)<1e-9
    report={'pass':True,'urdf':str(TORSO_URDF),'hand_configurations':12,'hand_fk_max_matrix_difference':maximum,'host_mujoco_role':'FK only; native 3.2.7 audits spring force separately','hardware_executed':False}
    snapshot=out/'native_pose_audit.json'
    if snapshot.exists():
        from scipy.spatial.transform import Rotation
        native=json.loads(snapshot.read_text());q=native['hinge_q']
        u.update_cfg({n:q.get(n,0)+q.get(n+'_backlash',0) for n,j in u.joint_map.items() if j.type=='revolute' and n in q})
        poses={}
        for b in native['bodies']:
            p=np.array(b['pose']);tf=np.eye(4);tf[:3,3]=p[:3];tf[:3,:3]=Rotation.from_quat(p[3:][[1,2,3,0]]).as_matrix();poses[b['name']]=tf
        error=[]
        for link in ('end_effector_left','end_effector_right'):
            a=np.linalg.inv(poses['torso'])@poses[link];b=u.get_transform(link,'torso')
            error.append(float(np.max(abs(a-b))));assert error[-1]<1e-8
        report['native_torso_ee_fk_with_backlash_max_matrix_difference']=max(error)
    placement=out/'room_layout_applied.json'
    if placement.exists():
        applied=json.loads(placement.read_text());meta=json.loads((out/'model.json').read_text());width=2+7*len(meta['bodies'])
        records=np.fromfile(out/'poses.bin').reshape(-1,width);at=np.searchsorted(records[:,1],applied['sim'])
        assert 0<at<len(records)
        poses=records[:,2:].reshape(-1,len(meta['bodies']),7)
        robot=[i for i,b in enumerate(meta['bodies']) if b['name']!='world' and not b['name'].startswith('room_')]
        room=[i for i,b in enumerate(meta['bodies']) if b['name'].startswith('room_')]
        jump=float(np.max(np.linalg.norm(poses[at,robot,:3]-poses[at-1,robot,:3],axis=1)))
        assert jump<.03, 'Robot jumped during room initialization'
        assert np.min(poses[at-1,room,0])>7 and np.max(poses[at,room,0])<7
        assert np.max(np.linalg.norm(np.diff(poses[at:,room,:3],axis=0),axis=2))<1, 'Room repositioned during replay'
        report['room_initialization_robot_step_max_m']=jump
        report['room_initialized_once_before_replay']=True
    (out/'hand_fk_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':check(Path(sys.argv[1]).resolve())
