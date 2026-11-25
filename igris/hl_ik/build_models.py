"""Build vendor torso-root topology with the complete latest RCV-hand subtree."""
import argparse
import copy
import json
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .common import *

def write_xml(path,root):
    path.parent.mkdir(parents=True,exist_ok=True);ET.indent(root)
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)

def vendor_robot(base='torso'):
    result=subprocess.run(['/home/rcv/.local/bin/xacro',str(VENDOR/'urdf/igris_c_v2.urdf.xacro'),
        f'base_type:={base}','parallel:=false','end_effector:=none'],capture_output=True,text=True,check=True)
    return ET.fromstring(result.stdout)

def build_urdf(out):
    root=vendor_robot();root.set('name','igris_c_rcv_hand_torso')
    source=ET.parse(DESCRIPTION/'urdf/igris_c_rcv_hand_pelvis.urdf').getroot()
    links={l.get('name'):l for l in source.findall('link')}
    # Traverse by graph, not by naming prefix: latest EE and camera names differ.
    existing={l.get('name') for l in root.findall('link')}
    frontier={'l_wrist_connector','r_wrist_connector'}
    copied=[]
    while frontier:
        following=set()
        for j in source.findall('joint'):
            p,c=j.find('parent').get('link'),j.find('child').get('link')
            if p in frontier and c not in existing:
                root.append(copy.deepcopy(j));root.append(copy.deepcopy(links[c]));copied.append(j.get('name'))
                existing.add(c);following.add(c)
        frontier=following
    assert all(n in existing for n in EE_NAMES)
    write_xml(out/'igris_c_rcv_hand_torso.urdf',root)
    return root,copied

def compile_urdf(root,out,stem,*,floating=True):
    r=copy.deepcopy(root)
    # Drop massless base_link so torso (or pelvis for parity) is actual root.
    bj=r.find("joint[@name='base_joint']")
    r.remove(bj);r.remove(r.find("link[@name='base_link']"))
    for j in r.findall('joint'):
        name=j.get('name')
        if name in HAND_Q:
            t=origin(j);axis=np.fromstring(j.find('axis').get('xyz'),sep=' ')
            t[:3,:3] = t[:3,:3] @ Rotation.from_rotvec(axis*HAND_Q[name]).as_matrix()
            o=j.find('origin');o.set('rpy',' '.join(map(str,Rotation.from_matrix(t[:3,:3]).as_euler('xyz'))))
            j.set('type','fixed')
        if name.startswith('neck_'): j.set('type','fixed')
        if j.get('type')=='fixed':
            for tag in ('axis','limit','mimic','dynamics'):
                for el in j.findall(tag):j.remove(el)
    for m in r.iter('mesh'):
        raw=m.get('filename');rel=raw.removeprefix('package://igris_c_description/')
        file=DESCRIPTION/rel
        if not file.exists(): file=VENDOR/rel
        if not file.exists():raise FileNotFoundError(file)
        m.set('filename',str(file))
    mj=ET.SubElement(r,'mujoco')
    ET.SubElement(mj,'compiler',strippath='false',discardvisual='false',fusestatic='false',balanceinertia='true')
    m=mujoco.MjModel.from_xml_string(ET.tostring(r,encoding='unicode'))
    with tempfile.NamedTemporaryFile(suffix='.xml') as f:
        mujoco.mj_saveLastXML(f.name,m);x=ET.parse(f.name).getroot()
    x.find('compiler').set('balanceinertia','true')
    body=x.find('worldbody/body')
    if floating: body.insert(0,ET.Element('freejoint',name='floating_base'))
    # GMR/FK uses explicit sphere QC. Disable native contact generation here.
    for g in x.iter('geom'):g.set('contype','0');g.set('conaffinity','0')
    op=x.find('option')
    if op is None:op=ET.SubElement(x,'option')
    op.set('timestep',str(1/30))
    path=out/f'{stem}.xml';write_xml(path,x)
    return mujoco.MjModel.from_xml_path(str(path))

def body_t(m,d,name):
    i=m.body(name).id;t=np.eye(4);t[:3,3]=d.xpos[i];t[:3,:3]=d.xmat[i].reshape(3,3);return t

def parity(mt,mp,out):
    dt,dp=mujoco.MjData(mt),mujoco.MjData(mp);rng=np.random.default_rng(42)
    pos=rot=0.;axes={}
    names=[mujoco.mj_id2name(mt,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(1,mt.nbody)]
    for trial in range(64):
        values=dict(zip(SDK_NAMES,np.zeros(31)))
        if trial:
            for n in SDK_NAMES:
                if n in ['neck_yaw','neck_pitch']:continue
                j=mt.joint(n).id;values[n]=float(rng.uniform(*mt.jnt_range[j])*.5)
        for m,d in [(mt,dt),(mp,dp)]:
            d.qpos[:]=m.qpos0
            for n,q in values.items():
                if n.startswith('neck_'):continue
                d.qpos[m.joint(n).qposadr[0]]=q
            mujoco.mj_kinematics(m,d)
        tp=np.linalg.inv(body_t(mp,dp,'torso'))
        for n in names:
            a=body_t(mt,dt,n);b=tp@body_t(mp,dp,n)
            if n in PELVIS_TO_TORSO_LINK_OFFSET:
                offset=np.eye(4);offset[:3,3]=PELVIS_TO_TORSO_LINK_OFFSET[n];a=a@offset
            pos=max(pos,float(np.linalg.norm(a[:3,3]-b[:3,3])))
            rot=max(rot,float(Rotation.from_matrix(a[:3,:3].T@b[:3,:3]).magnitude()))
        if trial==0:
            for n in EE_NAMES:axes[n]=body_t(mt,dt,n).tolist()
    # Vendor reverses the waist chain while retaining SDK angle conventions.
    assert pos<1e-6 and rot<1e-5,(pos,rot)
    # Both EE frames must describe the same anatomical axes at neutral pose.
    a,b=[np.array(axes[n])[:3,:3] for n in EE_NAMES]
    axis_error=float(Rotation.from_matrix(a.T@b).magnitude())
    assert axis_error<1e-4,axis_error
    report=dict(random_configurations=64,all_body_fk_max_position_m=pos,all_body_fk_max_orientation_rad=rot,
        neutral_left_right_ee_orientation_difference_rad=axis_error,neutral_ee_transforms=axes,
        fixed_hand_q=HAND_Q,root='torso',joint_order=[mt.joint(i).name for i in range(1,mt.njnt)],
        old_link_frame_in_new={k:v.tolist() for k,v in PELVIS_TO_TORSO_LINK_OFFSET.items()})
    save_json(out/'model_validation.json',report)
    return report

def gmr_config(out,model):
    old=json.loads((GMR/'general_motion_retargeting/ik_configs/smplx_to_robros_igris_c_v2_pelvis.json').read_text())
    mapping={'base_link':'pelvis','Link_Waist_Yaw':'torso','Link_Neck_Pitch':'head'}
    for j in ET.parse(DESCRIPTION/'urdf/igris_c_rcv_hand_pelvis.urdf').getroot().findall('joint'):
        n=j.get('name')
        if n not in SDK_NAMES:continue
        parts=n.split('_');side=''
        if parts[0] in ('l','r'):side='_Left' if parts.pop(0)=='l' else '_Right'
        mapping['Link_'+'_'.join(v.title() for v in parts)+side]=j.find('child').get('link')
    # Reversed torso waist topology changes the physical child role.
    mapping['Link_Waist_Yaw']='torso'
    for table in ['ik_match_table1','ik_match_table2']:
        old[table]={mapping[k]:v for k,v in old[table].items()}
        for k in old[table]:model.body(k)
    old['robot_root_name']='torso'
    old['provenance']='GMR pelvis config remapped by physical frames; latest hand/EE FK retained. Targets are anatomical wrist_connector frames.'
    save_json(out/'gmr_config.json',old)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUTPUT/'model');args=ap.parse_args();out=args.output
    out.mkdir(parents=True,exist_ok=True)
    root,copied=build_urdf(out)
    mt=compile_urdf(root,out,'igris_c_rcv_hand_torso')
    mp=compile_urdf(ET.parse(DESCRIPTION/'urdf/igris_c_rcv_hand_pelvis.urdf').getroot(),out,'reference_pelvis')
    report=parity(mt,mp,out);gmr_config(out,mt)
    sources=[VENDOR/'urdf/igris_c_v2.urdf.xacro',VENDOR/'urdf/igris_c_lowerarm_module.urdf.xacro',DESCRIPTION/'urdf/igris_c_rcv_hand_pelvis.urdf']
    save_json(out/'provenance.json',dict(sources={str(p):sha256(p) for p in sources},copied_joints=copied,
        output_urdf_sha256=sha256(out/'igris_c_rcv_hand_torso.urdf'),output_mjcf_sha256=sha256(out/'igris_c_rcv_hand_torso.xml'),
        mass_note='Source hand inertias retained for kinematics; physical box evaluation uses the existing 0.6 kg/hand simulation model.'))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
