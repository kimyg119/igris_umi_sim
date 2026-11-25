"""Nine controlled virtual-Quest trials through the unmodified vendor controller.
The recorder observes real mj_step outputs; rendered qpos are measured, not IK targets.
"""
from pathlib import Path
from dataclasses import asdict
import argparse,json,subprocess,time,threading,shutil,struct,queue,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import zmq
from igris_ik.assets import HAND_TOOL_FRAMES, hand_tool_transforms
WRIST_OFFSETS=np.array([-t[:3,:3].T@t[:3,3] for t in hand_tool_transforms()])
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
DURATION=48.
BOX_RADIUS=.40
BOX_Z=.90
HAND_FACE_ROTATION_DEG=45.
PALM_SEAT_ANGLE_DEG=12.
SQUEEZE_M=.035  # PD preload within the wrist collision limit: 15 cm -> 8 cm gap.
# Raised pregrasp home: palms face each other with clearance over both supports.
READY_JOINTS={
 'l_shoulder_pitch':-0.65,
 'l_shoulder_roll':0.4810562,
 'l_shoulder_yaw':-0.1048306,
 'l_elbow_pitch':-1.8404770,
 'l_wrist_yaw':0.4946118,
 'l_wrist_roll':0.1107351,
 'l_wrist_pitch':0.4036296,
 'r_shoulder_pitch':-0.65,
 'r_shoulder_roll':-0.4790805,
 'r_shoulder_yaw':0.0934642,
 'r_elbow_pitch':-1.8513741,
 'r_wrist_yaw':-0.4837484,
 'r_wrist_roll':-0.1106155,
 'r_wrist_pitch':0.4116523,
}
STAGES=[(0,'release'),(2,'clutch'),(5,'turn_right'),(11,'approach_box'),(19,'squeeze'),(23,'lift'),(27,'carry_left'),(39,'lower'),(43,'open'),(47,'release')]
def stage(t,motion='box'):
 if motion=='yaw' and 11<=t<27:return 'hold_right'
 if motion=='yaw' and 39<=t<47:return 'hold_left'
 return next(name for start,name in reversed(STAGES) if t>=start)
def smooth(t,a,b):
 x=np.clip((t-a)/(b-a),0,1);return x*x*(3-2*x)
def run(cmd,**kw):return subprocess.run(cmd,check=True,**kw)
def ctl(*args):run([str(ROOT/'sim_local/ctl.sh'),*args],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
def scene(mass,out,friction=.8):
 if not np.isfinite([mass,friction]).all() or mass<=0 or friction<=0:raise ValueError('mass and friction must be finite and positive')
 from igris_ik.constants import URDF_JOINT_NAMES
 standby=json.loads((ROOT/'configs/motions/WALK_STANDBY.json').read_text())['motions'][0]['position']
 ready={**dict(zip(URDF_JOINT_NAMES[3:15],standby[3:15])),**READY_JOINTS}
 tree=ET.parse(ROOT/'sim_local/igris_c_v2_rcv_hand.xml');r=tree.getroot()
 # Refine tangential friction for sustained grasp contact (no attachment or state writes).
 r.find('option').set('noslip_iterations','10')
 # Fixed hand meshes already have explicit body inertia. Enable box contact only.
 for side in ('l','r'):
  body=r.find(f'.//body[@name="{side}_hand_rcv"]')
  for g in body.findall('geom'):g.set('contype','2');g.set('conaffinity','4');g.set('friction',f'{friction} .005 .0001')
 # Separate pickup/release supports; the box is not attached to either hand.
 c=np.array([BOX_RADIUS,0.]);theta=-np.pi/4;R=Rotation.from_euler('z',theta);xy=R.apply([*c,0])[:2]
 world=r.find('worldbody')
 for sign in (-1,1):
  ET.SubElement(world,'geom',name=f'eval_table_{sign}',type='box',size='.05 .05 .025',pos=f'{xy[0]} {sign*abs(xy[1])} {BOX_Z-.1}',euler=f'0 0 {sign*np.pi/4}',contype='1',conaffinity='6',rgba='.3 .35 .4 1')
 b=ET.SubElement(world,'body',name='eval_box',pos=f'{xy[0]} {xy[1]} {BOX_Z}',quat=f'{np.cos(theta/2)} 0 0 {np.sin(theta/2)}')
 ET.SubElement(b,'freejoint',name='eval_box_free')
 ET.SubElement(b,'geom',name='eval_box_geom',type='box',size='.075 .075 .075',mass=str(mass),friction=f'{friction} .005 .0001',contype='4',conaffinity='3',condim='4',rgba='.85 .45 .10 1')
 addresses={};address=0
 for j in world.iter():
  if j.tag not in ('joint','freejoint'):continue
  addresses[j.get('name')]=address
  address+=7 if j.tag=='freejoint' or j.get('type')=='free' else 4 if j.get('type')=='ball' else 1
 for k in r.findall('./keyframe/key'):
  values=list(map(float,k.get('qpos').split()))
  for name,value in ready.items():values[addresses[name]]=value
  k.set('qpos',' '.join(map(str,values))+f' {xy[0]} {xy[1]} {BOX_Z} {np.cos(theta/2)} 0 0 {np.sin(theta/2)}')
 tree.write(out/'scene.xml',encoding='unicode')
 import yaml
 cfg=yaml.safe_load((ROOT/'sim_local/robot.yaml').read_text());cfg['igris_c']['description']['mujoco']['robot_model_path']=f'/igris/sim_local/vr_box_eval/results/{out.name}/scene.xml'
 shutil.copytree(ROOT/'configs/motions',out/'motions',dirs_exist_ok=True)
 from igris_ik.constants import URDF_JOINT_NAMES
 home=json.loads((out/'motions/HOME.json').read_text())
 for motion in home['motions']:
  for name,value in ready.items():motion['position'][URDF_JOINT_NAMES.index(name)]=value
 (out/'motions/HOME.json').write_text(json.dumps(home,indent=2))
 walking=json.loads((out/'motions/WALK_STANDBY.json').read_text())
 for motion in walking['motions']:
  for name,value in READY_JOINTS.items():motion['position'][URDF_JOINT_NAMES.index(name)]=value
 (out/'motions/WALK_STANDBY.json').write_text(json.dumps(walking,indent=2))
 cfg['igris_c']['controller']['motion_root_dir']=f'/igris/sim_local/vr_box_eval/results/{out.name}/motions'
 (out/'robot.yaml').write_text(yaml.safe_dump(cfg))
 return np.array([*xy,BOX_Z])
def virtual(t,zero,landmarks,closed,base,grasp_quats,motion="box",task_value=None,palm_seat=None):
 theta=-np.pi/4*smooth(t,5,11)+np.pi/2*smooth(t,27,39)
 if task_value is not None and t>=11:theta=float(task_value[3])
 base_pos,base_rot=base
 pivot=-base_rot.inv().apply(base_pos)
 rot=base_rot.inv()*Rotation.from_euler('z',theta)*base_rot
 poses=[];body=[]
 for i,name in enumerate(('left','right')):
  p0=np.asarray(getattr(zero,name).position);r0=Rotation.from_quat(getattr(zero,name).wxyz[[1,2,3,0]])
  # Approach with 2 cm surface clearance, then virtually clap through the cube.
  sign=1 if i==0 else -1
  grip=closed[i];rg=Rotation.from_quat(grasp_quats[i])
  if palm_seat is not None and task_value is not None:
   seat_rot=Rotation.from_rotvec(np.array(palm_seat['rotvecs'][i])*task_value[4+i]);pivot_hand=np.array(palm_seat['pivots'][i])
   grip=pivot_hand+seat_rot.apply(grip-pivot_hand);rg=seat_rot*rg
  target=grip+base_rot.inv().apply([0,sign*(SQUEEZE_M+.02)*(1-smooth(t,19,23)+smooth(t,43,47)),0])
  if task_value is not None:target=grip+base_rot.inv().apply([0,sign*(SQUEEZE_M-task_value[1]),0])
  alpha=smooth(t,11,19) if motion=='box' else 0.
  orient=smooth(t,11,15) if motion=='box' else 0.;p=p0.copy()
  hand_rotation=r0*Rotation.from_rotvec(orient*(r0.inv()*rg).as_rotvec())
  if motion=='box':
   # Orient with the forearms wide/high, then lower and close beside the box.
   offset=np.array(palm_seat.get('approach_offset',[.05,.08,.06]) if palm_seat is not None else [.05,.08,.06]);offset[1]*=sign
   waypoint=target+base_rot.inv().apply(offset);settle=smooth(t,15,19)
   # Interpolate the anatomical wrist while changing orientation. A straight EE
   # chord can swing the 16 cm tool offset outside reach midway through the turn.
   wrist=(1-orient)*(p0+r0.apply(WRIST_OFFSETS[i]))+orient*(waypoint+rg.apply(WRIST_OFFSETS[i]))
   p=(1-settle)*(wrist-hand_rotation.apply(WRIST_OFFSETS[i]))+settle*target
   lift=.08*(smooth(t,23,27)-smooth(t,39,43)) if task_value is None else task_value[2]
   p+=base_rot.inv().apply([0,0,lift])
  p=pivot+rot.apply(p-pivot);q=(rot*hand_rotation).as_quat();poses.append(dict(pos=p.tolist(),quat=q.tolist()))
  shoulder=pivot+rot.apply(landmarks[i,0]-pivot);upper=np.linalg.norm(landmarks[i,1]-landmarks[i,0]);lower=np.linalg.norm(landmarks[i,2]-landmarks[i,1])
  wrist=p+Rotation.from_quat(q).apply(WRIST_OFFSETS[i])
  v=wrist-shoulder;dist=np.linalg.norm(v);u=v/max(dist,1e-9)
  a=(upper**2-lower**2+dist**2)/(2*max(dist,1e-9));a=np.clip(a,0,upper)
  hint=rot.apply([0,1 if i==0 else -1,-.5]);normal=hint-u*np.dot(hint,u);normal/=max(np.linalg.norm(normal),1e-9)
  # Blend elbow swivel, not elbow positions, so both segment lengths stay fixed.
  initial=rot.apply(landmarks[i,1]-landmarks[i,0]);initial-=u*np.dot(initial,u);initial/=max(np.linalg.norm(initial),1e-9)
  normal=(1-alpha)*initial+alpha*normal;normal/=max(np.linalg.norm(normal),1e-9)
  elbow=shoulder+a*u+np.sqrt(max(upper**2-a*a,0))*normal
  body.append(dict(shoulder=shoulder.tolist(),elbow=elbow.tolist(),wrist=wrist.tolist()))
 enabled=2<=t<47
 return dict(seq=int(t*1000)+1,t_capture=time.time(),ee_frame='robot_base',head=dict(pos=(pivot+rot.apply(np.array([0,0,.65])-pivot)).tolist(),quat=rot.as_quat().tolist()),left=poses[0],right=poses[1],buttons={side:dict(hand_trigger=float(enabled),b=False) for side in ('left','right')},arms=dict(version=1,valid=True,frame='robot_base',origin_id='virtual-quest-eval',t_capture=time.time(),left=body[0],right=body[1])),theta


def physics_row(path,nq,nv):
 return last_record(path,2+nq+nv+7,8)

def last_record(path,width,header=0):
 with open(path,'rb') as f:
  count=(f.seek(0,2)-header)//(width*8)
  if count<1:raise RuntimeError('No recorded physics state')
  f.seek(header+(count-1)*width*8)
  return np.frombuffer(f.read(width*8),np.float64)

class WorldInput:
 """Synthetic operator with simulator base odometry; invert the real clutch mapping.

 No physics state/control writes. This is ideal world registration for an evaluation,
 not a claim that the Quest publisher already observes robot world odometry.
 """
 def __init__(self,base):
  self.base=base;self.zero=None
 def map(self,p,t,data,model):
  bp=data.xpos[model.body('pelvis').id].copy()
  br=Rotation.from_quat(data.xquat[model.body('pelvis').id][[1,2,3,0]])
  transform=br.inv()*self.base[1]
  world={s:self.base[0]+self.base[1].apply(p[s]['pos']) for s in ('left','right')}
  rotations={s:self.base[1]*Rotation.from_quat(p[s]['quat']) for s in ('left','right')}
  p['head']['pos']=br.inv().apply(self.base[0]+self.base[1].apply(p['head']['pos'])-bp).tolist()
  p['head']['quat']=(transform*Rotation.from_quat(p['head']['quat'])).as_quat().tolist()
  if self.zero is None and t>=2:
   h=Rotation.from_quat(p['head']['quat']).as_euler('xyz')[2]
   self.heading=Rotation.from_euler('z',h);self.zero={}
   self.torso_world_yaw=Rotation.from_quat(data.xquat[model.body('torso').id][[1,2,3,0]]).as_euler('xyz')[2]
   for s,short in [('left','l'),('right','r')]:
    bid=model.body(HAND_TOOL_FRAMES[0 if short=='l' else 1]).id
    rw=Rotation.from_quat(data.xquat[bid][[1,2,3,0]])
    self.zero[s]=(br.inv().apply(world[s]-bp),transform*Rotation.from_quat(p[s]['quat']),
                  br.inv().apply(data.xpos[bid]-bp),br.inv()*rw,
                  data.xpos[bid].copy()-world[s],rotations[s].inv()*rw)
  if self.zero is not None:
   # HMD yaw is clutched, but the box goals use world yaw. Remove the initial
   # torso heading during the first turn, including free-base drift at startup.
   head=Rotation.from_quat(p['head']['quat'])
   p['head']['quat']=(Rotation.from_euler('z',-self.torso_world_yaw*smooth(t,5,11))*head).as_quat().tolist()
  for s in ('left','right'):
   points=np.array([p['arms'][s][k] for k in ('shoulder','elbow','wrist')])
   points=br.inv().apply(self.base[0]+self.base[1].apply(points)-bp)
   if self.zero is None:
    pos=br.inv().apply(world[s]-bp);rot=br.inv()*rotations[s]
   else:
    op,oq,rp,rq,offset,spin=self.zero[s]
    # Preserve the clutched home through the waist turn. Blend the initial
    # physical/FK offset out while lowering, when the wrist has more range.
    fade=1-smooth(t,11,19)
    world[s]+=fade*offset
    rotations[s]=rotations[s]*Rotation.from_rotvec(fade*spin.as_rotvec())
    target=br.inv().apply(world[s]-bp)
    pos=op+self.heading.apply(target-rp)
    rot=self.heading*(br.inv()*rotations[s])*rq.inv()*self.heading.inv()*oq
    points=self.heading.apply(points-points[-1])+pos+rot.apply(WRIST_OFFSETS[0 if s=='left' else 1])
   p[s]['pos']=pos.tolist();p[s]['quat']=rot.as_quat().tolist()
   p['arms'][s]=dict(zip(('shoulder','elbow','wrist'),points.tolist()))
  return dict(world_hand_goals={s:world[s].tolist() for s in world},base_position=bp.tolist(),base_quat_xyzw=br.as_quat().tolist())

def setup(out,sim_arm_gains=None):
 # Only the dedicated simulation stack is restarted; no real robot connection.
 subprocess.run(['docker','exec','igris-sim','pkill','-INT','-f','^/usr/bin/python3 /opt/ros/jazzy/bin/ros2 launch igris_c_controller simulation.launch.py'])
 time.sleep(3)
 trigger=f'/igris/sim_local/vr_box_eval/results/{out.name}/record.trigger';log=f'/igris/sim_local/vr_box_eval/results/{out.name}/physics.bin'
 hand=f'IGRIS_HAND_DIR=/igris/sim_local/vr_box_eval/results/{out.name}' if (out/'compliance.json').exists() else ''
 preload='/tmp/igris_hand_motor.so:/tmp/igris_record_physics.so' if hand else '/tmp/igris_record_physics.so'
 cmd=f'source /opt/ros/jazzy/setup.bash; source /igris/ws/install/setup.bash; cd /tmp; export LD_PRELOAD={preload} {hand} IGRIS_EVAL_TRIGGER={trigger} IGRIS_EVAL_LOG={log}; exec ros2 launch igris_c_controller simulation.launch.py config_file:=/igris/sim_local/vr_box_eval/results/{out.name}/robot.yaml > /igris/sim_local/vr_box_eval/results/{out.name}/vendor.log 2>&1'
 run(['docker','exec','-d','igris-sim','bash','-lc',cmd]);time.sleep(10)
 gain_flag='--sim-arm-kp400' if sim_arm_gains is not None else '--sim-wrist-kp60'
 run([str(ROOT/'sim_local/start_dds_bridge.sh'),gain_flag,'--sim-joint-offsets',f'/igris/sim_local/vr_box_eval/results/{out.name}/physics.bin.offsets'],stdout=open(out/'bridge-start.log','w'))
 # Start the balance controller before resetting into the unsupported ready pose.
 # The cached arm-only SDK command holds the requested arms across the reset.
 from igris_ik.realtime.robot_io import RobotCommand,RobotCommandPublisher
 from igris_ik.realtime.robot_control import VENDOR_HOLD_KP,VENDOR_HOLD_KD
 with open(out/'bringup.log','w') as log:
  for args in [('sim','resume'),('torque','on'),('mode','home')]:
   run([str(ROOT/'sim_local/ctl.sh'),*args],stdout=log,stderr=log)
  run([str(ROOT/'sim_local/rt_physics.sh')],stdout=log,stderr=log)
  time.sleep(2)
  ready_q=np.array(json.loads((out/'motions/HOME.json').read_text())['motions'][0]['position'])
  kp=np.zeros(31);kd=np.zeros(31);kp[15:29]=VENDOR_HOLD_KP[15:29];kd[15:29]=VENDOR_HOLD_KD[15:29];kp[:3]=VENDOR_HOLD_KP[:3];kd[:3]=VENDOR_HOLD_KD[:3]
  kd[15:29]=np.minimum(kd[15:29],5.)  # Ready/clutch preparation obeys the same arm limit.
  publisher=RobotCommandPublisher('tcp://172.17.0.2:5561')
  try:
   for seq in range(50):
    publisher.send(RobotCommand(seq,time.time(),ready_q,np.zeros(31),kp,kd,np.ones(5,dtype=int)));time.sleep(.02)
  finally:publisher.close()
  # Both helpers spend 800 ms in DDS discovery. Stagger their launches so
  # WALK_LOW sees the upright reset state before the open-loop HOME can fall.
  reset=subprocess.Popen([str(ROOT/'sim_local/ctl.sh'),'sim','reset'],stdout=log,stderr=log)
  time.sleep(.08)
  walk=subprocess.Popen([str(ROOT/'sim_local/ctl.sh'),'mode','lowwalk'],stdout=log,stderr=log)
  if reset.wait()!=0 or walk.wait()!=0:raise RuntimeError('Reset/WALK_LOW initialization failed')
  time.sleep(5)



def palm_pad_alignment(palm,pad,tip,roll_deg=-45.,center_yz=None):
 """Touch the palm bump first, then seat the broad MCP plate about that contact.

 Patches are convex projections of mesh points within 1 mm of the support plane,
 not measured pressure/contact areas.
 """
 from scipy.optimize import brentq
 from scipy.spatial import ConvexHull
 from shapely.geometry import MultiPoint, box
 from shapely.ops import unary_union
 palm=palm[palm[:,2]>.05]
 def difference(angle):
  rot=Rotation.from_euler('y',angle)
  return rot.apply(palm)[:,0].max()-rot.apply(pad)[:,0].max()
 tilt=Rotation.from_euler('y',brentq(difference,-1.,1.))
 initial=Rotation.from_euler('y',brentq(lambda a:difference(a)-.005,-1.,tilt.as_rotvec()[1]))
 v=initial.apply(palm);patch=v[v[:,0]>v[:,0].max()-.001]
 pivot=patch.mean(0);pivot[0]=v[:,0].max();pivot=initial.inv().apply(pivot)
 # Place the fingers diagonally on the 15 cm face. The old horizontal pose
 # exposed only a ~1.5 mm strip of MCP plate below the distal central tongue.
 spin=Rotation.from_euler('x',roll_deg,degrees=True);initial=spin*initial;tilt=spin*tilt
 polygons=[]
 for points in (palm,pad):
  v=tilt.apply(points);polygons.append(MultiPoint(v[v[:,0]>v[:,0].max()-.001,1:]).convex_hull)
 plane=tilt.apply(pivot)[0];forbidden=[]
 from contact_carry import PALM_GEOMETRY_ARC
 for fraction in np.linspace(0,PALM_GEOMETRY_ARC,17):
  rotation=initial*Rotation.from_rotvec(fraction*(initial.inv()*tilt).as_rotvec())
  points=rotation.apply(tip)+tilt.apply(pivot)-rotation.apply(pivot)
  hull=ConvexHull(points);edges={tuple(sorted((i,j))) for tri in hull.simplices for i,j in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0]))}
  cut=plane-.001;clipped=list(points[points[:,0]>=cut])
  for i,j in edges:
   a,b=points[i],points[j]
   if (a[0]-cut)*(b[0]-cut)<0:clipped.append(a+(b-a)*(cut-a[0])/(b[0]-a[0]))
  if clipped:forbidden.append(MultiPoint(np.array(clipped)[:,1:]).convex_hull)
 forbidden=unary_union(forbidden).buffer(.002)
 ymin,zmin,ymax,zmax=polygons[0].bounds;best=None
 for y in np.arange(ymax-.065,ymin+.065,.002):
  for z in np.arange(zmax-.065,zmin+.065,.002):
   face=box(y-.075,z-.075,y+.075,z+.075)
   if forbidden.intersection(face).area>1e-12:continue
   area=polygons[1].intersection(face).area
   if best is None or area>best[0]:best=(area,y,z,face)
 if center_yz is not None:
  y,z=center_yz;face=box(y-.075,z-.075,y+.075,z+.075)
  if forbidden.intersection(face).area>1e-12 or polygons[0].difference(face).area>1e-10:
   raise ValueError('Shared bilateral contact center loses surface clearance')
  best=(polygons[1].intersection(face).area,y,z,face)
 if best is None or best[0]<.5e-4:raise ValueError('No usable palm/MCP contact face with distal-link clearance')
 _,y,z,face=best;center=np.array([plane,y,z])
 clipped=[p.intersection(face) for p in polygons]
 patches=[np.column_stack((np.full(len(p.exterior.coords)-1,plane),np.array(p.exterior.coords)[:-1])) for p in clipped]

 return initial,tilt,center,pivot,patches


def palm_first_rotation(clouds,roll_deg):
 """Approach at the measured/rest MCP shape, with fingers 5 mm behind the palm."""
 from scipy.optimize import brentq
 palm,pad,tip=clouds;palm=palm[palm[:,2]>.05]
 def gap(angle):
  r=Rotation.from_euler('y',angle)
  return r.apply(palm)[:,0].max()-max(r.apply(pad)[:,0].max(),r.apply(tip)[:,0].max())-.005
 initial=Rotation.from_euler('x',roll_deg,degrees=True)*Rotation.from_euler('y',brentq(gap,-1.,1.))
 points=initial.apply(palm);patch=points[points[:,0]>points[:,0].max()-.001]
 pivot=patch.mean(0);pivot[0]=points[:,0].max()
 return initial,initial.inv().apply(pivot)


def palm_tip_alignment(palm,pad,tip,roll_deg=-45.,center_yz=None):
 """Fit the palm bump and distal contact patches on the same 15 cm box face."""
 from scipy.optimize import brentq
 from shapely.geometry import MultiPoint,box
 initial,pivot=palm_first_rotation((palm,pad,tip),roll_deg)
 palm=palm[palm[:,2]>.05]
 def gap(angle):
  r=Rotation.from_euler('y',angle)
  return r.apply(palm)[:,0].max()-r.apply(tip)[:,0].max()
 tilt=Rotation.from_euler('x',roll_deg,degrees=True)*Rotation.from_euler('y',brentq(gap,-1.,1.))
 polygons=[]
 for points in (palm,tip):
  v=tilt.apply(points);polygons.append(MultiPoint(v[v[:,0]>v[:,0].max()-.001,1:]).convex_hull)
 bounds=np.array([p.bounds for p in polygons])
 center=(bounds[:,:2].min(0)+bounds[:,2:].max(0))/2 if center_yz is None else np.asarray(center_yz)
 # Keep the palm within 1 cm of the box COM, leaving a distal contact edge margin.
 if center_yz is None and abs(roll_deg)>1e-9:center[1]=polygons[0].centroid.y+.01
 face=box(center[0]-.075,center[1]-.075,center[0]+.075,center[1]+.075)
 if any(p.intersection(face).area<100e-6 for p in polygons):raise ValueError('Less than 100 mm2 of a palm/distal patch inside the box face')
 plane=tilt.apply(palm)[:,0].max()
 if tilt.apply(pad)[:,0].max()>plane+.001:raise ValueError('MCP plate obstructs palm/distal contact')
 # Unlike the old MCP-plate goal, the distal link is deliberately inside the face.
 return initial,tilt,np.r_[plane,center],pivot,polygons


def grasp_targets(q,base,*,palm_tip=False):
 import mujoco
 from igris_ik.sim.mujoco_model import load_fixed_hand_model
 from igris_ik.assets import FIXED_HAND_JOINT_VALUES
 approach=load_fixed_hand_model() if palm_tip else None
 previous=dict(FIXED_HAND_JOINT_VALUES)
 try:
  if palm_tip:FIXED_HAND_JOINT_VALUES.update({s+'_hand_dof_joint_dip':np.deg2rad(27.) for s in ('l','r')})
  bundle=load_fixed_hand_model()
 finally:FIXED_HAND_JOINT_VALUES.update(previous)
 base_pos,base_rot=base;m=bundle.model;md=mujoco.MjData(m)
 md.qpos[np.array(bundle.qpos_addresses)]=q;mujoco.mj_forward(m,md)
 def clouds_for(model,data,side):
  hi=model.body(side+'_hand_root').id;hr=Rotation.from_matrix(data.xmat[hi].reshape(3,3));clouds=[]
  for suffix in ('본체1_5','본체1_6','본체1_4'):
   points=[]
   for g in range(model.ngeom):
    if model.geom_bodyid[g]!=model.body(side+'_hand_'+suffix).id or model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
    mesh=model.geom_dataid[g];v=model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
    points.append(hr.inv().apply(v@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]-data.xpos[hi]))
   clouds.append(np.concatenate(points))
  return clouds
 if approach is not None:
  ad=mujoco.MjData(approach.model);ad.qpos[np.array(approach.qpos_addresses)]=q;mujoco.mj_forward(approach.model,ad)
 closed=[];grasp_quats=[];seat=dict(pivots=[],rotvecs=[]);shared_center=None
 if palm_tip:seat['approach_offset']=[0.,.08,0.]
 for i,side in enumerate(('l','r')):
  wi=m.body(HAND_TOOL_FRAMES[i]).id;hi=m.body(side+'_hand_root').id
  wr=Rotation.from_matrix(md.xmat[wi].reshape(3,3));hr=Rotation.from_matrix(md.xmat[hi].reshape(3,3))
  clouds=clouds_for(m,md,side)
  alignment=palm_tip_alignment if palm_tip else palm_pad_alignment
  roll=(-1 if i==0 else 1)*HAND_FACE_ROTATION_DEG
  initial,tilt,center,pivot,_=alignment(*clouds,roll_deg=roll,center_yz=None if shared_center is None else [-shared_center[1],shared_center[2]])
  if approach is not None:initial,pivot=palm_first_rotation(clouds_for(approach.model,ad,side),roll)
  if i==0:shared_center=center.copy()
  normal=np.array([0,-1 if i==0 else 1,0]);finger=np.array([1,0,0])
  contact_rotation=Rotation.from_matrix(np.column_stack((normal,np.cross(finger,normal),finger)))
  final_rotation=contact_rotation*tilt;hand_rotation=contact_rotation*initial
  wrist_rotation=hand_rotation*(wr.inv()*hr).inv()
  # Extend the hands so the whole palm patch lies inside the box face.
  # The box blocks the narrowed target gap, producing PD force from joint position error.
  contact_goal=np.array([BOX_RADIUS,(1 if i==0 else -1)*(.075-SQUEEZE_M),BOX_Z])
  pivot_position=contact_goal-contact_rotation.apply(center)+final_rotation.apply(pivot)
  hand_position=pivot_position-hand_rotation.apply(pivot)
  wrist_position=hand_position-wrist_rotation.apply(wr.inv().apply(md.xpos[hi]-md.xpos[wi]))
  closed.append(base_rot.inv().apply(wrist_position-base_pos));grasp_quats.append((base_rot.inv()*wrist_rotation).as_quat())
  seat['pivots'].append(base_rot.inv().apply(pivot_position-base_pos).tolist())
  seat['rotvecs'].append((base_rot.inv()*final_rotation*hand_rotation.inv()*base_rot).as_rotvec().tolist())
 if palm_tip:seat['fraction_limit']=min(4.,np.deg2rad(PALM_SEAT_ANGLE_DEG)/max(np.linalg.norm(seat['rotvecs'],axis=1)))
 return np.array(closed),grasp_quats,seat

def trial(mode,mass,*,hold_waist=False,motion="box",contact_driven=True,wrist_kp=30.,friction=.8,separate_waist=False,sim_arm_gains=None,hand_control=None,preload_limit_m=.035,palm_force_N=None,tip_force_N=None):
 from igris_ik import IKConfig,UpperBodyIK
 from igris_ik.assets import FIXED_HAND_JOINT_VALUES
 from igris_ik.realtime.vr_teleop import run_vr_teleop
 adaptive=contact_driven and motion=='box';duration=180. if adaptive else DURATION
 from contact_carry import ContactCarry, PALM_SEAT_LIMIT
 task=ContactCarry(mass,friction,palm_first=hand_control is not None,preload_limit_m=preload_limit_m,palm_force_N=palm_force_N,tip_force_N=tip_force_N) if adaptive else None;feedback=queue.SimpleQueue();attempts={};feedback_rows=[]
 def target_feedback(seq,result):feedback.put((seq,result.success,result.reasons,time.time(),asdict(result.diagnostics),result.collision_free,result.feasible))
 out=HERE/'results'/f'{time.strftime("%Y%m%d_%H%M%S")}_{motion}_{mode}_{mass:g}kg';out.mkdir(parents=True,exist_ok=False)
 for name in ('evaluate.py','contact_carry.py'):shutil.copy2(HERE/name,out/name)
 shutil.copy2(run_vr_teleop.__code__.co_filename,out/'vr_teleop.py')
 for name in ('ik.py','model.py'):
  shutil.copy2(Path(run_vr_teleop.__code__.co_filename).parents[1]/name,out/name)
 shutil.copy2(Path(run_vr_teleop.__code__.co_filename).parent/'robot_io.py',out/'robot_io.py')
 shutil.copy2(ROOT/'igris_curobo/tools/dds_zmq_bridge.py',out/'dds_zmq_bridge.py')
 shutil.copy2(HERE/'record_physics.cpp',out/'record_physics.cpp')
 shutil.copy2(Path(run_vr_teleop.__code__.co_filename).parent/'separated_waist.py',out/'separated_waist.py')
 if (out/'metadata.json').exists():raise FileExistsError(f'Existing trial: {out}')
 scene(mass,out,friction);setup(out,sim_arm_gains)
 if hand_control is not None:hand_control.start(out)
 # Observe actual post-settle physics before any enabled VR input.
 import mujoco
 from render_results import model_for
 (out/'record.trigger').write_text('preflight')
 time.sleep(.3)
 raw=(out/'physics.bin').read_bytes();nq,nv=struct.unpack('ii',raw[:8])
 rows=np.frombuffer(raw[8:],np.float64).reshape(-1,2+nq+nv+7)
 model=model_for(out);data=mujoco.MjData(model);data.qpos[:]=rows[-1,2:2+nq];mujoco.mj_forward(model,data)
 box=model.geom('eval_box_geom').id
 robot_geoms=[g for g in range(model.ngeom) if model.geom_bodyid[g]>0 and g!=box]
 clearance={}
 for name in ('eval_box_geom','eval_table_-1','eval_table_1'):
  clearance[name]=min(mujoco.mj_geomDistance(model,data,model.geom(name).id,g,.2,None) for g in robot_geoms)
 base_pos=data.xpos[model.body('pelvis').id].copy();base_rot=Rotation.from_quat(data.xquat[model.body('pelvis').id][[1,2,3,0]]);base=(base_pos,base_rot)
 (out/'preflight.json').write_text(json.dumps(dict(clearance_m=clearance,base_position=base_pos.tolist()),indent=2))
 if min(clearance.values())<.03:raise RuntimeError(f'Initial scene clearance below 3 cm: {clearance}')
 if base_pos[2]<.7 or np.linalg.norm(base_pos[:2])>.15:raise RuntimeError(f'Unsettled base after WALK_LOW preparation: {base_pos}')
 from PIL import Image
 renderer=mujoco.Renderer(model,720,1280);camera=mujoco.MjvCamera();camera.lookat[:]=[0,0,.85];camera.distance=2.8;camera.azimuth=125;camera.elevation=-15
 renderer.update_scene(data,camera=camera);Image.fromarray(renderer.render()).save(out/'ready.png');renderer.close()
 print('PRE-FLIGHT PASS',clearance,flush=True)
 ctx=zmq.Context();state=ctx.socket(zmq.SUB);state.setsockopt(zmq.SUBSCRIBE,b'');state.setsockopt(zmq.RCVTIMEO,5000);state.connect('tcp://172.17.0.2:5560');d=state.recv_json();q=np.array(d['q']);state.close(0)
 from igris_ik.constants import URDF_JOINT_NAMES
 ready_error=max(abs(q[URDF_JOINT_NAMES.index(name)]-value) for name,value in READY_JOINTS.items())
 if ready_error>.35:raise RuntimeError(f'Forward HOME not retained in WALK_LOW: {ready_error:.3f} rad')
 ik=UpperBodyIK(IKConfig(waist_mode='fixed',neck_mode='fixed',elbow_position_weight=.5));ik.lock_to(q,joint_offsets=d.get('joint_offsets'));zero=ik.forward(q);landmarks=ik.arm_landmarks(q)
 closed,grasp_quats,palm_seat=grasp_targets(q,base,palm_tip=hand_control is not None)
 if task is not None:task.seat_limit=palm_seat.get('fraction_limit',PALM_SEAT_LIMIT)
 pub=ctx.socket(zmq.PUB);pub.bind('tcp://127.0.0.1:5577')
 errors=[];stop_event=threading.Event();ready_event=threading.Event()
 def worker():
  try:result=run_vr_teleop('127.0.0.1',5577,state_endpoint='tcp://172.17.0.2:5560',cmd_endpoint='tcp://172.17.0.2:5561',viser_port=8093,simulation=True,hold_waist=hold_waist,delta_scale=1.,orientation_weight=.5 if mode=="off" else 1.,arm_tracking=mode,duration_s=duration+10,log_dir=str(out),target_feedback=target_feedback if adaptive else None,stop_event=stop_event,wrist_kp=None if sim_arm_gains is not None else wrist_kp,sim_arm_gains=sim_arm_gains,ready_event=ready_event,separate_waist=separate_waist,elbow_out_weight=.1 if hand_control is not None else .05,elbow_out_offset_m=.12 if hand_control is not None else .06);(out/'teleop_summary.json').write_text(json.dumps(result))
  except Exception as e:errors.append(repr(e))
 th=threading.Thread(target=worker);th.start()
 # Wait for solver/monitor/subscriber readiness before starting the task clock.
 for _ in range(3600):
  p,_=virtual(0,zero,landmarks,closed,base,grasp_quats,motion);p['seq']=_;pub.send_json(p);time.sleep(1/60)
  if ready_event.is_set():break
  if errors:raise RuntimeError(errors)
 else:
  stop_event.set();raise RuntimeError('teleop startup exceeded 60 s')
 start=time.time();(out/'record.trigger').write_text(str(start));inputs=[];registration=WorldInput(base)
 (out/'metadata.json').write_text(json.dumps(dict(mode=mode,separate_waist=separate_waist,mass=mass,motion=motion,world_registered=True,command_acceleration_rad_s2=2.,branch_guard_rad=.1,balance_relock_preserves_warm_start=True,squeeze_m=SQUEEZE_M,preload_limit_m=preload_limit_m,hand_orientation_weight=.5 if mode=="off" else 1.,contact_target="palm_tip_seated" if hand_control is not None else "palm_seated",hand_diagonal_deg=[-HAND_FACE_ROTATION_DEG,HAND_FACE_ROTATION_DEG],surface_force_N=task.surface_force_N if task else None,secondary_force_N=task.secondary_force_N if task else None,palm_primary_load=bool(task and task.palm_first),palm_preload_force_N=task.preload_force if task else None,noslip_iterations=10,palm_seat=palm_seat,palm_seat_limit=task.seat_limit if task else PALM_SEAT_LIMIT,approach_waypoint_world_offset_m=palm_seat.get("approach_offset",[.05,.08,.06]),approach_orientation_complete_s=15.,approach_rotation_path="linear_wrist_then_linear_tool",fixed_hand_joint_values=FIXED_HAND_JOINT_VALUES,box_radius=BOX_RADIUS,box_z=BOX_Z,base_position=base_pos.tolist(),base_quat_xyzw=base_rot.as_quat().tolist(),friction=friction,scale=1.,duration=duration,contact_driven=adaptive,wrist_kp=wrist_kp,wrist_kd=3.,start_wall=start,tools=list(HAND_TOOL_FRAMES),closed_ee=closed.tolist(),initial_q=q.tolist(),input='synthetic Quest-compatible observations; not a headset recording',preserve_relative=False,hold_waist=hold_waist,grasp_quats_xyzw=[q.tolist() for q in grasp_quats],ready_joint_error_rad=ready_error,initial_posture='raised pregrasp home: palms inward, elbows out'),indent=2))
 previous_wall=time.time();last_feedback=start;terminal_at=None;task_status_at=0.
 if sim_arm_gains is not None:
  meta=json.loads((out/'metadata.json').read_text());meta.update(sim_arm_gains=sim_arm_gains,wrist_kp=sim_arm_gains['wrist'][0],wrist_kd=sim_arm_gains['wrist'][1]);(out/'metadata.json').write_text(json.dumps(meta,indent=2))
 while time.time()-start<duration:
  wall=time.time();t=wall-start;dt=wall-previous_wall;previous_wall=wall
  row=physics_row(out/'physics.bin',nq,nv);data.qpos[:]=row[2:2+nq];mujoco.mj_forward(model,data)
  if adaptive:
   if (out/'stop.request').exists() and task.phase not in ('failed','done'):
    task.reason='diagnostic_stop';task.transition('failed')
   while not feedback.empty():
    seq,accepted,reasons,stamp,diagnostics,collision_free,feasible=feedback.get();last_feedback=stamp
    candidate=attempts.pop(seq,None)
    if candidate is not None:task.acknowledge(candidate,accepted,progressing=tuple(reasons)==('cartesian_progress',))
    feedback_rows.append(dict(seq=seq,t=stamp-start,accepted=accepted,reasons=reasons,diagnostics=diagnostics,collision_free=collision_free,feasible=feasible))
   patches=last_record(out/'physics.bin.contacts',7)
   pads=last_record(out/'physics.bin.pads',3)
   # A recorder can publish between loop start and these reads; compare after reading.
   sampled_at=time.time()
   fresh=all(0<=sampled_at-stamp<.15 for stamp in (row[0],patches[0],pads[0]))
   if fresh and row[-2]>.5 and task.phase not in ('done','failed'):
    task.reason='robot_support_contact';task.transition('failed')
   if task.value[0]<2:last_feedback=wall  # Start the watchdog when grip first enables IK.
   if task.phase not in ('done','failed') and task.value[0]>=2 and wall-last_feedback>.5 and t>3:
    task.reason='teleop_feedback_timeout';task.transition('failed')
   box_tilt=float(np.arccos(np.clip(data.xmat[model.body('eval_box').id][8],-1,1)))
   value=task.update(dt,fresh=fresh,forces=row[-7:-5],patches=[patches[1],patches[2],patches[4],patches[5]] if hand_control is not None else [patches[1],pads[1],patches[4],pads[2]],
                     box_lift=data.xpos[model.body('eval_box').id][2]-BOX_Z,supported=row[-1]>task.contact_threshold_N,box_tilt_rad=box_tilt)
   if task.value[0]<2:task.acknowledge(value,True)  # released startup has no IK acknowledgement
   if t>=task_status_at:
    print('TASK',task.phase,'clock',round(float(value[0]),2),'preload_mm',round(float(value[1])*1000,2),'force',task.force.round(2),'patch',task.patch.round(2),'pending_targets',task.rejections,flush=True);task_status_at=t+2
   task_t=float(value[0]);p,yaw=virtual(task_t,zero,landmarks,closed,base,grasp_quats,motion,task_value=value,palm_seat=palm_seat)
   if task.phase in ('done','failed'):
    for b in p['buttons'].values():b['hand_trigger']=0.
    if terminal_at is None:terminal_at=wall
   registration_info=registration.map(p,task_t,data,model)
  else:
   p,yaw=virtual(t,zero,landmarks,closed,base,grasp_quats,motion)
   registration_info=registration.map(p,t,data,model)
  p['seq']=len(inputs)+1000
  if adaptive:
   attempts[p['seq']]=value.copy()
   attempts={seq:q for seq,q in attempts.items() if seq>=p['seq']-120}
  pub.send_json(p);inputs.append(dict(t=t,stage=task.phase if adaptive else stage(t,motion),yaw=yaw,payload=p,
                                    **(dict(task_value=value.tolist(),contact_force=task.force.tolist(),surface_force=task.patch.tolist(),contact_fresh=fresh,stable_contact_s=task.stable,task_reason=task.reason,box_tilt_rad=box_tilt) if adaptive else {}),**registration_info))
  if terminal_at is not None and wall-terminal_at>2.:break
  time.sleep(1/30 if adaptive else 1/60)
 if adaptive:
  if task.phase not in ('done','failed'):task.reason='task_timeout';task.transition('failed')
  (out/'contact_task.json').write_text(json.dumps(dict(phase=task.phase,reason=task.reason,required_force_N=task.required_force,preload_force_N=task.preload_force,
       preload_m=float(task.value[1]),rejections=task.rejections,events=task.events),indent=2))
  (out/'target_feedback.json').write_text(json.dumps(feedback_rows))
  meta=json.loads((out/'metadata.json').read_text());meta['duration']=time.time()-start;(out/'metadata.json').write_text(json.dumps(meta,indent=2))
 (out/'input.json').write_text(json.dumps(inputs))
 stop_event.set()
 while th.is_alive() and time.time()-start<duration+10:
  for b in p['buttons'].values():b['hand_trigger']=0.
  p['seq']+=1;p['t_capture']=time.time();pub.send_json(p);time.sleep(.05)
 th.join(2);pub.close(0);ctx.term()
 if th.is_alive():raise RuntimeError('teleop did not finish')
 if errors:raise RuntimeError(errors)
 ctl('sim','pause')
 print('TRIAL COMPLETE',out,flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['off','elbow','arm']);ap.add_argument('mass',type=float);ap.add_argument('--hold-waist',action='store_true');ap.add_argument('--separate-waist',action='store_true');ap.add_argument('--motion',choices=['box','yaw'],default='box');ap.add_argument('--timed',action='store_true');ap.add_argument('--wrist-kp',type=float,choices=[30.,60.],default=30.);ap.add_argument('--friction',type=float,default=.8);a=ap.parse_args()
 try:trial(a.mode,a.mass,hold_waist=a.hold_waist,motion=a.motion,contact_driven=not a.timed,wrist_kp=a.wrist_kp,friction=a.friction,separate_waist=a.separate_waist)
 finally:ctl('sim','pause')
