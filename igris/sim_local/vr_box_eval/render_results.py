"""Render and score recorded vendor physics, including the actual floating base and box."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import argparse,json,subprocess,struct
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation
from evaluate import stage,smooth
ROOT=Path(__file__).resolve().parents[3]
def model_for(out):
 import xml.etree.ElementTree as ET
 r=ET.fromstring((out/'scene.xml').read_text())
 # Rendering host 3.12 compatibility; physics was vendor MuJoCo 3.2.7.
 for a in r.findall('./actuator/position'):
  if 'dampratio' in a.attrib and a.get('kv')=='0':a.attrib.pop('kv')
 for el in r.iter():
  for k,v in list(el.attrib.items()):
   if v.startswith('/igris/'):el.set(k,str(ROOT/'igris')+v[len('/igris'):])
   elif v.startswith('/robros/configs/'):el.set(k,str(ROOT/'igris/configs')+v[len('/robros/configs'):])
 vis=r.find('visual')
 if vis is None:vis=ET.SubElement(r,'visual')
 gl=vis.find('global')
 if gl is None:gl=ET.SubElement(vis,'global')
 gl.set('offwidth','1280');gl.set('offheight','720')
 m=mujoco.MjModel.from_xml_string(ET.tostring(r,encoding='unicode'))
 m.vis.headlight.ambient[:]=.2;m.vis.headlight.diffuse[:]=.3
 return m

def recorded_world_rotations(meta,inputs):
 """Undo WorldInput's clutch rotation using its recorded, fully seated reference.

 The publisher quaternions are controller observations, not direct world targets.
 This reconstruction also preserves the home spin correction during approach.
 """
 if not (meta.get('world_registered') and meta.get('contact_driven') and meta.get('motion')=='box'):
  raise ValueError('Target axes require a recorded world-registered contact trial')
 clock=np.array([f['task_value'][0] for f in inputs]);clutch=int(np.flatnonzero(clock>=2)[0])
 full=np.flatnonzero(clock>=19)
 if not len(full):raise ValueError('No fully approached reference for target orientation recovery')
 heading=Rotation.from_euler('z',Rotation.from_quat(inputs[clutch]['payload']['head']['quat']).as_euler('xyz')[2])
 base=Rotation.from_quat([f['base_quat_xyzw'] for f in inputs]);result=[]
 for i,side in enumerate(('left','right')):
  expected=Rotation.from_euler('z',np.array([inputs[k]['yaw'] for k in full])[:,None])*Rotation.from_quat(meta['base_quat_xyzw'])*Rotation.from_rotvec(np.array([inputs[k]['task_value'][4+i] for k in full])[:,None]*np.array(meta['palm_seat']['rotvecs'][i]))*Rotation.from_quat(meta['grasp_quats_xyzw'][i])
  observed=Rotation.from_quat([f['payload'][side]['quat'] for f in inputs])
  tail=expected[0].inv()*base[full[0]]*heading.inv()*observed[full[0]]
  recovered=base*heading.inv()*observed*tail.inv()
  if np.max((expected.inv()*recovered[full]).magnitude())>1e-7:
   raise ValueError('Recorded clutch mapping differs from orientation reconstruction')
  matrices=recovered.as_matrix();matrices[:clutch]=(base[:clutch]*observed[:clutch]).as_matrix()
  result.append(matrices)
 return np.stack(result,axis=1)

def pose_axes(draw,scene,point,rotation,actual=False):
 # Project overlays through the renderer's perspective camera; draw over surfaces
 # so the intentionally penetrating squeeze target is not hidden inside the hand.
 cam=scene.camera[0];eye=(cam.pos+scene.camera[1].pos)/2
 right=np.cross(cam.forward,cam.up);focal=720*cam.frustum_near/(cam.frustum_top-cam.frustum_bottom)
 points=np.vstack([point,point+rotation.T*(.05 if actual else .09)])-eye
 depth=points@cam.forward
 if np.any(depth<=cam.frustum_near):return
 uv=np.c_[640+focal*(points@right)/depth,360-focal*(points@cam.up)/depth]
 origin=tuple(uv[0])
 for axis,color in enumerate([(255,65,65),(30,235,60),(55,120,255)]):
  end=tuple(uv[axis+1]);color=tuple(int(.55*255+.45*c) for c in color) if actual else color
  draw.line([origin,end],fill=(10,10,10),width=4 if actual else 7)
  draw.line([origin,end],fill=color,width=2 if actual else 4)
  delta=uv[axis+1]-uv[0];length=np.linalg.norm(delta)
  if length>5:
   direction=delta/length;normal=np.array([-direction[1],direction[0]])
   draw.polygon([end,tuple(uv[axis+1]-7*direction+3*normal),tuple(uv[axis+1]-7*direction-3*normal)],fill=color)
 if actual:draw.ellipse((origin[0]-3,origin[1]-3,origin[0]+3,origin[1]+3),fill='white',outline='black')

def process(out,metrics_only=False,target_poses=False):
 meta=json.loads((out/'metadata.json').read_text());inputs=json.loads((out/'input.json').read_text());input_t=np.array([f['t'] for f in inputs]);base_rot=Rotation.from_quat(meta['base_quat_xyzw']);base_pos=np.array(meta['base_position']);raw=(out/'physics.bin').read_bytes();nq,nv=struct.unpack('ii',raw[:8]);width=2+nq+nv+7
 rows=np.frombuffer(raw[8:],np.float64).reshape(-1,width);t=rows[:,0]-meta['start_wall'];keep=(t>=0)&(t<=meta['duration']);rows=rows[keep];t=t[keep]
 if len(t)<30:raise RuntimeError('insufficient actual physics recording')
 m=model_for(out);assert m.nq==nq and m.nv==nv;d=mujoco.MjData(m)
 bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'eval_box');base=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pelvis');torso=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'torso')
 hands=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,name) for name in meta.get('tools',['l_wrist_connector','r_wrist_connector'])]
 elbows=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,s+'_elbow_bracket') for s in ('l','r')]
 from igris_ik.constants import URDF_JOINT_NAMES
 armj=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name) for name in URDF_JOINT_NAMES[15:29]]
 records=[];box_tilts=[]
 for row in rows:
  d.qpos[:]=row[2:2+nq];d.qvel[:]=row[2+nq:2+nq+nv];mujoco.mj_forward(m,d)
  box_tilts.append(float(np.arccos(np.clip(d.xmat[bid][8],-1,1))))
  R=Rotation.from_quat(d.xquat[torso][[1,2,3,0]]);Rb=Rotation.from_quat(d.xquat[base][[1,2,3,0]])
  records.append(np.r_[d.xpos[bid],d.xpos[base],R.as_euler('xyz')[2],(Rb.inv()*R).as_euler('xyz')[2],d.xpos[hands].ravel(),d.qvel[m.jnt_dofadr[armj]]])
 state=np.array(records);forces=rows[:,-7:];carry=(t>=27)&(t<39);squeeze=(t>=19)&(t<27)
 phases=np.array([inputs[min(np.searchsorted(input_t,sec),len(inputs)-1)]['stage'] for sec in t])
 phase_driven=meta.get('contact_driven') or meta.get('squeeze_hold')
 if phase_driven:
  carry=phases=='carry_left';squeeze=np.isin(phases,['squeeze','probe_lift','lift'])
 released=phases=='done' if phase_driven else t>=47
 threshold=min(.5,.1*meta['mass']*9.81)
 bilateral=(forces[:,0]>threshold)&(forces[:,1]>threshold);unsupported=forces[:,6]<threshold;lift=state[:,2]>meta['box_z']+.025
 from contact_carry import BOX_TILT_LIMIT_RAD,LANDING_TILT_LIMIT_RAD
 lifted=bilateral&unsupported&lift
 box_tilts=np.array(box_tilts);stable=lifted&(box_tilts<=BOX_TILT_LIMIT_RAD)
 def frac(a,mask):return float(np.mean(a[mask])) if np.any(mask) else None
 desired=np.rad2deg(np.interp(t,input_t,[f['yaw'] for f in inputs]))
 torso_relative=np.rad2deg(np.unwrap(state[:,7]));torso_relative-=torso_relative[0]
 torso_world=np.rad2deg(np.unwrap(state[:,6]))
 if not meta.get('world_registered',False):torso_world-=torso_world[0]
 error=(torso_relative-desired+180)%360-180
 armv=state[:,14:28];a=np.gradient(armv,rows[:,1],axis=0);jerk=np.gradient(a,rows[:,1],axis=0)
 carry_complete=not phase_driven or bool(np.any(np.isin(phases,['lower','open','done'])))
 success=bool(carry_complete and np.any(carry) and frac(stable,carry)>=.95 and float(state[carry,2].min())>meta['box_z']+.025)
 metrics=dict(waist_strategy="separated" if meta.get("separate_waist") else "coupled",mode=meta['mode'],mass_kg=meta['mass'],actual_frames=len(t),recorded_duration_s=float(t[-1]-t[0]),pickup_achieved=bool(np.any(lifted[squeeze|carry])),stable_carry=bool(success),bilateral_contact_fraction=frac(bilateral,carry),unsupported_lift_fraction=frac(unsupported&lift,carry),stable_carry_fraction=frac(stable,carry),box_min_z_carry_m=float(state[carry,2].min()) if np.any(carry) else None,box_max_z_m=float(state[:,2].max()),min_base_z_m=float(state[:,5].min()),fallen=bool(np.any(state[:,5]<.5)),torso_relative_yaw_rmse_deg=float(np.sqrt(np.mean(error[t>=5]**2))),torso_relative_yaw_span_deg=float(np.ptp(torso_relative)),physical_self_contact_fraction=float(np.mean(forces[:,3]>0)),max_self_penetration_m=float(-forces[:,4].min()),arm_velocity_p95_rad_s=float(np.percentile(np.abs(armv),95)),arm_jerk_p95_rad_s3=float(np.percentile(np.abs(jerk),95)),left_force_max_N=float(forces[:,0].max()),right_force_max_N=float(forces[:,1].max()),box_final_position_m=state[-1,:3].tolist())
 metrics.update(box_tilt_max_carry_deg=float(np.rad2deg(box_tilts[carry]).max()) if carry.any() else None,box_tilt_final_deg=float(np.rad2deg(box_tilts[-1])))
 metrics['rotation_criteria_deg']=dict(carry=float(np.rad2deg(BOX_TILT_LIMIT_RAD)),landing=float(np.rad2deg(LANDING_TILT_LIMIT_RAD)))
 base_matrices=Rotation.from_quat(rows[:,5:9][:,[1,2,3,0]]).as_matrix()
 base_tilt=np.rad2deg(np.arccos(np.clip(base_matrices[:,2,2],-1,1)))
 metrics.update(physics_realtime_ratio=float((rows[-1,1]-rows[0,1])/(t[-1]-t[0])),initial_box_support_force_mean_N=float(forces[t<2,6].mean()),base_tilt_max_deg=float(base_tilt.max()),balance_lost=bool(np.any(base_tilt>30) or metrics['fallen']),initial_robot_support_force_max_N=float(forces[t<2,5].max()),robot_support_force_max_N=float(forces[:,5].max()),robot_support_contact_fraction=float(np.mean(forces[:,5]>.5)),physics_elapsed_s=float(rows[-1,1]-rows[0,1]),max_record_gap_s=float(np.diff(t).max()),torso_world_yaw_span_deg=float(np.ptp(torso_world)), torso_world_yaw_rmse_deg=float(np.sqrt(np.mean(((torso_world[t>=5]-desired[t>=5]+180)%360-180)**2))), base_xy_max_displacement_m=float(np.linalg.norm(state[:,3:5]-state[0,3:5],axis=1).max()), arm_moving_fraction=float(np.mean(np.max(np.abs(armv),axis=1)>.02)), released_contact_free=bool(np.any(released) and np.all(forces[released,:2]<threshold)))
 release_xy=np.array([meta['box_radius']/2**.5]*2)
 metrics['released_on_target']=bool(metrics['pickup_achieved'] and metrics['stable_carry'] and metrics['released_contact_free'] and np.linalg.norm(state[-1,:2]-release_xy)<.04 and abs(state[-1,2]-meta['box_z'])<.015 and box_tilts[-1]<=LANDING_TILT_LIMIT_RAD)
 metrics['task_success']=bool(meta.get('motion','box')=='box' and metrics['released_on_target'] and not metrics['balance_lost'] and metrics['physical_self_contact_fraction']==0 and metrics['robot_support_contact_fraction']==0)
 patch_file=out/'physics.bin.contacts';patch_forces=None;pad_forces=None
 if patch_file.exists():
  patch_rows=np.fromfile(patch_file,np.float64).reshape(-1,7)
  patch_forces=patch_rows[np.clip(np.searchsorted(patch_rows[:,0],rows[:,0]),0,len(patch_rows)-1),1:]
  simultaneous=np.all(patch_forces[:,[0,1,3,4]]>min(.2,threshold),axis=1)
  metrics['palm_and_tip_contact_fraction_carry']=frac(simultaneous,carry)
  required_palm_force=1.5*meta['mass']*9.81/(2*meta['friction'])
  metrics['palm_support_fraction_carry']=frac(np.all(patch_forces[:,[0,3]]>=required_palm_force,axis=1),carry)
  metrics['palm_tip_force_max_N']=patch_forces[:,[0,1,3,4]].max(axis=0).tolist()
  metrics['other_hand_contact_force_max_N']=patch_forces[:,[2,5]].max(axis=0).tolist()
  contact_fraction=metrics['palm_support_fraction_carry'] if meta.get('contact_target')=='palm_priority' else metrics['palm_and_tip_contact_fraction_carry']
  if meta.get('contact_target') in ('palm_seated','palm_tip_seated'):
   contact_fraction=None
   if (out/'physics.bin.pads').exists():
    pad_rows=np.fromfile(out/'physics.bin.pads',np.float64).reshape(-1,3)
    pad_forces=pad_rows[np.clip(np.searchsorted(pad_rows[:,0],rows[:,0]),0,len(pad_rows)-1),1:]
    palms=patch_forces[:,[0,3]]
    surface_threshold=meta.get('surface_force_N') or threshold
    secondary=patch_forces[:,[1,4]] if meta['contact_target']=='palm_tip_seated' else pad_forces
    supported=np.all(palms>=surface_threshold,axis=1)&np.all(secondary>=surface_threshold,axis=1)&np.all(palms+secondary>=required_palm_force,axis=1)
    if meta.get('palm_primary_load'):
     palm_target=max(required_palm_force,meta.get('palm_preload_force_N') or required_palm_force)
     supported=np.all(palms>=required_palm_force,axis=1)&np.all(secondary>=meta['secondary_force_N'],axis=1)
     metrics['palm_preload_force_N']=palm_target
     metrics['palm_preload_target_fraction_carry']=frac(np.all(palms>=palm_target,axis=1),carry)
    contact_fraction=frac(supported,carry)
    metrics['selected_surface_support_fraction_carry']=contact_fraction
    if meta['contact_target']=='palm_seated':metrics['palm_pad_contact_fraction_carry']=contact_fraction
    metrics['broad_pad_force_max_N']=pad_forces.max(0).tolist()
  metrics['task_success']=metrics['task_success'] and contact_fraction is not None and contact_fraction>=.95
 elif meta.get('contact_target') in ('palm_priority','palm_seated','palm_tip_seated'):
  metrics['task_success']=False  # Palm-priority success requires recorded palm forces.
 hand_current=None;mcp_angles=None;dip_angles=None
 if (out/'hand_compliance_log.json').exists():
  hand=json.loads((out/'hand_compliance_log.json').read_text());hand_current=[];metrics['mcp_compliance']={'error':hand['error'],'sides':[]}
  metrics['task_success']=metrics['task_success'] and hand['error'] is None
  mcp_angles=np.rad2deg(rows[:,2+np.array([m.jnt_qposadr[m.joint(s+'_hand_mcp_compliant').id] for s in ('l','r')])])
  for side in (0,1):
   samples=[r for r in hand['rows'] if r['side']==side]
   stamps=np.array([r['wall_time'] for r in samples]);index=np.clip(np.searchsorted(stamps,rows[:,0],side='right')-1,0,len(stamps)-1)
   current=np.array([r['current_A'] for r in samples]);hand_current.append(current[index])
   active=(stamps>=meta['start_wall'])&(stamps<=meta['start_wall']+meta['duration'])
   metrics['mcp_compliance']['sides'].append(dict(angle_min_deg=float(mcp_angles[:,side].min()),angle_max_deg=float(mcp_angles[:,side].max()),
    modeled_current_max_A=float(abs(current[active]).max()),motor_torque_max_Nm=float(np.max([abs(r['motor_torque_Nm']) for r,keep in zip(samples,active) if keep]))))
  hand_current=1000*np.array(hand_current).T
  if (out/'compliance.json').exists() and 'dip_spring' in json.loads((out/'compliance.json').read_text()):
   dip_angles=np.rad2deg(rows[:,2+np.array([m.jnt_qposadr[m.joint(s+'_hand_dip_spring').id] for s in ('l','r')])])
   contact=np.min(forces[:,:2],axis=1)>.5
   metrics['dip_passive']=dict(min_deg=dip_angles.min(0).tolist(),max_deg=dip_angles.max(0).tolist(),
    median_contact_deg=np.median(dip_angles[contact],axis=0).tolist() if contact.any() else None,
    loaded_target_deg=27.,no_dip_actuator=True)
 torso_display=torso_world if meta.get('world_registered',False) else torso_relative
 (out/'metrics.json').write_text(json.dumps(metrics,indent=2));np.savez_compressed(out/'physics.npz',t=t,qpos=rows[:,2:2+nq],qvel=rows[:,2+nq:2+nq+nv],forces=forces,state=state)
 if metrics_only:
  print(json.dumps(metrics),flush=True);return
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
 gain_label='arm' if meta.get('sim_arm_gains') and len({tuple(v) for v in meta['sim_arm_gains'].values()})==1 else 'wrist'
 hand_pwm=json.loads((out/'compliance.json').read_text()).get('goal_pwm_raw') if (out/'compliance.json').exists() else None
 renderer=mujoco.Renderer(m,720,1280);cam=mujoco.MjvCamera();cam.lookat[:]=[0,0,.85];cam.distance=2.4;cam.azimuth=140;cam.elevation=-15
 if target_poses:
  target_rotations=recorded_world_rotations(meta,inputs)
  cam.lookat[:]=[.12,0,1.05];cam.distance=1.6;cam.elevation=-20
 dst=out/('teleoperation_targets.mp4' if target_poses else 'teleoperation.mp4');ff=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','24','-i','-','-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p',str(dst)],stdin=subprocess.PIPE)
 for sec in np.arange(0,min(meta['duration'],t[-1]),1/24):
  i=min(np.searchsorted(t,sec),len(t)-1);d.qpos[:]=rows[i,2:2+nq];mujoco.mj_forward(m,d);renderer.update_scene(d,camera=cam)
  fi=int(np.clip(np.searchsorted(input_t,t[i],side='right')-1,0,len(inputs)-1)) if target_poses else min(np.searchsorted(input_t,sec),len(inputs)-1)
  frame=inputs[fi];source=frame['payload']
  for hi,(side,color) in enumerate([('left',[0.,1.,1.,1.]),('right',[1.,0.,1.,1.])]):
   point=np.array(frame['world_hand_goals'][side]) if 'world_hand_goals' in frame else base_pos+base_rot.apply(source[side]['pos'])
   mujoco.mjv_initGeom(renderer.scene.geoms[renderer.scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.012]*3),point,np.eye(3).ravel(),np.array(color,dtype=np.float32));renderer.scene.ngeom+=1
  im=Image.fromarray(renderer.render());dr=ImageDraw.Draw(im);dr.rectangle((0,0,1280,194 if dip_angles is not None else 164 if hand_current is not None else 132 if patch_forces is not None else 104),fill=(20,25,30))
  if target_poses:
   for hi,side in enumerate(('left','right')):
    pose_axes(dr,renderer.scene,np.array(frame['world_hand_goals'][side]),target_rotations[fi,hi])
    pose_axes(dr,renderer.scene,d.xpos[hands[hi]],d.xmat[hands[hi]].reshape(3,3),actual=True)
  dr.text((18,8),f"IGRIS | {'ARMS14 + WAIST' if meta.get('separate_waist') else 'UPPER17 IK'} | {meta['mode'].upper()} | {meta['mass']:g} kg | mu {meta['friction']:g} | {gain_label} Kp {meta.get('wrist_kp',30):g} Kd {meta.get('wrist_kd',3):g} | VENDOR PHYSICS",font=font,fill='white')
  dr.text((18,38),f"{sec:05.1f}s  input: {frame['stage']} | torso desired {desired[i]:+.0f} / actual {torso_display[i]:+.1f} deg",font=font,fill='white')
  dr.text((18,68),f"L/R box force {forces[i,0]:.1f}/{forces[i,1]:.1f} N | box support {forces[i,6]:.1f} N | base z {state[i,5]:.2f}m",font=font,fill='white')
  if patch_forces is not None:
   lp,lt,lo,rp,rt,ro=patch_forces[i]
   label='tip'
   if pad_forces is not None and meta.get('contact_target')!='palm_tip_seated':lt,rt=pad_forces[i];label='broad pad'
   dr.text((18,98),f"Palm / {label}: LEFT {lp:.1f} / {lt:.1f} N | RIGHT {rp:.1f} / {rt:.1f} N",font=font,fill='white')
  if hand_current is not None:
   dr.text((18,130),f"MCP L/R {mcp_angles[i,0]:.1f}/{mcp_angles[i,1]:.1f} deg | current model {hand_current[i,0]:.0f}/{hand_current[i,1]:.0f} mA (cap 180) | PWM {hand_pwm}",font=font,fill='white')
  if dip_angles is not None:
   dr.text((18,160),f"Passive DIP L/R {dip_angles[i,0]:.1f}/{dip_angles[i,1]:.1f} deg | loaded geometry 27 deg | spring model, no DIP motor",font=font,fill='white')
  dr.rectangle((0,680,1280,720),fill=(20,25,30));dr.text((18,685),f"Measured physics | cyan/magenta: task EE goals | free base | waist: {'hold' if meta['hold_waist'] else 'prescribed HMD yaw' if meta.get('separate_waist') else 'HMD yaw IK'}",font=font,fill='white')
  if target_poses:
   dr.rectangle((0,606,1280,720),fill=(20,25,30))
   box_tilt=np.rad2deg(np.arccos(np.clip(d.xmat[bid][8],-1,1)))
   face=meta.get('hand_diagonal_deg',[0.,0.])
   dr.text((18,613),f"Outcome: {'SUCCESS' if metrics['task_success'] else 'FAILED'} | face rotation L/R {face[0]:+.0f}/{face[1]:+.0f} deg | box tilt {box_tilt:.1f} deg",font=font,fill=(120,255,160) if metrics['task_success'] else (255,145,130))
   dr.text((18,648),'RGB = XYZ | long: requested EE pose | short pale: actual EE FK | cyan: left / magenta: right',font=font,fill='white')
   dr.text((18,683),f"Inward target / hand: {1000*frame['task_value'][1]:+.1f} mm | axes drawn over surfaces | recorded free-base physics",font=font,fill='white')
  ff.stdin.write(im.tobytes())
 ff.stdin.close();assert ff.wait()==0;renderer.close()
 print(json.dumps(metrics),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--metrics-only',action='store_true');p.add_argument('--target-poses',action='store_true');a=p.parse_args();process(a.out.resolve(),a.metrics_only,a.target_poses)
