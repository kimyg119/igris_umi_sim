"""MCP articulation and HAND admittance adapter for the vendor simulator only."""
from dataclasses import asdict, replace
import argparse
import gc
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

import evaluate
from igris_ik.assets import FIXED_HAND_JOINT_VALUES, load_visual_urdf

HAND_ROOT = evaluate.ROOT.parent / 'HAND'
sys.path.insert(0, str(HAND_ROOT))
from compliance_control.admittance import Admittance1D
from compliance_control.config import ControllerConfig
from compliance_control.estimator import ExternalTorqueEstimator
from compliance_control.models import MotionDirection
from compliance_control.profile import ComplianceProfile

REST = np.deg2rad(30.)
MCP_ENVELOPE = np.deg2rad([20.,40.])  # monitored operating range around rest, not physical stops
DIP_STIFFNESS = 193e9 * .0008**4 / (64 * .0042 * 6)  # coil-only beam estimate, Nm/rad
DIP_ENVELOPE = np.deg2rad([-10.,45.])  # validation range, not invented mechanical stops


def check_hand_angles(mcp,dip):
    for name,values,bounds in [('MCP',mcp,MCP_ENVELOPE),('DIP',dip,DIP_ENVELOPE)]:
        values=np.asarray(values)
        if not np.isfinite(values).all() or np.any(values<bounds[0]) or np.any(values>bounds[1]):
            raise RuntimeError(f'{name} left the validated IK geometry range')


class T288SupplyModel:
    """Uncalibrated quasi-static DC model, matched to T288's 11.1 V stall point.

    XC330 reports supply current, not winding current. Copper loss is quadratic
    in output torque. Ideal gear/driver efficiency and zero idle draw are assumed;
    these are optimistic approximations, not a measured force calibration.
    """
    voltage=11.1
    stall_torque=.92
    stall_current=.80
    no_load_speed=65.*2*np.pi/60
    position_p_raw=900
    velocity=0.

    def current(self,torque,velocity):
        magnitude=self.stall_current*(torque/self.stall_torque)**2+max(torque*velocity,0.)/self.voltage
        return float(np.copysign(magnitude,torque))

    def evaluate(self,current,direction=None):
        a=self.stall_current/self.stall_torque**2
        b=max(float(np.sign(current))*self.velocity,0.)/self.voltage
        return float(np.sign(current))*(-b+np.sqrt(b*b+4*a*abs(current)))/(2*a)

    def drive_torque(self,reference,q,dq,profile):
        voltage_torque=self.stall_torque*(self.position_p_raw/128)*(4096/(2*np.pi))*(reference-q)/885
        pwm_torque=self.stall_torque*profile.goal_pwm_raw/885
        requested=np.clip(voltage_torque,-pwm_torque,pwm_torque)-self.stall_torque/self.no_load_speed*dq
        self.velocity=dq
        limit=abs(self.evaluate(np.copysign(profile.current_limit_A,requested)))
        return float(np.clip(requested,-limit,limit))


class NativeBaseline:
    """Simulation free-load baseline; hardware lookup is mount/friction specific."""
    value = 0.

    def evaluate(self, *args, **kwargs):
        return self.value


class MCPCompliance:
    def __init__(self, pwm400=False):
        self.profile = ComplianceProfile.load(HAND_ROOT/'config/gripper_id001_runtime.json')
        if pwm400:
            self.profile=replace(self.profile,goal_pwm_raw=400,pwm_limit_raw=max(400,self.profile.pwm_limit_raw))
        self.config = ControllerConfig.load(self.profile.controller_config)
        self.config.hardware.model='XC330-T288-T'
        self.calibration = T288SupplyModel()
        self.baseline = NativeBaseline()
        self.estimator = ExternalTorqueEstimator(self.calibration, self.baseline, self.config.filters, self.config.signs)
        self.admittance = Admittance1D(self.config.admittance, self.config.safety)
        self.admittance.reset(REST)
        self.anchor = None

    def update(self, q, dq, torque, free_torque, dt):
        current = round(self.calibration.current(torque,dq)/self.config.units.current_A_per_count)*self.config.units.current_A_per_count
        self.calibration.velocity=dq
        self.baseline.value = free_torque
        estimate = self.estimator.update(q,dq,current,MotionDirection.CLOSE,freeze_direction=True)
        if abs(current)>self.profile.current_limit_A+1e-6:
            raise RuntimeError('MCP current limit exceeded')
        if abs(estimate.tau_external_filtered)>self.profile.external_torque_limit_Nm:
            raise RuntimeError('MCP external torque limit exceeded')
        if estimate.contact and self.anchor is None:
            self.anchor=q;self.admittance.reset(q)
        if self.anchor is None:
            reference=REST
        else:
            reference=self.admittance.update(self.anchor,estimate.tau_external_filtered,dt).reference_position
        return reference, current, estimate


def _pose(element, matrix):
    element.set('pos',' '.join(map(str,matrix[:3,3])))
    element.set('quat',' '.join(map(str,Rotation.from_matrix(matrix[:3,:3]).as_quat()[[3,0,1,2]])))


def articulate(path, dip_preload_deg=0., pwm400=False):
    """Split the 0.6 kg hand into palm, actuated MCP and passive spring DIP."""
    from bake_fixed_hands import bake
    bake(path)
    tree=ET.parse(path);root=tree.getroot();world=root.find('worldbody')
    old_joints=[j for j in world.iter() if j.tag in ('joint','freejoint')]
    def width(j):return 7 if j.tag=='freejoint' or j.get('type')=='free' else 4 if j.get('type')=='ball' else 1
    keys=[]
    for key in root.findall('./keyframe/key'):
        values=np.fromstring(key.get('qpos'),sep=' ');at=0;mapping={}
        for j in old_joints:
            n=width(j);mapping[j.get('name')]=values[at:at+n];at+=n
        assert at==len(values)
        keys.append((key,mapping))
    u=load_visual_urdf();u.update_cfg(FIXED_HAND_JOINT_VALUES)
    meshes={Path(a.get('file')).name:a.get('name') for a in root.findall('./asset/mesh')}
    summary={}
    for side in ('l','r'):
        hand=root.find(f'.//body[@name="{side}_hand_rcv"]')
        joint=u.joint_map[side+'_hand_dof_joint_mcp'];dip_joint=u.joint_map[side+'_hand_dof_joint_dip']
        def descendants(name):
            moving={name}
            while True:
                following=moving|{j.child for j in u.robot.joints if j.parent in moving}
                if following==moving:return moving
                moving=following
        moving=descendants(joint.child);distal=descendants(dip_joint.child)
        frame=u.get_transform(joint.child,side+'_hand_root')
        finger=ET.SubElement(hand,'body',name=side+'_hand_mcp');_pose(finger,frame)
        ET.SubElement(finger,'joint',name=side+'_hand_mcp_compliant',type='hinge',axis=' '.join(map(str,joint.axis)),
                      ref=str(REST),limited='true',range='0 1.0471975511965976',damping='.002',armature='.00005')
        dip_frame=u.get_transform(dip_joint.child,side+'_hand_root')
        tip=ET.SubElement(finger,'body',name=side+'_hand_dip');_pose(tip,np.linalg.inv(frame)@dip_frame)
        ET.SubElement(tip,'joint',name=side+'_hand_dip_spring',type='hinge',axis=' '.join(map(str,dip_joint.axis)),
                      ref='0',limited='false',stiffness=str(DIP_STIFFNESS),springref=str(-np.deg2rad(dip_preload_deg)),
                      damping='.0001')  # small uncalibrated viscous damping; no motor or hard stop
        template=dict(hand.find('geom').attrib)
        for geom in list(hand.findall('geom')):hand.remove(geom)
        parts=[[],[],[]]
        for link in u.robot.links:
            if not link.name.startswith(side+'_hand_') or link.name.endswith('_cam'):continue
            group=2 if link.name in distal else int(link.name in moving);body=(hand,finger,tip)[group]
            transform=u.get_transform(link.name,side+'_hand_root')
            if group:transform=np.linalg.inv(dip_frame if group==2 else frame)@transform
            for visual in link.visuals:
                if visual.geometry.mesh is None:continue
                attrs={**template,'name':side+'_hand_'+link.name[len(side)+6:]+'_geom',
                       'mesh':meshes[Path(visual.geometry.mesh.filename).name]}
                geom=ET.SubElement(body,'geom',attrs);_pose(geom,transform@visual.origin)
            if link.inertial is not None:
                x=transform@link.inertial.origin
                parts[group].append((link.inertial.mass,x[:3,3],x[:3,:3]@link.inertial.inertia@x[:3,:3].T))
        scale=.6/sum(p[0] for group in parts for p in group)
        for body,group in zip((hand,finger,tip),parts):
            mass=sum(p[0] for p in group);com=sum(w*p for w,p,_ in group)/mass
            inertia=sum(I+w*((p-com)@(p-com)*np.eye(3)-np.outer(p-com,p-com)) for w,p,I in group)*scale
            inertial=body.find('inertial')
            if inertial is None:inertial=ET.SubElement(body,'inertial')
            inertial.attrib.clear();inertial.attrib.update(mass=str(mass*scale),pos=' '.join(map(str,com)),
                fullinertia=' '.join(map(str,inertia[[0,1,2,0,0,1],[0,1,2,1,2,2]])))
        summary[side]=dict(fixed_mass_kg=float(hand.find('inertial').get('mass')),
                           moving_mass_kg=float(finger.find('inertial').get('mass')),dip_mass_kg=float(tip.find('inertial').get('mass')),moving_links=sorted(moving))
    for key,mapping in keys:
        for side in ('l','r'):
            mapping[side+'_hand_mcp_compliant']=[REST];mapping[side+'_hand_dip_spring']=[0.]
        key.set('qpos',' '.join(str(v) for j in world.iter() if j.tag in ('joint','freejoint') for v in mapping[j.get('name')]))
    controller=MCPCompliance(pwm400);motor=controller.calibration
    custom=root.find('custom')
    if custom is None:custom=ET.SubElement(root,'custom')
    ET.SubElement(custom,'numeric',name='hand_motor_config',data=f'{REST} {controller.profile.current_limit_A} {motor.stall_torque} {motor.stall_current} {motor.voltage} {motor.position_p_raw} {controller.profile.goal_pwm_raw} {motor.no_load_speed}')
    tree.write(path,encoding='unicode')
    return summary


class CompliantHands:
    def __init__(self, dip_preload_deg=0., pwm400=False):
        if not np.isfinite(dip_preload_deg) or not 0<=dip_preload_deg<=30:raise ValueError('DIP preload sensitivity range is 0..30 degrees')
        self.dip_preload_deg=dip_preload_deg
        self.pwm400=pwm400
        self.stop=threading.Event();self.process=None;self.error=None;self.spheres={}
        self.previous=dict(FIXED_HAND_JOINT_VALUES)
        FIXED_HAND_JOINT_VALUES.update({s+'_hand_dof_joint_mcp':REST for s in ('l','r')})
        FIXED_HAND_JOINT_VALUES.update({s+'_hand_dof_joint_dip':0. for s in ('l','r')})
        # Evaluation-local model override; normal VR/SDK defaults are untouched.
        import igris_ik.model as model
        self.original=model.build_robot_dict
        def build(*args,**kwargs):
            result=self.original(*args,**kwargs)
            result['kinematics']['collision_spheres'].update(self.spheres)
            return result
        model.build_robot_dict=build

    def prepare(self,out):
        self.out=out
        summary=articulate(out/'scene.xml',self.dip_preload_deg,self.pwm400)
        from render_results import model_for
        import mujoco
        from scipy.cluster.vq import kmeans2
        m=model_for(out);d=mujoco.MjData(m)
        for side in ('l','r'):
            body=m.body(side+'_hand_rcv').id;finger=m.body(side+'_hand_mcp').id;tip=m.body(side+'_hand_dip').id
            joint=m.joint(side+'_hand_mcp_compliant').id;dip=m.joint(side+'_hand_dip_spring').id;cloud=[];max_lever=0.;dip_lever=0.
            from itertools import product
            for angle,dip_angle in product(np.linspace(*MCP_ENVELOPE,5),np.linspace(*DIP_ENVELOPE,12)):
                d.qpos[m.jnt_qposadr[joint]]=angle;d.qpos[m.jnt_qposadr[dip]]=dip_angle;mujoco.mj_forward(m,d)
                for g in np.flatnonzero(np.isin(m.geom_bodyid,[body,finger,tip])):
                    if m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
                    mesh=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
                    world=v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
                    cloud.append((world-d.xpos[body])@d.xmat[body].reshape(3,3))
                    if m.geom_bodyid[g] in (finger,tip):
                        max_lever=max(max_lever,float(np.linalg.norm(np.cross(world-d.xanchor[joint],d.xaxis[joint]),axis=1).max()))
                    if m.geom_bodyid[g]==tip:
                        dip_lever=max(dip_lever,float(np.linalg.norm(np.cross(world-d.xanchor[dip],d.xaxis[dip]),axis=1).max()))
            points=np.unique(np.concatenate(cloud).round(6),axis=0)
            centers,labels=kmeans2(points,64,iter=30,minit='++',seed=0)
            padding=.002+2*(max_lever+dip_lever)*np.sin(np.deg2rad(5.)/4)
            spheres=[dict(center=c.tolist(),radius=float(np.linalg.norm(points[labels==i]-c,axis=1).max()+padding)) for i,c in enumerate(centers)]
            self.spheres[side+'_hand_root']=spheres
        c=MCPCompliance(self.pwm400)
        metadata=dict(rest_deg=30.,mcp_validation_range_deg=np.rad2deg(MCP_ENVELOPE).tolist(),profile=str(c.profile.controller_config),current_limit_A=c.profile.current_limit_A,
            external_torque_limit_Nm=c.profile.external_torque_limit_Nm,admittance=asdict(c.config.admittance),
            safety=asdict(c.config.safety),motor_model=dict(name='XC330-T288-T',mcp_ratio=1.,voltage_V=11.1,
                stall_torque_Nm=.92,stall_current_A=.8,status='uncalibrated quasistatic DC supply-current approximation',
                source='https://emanual.robotis.com/docs/en/dxl/x/xc330-t288/',idle_current_A=0.,efficiency=1.),hand_bodies=summary,
            contact_pose_mcp_deg=30.,approach_mcp_deg=30.,contact_surfaces=['palm bump','distal phalanx'],hand_roll_deg=[-45.,45.],
            dip_spring=dict(wire_mm=.8,outer_diameter_mm=5.,active_coils=6,material='SUS304',youngs_modulus_GPa=193.,
                stiffness_Nm_rad=DIP_STIFFNESS,free_leg_angle_deg=180.,loaded_target_deg=27.,
                assumed_unloaded_urdf_deg=0.,preload_deg=self.dip_preload_deg,damping_Nms_rad=.0001,
                status='coil-only estimate; installed preload and leg compliance unmeasured',validation_range_deg=np.rad2deg(DIP_ENVELOPE).tolist()),
            hand_motor_record_width=23,
            position_p_raw=c.calibration.position_p_raw,position_pid_assumption='T288 factory P900 I0 D0; hardware readback not available',
            operating_mode=c.config.hardware.operating_mode,
            current_limit_assumption='ideal torque clamp in simulation; Mode 3 hardware uses the external current-stop guard',
            goal_pwm_raw=c.profile.goal_pwm_raw,no_load_speed_rpm=65.,
            baseline='native qfrc_bias - qfrc_passive; inertia/friction approximation, not hardware baseline',
            current='simulated from bounded motor torque; not measured hardware current',
            ik_collision='monitored MCP 20..40 and DIP -10..45 deg; stop outside envelope',spheres=self.spheres)
        (out/'compliance.json').write_text(json.dumps(metadata,indent=2))
        for path in (Path(__file__),Path(__file__).with_name('hand_motor.cpp'),c.profile.controller_config):
            shutil.copy2(path,out/path.name)

    def start(self,out):
        # cuRobo model construction can hold the GIL longer than the 100 ms hand deadline.
        self.process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--run',str(out.resolve())],
                                      stdout=(out/'hand_controller.log').open('w'),stderr=subprocess.STDOUT)
        deadline=time.monotonic()+5
        while not (out/'hand_reference.txt').exists():
            if self.process.poll() is not None or time.monotonic()>deadline:
                raise RuntimeError('MCP controller did not become ready; see hand_controller.log')
            time.sleep(.02)

    def _run(self):
        pwm400=json.loads((self.out/'compliance.json').read_text())['goal_pwm_raw']==400
        controllers=[MCPCompliance(pwm400),MCPCompliance(pwm400)];last=-1.;rows=[];failure=None;row=None
        # This worker runs for one bounded simulator trial. Its retained log
        # grows throughout the run; cyclic collection may scan it for hundreds
        # of milliseconds. Restore collection when command production ends.
        gc_was_enabled=gc.isenabled();gc.collect();gc.disable()
        previous_wall=None;max_wall_gap=0.
        cycle_timing=[];previous_iteration=time.perf_counter()
        try:
            while not self.stop.wait(.002):
                iteration=time.perf_counter();cpu_start=time.process_time()
                dispatch_gap=iteration-previous_iteration
                row=evaluate.last_record(self.out/'hand_motor.bin',23)
                read_end=time.perf_counter()
                if row[1]==last:
                    previous_iteration=read_end
                    continue
                if not 0<=time.time()-row[0]<.1:raise RuntimeError('stale MCP physics')
                dt=.02 if last<0 or row[1]<last else row[1]-last
                if previous_wall is not None:max_wall_gap=max(max_wall_gap,float(row[0])-previous_wall)
                previous_wall=float(row[0])
                if dt>.1:raise RuntimeError('MCP control period exceeded')
                last=row[1];references=[]
                check_hand_angles(row[[3,9]],row[[15,19]])
                for i,c in enumerate(controllers):
                    q,dq,tau,free,qref,contact_tau=row[3+6*i:9+6*i]
                    reference,current,estimate=c.update(q,dq,tau,free,dt);references.append(reference)
                    rows.append(dict(wall_time=float(row[0]),sim_time=float(row[1]),side=i,q=float(q),dq=float(dq),
                        reference=reference,current_A=current,motor_torque_Nm=float(tau),contact_torque_Nm=float(contact_tau),
                        native_reference=float(qref),free_torque_Nm=float(free),contact_anchor=c.anchor,
                        pwm_limited=bool(abs(c.calibration.position_p_raw/128*4096/(2*np.pi)*(qref-q))>=c.profile.goal_pwm_raw),
                        current_limited=bool(abs(c.calibration.current(tau,dq))>=.99*c.profile.current_limit_A),
                        estimated_external_Nm=estimate.tau_external_filtered,contact=estimate.contact,
                        dip_q=float(row[15+4*i]),dip_dq=float(row[16+4*i]),dip_spring_torque_Nm=float(row[17+4*i]),dip_constraint_torque_Nm=float(row[18+4*i])))
                update_end=time.perf_counter()
                temp=self.out/'hand_reference.tmp';temp.write_text(f'{time.time()} {references[0]} {references[1]}')
                temp.replace(self.out/'hand_reference.txt')
                published=time.perf_counter()
                cycle_timing.append(dict(native_wall=float(row[0]),dispatch_s=dispatch_gap,
                    read_s=read_end-iteration,update_s=update_end-read_end,
                    publish_s=published-update_end,cpu_s=time.process_time()-cpu_start))
                previous_iteration=published
        except Exception as exc:
            self.error=str(exc);(self.out/'stop.request').write_text('hand_compliance: '+self.error)
            failure=dict(reason=self.error,native_motor_row=None if row is None else row.tolist())
        finally:
            if gc_was_enabled:gc.enable()
            (self.out/'hand_compliance_log.json').write_text(json.dumps(dict(error=self.error,failure=failure,
                cyclic_gc_deferred=True,max_native_sample_wall_gap_s=max_wall_gap,rows=rows)))
            (self.out/'hand_cycle_timing.json').write_text(json.dumps(cycle_timing))

    def close(self):
        self.stop.set()
        if self.process is not None:
            self.process.send_signal(signal.SIGINT)
            self.process.wait(timeout=5)
        import igris_ik.model as model
        model.build_robot_dict=self.original
        FIXED_HAND_JOINT_VALUES.update(self.previous)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    runtime=object.__new__(CompliantHands);runtime.out=args.run;runtime.stop=threading.Event();runtime.error=None
    signal.signal(signal.SIGINT,lambda *args:runtime.stop.set())
    signal.signal(signal.SIGTERM,lambda *args:runtime.stop.set())
    runtime._run()
