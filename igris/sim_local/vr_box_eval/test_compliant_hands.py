"""Regression checks for MCP sign, current boundary and articulated native geometry."""
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

import evaluate
from compliant_hands import CompliantHands, MCPCompliance, REST, T288SupplyModel, DIP_STIFFNESS
from igris_ik.assets import FIXED_HAND_JOINT_VALUES, load_visual_urdf
from render_results import model_for


def test_compliance_yields_under_load_and_enforces_runtime_limits():
    c=MCPCompliance()
    for _ in range(150):
        reference,current,estimate=c.update(REST,0,.2,0,.02)
    assert np.deg2rad(28)<reference<np.deg2rad(29)
    assert estimate.contact and estimate.tau_external_filtered<0
    assert 0<current<.18
    for _ in range(150):reference,_,estimate=c.update(reference,0,0,0,.02)
    assert abs(reference-REST)<.006 and not estimate.contact
    with pytest.raises(RuntimeError,match='current limit'):
        c.update(REST,0,.7,0,.02)


def test_supply_current_is_not_linear_winding_current():
    model=T288SupplyModel()
    assert abs(model.current(.92,0)-.8)<1e-10
    assert abs(model.evaluate(.18)-.436394317)<1e-8
    for speed in (-.5,0,.5):
        model.velocity=speed
        for torque in (-.4,-.1,.1,.4):
            assert abs(model.evaluate(model.current(torque,speed))-torque)<1e-10
    profile=MCPCompliance().profile
    assert abs(model.drive_torque(REST,0,0,profile)-.92*133/885)<1e-10
    for speed in (-2.,0,2.):
        torque=model.drive_torque(REST,-1.,speed,profile)
        assert abs(model.current(torque,speed))<=profile.current_limit_A+1e-10


def test_physical_mcp_yields_and_returns_without_state_overwrites():
    m=mujoco.MjModel.from_xml_string('''<mujoco><option timestep="0.0005" gravity="0 0 0"/>
      <worldbody><body><joint type="hinge" axis="0 0 1" range="0 60" limited="true"/>
      <inertial mass="0.1" pos="0 .035 0" diaginertia=".0001 .0001 .0001"/></body></worldbody></mujoco>''')
    d=mujoco.MjData(m);d.qpos[0]=REST
    c=MCPCompliance();reference=REST;angles=[];currents=[]
    for k in range(14000):
        torque=c.calibration.drive_torque(reference,d.qpos[0],d.qvel[0],c.profile)
        currents.append(abs(c.calibration.current(torque,d.qvel[0])))
        if k%40==0:reference,_,_=c.update(d.qpos[0],d.qvel[0],torque,0,.02)
        d.qfrc_applied[0]=torque+(-.1 if 2000<=k<8000 else 0.)
        mujoco.mj_step(m,d);angles.append(d.qpos[0])
    assert np.mean(angles[6000:8000])<REST-.02
    assert abs(d.qpos[0]-REST)<.02
    assert max(currents)<=.18+1e-10


def test_articulation_preserves_mass_vendor_axes_and_urdf_mcp_fk():
    previous=dict(FIXED_HAND_JOINT_VALUES);control=CompliantHands()
    try:
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);evaluate.scene(1.,out,1.2)
            old=ET.parse(out/'scene.xml').getroot()
            actuators=ET.tostring(old.find('actuator'))
            control.prepare(out);model=model_for(out);data=mujoco.MjData(model)
            assert model.nu==93
            assert ET.tostring(ET.parse(out/'scene.xml').getroot().find('actuator'))==actuators
            u=load_visual_urdf()
            for angle in (0.,REST,np.pi/3):
                values={**FIXED_HAND_JOINT_VALUES,**{s+'_hand_dof_joint_mcp':angle for s in ('l','r')}}
                u.update_cfg(values)
                for side in ('l','r'):
                    palm=model.body(side+'_hand_rcv').id;finger=model.body(side+'_hand_mcp').id
                    tip=model.body(side+'_hand_dip').id;dip=model.joint(side+'_hand_dip_spring').id
                    joint=model.joint(side+'_hand_mcp_compliant').id
                    data.qpos[model.jnt_qposadr[joint]]=angle
                    assert abs(model.body_mass[palm]+model.body_mass[finger]+model.body_mass[tip]-.6)<1e-10
                    assert not model.jnt_limited[dip], 'Unknown DIP hard stops must not provide artificial grasp force'
                    assert all(abs(key.qpos[model.jnt_qposadr[joint]]-REST)<1e-10 for key in (model.key(i) for i in range(model.nkey)))
                    mujoco.mj_forward(model,data)
                    relative=np.eye(4);R=data.xmat[palm].reshape(3,3)
                    relative[:3,:3]=R.T@data.xmat[finger].reshape(3,3)
                    relative[:3,3]=R.T@(data.xpos[finger]-data.xpos[palm])
                    np.testing.assert_allclose(relative,u.get_transform(side+'_hand_본체1_7',side+'_hand_root'),atol=1e-8)
                    if angle==REST:
                        def clouds_at_current_pose():
                            clouds=[]
                            for suffix in ('___1_5','___1_6','___1_4'):
                                points=[]
                                for g in np.flatnonzero(np.isin(model.geom_bodyid,[palm,finger,tip])):
                                    mesh=model.geom_dataid[g]
                                    if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH or suffix not in model.mesh(mesh).name:continue
                                    v=model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
                                    points.append((v@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]-data.xpos[palm])@data.xmat[palm].reshape(3,3))
                                clouds.append(np.concatenate(points))
                            return clouds
                        clouds=clouds_at_current_pose()
                        rotation,pivot=evaluate.palm_first_rotation(clouds,-45. if side=='l' else 45.)
                        surface=rotation.apply(clouds[0][clouds[0][:,2]>.05])[:,0].max()
                        assert abs(surface-max(rotation.apply(c)[:,0].max() for c in clouds[1:])-.005)<1e-8
                        assert abs(rotation.apply(pivot)[0]-surface)<1e-8
                        data.qpos[model.jnt_qposadr[dip]]=np.deg2rad(27.)
                        mujoco.mj_forward(model,data);clouds=clouds_at_current_pose()
                        initial,tilt,center,_,patches=evaluate.palm_tip_alignment(*clouds,roll_deg=-45. if side=='l' else 45.)
                        assert np.rad2deg((initial.inv()*tilt).magnitude())<3.
                        for points in (clouds[0][clouds[0][:,2]>.05],clouds[2]):
                            assert abs(tilt.apply(points)[:,0].max()-center[0])<1e-8
                        from shapely.geometry import box
                        face=box(center[1]-.075,center[2]-.075,center[1]+.075,center[2]+.075)
                        assert all(p.intersection(face).area>=100e-6 for p in patches)
                        assert abs(patches[0].centroid.y+.01-center[2])<1e-8
                        _,horizontal,center,_,patches=evaluate.palm_tip_alignment(*clouds,roll_deg=0.)
                        np.testing.assert_allclose(horizontal.as_rotvec()[[0,2]],0.,atol=1e-12)
                        face=box(center[1]-.075,center[2]-.075,center[1]+.075,center[2]+.075)
                        assert all(p.intersection(face).area>=100e-6 for p in patches)
                    for dip_angle in (0.,np.deg2rad(27),np.pi/3):
                        data.qpos[model.jnt_qposadr[dip]]=dip_angle
                        u.update_cfg({**values,side+'_hand_dof_joint_dip':dip_angle});mujoco.mj_forward(model,data)
                        R=data.xmat[palm].reshape(3,3);relative[:3,:3]=R.T@data.xmat[tip].reshape(3,3);relative[:3,3]=R.T@(data.xpos[tip]-data.xpos[palm])
                        np.testing.assert_allclose(relative,u.get_transform(side+'_hand_본체1_4',side+'_hand_root'),atol=1e-8)
                        assert abs(data.qfrc_passive[model.jnt_dofadr[dip]]+DIP_STIFFNESS*dip_angle)<1e-9
                    data.qpos[model.jnt_qposadr[dip]]=0.;u.update_cfg(values)
    finally:control.close()
    assert FIXED_HAND_JOINT_VALUES==previous


def test_palm_first_seating_starts_at_contact_but_lift_requires_load_support():
    from contact_carry import ContactCarry
    c=ContactCarry(1.,1.2,palm_first=True);c.value[0]=19.;c.transition('squeeze')
    for _ in range(80):
        value=c.update(.02,fresh=True,forces=[.7,.7],patches=[.7,0,.7,0],box_lift=0,supported=True)
        c.acknowledge(value,True)
    assert c.phase=='seat_palm' and c.value[2]==0 and np.all(c.value[4:]>0)
    c.value[4:]=.2
    preload=c.value[1]
    value=c.update(.02,fresh=True,forces=[.7,.7],patches=[.7,0,.7,0],box_lift=0,supported=True)
    assert value[1]==preload and np.all(value[4:]>c.value[4:])
    for _ in range(120):
        value=c.update(.02,fresh=True,forces=[1.4,1.4],patches=[.7,.7,.7,.7],box_lift=0,supported=True)
        c.acknowledge(value,True)
    assert c.phase=='seat_palm' and c.value[2]==0 and np.all(c.value[4:]<=c.seat_limit)


def test_hand_operating_envelope_stops_without_clipping_angles():
    from compliant_hands import check_hand_angles
    check_hand_angles(np.deg2rad([20.,40.]),np.deg2rad([-10.,45.]))
    for mcp,dip in [([19.,30.],[0.,27.]),([30.,30.],[0.,46.]),([30.,float('nan')],[0.,0.])]:
        with pytest.raises(RuntimeError,match='validated IK geometry'):
            check_hand_angles(np.deg2rad(mcp),np.deg2rad(dip))


def test_five_centimeter_preload_is_a_ceiling_not_a_required_force():
    from contact_carry import ContactCarry
    c=ContactCarry(1.,1.2,palm_first=True,preload_limit_m=.05)
    c.phase='squeeze';c.value[0]=19.;c.value[1]=.04999
    assert c.update(.05,fresh=True,forces=[0.,0.],patches=[0.]*4,box_lift=0.,supported=True)[1]==.05
    for limit in [0.,.05001,float('nan')]:
        with pytest.raises(ValueError,match='preload limit'):
            ContactCarry(1.,preload_limit_m=limit)


def test_tip_recovery_uses_preload_after_tilt_is_exhausted():
    from contact_carry import ContactCarry
    c=ContactCarry(1.,1.2,palm_first=True,preload_limit_m=.05)
    c.force[:]=[8.,8.];c.patch[:]=[7.9,.1,7.3,.7]
    q=c.value.copy();q[1]=.015
    c.restore_surface_contact(q,.02)
    assert q[1]==.015 and q[4]>0
    q[4]=c.seat_limit;c.restore_surface_contact(q,.02)
    assert .015<q[1]<=.05


def test_compliant_carry_waits_for_actual_support_clearance():
    from contact_carry import ContactCarry
    c=ContactCarry(1.,1.2,palm_first=True);c.phase='lift';c.value[2]=.08
    c.force[:]=[8.,8.];c.patch[:]=[7.,1.,7.,1.];c.stable=.5
    q=c.update(.02,fresh=True,forces=[8.,8.],patches=[7.,1.,7.,1.],box_lift=.03,supported=True)
    assert c.phase=='lift' and .08<q[2]<=.12
    c.acknowledge(q,True)
    q=c.update(.02,fresh=True,forces=[8.,8.],patches=[7.,1.,7.,1.],box_lift=.04,supported=False)
    assert c.phase=='carry_left'
    c.acknowledge(q,True);yaw=q[3]
    q=c.update(.02,fresh=True,forces=[8.,8.],patches=[7.,1.,7.,1.],box_lift=.03,supported=True)
    assert q[3]==yaw and q[2]>c.value[2]


def test_probe_clears_tilted_box_corner_before_carrying():
    from contact_carry import ContactCarry
    c=ContactCarry(1.,1.2,palm_first=True);c.phase='probe_lift';c.value[2]=.04
    c.force[:]=[10.,10.];c.patch[:]=[9.5,.5,9.5,.5];c.stable=.5
    q=c.update(.02,fresh=True,forces=[10.,10.],patches=[9.5,.5,9.5,.5],box_lift=.012,supported=True)
    assert c.phase=='probe_lift' and .04<q[2]<=.08 and q[3]==c.value[3]
    c.acknowledge(q,True)
    c.update(.02,fresh=True,forces=[10.,10.],patches=[9.5,.5,9.5,.5],box_lift=.02,supported=False)
    assert c.phase=='lift'
    c.phase='probe_lift';c.value[2]=.08;c.probe_settled=2.
    c.update(.02,fresh=True,forces=[10.,10.],patches=[9.5,.5,9.5,.5],box_lift=.012,supported=True)
    assert c.phase=='failed' and c.reason=='probe_lift_not_secure'


def test_rejected_lift_can_relieve_excess_preload_without_advancing_motion():
    from contact_carry import ContactCarry
    c=ContactCarry(1.,1.2,palm_first=True);c.phase='probe_lift';c.value[1:3]=[.02,.018]
    c.force[:]=[8.,8.];c.patch[:]=[7.,1.,7.,1.]
    c.acknowledge(c.value,False)
    q=c.update(.02,fresh=True,forces=[8.,8.],patches=[7.,1.,7.,1.],box_lift=.003,supported=True)
    assert q[1]==.0195 and np.array_equal(q[2:],c.value[2:])
    c.acknowledge(q,True);assert not c.relieve_preload
    c.force[:]=[6.2,6.2];c.patch[:]=[5.6,.6,5.6,.6];c.acknowledge(c.value,False)
    q=c.update(.02,fresh=True,forces=[6.2,6.2],patches=[5.6,.6,5.6,.6],box_lift=.003,supported=True)
    assert q[1]==c.value[1], 'Insufficient force reserve must not permit pressure relief'


def test_passive_tip_is_auxiliary_but_palms_must_support_the_full_load():
    from contact_carry import ContactCarry
    c=ContactCarry(1.,1.2,palm_first=True);c.phase='seat_palm'
    c.force[:]=[7.3,7.3];c.patch[:]=[7.,.3,7.,.3]
    for _ in range(20):
        q=c.update(.02,fresh=True,forces=[7.3,7.3],patches=[7.,.3,7.,.3],box_lift=0,supported=True)
        c.acknowledge(q,True)
    assert c.phase=='probe_lift'
    c.phase='seat_palm';c.stable=0;c.force[:]=[7.3,7.3];c.patch[:]=[5.9,1.4,5.9,1.4]
    for _ in range(20):
        q=c.update(.02,fresh=True,forces=[7.3,7.3],patches=[5.9,1.4,5.9,1.4],box_lift=0,supported=True)
        c.acknowledge(q,True)
    assert c.phase=='seat_palm'


def test_extra_palm_force_preserves_payload_requirement_and_tip_gate():
    from contact_carry import ContactCarry
    assert ContactCarry(1.,1.2,palm_first=True,palm_force_N=1.).preload_force>6.
    c=ContactCarry(1.,1.2,palm_first=True,palm_force_N=20.);c.phase='seat_palm'
    c.force[:]=[12.,12.];c.patch[:]=[11.5,.5,11.5,.5]
    for _ in range(20):
        q=c.update(.02,fresh=True,forces=[12.,12.],patches=[11.5,.5,11.5,.5],box_lift=0,supported=True);c.acknowledge(q,True)
    assert c.phase=='seat_palm' and q[2]==0 and c.preload_force==20.
    for bad in [-1.,float('nan'),float('inf')]:
        with pytest.raises(ValueError,match='palm force'):ContactCarry(1.,palm_first=True,palm_force_N=bad)


def test_sim_pwm_headroom_keeps_current_and_external_torque_limits():
    from compliant_hands import MCPCompliance
    original,experiment=MCPCompliance(),MCPCompliance(pwm400=True)
    assert original.profile.goal_pwm_raw==133 and experiment.profile.goal_pwm_raw==400
    assert original.profile.current_limit_A==experiment.profile.current_limit_A==.18
    assert original.profile.external_torque_limit_Nm==experiment.profile.external_torque_limit_Nm
    torques=[]
    for c in (original,experiment):
        torque=c.calibration.drive_torque(.2,0.,0.,c.profile)
        assert abs(c.calibration.current(torque,0.))<=.18
        torques.append(torque)
    assert torques[1]>2.9*torques[0]


def test_release_uses_stable_physical_landing_despite_virtual_height_error():
    from contact_carry import ContactCarry
    c=ContactCarry(1.,1.2,palm_first=True);c.phase='lower';c.value[2]=.025
    for _ in range(24):
        c.update(.02,fresh=True,forces=[6.,6.],patches=[5.7,.3,5.7,.3],box_lift=.004,supported=True)
    assert c.phase=='lower'
    c.update(.02,fresh=False,forces=[6.,6.],patches=[5.7,.3,5.7,.3],box_lift=.004,supported=True)
    assert c.landing_stable==0
    for _ in range(26):
        c.update(.02,fresh=True,forces=[6.,6.],patches=[5.7,.3,5.7,.3],box_lift=.004,supported=True)
    assert c.phase=='open' and c.value[2]==.025
    c.value[1]=.021;c.step_scale=.01;c.retry_delay=.15
    q=c.update(.02,fresh=True,forces=[6.,6.],patches=[5.7,.3,5.7,.3],box_lift=.004,supported=True)
    assert abs(q[1]-(.021-.024*.02))<1e-10
    assert c.value[1]==.021, 'Outward motion still requires an IK acknowledgement'
