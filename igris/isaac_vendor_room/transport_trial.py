"""Run one isolated vendor trial; always stop its own container and save failures."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def validate_htd_loaded_rate(value,*,loaded_transport,current_grasp_reference,relaxed_grasp):
    """Only explicit relaxed transport may choose the slower native wall cap."""
    if (isinstance(value,bool) or not isinstance(value,(int,float))
            or not math.isfinite(value) or value not in (.1,.05)):
        raise ValueError('HTD loaded rate must be exactly0.1 or0.05')
    if value==.05 and not all(flag is True for flag in
            (loaded_transport,current_grasp_reference,relaxed_grasp)):
        raise ValueError('HTD loaded rate0.05 requires explicit relaxed SIM transport/current-grasp scope')
    return float(value)


def htd_pair_padding_metadata(enabled,*,loaded_transport,current_grasp_reference,relaxed_grasp,bilateral=False):
    """Reviewed SIM pair selection keeps raw spheres and every other buffer."""
    if type(enabled) is not bool or type(bilateral) is not bool:
        raise ValueError('HL-IK pair padding option must be bool')
    if enabled and bilateral:
        raise ValueError('HL-IK pair padding options are mutually exclusive')
    if not enabled and not bilateral:return None
    if not all(value is True for value in (loaded_transport,current_grasp_reference,relaxed_grasp)):
        raise ValueError('HL-IK pair padding requires explicit relaxed loaded transport/current-grasp SIM scope')
    if bilateral:
        return dict(schema='htd_pelvis_thighs_unbuffered/v2',
            pairs=[['pelvis','l_upper_leg'],['pelvis','r_upper_leg']],
            buffer_per_sphere_m=0.,default_buffer_per_sphere_m=.0025,interior_margin_m=.0005,
            physics_modified=False,evidence=[
                dict(path='outputs/hlik_transport_20260918_2005/native107_collision_seed01/REPORT.md',
                     sha256='8a48dce12b6a50abfed461f2e68d209beafb6e07484edf716ec1ecd00ff2db5a'),
                dict(path='outputs/hlik_transport_20260918_2005/native108_geometry_audit01/left_pair_proof.json',
                     sha256='ef19a1b3c1257a8279a5eb416dbb263c40c4a1c5b43ea4264250606be53e60ef')])
    return dict(schema='htd_pelvis_right_thigh_unbuffered/v1',pair=['pelvis','r_upper_leg'],
        buffer_per_sphere_m=0.,default_buffer_per_sphere_m=.0025,interior_margin_m=.0005,
        physics_modified=False,
        evidence_path='outputs/hlik_transport_20260918_2005/native107_collision_seed01/REPORT.md',
        evidence_sha256='8a48dce12b6a50abfed461f2e68d209beafb6e07484edf716ec1ecd00ff2db5a')


def htd_pair_buffer_overrides(manifest):
    """Validate the serialized opt-in before either host startup or IK assets."""
    if 'hl_ik_pair_padding' not in manifest:return None
    metadata=manifest['hl_ik_pair_padding'];scope=manifest.get('htd_loaded_zero')
    if (not isinstance(scope,dict) or scope.get('enabled') is not True
            or scope.get('scope')!='loaded_transport' or scope.get('current_grasp_reference') is not True
            or manifest.get('hardware_executed') is not False
            or manifest.get('physics_backend')!='vendor MuJoCo 3.2.7'
            or manifest.get('evaluation_profile')!='relaxed_transport'):
        raise ValueError('HL-IK pair padding requires explicit relaxed loaded transport/current-grasp SIM manifest')
    bilateral=isinstance(metadata,dict) and metadata.get('schema')=='htd_pelvis_thighs_unbuffered/v2'
    expected=htd_pair_padding_metadata(not bilateral,bilateral=bilateral,
        loaded_transport=True,current_grasp_reference=True,relaxed_grasp=True)
    if (not isinstance(metadata,dict) or metadata!=expected
            or metadata.get('physics_modified') is not False
            or any(type(metadata.get(k)) is not float for k in
                   ('buffer_per_sphere_m','default_buffer_per_sphere_m','interior_margin_m'))):
        raise ValueError('HL-IK pair padding metadata differs from the reviewed exact pair selection')
    return {tuple(pair):0. for pair in expected['pairs']} if bilateral else {('pelvis','r_upper_leg'):0.}


def configure_htd_pair_padding(trial,metadata):
    if metadata is None:return None
    path=Path(trial)/'manifest.json';manifest=json.loads(path.read_text())
    manifest['hl_ik_pair_padding']=metadata
    htd_pair_buffer_overrides(manifest)
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    return metadata


def validate_dip_damping_option(value,relaxed_grasp,loaded_transport):
    if value is None:return None
    if (type(value) not in (int,float) or not math.isfinite(value) or value!=.002
            or not relaxed_grasp or not loaded_transport):
        raise ValueError('DIP damping requires the reviewed0.002 SIM relaxed loaded-transport scope')
    return value


def relaxed_requirements(mass,mu,target=3.):
    if (not all(map(math.isfinite,(mass,mu,target))) or min(mass,mu,target)<=0
        or target>14. or target<1.5*mass*9.81/(2*mu)):
        raise ValueError('Relaxed grasp needs finite positive mass/friction and target within friction reserve..14 N')
    return dict(target_force_N=float(target),tip_min_force_N=0.,friction_safety_factor=1.5)


def validate_prepared_payload(trial,requested_mass):
    """Check the generated native mass before startup, including fixed layouts."""
    import xml.etree.ElementTree as ET
    trial=Path(trial);p=json.loads((trial/'manifest.json').read_text())['payload']
    tree=ET.parse(trial/'scene.xml');body=tree.find(f'.//body[@name="{p["body"]}"]')
    if body is None or body.find('freejoint') is None or body.find('inertial') is not None:
        raise ValueError('Prepared payload mass requires the original single free geom')
    geoms=body.findall('geom')
    if len(geoms)!=1 or geoms[0].get('name')!=p['geom']:
        raise ValueError('Prepared payload mass geom differs from manifest')
    actual=float(geoms[0].get('mass','nan'))
    if not all(math.isclose(float(v),requested_mass,rel_tol=1e-7,abs_tol=1e-9)
               for v in (p['mass_kg'],actual)):
        raise ValueError('Prepared payload mass differs from explicit requested mass')
    return dict(requested_mass_kg=requested_mass,manifest_mass_kg=p['mass_kg'],native_geom_mass_kg=actual)


def validate_prepared_inplace_contacts(manifest,audit,scene_sha,mu,pair_time,feet,hand_time,spin):
    pair=manifest['pair_friction_solver'];palm=manifest['palm_torsion']
    if (audit['mu']!=mu or manifest['contact_material']['mu']!=mu
            or pair['friction_time_constant_s']!=pair_time
            or pair['hand_friction_time_constant_s']!=hand_time
            or pair['foot_pair_count']!=(2 if feet else 0)
            or palm['coefficient_m']!=spin
            or pair['original_scene_sha256']!=audit['scene_sha256']
            or palm['original_scene_sha256']!=pair['scene_sha256']
            or palm['scene_sha256']!=scene_sha):
        raise ValueError('Prepared in-place contact profile or scene hash chain differs')


def run(out,motion,mu=.6,destination=(.33,.20),physics_cpu=None,live_model=None,live_acceleration=3.,force_grasp=False,
        live_limit_cost=None,live_soft_margin=None,collision_braking=False,source_base_clearance=0.,source_base_normal=0.,track_source=False,live_lead_steps=3,contact_noslip_iterations=0,pair_friction_time=None,refine_foot_friction=False,
        orientation_feedback=False,grasp_travel=.015,hand_friction_time=None,track_source_rotation=False,narrow_source_support=False,contact_impratio=None,synchronized_grasp=False,box_orientation_feedback=False,live_torso_filter_seconds=.4,simulation_rate_limit=None,native_collision_contacts=False,payload_mass=1.,relaxed_grasp=False,walk_to_rolltainer=False,pickup_stop_phase=110.,grasp_height_adjustment=0.,native_policy_clock=False,carry_clearance_lift=0.,simulation_rate_start_phase=0.,narrow_destination_support=False,palm_torsion_m=None,grasp_force_target=None,torso_relative_lowering=False,world_position_lowering=False,lowering_acceleration_limit=None,inplace_transfer=False,inplace_proposal=None,heading_cart_proposal=None,xy_first_lowering=False,arrival_settle_seconds=1.5,htd_loaded_zero=False,clearance_minimum_rise=None,arrival_brake_radius=None,htd_current_grasp_reference=False,htd_command_observer=False,htd_wrist_mode_entry=False,htd_loaded_goal_hold=False,htd_contact_grace=False,htd_policy_package=None,htd_loaded_transport=False,dip_damping=None,htd_posture_domain_diagnostic=False,htd_pelvis_right_thigh_unbuffered=False,htd_pelvis_thighs_unbuffered=False,htd_loaded_rate=.1,htd_upper_backlash_timeconst=None,htd_prealignment_coast_seconds=None):
    if out.exists():raise FileExistsError(out)
    if htd_prealignment_coast_seconds is not None:
        from htd_loaded import validate_prealignment_coast_option
        htd_prealignment_coast_seconds=validate_prealignment_coast_option(htd_prealignment_coast_seconds,
            enabled=htd_loaded_zero,transport=htd_loaded_transport,current_grasp_reference=htd_current_grasp_reference,
            command_observer=htd_command_observer,wrist_mode_entry=htd_wrist_mode_entry,goal_hold=htd_loaded_goal_hold,
            contact_grace=htd_contact_grace,relaxed_grasp=relaxed_grasp)
    if htd_upper_backlash_timeconst is not None:
        from htd_loaded import validate_upper_backlash_option
        htd_upper_backlash_timeconst=validate_upper_backlash_option(htd_upper_backlash_timeconst,
            enabled=htd_loaded_zero,transport=htd_loaded_transport,current_grasp_reference=htd_current_grasp_reference,
            command_observer=htd_command_observer,wrist_mode_entry=htd_wrist_mode_entry,goal_hold=htd_loaded_goal_hold,
            contact_grace=htd_contact_grace,relaxed_grasp=relaxed_grasp)
    htd_loaded_rate=validate_htd_loaded_rate(htd_loaded_rate,loaded_transport=htd_loaded_transport,
        current_grasp_reference=htd_current_grasp_reference,relaxed_grasp=relaxed_grasp)
    pair_padding=htd_pair_padding_metadata(htd_pelvis_right_thigh_unbuffered,
        bilateral=htd_pelvis_thighs_unbuffered,
        loaded_transport=htd_loaded_transport,current_grasp_reference=htd_current_grasp_reference,relaxed_grasp=relaxed_grasp)
    if type(htd_posture_domain_diagnostic) is not bool:
        raise ValueError('Posture domain diagnostic must be bool')
    if htd_posture_domain_diagnostic and not (htd_loaded_transport is True and htd_current_grasp_reference is True and htd_policy_package is not None):
        raise ValueError('Posture domain diagnostic requires explicit V2 transport/current-grasp scope')
    validate_dip_damping_option(dip_damping,relaxed_grasp,htd_loaded_transport)
    if htd_loaded_transport and not (htd_loaded_goal_hold and htd_contact_grace and heading_cart_proposal is not None and xy_first_lowering):
        raise ValueError('HTD transport requires explicit loaded goal/retention and reviewed XY-first heading route')
    from htd_candidate_package import require_candidate_scope, resolve_package, require_posture_domain_diagnostic
    require_candidate_scope(htd_policy_package, enabled=htd_loaded_zero,
        current_grasp_reference=htd_current_grasp_reference, command_observer=htd_command_observer,
        wrist_mode_entry=htd_wrist_mode_entry, goal_hold=htd_loaded_goal_hold)
    selection=None
    if htd_policy_package is not None:
        selection=resolve_package(htd_policy_package)  # All descriptor/artifact checks before preparation/mutations.
    if htd_prealignment_coast_seconds is not None:
        from htd_loaded import validate_prealignment_coast_policy
        validate_prealignment_coast_policy(selection)
    require_posture_domain_diagnostic(htd_posture_domain_diagnostic,selection,
        transport=htd_loaded_transport,current_grasp_reference=htd_current_grasp_reference)
    if htd_contact_grace and not htd_loaded_goal_hold:
        raise ValueError('Contact grace requires explicit goal hold scope')
    if htd_loaded_goal_hold and not (htd_loaded_zero and htd_current_grasp_reference and htd_command_observer and htd_wrist_mode_entry):
        raise ValueError('Goal hold requires observed current-grasp wrist mode entry scope')
    if htd_wrist_mode_entry and not (htd_loaded_zero and htd_current_grasp_reference and htd_command_observer):
        raise ValueError('Observed current-grasp loaded scope required for wrist mode entry')
    if htd_command_observer and not htd_loaded_zero:
        raise ValueError('Command observer requires explicit HTD loaded-zero experiment')
    if htd_current_grasp_reference and not htd_loaded_zero:
        raise ValueError('Current grasp reference requires explicit HTD loaded-zero experiment')
    if arrival_brake_radius is not None and (heading_cart_proposal is None or htd_loaded_zero
            or not relaxed_grasp or not math.isfinite(arrival_brake_radius) or not .015<=arrival_brake_radius<=.025):
        raise ValueError('Explicit15..25mm arrival braking requires relaxed heading cart; excludes stationary HTD')
    if clearance_minimum_rise is not None and (clearance_minimum_rise not in ((.015,.02) if htd_loaded_goal_hold else (.02,)) or carry_clearance_lift not in ((.04,.05) if htd_loaded_goal_hold else (.04,))
            or not relaxed_grasp or heading_cart_proposal is None or (htd_loaded_zero and not htd_current_grasp_reference)):
        raise ValueError('Explicit minimum clearance20mm (15mm goal-hold only) requires reviewed40mm heading route or explicit current-grasp HTD stationary scope')
    if htd_loaded_zero:
        from htd_loaded_setup import validate_options
        validate_options(live_model=live_model,walk_to_rolltainer=walk_to_rolltainer,inplace_transfer=inplace_transfer,
            force_grasp=force_grasp,relaxed_grasp=relaxed_grasp,payload_mass=payload_mass,mu=mu,
            carry_clearance_lift=carry_clearance_lift,simulation_rate_limit=simulation_rate_limit,
            simulation_rate_start_phase=simulation_rate_start_phase,goal_hold=htd_loaded_goal_hold)
    if inplace_transfer and walk_to_rolltainer:raise ValueError('Select one transfer controller')
    if xy_first_lowering and not (heading_cart_proposal is not None and world_position_lowering):
        raise ValueError('XY-first lowering requires heading cart and world-position lowering')
    if (not math.isfinite(arrival_settle_seconds) or not .3<=arrival_settle_seconds<=1.5
            or arrival_settle_seconds!=1.5 and heading_cart_proposal is None):
        raise ValueError('Short arrival dwell requires heading cart and0.3..1.5seconds')
    if inplace_proposal is not None and not inplace_transfer:raise ValueError('Near proposal requires in-place transfer')
    if inplace_transfer and (not relaxed_grasp or not world_position_lowering or payload_mass!=.1 or mu!=4. or palm_torsion_m!=.02 or carry_clearance_lift!=.04
                            or pickup_stop_phase!=106. or grasp_force_target!=6. or grasp_height_adjustment!=-.04):
        raise ValueError('In-place profile requires reviewed .1kg/mu4/spin.02/40mm-clearance hybrid-lowering configuration')
    if heading_cart_proposal is not None and (not walk_to_rolltainer or not narrow_destination_support or not relaxed_grasp
            or not world_position_lowering or payload_mass not in ((.01,.1) if htd_loaded_goal_hold else (.1,)) or mu!=4. or palm_torsion_m!=.02 or carry_clearance_lift not in ((.03,.04,.05) if htd_loaded_goal_hold else (.03,.04))
            or pickup_stop_phase!=106. or grasp_force_target!=6. or grasp_height_adjustment!=-.04 or live_acceleration!=3.):
        raise ValueError('Heading cart requires reviewed relaxed walking/clearance/hybrid-lowering profile')
    transfer_active=walk_to_rolltainer or inplace_transfer
    if grasp_force_target is not None and not relaxed_grasp:raise ValueError('Explicit force target requires relaxed grasp profile')
    if torso_relative_lowering and not transfer_active:raise ValueError('Torso-relative lowering requires walking transfer')
    if lowering_acceleration_limit is not None and (not math.isfinite(lowering_acceleration_limit) or not 0<lowering_acceleration_limit<=3. or not world_position_lowering or not relaxed_grasp):
        raise ValueError('Lowering acceleration override requires relaxed world-position lowering and a bound within (0,3] rad/s²')
    if world_position_lowering and not torso_relative_lowering:raise ValueError('World-position lowering requires torso-relative orientation mode')
    if not math.isfinite(live_torso_filter_seconds) or live_torso_filter_seconds<0:raise ValueError('Invalid torso filter time constant')
    if simulation_rate_limit is not None and (not math.isfinite(simulation_rate_limit) or not 0<simulation_rate_limit<=1):raise ValueError('Invalid simulation wall-rate limit')
    if not math.isfinite(simulation_rate_start_phase) or simulation_rate_start_phase<0 or (simulation_rate_start_phase and simulation_rate_limit is None):
        raise ValueError('Pacing start phase requires a rate cap and finite nonnegative phase')
    if narrow_destination_support and not walk_to_rolltainer:raise ValueError('Narrow destination requires rolltainer walking layout')
    if palm_torsion_m is not None and (not math.isfinite(palm_torsion_m) or not 0<palm_torsion_m<=.05):
        raise ValueError('Palm torsional friction must be finite and within (0,0.05] m')
    if force_grasp and live_model is None:raise ValueError('Force adaptation requires live HL-IK')
    if not math.isfinite(payload_mass) or payload_mass<=0:raise ValueError('Positive finite payload mass required')
    if relaxed_grasp and not force_grasp:raise ValueError('Relaxed grasp requires live force adaptation')
    requirements=relaxed_requirements(payload_mass,mu,3. if grasp_force_target is None else grasp_force_target) if relaxed_grasp else None
    if synchronized_grasp and not force_grasp:raise ValueError('Synchronized grasp requires force adaptation')
    if box_orientation_feedback and not force_grasp:raise ValueError('Payload orientation feedback requires force adaptation')
    if walk_to_rolltainer and not relaxed_grasp:raise ValueError('Walking prototype requires explicit relaxed grasp profile')
    if not math.isfinite(carry_clearance_lift)or not 0<=carry_clearance_lift<=.06 or (carry_clearance_lift and not transfer_active):
        raise ValueError('Walking clearance lift must be between0 and60mm')
    if not math.isfinite(pickup_stop_phase) or pickup_stop_phase<=0:raise ValueError('Invalid pickup handoff phase')
    if not math.isfinite(grasp_height_adjustment) or not -.06<=grasp_height_adjustment<=.04:raise ValueError('Invalid grasp height adjustment')
    if grasp_height_adjustment and live_model is None:raise ValueError('Grasp seating adjustment requires live HL-IK')
    if refine_foot_friction and pair_friction_time is None:raise ValueError('Foot refinement requires pair friction time')
    if hand_friction_time is not None and pair_friction_time is None:raise ValueError('Hand friction refinement requires pair friction time')
    env=os.environ.copy()
    env['IGRIS_HAND_PYTHON']=str(ROOT/'third_party/curobo/.venv/bin/python')
    env['PYTHONPATH']=os.pathsep.join(map(str,[ROOT,ROOT/'igris/igris_curobo/src',ROOT/'igris-core/igris_core',ROOT/'igris/real_client']))
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[name]='1'
    py=str(ROOT/'env_isaaclab/bin/python')
    gains=ROOT/'igris/sim_local/vr_box_eval/results/20260914_synthetic_motion100_1kg/replay_gains.toml'
    preparation=[py,str(HERE/'box_replay.py'),'prepare','--out',str(out),
                            '--compliant-hands','--motion',str(motion),'--destination',*map(str,destination),
                            '--source-base-clearance',str(source_base_clearance),
                            '--payload-mass',str(payload_mass),
                            '--source-base-normal',str(source_base_normal)]+(['--narrow-source-support'] if narrow_source_support else [])
    if walk_to_rolltainer:
        preparation=[py,str(HERE/'rolltainer_layout.py'),'--out',str(out),'--plan',str(motion.parent),'--payload-mass',str(payload_mass),'--mu',str(mu)]
        if narrow_destination_support:preparation+=['--narrow-destination-support']
        if heading_cart_proposal is not None:
            preparation=[py,str(HERE/'heading_cart_layout.py'),'--out',str(out),'--plan',str(motion.parent),
                         '--proposal',str(Path(heading_cart_proposal).resolve()),'--payload-mass',str(payload_mass)]
    if inplace_transfer:
        preparation=[py,str(HERE/'inplace_layout.py'),'--out',str(out),'--plan',str(motion.parent)]
        if inplace_proposal is not None:preparation+=['--proposal',str(Path(inplace_proposal).resolve())]
    prepared=subprocess.run(preparation,env=env,capture_output=True,text=True)
    if not out.exists():raise RuntimeError(prepared.stderr)
    (out/'prepare.log').write_text(prepared.stdout+prepared.stderr)
    prepared.check_returncode()
    mass_audit=validate_prepared_payload(out/'run',payload_mass)
    (out/'prepared_payload_mass.json').write_text(json.dumps(mass_audit,indent=2)+'\n')
    import hashlib
    source_snapshot=out/'source_snapshot';source_snapshot.mkdir()
    source_hashes={}
    for name in ('transport_trial.py','run.py','live_ik.py','walking_transfer.py','segment_delivery.py',
                 'htd_loaded_reference.py',
                 'grasp_feedback.py','box_orientation_feedback.py','reference_spline.py',
                 'room_setup.cpp','observer.cpp','contact_wrench.h','support_force_groups.h','box_replay.py','transport_metrics.py',
                 'vendor_policy_clock.cpp','vendor_policy_clock.py','simulation_pacer.cpp','rolltainer_layout.py',
                 'contact_solver.py','audit_material.cpp','contact_material.py','control_gc.py','lowering_targets.py','inplace_transfer.py','inplace_layout.py','inplace_reference.py','placement_support.py','heading_route.py'):
        source=HERE/name;data=source.read_bytes();(source_snapshot/name).write_bytes(data)
        source_hashes[str(source)]=hashlib.sha256(data).hexdigest()
    if heading_cart_proposal is not None:
        source=HERE/'heading_cart_layout.py';data=source.read_bytes();(source_snapshot/source.name).write_bytes(data)
        source_hashes[str(source)]=hashlib.sha256(data).hexdigest()
    source=ROOT/'igris/hl_ik/pipeline.py';data=source.read_bytes()
    (source_snapshot/'hl_ik_pipeline.py').write_bytes(data)
    source_hashes[str(source)]=hashlib.sha256(data).hexdigest()
    if pair_padding is not None:
        for name in ('geometry.py','retarget_quality.py','solver.py','common.py','native_contacts.py','native_contacts.cpp'):
            source=ROOT/'igris/hl_ik'/name;data=source.read_bytes()
            (source_snapshot/('hl_ik_'+name)).write_bytes(data)
            source_hashes[str(source)]=hashlib.sha256(data).hexdigest()
    if htd_loaded_zero:
        for name in ('htd_loaded_setup.py','htd_loaded.py','htd_policy.py','htd_serial_pd.py','htd_probe_config.py',
                     'htd_loaded_runtime.py','htd_candidate_package.py','dip_damping.py','dip_damping.cpp',
                     'htd_backlash_limits.py','htd_backlash_limits.cpp','htd_output_coordinates.py','htd_output_coordinates.cpp'):
            source=HERE/name;data=source.read_bytes();(source_snapshot/name).write_bytes(data)
            source_hashes[str(source)]=hashlib.sha256(data).hexdigest()
    if htd_command_observer:
        for name in ('htd_command_observer.py','htd_command_observer.cpp','htd_command_scope.py'):
            source=HERE/name;data=source.read_bytes();(source_snapshot/name).write_bytes(data)
            source_hashes[str(source)]=hashlib.sha256(data).hexdigest()
    if htd_wrist_mode_entry:
        source=HERE/'htd_wrist_entry.py';data=source.read_bytes();(source_snapshot/source.name).write_bytes(data)
        source_hashes[str(source)]=hashlib.sha256(data).hexdigest()
    if htd_loaded_goal_hold:
        for source in (HERE/'htd_goal_hold.py', ROOT/'igris/igris-c-wbc-base/src/wbc_base/world_xy.py'):
            data=source.read_bytes();(source_snapshot/source.name).write_bytes(data)
            source_hashes[str(source)]=hashlib.sha256(data).hexdigest()
    if htd_loaded_transport:
        for name in ('htd_navigation.py','htd_navigation_feedback.py','htd_reviewed_approach.py',
                     'htd_transport_reference.py','htd_transport_phase.py'):
            source=HERE/name;data=source.read_bytes();(source_snapshot/name).write_bytes(data)
            source_hashes[str(source)]=hashlib.sha256(data).hexdigest()
    if htd_contact_grace:
        source=HERE/'htd_grasp_retention.py';data=source.read_bytes();(source_snapshot/source.name).write_bytes(data)
        source_hashes[str(source)]=hashlib.sha256(data).hexdigest()
    (source_snapshot/'manifest.json').write_text(json.dumps(source_hashes,indent=2)+'\n')
    trial=out/'run'
    if native_policy_clock:
        manifest=json.loads((trial/'manifest.json').read_text())
        manifest['vendor_policy_native_clock']=True
        (trial/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if relaxed_grasp:
        manifest=json.loads((trial/'manifest.json').read_text())
        manifest.update(evaluation_profile='relaxed_transport',relaxed_rotation_limit_deg=15.,
                        grasp_height_adjustment_m=grasp_height_adjustment,
                        grasp_requirements=requirements,
                        authorized_relaxation='2026-09-20 user requested higher friction/lower mass and contact-based progression; original-condition hardening deferred')
        (trial/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if walk_to_rolltainer:
        manifest=json.loads((trial/'manifest.json').read_text())
        proposal=json.loads((out/'goal_proposal.json').read_text())
        manifest['walking_transfer']=dict(pickup_stop_phase_s=pickup_stop_phase,target_payload_xyz=proposal['payload_center_world_m'],
            clearance_lift_m=carry_clearance_lift,clearance_minimum_rise_m=clearance_minimum_rise,heading_preserving_route=heading_cart_proposal is not None,
            xy_first_lowering=xy_first_lowering,arrival_settle_seconds=arrival_settle_seconds,arrival_brake_radius_m=arrival_brake_radius,
            torso_relative_lowering=torso_relative_lowering,world_position_lowering=world_position_lowering,lowering_acceleration_limit=lowering_acceleration_limit,lowering_duration_s=16. if xy_first_lowering else 12. if torso_relative_lowering else 8.,
            target_torso_heading_rad=proposal['robot_goal_yaw_world_rad'],body_controller='vendor WALK_LOW',
            arm_controller='FiSTA HL-IK; measured torso-relative carry, '+('world-position/torso-orientation lowering/open' if world_position_lowering else 'torso-relative lowering/open' if torso_relative_lowering else 'world-fixed lowering'),
            goal_root_rule='computed from measured payload offset after optional clearance lift and fixed torso heading; waist retained from pickup')
        (trial/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if inplace_transfer:
        manifest=json.loads((trial/'manifest.json').read_text())
        manifest['inplace_transfer'].update(pickup_stop_phase_s=pickup_stop_phase,
            lowering_acceleration_limit=lowering_acceleration_limit,runtime_integration_pending=False,
            caller_reference_schema='explicit_inplace_waist/v1',arm_controller='FiSTA HL-IK',
            body_controller='vendor balance with zero WalkTwist')
        (trial/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if simulation_rate_limit is not None:
        manifest=json.loads((trial/'manifest.json').read_text())
        manifest['simulation_rate_limit']=simulation_rate_limit
        manifest['simulation_rate_start_phase_s']=simulation_rate_start_phase
        (trial/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report={'hardware_executed':False,'motion':str(motion),'start_wall':time.time(),'steps':[],
            'payload_mass_kg':payload_mass,'mu':mu,'evaluation_profile':'relaxed_transport' if relaxed_grasp else 'original_grasp_requirements'}
    def call(name,args):
        print(name,flush=True)
        with (out/(name+'.log')).open('w') as log:
            result=subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT)
        report['steps'].append({'name':name,'exit_code':result.returncode})
        (out/'trial_status.json').write_text(json.dumps(report,indent=2)+'\n')
        result.check_returncode()
    worker=None;worker_log=None
    try:
        if inplace_transfer:
            audit=json.loads((trial/'material_audit.json').read_text())
            manifest=json.loads((trial/'manifest.json').read_text())
            validate_prepared_inplace_contacts(manifest,audit,hashlib.sha256((trial/'scene.xml').read_bytes()).hexdigest(),
                mu,pair_friction_time,refine_foot_friction,hand_friction_time,palm_torsion_m)
            report['steps'].append(dict(name='material_and_pair_solver_prepared_by_inplace_layout',exit_code=0))
        elif walk_to_rolltainer:
            import hashlib
            audit=json.loads((trial/'material_audit.json').read_text())
            if audit['mu']!=mu or audit['scene_sha256']!=hashlib.sha256((trial/'scene.xml').read_bytes()).hexdigest():
                raise ValueError('Prepared rolltainer material audit differs from requested scene')
            report['steps'].append(dict(name='material_prepared_by_rolltainer_layout',exit_code=0))
        else:call('material',[py,str(HERE/'contact_material.py'),'--out',str(trial),'--mu',str(mu)])
        if palm_torsion_m is not None and not inplace_transfer:
            call('palm_torsion',[py,str(HERE/'contact_solver.py'),'--out',str(trial),'--palm-torsion-m',str(palm_torsion_m)])
        if contact_noslip_iterations:
            call('contact_solver',[py,str(HERE/'contact_solver.py'),'--out',str(trial),'--iterations',str(contact_noslip_iterations)])
        if pair_friction_time is not None and not inplace_transfer:
            call('pair_friction_solver',[py,str(HERE/'contact_solver.py'),'--out',str(trial),'--pair-friction-time',str(pair_friction_time)]+(['--include-feet'] if refine_foot_friction else [])+(['--hand-friction-time',str(hand_friction_time)] if hand_friction_time is not None else []))
        if contact_impratio is not None:
            call('friction_impedance',[py,str(HERE/'contact_solver.py'),'--out',str(trial),'--impratio',str(contact_impratio)])
        inc=ROOT/'igris/ws/install/igris_c_mujoco/include'
        lib=ROOT/'igris/ws/install/igris_c_mujoco/lib/igris_c_mujoco'
        call('compile_material',['g++','-O2','-std=c++17',f'-I{inc}',str(HERE/'audit_material.cpp'),
             f'-L{lib}',f'-Wl,-rpath,{lib}','-l:libmujoco.so.3.2.7','-o',str(trial/'audit_material')])
        audit_extra=([str(pair_friction_time)] if pair_friction_time is not None else [])
        if refine_foot_friction or hand_friction_time is not None:audit_extra+=['with-feet' if refine_foot_friction else 'without-feet']
        if hand_friction_time is not None:audit_extra+=[str(hand_friction_time)]
        if transfer_active and pair_friction_time is not None:
            manifest=json.loads((trial/'manifest.json').read_text())
            audit_extra=[str(pair_friction_time),'with-feet' if refine_foot_friction else 'without-feet',
                         str(hand_friction_time if hand_friction_time is not None else pair_friction_time),
                         str(manifest['pair_friction_solver']['support_pair_count'])]
        if palm_torsion_m is not None:
            manifest=json.loads((trial/'manifest.json').read_text())
            audit_extra=[str(pair_friction_time or 0.),'with-feet' if refine_foot_friction else 'without-feet',
                         str(hand_friction_time if hand_friction_time is not None else pair_friction_time or 0.),
                         str(manifest.get('pair_friction_solver',{}).get('support_pair_count',91)),str(palm_torsion_m)]
        call('native_material',[str(trial/'audit_material'),str(trial/'scene.xml'),str(trial/'native_material_audit.json'),str(mu)]+audit_extra)
        if dip_damping is not None:
            from dip_damping import configure
            manifest=json.loads((trial/'manifest.json').read_text())
            manifest.update(dip_damping_requested=dip_damping,dip_damping_scope='loaded_transport')
            (trial/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            report['dip_damping_variant']=configure(trial,dip_damping)
            report['steps'].append(dict(name='native_dip_damping_variant',exit_code=0))
            (out/'trial_status.json').write_text(json.dumps(report,indent=2)+'\n')
        if htd_loaded_zero:
            from htd_loaded_setup import prepare,require_runtime_hooks
            evidence=prepare(trial,enabled=True,environment=env,current_grasp_reference=htd_current_grasp_reference,
                             command_observer=htd_command_observer,wrist_mode_entry=htd_wrist_mode_entry,goal_hold=htd_loaded_goal_hold,contact_grace=htd_contact_grace,policy_package=htd_policy_package,transport=htd_loaded_transport,posture_domain_diagnostic=htd_posture_domain_diagnostic,loaded_rate=htd_loaded_rate,upper_backlash_timeconst=htd_upper_backlash_timeconst,prealignment_coast_seconds=htd_prealignment_coast_seconds)
            env.update(evidence['environment'])
            report['htd_loaded_zero']=evidence
            if htd_prealignment_coast_seconds is not None:
                report['htd_prealignment_coast']=evidence['manifest_scope']['prealignment_coast']
            if htd_upper_backlash_timeconst is not None:
                report['htd_upper_backlash_limit']=evidence['manifest_scope']['upper_backlash_limit']
            report['steps'].append(dict(name='htd_loaded_zero_host_setup',exit_code=0))
            (out/'trial_status.json').write_text(json.dumps(report,indent=2)+'\n')
            require_runtime_hooks()
        if pair_padding is not None:
            report['hl_ik_pair_padding']=configure_htd_pair_padding(trial,pair_padding)
            (out/'trial_status.json').write_text(json.dumps(report,indent=2)+'\n')
        for step in ('start','bringup'):
            args=[py,str(HERE/'run.py'),step,'--out',str(trial)]
            if step=='bringup' and physics_cpu is not None:args+=['--physics-cpu',str(physics_cpu)]
            call(step,args)
        if live_model is not None:
            worker_log=(out/'live_ik.log').open('w')
            live_options=[]
            if live_limit_cost is not None:live_options+=['--joint-limit-cost',str(live_limit_cost)]
            if live_soft_margin is not None:live_options+=['--soft-joint-margin',str(live_soft_margin)]
            if collision_braking:live_options+=['--collision-braking']
            if native_collision_contacts:live_options+=['--native-collision-contacts']
            if track_source:live_options+=['--track-source']
            if orientation_feedback:live_options+=['--orientation-feedback']
            if synchronized_grasp:live_options+=['--synchronized-grasp']
            if box_orientation_feedback:live_options+=['--box-orientation-feedback']
            if track_source_rotation:live_options+=['--track-source-rotation']
            live_options+=['--grasp-travel',str(grasp_travel)]
            live_options+=['--grasp-height-adjustment',str(grasp_height_adjustment)]
            live_options+=['--lead-steps',str(live_lead_steps)]
            live_options+=['--torso-filter-seconds',str(live_torso_filter_seconds)]
            worker=subprocess.Popen([str(ROOT/'.local/envs/hl-ik/bin/python'),str(HERE/'live_ik.py'),
                '--out',str(trial),'--plan',str(motion.parent),'--model',str(live_model),
                '--acceleration-limit',str(live_acceleration)]+(['--force-grasp'] if force_grasp else [])+live_options,
                env=env,stdout=worker_log,stderr=subprocess.STDOUT)
            # Keep the single-threaded solver on one core, separate from native
            # physics and the MCP process. This changes scheduling, not physics.
            available=os.sched_getaffinity(0)
            if 8 in available and physics_cpu not in (8,52):
                os.sched_setaffinity(worker.pid,{8});report['live_ik_cpu']=8
            deadline=time.monotonic()+30
            while not (trial/'live_ik_ready.json').exists():
                if worker.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Live HL-IK worker did not initialize')
                time.sleep(.05)
        args=[py,str(HERE/'run.py'),'command','--out',str(trial),
              '--compensate-backlash','--motion',str(motion),'--gains',str(gains)]
        if worker is not None:args+=['--live-ik']
        if walk_to_rolltainer:args+=['--walk-to-rolltainer']
        if inplace_transfer:args+=['--inplace-transfer']
        call('command',args)
    except Exception as exc:
        report['error']=str(exc)
    finally:
        if worker is not None and worker.poll() is None:
            worker.send_signal(signal.SIGTERM)
            try:worker.wait(timeout=10)
            except subprocess.TimeoutExpired:worker.kill();worker.wait()
        if worker_log is not None:worker_log.close()
        if (trial/'runtime.json').exists():
            try:call('stop',[py,str(HERE/'run.py'),'stop','--out',str(trial)])
            except Exception as exc:report['stop_error']=str(exc)
        if list(trial.glob('command_*.json')):
            try:
                call('analyze',[py,str(HERE/'box_replay.py'),'analyze','--out',str(trial)])
                call('rotation',[py,str(HERE/'transport_metrics.py'),'--out',str(trial)])
            except Exception as exc:report['analysis_error']=str(exc)
        report['end_wall']=time.time()
        metrics=json.loads((trial/'transport_result.json').read_text()) if (trial/'transport_result.json').exists() else {}
        report['strict_criteria_success']=metrics.get('pass',False)
        report['task_success']=metrics.get('relaxed_pass' if relaxed_grasp else 'pass',False)
        (out/'trial_status.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--mu',type=float,default=.6)
    p.add_argument('--payload-mass',type=float,default=1.)
    p.add_argument('--relaxed-grasp',action='store_true',help='Explicit easier-load profile: 3 N palms, optional tips, 15 degree rotation bound; strict metrics still saved')
    p.add_argument('--grasp-force-target',type=float,help='Explicit relaxed-profile palm normal-force target up to14 N; default3 N. Existing EE speed/travel and motor limits retained')
    p.add_argument('--walk-to-rolltainer',action='store_true')
    p.add_argument('--htd-loaded-transport',action='store_true',help='Explicit180s guarded HTD route, lower and supported release through the sole publisher')
    p.add_argument('--htd-prealignment-coast-seconds',type=float,help='Explicit SIM reviewed prealignment one-shot3.0 native-second XYZ-zero diagnostic; default off')
    p.add_argument('--htd-upper-backlash-timeconst',type=float,help='Explicit relaxed loaded SIM startup upper14 limit time constant0.005; default upper profile unchanged')
    p.add_argument('--htd-loaded-rate',type=float,default=.1,help='Loaded wall-rate cap0.1 (default) or explicit relaxed SIM transport0.05; acquisition0.5 and native actor/PD clocks unchanged')
    p.add_argument('--htd-posture-domain-diagnostic',action='store_true',help='Explicit SIM V2 transport diagnostic: record acquired posture outside the evaluated profile without relaxing physical/grasp guards')
    p.add_argument('--htd-pelvis-right-thigh-unbuffered',action='store_true',help='Explicit SIM loaded transport: retain raw pelvis/right-thigh spheres and all other buffers; native physics unchanged')
    p.add_argument('--htd-pelvis-thighs-unbuffered',action='store_true',help='Explicit SIM loaded transport: retain raw pelvis/left-thigh and pelvis/right-thigh spheres; all other buffers and native physics unchanged; exclusive with right-only option')
    p.add_argument('--htd-policy-package',type=Path,help='Explicit evaluated continuation package with pinned loaded_candidate.json; requires loaded current-grasp goal hold')
    p.add_argument('--htd-contact-grace',action='store_true',help='Explicit goal hold100ms contact grace with total-friction and actual box/EE relative pose guards')
    p.add_argument('--htd-loaded-goal-hold',action='store_true',help='Explicit SIM10s XY/heading hold with per-hand palm+tip qualification; requires observed wrist entry')
    p.add_argument('--htd-loaded-zero',action='store_true',help='SIM-only frozen1750/profile76 loaded zero10s after qualified clearance; requires integrated runtime hooks')
    p.add_argument('--htd-current-grasp-reference',action='store_true',help='Explicit SIM out-of-domain1750 evaluation: capture actual accepted held8 at handoff; keep bounded subsequent drift and physical guards')
    p.add_argument('--htd-command-observer',action='store_true',help='Logging-only effective command boundary trace for isolated loaded HTD experiments')
    p.add_argument('--htd-wrist-mode-entry',action='store_true',help='SIM observed loaded entry: hold actual wrist motors through SlowStart before resuming serial HL targets')
    p.add_argument('--arrival-brake-radius',type=float,help='Explicit final heading-route brake radius15..25mm; arrival/support gates unchanged')
    p.add_argument('--inplace-transfer',action='store_true',help='Reviewed near original-box layout, zero walking command, explicit waist arc')
    p.add_argument('--heading-cart-proposal',type=Path,help='Explicit reviewed heading-preserving cart layout and measured source departure')
    p.add_argument('--inplace-proposal',type=Path,help='Explicit reviewed near-layout proposal; default original near geometry')
    p.add_argument('--pickup-stop-phase',type=float,default=110.,help='Archive phase for loaded walking handoff; measured grasp and lift gates remain required')
    p.add_argument('--minimum-clearance-rise',type=float,default=None,help='Explicit relaxed20mm actual rise with reviewed40mm EE target; retain speed, contact and dwell checks')
    p.add_argument('--carry-clearance-lift',type=float,default=0.,help='Extra EE lift before velocity commands; measured rise, low speed, palm reserve and absence of support required')
    p.add_argument('--torso-relative-lowering',action='store_true',help='Opt-in12s lowering/opening in captured torso frame; retains measured support/position/speed release gates')
    p.add_argument('--world-position-lowering',action='store_true',help='Opt-in world-fixed EE position path with torso-following orientation; requires --torso-relative-lowering')
    p.add_argument('--xy-first-lowering',action='store_true',help='Reviewed heading route:4s horizontal world alignment, then12s vertical descent')
    p.add_argument('--arrival-settle-seconds',type=float,default=1.5,help='Reviewed heading-route experiment:0.3..1.5s measured arrival dwell; speed/yaw/release gates unchanged')
    p.add_argument('--lowering-acceleration-limit',type=float,help='Explicit relaxed profile: arm acceleration bound only during lowering/opening, at most3 rad/s²; carry unchanged')
    p.add_argument('--destination',type=float,nargs=2,default=(.33,.20))
    p.add_argument('--physics-cpu',type=int)
    p.add_argument('--live-model',type=Path)
    p.add_argument('--live-acceleration',type=float,default=3.)
    p.add_argument('--force-grasp',action='store_true')
    p.add_argument('--live-limit-cost',type=float)
    p.add_argument('--live-soft-margin',type=float)
    p.add_argument('--collision-braking',action='store_true')
    p.add_argument('--source-base-clearance',type=float,default=0.)
    p.add_argument('--source-base-normal',type=float,default=0.)
    p.add_argument('--track-source',action='store_true')
    p.add_argument('--live-lead-steps',type=int,default=3)
    p.add_argument('--contact-noslip-iterations',type=int,default=0)
    p.add_argument('--pair-friction-time',type=float)
    p.add_argument('--refine-foot-friction',action='store_true')
    p.add_argument('--orientation-feedback',action='store_true')
    p.add_argument('--grasp-travel',type=float,default=.015)
    p.add_argument('--grasp-height-adjustment',type=float,default=0.,help='Additional world-Z EE seating offset, smoothly applied before contact; object pose unchanged')
    p.add_argument('--hand-friction-time',type=float)
    p.add_argument('--track-source-rotation',action='store_true')
    p.add_argument('--narrow-source-support',action='store_true')
    p.add_argument('--contact-impratio',type=float)
    p.add_argument('--synchronized-grasp',action='store_true')
    p.add_argument('--box-orientation-feedback',action='store_true')
    p.add_argument('--live-torso-filter-seconds',type=float,default=.4)
    p.add_argument('--simulation-rate-limit',type=float,help='Maximum simulation seconds per wall second; unchanged native timestep')
    p.add_argument('--simulation-rate-start-phase',type=float,default=0.,help='Activate an explicit rate cap at this motion phase; default0 preserves replay-start pacing')
    p.add_argument('--narrow-destination-support',action='store_true',help='Use original boxes to build the verified narrow rolltainer support at the same height')
    p.add_argument('--palm-torsion-m',type=float,help='Explicit experimental torsional friction length for only two palm/payload pairs; other contacts retained')
    p.add_argument('--native-collision-contacts',action='store_true')
    p.add_argument('--native-policy-clock',action='store_true',help='Opt-in native-time vendor policy scheduler; controller/watchdog clocks remain wall time')
    p.add_argument('--dip-damping',type=float,choices=[.002],help='Explicit SIM loaded-transport DIP-only viscous damping variant; default unchanged')
    a=p.parse_args();run(a.out.resolve(),a.motion.resolve(),a.mu,a.destination,a.physics_cpu,
                        a.live_model.resolve() if a.live_model else None,a.live_acceleration,a.force_grasp,
                        a.live_limit_cost,a.live_soft_margin,a.collision_braking,a.source_base_clearance,a.source_base_normal,a.track_source,a.live_lead_steps,a.contact_noslip_iterations,a.pair_friction_time,a.refine_foot_friction,a.orientation_feedback,a.grasp_travel,a.hand_friction_time,a.track_source_rotation,a.narrow_source_support,a.contact_impratio,a.synchronized_grasp,a.box_orientation_feedback,a.live_torso_filter_seconds,a.simulation_rate_limit,a.native_collision_contacts,a.payload_mass,a.relaxed_grasp,a.walk_to_rolltainer,a.pickup_stop_phase,a.grasp_height_adjustment,a.native_policy_clock,a.carry_clearance_lift,a.simulation_rate_start_phase,a.narrow_destination_support,a.palm_torsion_m,a.grasp_force_target,a.torso_relative_lowering,a.world_position_lowering,a.lowering_acceleration_limit,a.inplace_transfer,a.inplace_proposal,a.heading_cart_proposal,a.xy_first_lowering,a.arrival_settle_seconds,a.htd_loaded_zero,a.minimum_clearance_rise,a.arrival_brake_radius,a.htd_current_grasp_reference,a.htd_command_observer,a.htd_wrist_mode_entry,a.htd_loaded_goal_hold,a.htd_contact_grace,a.htd_policy_package,a.htd_loaded_transport,a.dip_damping,a.htd_posture_domain_diagnostic,a.htd_pelvis_right_thigh_unbuffered,a.htd_pelvis_thighs_unbuffered,a.htd_loaded_rate,a.htd_upper_backlash_timeconst,a.htd_prealignment_coast_seconds)
