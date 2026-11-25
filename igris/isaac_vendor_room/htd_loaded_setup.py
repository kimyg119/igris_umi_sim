"""Opt-in fresh-trial loaded-zero configuration. Never starts physics or ROS.

Default paths are the frozen1750/profile76 package; evaluated continuations
require an explicit validated descriptor and current-grasp goal-hold scope. Model checks load
MuJoCo for static audits only. Runtime integration remains fail-closed until
run.py declares the reviewed command/HL/start contract implemented.
"""
import json
from pathlib import Path

import yaml

from htd_candidate_package import (PACKAGE, PROFILE, EXPECTED, sha,
                                   require_candidate_scope, resolve_package)


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
P = ROOT/'outputs/hlik_transport_20260918_2005'
PARAMS = ROOT/'igris/configs/controller/params.yaml'
CONTROLLER = ROOT/'igris/ws/install/igris_c_controller/lib/igris_c_controller/sim'
CONTRACT = 'htd_loaded_zero/v1'


def validate_options(*, live_model, walk_to_rolltainer, inplace_transfer, force_grasp,
                     relaxed_grasp, payload_mass, mu, carry_clearance_lift,
                     simulation_rate_limit, simulation_rate_start_phase, goal_hold=False):
    if (live_model is None or not walk_to_rolltainer or inplace_transfer or not force_grasp
            or not relaxed_grasp or payload_mass not in ((.01,.1) if goal_hold is True else (.1,)) or mu != 4.
            or carry_clearance_lift not in ((.04,.05) if goal_hold is True else (.04,))
            or simulation_rate_limit != .5 or simulation_rate_start_phase != 90.):
        raise ValueError('HTD loaded zero requires reviewed live force-grasp WALK_LOW acquisition, '
                         '.1kg/mu4/40mm clearance (10g/50mm explicit goal hold) and0.5 pacing from phase90')


def require_runtime_hooks(runtime_module=None):
    if runtime_module is None:
        import run as runtime_module
    if getattr(runtime_module, 'HTD_LOADED_ZERO_RUNTIME_CONTRACT', None) != CONTRACT:
        raise RuntimeError('HTD loaded-zero runtime integration is incomplete; refusing vendor-walking fallback')


def prepare(trial, *, enabled=False, current_grasp_reference=False, command_observer=False,
            wrist_mode_entry=False, goal_hold=False, contact_grace=False, environment=None,
            policy_package=None, transport=False, posture_domain_diagnostic=False, loaded_rate=.1, upper_backlash_timeconst=None, prealignment_coast_seconds=None):
    """Prepare a fresh directory; caller separately verifies private runtime.

    Returns environment additions without mutating the supplied environment.
    Absence of the opt-in flag is a complete no-op, including file reads.
    """
    if prealignment_coast_seconds is not None:
        from htd_loaded import validate_prealignment_coast_option
        prealignment_coast_seconds=validate_prealignment_coast_option(prealignment_coast_seconds,
            enabled=enabled,transport=transport,current_grasp_reference=current_grasp_reference,
            command_observer=command_observer,wrist_mode_entry=wrist_mode_entry,goal_hold=goal_hold,
            contact_grace=contact_grace)
    if upper_backlash_timeconst is not None:
        from htd_loaded import validate_upper_backlash_option
        upper_backlash_timeconst=validate_upper_backlash_option(upper_backlash_timeconst,
            enabled=enabled,transport=transport,current_grasp_reference=current_grasp_reference,
            command_observer=command_observer,wrist_mode_entry=wrist_mode_entry,goal_hold=goal_hold,
            contact_grace=contact_grace)
    if type(posture_domain_diagnostic) is not bool:
        raise ValueError('Posture domain diagnostic must be bool')
    if posture_domain_diagnostic and not (transport is True and current_grasp_reference is True and policy_package is not None):
        raise ValueError('Posture domain diagnostic requires explicit V2 transport/current-grasp scope')
    if type(transport) is not bool or (transport and not (enabled and current_grasp_reference
            and command_observer and wrist_mode_entry and goal_hold and contact_grace)):
        raise ValueError('Loaded transport requires all observed current-grasp goal/retention options')
    require_candidate_scope(policy_package, enabled=enabled, current_grasp_reference=current_grasp_reference,
        command_observer=command_observer, wrist_mode_entry=wrist_mode_entry, goal_hold=goal_hold)
    if type(contact_grace) is not bool or (contact_grace and not goal_hold):
        raise ValueError('Contact grace requires explicit goal hold scope')
    if type(goal_hold) is not bool or (goal_hold and not (enabled and current_grasp_reference and command_observer and wrist_mode_entry)):
        raise ValueError('Goal hold requires observed current-grasp wrist mode entry scope')
    if type(wrist_mode_entry) is not bool or (wrist_mode_entry and not (enabled and current_grasp_reference and command_observer)):
        raise ValueError('Observed current-grasp loaded scope required for wrist mode entry')
    if type(command_observer) is not bool or (command_observer and not enabled):
        raise ValueError('Command observer requires explicit loaded-zero scope')
    if type(current_grasp_reference) is not bool or (current_grasp_reference and not enabled):
        raise ValueError('Current-grasp reference requires explicit loaded-zero scope')
    if not enabled and type(loaded_rate) is float and loaded_rate == .1:
        return None
    from htd_loaded import validate_loaded_rate
    loaded_rate = validate_loaded_rate(loaded_rate, transport=transport,
        current_grasp_reference=current_grasp_reference)
    if not enabled:
        raise ValueError('Loaded pacing requires explicit enabled transport scope')
    trial = Path(trial).resolve()
    if not trial.is_relative_to(ROOT):
        raise ValueError('HTD loaded setup must remain inside this project')
    if (environment or {}).get('IGRIS_ROOM_PROFILE_LIBRARY'):
        raise ValueError('Refusing to replace an existing profile interposer')
    forbidden = ('runtime.json', 'physics_stats.json', 'pacing.request', 'htd_pacing.request',
                 'htd_pacing.request.ack', 'htd_initial_state.request', 'htd_initial_state_applied.json',
                 'htd_loaded_preflight.json', 'htd_gc_params.yaml', 'htd_backlash_original_scene.xml',
                 'htd_command_observer.arm', 'htd_command_observer.bin', 'htd_command_observer.json')
    if any((trial/name).exists() for name in forbidden):
        raise FileExistsError('HTD loaded setup requires a fresh trial without startup/activation/initializer artifacts')
    manifest = json.loads((trial/'manifest.json').read_text())
    if 'htd_prealignment_coast' in manifest:
        raise ValueError('Coast setup requires a fresh undeclared manifest')
    if prealignment_coast_seconds is not None:
        if (manifest.get('hardware_executed') is not False
                or manifest.get('physics_backend')!='vendor MuJoCo 3.2.7'
                or manifest.get('evaluation_profile')!='relaxed_transport'):
            raise ValueError('Coast requires fresh relaxed native SIM manifest')
    if 'htd_upper_backlash_limit' in manifest:
        raise FileExistsError('Upper backlash setup requires a fresh undeclared manifest')
    if upper_backlash_timeconst is not None and (manifest.get('evaluation_profile')!='relaxed_transport'
            or manifest.get('hardware_executed') is not False
            or manifest.get('physics_backend')!='vendor MuJoCo 3.2.7'):
        raise ValueError('Upper backlash requires relaxed native SIM manifest')
    if loaded_rate == .05 and manifest.get('evaluation_profile') != 'relaxed_transport':
        raise ValueError('Loaded pacing .05 requires explicit relaxed_transport SIM profile')
    from dip_damping import validate as validate_dip_damping
    dip_variant=validate_dip_damping(trial)
    if dip_variant is not None and not transport:
        raise ValueError('DIP damping variant requires explicit loaded transport setup')
    if 'htd_initial_state' in manifest:
        raise ValueError('Loaded setup forbids the empty-hand initializer')
    if 'htd_loaded_zero' in manifest:
        raise FileExistsError('HTD loaded setup requires a fresh unconfigured manifest')
    walking=manifest.get('walking_transfer',{})
    if transport and not all(walking.get(k) is True for k in ('heading_preserving_route',
            'xy_first_lowering','world_position_lowering','torso_relative_lowering')):
        raise ValueError('Loaded transport requires the reviewed heading route and XY-first world lowering')
    if (manifest.get('payload', {}).get('mass_kg') not in ((.01,.1) if goal_hold else (.1,))
            or manifest.get('payload', {}).get('friction') != 4.
            or manifest.get('walking_transfer', {}).get('clearance_lift_m') not in ((.04,.05) if goal_hold else (.04,))
            or manifest.get('simulation_rate_limit') != .5
            or manifest.get('simulation_rate_start_phase_s') != 90.):
        raise ValueError('Prepared acquisition manifest differs from the reviewed HTD loaded profile')
    robot = yaml.safe_load((trial/'robot.yaml').read_text())
    reference = yaml.safe_load(PARAMS.read_text())
    from htd_loaded import LoadedZero, scoped_gc_configuration, validate_loaded_runtime
    params, robot = scoped_gc_configuration(reference, robot)
    selection = resolve_package(policy_package)
    if prealignment_coast_seconds is not None:
        from htd_loaded import validate_prealignment_coast_policy
        validate_prealignment_coast_policy(selection)
    from htd_candidate_package import require_posture_domain_diagnostic
    require_posture_domain_diagnostic(posture_domain_diagnostic,selection,
        transport=transport,current_grasp_reference=current_grasp_reference)
    reviewed_approach = transport and selection['selection']=='evaluated_posture_continuation'
    package = Path(selection['package_directory'])
    profile = Path(selection['profile_directory'])
    if sha(CONTROLLER) != '5003375d46f80d7e1064b50a1933e0a1fb70f7566724ac682379ef909beb8b9b':
        raise ValueError('Loaded GC/SlowStart requires the audited installed controller')
    spec = json.loads((package/'deployment.json').read_text())
    from htd_probe_config import resolve_artifacts
    from htd_policy import HTDPolicy
    from htd_native_probe import TargetLimits, load_gains
    from htd_serial_pd import SerialPD, warm_policy_actor
    import torch
    torch.set_num_threads(1)
    artifacts = resolve_artifacts(policy_path=package/'policy.ts', deployment_path=package/'deployment.json',
        training_source=package/'runtime_env.yaml', native_hold_source=profile/'carry_joint_targets.json')
    policy = HTDPolicy(artifacts.policy_path, artifacts.deployment_path)
    kp, kd, _ = load_gains(training_source=artifacts.training_source)
    pd = SerialPD(artifacts.hold_q, kp, kd, TargetLimits(), delay_steps=(0, 0, 0, 0),
                  training_source=artifacts.training_source)
    warm_policy_actor(policy)
    coordinator = LoadedZero(policy, pd, artifacts.hold_q,
        mass_kg=manifest['payload']['mass_kg'], friction=manifest['payload']['friction'],
        clearance_lift_m=manifest['walking_transfer']['clearance_lift_m'],
        current_grasp_reference=current_grasp_reference, goal_hold=goal_hold, contact_grace=contact_grace,
        clearance_minimum_rise_m=manifest['walking_transfer'].get('clearance_minimum_rise_m'),
        duration_s=180. if transport else 10.,
        transport_target_xyz=walking['target_payload_xyz'] if transport else None,
        transport_heading=walking['target_torso_heading_rad'] if transport else None,
        reviewed_approach=reviewed_approach, loaded_rate=loaded_rate,
        prealignment_coast_seconds=prealignment_coast_seconds)
    # All semantic/input guards precede the first configuration mutation.
    from htd_backlash_limits import configure, validate as validate_backlash
    from htd_output_coordinates import build_observer
    backlash = configure(trial, .005, upper_timeconst=upper_backlash_timeconst)
    validate_backlash(trial, expected_upper_timeconst=upper_backlash_timeconst)
    observer = build_observer(trial, trial/'vendor.xml')
    command_diagnostic=None
    if command_observer:
        from htd_command_observer import build
        from htd_command_scope import SIM_EXE, validate
        command_diagnostic=build(trial)
        command_diagnostic['environment']['IGRIS_HTD_COMMAND_OBSERVER_EXE']=SIM_EXE
        validate(trial,command_diagnostic)
        (trial/'htd_command_observer.json').write_text(json.dumps(command_diagnostic,indent=2)+'\n')
    params_path = trial/'htd_gc_params.yaml'
    (trial/'htd_gc_original_robot.yaml').write_bytes((trial/'robot.yaml').read_bytes())
    (trial/'htd_gc_original_params.yaml').write_bytes(PARAMS.read_bytes())
    params_path.write_text(yaml.safe_dump(params, sort_keys=False))
    robot['igris_c']['controller']['gain_parameter_path'] = str(params_path)
    (trial/'robot.yaml').write_text(yaml.safe_dump(robot, sort_keys=False))
    scope = dict(enabled=True, schema=CONTRACT, scope='loaded_zero_only', duration_s=10.,
        current_grasp_reference=current_grasp_reference, runtime_domain_validated=False,
        clearance_minimum_rise_m=coordinator.clearance_minimum_rise_m,
        runtime_domain='unvalidated actual acquired grasp; weights and owned15 contract unchanged' if current_grasp_reference else 'frozen profile76',
        policy_path=str(artifacts.policy_path), deployment_path=str(artifacts.deployment_path),
        training_source=str(artifacts.training_source), native_hold_source=str(artifacts.native_hold_source),
        checkpoint_path=str(package/'model.pt'), profile_directory=str(profile),
        policy_selection=selection, historically_trained_on=selection['historically_trained_on'],
        configured_delay_steps=[0, 0, 0, 0], training_delay_range_steps=[0, 2],
        params_path=str(params_path), observer_library=observer['library'],
        pacing_request_path=str(trial/'htd_pacing.request'), pacing_ack_path=str(trial/'htd_pacing.request.ack'),
        acquisition_rate=.5, loaded_rate=loaded_rate, runtime_contract_required=CONTRACT,
        runtime_integration_pending=False, required_private_runtime=dict(network='none', ipc='private', hardware_executed=False),
        initializer_forbidden=True, phase_transition='measured clearance -> native slowdown ACK -> LOW -> zero10s -> pause/holdgrip',
        slow_start='unchanged; raw motor/PJS upper entry transient is unvalidated and guarded',
        acquisition_change='startup owned15 backlash .005 and waist GC off; new experiment, no mid-grasp model edit',
        artifact_sha256=artifacts.evidence['sha256'], profile_sha256=spec['carry_profile']['profile_sha256'],
        checkpoint_sha256=selection['artifact_sha256']['model.pt'])
    if selection['selection'] == 'evaluated_continuation':
        scope['runtime_domain'] = 'unvalidated actual acquired grasp; selected continuation weights; frozen profile76 owned15 contract'
    if selection['selection'] == 'evaluated_posture_continuation':
        scope['runtime_domain'] = 'evaluated fixed-posture proxy; actual free-box grasp remains unvalidated'
        scope['profile_handoff_max_error_rad']=selection['profile_handoff_max_error_rad']
    if posture_domain_diagnostic:
        scope['posture_domain_diagnostic']=True
        scope['runtime_domain']='explicit SIM posture-domain diagnostic; acquired held8 may differ from evaluated proxy; no domain match claimed'
    if dip_variant is not None:
        scope['dip_damping']=dip_variant
    if command_diagnostic is not None:
        scope['command_observer']=command_diagnostic
    if wrist_mode_entry:
        from htd_wrist_entry import WristModeEntry
        scope['wrist_mode_entry']=WristModeEntry().contract
    if goal_hold:
        scope.update(scope='loaded_goal_hold_only', goal_hold=coordinator.goal_hold.contract,
            grasp_contact_rule=coordinator.contract['grasp_contact_rule'],
            phase_transition='measured clearance -> native slowdown ACK -> LOW -> goal hold10s -> pause/holdgrip')
    if contact_grace:
        scope['grasp_retention']=coordinator.grasp_retention.contract
    if transport:
        scope.update(scope='loaded_transport',duration_s=180.,
            transport=dict(schema='htd_loaded_transport/v1',
                primary_qp_start='cold',
                primary_qp_iteration_limits={'outer':10,'inner':10},
                target_payload_xyz=walking['target_payload_xyz'],
                reviewed_torso_heading_rad=walking['target_torso_heading_rad'],
                adaptive_clearance=dict(enabled=True,ik_position_trigger_m=.010,
                    action='Freeze previous requested EE uplift only after measured rise/source-unload/palm/speed gates qualify',
                    retained='Nominal8s path,0.5s measured dwell,20s deadline and original minimum actual rise'),
                command_observer_scope='First4096 effective commands at handoff; later capacity drops are expected',
                actual_control_trace='Full actor/serialPD archives through placement and pause'),
            phase_transition='qualified clearance -> observed LOW entry -> HTD route/settle -> lower -> supported open -> verified release -> pause')
        if reviewed_approach:
            scope['transport']['reviewed_approach']=True
    scope['payload_loading']=dict(actual_mass_kg=manifest['payload']['mass_kg'],
        frozen_proxy_added_mass_kg=spec['carry_profile']['added_payload_mass_kg'],
        mass_difference_kg=manifest['payload']['mass_kg']-spec['carry_profile']['added_payload_mass_kg'],
        force_threshold_mass='actual payload', material_calibrated=False,
        scope='Explicit relaxed SIM loading; frozen proxy is not an exact native contact model')
    if upper_backlash_timeconst is not None:
        scope['upper_backlash_limit']=backlash['upper_backlash_limit']
        manifest['htd_upper_backlash_limit']=backlash['upper_backlash_limit']
    if prealignment_coast_seconds is not None:
        from htd_reviewed_approach import coast_descriptor
        scope['prealignment_coast']=coast_descriptor(prealignment_coast_seconds)
        manifest['htd_prealignment_coast']=coast_descriptor(prealignment_coast_seconds)
    manifest.update(htd_loaded_zero=scope, htd_backlash_limit_timeconst_s=.005)
    prospective_runtime=dict(network='none', ipc='private', hardware_executed=False, simulation_rate_limit=.5)
    validate_loaded_runtime(prospective_runtime, manifest, robot, params, reference)
    paths = [trial/'scene.xml', trial/'vendor.xml', trial/'robot.yaml',
             trial/'htd_backlash_original_scene.xml',trial/'htd_backlash_limit_audit.json',
             trial/'htd_backlash_audit',params_path, CONTROLLER,
             Path(observer['library']), PARAMS, *[HERE/name for name in
             ('htd_loaded.py', 'htd_loaded_setup.py', 'htd_loaded_runtime.py', 'htd_candidate_package.py', 'htd_loaded_reference.py', 'htd_serial_pd.py', 'htd_policy.py',
              'htd_probe_config.py', 'htd_backlash_limits.py', 'htd_backlash_limits.cpp',
              'htd_output_coordinates.py', 'htd_output_coordinates.cpp', 'run.py', 'live_ik.py', 'walking_transfer.py')]]
    if selection['selection']=='evaluated_posture_continuation':
        paths.append(ROOT/'igris/igris-c-wbc-base/src/wbc_base/carry_profile.py')
    if command_diagnostic is not None:
        paths += [Path(command_diagnostic['library']),trial/'htd_command_observer.json',
                  *[Path(p) for p in command_diagnostic['input_sha256']],
                  *[HERE/name for name in ('htd_command_observer.py','htd_command_observer.cpp','htd_command_scope.py')]]
    if wrist_mode_entry:
        paths.append(HERE/'htd_wrist_entry.py')
    if goal_hold:
        paths += [HERE/'htd_goal_hold.py', Path(coordinator.goal_hold.contract['controller_source'])]
    if contact_grace:
        paths.append(HERE/'htd_grasp_retention.py')
    if transport:
        paths.append(ROOT/'igris/hl_ik/pipeline.py')
        paths += [HERE/name for name in ('htd_navigation.py','htd_navigation_feedback.py',
                                        'htd_reviewed_approach.py',
                                        'htd_transport_reference.py','htd_transport_phase.py','heading_route.py')]
    if dip_variant is not None:
        from dip_damping import ORIGINALS
        paths += [trial/name for name in ('compliance.json','dip_damping.json','dip_damping_audit',
                                          'htd_backlash_original_scene.xml','htd_backlash_limit_audit.json')]
        paths += [trial/('dip_damping_original_'+name) for name in ORIGINALS]
        paths += [Path(name) for name in dip_variant['source_sha256']]
        # Confirm the exact DIP scene is the input to the existing owned15
        # audit; subsequent runtime checks require the same complete chain.
        validate_dip_damping(trial, manifest=manifest)
    if prealignment_coast_seconds is not None:
        from htd_reviewed_approach import COAST_REPORT
        paths += [ROOT/COAST_REPORT,HERE/'transport_trial.py']
    scope['setup_sha256'] = {str(path): sha(path) for path in paths}
    if prealignment_coast_seconds is not None:
        from htd_loaded import prealignment_coast_selection
        prealignment_coast_selection(manifest)
    evidence = dict(schema=CONTRACT, scope='host configuration and zero-step native model audits only',
        pass_=True, hardware_executed=False, physics_steps=0, runtime_private_sim_verified=False,
        runtime_validation_required='After start, verify actual runtime.json/Docker private namespace and selected config paths',
        environment={'IGRIS_ROOM_PROFILE_LIBRARY': observer['library']},
        native_start_environment_required={'IGRIS_ROOM_PACER_SLOWDOWN': scope['pacing_request_path'],
                                           'IGRIS_ROOM_SLOW_REALTIME_FACTOR': '.05' if loaded_rate == .05 else '.1'},
        manifest_scope=scope, backlash=backlash, observer=observer, serial_pd_contract=pd.contract,
        coordinator_contract=coordinator.contract, required_palm_force_N=coordinator.required_palm_force,
        host_actor_warmed_history_unadvanced=policy._step_index == 0,
        worker_actor_warmup_required=True)
    evidence['pass'] = evidence.pop('pass_')
    (trial/'htd_loaded_preflight.json').write_text(json.dumps(evidence, indent=2)+'\n')
    (trial/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return evidence
