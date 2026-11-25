"""SIM loaded-zero IO adapter; the caller remains the ONLY LowCmd publisher.

The constructor validates the selected trial files and warms CPU inference.
It does not request modes, pacing, torque, initialization, reset or HOME.
Snapshot JSON is read by the caller before PD; fresh() reads only the native
binary clock and updates ages after IO. The caller still validates HL segment
end_time against native control_time; wall freshness uses the independent
live_ik_heartbeat, not the native30Hz response wall timestamp.
"""
from dataclasses import fields, replace
import json
from pathlib import Path
import struct
import time

import numpy as np
import yaml

from htd_loaded import (LoadedSnapshot, LoadedZero, validate_loaded_runtime, validate_loaded_rate,
                        upper_backlash_selection, prealignment_coast_selection)
from htd_loaded_setup import CONTRACT, PROFILE, PARAMS, sha
from htd_candidate_package import validate_runtime_selection, check_posture_handoff
from htd_native_probe import TargetLimits, imu_mount, load_gains, torso_imu
from htd_policy import HTDPolicy
from htd_probe_config import resolve_artifacts
from htd_serial_pd import SerialPD, warm_policy_actor, vector
from htd_output_coordinates import read_output
from igris.hl_ik.native_clock import read_clock


AGES = ('state_age_s', 'mode_age_s', 'pose_age_s', 'force_age_s', 'hand_age_s', 'hl_age_s')


def _jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [_jsonable(v) for v in value]
    return value


class LoadedRuntime:
    def __init__(self, out, node, *, services=None, clock=None, output=None,
                 monotonic=None, wall=None, spin=None):
        self.out, self.node = Path(out).resolve(), node
        self._monotonic, self._wall = monotonic or time.monotonic, wall or time.time
        self._clock = clock or (lambda: read_clock(self.out/'native_clock.bin'))
        self._output = output or (lambda: read_output(self.out/'output_coordinates.bin',
                                      native_time=lambda: self._clock()['sim']))
        manifest = json.loads((self.out/'manifest.json').read_text())
        coast_seconds=prealignment_coast_selection(manifest)
        upper_timeconst=upper_backlash_selection(manifest)
        runtime = json.loads((self.out/'runtime.json').read_text())
        robot = yaml.safe_load((self.out/'robot.yaml').read_text())
        from dip_damping import validate as validate_dip_damping
        dip_variant=validate_dip_damping(self.out)
        self.scope = manifest['htd_loaded_zero']
        scope = self.scope
        if scope.get('dip_damping')!=dip_variant:
            raise ValueError('Loaded DIP damping selection differs from validated trial')
        self.command_observer=scope.get('command_observer')
        self._command_observer_armed_at=None
        self._command_observer_ready=False
        if self.command_observer is not None:
            from htd_command_scope import validate
            validate(self.out,self.command_observer)
        self.wrist_entry=None
        self.wrist_entry_started=False
        if scope.get('wrist_mode_entry') is not None:
            if self.command_observer is None or scope.get('current_grasp_reference') is not True:
                raise ValueError('Observed current-grasp loaded scope required for wrist mode entry')
            from htd_wrist_entry import WristModeEntry
            self.wrist_entry=WristModeEntry()
            if self.wrist_entry.contract!=scope['wrist_mode_entry']:
                raise ValueError('Wrist mode entry contract changed')
        goal_hold = scope.get('goal_hold') is not None
        transport = scope.get('transport') is not None
        self.loaded_rate = validate_loaded_rate(scope.get('loaded_rate'), transport=transport,
            current_grasp_reference=scope.get('current_grasp_reference'))
        if self.loaded_rate == .05 and (scope.get('enabled') is not True
                or manifest.get('evaluation_profile') != 'relaxed_transport'):
            raise ValueError('Loaded pacing .05 requires explicit relaxed_transport SIM profile')
        expected_scope = 'loaded_transport' if transport else 'loaded_goal_hold_only' if goal_hold else 'loaded_zero_only'
        if scope.get('schema') != CONTRACT or scope.get('duration_s') != (180. if transport else 10.) or scope.get('scope') != expected_scope:
            raise ValueError('Explicit bounded loaded runtime scope required')
        walking=manifest['walking_transfer']
        if transport and (not goal_hold or scope.get('grasp_retention') is None
                or scope['transport'].get('schema')!='htd_loaded_transport/v1'
                or scope['transport'].get('target_payload_xyz')!=walking.get('target_payload_xyz')
                or scope['transport'].get('reviewed_torso_heading_rad')!=walking.get('target_torso_heading_rad')
                or scope['transport'].get('adaptive_clearance',{}).get('enabled') is not True
                or scope['transport']['adaptive_clearance'].get('ik_position_trigger_m')!=.010
                or not all(walking.get(k) is True for k in ('heading_preserving_route','xy_first_lowering',
                                                          'world_position_lowering','torso_relative_lowering'))):
            raise ValueError('Loaded transport destination or manipulation scope differs')
        if goal_hold and self.wrist_entry is None:
            raise ValueError('Goal hold requires observed current-grasp wrist mode entry')
        selection = validate_runtime_selection(scope)
        self.policy_selection=selection
        self.posture_domain_diagnostic=scope.get('posture_domain_diagnostic',False)
        reviewed_approach = transport and selection['selection']=='evaluated_posture_continuation'
        declared_approach = scope['transport'].get('reviewed_approach',False) if transport else False
        if type(declared_approach) is not bool or declared_approach is not reviewed_approach:
            raise ValueError('Loaded reviewed approach differs from validated policy selection')
        if reviewed_approach:
            route_source=Path(__file__).with_name('htd_reviewed_approach.py')
            if scope.get('setup_sha256',{}).get(str(route_source))!=sha(route_source):
                raise ValueError('Loaded reviewed approach source hash is missing or changed')
        package = Path(selection['package_directory'])
        profile = Path(selection['profile_directory'])
        if scope.get('profile_handoff_max_error_rad')!=selection.get('profile_handoff_max_error_rad'):
            raise ValueError('Posture handoff qualification differs from selected package')
        if (scope.get('checkpoint_sha256') != selection['artifact_sha256']['model.pt']
                or scope.get('profile_sha256') != selection['profile_sha256']):
            raise ValueError('Loaded selection checkpoint/profile hash differs from setup')
        self.pacing_request_path = self.out/'htd_pacing.request'
        self.pacing_ack_path = self.out/'htd_pacing.request.ack'
        expected_paths = dict(policy_path=package/'policy.ts', deployment_path=package/'deployment.json',
            training_source=package/'runtime_env.yaml', native_hold_source=profile/'carry_joint_targets.json',
            checkpoint_path=package/'model.pt', profile_directory=profile,
            params_path=self.out/'htd_gc_params.yaml', pacing_request_path=self.pacing_request_path,
            pacing_ack_path=self.pacing_ack_path, observer_library=self.out/'htd_output_coordinates.so')
        for key, path in expected_paths.items():
            if scope.get(key) != str(path): raise ValueError('Unexpected loaded runtime path: '+key)
        if robot['igris_c']['controller']['gain_parameter_path'] != scope['params_path']:
            raise ValueError('Actual robot config does not select trial GC parameters')
        params = yaml.safe_load(Path(scope['params_path']).read_text())
        reference = yaml.safe_load(PARAMS.read_text())
        validate_loaded_runtime(runtime, manifest, robot, params, reference)
        if any((self.out/name).exists() for name in ('htd_pacing.request', 'htd_pacing.request.ack',
                'htd_initial_state.request', 'htd_initial_state_applied.json')):
            raise FileExistsError('Loaded runtime requires fresh pacing and no initializer activation')
        for path, expected in scope['setup_sha256'].items():
            if sha(path) != expected: raise ValueError('Loaded setup hash changed: '+path)
        for name, expected in scope['profile_sha256'].items():
            if sha(profile/name) != expected: raise ValueError('Loaded profile hash changed: '+name)
        for key in ('policy', 'deployment', 'training', 'native_hold'):
            path = expected_paths[{'policy':'policy_path', 'deployment':'deployment_path',
                                  'training':'training_source', 'native_hold':'native_hold_source'}[key]]
            if sha(path) != scope['artifact_sha256'][key]: raise ValueError('Loaded artifact changed: '+key)
        from htd_backlash_limits import validate as validate_backlash
        audit = validate_backlash(self.out, expected_upper_timeconst=upper_timeconst)
        observer = json.loads((self.out/'htd_output_observer.json').read_text())
        if (not audit['pass'] or audit['timeconst_s'] != .005 or audit['variant_scene_sha256'] != sha(self.out/'scene.xml')
                or observer['library_sha256'] != sha(scope['observer_library'])):
            raise ValueError('Native model/observer audit differs from loaded files')
        import torch
        torch.set_num_threads(1)
        artifacts = resolve_artifacts(**{k: expected_paths[k] for k in
            ('policy_path','deployment_path','training_source','native_hold_source')})
        policy = HTDPolicy(artifacts.policy_path, artifacts.deployment_path)
        kp, kd, _ = load_gains(training_source=artifacts.training_source)
        pd = SerialPD(artifacts.hold_q, kp, kd, TargetLimits(), delay_steps=(0, 0, 0, 0),
                      training_source=artifacts.training_source)
        warm_policy_actor(policy)
        self.coordinator = LoadedZero(policy, pd, artifacts.hold_q, mass_kg=manifest['payload']['mass_kg'],
            friction=manifest['payload']['friction'], clearance_lift_m=manifest['walking_transfer']['clearance_lift_m'],
            current_grasp_reference=scope.get('current_grasp_reference', False), goal_hold=goal_hold,
            contact_grace=scope.get('grasp_retention') is not None,
            clearance_minimum_rise_m=manifest['walking_transfer'].get('clearance_minimum_rise_m'),
            duration_s=scope['duration_s'],
            transport_target_xyz=walking['target_payload_xyz'] if transport else None,
            transport_heading=walking['target_torso_heading_rad'] if transport else None,
            reviewed_approach=reviewed_approach, loaded_rate=self.loaded_rate,
            prealignment_coast_seconds=coast_seconds)
        if scope.get('clearance_minimum_rise_m', self.coordinator.clearance_minimum_rise_m) != self.coordinator.clearance_minimum_rise_m:
            raise ValueError('Loaded clearance scope differs from acquisition manifest')
        if goal_hold and (scope['goal_hold'] != self.coordinator.goal_hold.contract
                         or scope.get('grasp_contact_rule') != self.coordinator.contract['grasp_contact_rule']):
            raise ValueError('Loaded goal hold contract changed')
        if scope.get('grasp_retention') is not None and scope['grasp_retention']!=self.coordinator.grasp_retention.contract:
            raise ValueError('Loaded grasp retention contract changed')
        self.mount = imu_mount(self.out/'vendor.xml')
        if services is None:
            from igris_c_sdk.srv import ControlModeCommandRequest, MujocoSimCmd
            services = dict(mode=ControlModeCommandRequest, sim=MujocoSimCmd)
        if spin is None:
            import rclpy
            spin = lambda: rclpy.spin_once(node, timeout_sec=.001)
        self._spin, self._services, self._clients = spin, services, {}
        deadline = self._monotonic()+2.
        discovered = {}
        while True:
            available = node.get_service_names_and_types()
            for key, name in [('mode', 'ControlModeCommandRequest'), ('sim', 'MujocoSimCmd')]:
                matches = [path for path, types in available if path.startswith('/igris_c_001/')
                           and 'igris_c_sdk/srv/'+name in types]
                if len(matches) > 1: raise RuntimeError('Ambiguous private SIM service: '+name)
                if matches:
                    if key in discovered and discovered[key] != matches[0]:
                        raise RuntimeError('Private SIM service identity changed: '+name)
                    if key not in self._clients:
                        discovered[key] = matches[0]
                        self._clients[key] = node.create_client(services[key], matches[0])
            if len(self._clients) == 2 and all(c.service_is_ready() for c in self._clients.values()):
                break
            if self._monotonic() >= deadline:
                raise RuntimeError('Timed out discovering ready private SIM services')
            self._spin()  # Graph discovery only; no service request or motor publication.
        self._pacing_requested_at = self._pacing_ack = self._low_future = None
        self.actor_records, self.pd_records, self.events = [], [], []
        self._last_environment = None
        self.preflight = dict(schema=CONTRACT, hardware_executed=False, artifacts=scope['artifact_sha256'],
            policy_selection=selection,
            scope=expected_scope, goal_hold=scope.get('goal_hold'),
            command_observer=self.command_observer,
            wrist_mode_entry=self.wrist_entry.contract if self.wrist_entry is not None else None,
            current_grasp_reference=self.coordinator.current_grasp_reference, runtime_domain_validated=False,
            clearance_minimum_rise_m=self.coordinator.clearance_minimum_rise_m,
            mounted_torso_imu_wxyz=self.mount.tolist(), actor_warmed=True, history_unadvanced=policy._step_index == 0,
            output_observer_warmed=False, serial_pd_contract=pd.contract,
            private_runtime='runtime.json and selected configuration validated; SIM RobotState gates services/commands',
            hl_freshness='Independent wall heartbeat; caller separately checks native response end_time lag')
        if self.posture_domain_diagnostic:
            self.preflight['posture_domain_diagnostic']=True
        if upper_timeconst is not None:
            self.preflight['upper_backlash_limit']=audit['upper_backlash_limit']
        if reviewed_approach:
            self.preflight['reviewed_approach']=True

    def warm_observer(self):
        sample = self._output()
        for key in ('q_output','dq_output','q_encoder','dq_encoder'): vector(sample[key], key)
        self.preflight['output_observer_warmed'] = True
        self.preflight['warm_output_sequence'] = int(sample['sequence'])
        return sample

    def note_environment(self, environment):
        """Called from RobotState callback, including during acquisition."""
        self._last_environment = environment
        if environment != 1:
            raise RuntimeError('Loaded runtime received non-SIM RobotState')

    def _clock_after_observations(self, observation_times):
        """Observe publication completion, without rewriting either timestamp.

        Inner producers can publish pose/force/output one2100Hz step before
        the clock interposer returns. Only that bounded gap may wait, for at
        most5ms wall time. The existing1ns JSON-force roundoff allowance stays;
        max(observation_times) is only the wait bound, never a synthetic clock.
        The caller's freshness and PD-deadline guards still run after this IO.
        """
        if not observation_times:
            raise ValueError('Captured observation timestamps required')
        stamps={}
        for key,value in observation_times.items():
            if (key not in ('pose','force','output') or isinstance(value,(bool,np.bool_))
                    or not isinstance(value,(int,float,np.integer,np.floating))
                    or not np.isfinite(value) or value<0):
                raise ValueError('Finite nonnegative observation timestamp required: '+str(key))
            stamps[key]=float(value)
        pose=stamps.get('pose');wait_until=max(stamps.values())
        started=self._monotonic()
        deadline=started+.005
        clock=self._clock()
        native=float(clock['sim'])
        if not np.isfinite(native) or native<0:
            raise RuntimeError('Invalid native clock after captured pose')
        if wait_until-native<=1e-9:
            return clock
        original=native;reads=1

        def fail(reason):
            event=dict(event='pose_clock_sync_failed',reason=reason,
                original_clock_sim=original,pose_sim=pose,last_clock_sim=native,
                observation_native_times_s=stamps.copy(),maximum_observation_sim=wait_until,
                ahead_s=wait_until-native,wall_wait_s=self._monotonic()-started,reads=reads)
            self.events.append(event)
            raise RuntimeError(f'Pose-clock synchronization {reason}: pose={pose!r}, '
                f'observations={stamps!r}, clock={native!r}, original_clock={original!r}, ahead_s={event["ahead_s"]!r}, '
                f'wall_wait_s={event["wall_wait_s"]!r}, reads={reads}')

        if wait_until-native>1/2100.+1e-9:
            fail('future beyond one native step')
        while True:
            now=self._monotonic();elapsed=now-started
            if not np.isfinite(elapsed) or elapsed<0:fail('monotonic clock moved backward or invalid')
            if now>=deadline:fail('timeout')
            time.sleep(min(.00005,deadline-now))
            if self._monotonic()>=deadline:fail('timeout')
            previous=native;reads+=1
            try:
                clock=self._clock()
                native=float(clock['sim'])
            except Exception as exc:
                fail('clock read failed: '+str(exc))
            if not np.isfinite(native) or native<0:fail('invalid native clock')
            if native<previous:fail('native clock moved backward')
            now=self._monotonic();elapsed=now-started
            if not np.isfinite(elapsed) or elapsed<0:fail('monotonic clock moved backward or invalid')
            if now>=deadline:fail('timeout')
            if wait_until-native<=1e-9:
                self.events.append(dict(event='pose_clock_synchronized',
                    original_clock_sim=original,pose_sim=pose,accepted_clock_sim=native,
                    observation_native_times_s=stamps.copy(),maximum_observation_sim=wait_until,
                    wall_wait_s=elapsed,reads=reads))
                return clock

    def snapshot(self, state, received, mode, environment, mode_at, stamp, poses, pelvis_index,
                 forces, hand_wall, hl_response, upper_output_target, walking_state, torso_index=None,
                 payload_index=None,hand_indices=None,applied_segment=None,control_time_s=None):
        for key in ('q','dq','status','imu_quat','imu_gyro'):
            if key not in state: raise ValueError('Missing documented SDK state field: '+key)
        for key in ('q','dq','status'): vector(state[key], key)
        if (not hl_response or hl_response.get('accepted') is not True or hl_response.get('prior_status') != 'used'):
            raise ValueError('Accepted live HL reference required')
        if 'heartbeat_wall' not in hl_response:
            raise ValueError('Independent HL heartbeat_wall is required; response wall is native-rate evidence')
        if forces.get('support_force_groups_configured') is not True:
            raise ValueError('Configured native support-force groups required')
        for key in ('clearance_measured_rise_m','clearance_ready_s','payload_speed_m_s'):
            if key not in walking_state: raise ValueError('Missing measured walking evidence: '+key)
        quat, gyro = torso_imu(state['imu_quat'], state['imu_gyro'], self.mount)
        sample = self._output()
        observation_times=dict(pose=stamp[1],output=sample['sim'])
        if 'sim' in forces:observation_times['force']=forces['sim']
        clock = self._clock_after_observations(observation_times)
        now, wall = self._monotonic(), self._wall()  # clocks AFTER IO
        self.note_environment(environment)
        transport_fields={}
        if self.coordinator.transport:
            if applied_segment is None or control_time_s is None:
                raise ValueError('Transport requires the actually applied live HL segment')
            # At the clearance/carry boundary the final clearance spline may
            # still execute during wrist priming. It authorizes no manipulation.
            if self.coordinator.state=='prepared' and applied_segment.get('walking_phase')=='clearance':
                transport_fields=dict(transport_phase='carry',phase_target_valid=False,
                    placement_target_xyz=np.asarray(walking_state['target_payload_xyz'],dtype=float),placement_done=False)
            else:
                from htd_transport_reference import applied_transport_reference
                transport_fields=applied_transport_reference(walking_state,applied_segment,
                                                             control_time_s,upper_output_target)
            transport_fields['destination_support_z_N']=float(forces['destination_support_force_world_z_N'])
        return LoadedSnapshot(native_time=clock['sim'], wall_time=now, environment=environment, mode=mode,
            state_age_s=now-received, mode_age_s=now-mode_at, joint_status=np.asarray(state['status']).copy(),
            output=sample, torso_quat_wxyz=quat, torso_gyro=gyro,
            pelvis_pose=np.asarray(poses[pelvis_index], dtype=float).copy(), pose_age_s=wall-stamp[0],
            force_age_s=wall-forces['wall'], hand_age_s=wall-hand_wall,
            hl_age_s=wall-hl_response['heartbeat_wall'], grasp_verified=hl_response.get('grasp_verified') is True,
            clearance_rise_m=walking_state['clearance_measured_rise_m'], clearance_ready_s=walking_state['clearance_ready_s'],
            payload_speed_m_s=walking_state['payload_speed_m_s'], source_support_z_N=forces['source_support_force_world_z_N'],
            palm_tip_force_N=np.asarray(forces['palm_tip_force_N'], dtype=float).copy(),
            upper_output_target=vector(upper_output_target, 'upper_output_target').copy(), pacing_ack=self._pacing_ack,
            torso_pose=None if torso_index is None else np.asarray(poses[torso_index], dtype=float).copy(),
            pose_native_time=float(stamp[1]),force_native_time=forces.get('sim'),
            payload_pose=None if payload_index is None else np.asarray(poses[payload_index],dtype=float).copy(),
            hand_poses=None if hand_indices is None else np.asarray(poses[list(hand_indices)],dtype=float).copy(),
            **transport_fields)

    def fresh(self, snapshot):
        """No JSON/callback/output reads; synchronize existing stamps, then age."""
        observation_times=dict(output=snapshot.output['sim'])
        if snapshot.pose_native_time is not None:observation_times['pose']=snapshot.pose_native_time
        if snapshot.force_native_time is not None:observation_times['force']=snapshot.force_native_time
        clock = self._clock_after_observations(observation_times)
        now = self._monotonic()
        elapsed = now-snapshot.wall_time
        if elapsed < 0: raise RuntimeError('Monotonic clock moved backwards')
        return replace(snapshot, native_time=clock['sim'], wall_time=now,
                       **{key: getattr(snapshot, key)+elapsed for key in AGES})

    def navigation_signal(self,control_time,stamp,poses,pelvis_index,torso_index):
        """Measured final-arrival qualification; never supplies a vendor twist."""
        if not self.coordinator.transport:
            return None
        feedback=self.coordinator.goal_hold
        if feedback.navigation is None:
            return None
        from types import SimpleNamespace
        snapshot=SimpleNamespace(native_time=self._clock_after_observations({'pose':stamp[1]})['sim'],pose_native_time=float(stamp[1]),
            torso_pose=np.asarray(poses[torso_index]),pelvis_pose=np.asarray(poses[pelvis_index]))
        return feedback.arrival_signal(control_time,snapshot)

    def refresh_output(self, snapshot):
        """Use AFTER actor and BEFORE PD computation; never after computing PD."""
        output = self._output()
        return self.fresh(replace(snapshot, output=output, pacing_ack=self._pacing_ack))

    def prepare_pd_snapshot(self, snapshot):
        """Resolve an actor deadline crossed by the newer PD measurement once.

        The300Hz observer may advance past a50Hz boundary after the caller's
        actor check. Run that due actor on the actual new sample, then reread
        output after inference. Existing missed-deadline and publication gates
        still reject slow inference; this never replays or holds an old effort.
        """
        current=self.refresh_output(snapshot)
        if self.coordinator.policy.policy_due(current.output['sim']):
            record=self.coordinator.advance_actor(current)
            if record is not None:
                self.record_actor(record,current)
                self.events.append(dict(event='actor_deadline_crossed_before_pd',
                    observation_native=current.output['sim'],native=current.native_time,
                    monotonic=current.wall_time,actor_count=self.coordinator.actor_count))
            if self.coordinator.state=='complete':return None
            current=self.refresh_output(current)
        return current

    def request_pacing(self):
        """Idempotently request once and poll; True only after native ACK."""
        if self.command_observer is not None and not self._command_observer_ready:
            if self._command_observer_armed_at is None:
                from htd_command_scope import validate
                validate(self.out,self.command_observer)
                self._command_observer_armed_at=self._monotonic()
                Path(self.command_observer['arm_path']).touch(exist_ok=False)
                self.events.append(dict(event='command_observer_armed',monotonic=self._command_observer_armed_at,
                    trace_path=self.command_observer['trace_path']))
            try:
                with Path(self.command_observer['trace_path']).open('rb') as trace:
                    header=trace.read(64)
            except FileNotFoundError:
                header=b''
            if len(header)==64:
                values=struct.unpack('<8sIIIIQQQQQ',header)
                self._command_observer_ready=(values[:5]==(b'HTDCMD01',1,64,8064,
                    self.command_observer['contract']['max_records']) and values[9]>0
                    and self._command_observer_armed_at<=values[8]/1e9<=self._monotonic())
            if not self._command_observer_ready:
                if self._monotonic()-self._command_observer_armed_at>.5:
                    raise RuntimeError('Command observer did not acknowledge armed SIM mapping boundary')
                return False
            self.events.append(dict(event='command_observer_ready',monotonic=self._monotonic(),pid=values[9]))
        if self._pacing_requested_at is None:
            if self.pacing_ack_path.exists():
                raise RuntimeError('Pacing ACK predates this runtime request')
            self._pacing_requested_at = self._monotonic()
            self.pacing_request_path.touch(exist_ok=False)
            self.events.append(dict(event='pacing_requested', monotonic=self._pacing_requested_at))
        return self.poll_pacing() is not None

    def poll_pacing(self):
        if self._pacing_ack is not None: return self._pacing_ack
        if not self.pacing_ack_path.exists(): return None
        ack = json.loads(self.pacing_ack_path.read_text())
        if (self._pacing_requested_at is None or ack.get('rate') != self.loaded_rate or ack.get('previous_rate') != .5
                or ack.get('monotonic_s', -1) < self._pacing_requested_at
                or ack['monotonic_s'] > self._monotonic() or not 0 <= ack.get('native_time_s', -1) <= self._clock()['sim']):
            raise RuntimeError('Invalid or preexisting native pacing acknowledgement')
        self._pacing_ack = ack
        self.events.append(dict(event='pacing_acknowledged', acknowledgement=ack))
        return ack

    def entry_command(self, snapshot, command):
        """Keep measured grasp guards while the scoped helper selects wrist coordinates."""
        if self.wrist_entry is None:return command
        if self.coordinator.state=='prepared':
            self.coordinator.check_handoff_ready(snapshot,require_fresh_output=not self.wrist_entry_started)
        else:
            self.coordinator._check(snapshot)
        incoming={**command,'kinematic_modes':np.ones(5,dtype=int)}
        result=self.wrist_entry.apply(incoming,snapshot)
        self.wrist_entry_started=True
        return result

    def entry_ready(self,snapshot):
        if self.coordinator.transport and (snapshot.transport_phase!='carry' or not snapshot.phase_target_valid):
            return False
        return self.wrist_entry is None or (self.wrist_entry_started and self.wrist_entry.ready_for_low(snapshot))

    def request_low(self, snapshot):
        if not self.entry_ready(snapshot):
            raise RuntimeError('Wrist motor hold must be primed before LOW')
        try:
            posture=check_posture_handoff(self.policy_selection,self.coordinator.hold_q,snapshot.upper_output_target,
                posture_domain_diagnostic=getattr(self,'posture_domain_diagnostic',False))
        except ValueError as error:
            self.events.append(dict(event='posture_handoff_rejected',native_time=snapshot.native_time,error=str(error)))
            raise
        if posture is not None:
            event='posture_handoff_qualified' if posture['profile_match'] else 'posture_handoff_out_of_domain_diagnostic'
            self.events.append(dict(event=event,native_time=snapshot.native_time,**posture))
        request_fields = self.coordinator.request_handoff(snapshot)
        request = self._services['mode'].Request()
        request.request_id = 'htd-loaded-low-'+str(time.time_ns())
        request.command_type = request_fields['mode_command_type']
        request.preset_id, request.is_cyclic = request_fields['preset_id'], request_fields['is_cyclic']
        self._last_environment = snapshot.environment
        self._low_future = self._clients['mode'].call_async(request)
        return self._low_future

    def low_ready(self):
        if self._low_future is None: return False
        if not self._low_future.done():
            if (self._monotonic()-self.coordinator.requested_wall >= .5
                    or self._clock()['sim']-self.coordinator.requested_native >= .05):
                raise RuntimeError('LOW transition acknowledgement timeout')
            return False
        response = self._low_future.result()
        if not response.success: raise RuntimeError('LOW service rejected: '+str(response.message))
        return True

    def waist_request(self):
        if self.coordinator.epoch <= 0: raise RuntimeError('Loaded ownership epoch required for actual waist')
        sample = self._output()
        clock, now = self._clock(), self._monotonic()
        q = vector(sample['q_output'], 'q_output').copy()
        spec = dict(schema='htd_loaded_actual_waist/v1', epoch=self.coordinator.epoch,
            native_sample_sim=float(sample['sim']), native_time_s=float(clock['sim']),
            sample_monotonic=float(sample['monotonic']), request_monotonic=float(now), waist=q[:3].tolist())
        from htd_loaded_reference import apply_loaded_waist
        apply_loaded_waist(q, dict(htd_loaded_waist=spec, nominal=q, measured=q), enabled=True)
        return spec, q

    def record_actor(self, record, snapshot, stage=None):
        row = {field.name: getattr(record, field.name) for field in fields(record)}
        row.update(q_output=snapshot.output['q_output'], dq_output=snapshot.output['dq_output'],
                   torso_quat_wxyz=snapshot.torso_quat_wxyz, torso_gyro=snapshot.torso_gyro,
                   inference_sample_native=snapshot.native_time, inference_sample_monotonic=snapshot.wall_time)
        if self.coordinator.goal_hold is not None:
            metrics = self.coordinator.goal_hold.metrics
            for key in ('command_xyz', 'torso_pose', 'pelvis_pose', 'pose_native_time',
                        'goal_xy_world', 'heading_goal_rad', 'velocity_xy_world', 'yaw_rate_rad_s',
                        'position_error_m', 'heading_error_rad', 'arrived', 'settled_s', 'success'):
                row['goal_hold_'+key] = metrics[key]
        if self.coordinator.transport:
            row.update(transport_phase=snapshot.transport_phase,
                       destination_support_z_N=snapshot.destination_support_z_N,
                       payload_pose=snapshot.payload_pose,
                       navigation_waypoint=self.coordinator.goal_hold.navigation.route.index,
                       navigation_recovery=self.coordinator.goal_hold.navigation.recovering)
        self.actor_records.append(row)
        if stage: self.events.append(dict(event='actor_stage', **stage))

    def record_pd(self, frame, snapshot, published, stage=None):
        row = {field.name: getattr(frame.pd, field.name) for field in fields(frame.pd)}
        row.update({f'published_{key}': value for key, value in frame.command.items()})
        row.update(compute_native_time=snapshot.native_time, compute_monotonic=snapshot.wall_time,
                   publication_clock_available=published is not None,
                   publication_native_time=published.native_time if published is not None else np.nan,
                   publication_monotonic=published.wall_time if published is not None else np.nan,
                   pelvis_pose=snapshot.pelvis_pose, upper_output_target=snapshot.upper_output_target,
                   palm_tip_force_N=snapshot.palm_tip_force_N, source_support_z_N=snapshot.source_support_z_N,
                   payload_speed_m_s=snapshot.payload_speed_m_s, clearance_rise_m=snapshot.clearance_rise_m,
                   **{key: getattr(published, key) if published is not None else np.nan for key in AGES})
        if self.coordinator.transport:
            row.update(transport_phase=snapshot.transport_phase,
                       destination_support_z_N=snapshot.destination_support_z_N,
                       phase_target_valid=snapshot.phase_target_valid,payload_pose=snapshot.payload_pose,
                       hand_poses=snapshot.hand_poses)
        self.pd_records.append(row)
        if stage: self.events.append(dict(event='pd_stage', **stage))

    def pause_cleanup(self, reason, last_command, publish):
        """Publish owned-zero once, request SIMpause0, save evidence; no grip release.

        Saves explicit evidence before raising any cleanup failure.
        It must keep hand/upper ownership until native pause is acknowledged.
        """
        completed = self.coordinator.state == 'complete'
        goal = self.coordinator.goal_hold
        if goal is not None and 'prealignment_coast' in goal.contract:
            goal.censor_coast(reason)  # No new measured timestamp exists at outer failure.
        result = dict(hardware_executed=False, bounded_zero_completed=completed and goal is None,
            bounded_goal_hold_completed=completed and goal is not None and not self.coordinator.transport,
            loaded_transport_completed=completed and self.coordinator.transport,
            goal_hold_metrics=goal.metrics if goal is not None else None,
            goal_hold_events=goal.events if goal is not None else [],
            pause_acknowledged=False, hold_grip=not (completed and self.coordinator.transport), reason=str(reason), cleanup_errors=[])
        retention=self.coordinator.grasp_retention
        result.update(grasp_retention_metrics=retention.metrics if retention is not None else None,
                      grasp_retention_metrics_scope='last accepted guard check; failure reason is separate',
                      grasp_retention_events=retention.events if retention is not None else [])
        if self.coordinator.transport:
            placement=self.coordinator.placement
            result.update(placement_metrics=placement.metrics,placement_events=placement.events,
                          placement_contract=placement.contract,
                          hold_grip=placement.phase not in ('open','done'))
        try:
            cleanup = self.coordinator.abort(reason, last_command)
            publish(cleanup.command)
        except Exception as exc:
            result['cleanup_errors'].append('zero publication: '+str(exc))
        try:
            if self._last_environment != 1:
                raise RuntimeError('Fresh SIM observation required for pause service')
            request = self._services['sim'].Request()
            request.request_id = 'htd-loaded-pause-'+str(time.time_ns())
            request.command_type = 0
            future = self._clients['sim'].call_async(request)
            deadline = self._monotonic()+2.
            while not future.done():
                if self._monotonic() >= deadline: raise RuntimeError('SIM pause acknowledgement timeout')
                self._spin()
            response = future.result()
            if not response.success: raise RuntimeError('SIM pause rejected: '+str(response.message))
            result['pause_acknowledged'] = True
        except Exception as exc:
            result['cleanup_errors'].append('pause: '+str(exc))
        result.update(actor_records=len(self.actor_records), pd_records=len(self.pd_records),
            wrist_mode_entry_events=self.wrist_entry.events if self.wrist_entry is not None else [],
            actor_count=self.coordinator.actor_count, published_pd_count=self.coordinator.published_count,
            preflight=self.preflight, coordinator_events=self.coordinator.events, events=self.events,
            coordinator_contract=self.coordinator.contract,
            current_grasp_reference_evidence=self.coordinator.current_grasp_evidence,
            strict_settling_verified=False, loaded_transport_verified=completed and self.coordinator.transport)
        for kind, rows in [('actor', self.actor_records), ('pd', self.pd_records)]:
            if rows:
                np.savez_compressed(self.out/f'htd_loaded_{kind}.npz',
                    **{key: np.stack([row[key] for row in rows]) for key in rows[0]})
        (self.out/'htd_loaded_result.json').write_text(json.dumps(_jsonable(result), indent=2, allow_nan=False)+'\n')
        if result['cleanup_errors'] or not result['pause_acknowledged']:
            raise RuntimeError('Loaded cleanup failed; saved htd_loaded_result.json: '+ '; '.join(result['cleanup_errors']))
        return result
