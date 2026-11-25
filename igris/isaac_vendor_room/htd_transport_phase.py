"""Measured SIM placement authorization; no IO, policy reset or command writer."""
import numpy as np
from scipy.spatial.transform import Rotation

from htd_goal_hold import GoalHold


class TransportPlacement:
    phases = ('carry', 'lower', 'open', 'done')

    def __init__(self, target_xyz, mass_kg, *, friction=4., qualified_translation_limit_m=.015):
        target = np.asarray(target_xyz, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all() or not np.isfinite(mass_kg) or mass_kg <= 0:
            raise ValueError('Finite transport destination and positive payload mass required')
        self.target = target.copy()
        if not np.isfinite(friction) or friction<=0 or qualified_translation_limit_m not in (.015,.025):
            raise ValueError('Positive friction and explicit15/25mm qualified geometry limit required')
        self.required_each=1.5*mass_kg*9.81/(2*friction)
        self.qualified_translation_limit=qualified_translation_limit_m
        self.required_support = .5*mass_kg*9.81
        self.phase = 'carry'
        self.captured = False
        self.completion_ready = False
        self.hand_contact_required = True
        self._last_native = self._pose_stamp = self._force_stamp = None
        self._support_since = self._support_last = None
        self._lower_since = self._open_since = self._done_since = None
        self.events, self.metrics = [], {}
        self.contract = dict(schema='htd_loaded_transport_placement/v1', phases=list(self.phases),
            phase_authority='Actually executing fresh accepted HL segment; never latest unapplied plan',
            target_payload_xyz=self.target.tolist(), duration_s=180., lower_timeout_s=25.,
            opening_minimum_s=7., done_verification_timeout_s=3., support_dwell_s=.5,
            required_destination_support_z_N=self.required_support,
            payload_xy_error_limit_m=.03, payload_z_error_limit_m=.025, payload_speed_limit_m_s=.02,
            maximum_native_sample_age_s=.05, maximum_relative_translation_m=.015,
            qualified_translation_limit_m=qualified_translation_limit_m,
            wider_geometry_condition='Both hands friction-qualified, or lower with actual qualifying destination support/pose/speed',
            maximum_payload_rotation_rad=float(np.deg2rad(15)), minimum_each_outward_open_m=.045,
            maximum_each_open_hand_force_N=.05,
            opening_measurement='Each actual EE relative to payload, projected outward on captured hand-to-hand axis',
            contact_unloading='Only lower with measured qualifying destination support/pose/speed; geometry retained',
            completion='Explicit placement_done plus measured opening/support; coordinator also verifies PD coverage',
            limitation='Measured support and pose do not certify exact COM contact distribution or hardware safety')

    def _read(self, s):
        native = float(s.native_time)
        stamps = [float(s.pose_native_time), float(s.force_native_time)]
        if (not np.isfinite([native, *stamps]).all() or min(native, *stamps) < 0
                or any(not -1e-9 <= native-t <= .05+1e-9 for t in stamps)):
            raise ValueError('Fresh finite native placement pose/force samples required')
        if (self._last_native is not None and native < self._last_native-1e-9
                or self._pose_stamp is not None and stamps[0] < self._pose_stamp-1e-9
                or self._force_stamp is not None and stamps[1] < self._force_stamp-1e-9):
            raise ValueError('Placement native sample clock moved backward')
        box = GoalHold._pose(s.payload_pose, 'payload_pose')
        hands = np.asarray(s.hand_poses, dtype=float)
        if hands.shape != (2, 7): raise ValueError('Two actual placement hand poses required')
        hands = np.stack([GoalHold._pose(p, 'hand_pose') for p in hands])
        force = np.asarray(s.palm_tip_force_N, dtype=float)
        support, speed = float(s.destination_support_z_N), float(s.payload_speed_m_s)
        if (force.shape != (4,) or not np.isfinite(force).all() or np.any(force < 0)
                or not np.isfinite([support, speed]).all() or speed < 0):
            raise ValueError('Finite measured placement support, forces and speed required')
        if self._pose_stamp == stamps[0] and (not np.array_equal(box, self._box)
                                              or not np.array_equal(hands, self._hands)):
            raise ValueError('Changed placement pose at identical native stamp')
        measured_force = np.r_[force, support]
        if self._force_stamp == stamps[1] and not np.array_equal(measured_force, self._forces):
            raise ValueError('Changed placement force at identical native stamp')
        return native, stamps, box, hands, force.reshape(2, 2).sum(axis=1), support, speed, measured_force

    @staticmethod
    def _relative(box, hands):
        hr = Rotation.from_quat(hands[:, 3:].copy(), scalar_first=True)
        br = Rotation.from_quat(box[3:].copy(), scalar_first=True)
        return hr.inv().apply(box[:3]-hands[:, :3]), hr.inv()*br, br

    def capture(self, s):
        if self.captured: raise RuntimeError('Transport grasp geometry may be captured only once')
        _, _, box, hands, _, _, _, _ = self._read(s)
        self._xyz, self._rot, self._box_rotation = self._relative(box, hands)
        self.captured = True
        self.events.append(dict(event='captured_transport_grasp', native_time=s.native_time))

    def check(self, s, *, navigation_ready):
        phase = s.transport_phase
        if phase not in self.phases: raise ValueError('Unknown transport phase')
        old, new = self.phases.index(self.phase), self.phases.index(phase)
        if new < old or new > old+1: raise RuntimeError('Illegal transport phase transition')
        if type(s.placement_done) is not bool or s.placement_done and phase != 'done':
            raise RuntimeError('Placement completion requires explicit done phase')
        if phase != 'carry':
            if s.phase_target_valid is not True:
                raise RuntimeError('Fresh accepted phase target required for placement')
            target = np.asarray(s.placement_target_xyz, dtype=float)
            if target.shape != (3,) or not np.isfinite(target).all() or not np.allclose(target, self.target, atol=1e-9, rtol=0):
                raise ValueError('Placement destination differs from explicit transport target')
            if not self.captured: raise RuntimeError('Captured transport grasp required')
        native, stamps, box, hands, forces, support, speed, measured_force = self._read(s)
        xyz, rot, box_rotation = self._relative(box, hands)
        xy_error, z_error = float(np.linalg.norm(box[:2]-self.target[:2])), float(abs(box[2]-self.target[2]))
        supported = support > self.required_support and xy_error < .03 and z_error < .025 and speed < .02
        geometry_limit=(self.qualified_translation_limit if np.all(forces>=self.required_each)
                        or phase=='lower' and supported else .015)
        if self.captured:
            if phase in ('carry', 'lower'):
                if np.any(np.linalg.norm(xyz-self._xyz, axis=1) > geometry_limit):
                    raise RuntimeError(f'Payload relative translation exceeds{geometry_limit*1000:g}mm during carry/lower')
                if np.any((self._rot.inv()*rot).magnitude() > np.deg2rad(15)):
                    raise RuntimeError('Payload relative rotation exceeds15degrees during carry/lower')
            if (self._box_rotation.inv()*box_rotation).magnitude() > np.deg2rad(15):
                raise RuntimeError('Payload placement rotation exceeds15degrees')
        sample_time = min(stamps)
        since, last = self._support_since, self._support_last
        if not supported: since = last = None
        elif phase != 'carry':
            if last is None or sample_time-last > .05+1e-9: since = sample_time
            last = sample_time
        dwell = 0. if since is None else sample_time-since
        if new > old:
            if phase == 'lower' and not navigation_ready:
                raise RuntimeError('Settled navigation required before lower transition')
            if phase == 'open' and (not supported or dwell < .5-1e-9):
                raise RuntimeError('Measured destination support dwell0.5s required before opening')
        if phase in ('open', 'done') and not supported:
            raise RuntimeError('Measured destination support/pose/speed lost after opening')
        if phase == 'lower' and self._lower_since is not None and native-self._lower_since > 25.+1e-9:
            raise RuntimeError('Transport lower timeout')
        if phase == 'done' and self._done_since is not None and native-self._done_since > 3.+1e-9:
            raise RuntimeError('Transport done verification timeout')
        if phase == 'open' and self._open_since is not None and native-self._open_since > 10.+1e-9:
            raise RuntimeError('Transport opening timeout awaiting done phase')
        if new > old:
            if phase == 'lower': self._lower_since = native
            if phase == 'open':
                axis = hands[0, :3]-hands[1, :3]
                if np.linalg.norm(axis) < .05: raise ValueError('Distinct measured hands required for opening axis')
                self._open_axis = axis/np.linalg.norm(axis)
                self._open_relative = hands[:, :3]-box[:3]
                self._open_since = native
            if phase == 'done': self._done_since = native
            self.phase = phase
            self.events.append(dict(event='transport_phase', phase=phase, native_time=native, support_dwell_s=dwell))
        outward = np.zeros(2)
        if self._open_since is not None:
            outward = ((hands[:, :3]-box[:3]-self._open_relative)@self._open_axis)*[1., -1.]
        self.completion_ready = bool(phase == 'done' and s.placement_done and supported
            and native-self._open_since >= 7.-1e-9 and np.all(outward >= .045) and np.all(forces <= .05))
        self.hand_contact_required = phase == 'carry' or (phase == 'lower' and not supported)
        self._support_since, self._support_last = since, last
        self._last_native, self._pose_stamp, self._force_stamp = native, *stamps
        self._box, self._hands, self._forces = box, hands, measured_force
        self.metrics = dict(phase=phase, native_time=native, pose_native_time=stamps[0], force_native_time=stamps[1],
            destination_support_z_N=support, support_qualified=supported, support_dwell_s=dwell,
            payload_xy_error_m=xy_error, payload_z_error_m=z_error, payload_speed_m_s=speed,
            hand_contact_required=self.hand_contact_required, outward_open_m=outward.tolist(),
            active_translation_limit_m=geometry_limit,
            open_elapsed_s=0. if self._open_since is None else native-self._open_since,
            force_per_hand_N=forces.tolist(), completion_ready=self.completion_ready)
