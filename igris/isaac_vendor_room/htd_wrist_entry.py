"""Pure, opt-in mixed-coordinate wrist preparation for a guarded SIM handoff.

The caller validates fresh native samples, SIM scope, zero WalkTwist, physical
upper targets, grasp/clearance, timing and sole publication. This helper does
not publish, request modes, reset an actor or control object, or read/write IO.
It infers a conservative SlowStart expiration condition from the audited binary;
it does not observe or modify the internal active flag.
"""
from copy import deepcopy

import numpy as np


WRIST_MOTORS = (20, 21, 27, 28)


class WristModeEntry:
    def __init__(self):
        self.state = 'unseeded'
        self.events = []
        self.contract = dict(schema='htd_wrist_mode_entry/v1', scope='SIM guarded loaded hold with zero vendor WalkTwist',
            wrist_motor_indices=list(WRIST_MOTORS), held_kinematic_modes=[1,1,1,0,0],
            restored_kinematic_modes=[1,1,1,1,1], original_wrist_kp=400., original_wrist_kd=2.,
            original_wrist_dq=0., original_wrist_host_tau=0.,
            priming_native_s=.05, priming_timeout_wall_s=2.,
            minimum_low_wall_s=3.1, maximum_upper_raw_error_rad=.18,
            continuous_error_wall_s=.1, maximum_observation_gap_wall_s=.1,
            low_timeout_wall_s=4.5, deactivation_observed=False,
            expiration_evidence='Code-derived conservative condition; internal active flag is not observed',
            caller_guards='Fresh native/SIM/zero WalkTwist/grasp/clearance/physical upper/timing/sole publisher remain required',
            gravity='Preserve zero host wrist feedforward and existing native upper GC; never add GC here')
        self._frozen = None
        self._seed_wall = self._seed_native = None
        self._last_wall = self._last_native = None
        self._low_wall = None
        self._qualified_since = self._qualified_observation_wall = None

    @staticmethod
    def _vector(value, name):
        result = np.asarray(value, dtype=float)
        if result.shape != (31,) or not np.isfinite(result).all():
            raise ValueError('Finite full31 '+name+' required')
        return result

    def _sample(self, snapshot):
        wall, native = float(snapshot.wall_time), float(snapshot.native_time)
        if not np.isfinite([wall,native]).all() or min(wall,native) < 0:
            raise ValueError('Finite nonnegative monotonic/native clocks required')
        if (self._last_wall is not None and
                (wall < self._last_wall-1e-12 or native < self._last_native-1e-12)):
            raise RuntimeError('Wrist mode-entry clock moved backward')
        if snapshot.mode not in (6,3):
            raise RuntimeError('Wrist mode entry requires WALK_LOW6 or LOW3 mode')
        if self.state in ('low_hold','released') and snapshot.mode != 3:
            raise RuntimeError('Wrist mode regressed after LOW')
        motor = self._vector(snapshot.output['q_motor'], 'q_motor')
        self._last_wall, self._last_native = wall, native
        return wall, native, motor

    def _advance_priming(self, wall, native):
        if self.state != 'priming':
            return
        if wall-self._seed_wall > 2.+1e-12:
            raise RuntimeError('Wrist motor-space priming exceeded2wall seconds')
        if native-self._seed_native >= .05-1e-12:
            self.state = 'ready'
            self.events.append(dict(event='primed_for_low',monotonic=wall,native=native,
                                    elapsed_native_s=native-self._seed_native))

    def ready_for_low(self, snapshot):
        """Latch native priming readiness; coordinator owns the subsequent LOW ACK timeout."""
        wall, native, _ = self._sample(snapshot)
        if snapshot.mode != 6:
            return False
        self._advance_priming(wall,native)
        return self.state == 'ready'

    def apply(self, command, snapshot):
        """Transform a fresh original PJS command; never feed the returned command back."""
        values = {key:self._vector(command[key],key) for key in ('q','dq','tau','kp','kd')}
        modes = np.asarray(command['kinematic_modes'])
        if modes.shape != (5,) or not np.all(modes == 1):
            raise ValueError('Original live all-PJS command required; transformed feedback is forbidden')
        wrist = list(WRIST_MOTORS)
        if any(not np.all(values[key][wrist] == expected)
               for key,expected in [('kp',400.),('kd',2.),('dq',0.),('tau',0.)]):
            raise ValueError('Reviewed wrist400/2 gains and zero dq/host tau required')
        wall, native, motor = self._sample(snapshot)
        result = deepcopy(command)
        if self.state == 'released':
            return result
        if self.state == 'unseeded':
            if snapshot.mode != 6:
                raise RuntimeError('Wrist hold must be primed under WALK_LOW before LOW')
            self._frozen = motor[wrist].copy()
            self._seed_wall, self._seed_native = wall, native
            self.state = 'priming'
            self.events.append(dict(event='seeded_motor_wrist_hold',monotonic=wall,native=native,
                                    q_motor=self._frozen.tolist(),sdk_indices=list(WRIST_MOTORS)))
        if snapshot.mode == 6:
            self._advance_priming(wall,native)
        elif self.state != 'low_hold':
            if self.state != 'ready':
                raise RuntimeError('Wrist hold was not primed before LOW')
            self.state = 'low_hold'
            self._low_wall = wall
            self.events.append(dict(event='first_low_observed',monotonic=wall,native=native))
        result['q'] = values['q'].copy()
        result['q'][wrist] = self._frozen
        result['kinematic_modes'] = [1,1,1,0,0]
        if self.state == 'low_hold':
            if wall-self._low_wall > 4.5+1e-12:
                raise RuntimeError('Wrist motor-space LOW hold exceeded4.5wall seconds')
            error = float(np.max(np.abs(result['q'][15:29]-motor[15:29])))
            continuous = (self._qualified_observation_wall is not None and
                          wall-self._qualified_observation_wall <= .1+1e-12)
            if error <= .18:
                if self._qualified_since is None or not continuous:
                    self._qualified_since = wall
            else:
                self._qualified_since = None
            self._qualified_observation_wall = wall
            if (wall-self._low_wall >= 3.1-1e-12 and self._qualified_since is not None and
                    wall-self._qualified_since >= .1-1e-12):
                self.state = 'released'
                self.events.append(dict(event='restored_live_pjs',monotonic=wall,native=native,
                    elapsed_low_wall_s=wall-self._low_wall,
                    continuous_error_wall_s=wall-self._qualified_since,
                    maximum_upper_raw_error_rad=error,deactivation_observed=False,
                    evidence='Conservative audited-code expiration condition; observer must verify continuity'))
                return deepcopy(command)
        return result
