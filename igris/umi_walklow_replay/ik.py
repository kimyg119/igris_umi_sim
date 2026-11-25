"""Seventeen-joint upper-body IK: waist 3 + both arms, pelvis root.

Each control tick takes the previous command and two hand targets and returns
the next command.  Hand poses are the primary task; posture preferences act in
its null space.  Joint limits, the vendor coupled wrist/waist safe sets and a
per-tick velocity bound are hard: a joint that would cross a bound is fixed
there and the remaining joints re-solve.  Only numpy is used.
"""
import numpy as np

from .kinematics import ACTIVE, JOINTS, coupled_halfspaces, rotation_vector

# Null-space posture, in ACTIVE order: waist yaw/roll/pitch, left arm 7, right arm 7.
# Elbows bent and slightly away from the trunk; everything else neutral.
POSTURE = np.array([0., 0., 0.,
                    -.3, .35, 0., -1.0, 0., 0., 0.,
                    -.3, -.35, 0., -1.0, 0., 0., 0.])
# Null-space return rate toward the posture [1/s].  Waist roll and pitch return
# fastest: leaning the trunk moves the centre of mass, and the vendor walking
# controller answers by shifting the pelvis.
POSTURE_RATE = np.array([4., 8., 8.] + [1.5, 2., 1.5, 1.5, 1., 1., 1.] * 2)
# Cost of moving each joint.  The arms move first; waist roll and pitch are
# used only when the arms cannot reach (offline study on episode 0: peak
# forward lean 0.22 rad instead of 0.64 rad with equal costs).
JOINT_COST = np.array([2., 25., 25.] + [1.] * 14)


# Elbow prior (HL-IK): rate at which each elbow is drawn to its predicted place
# inside the null space of the hands [1/s].  The prediction is 33 ms ahead.
ELBOW_RATE = 12.
ELBOW_STEP_LIMIT = .08          # largest elbow error acted on per tick [m]
ELBOW_DAMPING = 1e-3            # [m2/rad2] keeps the swivel calm where it barely moves the elbow (arm nearly straight)
ELBOW_SPEED_SHARE = .5          # share of a joint's per-tick range the swivel may use; the rest stays for the hands
# Where the swivel does not follow the prior.  Both were found in simulation (2026-11-23):
# - Near a joint limit.  Following the predicted elbow took three left-arm joints to their limits;
#   the hands then needed the trunk to lean, the walking controller moved the pelvis 12 cm and the
#   box being set down fell.
# - Near a box (``keep_out`` of ``IK.step``).  An elbow held out to the side sweeps over the waiting
#   stack: the right upper arm came within 4-8 cm of the next box (centre line to surface; the link
#   is about 4.5 cm thick) and knocked it over once.
# Outside these zones a step that would enter one is dropped; inside, the arm swivels out of it.
ELBOW_LIMIT_MARGIN = .10        # [rad] distance to an arm joint limit
ELBOW_KEEP_OUT = .08            # [m] arm centre line to box surface
ELBOW_KEEP_OUT_BAND = .03       # [m] depth inside the box margin that counts like a joint at its limit
UPPER_ARM_SAMPLES = np.linspace(.4, 1., 4)      # shoulder (0) to elbow (1)
FOREARM_SAMPLES = np.array([.2, .4])            # elbow (0) to hand (1); nearer the hand the arm is meant to be close
# With an elbow target the arm has no freedom left; the joint posture only breaks ties.
ARM_POSTURE_SCALE_WITH_ELBOWS = .15


# Free toe-in (``toe_range`` of ``IK``): each hand may turn about its x axis through a pivot on the
# palm, between the two given angles, so that the fingers converge.  Inside the range the angle is
# only drawn weakly to its preferred value; the freedom is spent on keeping arm joints off their limits.
TOE_PREFERENCE_WEIGHT = .05     # share of the orientation weight that holds the preferred angle
TOE_LIMIT_MARGIN = .30          # [rad] distance to a limit below which an arm joint is pushed away
TOE_LIMIT_RATE = 2.             # [rad/s] push at the limit itself, as a null-space preference


def _skew(v):
    return np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])


def project_polygon(point, normal, offset):
    """Closest point of a convex polygon (normal @ x <= offset) to a 2-D point."""
    point = np.asarray(point, dtype=float)
    if np.all(normal @ point <= offset + 1e-12):
        return point
    best = None
    count = len(offset)
    for i in range(count):
        # Vertices are the intersections of consecutive edges.
        a = np.linalg.solve(np.vstack((normal[i - 1], normal[i])), np.array([offset[i - 1], offset[i]]))
        b = np.linalg.solve(np.vstack((normal[i], normal[(i + 1) % count])),
                            np.array([offset[i], offset[(i + 1) % count]]))
        edge = b - a
        u = np.clip(np.dot(point - a, edge) / np.dot(edge, edge), 0., 1.)
        candidate = a + u * edge
        if best is None or np.linalg.norm(candidate - point) < np.linalg.norm(best - point):
            best = candidate
    return best


class IK:
    def __init__(self, model, *, dt=.01, velocity_limit=2.0, waist_velocity_limit=1.2, position_weight=1.,
                 rotation_weight=.12, damping=.02, limit_margin=.035, iterations=4,
                 max_position_step=.05, max_rotation_step=.35, waist_roll_limit=None, waist_pitch_limits=None,
                 pin_waist_yaw=False, toe_range=None, toe_preference=None, toe_pivot=None):
        self.model = model
        # ``toe_range`` (lowest, highest toe-in [rad]) frees the turn of each hand about its own x axis
        # through ``toe_pivot`` (2 x 3, EE frame); the pivot is then what the position target holds.
        self.toe_range = None if toe_range is None else (float(min(toe_range)), float(max(toe_range)))
        if self.toe_range is not None:
            if toe_pivot is None:
                raise ValueError('toe_range needs toe_pivot, the palm point each hand turns about')
            self.toe_pivot = np.asarray(toe_pivot, dtype=float).reshape(2, 3)
            self.toe_preference = float(np.mean(self.toe_range) if toe_preference is None else toe_preference)
            self.position_weight, self.rotation_weight = float(position_weight), float(rotation_weight)
        # The waist yaw follows its reference directly (within its speed and range) and the hands are
        # solved with the other sixteen joints.  As a null-space preference the trunk trails the hands
        # by about a quarter second when they swing across, and the trailing arm folds over the chest.
        self.pin_waist_yaw = bool(pin_waist_yaw)
        self.dt = float(dt)
        self.velocity = np.r_[[waist_velocity_limit] * 3, [velocity_limit] * 14].astype(float)
        self.weight = np.r_[[position_weight] * 3, [rotation_weight] * 3] * 1.
        self.weight = np.r_[self.weight, self.weight]
        self.damping2 = float(damping) ** 2
        self.iterations = int(iterations)
        self.max_position_step = float(max_position_step)
        self.max_rotation_step = float(max_rotation_step)
        self.lower = model.lower[ACTIVE] + limit_margin
        self.upper = model.upper[ACTIVE] - limit_margin
        # Optional tighter trunk-lean range (ACTIVE order: waist yaw, roll, pitch).
        if waist_roll_limit is not None:
            self.lower[1] = max(self.lower[1], -abs(waist_roll_limit)); self.upper[1] = min(self.upper[1], abs(waist_roll_limit))
        if waist_pitch_limits is not None:
            self.lower[2] = max(self.lower[2], min(waist_pitch_limits)); self.upper[2] = min(self.upper[2], max(waist_pitch_limits))
        column = {joint: i for i, joint in enumerate(ACTIVE)}
        self.coupled = [([column[j] for j in joints], normal, offset - 1e-3)
                        for joints, normal, offset in coupled_halfspaces()]
        self.inverse_cost = 1. / JOINT_COST

    def _solve(self, error, jacobian, desire, lower, upper, pinned):
        """One bounded resolved-rate step with strict task priority."""
        q = lower  # only its length is used below
        step = np.zeros(len(q)); free = np.ones(len(q), dtype=bool)
        for index, value in pinned.items():
            step[index] = value; free[index] = False
        identity = np.eye(len(error))
        for _ in range(len(q) + 1):
            if not free.any():
                break
            j = jacobian[:, free]; inverse = self.inverse_cost[free]
            residual = error - jacobian[:, ~free] @ step[~free]
            gram = (j * inverse) @ j.T + self.damping2 * identity
            task = inverse * (j.T @ np.linalg.solve(gram, residual))
            wish = desire[free]
            posture = wish - inverse * (j.T @ np.linalg.solve(gram, j @ wish))
            candidate = task + posture
            excess = np.maximum(lower[free] - candidate, candidate - upper[free])
            worst = int(np.argmax(excess))
            if excess[worst] <= 1e-12:
                step[free] = candidate
                break
            index = np.flatnonzero(free)[worst]
            step[index] = np.clip(candidate[worst], lower[index], upper[index]); free[index] = False
        return step

    def _elbow_wish(self, q31, elbow_targets, hand_jacobian, keep_out=None):
        """Joint step that swivels each elbow toward its target and leaves its hand where it is.

        The step is taken from the null space of each arm's own hand Jacobian.  A
        general joint step would have to be cancelled at the hand by other joints,
        and the cheapest of those is the waist yaw: the trunk then turns to let an
        elbow move (24 deg in the ready pose before this was changed).  The step is
        shortened so that no arm joint leaves this tick's bounds; an arm whose
        swivel is blocked by a limit stays as it is.
        """
        _, elbows, hands, elbow_jacobian = self.model.arm_frames(q31)
        start = q31[ACTIVE]
        lower = np.minimum(np.maximum(self.lower, start - self.velocity * self.dt), start) - start
        upper = np.maximum(np.minimum(self.upper, start + self.velocity * self.dt), start) - start
        wish = np.zeros(len(ACTIVE))
        for side in range(2):
            columns = slice(3 + 7 * side, 10 + 7 * side)
            hand = hand_jacobian[6 * side:6 * side + 6, columns]
            null = np.eye(7) - hand.T @ np.linalg.solve(hand @ hand.T + 1e-6 * np.eye(6), hand)
            error = elbow_targets[side] - elbows[side][:3, 3]
            norm = np.linalg.norm(error)
            if norm > ELBOW_STEP_LIMIT:
                error = error * ELBOW_STEP_LIMIT / norm
            swivel = elbow_jacobian[3 * side:3 * side + 3, columns] @ null     # elbow motion per null-space joint motion
            step = min(1., ELBOW_RATE * self.dt) * null @ (
                swivel.T @ np.linalg.solve(swivel @ swivel.T + ELBOW_DAMPING * np.eye(3), error))
            scale = 1.
            for index in range(7):
                if step[index] > 1e-12:
                    scale = min(scale, ELBOW_SPEED_SHARE * upper[columns][index] / step[index])
                elif step[index] < -1e-12:
                    scale = min(scale, ELBOW_SPEED_SHARE * lower[columns][index] / step[index])
            wish[columns] = self._guarded_swivel(q31, side, scale * step, hand, hands[side][:3, 3],
                                                 lower[columns], upper[columns], keep_out)
        return wish

    def _arm_gap(self, q31, side, hand, keep_out):
        """Smallest distance from the elbow side of one arm's centre lines to the keep-out boxes [m]."""
        shoulder, elbow = self.model.upper_arm(q31, side)
        points = np.vstack((shoulder + UPPER_ARM_SAMPLES[:, None] * (elbow - shoulder),
                            elbow + FOREARM_SAMPLES[:, None] * (hand - elbow)))
        return min(np.linalg.norm(np.maximum(np.abs((points - centre) @ rotation) - half, 0.), axis=1).min()
                   for centre, rotation, half in keep_out)

    def _swivel_cost(self, q31, side, hand, keep_out):
        """Zero while one arm is clear; grows as its joints near their limits and its elbow side nears a box."""
        columns = slice(3 + 7 * side, 10 + 7 * side)
        q = q31[ACTIVE][columns]
        room = np.minimum(q - self.lower[columns], self.upper[columns] - q)
        cost = float(np.sum(np.maximum(0., 1. - room / ELBOW_LIMIT_MARGIN) ** 2))
        if keep_out:
            cost += max(0., (ELBOW_KEEP_OUT - self._arm_gap(q31, side, hand, keep_out)) / ELBOW_KEEP_OUT_BAND) ** 2
        return cost

    def _guarded_swivel(self, q31, side, step, hand_jacobian, hand, lower, upper, keep_out):
        """Swivel step of one arm: the prior's ``step`` where the arm is clear, a way out where it is not.

        The hand does not move during a swivel, so its current place is used for every trial.
        """
        joints = ACTIVE[3 + 7 * side:10 + 7 * side]

        def cost(delta):
            trial = q31.copy(); trial[joints] += delta
            return self._swivel_cost(trial, side, hand, keep_out)

        now = cost(np.zeros(7))
        if now <= 0.:
            return step if not step.any() or cost(step) <= 0. else np.zeros(7)
        direction = np.linalg.svd(hand_jacobian)[2][-1]              # the arm's swivel direction in joint space
        urgency = min(1., np.sqrt(now))
        candidates = [step] if step.any() else []
        for move in (direction, -direction):
            room = min((upper[k] if move[k] > 0 else lower[k]) / move[k] for k in range(7) if abs(move[k]) > 1e-9)
            candidates.append(ELBOW_SPEED_SHARE * urgency * room * move)
        best, best_cost = np.zeros(7), now
        for candidate in candidates:
            value = cost(candidate)
            if value < best_cost - 1e-9:
                best, best_cost = candidate, value
        return best

    def toe_angles(self, q31, left, right):
        """Toe-in of both hands [rad] relative to the target orientations: + brings the fingertips together."""
        current = self.model.fk(q31)
        return np.array([sign * rotation_vector(target[:3, :3].T @ pose[:3, :3])[0]
                         for sign, pose, target in zip((-1., 1.), current, (left, right))])

    def _toe_task(self, current, targets, jacobian, hold=0.):
        """Weighted hand task with the toe-in turn left free: pivot position, the two other turns, and a
        weak (inside the range) or full (outside it) pull on the toe angle.  Rotations are in the target frame.
        ``hold`` (0..1) narrows the range onto the preferred angle: 1 while a rigid pad lies on a box."""
        errors, rows, residual = [], [], []
        hold = float(np.clip(hold, 0., 1.))
        lowest = self.toe_range[0] + hold * (self.toe_preference - self.toe_range[0])
        highest = self.toe_range[1] - hold * (self.toe_range[1] - self.toe_preference)
        for side, (pose, target) in enumerate(zip(current, targets)):
            linear, angular = jacobian[6 * side:6 * side + 3], jacobian[6 * side + 3:6 * side + 6]
            arm = pose[:3, :3] @ self.toe_pivot[side]
            position = target[:3, 3] + target[:3, :3] @ self.toe_pivot[side] - pose[:3, 3] - arm
            turn = rotation_vector(target[:3, :3].T @ pose[:3, :3])        # current seen from the target frame
            sign = -1. if side == 0 else 1.
            toe = sign * turn[0]
            if toe < lowest:
                wanted, pull = lowest, 1.
            elif toe > highest:
                wanted, pull = highest, 1.
            else:
                wanted, pull = self.toe_preference, TOE_PREFERENCE_WEIGHT
            rotation = np.array([sign * wanted - turn[0], -turn[1], -turn[2]])
            for vector, cap in ((position, self.max_position_step), (rotation, self.max_rotation_step)):
                norm = np.linalg.norm(vector)
                if norm > cap:
                    vector *= cap / norm
            weights = np.r_[[self.position_weight] * 3, self.rotation_weight * pull, [self.rotation_weight] * 2]
            errors.append(weights * np.r_[position, rotation])
            rows.append(weights[:, None] * np.vstack((linear - _skew(arm) @ angular, target[:3, :3].T @ angular)))
            residual.append((np.linalg.norm(target[:3, 3] + target[:3, :3] @ self.toe_pivot[side] - pose[:3, 3] - arm),
                             np.linalg.norm(turn[1:])))
        return np.concatenate(errors), np.vstack(rows), np.array(residual)

    def step(self, q31, left, right, *, waist_yaw_reference=None, elbow_targets=None, keep_out=None, toe_hold=0.,
             posture=None, orientation=1.):
        """Next 31-joint command and diagnostics for 4x4 pelvis-frame targets.

        ``elbow_targets`` (2 x 3, pelvis frame) are followed only as far as the
        hand poses and the joint limits leave room.  ``keep_out`` lists boxes as
        (centre 3, rotation 3x3, half extents 3) in the pelvis frame that the
        elbow side of the arms stays clear of while following them.  ``toe_hold``
        (0..1) fixes a free toe-in at its preferred angle, for the time a box is held.
        """
        q31 = np.array(q31, dtype=float, copy=True)
        start = q31[ACTIVE].copy()
        # The measured start may sit slightly outside the margin; never demand a jump back.
        lower = np.minimum(np.maximum(self.lower, start - self.velocity * self.dt), start)
        upper = np.maximum(np.minimum(self.upper, start + self.velocity * self.dt), start)
        reference = POSTURE.copy()
        if posture is not None:                              # preferred arm joints for this tick (14, SDK order)
            reference[3:] = np.clip(posture, self.lower[3:], self.upper[3:])
        if waist_yaw_reference is not None:
            reference[0] = np.clip(waist_yaw_reference, self.lower[0], self.upper[0])
        q = start.copy()
        saturated = False
        for iteration in range(self.iterations):
            q31[ACTIVE] = q
            # Posture is a rate: apply it once per tick, then only refine the hands.
            current_left, current_right, jacobian = self.model.fk_jacobian(q31)
            desire = np.zeros(len(q))
            if iteration == 0:
                desire = POSTURE_RATE * self.dt * (reference - q)
                if elbow_targets is not None:
                    desire[3:] *= ARM_POSTURE_SCALE_WITH_ELBOWS
                    desire += self._elbow_wish(q31, elbow_targets, jacobian, keep_out)
                if self.toe_range is not None:
                    push = (np.maximum(0., 1. - (q - self.lower) / TOE_LIMIT_MARGIN)
                            - np.maximum(0., 1. - (self.upper - q) / TOE_LIMIT_MARGIN))
                    push[:3] = 0.                                # arms only; the waist has its own preference
                    desire += TOE_LIMIT_RATE * self.dt * push
            if self.toe_range is not None:
                weighted_error, weighted_jacobian, _ = self._toe_task((current_left, current_right), (left, right), jacobian,
                                                                     toe_hold)
            else:
                error = np.r_[left[:3, 3] - current_left[:3, 3], rotation_vector(left[:3, :3] @ current_left[:3, :3].T),
                              right[:3, 3] - current_right[:3, 3], rotation_vector(right[:3, :3] @ current_right[:3, :3].T)]
                for rows, cap in ((slice(0, 3), self.max_position_step), (slice(3, 6), self.max_rotation_step),
                                  (slice(6, 9), self.max_position_step), (slice(9, 12), self.max_rotation_step)):
                    norm = np.linalg.norm(error[rows])
                    if norm > cap:
                        error[rows] *= cap / norm
                weighted_error = self.weight * error
                weighted_jacobian = self.weight[:, None] * jacobian
            pinned = {}
            if self.pin_waist_yaw and waist_yaw_reference is not None:
                pinned[0] = float(np.clip(reference[0], lower[0], upper[0]) - q[0])
            for _attempt in range(len(self.coupled) + 1):
                delta = self._solve(weighted_error, weighted_jacobian, desire, lower - q, upper - q, pinned)
                violated = False
                for joints, normal, offset in self.coupled:
                    point = q[joints] + delta[joints]
                    if np.any(normal @ point > offset + 1e-9):
                        inside = project_polygon(point, normal, offset)
                        inside = np.clip(inside, lower[joints], upper[joints])
                        for joint, value in zip(joints, inside):
                            pinned[joint] = value - q[joint]
                        violated = True
                if not violated:
                    break
            q = q + delta
            saturated = saturated or bool(np.any(q <= lower + 1e-9) or np.any(q >= upper - 1e-9))
            if np.max(abs(delta)) < 1e-6:
                break
        q31[ACTIVE] = q
        final_left, final_right = self.model.fk(q31)
        elbow_error = np.zeros(2)
        if elbow_targets is not None:
            elbow_error = np.array([np.linalg.norm(elbow_targets[i] - self.model.upper_arm(q31, i)[1]) for i in range(2)])
        info = dict(elbow_error=elbow_error,
            position_error=np.array([np.linalg.norm(left[:3, 3] - final_left[:3, 3]),
                                     np.linalg.norm(right[:3, 3] - final_right[:3, 3])]),
            rotation_error=np.array([np.linalg.norm(rotation_vector(left[:3, :3] @ final_left[:3, :3].T)),
                                     np.linalg.norm(rotation_vector(right[:3, :3] @ final_right[:3, :3].T))]),
            saturated=saturated)
        if self.toe_range is not None:
            # With the toe-in turn free the residuals are those of the task: pivot position and the two held turns.
            residual = self._toe_task((final_left, final_right), (left, right), np.zeros((12, len(ACTIVE))))[2]
            info.update(position_error=residual[:, 0], rotation_error=residual[:, 1],
                        toe=self.toe_angles(q31, left, right))
        return q31, info

    def converge(self, q31, left, right, *, waist_yaw_reference=None, elbow_targets=None, keep_out=None, posture=None,
                 ticks=400):
        """Repeat ``step`` until the pose stops improving (offline use)."""
        info = None
        for _ in range(ticks):
            previous = q31
            q31, info = self.step(q31, left, right, waist_yaw_reference=waist_yaw_reference, elbow_targets=elbow_targets,
                                  keep_out=keep_out, posture=posture)
            if np.max(abs(q31 - previous)) < 1e-7:
                break
        return q31, info


def joint_names():
    return [JOINTS[i] for i in ACTIVE]
