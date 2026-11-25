"""Upper-body IK through mink (QP differential IK on a MuJoCo model), with the interface of ``ik.IK``.

The numpy solver in ``ik.py`` enforces each bound by its own rule (freeze and re-solve, polygon
projection).  Here joint ranges, joint speeds, the vendor coupled limits and the self-collision
gaps are inequalities of one QP per tick, solved by an established library.  It needs mink,
qpsolvers with daqp and MuJoCo >= 3.10; ``vendor_py`` (tools/install_packages.sh) holds copies that also load in
the simulator container.  Only kinematics are taken from the vendor model: no geoms, legs welded.
"""
import os
import xml.etree.ElementTree as ET

import mink
import mujoco
import numpy as np

from .ik import ELBOW_DAMPING, ELBOW_STEP_LIMIT, POSTURE
from .kinematics import ACTIVE, JOINTS, VENDOR_MJCF, UpperBody, coupled_halfspaces, rotation_vector

# Posture preference: cost per joint (against 1 per metre of hand error) and how much of the
# distance is asked for per tick.  Waist roll and pitch cost most: leaning moves the centre of mass.
POSTURE_COST = np.array([.03, .30, .30] + [.01] * 14)
POSTURE_GAIN = .04
ROTATION_COST = .12               # per radian of hand orientation error, as in ik.py
QP_DAMPING = 1e-5
ARM_POSTURE_COST_WITH_LOOKUP = .03    # arm posture cost when the preferred posture is looked up per tick (teleop tendency)
FREE_ORIENTATION = .04                # share of the orientation cost left where the hand orientation does not matter
# Elbow prior (FiSTA).  The predicted elbow places are turned into a joint step of each arm in the null space of its
# own hand (the swivel, ``_swivel_step``), shortened so that every arm joint can take it within this tick, and the
# arm posture term of the QP asks for exactly that step - the same door the teleop posture lookup comes in by.  The
# arm can then give all of it without the hands moving, and the waist has nothing to add.
# What did not work, each tried here: a position task on the elbow (pelvis frame or relative to the torso), only
# the swivel part of the elbow error as that task's target, and a swivel step that was not kept inside the joint
# bounds.  In every one the QP turned the waist to bring an elbow nearer or to make up for a joint at its bound:
# 7-20 deg for a point out of reach, and on held-out teleop sessions the elbows ended further from the operator's
# the stronger the task was (1.9 -> 3.0 cm).  ik.py documents the same trap for its own elbow step.
# Its cost is that of the looked-up posture.  Stiffer follows the prior a little closer on teleop hand poses (elbows 1.8 -> 1.7 cm
# from the operator's at .10) but takes it from the hands and the trunk on UMI ones: hand error p99 1.3 -> 2.7 mm, trunk lean
# in use 3 -> 10 % of the time (37 % at .20).
ARM_POSTURE_COST_WITH_ELBOWS = .03    # cost of the arm joints not taking the swivel step
# Pace of the swivel.  The elbow error closes at the rate the looked-up posture does (POSTURE_GAIN per tick, 4 /s at
# 100 Hz), and no arm joint is asked for more than a quarter of its speed limit by it (0.5 rad/s).  At 12 /s and half
# the limit, as in ik.py, the swivel alone reached the 1 rad/s of the playback speed limit, and the playback of
# episode 7 took 93-125 s in simulation instead of 81-84 s.
ELBOW_RATE = float(os.environ.get('UMI_ELBOW_RATE', 4.))
ELBOW_SPEED_SHARE = float(os.environ.get('UMI_ELBOW_SPEED_SHARE', .25))
# Trust in a prediction.  The swivel moves the elbow along one line; what is left of the elbow error across that line
# is how far the predicted place is from anything the arm can do with its hand held.  On teleop hand poses that is
# 0.5 cm, on UMI ones 1.3 cm.  In simulation, carrying the third box high, it was 6-18 cm - hand poses the prior had
# not seen - and following such a prediction swivelled the right arm to 114 deg and onto its joint limits (3 of 6
# and 4 of 6 runs stacked, against 6 of 6 with the posture lookup).  The step is scaled down from full at
# ``ELBOW_TRUST_M[0]`` across the line to nothing at ``ELBOW_TRUST_M[1]``, and the share taken off goes to the
# posture reference the arm would have had without the prior (in the replay: the looked-up teleop posture), so
# that an arm whose prediction is not believed is not left without any posture to lean on.
# A prediction far along the line is no more believable than one far across it (the elbow moves on a circle, not
# on the line): the same fading applies to the whole distance, between ``ELBOW_TRUST_REACH_M``.  A rollout that is
# working keeps its elbows within 1-2 cm of the prediction.
ELBOW_TRUST_M = (float(os.environ.get('UMI_ELBOW_TRUST_FULL', .02)), float(os.environ.get('UMI_ELBOW_TRUST_NONE', .06)))
ELBOW_TRUST_REACH_M = (.04, .12)
COLLISION_GAP_M = .002            # smallest sphere-model gap allowed (the spheres carry 2.5 mm each already)
COLLISION_GAIN = .85              # share of the remaining gap a tick may use up
COLLISION_ROWS = 16               # sphere pairs bounded per link pair; with one, its neighbours close in instead
WAIST_HELD_RAD_S = 1e-6           # speed left to the waist joints in the QP when their angles are given from outside


def upper_body_xml(path=VENDOR_MJCF):
    """Kinematics-only MJCF of the vendor robot: pelvis fixed, only the waist and arm joints kept."""
    root = ET.parse(path).getroot()
    for tag in ('asset', 'contact', 'equality', 'actuator', 'sensor', 'keyframe', 'visual', 'default'):
        for element in root.findall(tag):
            root.remove(element)
    compiler = root.find('compiler')
    for key in ('meshdir', 'texturedir'):
        compiler.attrib.pop(key, None)
    keep = {JOINTS[index] for index in ACTIVE}
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag in ('geom', 'site', 'camera', 'light', 'freejoint') or (child.tag == 'joint' and child.get('name') not in keep):
                parent.remove(child)
            elif child.tag in ('joint', 'body', 'inertial'):
                child.attrib.pop('class', None); child.attrib.pop('childclass', None)
    return ET.tostring(root, encoding='unicode')


class CoupledLimit(mink.Limit):
    """Vendor convex safe sets of joint pairs (wrist roll/pitch, waist roll/pitch) as QP inequalities."""

    def __init__(self, columns, margin=1e-3):
        self.sets = [([columns[joint] for joint in joints], normal, offset - margin) for joints, normal, offset in coupled_halfspaces()]
        self.size = max(max(joints) for joints, _, _ in self.sets) + 1

    def compute_qp_inequalities(self, configuration, dt):
        rows = []; bounds = []
        for joints, normal, offset in self.sets:
            block = np.zeros((len(offset), configuration.nv)); block[:, joints] = normal
            # Never below zero: a set already left may not be left further, and standing still stays feasible.
            rows.append(block); bounds.append(np.maximum(offset - normal @ configuration.q[joints], 0.))
        return mink.limits.limit.Constraint(G=np.vstack(rows), h=np.concatenate(bounds))


class SelfCollisionLimit(mink.Limit):
    """Keeps every near pair of collision spheres (``collision.SelfCollision``) at least ``minimum`` apart."""

    def __init__(self, model, checker, to_sdk, minimum=COLLISION_GAP_M, gain=COLLISION_GAIN):
        self.model, self.checker, self.to_sdk, self.minimum, self.gain = model, checker, to_sdk, float(minimum), float(gain)
        self.bodies = {link: model.body(body).id for link, body in zip(checker.links, checker.bodies)}
        self.last_gap = checker.cutoff

    def compute_qp_inequalities(self, configuration, dt):
        pairs = self.checker.near_pairs(self.to_sdk(configuration.q), per_pair=COLLISION_ROWS)
        self.last_gap = min((entry[0] for entry in pairs), default=self.checker.cutoff)
        if not pairs:
            return mink.limits.limit.Constraint()
        rows = []; bounds = []
        first = np.empty((3, self.model.nv)); second = np.empty((3, self.model.nv))
        for gap, link_a, point_a, link_b, point_b in pairs:
            direction = point_a - point_b
            direction /= max(np.linalg.norm(direction), 1e-9)
            mujoco.mj_jac(self.model, configuration.data, first, None, point_a, self.bodies[link_a])
            mujoco.mj_jac(self.model, configuration.data, second, None, point_b, self.bodies[link_b])
            rows.append(-direction @ (first - second))            # minus the change of the gap per joint step
            bounds.append(max(self.gain * (gap - self.minimum), 0.))      # closer than allowed: may not close further
        return mink.limits.limit.Constraint(G=np.array(rows), h=np.array(bounds))


class MinkIK:
    """Seventeen-joint upper-body IK (waist 3 + arms 14): hands first, posture second, bounds as constraints."""

    def __init__(self, model=None, *, dt=.01, velocity_limit=2.0, waist_velocity_limit=1.2, limit_margin=.035,
                 waist_roll_limit=None, waist_pitch_limits=None, shoulder_roll_inward=None, self_collision=None,
                 solver='daqp', arm_posture_cost=None):
        self.kinematics = model or UpperBody.from_mjcf()
        self.model = mujoco.MjModel.from_xml_string(upper_body_xml())
        self.dt, self.solver = float(dt), solver
        names = [JOINTS[index] for index in ACTIVE]
        self.columns = np.array([self.model.joint(name).qposadr[0] for name in names])     # ACTIVE order -> model order
        if self.model.nq != len(ACTIVE) or sorted(self.columns) != list(range(len(ACTIVE))):
            raise ValueError('The reduced model must have exactly the seventeen commanded joints')
        lower = self.kinematics.lower[ACTIVE] + limit_margin; upper = self.kinematics.upper[ACTIVE] - limit_margin
        if waist_roll_limit is not None:
            lower[1] = max(lower[1], -abs(waist_roll_limit)); upper[1] = min(upper[1], abs(waist_roll_limit))
        if waist_pitch_limits is not None:
            lower[2] = max(lower[2], min(waist_pitch_limits)); upper[2] = min(upper[2], max(waist_pitch_limits))
        if shoulder_roll_inward is not None:
            # Shoulder roll: + is outward for the left arm (ACTIVE 4), - for the right (ACTIVE 11).
            lower[4] = max(lower[4], -abs(shoulder_roll_inward)); upper[11] = min(upper[11], abs(shoulder_roll_inward))
        self.lower, self.upper = lower, upper
        for k, name in enumerate(names):
            self.model.jnt_range[self.model.joint(name).id] = (lower[k], upper[k])
        self.configuration = mink.Configuration(self.model)
        self.hands = [mink.FrameTask(frame_name=name, frame_type='body', position_cost=1., orientation_cost=ROTATION_COST)
                      for name in ('end_effector_left', 'end_effector_right')]
        costs = POSTURE_COST.copy()
        if arm_posture_cost is not None:
            costs[3:] = arm_posture_cost
        cost = np.empty(len(ACTIVE)); cost[self.columns] = costs
        self.posture = mink.PostureTask(self.model, cost=cost, gain=POSTURE_GAIN)
        # With elbow targets: the waist keeps its posture term, the arms get one of their own that asks for a whole
        # step per tick (gain 1) instead of a share of the way to a far reference.
        waist = np.zeros(len(ACTIVE)); waist[self.columns[:3]] = costs[:3]
        arms = np.zeros(len(ACTIVE)); arms[self.columns[3:]] = ARM_POSTURE_COST_WITH_ELBOWS
        self.waist_posture = mink.PostureTask(self.model, cost=waist, gain=POSTURE_GAIN)
        self.arm_step = mink.PostureTask(self.model, cost=arms, gain=1.)
        self.velocity = np.r_[[waist_velocity_limit] * 3, [velocity_limit] * 14].astype(float)
        self.legs = np.zeros(len(JOINTS))                        # last full configuration, for the leg links of the collision model
        self.believed = np.zeros(2)
        self.collision = SelfCollisionLimit(self.model, self_collision, self._sdk) if self_collision is not None else None
        self.limits = [mink.ConfigurationLimit(self.model), mink.VelocityLimit(self.model, dict(zip(names, self.velocity))),
                       CoupledLimit({ACTIVE[k]: int(self.columns[k]) for k in range(len(ACTIVE))})]
        if self.collision is not None:
            self.limits.append(self.collision)
        # The same bounds with the waist held where it is: for a waist that is given, not solved for (``waist_pinned``).
        held = dict(zip(names, np.r_[[WAIST_HELD_RAD_S] * 3, self.velocity[3:]]))
        self.limits_waist_held = [self.limits[0], mink.VelocityLimit(self.model, held), *self.limits[2:]]

    def _sdk(self, q):
        out = self.legs.copy(); out[ACTIVE] = q[self.columns]
        return out

    def _swivel_step(self, q31, elbow_targets):
        """Joint step of the 14 arm joints that swivels each elbow toward its target (2 x 3, pelvis frame) and
        leaves its hand where it is: ``ELBOW_RATE`` of the error per second, at most ``ELBOW_STEP_LIMIT`` of it,
        and shortened so that no arm joint uses more than ``ELBOW_SPEED_SHARE`` of its room in this tick.
        Also returns how far each prediction is believed (2 values, 0..1; see ``ELBOW_TRUST_M``)."""
        _, _, hand_jacobian = self.kinematics.fk_jacobian(q31)
        _, elbows, _, elbow_jacobian = self.kinematics.arm_frames(q31)
        start = np.clip(q31[ACTIVE], self.lower, self.upper)
        down = np.minimum(np.maximum(self.lower, start - self.velocity * self.dt), start) - start
        up = np.maximum(np.minimum(self.upper, start + self.velocity * self.dt), start) - start
        step = np.zeros(14); believed = np.zeros(2)
        for side in range(2):
            columns = slice(3 + 7 * side, 10 + 7 * side)
            hand = hand_jacobian[6 * side:6 * side + 6, columns]
            null = np.eye(7) - hand.T @ np.linalg.solve(hand @ hand.T + 1e-6 * np.eye(6), hand)
            error = np.asarray(elbow_targets[side], dtype=float) - elbows[side][:3, 3]
            swivel = elbow_jacobian[3 * side:3 * side + 3, columns] @ null         # elbow motion per null-space joint motion
            line = np.linalg.svd(swivel)[0][:, 0]                                  # the way the elbow goes when the arm swivels
            across = np.linalg.norm(error - line * (line @ error))                 # of the whole error, before it is cut short
            distance = np.linalg.norm(error)
            trust = float(np.clip(min((ELBOW_TRUST_M[1] - across) / (ELBOW_TRUST_M[1] - ELBOW_TRUST_M[0]),
                                      (ELBOW_TRUST_REACH_M[1] - distance) / (ELBOW_TRUST_REACH_M[1] - ELBOW_TRUST_REACH_M[0])), 0., 1.))
            if distance > ELBOW_STEP_LIMIT:
                error = error * ELBOW_STEP_LIMIT / distance
            wish = trust * min(1., ELBOW_RATE * self.dt) * null @ (
                swivel.T @ np.linalg.solve(swivel @ swivel.T + ELBOW_DAMPING * np.eye(3), error))
            scale = 1.
            for index in range(7):
                if wish[index] > 1e-12:
                    scale = min(scale, ELBOW_SPEED_SHARE * up[columns][index] / wish[index])
                elif wish[index] < -1e-12:
                    scale = min(scale, ELBOW_SPEED_SHARE * down[columns][index] / wish[index])
            step[7 * side:7 * side + 7] = scale * wish; believed[side] = trust
        return step, believed

    def step(self, q31, left, right, *, waist_yaw_reference=None, posture=None, orientation=1., elbow_targets=None,
             waist_target=None, waist_pinned=False, **_):
        """Next 31-joint command and diagnostics for 4x4 pelvis-frame hand targets.

        ``waist_target`` (yaw, roll, pitch) is what somebody else wants of the waist - a policy that commands the
        chest as its own end effector.  By default it is the posture reference of all three waist joints, and
        the QP may still lean or turn the trunk for the hands.  With ``waist_pinned`` the waist goes to it at its
        speed limit whatever the hands ask, and the QP solves for the arms alone on that moving chest: the hands
        can no longer pull the trunk over, a target the arms cannot reach is left short.

        ``elbow_targets`` (2 x 3, pelvis frame; the elbow prior's prediction) make the arm posture term ask for the
        swivel toward them; ``posture`` is then used for an arm only as far as its prediction is not believed.
        ``posture`` (14 arm joints, SDK order) replaces the constant preferred arm posture for this tick.
        ``orientation`` (0..1) says how much the hand orientation matters now: 1 near and during a grasp,
        0 in free motion, where only ``FREE_ORIENTATION`` of its cost is kept and the posture decides.
        """
        q31 = np.array(q31, dtype=float, copy=True)
        self.legs = q31.copy()
        start = np.clip(q31[ACTIVE], self.lower, self.upper)     # mink refuses configurations outside the limits
        limits = self.limits
        moved = start.copy()                                     # the pose the QP starts from
        if waist_target is not None and waist_pinned:
            reach = self.velocity[:3] * self.dt
            wanted_waist = np.clip(np.asarray(waist_target, dtype=float), self.lower[:3], self.upper[:3])
            moved[:3] = start[:3] + np.clip(wanted_waist - start[:3], -reach, reach)
            limits = self.limits_waist_held
        q = np.empty(len(ACTIVE)); q[self.columns] = moved
        self.configuration.update(q)
        share = FREE_ORIENTATION + (1. - FREE_ORIENTATION) * float(np.clip(orientation, 0., 1.))
        for task, target in zip(self.hands, (left, right)):
            task.set_target(mink.SE3.from_matrix(np.asarray(target, dtype=float)))
            task.set_orientation_cost(ROTATION_COST * share)
        reference = POSTURE.copy()
        if posture is not None:
            reference[3:] = np.clip(posture, self.lower[3:], self.upper[3:])
        if waist_yaw_reference is not None:
            reference[0] = np.clip(waist_yaw_reference, self.lower[0], self.upper[0])
        if waist_target is not None:
            reference[:3] = np.clip(np.asarray(waist_target, dtype=float), self.lower[:3], self.upper[:3])
        wanted = np.empty(len(ACTIVE)); wanted[self.columns] = reference
        if elbow_targets is None:
            self.posture.set_target(wanted)
            tasks = [*self.hands, self.posture]
        else:
            self.waist_posture.set_target(wanted)
            swivel, believed = self._swivel_step(q31, elbow_targets)
            self.believed = believed
            # What is not believed of a prediction is made up by the pull of the posture reference (one share per tick).
            fallback = POSTURE_GAIN * (reference[3:] - start[3:]) * np.repeat(1. - believed, 7)
            stepped = moved.copy(); stepped[3:] += swivel + fallback
            goal = np.empty(len(ACTIVE)); goal[self.columns] = stepped
            self.arm_step.set_target(goal)
            tasks = [*self.hands, self.waist_posture, self.arm_step]
        # Every bound admits standing still, so the QP is feasible by construction.  Should the solver
        # still fail (numerics), drop the self-collision rows, then hold the pose; ``relaxed`` reports it.
        relaxed = 0
        try:
            velocity = mink.solve_ik(self.configuration, tasks, self.dt, self.solver, damping=QP_DAMPING,
                                     limits=limits)
        except mink.exceptions.NoSolutionFound:
            relaxed = 1
            try:
                velocity = mink.solve_ik(self.configuration, tasks, self.dt, self.solver, damping=QP_DAMPING,
                                         limits=limits[:3])
            except mink.exceptions.NoSolutionFound:
                relaxed = 2; velocity = np.zeros(self.model.nv)
        new = self.configuration.integrate(velocity, self.dt)[self.columns]
        new = np.clip(new, self.lower, self.upper)
        if limits is self.limits_waist_held:
            new[:3] = moved[:3]                                  # exactly the given step, not the QP's 1e-6 rad/s of play
        q31[ACTIVE] = new
        final_left, final_right = self.kinematics.fk(q31)
        step = np.abs(new - start)
        info = dict(
            position_error=np.array([np.linalg.norm(left[:3, 3] - final_left[:3, 3]), np.linalg.norm(right[:3, 3] - final_right[:3, 3])]),
            rotation_error=np.array([np.linalg.norm(rotation_vector(left[:3, :3] @ final_left[:3, :3].T)),
                                     np.linalg.norm(rotation_vector(right[:3, :3] @ final_right[:3, :3].T))]),
            elbow_error=np.zeros(2) if elbow_targets is None else np.linalg.norm(
                np.asarray(elbow_targets, dtype=float) - [elbow[:3, 3] for elbow in self.kinematics.arm_frames(q31)[1]], axis=1),
            saturated=bool(np.any(new <= self.lower + 1e-6) or np.any(new >= self.upper - 1e-6) or np.any(step >= self.velocity * self.dt - 1e-9)),
            # How far each elbow prediction was believed in this tick (0..1); zero without elbow targets.
            elbow_trust=self.believed.copy() if elbow_targets is not None else np.zeros(2),
            gap=self.collision.last_gap if self.collision is not None else np.nan, relaxed=relaxed)
        return q31, info

    def converge(self, q31, left, right, *, waist_yaw_reference=None, posture=None, elbow_targets=None, ticks=400, **_):
        """Repeat ``step`` until the pose stops changing (offline use)."""
        info = None
        for _ in range(ticks):
            previous = q31
            q31, info = self.step(q31, left, right, waist_yaw_reference=waist_yaw_reference, posture=posture,
                                  elbow_targets=elbow_targets)
            if np.max(abs(q31 - previous)) < 1e-7:
                break
        return q31, info
