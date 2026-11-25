import numpy as np

from igris.umi_walklow_replay.ik import POSTURE
from igris.umi_walklow_replay.kinematics import ACTIVE, JOINTS, UpperBody
from igris.umi_walklow_replay.posture import arm_posture, swivel_deg


def test_swivel_is_zero_straight_down_and_positive_outward_on_both_sides():
    hand = np.array([.4, 0., 0.])                       # hand straight ahead of the shoulder
    assert swivel_deg(np.array([.2, 0., -.1]), hand, 0) == 0.
    assert np.isclose(swivel_deg(np.array([.2, .1, 0.]), hand, 0), 90.)       # left elbow out to the left
    assert np.isclose(swivel_deg(np.array([.2, -.1, 0.]), hand, 1), 90.)      # right elbow out to the right
    assert np.isclose(swivel_deg(np.array([.2, -.1, 0.]), hand, 0), -90.)     # left elbow across the body
    assert swivel_deg(np.array([.2, 0., 0.]), hand, 0) == 0.                  # elbow on the line: undefined -> 0


def test_arm_posture_reports_limits_and_mirror_symmetry():
    model = UpperBody.from_mjcf()
    q = np.zeros((3, len(JOINTS))); q[:, ACTIVE] = POSTURE
    lower, upper = model.lower[ACTIVE] + .035, model.upper[ACTIVE] - .035
    q[2, ACTIVE[4]] = lower[4]                           # one sample with the left shoulder roll at its bound
    report, elbows, swivel = arm_posture(model, q, lower, upper)
    assert report['samples'] == 3 and elbows.shape == (3, 2, 3) and swivel.shape == (3, 2)
    assert np.isclose(swivel[0, 0], swivel[0, 1], atol=1e-6)                  # mirrored posture, same swivel
    assert np.isclose(report['joints_at_limit_fraction']['l_shoulder_roll'], 1 / 3)
    assert np.isclose(report['any_arm_joint_at_limit_fraction'], 1 / 3)
