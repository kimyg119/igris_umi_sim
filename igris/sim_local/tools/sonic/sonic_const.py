"""SONIC G1 constants — joint-order permutations, default pose, action scales, names.

Source of truth (copied VERBATIM):
  GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp

  - isaaclab_to_mujoco / mujoco_to_isaaclab .............. hpp:100-104
  - default_angles (29, hardware/MuJoCo order) ........... hpp:210-240
  - g1_action_scale construction ........................ hpp:106-139
  - motor armature / effort / natural-freq constants .... hpp:37-65

Action scales are COMPUTED here from the motor constants (not hardcoded floats),
so a reader can check them:  stiffness = armature * omega^2 ;
action_scale = 0.25 * effort / stiffness ;  omega = 10 Hz * 2*pi.

Everything is plain Python lists (float) so callers wrap in numpy as needed.
"""
import math

# --- joint-order permutations (hpp:100-104, verbatim) ------------------------
# isaaclab_to_mujoco: "mujoco order in isaaclab index".
# Used to GATHER an isaaclab-order action into hardware order (spec §2.4, trap 7):
#   q_target_hw[i] = default[i] + action_isaaclab[ISAACLAB_TO_MUJOCO[i]] * scale[i]
ISAACLAB_TO_MUJOCO = [
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
    11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
]
# mujoco_to_isaaclab: "isaaclab order in mujoco index".
# Used to convert hardware-order proprio into isaaclab order (spec §2.4, trap 7):
#   body_q_isaaclab[i] = q_hw[MUJOCO_TO_ISAACLAB[i]] - default[MUJOCO_TO_ISAACLAB[i]]
MUJOCO_TO_ISAACLAB = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
    16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]

# --- default standing pose (29, hardware/MuJoCo order, hpp:210-240) ----------
DEFAULT_ANGLES = [
    -0.312, 0.0,  0.0,  0.669, -0.363, 0.0,   # left leg
    -0.312, 0.0,  0.0,  0.669, -0.363, 0.0,   # right leg
     0.0,   0.0,  0.0,                        # waist yaw/roll/pitch
     0.2,   0.2,  0.0,  0.6,   0.0,   0.0, 0.0,  # left arm  (sh_p,sh_r,sh_y,elbow,wr_r,wr_p,wr_y)
     0.2,  -0.2,  0.0,  0.6,   0.0,   0.0, 0.0,  # right arm
]

# --- G1 joint names, hardware/MuJoCo order (hpp default_angles comments) ------
G1_HW_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# --- action scale, computed from motor constants (hpp:37-139) -----------------
# Motor armature constants (hpp:40-43)
_ARMATURE = {
    "5020": 0.003609725,
    "7520_14": 0.010177520,
    "7520_22": 0.025101925,
    "4010": 0.00425,
}
# Effort limits (hpp:62-65)
_EFFORT = {
    "5020": 25.0,
    "7520_14": 88.0,
    "7520_22": 139.0,
    "4010": 5.0,
}
_OMEGA = 10.0 * 2.0 * math.pi  # natural freq, 10 Hz (hpp:46)


def _stiffness(motor_type):
    return _ARMATURE[motor_type] * _OMEGA * _OMEGA


def _action_scale(motor_type):
    return 0.25 * _EFFORT[motor_type] / _stiffness(motor_type)


# Per-joint motor type in hardware order (from g1_action_scale comments, hpp:109-139)
_MOTOR_TYPE_HW = [
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",   # left leg
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",   # right leg
    "7520_14", "5020", "5020",                                    # waist
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",       # left arm
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",       # right arm
]
ACTION_SCALE = [_action_scale(mt) for mt in _MOTOR_TYPE_HW]

# --- G1 joint names, IsaacLab order ------------------------------------------
# isaaclab position i corresponds to hardware joint MUJOCO_TO_ISAACLAB[i]
# (verified against the proprio conversion in g1_deploy_onnx_ref.cpp:2830).
G1_ISAACLAB_NAMES = [G1_HW_NAMES[MUJOCO_TO_ISAACLAB[i]] for i in range(29)]


if __name__ == "__main__":
    # self-check: permutations are true inverses, scales match spec §2.4 approximations
    assert all(MUJOCO_TO_ISAACLAB[ISAACLAB_TO_MUJOCO[i]] == i for i in range(29))
    assert sorted(ISAACLAB_TO_MUJOCO) == list(range(29))
    assert abs(ACTION_SCALE[0] - 0.3506) < 1e-3   # 7520_22
    assert abs(ACTION_SCALE[2] - 0.5475) < 1e-3   # 7520_14
    assert abs(ACTION_SCALE[4] - 0.4386) < 1e-3   # 5020
    assert abs(ACTION_SCALE[20] - 0.0745) < 1e-3  # 4010
    print("sonic_const self-check OK")
    for i in range(29):
        print(f"{i:2d} hw={G1_HW_NAMES[i]:28s} scale={ACTION_SCALE[i]:.4f} "
              f"il={G1_ISAACLAB_NAMES[i]}")
