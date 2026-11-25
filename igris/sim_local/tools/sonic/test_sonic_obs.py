"""TDD tests for the SONIC obs assembly + constants (S2 Task 2).

Contract authority: docs/superpowers/specs/2026-08-26-s2-sonic-zeroshot-tracker-design.md
(§2.1 decoder 994, §2.2 encoder 1762 g1-mode, §2.4 action formula, §7 the 6 traps).

Run:
  conda run --no-capture-output -n gmr python -m pytest \
      sim_local/tools/sonic/test_sonic_obs.py -q
"""
import numpy as np
import pytest

import sonic_const as C
from sonic_obs import (
    StateLog,
    assemble_decoder_obs,
    assemble_encoder_obs,
    anchor6,
    action_to_qtarget_g1,
    MotionRefWindow,
)


# --- constants ---------------------------------------------------------------

def test_perm_inverse():
    A, B = C.ISAACLAB_TO_MUJOCO, C.MUJOCO_TO_ISAACLAB
    assert len(A) == 29 and len(B) == 29
    assert all(B[A[i]] == i for i in range(29))
    assert all(A[B[i]] == i for i in range(29))


def test_perm_are_permutations():
    assert sorted(C.ISAACLAB_TO_MUJOCO) == list(range(29))
    assert sorted(C.MUJOCO_TO_ISAACLAB) == list(range(29))


def test_default_angles_shape_and_values():
    assert len(C.DEFAULT_ANGLES) == 29
    # spec §2.4: leg hip_pitch -0.312, knee 0.669, ankle_pitch -0.363 (hardware order)
    assert C.DEFAULT_ANGLES[0] == pytest.approx(-0.312)   # left_hip_pitch
    assert C.DEFAULT_ANGLES[3] == pytest.approx(0.669)    # left_knee
    assert C.DEFAULT_ANGLES[4] == pytest.approx(-0.363)   # left_ankle_pitch


def test_action_scale_matches_spec():
    assert len(C.ACTION_SCALE) == 29
    # spec §2.4 approximations, hardware order
    assert C.ACTION_SCALE[0] == pytest.approx(0.3506, abs=1e-3)   # 7520_22 (left_hip_pitch)
    assert C.ACTION_SCALE[2] == pytest.approx(0.5475, abs=1e-3)   # 7520_14 (left_hip_yaw)
    assert C.ACTION_SCALE[4] == pytest.approx(0.4386, abs=1e-3)   # 5020 (left_ankle_pitch)
    assert C.ACTION_SCALE[20] == pytest.approx(0.0745, abs=1e-3)  # 4010 (left_wrist_pitch)


def test_names_lengths_and_isaaclab_mapping():
    assert len(C.G1_HW_NAMES) == 29
    assert len(C.G1_ISAACLAB_NAMES) == 29
    # isaaclab name at position i is the hardware joint mujoco_to_isaaclab[i] (cpp:2830 proprio conv)
    assert all(
        C.G1_ISAACLAB_NAMES[i] == C.G1_HW_NAMES[C.MUJOCO_TO_ISAACLAB[i]]
        for i in range(29)
    )
    assert C.G1_HW_NAMES[0] == "left_hip_pitch_joint"
    assert C.G1_HW_NAMES[12] == "waist_yaw_joint"


# --- action -> joint target (trap 1 direction, trap 7 scatter/gather) ---------

def test_action_to_qtarget_zero_is_default():
    q = action_to_qtarget_g1(np.zeros(29))
    assert q.shape == (29,)
    np.testing.assert_allclose(q, np.asarray(C.DEFAULT_ANGLES), rtol=0, atol=0)


def test_action_to_qtarget_gather_direction():
    # hardware i pulls action[isaaclab_to_mujoco[i]]; set one isaaclab action nonzero
    a = np.zeros(29)
    il = 5
    a[il] = 1.0
    q = action_to_qtarget_g1(a)
    hw = C.ISAACLAB_TO_MUJOCO.index(il)  # hardware joint that gathers from isaaclab index il
    expected = np.asarray(C.DEFAULT_ANGLES, float).copy()
    expected[hw] += C.ACTION_SCALE[hw] * 1.0
    np.testing.assert_allclose(q, expected)


# --- decoder obs (994) -------------------------------------------------------

def _fill_statelog(sl, n=10):
    for f in range(n):
        sl.push(
            joint_pos_rel=np.full(29, f + 0.01),
            joint_vel=np.full(29, f + 0.02),
            base_angvel=np.full(3, f + 0.03),
            gravity=np.full(3, f + 0.04),
            last_action=np.full(29, f + 0.05),
        )


def test_decoder_obs_shape_and_dtype():
    sl = StateLog()
    _fill_statelog(sl)
    tokens = np.arange(64, dtype=float)
    obs = assemble_decoder_obs(tokens, sl)
    assert obs.shape == (994,)
    assert obs.dtype == np.float32


def test_decoder_obs_layout_offsets():
    sl = StateLog()
    _fill_statelog(sl)
    tokens = np.arange(64, dtype=float)
    obs = assemble_decoder_obs(tokens, sl)
    # token_state (64)
    np.testing.assert_allclose(obs[0:64], tokens)
    # his_base_angular_velocity x10 (30) — frame-major, oldest->newest
    assert obs[64:94].shape == (30,)
    np.testing.assert_allclose(obs[64:67], np.full(3, 0.03))   # oldest frame
    np.testing.assert_allclose(obs[91:94], np.full(3, 9.03))   # newest frame
    # his_body_joint_positions x10 (290)
    np.testing.assert_allclose(obs[94:123], np.full(29, 0.01))     # oldest
    np.testing.assert_allclose(obs[355:384], np.full(29, 9.01))    # newest
    # his_body_joint_velocities x10 (290)
    np.testing.assert_allclose(obs[384:413], np.full(29, 0.02))
    # his_last_actions x10 (290)
    np.testing.assert_allclose(obs[674:703], np.full(29, 0.05))
    # his_gravity_dir x10 (30)
    np.testing.assert_allclose(obs[964:967], np.full(3, 0.04))     # oldest
    np.testing.assert_allclose(obs[991:994], np.full(3, 9.04))     # newest


# --- StateLog ordering (trap 2: oldest->newest) ------------------------------

def test_statelog_oldest_to_newest():
    sl = StateLog()
    for f in range(10):
        sl.push(
            joint_pos_rel=np.full(29, float(f)),
            joint_vel=np.zeros(29),
            base_angvel=np.zeros(3),
            gravity=np.zeros(3),
            last_action=np.zeros(29),
        )
    stacked = sl.stack("joint_pos_rel")
    assert stacked.shape == (290,)
    # last-pushed (9) lands in the newest (final) slot; first-pushed (0) is oldest
    np.testing.assert_allclose(stacked[0:29], np.full(29, 0.0))
    np.testing.assert_allclose(stacked[-29:], np.full(29, 9.0))


def test_statelog_ring_evicts_oldest():
    sl = StateLog()
    for f in range(15):  # 5 more than capacity
        sl.push(
            joint_pos_rel=np.full(29, float(f)),
            joint_vel=np.zeros(29),
            base_angvel=np.zeros(3),
            gravity=np.zeros(3),
            last_action=np.zeros(29),
        )
    stacked = sl.stack("joint_pos_rel")
    # frames 5..14 remain, oldest->newest
    np.testing.assert_allclose(stacked[0:29], np.full(29, 5.0))
    np.testing.assert_allclose(stacked[-29:], np.full(29, 14.0))


def test_statelog_warmup_pads_to_ten():
    sl = StateLog()
    for f in range(3):
        sl.push(
            joint_pos_rel=np.full(29, float(f)),
            joint_vel=np.zeros(29),
            base_angvel=np.zeros(3),
            gravity=np.zeros(3),
            last_action=np.zeros(29),
        )
    stacked = sl.stack("joint_pos_rel")
    assert stacked.shape == (290,)          # padded to 10 frames
    np.testing.assert_allclose(stacked[-29:], np.full(29, 2.0))  # newest is last-pushed


# --- encoder obs (1762) ------------------------------------------------------

def _make_window():
    return MotionRefWindow(
        joint_positions=np.arange(10 * 29, dtype=float).reshape(10, 29) + 1.0,
        joint_velocities=np.arange(10 * 29, dtype=float).reshape(10, 29) + 1000.0,
        anchor6=np.arange(10 * 6, dtype=float).reshape(10, 6) + 1.0,
    )


def test_encoder_obs_shape_and_dtype():
    obs = assemble_encoder_obs(_make_window())
    assert obs.shape == (1762,)
    assert obs.dtype == np.float32


def test_encoder_obs_g1_blocks_and_zeros():
    w = _make_window()
    obs = assemble_encoder_obs(w)
    # encoder_mode_4 = [0,0,0,0] (g1 mode_id=0), not one-hot
    np.testing.assert_allclose(obs[0:4], np.zeros(4))
    # motion_joint_positions_10frame_step5 @ 4..294 (frame-major, absolute)
    np.testing.assert_allclose(obs[4:294], w.joint_positions.reshape(-1))
    # motion_joint_velocities_10frame_step5 @ 294..584
    np.testing.assert_allclose(obs[294:584], w.joint_velocities.reshape(-1))
    # motion_anchor_orientation_10frame_step5 @ 601..661
    np.testing.assert_allclose(obs[601:661], w.anchor6.reshape(-1))
    # every other offset is exactly 0
    zero_mask = np.ones(1762, dtype=bool)
    zero_mask[0:4] = False
    zero_mask[4:294] = False
    zero_mask[294:584] = False
    zero_mask[601:661] = False
    np.testing.assert_array_equal(obs[zero_mask], np.zeros(int(zero_mask.sum()), np.float32))


def test_encoder_obs_g1_blocks_nonzero_capable():
    obs = assemble_encoder_obs(_make_window())
    assert np.any(obs[4:294] != 0)
    assert np.any(obs[294:584] != 0)
    assert np.any(obs[601:661] != 0)


# --- anchor6 (trap 5: rotation matrix first 2 cols row-wise) -----------------

def test_anchor6_identity_pair():
    q = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz identity
    np.testing.assert_allclose(anchor6(q, q), [1, 0, 0, 1, 0, 0], atol=1e-12)


def test_anchor6_equal_nonidentity_pair():
    # any equal unit quats -> relative rotation is identity -> [1,0,0,1,0,0]
    q = np.array([0.5, 0.5, 0.5, 0.5])  # unit wxyz
    q = q / np.linalg.norm(q)
    np.testing.assert_allclose(anchor6(q, q), [1, 0, 0, 1, 0, 0], atol=1e-12)


def test_anchor6_row_wise_first_two_columns():
    # ref rotated +90deg about z relative to base (base=identity)
    base = np.array([1.0, 0.0, 0.0, 0.0])
    a = np.pi / 2
    ref = np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])  # wxyz z-rot
    out = anchor6(base, ref)
    # Rz(90) = [[0,-1,0],[1,0,0],[0,0,1]]; first 2 cols row-wise: [0,-1, 1,0, 0,0]
    np.testing.assert_allclose(out, [0, -1, 1, 0, 0, 0], atol=1e-9)
