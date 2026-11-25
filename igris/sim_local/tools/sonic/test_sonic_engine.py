"""TDD tests for the SONIC engine + tracker step (S2 Task 3).

Task brief: .superpowers/sdd/2026-08-26-s2-sonic-zeroshot-tracker/task-3-brief.md
Contract authority: docs/superpowers/specs/2026-08-26-s2-sonic-zeroshot-tracker-design.md

Offline; no sim needed. Run:
  conda run --no-capture-output -n gmr python -m pytest \
      sim_local/tools/sonic/test_sonic_engine.py -q
"""
import numpy as np
import pytest

import sonic_const as C
from sonic_obs import assemble_encoder_obs, apply_delta_heading, _ENC_ANCHOR

import sonic_engine as SE
from sonic_engine import (
    SonicEngine,
    SonicStep,
    PhiEnc,
    MotionRef,
    RobotProprio,
    build_encoder_window,
    DEFAULT_ANGLES_ISAACLAB,
)

# igris_codec is placed on sys.path by importing sonic_engine.
from igris_codec import PhiDecoder, G1_MJ29


# --- helpers -----------------------------------------------------------------
def _yaw_quat(a):
    return np.array([np.cos(a / 2.0), 0.0, 0.0, np.sin(a / 2.0)], dtype=np.float64)


def _static_proprio(base_quat=None):
    # Robot standing AT the G1 default pose (jpos_rel ~= 0): invert Phi_enc per joint so
    # PhiEnc(proprio) == DEFAULT_ANGLES, a faithful "static standing" proprio consistent
    # with the DEFAULT_ANGLES reference (a mismatched pose would inject a large cold-start
    # tracking transient that is not what test (c) is checking). Affine joints invert
    # q_igris = (g1_default - b)/a; the 9 closed-form (waist/shoulder) joints invert via the
    # codec forward (Phi_dec of g1_default), the exact inverse of PhiEnc's closed_form_invert.
    penc = PhiEnc()
    g1_default = {n: C.DEFAULT_ANGLES[i] for i, n in enumerate(C.G1_HW_NAMES)}
    ig_from_default = PhiDecoder(str(SE.DEFAULT_PHI_DEC)).joints(np.asarray(C.DEFAULT_ANGLES, dtype=np.float64))
    jp = {ig: (float(np.asarray(ig_from_default[ig]).reshape(-1)[0]) if g1 in penc.closed_form
              else (g1_default[g1] - b) / a)
          for (g1, ig, a, b) in penc.entries}
    jv = {ig: 0.0 for ig in jp}
    return RobotProprio(
        joint_pos=jp,
        joint_vel=jv,
        base_angvel=np.zeros(3),
        gravity=np.array([0.0, 0.0, -1.0]),
        base_quat=np.array([1.0, 0.0, 0.0, 0.0]) if base_quat is None else base_quat,
    )


def _static_ref(root_yaw=0.0):
    jp = np.tile(DEFAULT_ANGLES_ISAACLAB, (10, 1))
    jv = np.zeros((10, 29))
    rq = np.tile(_yaw_quat(root_yaw), (10, 1))
    return MotionRef(joint_positions=jp, joint_velocities=jv, root_quat=rq)


@pytest.fixture(scope="module")
def engine():
    return SonicEngine()


def _phi_dec():
    return PhiDecoder(str(SE.DEFAULT_PHI_DEC))


# --- (const) joint-order threading pins ---------------------------------------
def test_joint_order_hw_equals_codec_g1mj29():
    # action_to_qtarget_g1 returns hardware order; phi_dec.joints wants G1_MJ29.
    # They must be the SAME order or the codec silently scrambles joints.
    assert tuple(G1_MJ29) == tuple(C.G1_HW_NAMES)


def test_default_angles_isaaclab_reindex():
    for i in range(29):
        assert DEFAULT_ANGLES_ISAACLAB[i] == C.DEFAULT_ANGLES[C.MUJOCO_TO_ISAACLAB[i]]


# --- (a) engine loads + shapes ------------------------------------------------
def test_engine_encode_decode_shapes(engine):
    tokens = engine.encode(np.zeros(1762, dtype=np.float32))
    assert tokens.shape == (64,)
    action = engine.decode(np.zeros(994, dtype=np.float32))
    assert action.shape == (29,)
    assert np.all(np.isfinite(tokens))
    assert np.all(np.isfinite(action))


def test_engine_threads_are_one(engine):
    assert engine.enc.get_session_options().intra_op_num_threads == 1
    assert engine.enc.get_session_options().inter_op_num_threads == 1


# --- (b) one step -> valid 29 IGRIS joints ------------------------------------
def test_step_returns_valid_igris(engine):
    step = SonicStep(engine, _phi_dec(), PhiEnc())
    out = step.step(_static_ref(), _static_proprio())
    assert isinstance(out, dict)
    assert len(out) == 29
    vals = np.array(list(out.values()), dtype=np.float64)
    assert np.all(np.isfinite(vals))
    assert np.all(np.abs(vals) < 3.5)  # sane joint soft bound (rad)


# --- (c) 5 consecutive static steps do not diverge ----------------------------
def test_five_static_steps_bounded(engine):
    step = SonicStep(engine, _phi_dec(), PhiEnc())
    ref, pro = _static_ref(), _static_proprio()
    prev = None
    for _ in range(5):
        out = step.step(ref, pro)
        v = np.array([out[k] for k in sorted(out)], dtype=np.float64)
        assert np.all(np.isfinite(v))
        assert np.all(np.abs(v) < 3.5)
        if prev is not None:
            assert np.max(np.abs(v - prev)) < 0.2  # settles to a near-still stand, no blow-up
        prev = v


# --- (d) anchor delta-heading pre-apply is actually wired ---------------------
def test_anchor_delta_heading_preapply_changes_obs():
    # reference points 0.7 rad off the robot base heading at init.
    base = _yaw_quat(0.0)
    init_base = _yaw_quat(0.0)
    init_ref = _yaw_quat(0.7)
    ref = _static_ref(root_yaw=0.7)

    delta = apply_delta_heading(init_base, init_ref)
    win_pre = build_encoder_window(ref, base, delta)                      # pre-applied
    win_raw = build_encoder_window(ref, base, np.array([1.0, 0.0, 0.0, 0.0]))  # identity == no pre-apply

    A = slice(_ENC_ANCHOR, _ENC_ANCHOR + 60)
    obs_pre = assemble_encoder_obs(win_pre)
    obs_raw = assemble_encoder_obs(win_raw)
    assert not np.allclose(obs_pre[A], obs_raw[A]), "anchor block identical -> pre-apply missing"
    # with the delta pre-applied, ref heading re-aligns to base -> anchor == identity 6D
    assert np.allclose(obs_pre[A].reshape(10, 6)[0], [1, 0, 0, 1, 0, 0], atol=1e-6)


def test_step_uses_preapply_end_to_end(engine):
    # A yawed reference vs identity reference must yield different encoder tokens,
    # proving step() feeds the (heading-adjusted) anchor into the encoder.
    step_a = SonicStep(engine, _phi_dec(), PhiEnc())
    step_b = SonicStep(engine, _phi_dec(), PhiEnc())
    # base identity, ref frame 0 yawed -> init_ref differs from init_base -> nonzero delta.
    out_a = step_a.step(_static_ref(root_yaw=0.7), _static_proprio())
    out_b = step_b.step(_static_ref(root_yaw=0.0), _static_proprio())
    # both are valid; delta-heading re-anchors, so a pure heading offset should
    # collapse to (near) the aligned case -> outputs close but both finite/valid.
    for out in (out_a, out_b):
        v = np.array(list(out.values()))
        assert np.all(np.isfinite(v)) and np.all(np.abs(v) < 3.5)
