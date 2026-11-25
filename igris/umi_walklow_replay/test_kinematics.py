"""Kinematics must agree with MuJoCo on the unchanged vendor model."""
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from igris.umi_walklow_replay.kinematics import (ACTIVE, ELBOWS, END_EFFECTORS, JOINTS, ROOT, SHOULDERS, TORSO,
                                                 VENDOR_MJCF, UpperBody, coupled_halfspaces, matrix_quat, quat_matrix,
                                                 rotation_vector, axis_rotation)


def host_model_path(directory):
    """Vendor MJCF with container asset prefixes mapped to this checkout."""
    root = ET.parse(VENDOR_MJCF).getroot()
    for element in root.iter():
        for key, value in list(element.attrib.items()):
            for prefix, target in (('/robros/configs/', ROOT / 'igris/configs'), ('/igris/', ROOT / 'igris')):
                if value.startswith(prefix):
                    element.set(key, str(target / value[len(prefix):]))
    # Host MuJoCo is newer than the vendor's 3.2.7 and rejects its actuator
    # syntax.  Body frames and joint axes, the only things compared, are kept.
    for tag in ('actuator', 'sensor'):
        for element in root.findall(tag):
            root.remove(element)
    path = Path(directory) / 'vendor.xml'
    ET.ElementTree(root).write(path, encoding='unicode')
    return path


@pytest.fixture(scope='module')
def model():
    return UpperBody.from_mjcf()


def random_q(model, rng):
    q = np.zeros(len(JOINTS))
    q[ACTIVE] = rng.uniform(model.lower[ACTIVE], model.upper[ACTIVE])
    return q


def test_fk_matches_mujoco(model):
    mujoco = pytest.importorskip('mujoco')
    with tempfile.TemporaryDirectory() as directory:
        native = mujoco.MjModel.from_xml_path(str(host_model_path(directory)))
    data = mujoco.MjData(native)
    pelvis = native.body('pelvis').id
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = random_q(model, rng)
        mujoco.mj_resetData(native, data)
        for index in ACTIVE:
            data.qpos[native.joint(JOINTS[index]).qposadr[0]] = q[index]
        mujoco.mj_kinematics(native, data)
        base_r = data.xmat[pelvis].reshape(3, 3); base_p = data.xpos[pelvis]
        for name, pose in zip(END_EFFECTORS, model.fk(q)):
            body = native.body(name).id
            position = base_r.T @ (data.xpos[body] - base_p)
            rotation = base_r.T @ data.xmat[body].reshape(3, 3)
            assert np.allclose(pose[:3, 3], position, atol=1e-6)
            assert np.allclose(pose[:3, :3], rotation, atol=1e-6)


def test_jacobian_matches_finite_difference(model):
    rng = np.random.default_rng(1)
    for _ in range(5):
        q = random_q(model, rng)
        left, right, jacobian = model.fk_jacobian(q)
        for column, joint in enumerate(ACTIVE):
            step = np.zeros(len(JOINTS)); step[joint] = 1e-6
            for rows, base, moved in ((slice(0, 6), left, model.fk(q + step)[0]),
                                      (slice(6, 12), right, model.fk(q + step)[1])):
                linear = (moved[:3, 3] - base[:3, 3]) / 1e-6
                angular = rotation_vector(moved[:3, :3] @ base[:3, :3].T) / 1e-6
                assert np.allclose(jacobian[rows, column], np.r_[linear, angular], atol=1e-5)


def test_rotation_vector_roundtrip():
    rng = np.random.default_rng(2)
    for angle in (0., 1e-9, .3, 2.5, np.pi - 1e-7, np.pi):
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        recovered = rotation_vector(axis_rotation(axis, angle))
        assert np.allclose(axis_rotation(recovered / max(np.linalg.norm(recovered), 1e-12),
                                         np.linalg.norm(recovered)), axis_rotation(axis, angle), atol=1e-6)
    assert np.allclose(quat_matrix([1, 0, 0, 0]), np.eye(3))


def test_limits_and_coupled_sets(model):
    assert model.lower[JOINTS.index('l_elbow_pitch')] == pytest.approx(-2.094395102)
    assert model.upper[JOINTS.index('waist_yaw')] == pytest.approx(1.570796327)
    sets = coupled_halfspaces()
    assert [[JOINTS[i] for i in joints] for joints, _, _ in sets] == [
        ['waist_roll', 'waist_pitch'], ['l_wrist_roll', 'l_wrist_pitch'], ['r_wrist_roll', 'r_wrist_pitch']]
    for _, normal, offset in sets:
        assert np.all(normal @ np.zeros(2) <= offset)      # the neutral pose is inside every safe set


def test_arm_frames_match_mujoco_and_finite_difference(model):
    mujoco = pytest.importorskip('mujoco')
    with tempfile.TemporaryDirectory() as directory:
        native = mujoco.MjModel.from_xml_path(str(host_model_path(directory)))
    data = mujoco.MjData(native)
    pelvis = native.body('pelvis').id
    rng = np.random.default_rng(4)
    for _ in range(10):
        q = random_q(model, rng)
        mujoco.mj_resetData(native, data)
        for index in ACTIVE:
            data.qpos[native.joint(JOINTS[index]).qposadr[0]] = q[index]
        mujoco.mj_kinematics(native, data)
        base_r = data.xmat[pelvis].reshape(3, 3); base_p = data.xpos[pelvis]
        torso, elbows, hands, jacobian = model.arm_frames(q)
        for name, pose in ((TORSO, torso), (ELBOWS[0], elbows[0]), (ELBOWS[1], elbows[1]),
                           (END_EFFECTORS[0], hands[0]), (END_EFFECTORS[1], hands[1])):
            body = native.body(name).id
            assert np.allclose(pose[:3, 3], base_r.T @ (data.xpos[body] - base_p), atol=1e-6)
            assert np.allclose(pose[:3, :3], base_r.T @ data.xmat[body].reshape(3, 3), atol=1e-6)
        for column, joint in enumerate(ACTIVE):
            step = np.zeros(len(JOINTS)); step[joint] = 1e-6
            moved = model.arm_frames(q + step)[1]
            numeric = np.r_[moved[0][:3, 3] - elbows[0][:3, 3], moved[1][:3, 3] - elbows[1][:3, 3]] / 1e-6
            assert np.allclose(jacobian[:, column], numeric, atol=1e-5)


def test_shoulder_origins_and_quaternion_roundtrip(model):
    torso, elbows, _, _ = model.arm_frames(np.zeros(len(JOINTS)))
    for side in range(2):
        # At the zero pose the upper arm hangs straight down from the shoulder origin.
        offset = np.linalg.inv(torso)[:3, :3] @ (elbows[side][:3, 3] - torso[:3, 3]) - SHOULDERS[side]
        assert abs(offset[1]) < 1e-9 and offset[2] < -.15
    rng = np.random.default_rng(5)
    for _ in range(20):
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        rotation = axis_rotation(axis, rng.uniform(-np.pi, np.pi))
        assert np.allclose(quat_matrix(matrix_quat(rotation)), rotation, atol=1e-9)
