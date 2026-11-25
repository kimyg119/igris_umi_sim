from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from scipy.spatial.transform import Rotation
from igris_ik.geometry import Pose
from umi_replay import reanchor, pose_error, prepare_scene, inferred_yaw
from igris_ik.realtime.retarget import hmd_waist_yaw


def test_relative_motion_uses_each_initial_tool_frame():
    rotation=Rotation.from_euler('z',90,degrees=True)
    start=SimpleNamespace(left=Pose([1,2,3],rotation.as_quat()[[3,0,1,2]]),right=Pose([0,0,0],[1,0,0,0]))
    poses=np.zeros((2,2,7));poses[:,:,6]=1;poses[1,:,:3]=[.1,0,0]
    poses[1,:,3:]=Rotation.from_euler('x',.2).as_quat()
    result=reanchor(poses,start)
    np.testing.assert_allclose(result[1,0,:3],[1,2.1,3],atol=1e-9)
    np.testing.assert_allclose(result[1,1,:3],[.1,0,0],atol=1e-9)
    np.testing.assert_allclose(Rotation.from_quat(result[1,0,3:]).as_matrix(),(rotation*Rotation.from_euler('x',.2)).as_matrix())
    opposite=result.copy();opposite[:,:,3:]*=-1
    position,angle=pose_error(result,opposite)
    assert position.max()<1e-9 and angle.max()<1e-5


def test_scene_removes_box_and_preserves_free_base():
    with TemporaryDirectory() as folder:
        prepare_scene(Path(folder))  # Asserts free base and absent box after MJCF compilation.


def test_pair_yaw_rotation_translation_degeneracy_and_sdk_sign():
    poses=np.zeros((4,2,7));poses[:,:,6]=1
    poses[0,:,:3]=[[.4,.2,.3],[.4,-.2,.3]]
    poses[1,:,:3]=Rotation.from_euler('z',45,degrees=True).apply(poses[0,:,:3])
    poses[2,:,:3]=poses[1,:,:3]+[.3,-.1,.2]
    poses[3,:,:3]=[.4,0,.3]
    np.testing.assert_allclose(np.rad2deg(inferred_yaw(poses)),[0,45,45,45],atol=1e-10)
    head=Pose([0,0,0],Rotation.from_euler('z',45,degrees=True).as_quat()[[3,0,1,2]])
    assert np.isclose(hmd_waist_yaw(head,0.,0.,[-1.5,1.5]),-np.pi/4)
