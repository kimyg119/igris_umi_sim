import json
import pytest
from quest_trajectory_replay import load_capture


def test_loaded_file_reference_stays_alive_without_ik_frame_completion():
    import threading
    from types import SimpleNamespace
    from quest_trajectory_replay import refresh_file_reference
    refreshed=threading.Event();stopped=threading.Event();calls=[]
    def refresh(active):
        calls.append(active);refreshed.set()
    worker=threading.Thread(target=refresh_file_reference,args=(SimpleNamespace(set_reference_active=refresh),stopped))
    worker.start()
    try:
        assert refreshed.wait(1.)
        refreshed.clear()
        assert refreshed.wait(1.)  # No IK step has completed between these refreshes.
    finally:
        stopped.set();worker.join(1.)
    assert not worker.is_alive() and len(calls)>=2 and all(calls)


def test_capture_keeps_sampling_gaps_and_reports_live_reference_jumps(tmp_path):
    pose=dict(pos=[0,0,0],quat=[0,0,0,1])
    rows=[dict(seq=i,t_capture=100.+t,t_capture_monotonic_ns=int(t*1e9),ee_frame='robot_base',left=pose,right=pose,head=pose)
          for i,t in enumerate((1.,1.033,2.3))]
    rows[-1]['right']=dict(pos=[.2,0,0],quat=[0,0,0,1])
    path=tmp_path/'input.jsonl';path.write_text('\n'.join(map(json.dumps,rows)))
    messages,t,audit=load_capture(path)
    assert len(messages)==3 and t[-1]==pytest.approx(1.3)
    assert audit['gaps'][0]['gap_s']==pytest.approx(1.267)
    assert audit['jumps']['right'][0]['step_mm']==pytest.approx(200)
    assert audit['missing_buttons']==3
    rows[-1]['t_capture_monotonic_ns']=rows[0]['t_capture_monotonic_ns']
    path.write_text('\n'.join(map(json.dumps,rows)))
    with pytest.raises(ValueError,match='strictly increase'):load_capture(path)


def test_pelvis_targets_preserve_absolute_pose_and_head_yaw():
    import numpy as np
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    from igris_ik.geometry import Pose, UpperBodyTargets
    from quest_trajectory_replay import make_targets, pose_array

    def pose(p, yaw):
        return Pose(p, Rotation.from_euler('z', yaw).as_quat()[[3,0,1,2]])
    home=UpperBodyTargets(pose([.6,.25,.35],0),pose([.6,-.25,.35],0),torso=pose([0,0,.1],0))
    ik=SimpleNamespace(forward=lambda q:home,config=SimpleNamespace(waist_limits_deg={'waist_yaw':(-88,88)}))
    messages=[SimpleNamespace(left=pose([-.16,.3,.25],.7),right=pose([-.27,.1,.34],-.9),hmd=pose([0,0,.8],yaw))
              for yaw in (.3,.5)]
    q=np.zeros(31);q[0]=.2
    targets,waist=make_targets(messages,ik,q)
    for source,target in zip(messages,targets):
        np.testing.assert_allclose(pose_array(target),pose_array(UpperBodyTargets(source.left,source.right,head=source.hmd)))
    np.testing.assert_allclose(waist,[-.3,-.5])
