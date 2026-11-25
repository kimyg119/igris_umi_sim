"""Regress common target rotation, intentional yaw, and bounded feedback."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from box_orientation_feedback import BoxOrientationFeedback, rotate_target


def check():
    # Omitting/reversing feedback lets a constant wrist disturbance tip the load.
    f=BoxOrientationFeedback();disturbance=R.from_rotvec(np.deg2rad([4.,4.,0.]))
    measured=disturbance
    previous=R.identity()
    for _ in range(1200):
        correction=R.from_matrix(f.update(np.eye(3),measured.as_matrix(),0.,1/30,True))
        assert (correction*previous.inv()).magnitude()<=np.deg2rad(.3)/30+1e-12
        measured=correction*disturbance;previous=correction
    assert measured.magnitude()<np.deg2rad(.1), 'Shared EE correction did not reject payload tilt'
    assert f.correction.magnitude()<=np.deg2rad(10)+1e-12
    # The same local correction must follow the permitted yaw, without opposing it.
    yaw=1.1;y=R.from_euler('z',yaw);before=f.correction.as_matrix().copy()
    observed=y*correction*disturbance
    applied=f.update(y.as_matrix(),observed.as_matrix(),yaw,1/30,False)
    assert np.allclose(f.correction.as_matrix(),before), 'Inactive controller integrated'
    assert np.allclose(applied,y.as_matrix()@before@y.as_matrix().T)
    fresh=BoxOrientationFeedback()
    assert np.allclose(fresh.update(y.as_matrix(),y.as_matrix(),yaw,1/30,True),np.eye(3))
    for _ in range(1500):fresh.update(np.eye(3),disturbance.as_matrix(),0.,1/30,True)
    assert fresh.correction.magnitude()<=np.deg2rad(10)+1e-12
    # Both hand poses rotate rigidly around the planned box center; their relative
    # pose, gap and center distances must be preserved (no artificial squeeze).
    a=np.eye(4);a[:3,3]=[.3,.1,.8];b=np.eye(4);b[:3,3]=[.3,-.1,.8]
    center=np.array([.3,0.,.85]);rot=R.from_rotvec([.03,-.02,.01]).as_matrix()
    aa=rotate_target(a,center,rot);bb=rotate_target(b,center,rot)
    assert not np.allclose(aa,a)
    assert np.allclose(np.linalg.inv(aa)@bb,np.linalg.inv(a)@b)
    assert np.isclose(np.linalg.norm(aa[:3,3]-center),np.linalg.norm(a[:3,3]-center))
    assert np.allclose(a[:3,3],[.3,.1,.8]), 'Input target was mutated'
    print('PASS: tilt rejection, rate/angle bounds, yaw following, inactive freeze, rigid pair transform')


if __name__=='__main__':check()
