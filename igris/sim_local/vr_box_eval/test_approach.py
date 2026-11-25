"""Waypoint pose, final grasp invariance, and continuity in a tilted base frame."""
import numpy as np
from scipy.spatial.transform import Rotation
from igris_ik.geometry import Pose
from types import SimpleNamespace
from evaluate import virtual,SQUEEZE_M


def check():
    base=(np.array([.03,-.02,.86]),Rotation.from_euler('xyz',[.05,.08,.15]))
    zero=SimpleNamespace(**{s:Pose([.25,sign*.2,.4],[1,0,0,0]) for s,sign in [('left',1),('right',-1)]})
    land=np.array([[[0,.2,.5],[.15,.25,.4],[.25,.2,.4]],[[0,-.2,.5],[.15,-.25,.4],[.25,-.2,.4]]])
    closed=np.array([[.3,.08,.08],[.3,-.08,.08]]);quats=[Rotation.from_euler('y',.4).as_quat()]*2
    turn=Rotation.from_euler('z',-np.pi/4)
    def world(packet,s):return base[0]+base[1].apply(packet[s]['pos'])
    middle,_=virtual(15,zero,land,closed,base,quats)
    opened,_=virtual(19,zero,land,closed,base,quats)
    pressed,_=virtual(23,zero,land,closed,base,quats)
    for i,s in enumerate(['left','right']):
        sign=1 if i==0 else -1
        expected=base[0]+base[1].apply(closed[i])+[0,sign*(SQUEEZE_M+.02),0]
        np.testing.assert_allclose(world(opened,s),turn.apply(expected),atol=1e-9)
        np.testing.assert_allclose(world(middle,s),turn.apply(expected+[.05,sign*.08,.06]),atol=1e-9)
        np.testing.assert_allclose(world(pressed,s),turn.apply(base[0]+base[1].apply(closed[i])),atol=1e-9)
        for t in [11,15,19,23]:
            a,_=virtual(t-1e-5,zero,land,closed,base,quats);b,_=virtual(t+1e-5,zero,land,closed,base,quats)
            assert np.linalg.norm(world(a,s)-world(b,s))<1e-6
            assert (Rotation.from_quat(a[s]['quat']).inv()*Rotation.from_quat(b[s]['quat'])).magnitude()<1e-6
        expected_q=turn*base[1]*Rotation.from_quat(quats[i])
        actual_q=base[1]*Rotation.from_quat(middle[s]['quat'])
        assert (expected_q.inv()*actual_q).magnitude()<1e-9
    print('PASS: wide waypoint, orientation staging, world-frame opening and unchanged squeeze')

if __name__=='__main__':check()
