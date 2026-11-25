"""An early touching palm must not shove the box before the second palm arrives."""
import numpy as np
from grasp_feedback import GraspFeedback


def check():
    f=GraspFeedback(1.,.6,travel=.024,synchronized=True)
    for _ in range(90):offset,ready=f.update([8.,.3,0.,0.],1/30,active=[True,True])
    assert offset[0]<0 and offset[1]>0, 'Early palm kept pressing instead of waiting for the opposite palm'
    assert not ready
    # Four light contacts alone must not satisfy the unchanged load reserve.
    for _ in range(40):offset,ready=f.update([3.,.3,3.,.3],1/30,active=[True,True])
    assert not ready
    assert 3<f.commanded_target<f.target
    last=offset
    for _ in range(150):
        offset,ready=f.update([14.,.3,14.,.3],1/30,active=[True,True])
        assert np.max(abs(offset-last))<=.0015/30+1e-12;last=offset
    assert ready and f.qualified and f.commanded_target==14.
    # Once supporting a load, losing one contact must not reduce the other target.
    for _ in range(40):offset,ready=f.update([14.,.3,0.,0.],1/30,active=[True,True])
    assert not ready and f.commanded_target==14.
    assert np.max(offset)<=.024
    # Legacy control and load thresholds remain available.
    old=GraspFeedback(1.,.6)
    for _ in range(90):offset,ready=old.update([8.,.3,0.,0.],1/30,active=[True,True])
    assert offset[0]>0 and not ready
    print('PASS: first-palm staging, shared force ramp, original load/rate/travel guards, load target retained')

if __name__=='__main__':check()
