"""Independent geometry, derivative and data-leakage checks for HL-IK."""
import numpy as np
import torch
import mujoco
import mink
from .common import *
from .solver import HLIK
from .network import batch_from_states,canonicalize
from .geometry import SphereGeometry

def test_se3_log_roundtrip():
    for angle in (1e-9,.7,np.pi-1e-5):
        t=np.eye(4);t[:3,:3]=Rotation.from_rotvec(np.array([1,2,3])/np.sqrt(14)*angle).as_matrix();t[:3,3]=[.1,-.2,.3]
        x=mink.SE3.from_matrix(t)
        np.testing.assert_allclose(mink.SE3.exp(x.log()).as_matrix(),t,atol=1e-8)

def test_lm_jacobian_matches_finite_difference():
    solver=HLIK(collision=False);q=np.zeros(31);q[15:29]=[-.4,.5,.1,-1.2,.1,.1,.1,-.4,-.5,-.1,-1.2,-.1,-.1,.1]
    solver.reset(q);base=solver.configuration.q.copy();arm=base[solver.qadr].copy()
    for task in solver.tasks[:2]:
        t=solver.configuration.get_transform_frame_to_world(task.frame_name,'body').as_matrix()
        t[:3,3]+=[.03,-.01,.02];t[:3,:3]=t[:3,:3]@Rotation.from_euler('xyz',[.1,-.05,.2]).as_matrix()
        task.set_target(mink.SE3.from_matrix(t))
    r,j=solver.residual(arm,arm,base);numeric=np.empty_like(j)
    for i in range(14):
        delta=np.zeros(14);delta[i]=1e-6
        numeric[:,i]=(solver.residual(arm+delta,arm,base,jacobian=False)-solver.residual(arm-delta,arm,base,jacobian=False))/(2e-6)
    np.testing.assert_allclose(j,numeric,atol=2e-5,rtol=1e-5)

def test_history_excludes_future_elbow():
    states=torch.randn(20,2,14);states[:,:,3:7]=torch.tensor([1.,0,0,0]);states[:,:,10:]=torch.tensor([1.,0,0,0])
    mean=torch.zeros(2,14);std=torch.ones(2,14);starts=torch.tensor([2,10])
    h,g,y,_=batch_from_states(states,starts,mean,std)
    states[7,:,7:10]+=10
    h2,g2,y2,_=batch_from_states(states,starts,mean,std)
    torch.testing.assert_close(h,h2);torch.testing.assert_close(g,g2)
    assert not torch.equal(y,y2)

def test_quaternion_sign_invariant():
    torch.manual_seed(42)
    h=torch.randn(13,5,2,14);g=torch.randn(13,2,7);y=torch.randn(13,2,7)
    expected=canonicalize(h,g,y)
    for off in (3,10):h[:,:,:,off:off+4]*=torch.where(torch.rand(13,5,2,1)<.5,-1.,1.)
    g[:,:,3:]*=torch.where(torch.rand(13,2,1)<.5,-1.,1.)
    y[:,:,3:]*=torch.where(torch.rand(13,2,1)<.5,-1.,1.)
    for a,b in zip(expected,canonicalize(h,g,y)):torch.testing.assert_close(a,b)

def test_small_reachable_motion():
    solver=HLIK(collision=False);q=np.zeros(31);q[15:29]=[-.4,.5,.1,-1.2,.1,.1,.1,-.4,-.5,-.1,-1.2,-.1,-.1,.1]
    target=q.copy();target[15]+=.005;target[22]+=.005
    solver.set_q(target);goals=np.array([pose7(solver.configuration.get_transform_frame_to_world(n,'body').as_matrix()) for n in EE_NAMES])
    solver.reset(q);result=solver.solve(goals,q)
    assert result.success and result.position_error_m<.001,result
    np.testing.assert_array_equal(result.q[:15],q[:15])

if __name__=='__main__':
    for test in [test_se3_log_roundtrip,test_lm_jacobian_matches_finite_difference,test_history_excludes_future_elbow,test_quaternion_sign_invariant,test_small_reachable_motion]:
        test();print(test.__name__,'PASS')
