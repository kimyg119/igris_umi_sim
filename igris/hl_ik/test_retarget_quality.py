"""Numerical checks independent of the sequence optimizer's outputs."""
import numpy as np
import mujoco
from .retarget_quality import Refiner, Distances, differences, fk, retime
from .common import Rotation
from .geometry import SphereGeometry


def test_distance_derivative():
    refiner=Refiner();model=refiner.model;data=refiner.data;q=model.qpos0.copy()
    q[7:10]=[.15,.05,-.07]
    q[22:]=[-.4,.4,.1,-1.,.1,.1,.1,-.3,-.3,-.1,-1.,-.1,-.1,.1]
    fk(model,data,q);contacts=refiner.geometry.contacts(data,.10)
    assert contacts
    # Hold the same sphere pair fixed to avoid the nonsmooth nearest-pair switch.
    for gap,grad,names,indices in contacts[:12]:
        a,b=[refiner.geometry.names.index(n) for n in names]
        def distance(configuration):
            fk(model,data,configuration)
            p=[]
            for j,k in ((a,indices[0]),(b,indices[1])):
                body=refiner.geometry.ids[j]
                p.append(data.xpos[body]+data.xmat[body].reshape(3,3)@refiner.geometry.local[j][k])
            return np.linalg.norm(p[0]-p[1])-refiner.geometry.radius[a][indices[0]]-refiner.geometry.radius[b][indices[1]]
        numeric=[]
        for i in range(29):
            step=np.zeros_like(q);step[7+i]=1e-6
            numeric.append((distance(q+step)-distance(q-step))/2e-6)
        np.testing.assert_allclose(grad,numeric,atol=2e-7)


def test_pose_derivative():
    refiner=Refiner();q=refiner.model.qpos0.copy();q[7:]=.05
    fk(refiner.model,refiner.data,q)
    p=refiner.data.xpos[refiner.ids].copy()+.01
    r=refiner.data.xmat[refiner.ids].reshape(-1,3,3).copy()
    e,jac=refiner.pose_objective(q,p,r);numeric=np.empty_like(jac)
    for i in range(29):
        step=np.zeros_like(q);step[7+i]=1e-6
        numeric[:,i]=(refiner.pose_objective(q+step,p,r)[0]-refiner.pose_objective(q-step,p,r)[0])/2e-6
    np.testing.assert_allclose(jac,numeric,atol=2e-6)


def test_temporal_units():
    dt=1/30;t=np.arange(8)*dt
    trajectory=np.repeat((.5*t*t)[:,None],29,axis=1)
    np.testing.assert_allclose((differences(8,1,dt)@trajectory.ravel()).reshape(-1,29)[:,0],(t[1:]+t[:-1])/2)
    np.testing.assert_allclose(differences(8,2,dt)@trajectory.ravel(),1,atol=1e-12)


def test_elbow_hinge_mapping():
    refiner=Refiner();q=refiner.model.qpos0.copy()
    q[22:]=[-.4,.4,.1,-1.,.1,.1,.1,-.3,-.3,-.1,-1.,-.1,-.1,.1]
    fk(refiner.model,refiner.data,q);rot=refiner.data.xmat[refiner.ids].reshape(1,-1,3,3).copy();directions=[]
    for s in ('l','r'):
        points=[refiner.data.xpos[refiner.model.body(s+'_'+n).id] for n in ('shoulder_frame','elbow_bracket','wrist_connector')]
        v=np.diff(points,axis=0);directions.append(v/np.linalg.norm(v,axis=1,keepdims=True))
    mapped=refiner.elbow_frames(np.zeros((1,len(refiner.links),3)),rot,np.array(directions)[None])
    for s in ('l','r'):
        i=refiner.links.index(s+'_elbow_bracket')
        assert Rotation.from_matrix(mapped[0,i].T@rot[0,i]).magnitude()<np.deg2rad(3)


def test_retime_path_and_limits():
    ts=np.arange(91)/30;q=np.zeros((len(ts),36));q[:,3]=1
    q[:,7:]=np.sin(ts[:,None]*np.linspace(3,7,29)[None])
    clip=dict(time=ts,human_positions=np.zeros((len(ts),22,3)),human_quaternions=np.tile([1.,0,0,0],(len(ts),22,1)))
    slowed,result,factor=retime(clip,q)
    assert factor>1
    expected=np.column_stack([np.interp(slowed['source_time'],ts,q[:,j]) for j in range(7,36)])
    np.testing.assert_array_equal(result[:,7:],expected)
    np.testing.assert_allclose(np.diff(slowed['time']),1/30,atol=1e-12)
    velocity=np.diff(result[:,7:],axis=0)*30;accel=np.diff(velocity,axis=0)*30
    assert abs(velocity).max()<=2.8+1e-6
    assert abs(accel).max()<=55+1e-6


def test_cached_collision_parity():
    refiner=Refiner();reference=SphereGeometry(refiner.model);rng=np.random.default_rng(101)
    clear=0;colliding=0
    for i in range(300):
        q=refiner.model.qpos0.copy()
        if i<100:q[7:]=rng.uniform(refiner.model.jnt_range[1:,0],refiner.model.jnt_range[1:,1])
        elif i<200:q[7:]+=rng.normal(0,.01,29)
        else:
            q[22:]=[-.65,.48,-.10,-1.84,.49,.11,.40,-.65,-.48,.09,-1.85,-.48,-.11,.41]
            q[7:]+=rng.normal(0,.01,29)
        fk(refiner.model,refiner.data,q)
        expected=reference.collision(refiner.data) is None
        assert expected==(refiner.geometry.collision(refiner.data) is None)
        clear+=expected;colliding+=not expected
    assert clear>50 and colliding>50


def test_paper_gate_and_source_time_weighting():
    from .retarget_quality_batch import summary
    from copy import deepcopy
    def metrics(frames,valid,error,rejected):
        return dict(frames=frames,valid_fraction=valid,valid_windows=50,position_mean_m=error,orientation_mean_deg=error,
                    direction_mean_deg=error,strict_30mm_20deg_fraction=valid,
                    rejection_counts={k:rejected for k in ('self_collision','corridor_collision','velocity','acceleration')})
    rows=[dict(status='processed',path=str(i),result=dict(time_scale=n/100,baseline=metrics(100,.1,.2,50),
          refined=metrics(100,.8,.02,0),retimed=metrics(n,.8,e,0))) for i,n,e in [(0,100,.02),(1,1000,.04)]]
    config=dict(count=2,truncated=False,scope='paper',split='all')
    result=summary(rows,config)
    assert result['ready_for_full']
    assert abs(result['stages']['retimed']['position_mean_m']-.03)<1e-12
    assert not summary(rows,dict(config,split='test'))['ready_for_full']
    assert not summary(rows,dict(config,truncated=True))['ready_for_full']
    assert not summary(rows[:1],config)['ready_for_full']
    failed=deepcopy(rows);failed[1]=dict(status='failed',path='1',error='fixture conversion failure')
    assert not summary(failed,config)['ready_for_full']


if __name__=='__main__':
    for test in (test_distance_derivative,test_pose_derivative,test_temporal_units,test_elbow_hinge_mapping,test_retime_path_and_limits,test_cached_collision_parity,test_paper_gate_and_source_time_weighting):
        test();print(test.__name__,'PASS')
