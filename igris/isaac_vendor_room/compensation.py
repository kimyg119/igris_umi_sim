"""Bounded motor-space compensation for measured passive backlash angles."""
import numpy as np


def compensate(reference,offset,previous,dt,lo,hi,a,b):
    reference,offset=np.asarray(reference),np.asarray(offset)
    if not np.isfinite([reference,offset]).all() or not 0<dt<=.1:raise ValueError('Invalid compensation input')
    if np.any(reference<lo-1e-8) or np.any(reference>hi+1e-8) or np.any(a@reference>b+1e-8):raise ValueError('Reference outside motor limits')
    direction=-offset
    # Project along the correction ray into the same convex vendor safe set.
    normals=np.vstack([np.eye(len(reference)),-np.eye(len(reference)),a]);bounds=np.r_[hi,-lo,b]
    slope=normals@direction;active=slope>1e-12
    alpha=min(1.,float(np.min(np.maximum(0,bounds[active]-normals[active]@reference)/slope[active]))) if active.any() else 1.
    result=reference+alpha*direction
    if previous is not None:
        previous=np.asarray(previous)
        if np.any(normals@previous>bounds+1e-8):raise ValueError('Previous motor command outside limits')
        delta=result-previous;scale=min(1.,.5*dt/max(float(np.max(abs(delta))),1e-12));result=previous+scale*delta
    if np.any(normals@result>bounds+1e-8):raise RuntimeError('Compensation exceeded vendor limits')
    return result


if __name__=='__main__':
    lo=np.full(2,-1.);hi=np.ones(2);a=np.array([[1.,1.]]);b=np.array([.9]);ref=np.array([.4,.4])
    q=compensate(ref,[-.2,-.1],None,.01,lo,hi,a,b);assert abs(q.sum()-.9)<1e-10
    q=compensate(ref,[-.2,-.1],ref,.01,lo,hi,a,b);assert np.max(abs(q-ref))<=.005+1e-10 and q.sum()<=.9
    np.testing.assert_array_equal(compensate(ref,[0.,0.],ref,.01,lo,hi,a,b),ref)
    try:compensate(ref,[np.nan,0],ref,.01,lo,hi,a,b)
    except ValueError:pass
    else:raise AssertionError('Non-finite offset accepted')
    print('PASS: zero correction, coupled limits, 0.5 rad/s bound, invalid offset rejection')
