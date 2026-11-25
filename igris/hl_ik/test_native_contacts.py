"""Compare native selection against exhaustive SciPy sphere distances."""
import numpy as np
from scipy.spatial.distance import cdist


def test_native_contacts_match_exhaustive_distances():
    from igris.hl_ik.native_contacts import NativeContacts
    kernel=NativeContacts();rng=np.random.default_rng(491)
    pairs=np.array([[0,1],[0,2],[1,2]],dtype=np.int32)
    for sample in range(100):
        points=[rng.normal(size=(n,3))*.08 for n in (15,23,7)]
        radii=[rng.uniform(.005,.03,len(p)) for p in points]
        thresholds=np.array([0.,.02,.15])
        count=1 if sample%2 else 4
        actual=kernel.select(points,radii,pairs,thresholds,count)
        expected=[]
        for k,(a,b) in enumerate(pairs):
            gaps=cdist(points[a],points[b])-radii[a][:,None]-radii[b][None,:]
            selected=np.flatnonzero(gaps.ravel()<thresholds[k])
            selected=selected[np.argsort(gaps.ravel()[selected])[:count]]
            for index in selected:
                i,j=np.unravel_index(index,gaps.shape)
                expected.append((k,int(i),int(j),float(gaps[i,j])))
        assert actual is not None
        assert [r[:3] for r in actual]==[r[:3] for r in expected]
        np.testing.assert_array_equal([r[3] for r in actual],[r[3] for r in expected])
        first=kernel.select(points,radii,pairs,thresholds,count,stop_at_first=True)
        assert first==expected[:1]


def test_ties_request_the_original_numpy_ordering():
    from igris.hl_ik.native_contacts import NativeContacts
    kernel=NativeContacts()
    points=[np.array([[0.,0,0],[0,1,0]]),np.array([[1.,0,0],[1,1,0]])]
    assert kernel.select(points,[np.zeros(2),np.zeros(2)],np.array([[0,1]]),np.array([2.]),1) is None


def test_disjoint_bodies_have_no_contacts():
    from igris.hl_ik.native_contacts import NativeContacts
    kernel=NativeContacts()
    assert kernel.select([np.zeros((1,3)),np.ones((1,3))*10],[np.ones(1),np.ones(1)],
                         np.array([[0,1]]),np.array([.02]),4)==[]
