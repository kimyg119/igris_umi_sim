import numpy as np
import pytest
from segment_delivery import enqueue_segments


def segment(i):
    point=lambda j:np.full(14,j*.001).tolist()
    return dict(start_time=1+i/30,end_time=1+(i+1)/30,previous=point(i),target=point(i+1),
                control_points=[point(j) for j in range(i-2,i+2)],accepted=True,prior_status='used')


def test_producer_burst_keeps_every_continuous_segment():
    first=segment(0);third=segment(2)
    packet=dict(third,segments=[first,segment(1),third])
    pending,last=enqueue_segments([first],first,packet,.99)
    assert [x['start_time'] for x in pending]==[1,1+1/30,1+2/30]
    assert last==third
    again,last2=enqueue_segments(pending,last,packet,1.)
    assert again==pending and last2==last


def test_missing_or_late_or_noncontinuous_segments_still_reject():
    first=segment(0)
    with pytest.raises(RuntimeError,match='sequence'):enqueue_segments([first],first,segment(2),.99)
    with pytest.raises(RuntimeError,match='Late'):enqueue_segments([first],first,segment(1),1.05)
    broken=segment(1);broken['control_points'][0][0]+=.1
    with pytest.raises(RuntimeError,match='spline'):enqueue_segments([first],first,broken,.99)
    bad=segment(1);bad['prior_status']='disabled'
    with pytest.raises(RuntimeError,match='Unvalidated'):enqueue_segments([first],first,bad,.99)
