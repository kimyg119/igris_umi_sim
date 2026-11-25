import numpy as np
import pytest

from test_walking_transfer import fixture
from walking_transfer import WalkingTransfer


def test_later_braking_keeps_move_then_coast_without_relaxing_arrival():
    _,p,t,b,h=fixture()
    ctl=WalkingTransfer([1.16,-.26,.758],-np.pi/4,.1,4.,heading_preserving_route=True,
                        arrival_settle_seconds=.3,arrival_brake_radius=.02)
    ctl.start(0.,p,t,b,h);ctl.route.index=2;ctl.goal_xy=ctl.route.goal
    for pose in (p,t,b):pose[0]+=.86
    ctl.last_xy=p[:2].copy();ctl.velocity[:]=[.15,0.]
    state=ctl.update(.02,p,t,b,[6,0,6,0],0.)
    assert state['command_xyz'][0]>0 and not state['arrival_coasting']
    for pose in (p,t,b):pose[0]+=.021
    state=ctl.update(.04,p,t,b,[6,0,6,0],0.)
    assert not np.any(state['command_xyz']) and state['arrival_coasting']
    assert state['phase']=='carry' and state['release_fraction']==0.
    # A40mm stopped position remains outside the original35mm arrival gate.
    for pose in (p,t,b):pose[0]-=.021
    for stamp in np.arange(.06,2.,.02):state=ctl.update(float(stamp),p,t,b,[6,0,6,0],0.)
    assert state['phase']=='carry' and state['settled_s']==0.


@pytest.mark.parametrize('radius',[0.,-.02,.01,.026,float('nan')])
def test_braking_override_is_bounded(radius):
    with pytest.raises(ValueError,match='brak'):
        WalkingTransfer([1.,0.,.75],0.,.1,4.,heading_preserving_route=True,arrival_brake_radius=radius)


def test_braking_override_requires_heading_route():
    with pytest.raises(ValueError,match='brak'):
        WalkingTransfer([1.,0.,.75],0.,.1,4.,arrival_brake_radius=.02)
