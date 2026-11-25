import numpy as np
import pytest

from heading_route import HeadingRoute


def test_waypoints_rotate_about_actual_pickup_and_use_actual_final_goal():
    route=HeadingRoute([1.,2.],[1.,2.9],np.pi/2)
    np.testing.assert_allclose(route.goal,[.915,2.])
    assert not route.advance([.96,2.],settled=True)  # within45mm, but source not cleared
    assert not route.advance([.94,2.],settled=False)
    assert route.advance([.93,2.],settled=False)
    np.testing.assert_allclose(route.goal,[.915,2.4])
    assert not route.final
    assert route.advance([.915,2.38],settled=False)
    np.testing.assert_allclose(route.goal,[1.,2.9])
    assert route.final
    assert not route.advance([1.,2.9],settled=True)


def test_cannot_skip_source_departure_or_advance_from_distant_pose():
    route=HeadingRoute([0.,0.],[.9,0.],0.)
    assert not route.advance([.4,.085],settled=True)
    assert not route.advance([0.,.04],settled=True)
    assert route.advance([0.,.07],settled=True)
    assert not route.advance([.34,.085],settled=True)
    assert route.advance([.37,.085],settled=True)


def test_moving_plane_crossing_preserves_measured_lateral_offset():
    route=HeadingRoute([0.,0.],[.9,0.],0.)
    assert not route.advance([0.,.064],settled=False)
    assert route.advance([-.004,.11],settled=False)
    np.testing.assert_allclose(route.goal,[.4,.11])
    assert not route.advance([.34,.11],settled=False)
    assert not route.advance([.36,.04],settled=False)
    assert route.advance([.36,.11],settled=False)
    assert route.final


@pytest.mark.parametrize('start,end,heading',[
    ([0.,0.],[.3,0.],0.),([0.,0.],[-.9,0.],0.),
    ([0.,0.],[.9,.3],0.),([0.,0.],[.9,0.],float('nan')),
])
def test_reject_route_outside_reviewed_forward_corridor(start,end,heading):
    with pytest.raises(ValueError):HeadingRoute(start,end,heading)
