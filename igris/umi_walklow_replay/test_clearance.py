import numpy as np

from igris.umi_walklow_replay.clearance import clear_supports, cone_envelope

DT = .01
# Contact pad of a hand in its EE frame: 8 cm square, 2 cm thick, hanging below the EE origin by up to 10 cm.
CORNERS = np.array([[x, y, z] for x in (-.04, .04) for y in (-.01, .01) for z in (-.10, .0)])
TABLE = dict(centre=[.5, 0.], yaw=0., half=[.3, .2], top=.77)
MARGIN = .02


def hover(heights, x=.5, y=0.):
    """Level EE targets over (x, y) with the EE origin at the given heights [m]."""
    poses = np.tile(np.eye(4), (len(heights), 1, 1))
    poses[:, 0, 3] = x; poses[:, 1, 3] = y; poses[:, 2, 3] = heights
    return poses


def lowest(poses):
    return poses[:, 2, 3] - .10


def test_hand_too_close_to_the_surface_is_raised_to_the_margin():
    low = hover(np.full(300, .77 + .10 + .005))                   # pad 5 mm above the table
    raised, lift = clear_supports(low, CORNERS, np.zeros(300), [TABLE], DT, margin=MARGIN)
    assert np.allclose(lift, .015) and np.allclose(lowest(raised), .77 + MARGIN)
    assert np.array_equal(raised[:, :3, :3], low[:, :3, :3]) and np.array_equal(raised[:, :2, 3], low[:, :2, 3])


def test_hand_with_room_to_spare_is_left_alone():
    high = hover(np.full(100, .77 + .10 + .05))
    raised, lift = clear_supports(high, CORNERS, np.zeros(100), [TABLE], DT, margin=MARGIN)
    assert np.array_equal(raised, high) and not lift.any()


def test_hand_beside_the_surface_is_left_alone():
    """Next to a box that stands on another box nothing is under the hand."""
    beside = hover(np.full(100, .77 + .10 - .03), x=1.2)
    raised, lift = clear_supports(beside, CORNERS, np.zeros(100), [TABLE], DT, margin=MARGIN)
    assert np.array_equal(raised, beside) and not lift.any()


def test_height_offset_of_the_samples_counts():
    """The replay adds a per-sample height to the targets (surface height, lower pick side)."""
    poses = hover(np.full(100, .10 + .005))                       # 5 mm above the table once 0.77 m is added
    raised, lift = clear_supports(poses, CORNERS, np.full(100, .77), [TABLE], DT, margin=MARGIN)
    assert np.allclose(lift, .015)


def test_tilted_hand_is_judged_by_its_lowest_corner():
    poses = hover(np.full(100, .77 + .10 + .025))                 # level, the pad would have 2.5 cm of air
    angle = np.radians(30.)                                       # turned about y: one edge of the pad comes down
    poses[:, :3, :3] = [[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]]
    low_point = (poses[0, :3, :3] @ CORNERS.T).T[:, 2].min() + poses[0, 2, 3]
    raised, lift = clear_supports(poses, CORNERS, np.zeros(100), [TABLE], DT, margin=MARGIN)
    assert np.allclose(lift, .77 + MARGIN - low_point) and lift[0] > 0


def test_a_short_dip_is_met_by_a_lift_that_comes_and_goes_at_a_bounded_speed():
    heights = np.full(400, .77 + .10 + .05)
    heights[200:210] = .77 + .10 + .004                           # 0.1 s at 4 mm
    raised, lift = clear_supports(hover(heights), CORNERS, np.zeros(400), [TABLE], DT, margin=MARGIN, speed=.05)
    assert np.all(lowest(raised)[200:210] >= .77 + MARGIN - 1e-12)          # the dip itself is cleared
    assert np.abs(np.diff(lift)).max() <= .05 * DT + 1e-12                  # 5 cm/s
    assert lift[170] > 0 and lift[240] > 0 and lift[100] == 0 and lift[300] == 0
    assert np.all(raised[:, 2, 3] >= heights)                               # never lowered


def test_cone_envelope_covers_every_sample_with_the_gentlest_slope():
    deficit = np.zeros(100); deficit[50] = .01; deficit[60] = .004
    lift = cone_envelope(deficit, step=.001)
    assert np.all(lift >= deficit) and np.abs(np.diff(lift)).max() <= .001 + 1e-15
    assert np.isclose(lift[50], .01) and np.isclose(lift[45], .005) and lift[39] == 0.
    assert np.isclose(lift[60], .004) and np.isclose(lift[62], .002)     # the second, smaller need has its own cone
