"""Pose rows at video frame times must lie between the recorded rows, not jump to the next one."""
import numpy as np

from pose_frames import frames_at


def turn_z(degrees):
    """Quaternion (w, x, y, z) of a turn about z."""
    half = np.radians(degrees) / 2
    return [np.cos(half), 0., 0., np.sin(half)]


def row(wall, sim, *bodies):
    """One recorded row: wall time, simulation time, then position and quaternion per body."""
    return np.array([wall, sim, *np.ravel(bodies)], dtype=float)


def body(x, degrees, sign=1.):
    return [x, 0., .5, *(sign * np.array(turn_z(degrees)))]


def test_a_frame_time_between_two_rows_gets_the_pose_in_between():
    rows = np.array([row(100., 2.00, body(0., 0.), body(1., 0.)),
                     row(100.4, 2.04, body(.3, 90.), body(1., 0.))])
    frame, = frames_at(rows, 1, [2.01])
    assert np.allclose(frame[:2], [100.1, 2.01])                   # both clocks
    assert np.allclose(frame[2:9], body(.075, 22.5))               # a quarter of the way, position and turn
    assert np.allclose(frame[9:], body(1., 0.))                    # the body that stood still


def test_frame_times_on_recorded_rows_return_those_rows():
    rows = np.array([row(0., 0., body(0., 0.)), row(1., .03, body(.1, 20.)), row(2., .07, body(.4, -30.))])
    assert np.allclose(frames_at(rows, 1, [0., .03, .07]), rows)
    assert np.allclose(frames_at(rows, 0, [1.]), rows[1:2])        # the wall clock works the same way


def test_orientation_takes_the_shortest_turn_when_the_recorded_sign_flips():
    rows = np.array([row(0., 0., body(0., 0.)), row(1., 1., body(0., 10., sign=-1.))])      # -q is the same orientation
    frame, = frames_at(rows, 1, [.5])
    quaternion = frame[5:9]
    assert np.isclose(np.linalg.norm(quaternion), 1.)
    assert np.allclose(quaternion, turn_z(5.)) or np.allclose(quaternion, -np.array(turn_z(5.)))


def test_evenly_spaced_frames_move_evenly_when_the_recording_is_not_evenly_spaced():
    stamps = np.array([0., .033, .067, .105, .141, .166, .204, .233, .268, .31])    # recorded at about 30 Hz
    rows = np.array([row(t, t, body(2. * t, 90. * t)) for t in stamps])             # constant speed and turn rate
    frames = frames_at(rows, 1, [0., .1, .2, .3])                                    # 10 frames per second
    assert np.allclose(np.diff(frames[:, 2]), .2)                                    # the next rows give 0.21, 0.198, 0.212
    turned = 2 * np.degrees(np.arctan2(frames[:, 8], frames[:, 5]))
    assert np.allclose(np.diff(turned), 9.)


def test_a_frame_time_outside_the_recording_holds_the_end_pose():
    rows = np.array([row(0., 1., body(0., 0.)), row(1., 2., body(1., 40.))])
    assert np.allclose(frames_at(rows, 1, [.5, 2.5]), rows)
