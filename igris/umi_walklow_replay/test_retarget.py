import numpy as np
import pytest

from igris.umi_walklow_replay import retarget
from igris.umi_walklow_replay.kinematics import rotation_vector


def facing_palms(count=40, separation=.27):
    """Two palms facing each other across a box whose long axis is pelvis +y."""
    t = np.arange(count) / 15.
    quaternion = [0., 0., 0., 1.]                       # EE axes = pelvis axes: left palm -y, right palm +y
    left = np.array([[.4, separation / 2, .2, *quaternion]] * count)
    right = np.array([[.4, -separation / 2, .2, *quaternion]] * count)
    return dict(t=t, left=left, right=right)


def config(**changes):
    out = {**retarget.DEFAULT, 'palm_normal_correction': False, **changes}
    return out


def test_free_motion_keeps_clear_of_the_recorded_palm_distance():
    episode = facing_palms()
    cfg = config(object_centric=False, clearance_m=.008)
    left = retarget.ee_targets(episode['left'], 'left', cfg); right = retarget.ee_targets(episode['right'], 'right', cfg)
    gap = retarget.contact_points(left, 'left', cfg) - retarget.contact_points(right, 'right', cfg)
    assert np.allclose(np.linalg.norm(gap, axis=1), .27 + 2 * .008)


def test_object_centric_makes_palms_parallel_at_box_length():
    episode = facing_palms(separation=.29)
    # Tilt the recorded right palm by 15 deg, as the recording error does.
    half = np.deg2rad(15) / 2
    episode['right'][:, 3:] = [np.sin(half), 0., 0., np.cos(half)]
    segments = [dict(pick_index=15, place_index=30)]
    cfg = config(squeeze_m=.01, box_length_m=.27, clearance_m=.008, grasp_turn_in_s=.6)
    left, right = retarget.bimanual_targets(episode, segments, cfg)
    # After the set-down each palm is back on its own recording, clear of the box.
    assert np.allclose(left[-1], retarget.ee_targets(episode['left'], 'left', cfg)[-1])
    released = retarget.contact_points(left, 'left', cfg)[-1] - retarget.contact_points(right, 'right', cfg)[-1]
    assert np.linalg.norm(released) > .29
    held = 22
    assert np.linalg.norm(rotation_vector(left[held, :3, :3].T @ right[held, :3, :3])) < 1e-9
    gap = retarget.contact_points(left, 'left', cfg)[held] - retarget.contact_points(right, 'right', cfg)[held]
    assert np.linalg.norm(gap) == pytest.approx(.27 - 2 * .01)
    # Free motion keeps the recording: the first frame is untouched by the pairing.
    alone = retarget.ee_targets(episode['right'], 'right', cfg)
    assert np.allclose(right[0], alone[0]) and not np.allclose(right[held], alone[held])


def test_pair_attitude_is_taken_up_before_the_position():
    episode = facing_palms(count=70)
    half = np.deg2rad(15) / 2
    episode['right'][:, 3:] = [np.sin(half), 0., 0., np.cos(half)]      # the right palm arrives tilted
    fps = 15.; pick = 45
    cfg = config(squeeze_m=.01, box_length_m=.27, clearance_m=.008, grasp_blend_in_s=.6, grasp_turn_in_s=1.5)
    left, right = retarget.bimanual_targets(episode, [dict(pick_index=pick, place_index=60)], cfg)
    alone = retarget.ee_targets(episode['right'], 'right', cfg)
    early = pick - int(round(1. * fps))                                   # turning already, not yet closing in
    turned = np.linalg.norm(rotation_vector(alone[early, :3, :3].T @ right[early, :3, :3]))
    assert 0 < turned < np.linalg.norm(rotation_vector(alone[pick, :3, :3].T @ right[pick, :3, :3]))
    # The palm stays where it was recorded (only its 8 mm clearance turns with it).
    assert np.allclose(retarget.contact_points(right, 'right', cfg)[early], retarget.contact_points(alone, 'right', cfg)[early], atol=1e-3)
    # Both hands share the attitude at the grasp and the turn in the last 0.6 s is the smaller part.
    assert np.linalg.norm(rotation_vector(left[pick, :3, :3].T @ right[pick, :3, :3])) < 1e-9
    late = pick - int(round(.6 * fps))
    whole = np.linalg.norm(rotation_vector(alone[pick, :3, :3].T @ right[pick, :3, :3]))
    assert np.linalg.norm(rotation_vector(right[late, :3, :3].T @ right[pick, :3, :3])) < .6 * whole    # most of it is done by then
    assert np.allclose(right[0], alone[0])                                # free motion keeps the recording


def test_palm_correction_aligns_measured_axis_with_y():
    for side in ('left', 'right'):
        correction = retarget.palm_correction(side)
        measured = np.array(retarget.MEASURED_BOX_AXIS[side]); measured /= np.linalg.norm(measured)
        assert np.allclose(correction @ [0, 1, 0], measured, atol=1e-9)


def test_resample_takes_the_short_way_and_validates_time():
    t = np.array([0., 1.])
    a = np.eye(4); b = np.eye(4)
    b[:3, :3] = retarget.axis_rotation(np.array([0., 0., 1.]), np.deg2rad(170))
    times, poses = retarget.resample(t, np.array([a, b]), 10.)
    angles = [np.linalg.norm(rotation_vector(p[:3, :3])) for p in poses]
    assert len(times) == 11 and np.all(np.diff(angles) > 0) and angles[5] == pytest.approx(np.deg2rad(85))
    with pytest.raises(ValueError):
        retarget.resample(np.array([0., 0.]), np.array([a, b]), 10.)


def test_grip_height_shift_centres_every_carry():
    # Three boxes of 0.15 m taken top first, held 10.7, 4.2 and 7.4 cm above their bottoms.
    segments = [dict(pick_mid=np.array([0., 0., level * .15 + grip]))
                for level, grip in ((2, .107), (1, .042), (0, .074))]
    shifts = retarget.grip_height_shifts(segments, .15)
    grips = np.array([.107, .042, .074]) + shifts
    assert np.allclose(grips, grips.mean())
    assert retarget.grip_height_shifts(segments[:1], .15).tolist() == [0.]
    # A pick detected in mid-air must not drag the hands down by more than half a box.
    segments[1]['pick_mid'] = np.array([0., 0., .48])
    assert abs(retarget.grip_height_shifts(segments, .15)[1]) <= .075


def test_toe_in_turns_the_fingers_inward_about_the_contact_point():
    from igris.umi_walklow_replay.retarget import DEFAULT, toe_in, toe_in_transform
    config = dict(DEFAULT, toe_in_deg=12.)
    poses = np.repeat(np.eye(4)[None], 3, axis=0); poses[:, :3, 3] = [[.4, .15, .2], [.4, .15, .3], [.5, .1, .2]]
    for side, inward in (('left', -1.), ('right', 1.)):
        turned = toe_in(poses, side, config)
        contact = np.r_[DEFAULT['contact'][side], 1.]
        assert np.allclose(turned @ contact, poses @ contact)                      # the palm contact point stays put
        fingers = -turned[:, :3, 2]                                               # fingers run along -z of the EE frame
        assert np.allclose(np.degrees(np.arcsin(inward * fingers[:, 1])), 12.)     # toward the palm normal: both converge
    assert np.array_equal(toe_in(poses, 'left', DEFAULT), poses)                   # zero by default
    assert np.allclose(toe_in_transform('left', 0., DEFAULT['contact']['left']), np.eye(4))
