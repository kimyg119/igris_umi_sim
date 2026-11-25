import pytest

from transport_trial import relaxed_requirements, run


def test_relaxed_default_is_preserved_and_force_target_is_explicit():
    assert relaxed_requirements(.25, 2.) == dict(target_force_N=3., tip_min_force_N=0., friction_safety_factor=1.5)
    assert relaxed_requirements(.1, 4., 6.) == dict(target_force_N=6., tip_min_force_N=0., friction_safety_factor=1.5)


@pytest.mark.parametrize('mass,mu,target', [(1., .6, 6.), (.1, 4., 0.), (.1, 4., 15.),
    (.1, 4., float('nan')), (.1, float('inf'), 6.), (-.1, 4., 6.)])
def test_invalid_force_profile_is_rejected(mass, mu, target):
    with pytest.raises(ValueError, match='grasp'):
        relaxed_requirements(mass, mu, target)


def test_explicit_target_requires_relaxed_profile_before_any_preparation(tmp_path):
    with pytest.raises(ValueError, match='relaxed'):
        run(tmp_path/'unused', tmp_path/'absent_motion', grasp_force_target=6.)
    assert not (tmp_path/'unused').exists()


def test_heading_cart_requires_explicit_walking_profile_before_preparation(tmp_path):
    with pytest.raises(ValueError,match='Heading cart'):
        run(tmp_path/'unused',tmp_path/'absent_motion',heading_cart_proposal=tmp_path/'proposal.json')
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('lift', [.03, .04])
def test_heading_cart_accepts_reviewed30_and40mm_clearance_only(tmp_path, lift):
    # A later independent validation error demonstrates acceptance without
    # invoking scene preparation, a container or physics.
    with pytest.raises(ValueError, match='Invalid torso filter'):
        run(tmp_path/'unused',tmp_path/'absent_motion',heading_cart_proposal=tmp_path/'proposal.json',
            walk_to_rolltainer=True,narrow_destination_support=True,relaxed_grasp=True,
            world_position_lowering=True,torso_relative_lowering=True,payload_mass=.1,mu=4.,
            palm_torsion_m=.02,carry_clearance_lift=lift,pickup_stop_phase=106.,
            grasp_force_target=6.,grasp_height_adjustment=-.04,live_acceleration=3.,
            live_torso_filter_seconds=-1.)
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('change', [dict(carry_clearance_lift=.025),dict(mu=2.),dict(grasp_force_target=3.)])
def test_heading_cart30mm_keeps_remaining_reviewed_profile_guards(tmp_path, change):
    options=dict(heading_cart_proposal=tmp_path/'proposal.json',walk_to_rolltainer=True,
        narrow_destination_support=True,relaxed_grasp=True,world_position_lowering=True,
        torso_relative_lowering=True,payload_mass=.1,mu=4.,palm_torsion_m=.02,
        carry_clearance_lift=.03,pickup_stop_phase=106.,grasp_force_target=6.,
        grasp_height_adjustment=-.04,live_acceleration=3.)
    options.update(change)
    with pytest.raises(ValueError, match='Heading cart'):
        run(tmp_path/'unused',tmp_path/'absent_motion',**options)
