"""The explicit10g goal-hold experiment must use actual mass in grasp guards."""
import pytest
from htd_loaded_setup import validate_options


def options(mass, goal_hold):
    return dict(live_model='learned.pt',walk_to_rolltainer=True,inplace_transfer=False,
                force_grasp=True,relaxed_grasp=True,payload_mass=mass,mu=4.,
                carry_clearance_lift=.05 if goal_hold else .04,
                simulation_rate_limit=.5,simulation_rate_start_phase=90.,goal_hold=goal_hold)


def test_ten_gram_payload_is_allowed_only_in_explicit_goal_hold():
    validate_options(**options(.01,True))
    validate_options(**options(.1,True))
    validate_options(**options(.1,False))
    with pytest.raises(ValueError):validate_options(**options(.01,False))
    for mass in [0.,.001,.05,1.]:
        with pytest.raises(ValueError):validate_options(**options(mass,True))


def test_prepared_payload_mass_mismatch_is_rejected_before_launch(tmp_path):
    import json
    from transport_trial import validate_prepared_payload
    (tmp_path/'manifest.json').write_text(json.dumps({'payload':{'mass_kg':.01,'body':'box','geom':'box_geom'}}))
    xml='<mujoco><worldbody><body name="box"><freejoint/><geom name="box_geom" type="box" size=".075 .075 .075" mass=".1"/></body></worldbody></mujoco>'
    (tmp_path/'scene.xml').write_text(xml)
    with pytest.raises(ValueError,match='mass'):validate_prepared_payload(tmp_path,.01)
    (tmp_path/'scene.xml').write_text(xml.replace('mass=".1"','mass=".01"'))
    assert validate_prepared_payload(tmp_path,.01)['native_geom_mass_kg']==.01
    with pytest.raises(ValueError,match='mass'):validate_prepared_payload(tmp_path,.1)
