import json
from pathlib import Path
import numpy as np
import pytest

from rolltainer_staging import alignment_transform


def test_near_selector_and_goal_share_startup_transform():
    from inplace_layout import near_staging_contract,align_near_goal
    manifest=dict(room_bodies=[{'body':n}for n in ('source','payload','dest','parked')],
                  source_support_bodies=['source'],payload={'body':'payload'},destination_bodies=['dest'])
    contract=near_staging_contract(manifest)
    assert contract['transform_bodies']==['source','payload','dest']
    assert contract['fixed_bodies']==['parked'] and contract['transforms_destination']
    transform=alignment_transform([0,0,0],[.1,.2,np.pi/2])
    goal=align_near_goal([.33,.20,.7561],[0,0,0],[.1,.2,np.pi/2])
    np.testing.assert_allclose(goal,(transform@np.array([.33,.20,.7561,1]))[:3])
    assert goal[2]==pytest.approx(.7561)
    manifest['destination_bodies']=['source']
    with pytest.raises(ValueError):near_staging_contract(manifest)


def test_original_asset_stack_configuration_preserves_height_and_discloses_overhang():
    from inplace_layout import load_layout_proposal,DEFAULT_PROPOSAL
    config=load_layout_proposal(DEFAULT_PROPOSAL)
    assert len(config['destination_bodies'])==4
    assert config['support_top_m']==pytest.approx(.6808)
    assert config['support']['full_footprint_supported'] is False
    assert config['support']['com_axis_margins_m']==[.09,.05]
    assert config['support']['middle_box_depth_overhang_m']==.005
    assert config['support']['planned_footprint_overlap_fraction']==pytest.approx(.6631581661687497)
    assert len(config['nominal_room_poses'])==34
    for name,pose in config['nominal_room_poses'].items():
        assert np.isfinite(pose).all() and np.linalg.norm(pose[3:])==pytest.approx(1.)


def test_original_mass_comparison_accepts_only_same_usd_float32_value():
    from inplace_layout import same_authored_mass
    assert same_authored_mass(.15999999642372129,.1599999964237213)
    assert not same_authored_mass(.1599999964237213,float(np.nextafter(np.float32(.16),np.float32(1.))))
