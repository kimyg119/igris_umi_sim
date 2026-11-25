"""Pre-room-staging heading guard; recorded102 must stop before any room move."""
import copy
import math

import numpy as np
import pytest

import run


HEADING=-.7885157172889831
# Nearest pre-staging torso rows in transfer_trial{96,100,102}/run/poses.bin.
RECORDED={
    96:[-.18776895600890092,.05621810019520639,.9365158286824963,
        .9308592738768091,-.016264967215705438,.08995573951496405,-.3537575836746493],
    100:[-.0838223191716429,.008800199313837225,.942898896810761,
         .9384359280951681,-.03102922595382227,.07636135666059714,-.3354759890153401],
    102:[-.2702661594227038,-.22341588523720593,.9406883047371445,
         .9785911066405603,-.041972665178336935,.07947901437446345,-.18515082407565592],
}


def manifest():
    return dict(htd_loaded_zero=dict(enabled=True,scope='loaded_transport'),
                walking_transfer=dict(heading_preserving_route=True,target_torso_heading_rad=HEADING),
                room_reference_base_xyyaw=[-.009058234144107231,.005064025248500327,-.020928289015538396])


def pose(yaw):return np.array([.1,-.2,.9,math.cos(yaw/2),0.,0.,math.sin(yaw/2)])


@pytest.mark.parametrize('trial,expected',[(96,.054685532125343994),(100,.09506006761317765)])
def test_recorded_qualified_startups_keep_exact_room_alignment(tmp_path,trial,expected):
    m=manifest();before=copy.deepcopy(m);pelvis=pose(.04)
    result=run.request_room_layout(tmp_path,m,pelvis,RECORDED[trial],pose_native_time=35.)
    assert result['heading_error_rad']==pytest.approx(expected)
    assert result['limit_rad']==.12 and result['pose_native_time']==35.
    assert result['measured_torso_heading_rad']==pytest.approx(HEADING+expected)
    assert m==before
    actual=np.fromstring((tmp_path/'room_layout.request').read_text(),sep=' ')
    reference=m['room_reference_base_xyyaw'];yaw=.04-reference[2]
    c,s=np.cos(yaw),np.sin(yaw)
    np.testing.assert_allclose(actual,[pelvis[0]-c*reference[0]+s*reference[1],
                                     pelvis[1]-s*reference[0]-c*reference[1],yaw],atol=1e-15,rtol=0)
    assert not (tmp_path/'room_layout.request.tmp').exists()


def test_recorded102_fails_before_room_request_or_any_other_mutation(tmp_path):
    m=manifest();before=copy.deepcopy(m)
    with pytest.raises(RuntimeError,match=r'Pre-grasp.*measured=.*reviewed=.*error=.*limit=0.12'):
        run.request_room_layout(tmp_path,m,pose(.3779),RECORDED[102],pose_native_time=34.31333360787005)
    assert not list(tmp_path.iterdir()) and m==before


@pytest.mark.parametrize('scope',[None,'loaded_zero_only','loaded_goal_hold_only','inplace'])
def test_default_and_inplace_staging_do_not_add_heading_requirement(tmp_path,scope):
    m=manifest()
    if scope is None:m.pop('htd_loaded_zero')
    else:m['htd_loaded_zero']['scope']=scope
    assert run.request_room_layout(tmp_path,m,pose(.3),None) is None
    assert (tmp_path/'room_layout.request').exists()


@pytest.mark.parametrize('delta',[-.120001,.120001])
def test_same_point12_heading_limit_on_both_sides(tmp_path,delta):
    with pytest.raises(RuntimeError,match='Pre-grasp'):
        run.request_room_layout(tmp_path,manifest(),pose(0),pose(HEADING+delta),pose_native_time=30.)
    assert not list(tmp_path.iterdir())


def test_yaw_wrap_does_not_reject_equivalent_heading(tmp_path):
    m=manifest();m['walking_transfer']['target_torso_heading_rad']=math.pi-.03
    result=run.request_room_layout(tmp_path,m,pose(0),pose(-math.pi+.03),pose_native_time=30.)
    assert result['heading_error_rad']==pytest.approx(.06)


@pytest.mark.parametrize('bad',[None,[0.]*7,[0.]*6,[0.,0.,.9,float('nan'),0.,0.,0.],
                              [0.,0.,.9,2.,0.,0.,0.]])
def test_invalid_selected_torso_cannot_publish_staging_request(tmp_path,bad):
    with pytest.raises(ValueError,match='torso'):
        run.request_room_layout(tmp_path,manifest(),pose(0),bad,pose_native_time=30.)
    assert not list(tmp_path.iterdir())
