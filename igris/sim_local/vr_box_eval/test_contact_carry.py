import numpy as np


def test_rotation_and_corner_landing_cannot_advance_or_wind_up_targets():
    c=ContactCarry(1.,1.2,palm_first=True);c.phase='probe_lift';c.value[2]=.04
    observation=dict(fresh=True,forces=(12,12),patches=(11,1,11,1),box_lift=.016,supported=False)
    for _ in range(9):
        q=c.update(.02,**observation,box_tilt_rad=np.deg2rad(11))
        np.testing.assert_array_equal(q,c.value)
    assert c.phase=='failed' and c.reason=='box_rotation_unstable'
    c=ContactCarry(1.,1.2,palm_first=True);c.phase='lower';c.value[2]=.001
    observation.update(box_lift=.024,supported=True)
    for _ in range(30):
        q=c.update(.02,**observation,box_tilt_rad=np.deg2rad(35))
        assert q[2]==c.value[2], 'Contact must stop downward virtual preload'
    assert c.phase=='lower', 'A zero target height is not an actual flat landing'
    for _ in range(160):c.update(.02,**observation,box_tilt_rad=np.deg2rad(35))
    assert c.phase=='failed' and c.reason=='unstable_landing'
from contact_carry import ContactCarry, PALM_SEAT_LIMIT


def test_raised_distal_force_blocks_lift_until_both_tips_support():
    import pytest
    for value in (0.,-1.,float('nan'),float('inf')):
        with pytest.raises(ValueError):ContactCarry(1.,palm_first=True,tip_force_N=value)
    assert ContactCarry(1.,palm_first=True,tip_force_N=.1).secondary_force_N == .2
    c=ContactCarry(1.,1.2,palm_first=True,tip_force_N=.8)
    c.transition('probe_lift');c.value[0]=19.
    for _ in range(40):
        q=c.update(.02,fresh=True,forces=(15,15),patches=(14,.4,14,.4),box_lift=0.,supported=True)
        c.acknowledge(q,True)
    assert c.value[2]==0 and np.all(c.value[4:6]>0)
    for _ in range(40):
        q=c.update(.02,fresh=True,forces=(15,15),patches=(14,1.,14,1.),box_lift=0.,supported=True)
        c.acknowledge(q,True)
    assert c.value[2]>0


def test_ten_gram_box_keeps_contact_detection_and_preload_margin_separate():
    c=ContactCarry(.01,1.2)
    assert 0 < c.contact_threshold_N < .01*9.81
    c.transition('carry_left');c.value[2]=.08
    c.force[:]=.1;c.patch[:]=(.05,.05,.05,.05)
    q=c.update(.02,fresh=True,forces=(.1,.1),patches=(.05,)*4,box_lift=.08,supported=False)
    assert c.required_force<.1<c.preload_force
    assert c.phase=='carry_left' and q[3]==c.value[3] and q[1]>c.value[1]


def test_palm_contact_and_actual_lift_gate_the_task():
    c=ContactCarry(.5);c.value[0]=19.;c.transition('squeeze')
    def tick(forces=(0,0),patches=(0,0,0,0),lift=0,supported=True,fresh=True):
        p=c.update(.02,fresh=fresh,forces=forces,patches=patches,box_lift=lift,supported=supported)
        c.acknowledge(p,True)
    for _ in range(300):tick()
    assert c.phase=='squeeze' and c.value[2]==0
    for _ in range(60):tick((10,10),(5,0,5,0))
    assert c.phase=='seat_palm'
    preload=c.value[1]
    for _ in range(120):tick((10,10),(5,5,5,5))
    assert c.phase=='probe_lift'
    assert np.all((0 < c.value[4:6]) & (c.value[4:6] <= PALM_SEAT_LIMIT)) and c.value[1]==preload
    for _ in range(20):tick((10,10),(5,5,5,5),lift=.02,supported=False)
    assert c.phase=='lift' and c.value[1]==preload
    for _ in range(220):tick((10,10),(5,5,5,5),lift=.08,supported=False)
    assert c.phase=='carry_left' and c.value[1]==preload
    before=c.value.copy();tick(fresh=False)
    np.testing.assert_array_equal(c.value,before)


def test_broad_pad_cannot_replace_lost_palm_bump():
    c=ContactCarry(.5);c.transition('squeeze')
    for _ in range(100):
        q=c.update(.02,fresh=True,forces=(10,10),patches=(0,10,0,10),box_lift=0,supported=True)
        c.acknowledge(q,True)
    assert c.phase=='squeeze' and c.value[4]==0
    c.transition('carry_left');c.value[2]=.08
    q=c.update(.02,fresh=True,forces=(10,10),patches=(0,10,0,10),box_lift=.08,supported=False)
    assert c.phase=='carry_left' and q[3]==c.value[3] and q[1]==c.value[1]
    c.update(.02,fresh=True,forces=(10,10),patches=(0,10,0,10),box_lift=0,supported=True)
    assert c.phase=='failed' and c.reason=='contact_lost'


def test_lift_waits_for_broad_contact_to_recover():
    c=ContactCarry(.01,1.2);c.transition('lift');c.value[1]=.004;c.value[2]=.06
    c.value[4:6]=1.5
    c.patch[:]=(1.1,.1,1.1,0);c.force[:]=(1.2,1.1)
    q=c.update(.02,fresh=True,forces=(1.2,1.1),patches=c.patch,box_lift=.05,supported=False)
    assert c.phase=='lift' and q[2]==c.value[2] and q[1]==c.value[1]
    assert q[4]==c.value[4] and c.value[5]<q[5]<=PALM_SEAT_LIMIT
    c.value[5]=PALM_SEAT_LIMIT
    capped=c.update(.02,fresh=True,forces=(1.2,1.1),patches=c.patch,box_lift=.05,supported=False)
    np.testing.assert_array_equal(capped,c.value)
    c.acknowledge(q,True);c.patch[:]=(1.1,.1,1.1,.1)
    q=c.update(.02,fresh=True,forces=(1.2,1.2),patches=c.patch,box_lift=.05,supported=False)
    assert q[2]>c.value[2] and q[1]==c.value[1]


def test_probe_holds_height_while_reseating_contact():
    c=ContactCarry(.01,1.2);c.transition('probe_lift');c.value[2]=.005
    c.patch[:]=(1.1,.1,1.1,0);c.force[:]=(1.2,1.1)
    q=c.update(.02,fresh=True,forces=(1.2,1.1),patches=c.patch,box_lift=0,supported=True)
    assert q[2]==c.value[2] and q[5]>c.value[5]


def test_seated_angle_without_broad_contact_cannot_start_lifting():
    c=ContactCarry(.01,1.2);c.transition('seat_palm');c.value[4]=1.
    c.patch[:]=(1.1,0,1.1,0);c.force[:]=1.1
    for _ in range(1010):
        q=c.update(.02,fresh=True,forces=(1.1,1.1),patches=(1.1,0,1.1,0),box_lift=0,supported=True)
        c.acknowledge(q,True)
        assert c.value[2]==0
    assert c.phase=='failed' and c.reason=='broad_palm_not_seated'
    np.testing.assert_array_equal(c.value[4:6],PALM_SEAT_LIMIT)
    assert c.value[1]==-.02


def test_seated_hand_stops_tilting_while_other_hand_seeks_contact():
    c=ContactCarry(.01,1.2);c.transition('seat_palm');c.patch[:]=(1.1,.1,1.1,0);c.force[:]=(1.2,1.1)
    q=c.update(.02,fresh=True,forces=(1.2,1.1),patches=c.patch,box_lift=0,supported=True)
    assert q[4]==0 and q[5]>0 and q[1]==c.value[1]


def test_seating_restores_palm_pressure_before_rotating_further():
    c=ContactCarry(.01,1.2);c.transition('seat_palm')
    c.value[1]=.003;c.value[4:6]=.8
    def tick():
        return c.update(.02,fresh=True,forces=(0,0),patches=(0,0,0,0),box_lift=0,supported=True)
    q=tick()
    assert c.phase=='seat_palm' and c.value[1]<q[1]<=c.ceiling
    np.testing.assert_array_equal(q[4:6],c.value[4:6])
    c.acknowledge(q,False)
    assert c.value[1]==.003
    for _ in range(20):
        q=tick();c.acknowledge(q,True)
    assert c.value[1]>.003
    c.value[1]=c.ceiling
    assert tick()[1]==c.ceiling
    c.phase_age=10.1  # Measured broad contact arrived just after the old 10 s deadline.
    for _ in range(60):
        q=c.update(.02,fresh=True,forces=(1.3,1.3),patches=(1.1,.2,1.1,.2),box_lift=0,supported=True)
        c.acknowledge(q,True)
    assert c.phase=='probe_lift'


def test_seating_undoes_pad_only_tilt_and_counts_shared_support():
    c=ContactCarry(.01,1.2);c.transition('seat_palm');c.value[4:6]=1.5
    c.patch[:]=(0,1.1,1.1,0);c.force[:]=1.1
    q=c.update(.02,fresh=True,forces=(1.1,1.1),patches=c.patch,box_lift=0,supported=True)
    assert q[4]<c.value[4] and q[5]>c.value[5] and q[1]==c.value[1]
    c.patch[:]=.6  # Each hand supplies 1.2 N; neither patch alone supplies the 1 N preload.
    for _ in range(20):
        q=c.update(.02,fresh=True,forces=(1.2,1.2),patches=(.6,)*4,box_lift=0,supported=True)
        c.acknowledge(q,True)
    assert c.phase=='probe_lift'


def test_rejected_preload_never_becomes_the_accepted_target():
    c=ContactCarry(2.);c.transition('squeeze');c.value[1]=.02
    candidate=c.value.copy();candidate[1]=.021;c.acknowledge(candidate,False)
    assert c.value[1]==.02 and c.step_scale<1.
    for _ in range(100):
        q=c.update(.02,fresh=True,forces=(0,0),patches=(0,0,0,0),box_lift=0,supported=True)
        c.acknowledge(q,False)
    assert c.value[1]==.02 and c.phase=='squeeze'
    for _ in range(10):
        q=c.update(.02,fresh=True,forces=(0,0),patches=(0,0,0,0),box_lift=0,supported=True)
        c.acknowledge(q,True)
    assert c.value[1]>.02, 'a temporary IK failure must not permanently cap preload'


def test_cartesian_progress_waits_without_backing_off_valid_motion():
    c=ContactCarry(.01,1.2);proposal=c.value.copy();proposal[0]=1.
    c.acknowledge(proposal,False,progressing=True)
    assert c.value[0]==0 and c.step_scale==1 and c.retry_delay==0
    c.acknowledge(proposal,False)
    assert c.value[0]==0 and c.step_scale==.5 and c.retry_delay>0


def test_auxiliary_contact_limits_preload_but_cannot_qualify_as_seated():
    c=ContactCarry(.01,1.2);c.transition('seat_palm');c.value[4:]=2.
    c.force[:]=3.;c.patch[:]=(0,0,1.5,.5)
    q=c.update(.02,fresh=True,forces=c.force,patches=c.patch,box_lift=0,supported=True)
    assert q[1]==c.value[1] and q[4]<c.value[4] and q[5]==c.value[5]
    assert c.phase=='seat_palm' and q[2]==0
