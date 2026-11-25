import numpy as np
import pytest
from walking_transfer import WalkingTransfer
from test_walking_transfer import fixture
from transport_trial import run


def test_explicit_measured20mm_clearance_keeps_force_speed_and_dwell_gates():
    _,p,t,b,h=fixture()
    ctl=WalkingTransfer([1.16,-.26,.758],-np.pi/4,.1,4.,clearance_lift=.04,
        heading_preserving_route=True,clearance_minimum_rise=.02)
    ctl.start(0.,p,t,b,h)
    for stamp in np.arange(.02,9.,.02):
        b[2]=.85+.019
        s=ctl.update(float(stamp),p,t,b,[6,0,6,0],0.)
    assert s['phase']=='clearance'
    b[2]=.85+.0235
    for stamp in np.arange(9.,10.,.02):
        s=ctl.update(float(stamp),p,t,b,[6,0,6,0],.981)
    assert s['phase']=='clearance' and s['clearance_ready_s']==0
    for stamp in np.arange(10.,10.44,.02):
        s=ctl.update(float(stamp),p,t,b,[6,0,6,0],0.)
    assert s['phase']=='clearance'
    for stamp in np.arange(10.44,11.,.02):
        s=ctl.update(float(stamp),p,t,b,[6,0,6,0],0.)
    assert s['phase']=='carry' and s['clearance_minimum_rise_m']==.02
    ordinary=WalkingTransfer([1.16,-.26,.758],-np.pi/4,.1,4.,clearance_lift=.04,heading_preserving_route=True)
    b[2]=.85;ordinary.start(0.,p,t,b,h);b[2]+=.0235
    for stamp in np.arange(.02,12.,.02):s=ordinary.update(float(stamp),p,t,b,[6,0,6,0],0.)
    assert s['phase']=='clearance'


@pytest.mark.parametrize('minimum',[0.,-.01,.04,float('nan')])
def test_clearance_override_is_bounded(minimum):
    with pytest.raises(ValueError):WalkingTransfer([1,0,.75],0,.1,4.,clearance_lift=.04,
        heading_preserving_route=True,clearance_minimum_rise=minimum)


def test_cli_rejects_clearance_override_outside_reviewed_relaxed_profile(tmp_path):
    with pytest.raises(ValueError,match='minimum clearance'):
        run(tmp_path/'unused',tmp_path/'missing',clearance_minimum_rise=.02)
    assert not(tmp_path/'unused').exists()
