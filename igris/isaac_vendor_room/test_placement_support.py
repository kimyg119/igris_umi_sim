import numpy as np


def scenario():
    stamps=np.column_stack([100.+np.arange(41)*.05,10.+np.arange(41)*.025])
    events=[dict(wall=round(w,9),sim=s,geoms=['payload','receiver'],normal_force_N=.24525,
                 force_on_payload_world_N=[0,0,.24525]) for w,s in stamps for _ in range(4)]
    return stamps,events


def test_light_box_weight_split_over_four_contacts_is_supported():
    from placement_support import final_support_evidence
    stamps,events=scenario()
    r=final_support_evidence(stamps,events,{'receiver'},.1)
    assert r['pass'] and r['supported_fraction']==1
    assert abs(r['minimum_force_z_N']-.981)<1e-12
    assert all(e['normal_force_N']<.5 for e in events)


def test_forearm_or_source_load_cannot_replace_destination_support():
    from placement_support import final_support_evidence
    stamps,events=scenario()
    for event in events:event['geoms']=['payload','forearm']
    r=final_support_evidence(stamps,events,{'receiver'},.1)
    assert not r['pass'] and r['supported_fraction']==0


def test_transient_or_signed_downward_contact_is_not_stable_support():
    from placement_support import final_support_evidence
    stamps,events=scenario()
    assert not final_support_evidence(stamps,events[:8],{'receiver'},.1)['pass']
    for event in events:event['force_on_payload_world_N'][2]*=-1
    assert not final_support_evidence(stamps,events,{'receiver'},.1)['pass']


def test_window_uses_native_time_and_counts_missing_contact_frames():
    from placement_support import final_support_evidence
    stamps,events=scenario()
    assert not final_support_evidence(stamps[-10:],events,{'receiver'},.1)['pass']
    absent=set(stamps[-5:,0]);events=[e for e in events if e['wall']not in absent]
    assert not final_support_evidence(stamps,events,{'receiver'},.1)['pass']
