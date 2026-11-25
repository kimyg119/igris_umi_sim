"""Explicit caller waist references shared by SDK publication, IK and audit."""
import numpy as np
from inplace_transfer import _waist


def apply_waist_reference(reference,state):
    q=np.asarray(reference,dtype=float).copy()
    if q.shape!=(31,) or not np.isfinite(q).all():raise ValueError('Expected finite31-axis reference')
    if state is None:return q
    if (state.get('caller_reference_schema')!='explicit_inplace_waist/v1'
            or state.get('caller_sdk_indices')!=[0,1,2]
            or not np.array_equal(state.get('command_xyz'),[0.,0.,0.])):
        raise ValueError('Invalid explicit in-place caller contract')
    waist=_waist(state['waist_target_rad'])
    if not np.array_equal(waist,state.get('nominal_waist_target_rad')):
        raise ValueError('Publisher and IK nominal waist targets differ')
    q[:3]=waist
    return q


def reference_error(actual,archive,state):
    return float(np.max(abs(np.asarray(actual)-apply_waist_reference(archive,state))))
