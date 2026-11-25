"""Bind manipulation guard authorization to the actually applied HL segment.

The latest request may already ask for opening while a previously accepted
lower segment is still executing. Guard phases follow the executing segment,
whose interpolation must equal the target sent to the upper controller.
"""
import numpy as np
from reference_spline import evaluate


def applied_transport_reference(walking, segment, control_time, target):
    phases=('carry','lower','open','done')
    phase=segment.get('walking_phase')
    planned=walking.get('phase')
    if (phase not in phases or planned not in phases
            or phases.index(phase)>phases.index(planned)
            or phases.index(planned)-phases.index(phase)>1
            or segment.get('accepted') is not True or segment.get('prior_status')!='used'):
        raise ValueError('Accepted applied transport phase required')
    start,end=segment['start_time'],segment['end_time']
    request=segment.get('reference_control_time_s',np.nan)
    if (not np.isfinite([start,end,request,control_time]).all() or end<=start
            or not start-1e-6<=control_time<=end+.1
            or not 0<=request<=control_time+1e-6):
        raise ValueError('Fresh applied transport reference timeline required')
    points=np.asarray(segment['control_points'],dtype=float)
    q=np.asarray(target,dtype=float)
    if (points.shape!=(4,14) or q.shape!=(31,) or not np.isfinite(points).all()
            or not np.isfinite(q).all()
            or not np.allclose(q[15:29],evaluate(points,np.clip((control_time-start)/(end-start),0.,1.)),
                               atol=1e-10,rtol=0.)):
        raise ValueError('Upper target differs from the accepted applied spline')
    destination=np.asarray(walking['target_payload_xyz'],dtype=float)
    if destination.shape!=(3,) or not np.isfinite(destination).all():
        raise ValueError('Finite placement target required')
    return dict(transport_phase=phase,phase_target_valid=True,
                placement_target_xyz=destination.copy(),
                placement_done=walking.get('done') is True and phase=='done')
