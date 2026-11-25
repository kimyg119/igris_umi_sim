"""Receive a bounded burst of validated future IK segments without losing knots."""
import numpy as np


def enqueue_segments(pending, last, message, control_time):
    segments=message.get('segments',[message])
    if not isinstance(segments,list) or not 1<=len(segments)<=16:
        raise RuntimeError('Invalid live HL-IK segment packet')
    pending=list(pending)
    previous_start=-np.inf
    for response in segments:
        start,end=response['start_time'],response['end_time']
        if not np.isfinite([start,end]).all() or end<=start or start<=previous_start:
            raise RuntimeError('Invalid live HL-IK segment timeline')
        previous_start=start
        if not response['accepted'] or response['prior_status']!='used':
            raise RuntimeError('Unvalidated live HL-IK response')
        if last is not None and start<=last['start_time']:continue
        if control_time>start+1e-6:
            raise RuntimeError('Late HL-IK segment would break reference rate bounds')
        before=np.asarray(response['previous']);after=np.asarray(response['target'])
        points=np.asarray(response['control_points'])
        if before.shape!=(14,) or after.shape!=(14,) or points.shape!=(4,14) or not np.isfinite([before,after]).all() or not np.isfinite(points).all():
            raise RuntimeError('Invalid live arm reference')
        if last is not None:
            if abs(start-last['end_time'])>1e-6 or not np.allclose(before,last['target'],atol=1e-9,rtol=0):
                raise RuntimeError('Discontinuous live HL-IK segment sequence')
            if not np.allclose(np.asarray(last['control_points'])[1:],points[:-1],atol=1e-9,rtol=0):
                raise RuntimeError('Discontinuous live HL-IK spline window')
        pending.append(response);last=response
    return pending,last
