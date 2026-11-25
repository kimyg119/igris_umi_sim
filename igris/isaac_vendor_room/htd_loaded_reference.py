"""Explicit actual-waist ownership for the loaded HTD/HL-IK SIM epoch."""
import numpy as np


def apply_loaded_waist(reference, request, *, enabled):
    q=np.array(reference,dtype=float,copy=True)
    if q.shape!=(31,) or not np.isfinite(q).all():raise ValueError('Finite31 reference required')
    spec=request.get('htd_loaded_waist')
    if spec is None:return q
    if enabled is not True or spec.get('schema')!='htd_loaded_actual_waist/v1':
        raise ValueError('Explicit loaded HTD waist namespace required')
    epoch=spec.get('epoch')
    if type(epoch) is not int or epoch<=0:raise ValueError('Positive loaded ownership epoch required')
    try:
        sample,native,wall,now=[float(spec[k])for k in
            ('native_sample_sim','native_time_s','sample_monotonic','request_monotonic')]
        waist=np.asarray(spec['waist'],dtype=float)
        nominal=np.asarray(request['nominal'],dtype=float)
        measured=np.asarray(request['measured'],dtype=float)
    except (KeyError,TypeError,ValueError)as exc:raise ValueError('Incomplete actual-waist sample')from exc
    if (not np.isfinite([sample,native,wall,now]).all() or sample<0 or wall<0
            or not -1e-9<=native-sample<.01 or not 0<=now-wall<.1):
        raise ValueError('Fresh native actual-waist sample required')
    if (waist.shape!=(3,) or not np.isfinite(waist).all() or nominal.shape!=(31,)
            or measured.shape!=(31,) or not np.isfinite(nominal).all() or not np.isfinite(measured).all()
            or not np.array_equal(nominal[:3],waist) or not np.array_equal(measured[:3],waist)):
        raise ValueError('Actual waist differs between measured and caller nominal')
    q[:3]=waist
    return q
