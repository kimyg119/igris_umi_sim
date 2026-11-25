"""Measured receiving support: sum signed native wrenches over contact patches."""
import numpy as np


def final_support_evidence(stamps,events,destination_geoms,mass):
    stamps=np.asarray(stamps,dtype=float)
    if (stamps.ndim!=2 or stamps.shape[1]!=2 or not len(stamps) or not np.isfinite(stamps).all()
            or not np.isfinite(mass) or mass<=0 or np.any(np.diff(stamps,axis=0)<0)):
        raise ValueError('Finite monotonic native pose samples and positive mass required')
    selected=stamps[stamps[:,1]>=stamps[-1,1]-1.];forces=np.zeros(len(selected))
    for e in events:
        if not destination_geoms.intersection(e['geoms']):continue
        wall=float(e['wall']);slot=int(np.searchsorted(selected[:,0],wall))
        candidates=[i for i in (slot-1,slot) if 0<=i<len(selected)]
        i=min(candidates,key=lambda i:abs(selected[i,0]-wall))
        if abs(selected[i,0]-wall)>1e-6:continue
        force=float(e['force_on_payload_world_N'][2])
        if not np.isfinite(force):raise ValueError('Nonfinite native support wrench')
        forces[i]+=force
    fraction=float(np.mean(forces>.5*mass*9.81));duration=float(np.ptp(selected[:,1]))
    return dict(native_duration_s=duration,
        supported_fraction=fraction,samples=len(selected),minimum_force_z_N=float(forces.min()),
        maximum_force_z_N=float(forces.max()),mean_force_z_N=float(forces.mean()),payload_weight_N=mass*9.81,
        start_wall=float(selected[0,0]),end_wall=float(selected[-1,0]),
        semantics='Sum signed world-Z force on payload from receiver contacts; final1native second; at least95percent of frames above half weight')|{'pass':bool(duration>=.9 and fraction>=.95)}
