"""EE positions anchored in world with orientations following measured torso.

Capture accepted carry goals, including their current preload, exactly once.
This changes target frames only; measured destination support still controls
release in WalkingTransfer, and the live IK worker retains every solver guard.
"""
import numpy as np


def phase_acceleration_limit(base,override,walking):
    if override is not None and (not np.isfinite(override) or not 0<override<=3.):
        raise ValueError('Lowering acceleration must be finite and within (0,3] rad/s²')
    return override if override is not None and walking is not None and walking['phase'] in ('lower','open','done') else base


class WorldPositionLowering:
    def __init__(self,local_hands,torso,delta_world,preload_at_capture):
        self.local=np.asarray(local_hands,dtype=float).copy()
        self.delta=np.asarray(delta_world,dtype=float).copy()
        self.preload=np.asarray(preload_at_capture,dtype=float).copy()
        if self.local.shape!=(2,4,4) or self.delta.shape!=(3,) or self.preload.shape!=(2,3):
            raise ValueError('Invalid lowering capture shapes')
        self.world=np.asarray(torso,dtype=float)[None]@self.local
        if not all(np.isfinite(v).all() for v in (self.local,self.delta,self.preload,self.world)):
            raise ValueError('Nonfinite lowering capture')
        normal=self.local[0,:3,3]-self.local[1,:3,3]
        if np.linalg.norm(normal)<.001:raise ValueError('Coincident lowering hands')
        self.normal=normal/np.linalg.norm(normal)

    def targets(self,torso,fraction,release,preload,*,xy_fraction=None):
        if not np.isfinite([fraction,release]).all() or not 0<=fraction<=1 or not 0<=release<=1:
            raise ValueError('Invalid lowering/opening fraction')
        if xy_fraction is None:xy_fraction=fraction
        if not np.isfinite(xy_fraction) or not 0<=xy_fraction<=1:
            raise ValueError('Invalid horizontal lowering fraction')
        preload=np.asarray(preload,dtype=float)
        if preload.shape!=(2,3) or not np.isfinite(preload).all():raise ValueError('Invalid lowering preload')
        inverse=np.linalg.inv(np.asarray(torso,dtype=float))
        if inverse.shape!=(4,4) or not np.isfinite(inverse).all():raise ValueError('Invalid lowering torso')
        world_positions=self.world[:,:3,3]+np.array([xy_fraction,xy_fraction,fraction])*self.delta
        result=self.local.copy()
        result[:,:3,3]=world_positions@inverse[:3,:3].T+inverse[:3,3]
        result[:,:3,3]+=preload-(1-release)*self.preload
        result[:,:3,3]+=np.array([1.,-1.])[:,None]*.055*release*self.normal
        return result
