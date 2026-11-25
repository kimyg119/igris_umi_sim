"""Measured source departure for a heading-preserving, forward cart approach.

Only navigation goals are emitted. No model, joint or object state is changed.
Intermediate goals advance on measured clearance planes while moving. Only
the final placement goal requires the caller's continuous settled-state gate.
"""
import numpy as np


class HeadingRoute:
    def __init__(self,start_xy,final_xy,heading):
        self.start=np.asarray(start_xy,dtype=float).copy()
        final=np.asarray(final_xy,dtype=float).copy()
        if self.start.shape!=(2,) or final.shape!=(2,) or not np.isfinite([*self.start,*final,heading]).all():
            raise ValueError('Finite planar route and heading required')
        c,s=np.cos(heading),np.sin(heading)
        self.rotation=np.array([[c,-s],[s,c]])
        local=self.rotation.T@(final-self.start)
        if not .5<=local[0]<=1.2 or abs(local[1])>.15:
            raise ValueError('Final root goal is outside reviewed forward corridor')
        self.waypoints=[self.start+self.rotation@p for p in (np.array([0.,.085]),np.array([.4,.085]))]+[final]
        self.index=0
        self.minimum_departure=.065

    @property
    def goal(self):return self.waypoints[self.index].copy()

    @property
    def final(self):return self.index==2

    def advance(self,position,*,settled):
        # `settled` is retained for callers of v1, but does not gate transit.
        position=np.asarray(position,dtype=float)
        if position.shape!=(2,) or not np.isfinite(position).all():raise ValueError('Finite measured root position required')
        if self.final:return False
        local=self.rotation.T@(position-self.start)
        if not self.minimum_departure<=local[1]<=.20:return False
        if self.index==0:
            if abs(local[0])>.045:return False
            self.waypoints[1]=self.start+self.rotation@np.array([.4,max(.085,local[1])])
        elif not .35<=local[0]<=.45:return False
        self.index+=1
        return True

    def clearance_recovery_goal(self,position):
        local=self.rotation.T@(np.asarray(position)-self.start)
        if self.index==1 and local[1]<self.minimum_departure:
            # Regaining the lateral corridor takes priority over moving ahead.
            return self.start+self.rotation@np.array([local[0],.085])
        return None

    def evidence(self):
        return dict(schema='measured_heading_route/v2',waypoint_index=self.index,final_waypoint=self.final,
                    start_xy=self.start.tolist(),waypoints_xy=[p.tolist()for p in self.waypoints],
                    minimum_measured_departure_m=self.minimum_departure,minimum_measured_source_forward_m=.35,
                    intermediate_settling_required=False)
