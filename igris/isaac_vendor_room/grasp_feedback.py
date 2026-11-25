"""Bounded EE preload adaptation from native payload contact forces only."""
import numpy as np


class GraspFeedback:
    def __init__(self,mass,mu,target=14.,gain=.0002,rate=.0015,travel=.015,synchronized=False,tip_min_force=.2):
        if not np.isfinite([mass,mu,target,tip_min_force]).all() or min(mass,mu,target)<=0 or tip_min_force<0:
            raise ValueError('Invalid mass, friction or grasp force requirements')
        self.required=1.5*mass*9.81/(2*mu)
        if target<self.required:raise ValueError('Target is below friction reserve')
        self.target=target;self.gain=gain;self.rate=rate;self.travel=travel
        self.tip_min_force=tip_min_force
        self.offset=np.zeros(2);self.filtered=np.zeros(4);self.held=0.
        self.roll_offset=np.zeros(2)
        self.synchronized=synchronized;self.qualified=False;self.paired_time=0.
        self.commanded_target=3. if synchronized else target

    def update_roll(self,error,dt,active):
        """Slow bounded correction of actual EE local-X rotation under load."""
        error=np.asarray(error,float);active=np.asarray(active,bool)
        if error.shape!=(2,) or active.shape!=(2,) or not np.isfinite(error).all() or not 0<dt<=.1:
            raise ValueError('Invalid measured EE roll error')
        velocity=np.clip(-.8*error,-np.deg2rad(.3),np.deg2rad(.3))
        self.roll_offset=np.clip(self.roll_offset+dt*np.where(active,velocity,0.),-np.deg2rad(4),np.deg2rad(4))
        return self.roll_offset.copy()

    def update(self,forces,dt,active=None):
        force=np.asarray(forces,float)
        if force.shape!=(4,) or not np.isfinite(force).all() or np.any(force<0) or not 0<dt<=.1:
            raise ValueError('Invalid native contact sample')
        self.filtered+=dt/(.15+dt)*(force-self.filtered)
        if self.synchronized and not self.qualified:
            paired=bool(np.all(self.filtered[[0,2]]>=2.))
            self.paired_time=self.paired_time+dt if paired else 0.
            if not paired:self.commanded_target=3.
            elif self.paired_time>=.2:self.commanded_target=min(self.target,self.commanded_target+2.*dt)
        else:self.commanded_target=self.target
        velocity=np.clip(self.gain*(self.commanded_target-self.filtered[[0,2]]),-self.rate,self.rate)
        if active is not None:
            active=np.asarray(active,bool)
            if active.shape!=(2,):raise ValueError('Expected two contact-active flags')
            velocity=np.where(active,velocity,0.)
        self.offset=np.clip(self.offset+dt*velocity,-.01,self.travel)
        supported=bool(np.all(self.filtered[[0,2]]>=self.required) and np.all(self.filtered[[1,3]]>=self.tip_min_force))
        self.held=self.held+dt if supported else 0.
        self.qualified=self.qualified or self.held>=.3
        return self.offset.copy(),self.held>=.3
