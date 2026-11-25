"""C2 uniform cubic reference interpolation for bounded HL-IK command knots."""
import numpy as np


def evaluate(points,u,derivative=0,dt=1/30):
    p=np.asarray(points,float);u=float(np.clip(u,0.,1.))
    if p.shape!=(4,14) or not np.isfinite(p).all():raise ValueError('Four finite arm knots required')
    if derivative==0:w=np.array([(1-u)**3,3*u**3-6*u*u+4,-3*u**3+3*u*u+3*u+1,u**3])/6
    elif derivative==1:w=np.array([-3*(1-u)**2,9*u*u-12*u,-9*u*u+6*u+3,3*u*u])/(6*dt)
    elif derivative==2:w=np.array([6*(1-u),18*u-12,-18*u+6,6*u])/(6*dt*dt)
    else:raise ValueError('Only position, velocity and acceleration are supported')
    return w@p
