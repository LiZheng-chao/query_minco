"""Explicit native-motor initial state;not a ground spin-up or dynamics modification."""
import math
def equilibrium_rpm(mass,gravity,kf):
 if not all(math.isfinite(v) and v>0 for v in [mass,gravity,kf]):raise ValueError('PHYSICS_CONSTANT')
 return math.sqrt(mass*gravity/(4*kf))
def initial_motor_rpm(mode,mass,gravity,kf):
 if mode=='zero':return 0.
 if mode=='equilibrium_hover':return equilibrium_rpm(mass,gravity,kf)
 raise ValueError('INITIAL_MOTOR_STATE')
