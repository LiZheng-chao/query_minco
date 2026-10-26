"""Bounded non-Query reference primitives; no ROS publishing capability."""
import numpy as np
from query_minco.inference.constrained_prefix import validate_activation
from query_minco.inference.trajectory_execution import quintic,MotionLimits

def soft_brake(head,limits=MotionLimits(),soft_jerk=.35,max_time=6.):
    h=np.asarray(head,float)
    if h.shape!=(3,3) or not np.isfinite(h).all():raise ValueError('BRAKE_STATE')
    for T in np.arange(2.,max_time+1e-9,.1):
        endpoint=h[0]+h[1]*T/2+h[2]*T*T/12
        p=quintic(h,np.stack([endpoint,np.zeros(3),np.zeros(3)]),T)
        try:p.validate(limits)
        except ValueError:continue
        ex=p.extrema()
        if ex[2]>soft_jerk:continue
        return p,dict(duration_s=float(T),endpoint_displacement_m=float(np.linalg.norm(endpoint-h[0])),
            travel_distance_upper_bound_m=float(ex[0]*T),extrema=ex,soft_jerk=soft_jerk,
            requires_swept_tube=True)
    raise ValueError('NO_SOFT_BOUNDED_BRAKE')

def activation_guard(requested,actual,measurement_stamp,publishers,clock_delta=0.):
    # All failures are latching at the ROS adapter. Never change ROS timestamps.
    if not np.isfinite(clock_delta) or abs(clock_delta)>.005:raise ValueError('CLOCK_JUMP')
    if publishers!=['/f3_reference_arbiter']:raise ValueError('PUBLISHER_CONFLICT')
    if not 0<=actual-measurement_stamp<=.05:raise ValueError('STALE_ODOM')
    return validate_activation(requested,actual)

def cruise_segment(head,duration):
    c=np.zeros((1,3,6));c[0,:,0]=head[0];c[0,:,1]=head[1]
    from query_minco.inference.trajectory_execution import Program
    if np.linalg.norm(head[2])>1e-9:raise ValueError('CRUISE_ACCELERATION')
    return Program(c,np.array([duration]))
