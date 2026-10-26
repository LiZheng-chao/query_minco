"""Frozen MINCO lookahead connection + exactly retimed short suffix; no ROS.

The connection replaces the unexecuted original path up to source_phase.
Only the final short suffix follows the original geometry. This is execution
postprocessing, never claimed to be the untouched raw MINCO trajectory.
"""
import time
import numpy as np
from query_minco.inference.trajectory_execution import Program,MotionLimits,Measurement,quintic,trim,concatenate,brake
TEMPLATES=((1.,.6,.4),(1.,.7,.5),(1.25,.6,.4),(1.25,.7,.4),(1.25,.8,.4),
           (1.25,.5,.25),(1.,.6,.25),(1.25,.7,.25),(1.25,.8,.25))
def slowed(program,rate):
    if not np.isfinite(rate) or not 0<rate<=1:raise ValueError('RATE')
    return Program(program.coefficients*rate**np.arange(6)[None,None,:],program.durations/rate)
def constrained_prefix(world,measurement,activation,volume,now,retained_s=.1,active=None,source_stamp=None):
    start=time.perf_counter()
    if source_stamp is None:source_stamp=measurement.stamp
    if not 0<=now-source_stamp<=.25 or not .02<=activation-now<=.1:raise ValueError('TIME_CONTRACT')
    if not 0<=now-measurement.stamp<=.05:raise ValueError('STALE_CURRENT_ODOM')
    if not .8-1e-6<=world.duration<=2.2+1e-6:raise ValueError('SOURCE_HORIZON')
    predicted=measurement.predict(activation)
    head=predicted if active is None else active.pva(activation)
    if (np.max(np.abs(predicted-head),axis=1)>[.02,.05,.1]).any():raise ValueError('PREDICTED_TRACKING_TUBE')
    failures=[]
    for connection_s,source_phase,rate in TEMPLATES:
        if source_phase+retained_s*rate>world.duration:continue
        tail=world.pva(source_phase).copy();tail[1]*=rate;tail[2]*=rate**2
        connection=quintic(head,tail,connection_s)
        retained=slowed(trim(world,source_phase,source_phase+retained_s*rate),rate)
        prefix=concatenate(connection,retained)
        try:
            prefix.validate(MotionLimits())
            stop,stop_bounds=brake(prefix.pva(prefix.duration))
            full=concatenate(prefix,stop).validate(MotionLimits())
        except ValueError as e:
            failures.append(str(e));continue
        query_progress=float(np.linalg.norm(retained.pva(retained.duration)[0]-retained.pva(0)[0]))
        progress=float(np.linalg.norm(prefix.pva(prefix.duration)[0]-head[0]))
        if query_progress<.005 or progress<.05:
            failures.append('INSUFFICIENT_NONZERO_QUERY_PROGRESS');continue
        proof=volume.certify(full.coefficients,full.durations,now,required_until=activation+full.duration)
        if proof['status']!='CERTIFIED_FREE':
            failures.append('SPACE_'+proof['reason']);continue
        # Recalculate actual physical trajectory extrema including connection,
        # retained source suffix and the complete reserve.
        return dict(prefix=prefix,brake=stop,full=full,certificate=proof,activation=activation,
            source_phase=source_phase,source_interval=[source_phase,source_phase+retained_s*rate],
            source_clock_rate=rate,connection_s=connection_s,retained_physical_s=retained_s,
            retained_progress_m=query_progress,prefix_progress_m=progress,full_extrema=full.extrema(),
            brake_bounds=stop_bounds,compute_s=time.perf_counter()-start,attempts=len(failures)+1)
    raise ValueError('NO_CONSTRAINED_PREFIX '+('|'.join(sorted(set(failures))) or 'NO_TEMPLATE'))

def validate_activation(requested,actual,max_lateness=.01):
    """Mandatory fail-closed guard for the NEXT native adapter.
    Old recorded arbiter is immutable and did not wire this check; never relabel
    its run. A20ms late event at50Hz is not acceptable under a10ms contract.
    """
    if not np.isfinite([requested,actual,max_lateness]).all() or max_lateness<0:raise ValueError('ACTIVATION_TIME')
    if actual<requested:raise ValueError('EARLY_ACTIVATION')
    if actual-requested>max_lateness:raise ValueError('LATE_ACTIVATION')
    return actual-requested
