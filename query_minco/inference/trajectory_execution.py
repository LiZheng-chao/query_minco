"""Dry-run-only time-aware reference execution. Contains NO ROS publisher.

Reference continuity is not a tracking guarantee. Space checks must include the
connection/brake and remain valid through their execution. Missing proof => halt
with an explicit UNPROVEN outcome, never a fake safe hover or zero-velocity reset.
"""
from dataclasses import dataclass
import math
import numpy as np
from query_minco.evaluation.validation import evaluate,norm_max,axis_range
from query_minco.inference.observed_space import readonly,trajectory_digest
from query_minco.inference.selector_safety import world_coefficients

@dataclass(frozen=True)
class MotionLimits:
    speed:float=1.
    acceleration:float=1.5
    jerk:float=10.
    def __post_init__(self):
        if not all(np.isfinite(v) and v>0 for v in self.__dict__.values()):raise ValueError('LIMITS')

@dataclass(frozen=True)
class Program:
    coefficients:np.ndarray
    durations:np.ndarray
    def __post_init__(self):
        c=readonly(self.coefficients);d=readonly(self.durations)
        if c.ndim!=3 or c.shape[1:]!=(3,6) or d.shape!=(len(c),) or len(c)==0 or not np.isfinite(c).all() or not np.isfinite(d).all() or (d<=0).any():raise ValueError('PROGRAM_ARRAYS')
        object.__setattr__(self,'coefficients',c);object.__setattr__(self,'durations',d)
    @property
    def duration(self):return float(self.durations.sum())
    @property
    def digest(self):return trajectory_digest(self.coefficients,self.durations)
    def pva(self,t):
        if not np.isfinite(t) or not 0<=t<=self.duration+1e-10:raise ValueError('NO_EXTRAPOLATION')
        t=min(t,self.duration);bounds=np.cumsum(self.durations);i=min(int(np.searchsorted(bounds,t,side='right')),len(bounds)-1)
        local=t-(bounds[i-1] if i else 0)
        return np.stack([evaluate(self.coefficients[i],local,k) for k in range(3)])
    def extrema(self):
        return [max(norm_max(c,t,k) for c,t in zip(self.coefficients,self.durations)) for k in (1,2,3)]
    def validate(self,limits):
        if any(v>lim+1e-8 for v,lim in zip(self.extrema(),[limits.speed,limits.acceleration,limits.jerk])):raise ValueError('MOTION_LIMIT')
        for i in range(len(self.durations)-1):
            residual=np.abs(np.stack([evaluate(self.coefficients[i],self.durations[i],k)-evaluate(self.coefficients[i+1],0,k) for k in range(3)]))
            if (residual.max(1)>[.0002,.002,.01]).any():raise ValueError('PVA_DISCONTINUITY')
        return self

def quintic(head,tail,duration):
    h=np.asarray(head,float);tail=np.asarray(tail,float);T=float(duration)
    if h.shape!=(3,3) or tail.shape!=(3,3) or not np.isfinite(h).all() or not np.isfinite(tail).all() or not np.isfinite(T) or T<=0:raise ValueError('PVA_BOUNDARY')
    c=np.zeros((3,6));c[:,:3]=np.stack([h[0],h[1],h[2]/2],axis=1)
    A=np.array([[T**3,T**4,T**5],[3*T*T,4*T**3,5*T**4],[6*T,12*T*T,20*T**3]])
    residual=tail-np.stack([evaluate(c,T,k) for k in range(3)])
    c[:,3:]=np.linalg.solve(A,residual).T
    return Program(c[None],np.array([T]))

def shift_polynomial(c,h):
    return np.array([[sum(c[a,k]*math.comb(k,l)*h**(k-l) for k in range(l,6)) for l in range(6)] for a in range(3)])
def trim(program,start,end=None):
    end=program.duration if end is None else end
    if not 0<=start<end<=program.duration+1e-9:raise ValueError('TRIM_RANGE')
    c=[];d=[];offset=0.
    for coeff,T in zip(program.coefficients,program.durations):
        a=max(start,offset);b=min(end,offset+T)
        if b>a+1e-10:c.append(shift_polynomial(coeff,a-offset));d.append(b-a)
        offset+=T
    return Program(np.array(c),np.array(d))
def concatenate(a,b):
    return Program(np.concatenate([a.coefficients,b.coefficients]),np.r_[a.durations,b.durations])

def brake(head,limits=MotionLimits(),max_time=4.):
    """PVA-continuous quintic stop; search bounds, no instantaneous velocity reset.
    Endpoint heuristic is NOT a universal stopping-distance formula.
    """
    head=np.asarray(head,float)
    if head.shape!=(3,3) or not np.isfinite(head).all():raise ValueError('BRAKE_STATE')
    for T in np.arange(.4,max_time+1e-8,.1):
        endpoint=head[0]+head[1]*T/2+head[2]*T*T/12
        tail=np.stack([endpoint,np.zeros(3),np.zeros(3)])
        p=quintic(head,tail,T)
        try:p.validate(limits)
        except ValueError:continue
        stop_distance=float(np.linalg.norm(endpoint-head[0]))
        # Arc length upper bound, NOT endpoint displacement; includes reversals.
        bound=norm_max(p.coefficients[0],T,1)*T
        return p,dict(duration_s=float(T),endpoint_displacement_m=stop_distance,
            travel_distance_upper_bound_m=float(bound),extrema=p.extrema(),requires_swept_tube=True)
    raise ValueError('NO_BOUNDED_BRAKE')

def local_goal(position,rotation,global_goal):
    delta=np.asarray(global_goal,float)-np.asarray(position,float);distance=float(np.linalg.norm(delta))
    if not np.isfinite(distance):raise ValueError('GOAL_FINITE')
    if distance<2.4:return dict(mode='TERMINAL_ANALYTIC_ONLY',network_goal=None,distance=distance)
    g=np.asarray(rotation,float).T@delta*(2.4/distance)
    if g[0]<=0 or (np.abs(g[1:]/g[0])>.175+1e-6).any():return dict(mode='DIRECTION_REJECT',network_goal=None,distance=distance)
    return dict(mode='NETWORK_ROLLING',network_goal=g,distance=distance)

def terminal(head,goal,limits=MotionLimits(),max_time=8.):
    """Analytic terminal reference, not a new network Goal distribution."""
    head=np.asarray(head,float);goal=np.asarray(goal,float)
    for T in np.arange(.8,max_time+1e-8,.2):
        p=quintic(head,np.stack([goal,np.zeros(3),np.zeros(3)]),T)
        try:p.validate(limits)
        except ValueError:continue
        delta=goal-head[0];distance=np.linalg.norm(delta)
        if distance<1e-9:
            if np.linalg.norm(head[1:])>1e-9:continue
        else:
            projected=np.zeros((3,6));projected[0]=(delta/distance)@p.coefficients[0];projected[0,0]-=(delta/distance)@head[0]
            bounds=axis_range(projected,T)[0]
            if bounds[0]<-1e-8 or bounds[1]>distance+1e-8:continue
        return p
    raise ValueError('NO_BOUNDED_MONOTONE_TERMINAL')

@dataclass(frozen=True)
class Measurement:
    stamp:float
    pva:np.ndarray
    def __post_init__(self):
        p=readonly(self.pva)
        if p.shape!=(3,3) or not np.isfinite(p).all() or not np.isfinite(self.stamp):raise ValueError('MEASUREMENT')
        object.__setattr__(self,'pva',p)
    def predict(self,t):
        dt=t-self.stamp
        if not 0<=dt<=.15:raise ValueError('MEASUREMENT_PREDICTION_HORIZON')
        p=self.pva.copy();p[0]+=p[1]*dt+.5*p[2]*dt*dt;p[1]+=p[2]*dt;return p

@dataclass(frozen=True)
class Scheduled:
    program:Program
    activation:float
    source_stamp:float
    expiry:float
    certificate:dict
    brake_reserve:Program
    def pva(self,now):return self.program.pva(now-self.activation)

class Authority:
    """Offline handover state machine. Deliberately cannot grant ROS permission."""
    def __init__(self):self.owner=None;self.epoch=0;self.lease_until=-np.inf
    def release(self,owner):
        if self.owner!=owner:raise ValueError('NOT_OWNER')
        self.owner=None;self.lease_until=-np.inf;self.epoch+=1
    def acquire_dryrun(self,observed_publishers,now,lease_s=.5):
        if observed_publishers or self.owner is not None:raise ValueError('PUBLISHER_CONFLICT')
        if not np.isfinite(now) or not 0<lease_s<=1:raise ValueError('LEASE')
        self.owner='query_executor_DRYRUN';self.epoch+=1;self.lease_until=now+lease_s;return self.epoch
    def verify(self,observed_publishers,now,epoch):
        if not np.isfinite(now) or observed_publishers or self.owner!='query_executor_DRYRUN' or epoch!=self.epoch or now>self.lease_until:raise ValueError('AUTHORITY_REVOKED')
    def renew_dryrun(self,observed_publishers,now,epoch,lease_s=.5):
        self.verify(observed_publishers,now,epoch)
        if not 0<lease_s<=1:raise ValueError('LEASE')
        self.lease_until=now+lease_s
    def publish_ros(self,*_args,**_kwargs):raise PermissionError('QUERY_ROS_PUBLICATION_NOT_AUTHORIZED_IMPLEMENTATION_ABSENT')

def prepare(candidate,source_stamp,origin,rotation,measurement,now,activation,volume,
            active=None,limits=MotionLimits(),splice_s=.25,prefix_s=.1):
    if not all(np.isfinite(x) for x in [source_stamp,now,activation,splice_s,prefix_s]):raise ValueError('TIME_FINITE')
    if not 0<=now-source_stamp<=.25 or not 0<=now-measurement.stamp<=.05:raise ValueError('STALE_SOURCE_OR_MEASUREMENT')
    if not .02<=activation-now<=.1:raise ValueError('LATE_OR_FAR_ACTIVATION')
    if not .8-1e-6<=candidate.duration<=2.2+1e-6 or min(candidate.durations)<.12-1e-6:raise ValueError('SOURCE_MINCO_TIME_CONTRACT')
    world=Program(world_coefficients(candidate.coefficients,origin,rotation),candidate.durations)
    phase=activation-source_stamp
    target_phase=phase+splice_s
    if phase<0 or splice_s<=0 or prefix_s<=0 or target_phase+prefix_s>world.duration:raise ValueError('SOURCE_HORIZON')
    predicted=measurement.predict(activation)
    head=active.pva(activation) if active is not None else predicted
    if (np.max(np.abs(predicted-head),axis=1)>[.02,.05,.1]).any():raise ValueError('PREDICTED_TRACKING_TUBE')
    connection=quintic(head,world.pva(target_phase),splice_s)
    program=concatenate(connection,trim(world,target_phase,target_phase+prefix_s)).validate(limits)
    end=activation+program.duration
    stop,reserve=brake(program.pva(program.duration),limits)
    # Reservation covers the actually scheduled prefix AND its nonzero-velocity stop.
    combined=concatenate(program,stop).validate(limits)
    cert=volume.certify(combined.coefficients,combined.durations,now,required_until=end+stop.duration)
    if cert['status']!='CERTIFIED_FREE':raise ValueError('EXECUTION_AND_BRAKE_SPACE_'+cert['reason'])
    return Scheduled(program,activation,source_stamp,min(end,cert['valid_until']),cert,stop)

class DryExecutor:
    """50Hz dry reference stream. Caller must supply fresh measurement and space."""
    period=.02
    def __init__(self,limits=MotionLimits()):
        self.limits=limits;self.active=None;self.pending=None;self.last_now=None;self.last_tick=None
        self.mode='IDLE';self.events=[];self.authority=Authority();self.epoch=None;self.hard_fault=False
    def arm_dryrun(self,publishers,now,lease_s=.5):
        self.epoch=self.authority.acquire_dryrun(publishers,now,lease_s)
    def submit(self,scheduled,now):
        if self.hard_fault:raise ValueError('FAULT_LATCHED')
        if not now<scheduled.activation or now>scheduled.expiry:raise ValueError('LATE_SUBMISSION')
        scheduled.program.validate(self.limits)
        scheduled.brake_reserve.validate(self.limits)
        if scheduled.expiry!=scheduled.activation+scheduled.program.duration:raise ValueError('EXPIRY_CONTRACT')
        self.pending=scheduled
    def stop(self,measurement,now,volume,reason):
        self.events.append(dict(type='WATCHDOG',stamp=now,reason=reason))
        try:
            if not 0<=now-measurement.stamp<=.05:raise ValueError('NO_FRESH_BRAKE_STATE')
            head=self.active.pva(now) if self.active and self.active.activation<=now<=self.active.expiry else measurement.pva
            if (np.max(np.abs(head-measurement.pva),axis=1)>[.02,.05,.1]).any():raise ValueError('TRACKING_TUBE_EXCEEDED')
            p,_=brake(head,self.limits)
            cert=volume.certify(p.coefficients,p.durations,now,required_until=now+p.duration)
            if cert['status']!='CERTIFIED_FREE':raise ValueError('BRAKE_SPACE_UNKNOWN')
            self.active=Scheduled(p,now,measurement.stamp,now+p.duration,cert,p);self.pending=None;self.mode='BRAKING'
            return dict(mode=self.mode,pva=p.pva(0),reason=reason,ros_published=False)
        except ValueError as e:
            self.hard_fault=True;self.mode='EMERGENCY_STOP_UNPROVEN';self.pending=None
            return dict(mode=self.mode,pva=None,reason=str(e),ros_published=False,physical_stop_guaranteed=False)
    def tick(self,now,measurement,volume,publishers=()):
        if self.hard_fault:return dict(mode=self.mode,pva=None,ros_published=False,physical_stop_guaranteed=False)
        if not np.isfinite(now) or self.last_now is not None and now<=self.last_now:
            self.hard_fault=True;self.mode='CLOCK_FAULT';return dict(mode=self.mode,pva=None,ros_published=False)
        gap=self.last_now is not None and now-self.last_now>.04+1e-9;self.last_now=now
        try:self.authority.verify(publishers,now,self.epoch)
        except ValueError:
            self.hard_fault=True;self.mode='AUTHORITY_FAULT';return dict(mode=self.mode,pva=None,ros_published=False)
        if gap:return self.stop(measurement,now,volume,'MISSED_50HZ_TICKS')
        if not 0<=now-measurement.stamp<=.05:return self.stop(measurement,now,volume,'STALE_ODOM')
        if self.pending and now>=self.pending.activation:
            if now-self.pending.activation>.01:return self.stop(measurement,now,volume,'LATE_ACTIVATION')
            expected=self.pending.pva(now)
            if (np.max(np.abs(expected-measurement.pva),axis=1)>[.02,.05,.1]).any():return self.stop(measurement,now,volume,'ACTIVATION_TRACKING_ERROR')
            self.active=self.pending;self.pending=None;self.mode='EXECUTING'
        if self.active is None and self.pending and now<self.pending.activation:
            return dict(mode='WAITING_EXTERNAL_REFERENCE',pva=None,ros_published=False,requires_external_reference_until_activation=True)
        if self.active is None:return self.stop(measurement,now,volume,'NO_VALID_CANDIDATE')
        if now<self.active.activation:return dict(mode='WAITING',pva=None,ros_published=False)
        if self.mode=='STOPPED':
            ref=self.active.program.pva(self.active.program.duration)
            hold=Program(np.pad(ref[0][:,None],((0,0),(0,5)))[None],np.array([.02]))
            proof=volume.certify(hold.coefficients,hold.durations,now,required_until=now+.02)
            if proof['status']!='CERTIFIED_FREE' or (np.max(np.abs(ref-measurement.pva),axis=1)>[.02,.03,.05]).any():
                return self.stop(measurement,now,volume,'STOPPED_HOLD_INVALID')
            return dict(mode='STOPPED',pva=ref,ros_published=False)
        if self.mode=='EXECUTING' and now>=self.active.expiry-1e-10:
            # Reserve was connected at the exact end, even if that time falls
            # between two 50Hz ticks. Never extrapolate the expired prefix.
            old=self.active;end=old.expiry;p=old.brake_reserve
            proof=volume.certify(p.coefficients,p.durations,now,required_until=end+p.duration)
            if now-end>.04 or proof['status']!='CERTIFIED_FREE':return self.stop(measurement,now,volume,'EXPIRED_WITHOUT_VALID_RESERVE')
            self.active=Scheduled(p,end,old.source_stamp,end+p.duration,proof,p);self.mode='BRAKING'
            self.events.append(dict(type='RESERVED_BRAKE',stamp=now,activation=end))
        if self.mode=='BRAKING' and now>=self.active.expiry-1e-10:
            ref=self.active.program.pva(self.active.program.duration)
            if (np.max(np.abs(ref-measurement.pva),axis=1)>[.02,.03,.05]).any():return self.stop(measurement,now,volume,'NOT_STOPPED')
            self.mode='STOPPED'
            return dict(mode='STOPPED',pva=ref,ros_published=False)
        if now>self.active.expiry+1e-10:return self.stop(measurement,now,volume,'TRAJECTORY_EXPIRED')
        ref=self.active.pva(now)
        if (np.max(np.abs(ref-measurement.pva),axis=1)>[.02,.05,.1]).any():return self.stop(measurement,now,volume,'TRACKING_TUBE_EXCEEDED')
        # Recheck proof at each command time, not just an old certificate boolean.
        proof=volume.certify(self.active.program.coefficients,self.active.program.durations,now,required_until=self.active.expiry)
        if proof['status']!='CERTIFIED_FREE':return self.stop(measurement,now,volume,'SPACE_EVIDENCE_EXPIRED')
        if self.mode=='BRAKING' and now>=self.active.expiry-1e-9:
            if np.linalg.norm(measurement.pva[1])>.03 or np.linalg.norm(measurement.pva[2])>.05:return self.stop(measurement,now,volume,'NOT_STOPPED')
            self.mode='STOPPED'
        return dict(mode=self.mode,pva=ref,ros_published=False,controller_branch='NATIVE_EMPTY_PVA_FEEDBACK_NOT_BRAKE_CERTIFICATE')

class TerminalMonitor:
    """Goal success requires actual measured stop dwell; not endpoint fitting."""
    def __init__(self,goal,dwell_s=.5):
        self.goal=np.asarray(goal,float);self.dwell=dwell_s;self.since=None;self.last=None
        if self.goal.shape!=(3,) or not np.isfinite(self.goal).all() or not 0<dwell_s<=5:raise ValueError('TERMINAL_MONITOR_CONTRACT')
    def update(self,measurement):
        if self.last is not None and measurement.stamp<=self.last:
            self.since=None;raise ValueError('TERMINAL_CLOCK_RESET')
        if self.last is not None and measurement.stamp-self.last>.04+1e-9:self.since=None
        self.last=measurement.stamp
        p=measurement.pva
        within=np.linalg.norm(p[0]-self.goal)<=.1 and np.linalg.norm(p[1])<=.03 and np.linalg.norm(p[2])<=.05
        if not within:self.since=None;return False
        if self.since is None:self.since=measurement.stamp
        return measurement.stamp-self.since>=self.dwell
