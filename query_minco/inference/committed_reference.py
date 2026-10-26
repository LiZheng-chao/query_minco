"""Prepared prefix + complete stop. No ROS, disk I/O, or planning in tick().
Worker-owned certification; immutable byte payloads; single-consumer execution.
Not a hard-real-time or native Controller qualification claim.
"""
from dataclasses import dataclass
import bisect,collections,json,math,time
import numpy as np
from query_minco.inference.trajectory_execution import Program,MotionLimits,trim,concatenate
from query_minco.inference.execution_clock_v3 import soft_brake
from query_minco.inference.constrained_prefix import validate_activation

@dataclass(frozen=True)
class FrozenPolynomial:
    coefficients:bytes
    durations:tuple
    def __post_init__(self):
        if not isinstance(self.coefficients,bytes):raise ValueError('IMMUTABLE_BYTES_REQUIRED')
        if len(self.coefficients)!=len(self.durations)*3*6*8:raise ValueError('SHAPE')
    @classmethod
    def freeze(cls,p):
        return cls(p.coefficients.astype('<f8').tobytes(),tuple(map(float,p.durations)))
    @property
    def duration(self):return sum(self.durations)
    def pva(self,t):
        if not math.isfinite(t) or t<0 or t>self.duration+1e-8:raise ValueError('NO_EXTRAPOLATION')
        c=np.frombuffer(self.coefficients,dtype='<f8').reshape(-1,3,6)
        boundaries=np.cumsum(self.durations)
        i=min(bisect.bisect_right(boundaries,t),len(self.durations)-1)
        local=min(t,self.duration)-(boundaries[i-1] if i else 0)
        out=np.zeros((3,3))
        # Bounded coefficient evaluation only, no polynomial roots/certification.
        for deriv in range(3):
            for k in range(5,deriv-1,-1):
                out[deriv]=out[deriv]*local+c[i,:,k]*(math.factorial(k)/math.factorial(k-deriv))
        return out

@dataclass(frozen=True)
class StopEntry:
    phase:float
    polynomial:FrozenPolynomial
    proof:bytes

@dataclass(frozen=True)
class PreparedReference:
    prefix:FrozenPolynomial
    stops:tuple
    prefix_proof:bytes
    valid_until_ros:float
    valid_until_mono:float
    scene_id:str
    prepared_ros:float
    prepared_mono:float

def reserved_stop(head):
    # Slow any positive acceleration continuously before the ordinary soft stop.
    # All actual extrema and geometry are still checked by the worker.
    h=np.array(head,float);speed=np.linalg.norm(h[1]);acc=np.linalg.norm(h[2])
    if np.dot(h[1],h[2])>0 and acc>.02:
        T=acc/.35
        c=np.zeros((1,3,6));c[0,:,0]=h[0];c[0,:,1]=h[1]
        c[0,:,2]=h[2]/2;c[0,:,3]=-h[2]/(6*T)
        transition=Program(c,[T]).validate(MotionLimits())
        stop,_=soft_brake(transition.pva(T))
        return concatenate(transition,stop).validate(MotionLimits())
    return soft_brake(h)[0]

def prepare_reference(prefix,volume,now_ros,now_mono,valid_for=120.,period=.02):
    """Planning worker only. Reject whole bundle if any stop tube is unproved.
    Each phase is a legal future activation; full delayed prefix + stop checked.
    """
    prefix.validate(MotionLimits())
    if abs(period-.02)>1e-12 or not 1<=valid_for<=120:raise ValueError('CONTRACT')
    count=int(round(prefix.duration/period))
    if not 1<=count<=1024:raise ValueError('BOUNDED_PREFIX_GRID')
    if abs(count*period-prefix.duration)>1e-9:raise ValueError('PREFIX_ON_50HZ_GRID')
    entries=[]
    for k in range(count+1):
        phase=k*period
        stop=reserved_stop(prefix.pva(phase))
        full=stop if k==0 else concatenate(trim(prefix,0,phase),stop)
        full.validate(MotionLimits())
        # valid_for includes all delayed execution and the complete stop.
        if phase+stop.duration>=valid_for:raise ValueError('CERTIFICATE_HORIZON')
        proof=volume.certify(full.coefficients,full.durations,now_ros,
                             required_until=now_ros+valid_for)
        if proof['status']!='CERTIFIED_FREE':raise ValueError('STOP_TUBE_'+proof['reason'])
        entries.append(StopEntry(phase,FrozenPolynomial.freeze(stop),
                                json.dumps(proof,sort_keys=True).encode()))
    proof=volume.certify(prefix.coefficients,prefix.durations,now_ros,
                         required_until=now_ros+valid_for)
    if proof['status']!='CERTIFIED_FREE':raise ValueError('PREFIX_SPACE')
    return PreparedReference(FrozenPolynomial.freeze(prefix),tuple(entries),
        json.dumps(proof,sort_keys=True).encode(),now_ros+valid_for,
        now_mono+valid_for,'EXTERNAL_SUPERVISOR',now_ros,now_mono)

@dataclass(frozen=True)
class StateSnapshot:
    stamp_ros:float
    received_mono:float
    pva:tuple
    publishers:tuple=('/f3_reference_arbiter',)
    topology_received_mono:float=0.

class ReferenceKernel:
    """Tick owner only: immutable pending assignment, no worker locks.
    Keeps active finite-stop commitment when a new proposal is rejected.
    Transport/physical publication is deliberately separate and NOT enabled here.
    """
    def __init__(self):
        self.active=None;self.pending=None;self.start_ros=None;self.start_mono=None
        self.stop_entry=None;self.stop_deadline=None;self.mode='DISARMED'
        self.events=collections.deque(maxlen=10000);self.last_ros=None
        self.last_mono=None;self.qualification_failed=False;self.last_request=None
    def event(self,reason,ros,mono,**data):
        self.events.append(dict(reason=reason,ros=ros,mono=mono,**data))
    def submit(self,bundle,activation_ros,activation_mono):
        # Only independently authenticated worker output can reach this local interface.
        # No network or arbitrary bytes deserialization is exposed by the kernel.
        if not isinstance(bundle,PreparedReference):raise ValueError('PREPARED_REQUIRED')
        self.pending=(bundle,float(activation_ros),float(activation_mono))
    def _valid(self,bundle,ros,mono,remaining=0.):
        return (bundle.prepared_ros<=ros and bundle.prepared_mono<=mono
            and ros+remaining<=bundle.valid_until_ros
            and mono+remaining<=bundle.valid_until_mono)
    def _state_gate(self,s,ros,mono,expected):
        if s is None or not 0<=mono-s.received_mono<=.05:raise ValueError('DROPPED_ODOM')
        if not 0<=ros-s.stamp_ros<=.05:raise ValueError('STALE_OR_FUTURE_ODOM')
        if s.publishers!=('/f3_reference_arbiter',):raise ValueError('PUBLISHER_CONFLICT')
        if not 0<=mono-s.topology_received_mono<=.5:raise ValueError('AUTHORITY_EXPIRED')
        observed=np.asarray(s.pva,float)
        if observed.shape!=(3,3) or not np.isfinite(observed).all():raise ValueError('PVA')
        dt=ros-s.stamp_ros
        predicted=observed.copy()
        predicted[0]+=observed[1]*dt+.5*observed[2]*dt*dt
        predicted[1]+=observed[2]*dt
        if (np.linalg.norm(predicted-expected,axis=1)>[.02,.05,.1]).any():
            raise ValueError('TRACKING_TUBE')
    def request_stop(self,reason,ros,mono):
        self.event('STOP_REQUEST',ros,mono,cause=reason);self.last_request=reason
        if self.active is None:
            self.mode='UNPROVEN_NO_ACTIVE_REFERENCE';return False
        if self.stop_entry is not None:return True # never abandon a committed stop
        elapsed=max(0.,mono-self.start_mono)
        target=elapsed+.06
        entry=next((x for x in self.active.stops if x.phase>=target-1e-9),self.active.stops[-1])
        # Near prefix end, the final reserve is already scheduled, not invented hover.
        activation=self.start_mono+entry.phase
        remaining=max(0,activation-mono)+entry.polynomial.duration
        if not self._valid(self.active,ros,mono,remaining):
            self.mode='UNPROVEN_EXPIRED_RESERVE';self.qualification_failed=True;return False
        self.stop_entry=entry;self.stop_deadline=activation;self.mode='BACKUP_ARMED'
        return True
    def _reference(self,mono):
        elapsed=max(0.,mono-self.start_mono)
        if self.stop_entry is not None and mono>=self.stop_deadline:
            return self.stop_entry.polynomial.pva(min(mono-self.stop_deadline,
                self.stop_entry.polynomial.duration))
        if elapsed>=self.active.prefix.duration:
            tail=self.active.stops[-1]
            return tail.polynomial.pva(min(elapsed-tail.phase,tail.polynomial.duration))
        return self.active.prefix.pva(elapsed)
    def tick(self,ros,mono,snapshot,watchdog=False):
        # Real clocks are not rewritten. Jump prevents new activation.
        jump=self.last_ros is not None and abs((ros-self.last_ros)-(mono-self.last_mono))>.005
        gap=self.last_mono is not None and mono-self.last_mono>.04
        self.last_ros=ros;self.last_mono=mono
        if jump or gap:
            self.qualification_failed=True
            self.event('CLOCK_JUMP' if jump else 'SCHEDULING_GAP',ros,mono)
            self.pending=None
            self.request_stop('CLOCK_JUMP' if jump else 'SCHEDULING_GAP',ros,mono)
        if self.pending and mono>=self.pending[2]:
            bundle,ar,am=self.pending;self.pending=None
            try:
                if jump:raise ValueError('CLOCK_JUMP')
                validate_activation(ar,ros)
                if mono-am>.01:raise ValueError('MONOTONIC_LATE')
                maxstop=max(x.phase+x.polynomial.duration for x in bundle.stops)
                if not self._valid(bundle,ros,mono,maxstop):raise ValueError('EXPIRED_BUNDLE')
                head=bundle.prefix.pva(0)
                if self.active is not None and (np.max(np.abs(head-self._reference(mono)),axis=1)>[.0002,.002,.01]).any():raise ValueError('COMMAND_PVA_DISCONTINUITY')
                self._state_gate(snapshot,ros,mono,head)
                self.active=bundle;self.start_ros=ar;self.start_mono=am
                self.stop_entry=None;self.stop_deadline=None;self.mode='PREFIX'
                self.event('ACTIVATED',ros,mono,lateness=ros-ar)
            except ValueError as e:
                # Do NOT discard active reference, kill transport, or invent hover.
                self.event('PROPOSAL_REJECT',ros,mono,cause=str(e))
        if self.active is None:return None
        if self.stop_entry is None and mono-self.start_mono>=self.active.prefix.duration:
            self.stop_entry=self.active.stops[-1]
            self.stop_deadline=self.start_mono+self.stop_entry.phase
            self.mode='BACKUP_ARMED'
        if watchdog:self.request_stop('WATCHDOG',ros,mono)
        expected=self._reference(mono)
        try:
            if not self._valid(self.active,ros,mono):raise ValueError('EXPIRED_ACTIVE')
            self._state_gate(snapshot,ros,mono,expected)
        except ValueError as e:
            self.qualification_failed=True;self.event('MEASUREMENT_FAULT',ros,mono,cause=str(e))
            if not self.request_stop(str(e),ros,mono):return None
            # Missing measurement is NOT a new actual tracking certificate.
            # The saved trajectory remains a conditional command reserve only.
            self.mode='BACKUP_TRACKING_UNPROVEN'
        if self.stop_entry is not None and mono>=self.stop_deadline:
            late=mono-self.stop_deadline
            if self.mode=='BACKUP_ARMED':
                if late>.01:self.qualification_failed=True;self.event('BACKUP_LATE',ros,mono,lateness=late)
                self.event('BACKUP_ACTIVATED',ros,mono,lateness=late)
                self.mode='BRAKING'
            if late>=self.stop_entry.polynomial.duration and self.mode=='BRAKING':
                self.mode='STOP_REFERENCE_COMPLETE' # measured stop dwell still required
        return expected
