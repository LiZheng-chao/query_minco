"""Independent simulation test references. No network/Selector/ordinary-navigation authority."""
import hashlib,json,math
import numpy as np

def digest(plan):return hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def evaluate(c,t,k=0):
    a=np.asarray(c,float)
    for _ in range(k):a=a[:,1:]*np.arange(1,a.shape[1])
    return (a*np.power(float(t),np.arange(a.shape[1]))).sum(1)
def quintic(p,v,a,p1,T):
    c=np.zeros((3,6));c[:,:3]=np.stack([p,v,np.asarray(a)/2],axis=1)
    A=np.array([[T**3,T**4,T**5],[3*T**2,4*T**3,5*T**4],[6*T,12*T**2,20*T**3]])
    rhs=np.stack([np.asarray(p1)-evaluate(c,T),-evaluate(c,T,1),-evaluate(c,T,2)],axis=1)
    c[:,3:]=np.linalg.solve(A,rhs.T).T;return c

def extrema(c,T,k):
    a=np.asarray(c,float)
    for _ in range(k):a=a[:,1:]*np.arange(1,a.shape[1])
    sq=np.zeros(1)
    for x in a:sq=np.polynomial.polynomial.polyadd(sq,np.polynomial.polynomial.polymul(x,x))
    roots=np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(sq));ts=[0.,T]+[float(r.real) for r in roots if abs(r.imag)<1e-7 and 0<r.real<T]
    return max(float(np.linalg.norm(evaluate(c,t,k))) for t in ts)

def brake(p,v,a,max_a=.8,max_j=2.):
    p=np.asarray(p,float);v=np.asarray(v,float);a=np.asarray(a,float)
    if any(x.shape!=(3,) or not np.isfinite(x).all() for x in (p,v,a)):raise ValueError('BRAKE_STATE')
    # Fixed registered finite search, never hidden acceleration/jerk clipping.
    for T in (2.,2.5,3.,4.):
        end=p+v*T/2+a*T*T/12;c=quintic(p,v,a,end,T)
        if extrema(c,T,2)<=max_a and extrema(c,T,3)<=max_j:
            return dict(coefficients=[c.tolist()],durations=[T],endpoint=end.tolist(),max_reference_a=extrema(c,T,2),max_reference_j=extrema(c,T,3),certificate=False)
    raise ValueError('NO_BOUNDED_BRAKE_REFERENCE')

def pva(plan,now):
    t=now-plan['source_stamp'];d=np.asarray(plan['durations'],float)
    if not np.isfinite(t) or t<0 or t>sum(d):raise ValueError('REFERENCE_TIME')
    ends=d.cumsum();i=min(int(np.searchsorted(ends,t,side='right')),len(d)-1);local=t-(ends[i-1] if i else 0)
    return np.stack([evaluate(plan['coefficients'][i],local,k) for k in range(3)])

class Lease:
    def __init__(self,owner,ttl=.1):
        self.owner=owner;self.ttl=ttl;self.plan=None;self.seq=-1;self.hb=-1;self.rx=None;self.last_clock=None;self.fault=None
    def fail(self,reason):
        self.fault=self.fault or reason;raise ValueError(self.fault)
    def clock(self,now):
        if not math.isfinite(now):self.fail('CLOCK_NONFINITE')
        if self.last_clock is not None and now<self.last_clock:self.fail('CLOCK_RESET')
        self.last_clock=now
    def receive(self,msg,now,steady):
        if self.fault:raise ValueError(self.fault)
        self.clock(now)
        try:
            seq=msg['sequence'];hb=msg['heartbeat'];plan=msg['plan'];h=digest(plan)
            if msg['owner']!=self.owner:self.fail('OWNER_CONFLICT')
            if type(seq)!=int or type(hb)!=int or hb<0:self.fail('REFERENCE_SEQUENCE_TYPE')
            c=np.asarray(plan['coefficients'],float);d=np.asarray(plan['durations'],float)
            if c.ndim!=3 or c.shape[1:]!=(3,6) or len(c)!=len(d) or not 1<=len(d)<=3 or not np.isfinite(c).all() or not np.isfinite(d).all() or np.any(d<=0) or float(sum(d))>30:self.fail('REFERENCE_ARRAYS')
            stamp=float(plan['source_stamp']);end=stamp+float(sum(d))
            if not math.isfinite(stamp) or stamp<=0 or stamp>now or now>end:self.fail('FUTURE_OR_EXPIRED_REFERENCE')
            if seq==self.seq:
                if h!=digest(self.plan) or hb<=self.hb:self.fail('DUPLICATE_OR_CHANGED_HEARTBEAT')
            else:
                if seq<=self.seq or now-stamp>self.ttl:self.fail('STALE_OR_REORDERED_REFERENCE')
                self.plan=json.loads(json.dumps(plan));self.seq=seq
            self.hb=hb;self.rx=steady
        except (KeyError,TypeError,OverflowError):self.fail('REFERENCE_SCHEMA')
        except ValueError:
            if self.fault:raise
            self.fail('REFERENCE_SCHEMA')
    def check(self,now,steady):
        if self.fault:raise ValueError(self.fault)
        self.clock(now)
        if self.plan is None:self.fail('NO_REFERENCE')
        if self.rx is None or steady-self.rx>self.ttl:self.fail('RECEIPT_TIMEOUT')
        if now>self.plan['source_stamp']+sum(self.plan['durations']):self.fail('REFERENCE_EXPIRED')
        return pva(self.plan,now)
