"""Bounded CPU diagnostics for frozen MINCO; no ROS/control or learned scores.
The inherited-lifetime adapter and existing stop family remain authoritative.
A negative soft-jerk witness is a necessary-condition rejection, never a proof
of infeasibility under the separate hard jerk=10 contract.
"""
import numpy as np
from query_minco.inference.trajectory_execution import Program,MotionLimits,quintic,trim,concatenate
from query_minco.inference.constrained_prefix import slowed
from query_minco.inference.proposal_preparation import prepare_bound_reference

def build_prefix(world,head,connection_s,source_phase,rate,retained_s=.1):
    if source_phase+retained_s*rate>world.duration:raise ValueError('SOURCE_INTERVAL')
    tail=world.pva(source_phase).copy()
    tail[1]*=rate;tail[2]*=rate**2
    connection=quintic(head,tail,connection_s)
    retained=slowed(trim(world,source_phase,source_phase+retained_s*rate),rate)
    prefix=concatenate(connection,retained)
    return connection,retained,prefix

def soft_jerk_witness(prefix,jerk=.35,period=.02):
    # At each legal backup phase, project onto the *current acceleration*
    # direction. Even maximal opposite jerk needs |a|/J to cancel acceleration.
    # Thus v_projected+|a|^2/(2J)>vmax is a valid impossibility witness for
    # ANY stop bounded by this soft jerk, not just the searched quintics.
    if not np.isfinite(jerk) or jerk<=0:raise ValueError('JERK')
    for t in np.arange(0,prefix.duration+1e-8,period):
        pva=prefix.pva(min(t,prefix.duration));a=np.linalg.norm(pva[2])
        if a<1e-9:continue
        projection=float(pva[1]@pva[2]/a)
        unavoidable=projection+a*a/(2*jerk)
        if unavoidable>1+1e-8:
            return dict(phase=float(t),projected_velocity=projection,acceleration=a,
                        minimum_peak_projected_speed=unavoidable,soft_jerk=jerk,
                        reason='SOFT_JERK_NECESSARY_SPEED_VIOLATION')
    return None

def contribution(program):
    # Arc length is numerical reporting only; NOT a safety certificate.
    times=np.linspace(0,program.duration,201)
    points=np.array([program.pva(float(t))[0] for t in times])
    velocities=np.array([program.pva(float(t))[1] for t in times])
    good=np.linalg.norm(velocities,axis=1)>1e-8
    turns=0.
    vv=velocities[good]
    if len(vv)>1:
        vv=vv/np.linalg.norm(vv,axis=1)[:,None]
        turns=float(np.arccos(np.clip((vv[1:]*vv[:-1]).sum(1),-1,1)).sum())
    return dict(duration_s=program.duration,displacement_m=(points[-1]-points[0]).tolist(),
                progress_x_m=float(points[-1,0]-points[0,0]),
                arc_length_estimate_m=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
                direction_change_rad_estimate=turns)

def complete_bank(prefix,volume,now_ros,now_mono):
    # New independent experiment uses old receipt's remaining lifetime exactly.
    return prepare_bound_reference(prefix,volume,now_ros,now_mono,activation_lead=.06)

def _subdivision_matrices(degree,levels=2):
    # Pure numerical basis, safe to reuse across states/times/evidence.
    import math
    m=np.eye(degree+1);parts=[m]
    for _ in range(levels):
        nxt=[]
        for a in parts:
            work=a.copy();left=[work[0]];right=[work[-1]]
            for k in range(degree):
                work=(work[:-1]+work[1:])/2
                left.append(work[0]);right.append(work[-1])
            nxt.extend([np.array(left),np.array(right[::-1])])
        parts=nxt
    return np.array(parts)
_VSUB=_subdivision_matrices(3)
_ASUB=_subdivision_matrices(2)

def fast_soft_stops(heads):
    """Same quartic endpoint family and 0.1s duration grid as soft_brake.
    Conservative continuous Bernstein norm bounds replace repeated root search.
    A rejected hull is NOT mathematical infeasibility. Returns no certificates.
    Positive-acceleration transition is identical to reserved_stop.
    """
    h=np.asarray(heads,float)
    if h.ndim!=3 or h.shape[1:]!=(3,3) or not np.isfinite(h).all():raise ValueError('BRAKE_STATE')
    n=len(h);an=np.linalg.norm(h[:,2],axis=1)
    use=(np.sum(h[:,1]*h[:,2],axis=1)>0)&(an>.02)
    transition_time=np.where(use,an/.35,0.)
    v=h[:,1]+.5*h[:,2]*transition_time[:,None]
    p=h[:,0]+h[:,1]*transition_time[:,None]+h[:,2]*transition_time[:,None]**2/3
    a=np.where(use[:,None],0.,h[:,2])
    if np.any(np.linalg.norm(h[:,1],axis=1)>1+1e-10) or np.any(an>1.5+1e-10):raise ValueError('BRAKE_HEAD_HARD_LIMIT')
    if np.any(np.linalg.norm(v,axis=1)>1+1e-10):raise ValueError('SOFT_TRANSITION_SPEED_LIMIT')
    ts=np.arange(2.,6.+1e-9,.1)[None,:,None]
    vv=v[:,None,:];aa=a[:,None,:]
    j0=-6*vv/ts**2-4*aa/ts
    jt=6*vv/ts**2+2*aa/ts
    jbound=np.maximum(np.linalg.norm(j0,axis=-1),np.linalg.norm(jt,axis=-1))
    zeros=np.zeros_like(vv+aa*ts)
    vb=np.stack([np.broadcast_to(vv,zeros.shape),vv+aa*ts/3,zeros,zeros],axis=-2)
    ab=np.stack([np.broadcast_to(aa,zeros.shape),-3*vv/ts-aa,zeros],axis=-2)
    vbound=np.linalg.norm(np.einsum('skl,ntlz->ntskz',_VSUB,vb),axis=-1).max(axis=(-1,-2))
    abound=np.linalg.norm(np.einsum('skl,ntlz->ntskz',_ASUB,ab),axis=-1).max(axis=(-1,-2))
    ok=(jbound<=.35)&(vbound<=1)&(abound<=1.5)
    if not ok.any(axis=1).all():raise ValueError('NO_CONSERVATIVE_SOFT_BRAKE')
    chosen=ok.argmax(axis=1);times=ts[0,:,0][chosen];programs=[]
    for i,T in enumerate(times):
        c=np.zeros((1,3,6));c[0,:,0]=p[i];c[0,:,1]=v[i];c[0,:,2]=a[i]/2
        c[0,:,3]=-v[i]/T**2-2*a[i]/(3*T)
        c[0,:,4]=v[i]/(2*T**3)+a[i]/(4*T**2)
        stop=Program(c,[T])
        if use[i]:
            tt=transition_time[i];tr=np.zeros((1,3,6));tr[0,:,0]=h[i,0];tr[0,:,1]=h[i,1]
            tr[0,:,2]=h[i,2]/2;tr[0,:,3]=-h[i,2]/(6*tt)
            stop=concatenate(Program(tr,[tt]),stop)
        programs.append(stop)
    return programs


def evaluate_pva_batch(program,times):
    """Exact coefficient evaluation at many times; no finite-sampling proof."""
    import math
    times=np.asarray(times,float)
    if times.ndim!=1 or not np.isfinite(times).all() or (times<0).any() or (times>program.duration+1e-9).any():raise ValueError('NO_EXTRAPOLATION')
    bounds=np.cumsum(program.durations)
    indices=np.minimum(np.searchsorted(bounds,times,side='right'),len(bounds)-1)
    local=times-np.r_[0.,bounds[:-1]][indices]
    c=program.coefficients[indices];out=np.zeros((len(times),3,3))
    for d in range(3):
        for k in range(5,d-1,-1):
            out[:,d]=out[:,d]*local[:,None]+c[:,:,k]*(math.factorial(k)/math.factorial(k-d))
    return out

def prepare_fast_bound_reference(prefix,volume,now_ros,now_mono,activation_lead=.06):
    """Worker-only faster certification, same immutable PreparedReference type.
    Geometry proof composes complete prefix containment with each whole stop.
    No point sampling; same float64 Bernstein/error-budget guarantee and scope.
    Nothing here grants physical publication or authenticates dynamic proposals.
    """
    import json
    from query_minco.inference.committed_reference import PreparedReference,FrozenPolynomial,StopEntry
    if not .02<=activation_lead<=.1:raise ValueError('ACTIVATION_LEAD')
    if not volume.cells:raise ValueError('NO_CONTINUOUS_CELL')
    until=min(c.valid_until for c in volume.cells)
    valid_for=min(120.,until-now_ros-.001)
    if not np.isfinite(valid_for) or valid_for<1:raise ValueError('INHERITED_CELL_EXPIRED')
    prefix.validate(MotionLimits())
    count=int(round(prefix.duration/.02))
    if not 1<=count<=1024 or abs(count*.02-prefix.duration)>1e-9:raise ValueError('PREFIX_ON_50HZ_GRID')
    phases=np.arange(count+1)*.02
    heads=evaluate_pva_batch(prefix,np.minimum(phases,prefix.duration))
    stops=fast_soft_stops(heads)
    if max(t+s.duration for t,s in zip(phases,stops))+activation_lead>valid_for:raise ValueError('FULL_STOP_EXCEEDS_INHERITED_LIFETIME')
    prefix_proof=volume.certify(prefix.coefficients,prefix.durations,now_ros,required_until=now_ros+valid_for)
    if prefix_proof['status']!='CERTIFIED_FREE':raise ValueError('PREFIX_SPACE_'+prefix_proof['reason'])
    first=np.array([s.coefficients[0,:,:3].T for s in stops]);first[:,2]*=2
    if (np.linalg.norm(first-heads,axis=2)>[.0002,.002,.01]).any():raise ValueError('BRAKE_HEAD_PVA')
    last=np.array([s.coefficients[-1] for s in stops]);ends=np.array([s.durations[-1] for s in stops])
    terminal=np.zeros((len(stops),2,3))
    import math
    for d in (1,2):
        for k in range(5,d-1,-1):
            terminal[:,d-1]=terminal[:,d-1]*ends[:,None]+last[:,:,k]*(math.factorial(k)/math.factorial(k-d))
    if (np.linalg.norm(terminal,axis=2)>1e-8).any():raise ValueError('BRAKE_TERMINAL_VA')
    entries=[]
    for phase,head,stop in zip(phases,heads,stops):
        proof=volume.certify(stop.coefficients,stop.durations,now_ros,required_until=now_ros+valid_for)
        if proof['status']!='CERTIFIED_FREE':raise ValueError('STOP_TUBE_'+proof['reason'])
        proof=dict(proof,composition='PREFIX_AND_WHOLE_STOP_CONTINUOUS_CONTAINMENT',
                   certified_prefix_sha256=prefix.digest,stop_phase=float(phase))
        entries.append(StopEntry(float(phase),FrozenPolynomial.freeze(stop),json.dumps(proof,sort_keys=True).encode()))
    return PreparedReference(FrozenPolynomial.freeze(prefix),tuple(entries),
         json.dumps(prefix_proof,sort_keys=True).encode(),now_ros+valid_for,now_mono+valid_for,
         volume.scope.value,now_ros,now_mono)
