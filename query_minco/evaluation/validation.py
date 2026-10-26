"""Independent NumPy polynomial reference, extrema and conservative hull collision checks."""
import math
import numpy as np
from .config import CONFIG
from .geometry import point_box_distance,hull_box_distance


def evaluate(c,t,order=0):
    c=np.asarray(c,dtype=np.float64)
    dc=np.stack([c[:,k]*math.factorial(k)/math.factorial(k-order) for k in range(order,6)],axis=1)
    return np.stack([np.polynomial.polynomial.polyval(t,axis) for axis in dc],axis=-1)


def roots(poly,T):
    p=np.polynomial.polynomial.polytrim(poly,tol=1e-13)
    if len(p)<2:return []
    return [float(x.real) for x in np.polynomial.polynomial.polyroots(p)
            if abs(x.imag)<1e-7 and 0<float(x.real)<T]


def norm_max(c,T,order):
    dc=np.stack([c[:,k]*math.factorial(k)/math.factorial(k-order) for k in range(order,6)],axis=1)
    sq=np.zeros(1)
    for axis in dc:sq=np.polynomial.polynomial.polyadd(sq,np.polynomial.polynomial.polymul(axis,axis))
    ts=[0.,float(T)]+roots(np.polynomial.polynomial.polyder(sq),T)
    return float(np.linalg.norm(evaluate(c,np.array(ts),order),axis=-1).max())


def axis_range(c,T):
    result=[]
    for axis in c:
        ts=[0.,float(T)]+roots(np.polynomial.polynomial.polyder(axis),T)
        values=np.polynomial.polynomial.polyval(ts,axis)
        result.append([float(min(values)),float(max(values))])
    return np.array(result)


def bernstein_controls(c,T):
    # Power coefficients in t -> Bernstein controls over normalized u in [0,1].
    n=c.shape[1]-1
    power=np.asarray(c,dtype=np.float64)*(float(T)**np.arange(n+1))[None]
    return np.array([sum(power[:,k]*math.comb(i,k)/math.comb(n,k) for k in range(i+1))
                     for i in range(n+1)])


def split_controls(controls):
    levels=[np.asarray(controls,dtype=np.float64)]
    while len(levels[-1])>1:levels.append((levels[-1][:-1]+levels[-1][1:])/2)
    return np.array([x[0] for x in levels]),np.array([x[-1] for x in levels[::-1]])


def certify_piece(c,T,boxes):
    threshold=CONFIG["body_radius_m"]+CONFIG["required_surface_margin_m"]
    stack=[(bernstein_controls(c,T),0)]
    minimum=np.inf;leaves=0;max_depth=0
    while stack:
        controls,depth=stack.pop();max_depth=max(max_depth,depth)
        bounds=[hull_box_distance(controls,b) for b in boxes]
        lower=min(bounds) if bounds else np.inf
        # Include the outward reserve before stopping subdivision, not afterwards.
        if lower>=threshold+1e-6:
            minimum=min(minimum,lower);leaves+=1;continue
        left,right=split_controls(controls)
        witnesses=np.stack((controls[0],left[-1],controls[-1]))
        if any(float(point_box_distance(witnesses,b).min())<threshold for b in boxes):
            return {"passed":False,"reason":"collision_or_required_margin_violation","depth":depth}
        if depth>=CONFIG["max_certificate_depth"]:
            return {"passed":False,"reason":"continuous_clearance_unproven","depth":depth}
        stack.extend(((left,depth+1),(right,depth+1)))
    # Outward numerical reserve. This is a conservative numerical certificate, not interval arithmetic.
    lower=float(minimum-CONFIG["body_radius_m"]-1e-6)
    return {"passed":lower>=CONFIG["required_surface_margin_m"],"reason":"bounded_bernstein_hull",
            "surface_clearance_lower_bound_m":lower,"leaves":leaves,"depth":max_depth}


def numerical_reference(head,tail,inner,durations):
    A=np.zeros((18,18));rhs=[];row=0
    def constraint(parts,value):
        nonlocal row
        for piece,t,order,sign in parts:
            for k in range(order,6):
                A[row,piece*6+k]+=sign*math.factorial(k)/math.factorial(k-order)*t**(k-order)
        rhs.append(value);row+=1
    for d in range(3):
        constraint([(0,0.,d,1)],head[d]);constraint([(2,float(durations[2]),d,1)],tail[d])
    for i in range(2):
        constraint([(i,float(durations[i]),0,1)],inner[i])
        constraint([(i+1,0.,0,1)],inner[i])
        for d in range(1,5):constraint([(i,float(durations[i]),d,1),(i+1,0.,d,-1)],np.zeros(3))
    return np.linalg.solve(A,np.asarray(rhs)).reshape(3,6,3).transpose(0,2,1)


def validate_teacher(coeffs,durations,head,tail,inner,scene):
    c=np.asarray(coeffs,dtype=np.float64);d=np.asarray(durations,dtype=np.float64)
    failed=[];bounds=[];res=[]
    if c.shape!=(3,3,6) or d.shape!=(3,) or not np.isfinite(c).all() or not np.isfinite(d).all():
        return {"passed":False,"reasons":["invalid_arrays"]}
    total=float(d.sum())
    if not .8<=total<=2.2 or d.min()<.12:failed.append("time_contract")
    boundary=[max(float(np.abs(evaluate(c[0],0.,k)-head[k]).max()),
                  float(np.abs(evaluate(c[2],d[2],k)-tail[k]).max())) for k in range(3)]
    for k,err in enumerate(boundary):
        if err>CONFIG["boundary_atol_pva"][k]:failed.append("boundary_"+str(k))
    for k in range(5):
        err=max(float(np.abs(evaluate(c[i],d[i],k)-evaluate(c[i+1],0.,k)).max()) for i in range(2))
        res.append(err)
        if err>CONFIG["continuity_atol_orders_0_to_4"][k]:failed.append("continuity_"+str(k))
    wp=max(float(np.abs(evaluate(c[i],d[i],0)-inner[i]).max()) for i in range(2))
    if wp>CONFIG["boundary_atol_pva"][0]:failed.append("inner_waypoints")
    domain=np.asarray(scene["flight_center_domain_m"])
    ranges=np.stack([axis_range(a,t) for a,t in zip(c,d)])
    if (ranges[:,:,0]<domain[0]-.0002).any() or (ranges[:,:,1]>domain[1]+.0002).any():failed.append("flight_domain")
    physical=np.asarray(CONFIG["position_decoder_bounds_m"])
    points=np.vstack((inner,tail[0]))
    if (points<physical[0]).any() or (points>physical[1]).any():failed.append("decoder_waypoint_bounds")
    if (np.abs(tail[1])>np.array([6.,4.,3.])).any() or (np.abs(tail[2])>np.array([6.,6.,4.])).any():failed.append("decoder_tail_bounds")
    maxima=[max(norm_max(a,t,k) for a,t in zip(c,d)) for k in [1,2,3]]
    for name,value,limit in zip(["speed","acceleration","jerk"],maxima,[3.,8.,100.]):
        if value>limit:failed.append(name+"_limit")
    certs=[certify_piece(a,t,scene["obstacles"]) for a,t in zip(c,d)]
    if not all(x["passed"] for x in certs):failed.extend(sorted(set(x["reason"] for x in certs if not x["passed"])))
    ref=numerical_reference(head,tail,inner,d)
    referrors=[max(float(np.abs(evaluate(a,np.linspace(0,t,257),k)-evaluate(b,np.linspace(0,t,257),k)).max())
                   for a,b,t in zip(c,ref,d)) for k in range(3)]
    if any(e>tol for e,tol in zip(referrors,[.0002,.002,.01])):failed.append("independent_reference")
    dense=np.concatenate([evaluate(a,np.linspace(0,t,1025)) for a,t in zip(c,d)])
    sampled=min((float(point_box_distance(dense,b).min()) for b in scene["obstacles"]),default=np.inf)-CONFIG["body_radius_m"]
    clear=min([x.get("surface_clearance_lower_bound_m",-1.) for x in certs])
    return {"passed":not failed,"reasons":sorted(set(failed)),"boundary_max_abs_pva":boundary,
            "continuity_max_abs_orders_0_to_4":res,"waypoint_max_abs_m":wp,
            "reference_max_abs_pva":referrors,"max_speed_mps":maxima[0],
            "max_acceleration_mps2":maxima[1],"max_jerk_mps3":maxima[2],
            "total_time_s":total,"segment_times_s":d.tolist(),
            "surface_clearance_lower_bound_m":clear,"dense_surface_clearance_m":sampled,
            "collision_label":"offline_bounded_bernstein_feasible" if not failed else "rejected",
            "certificates":certs,"axis_ranges_m":ranges.tolist()}


def sample_curve(coeffs,durations,count=257):
    durations=np.asarray(durations,dtype=np.float64)
    total=float(np.sum(durations));times=np.linspace(0,total,count)
    boundaries=np.cumsum(durations)
    idx=np.minimum(np.searchsorted(boundaries,times,side="right"),2)
    starts=np.r_[0,boundaries[:-1]]
    return times,np.stack([evaluate(coeffs[i],t-starts[i]) for t,i in zip(times,idx)])


def crossing_side(coeffs,durations,crossing_x):
    for c,T in zip(coeffs,durations):
        poly=c[0].astype(np.float64).copy();poly[0]-=crossing_x
        for t in sorted(roots(poly,float(T))):
            y=float(evaluate(c,t)[1])
            return "left" if y>.15 else "right" if y<-.15 else "center"
    return "no_crossing"
