"""Continuous crossing metric, independent from immutable Hard task thresholds."""
import numpy as np
from .validation import roots,evaluate
COMPONENT_ID="continuous_forward_wall_crossing_v1"

def crossing_metric(coeffs,durations,planes,gap,expected_sign):
    c=np.asarray(coeffs,dtype=float);dt=np.asarray(durations,dtype=float)
    if c.shape!=(3,3,6) or dt.shape!=(3,) or not np.isfinite(c).all() or not np.isfinite(dt).all() or (dt<=0).any():
        raise ValueError("CROSSING_INPUT")
    all_hits=[]
    for plane in planes:
        hits=[];offset=0.
        for piece,(a,t) in enumerate(zip(c,dt)):
            poly=a[0].copy();poly[0]-=plane
            ts=roots(poly,t)
            for end in [0.,float(t)]:
                if abs(float(np.polynomial.polynomial.polyval(end,poly)))<1e-7:ts.append(end)
            for local in sorted(set(ts)):
                pos=evaluate(a,local);vel=evaluate(a,local,1)
                if vel[0]>1e-7:
                    hits.append(dict(time_s=offset+local,position=pos.tolist(),velocity_x=float(vel[0]),piece=piece))
            offset+=t
        all_hits.append(dict(plane_x_m=float(plane),forward_hits=hits))
    center=all_hits[1]["forward_hits"];margin=.150001
    inside=bool(center) and all(gap[0]+margin<=x["position"][1]<=gap[1]-margin for x in center)
    sign_ok=bool(center) and all(expected_sign*x["position"][1]>0 for x in center)
    beyond=float(evaluate(c[-1],dt[-1])[0])>=planes[-1]+.15
    ordered=all(x["forward_hits"] for x in all_hits)
    if ordered:
        ordered=all_hits[0]["forward_hits"][0]["time_s"]<=center[0]["time_s"]<=all_hits[2]["forward_hits"][0]["time_s"]
    return dict(passed=bool(ordered and inside and sign_ok and beyond),crossed_all_planes=bool(ordered),
        center_in_eroded_gap=inside,correct_direction=sign_ok,terminal_beyond_exit=bool(beyond),planes=all_hits,
        center_crossing_y_m=[x["position"][1] for x in center],expected_sign=int(expected_sign),component_id=COMPONENT_ID)
