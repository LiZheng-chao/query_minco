"""Direction-preserving local target for the forest10m diagnostic planner."""
import math
import numpy as np


def ray_box_exit_distance(position, direction, lower, upper):
    p, d, lo, hi = (np.asarray(x,dtype=float) for x in (position,direction,lower,upper))
    if (any(x.shape!=(3,) or not np.isfinite(x).all() for x in (p,d,lo,hi))
            or np.any(lo>=hi) or np.any(p<lo) or np.any(p>hi)
            or not np.any(d)):
        raise ValueError("LOCAL_RAY_OUT_OF_BOUNDS")
    distances=[(hi[i]-p[i])/d[i] if d[i]>0 else (lo[i]-p[i])/d[i]
               for i in range(3) if d[i]!=0]
    exit_distance=min(distances)
    if not math.isfinite(exit_distance) or exit_distance<=0:
        raise ValueError("LOCAL_RAY_OUT_OF_BOUNDS")
    return exit_distance


def choose_local_goal(space, position, goal, search):
    """Existing Depth hit veto only. The result carries no execution authority."""
    p,g=np.asarray(position,dtype=float),np.asarray(goal,dtype=float)
    lo=space.origin+.33
    hi=space.origin+np.asarray(space.cfg.shape)*.1-.33
    if (any(x.shape!=(3,) or not np.isfinite(x).all() for x in (p,g,lo,hi))
            or np.any(lo>=hi) or np.any(p<lo) or np.any(p>hi)):
        raise ValueError("LOCAL_RAY_OUT_OF_BOUNDS")
    delta=g-p
    distance=float(np.linalg.norm(delta))
    if not math.isfinite(distance):
        raise ValueError("LOCAL_RAY_OUT_OF_BOUNDS")
    if distance<=1e-9:
        target=g
    else:
        direction=delta/distance
        limit=ray_box_exit_distance(p,direction,lo,hi)
        target=p+min(distance,.4,limit)*direction
    segments=max(1,math.ceil(float(np.linalg.norm(target-p))/.05))
    if all(space.ball_free(q,.30+.03) for q in np.linspace(p,target,segments+1)):
        return target,target,'ray'
    local=search(space,p,target)
    if np.linalg.norm(local-p)<=1e-9:
        raise ValueError('NO_SENSOR_CERTIFIED_LOCAL_ROUTE')
    return local,target,'astar'
