"""B0 FLU axis-aligned solids and deterministic optical pinhole ray casting."""
import numpy as np
from .config import CONFIG


def make_scene(kind,cx,half_y,thickness):
    # The bounded flight domain prevents bypassing an impassable slab at floor/ceiling.
    boxes=[
        {"name":"left_wall","min":[-.5,1.8,-.8],"max":[3.6,2.,.8]},
        {"name":"right_wall","min":[-.5,-2.,-.8],"max":[3.6,-1.8,.8]},
        {"name":"floor","min":[-.5,-2.,-.8],"max":[3.6,2.,-.6]},
        {"name":"ceiling","min":[-.5,-2.,.6],"max":[3.6,2.,.8]},
        {"name":"backdrop","min":[3.5+.1*(cx-1.4),-2.,-.8],"max":[3.6+.1*(cx-1.4),2.,.8]}]
    def slab(name,a,b,x=cx,w=thickness):
        boxes.append({"name":name,"min":[x-w/2,a,-.8],"max":[x+w/2,b,.8]})
    if kind=="left_only":slab("right_and_center_blocker",-2.,half_y)
    elif kind=="right_only":slab("left_and_center_blocker",-half_y,2.)
    elif kind=="dual":slab("center_blocker",-half_y,half_y)
    elif kind=="narrow":
        slab("lower_gate",-2.,.44);slab("upper_gate",1.20,2.)
    elif kind=="blocked":slab("separating_slab",-2.,2.)
    elif kind=="staggered_double":
        slab("front_center",-half_y,half_y,x=cx-.14,w=.10)
        slab("rear_center",-half_y*.9,half_y*.9,x=cx+.14,w=.10)
    elif kind!="free":raise ValueError("unknown layout")
    return {"kind":kind,"obstacles":boxes,"crossing_x_m":cx,
            "parameters":{"center_x_m":cx,"half_y_m":half_y,"thickness_m":thickness},
            "flight_center_domain_m":CONFIG["flight_center_domain_m"],
            "left_definition":"positive B0 y; projects to image left",
            "no_route_proof":({"type":"full_cross_section_separating_slab",
              "scope":"continuous curves inside declared flight-center domain",
              "x_between_start_and_goal":True} if kind=="blocked" else None)}


def camera_rays(height=128,width=160,K=None):
    K=np.asarray(CONFIG["K"] if K is None else K,dtype=np.float64)
    v,u=np.indices((height,width),dtype=np.float64)
    optical=np.stack(((u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],np.ones_like(u)),-1)
    return optical@np.asarray(CONFIG["optical_to_B0_rotation"]).T


def raycast(scene,height=128,width=160,K=None):
    rays=camera_rays(height,width,K)
    origin=np.asarray(CONFIG["camera_origin_B0_m"])
    nearest=np.full((height,width),np.inf)
    hit_id=np.full((height,width),-1,dtype=np.int16)
    for i,box in enumerate(scene["obstacles"]):
        lo=np.asarray(box["min"])-origin;hi=np.asarray(box["max"])-origin
        parallel=np.abs(rays)<1e-14
        inv=np.divide(1.,rays,out=np.zeros_like(rays),where=~parallel)
        t0=lo*inv;t1=hi*inv
        enter=np.where(parallel,-np.inf,np.minimum(t0,t1))
        leave=np.where(parallel,np.inf,np.maximum(t0,t1))
        impossible=np.any(parallel & ((lo>0)|(hi<0)),axis=-1)
        near=enter.max(-1);far=leave.min(-1)
        # Nearest surface of a forward solid; origin-inside returns its exit.
        value=np.where(near>=0,near,far)
        valid=(far>=np.maximum(near,0)) & ~impossible & (value>=0)
        update=valid & (value<nearest)
        nearest=np.where(update,value,nearest);hit_id=np.where(update,i,hit_id)
    valid=np.isfinite(nearest)&(nearest>CONFIG["depth_range_m"][0])&(nearest<CONFIG["depth_range_m"][1])
    metric=np.where(valid,nearest,0.).astype(np.float32)
    hit_id=np.where(valid,hit_id,-1)
    network=np.stack((metric/20.,valid.astype(np.float32))).astype(np.float32)
    return metric,valid,network,hit_id


def point_box_distance(points,box):
    p=np.asarray(points,dtype=np.float64)
    gap=np.maximum(np.maximum(np.asarray(box["min"])-p,p-np.asarray(box["max"])),0.)
    return np.linalg.norm(gap,axis=-1)


def hull_box_distance(controls,box):
    lo=np.asarray(controls).min(0);hi=np.asarray(controls).max(0)
    gap=np.maximum(np.maximum(np.asarray(box["min"])-hi,lo-np.asarray(box["max"])),0.)
    return float(np.linalg.norm(gap))
