"""Task metrics, independent of trajectory construction and network outputs."""
import numpy as np
from .goal_config import CONFIG
def task_metrics(endpoint,terminal_velocity,goal):
 p=np.asarray(endpoint,float);v=np.asarray(terminal_velocity,float);g=np.asarray(goal,float);d=float(np.linalg.norm(g))
 if d<=0 or not np.isfinite(np.r_[p,v,g]).all():raise ValueError("finite nonzero local goal required")
 unit=g/d;remaining=float(np.linalg.norm(p-g));reduction=d-remaining;projected=float(p@unit);lateral=float(np.linalg.norm(p-projected*unit))
 cosine=projected/max(float(np.linalg.norm(p)),1e-12)
 near=d<=CONFIG["near_distance_threshold_m"]
 if near:
  reasons=(["near_goal_endpoint"] if remaining>CONFIG["near_endpoint_tolerance_m"] else [])+(["near_goal_stop"] if np.linalg.norm(v)>CONFIG["near_terminal_speed_max"] else [])
 else:
  reasons=[]
  for failed,name in [(reduction<1.,"insufficient_distance_reduction"),(projected<1.,"insufficient_projected_progress"),(cosine<.5,"wrong_progress_direction"),(lateral>.8,"excessive_goal_ray_offset")]:
   if failed:reasons.append(name)
 return {"mode":"near_arrival" if near else "far_progress","passed":not reasons,"reasons":reasons,"goal_distance_m":d,"endpoint_goal_error_m":remaining,
  "distance_reduction_m":reduction,"projected_progress_m":projected,"terminal_ray_offset_m":lateral,"progress_direction_cosine":cosine,"terminal_speed_mps":float(np.linalg.norm(v))}
def expert_tail(state):
 s=np.asarray(state,np.float32);g=s[6:9];d=np.linalg.norm(g);unit=g/d
 near=d<=CONFIG["near_distance_threshold_m"]
 p=g.copy() if near else unit*CONFIG["expert_far_endpoint_radius_m"]
 v=np.zeros(3,np.float32) if near else unit*CONFIG["expert_far_terminal_speed_mps"]
 return np.stack([p,v,np.zeros(3,np.float32)]).astype(np.float32)
def separating_slab_proof(scene,state):
 if scene["kind"]!="blocked":return None
 g=np.asarray(state[6:9],float);box=next(b for b in scene["obstacles"] if b["name"]=="separating_slab")
 domain=np.asarray(scene["flight_center_domain_m"],float);xmin=box["min"][0]
 covers=box["min"][1]<=domain[0,1] and box["max"][1]>=domain[1,1] and box["min"][2]<=domain[0,2] and box["max"][2]>=domain[1,2]
 xmax=xmin-CONFIG["body_radius_m"] # body collision only, no reliance on required margin
 nearest=np.minimum(np.maximum(g,domain[0]),[xmax,domain[1,1],domain[1,2]])
 lower=float(np.linalg.norm(g-nearest));max_reduction=float(np.linalg.norm(g))-lower
 near=np.linalg.norm(g)<=CONFIG["near_distance_threshold_m"]
 impossible=(lower>CONFIG["near_endpoint_tolerance_m"]) if near else max_reduction<CONFIG["far_min_distance_reduction_m"]
 valid=bool(covers and 0<xmax<g[0] and impossible)
 return {"proven_infeasible":valid,"reason":"continuous curve must cross body-inflated full-section slab to satisfy task",
  "domain":domain.tolist(),"pre_slab_component_x_max_m":xmax,"best_remaining_distance_lower_bound_m":lower,
  "distance_reduction_upper_bound_m":max_reduction,"near_arrival":bool(near),"obstacle":box,"no_expert_search_needed":valid}
