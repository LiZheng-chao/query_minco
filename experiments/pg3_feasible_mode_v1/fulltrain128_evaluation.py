"""Complete model-only evaluation with unchanged original hard validator."""
import csv,json,time
import numpy as np
import torch
from query_minco.training.io import write,require,sha
from query_minco.trajectory import TrajectoryPostprocessor
from query_minco.training.losses.imitation import cost_matrix
from query_minco.training.matching.hungarian import match
from query_minco.evaluation.validation import validate_teacher,evaluate
from query_minco.evaluation.goal import task_metrics
from query_minco.evaluation.signed_geometry import fine_geometry
from query_minco.training.pg3_feasible_mode_v1.bottleneck import crossing_events,bottleneck_loss
from query_minco.training.pg3_feasible_mode_v1.losses import mode_query
from query_minco.training.pg3_feasible_mode_v1.spec import recipe
from query_minco.inference.identity import state_sha
ENCLOSURE={"left_wall","right_wall","floor","ceiling","backdrop"}
def integral(c,d,order):
 result=0.
 for cc,T in zip(np.asarray(c,float),np.asarray(d,float)):
  a=np.stack([cc[:,k]*__import__("math").factorial(k)/__import__("math").factorial(k-order) for k in range(order,6)],1)
  k=np.arange(a.shape[-1]);M=T**(k[:,None]+k[None,:]+1)/(k[:,None]+k[None,:]+1)
  result+=float(np.einsum("ai,ij,aj->",a,M,a))
 return result
def signature(c,d,scene):
 result=[]
 for i,box in enumerate(scene["obstacles"]):
  if box["name"] in ENCLOSURE:continue
  e=crossing_events(c,d,(box["min"][0]+box["max"][0])/2)
  if e["status"]!="UNIQUE_FORWARD_CROSSING":return None
  p=e["selected"]["position"];faces=[]
  for axis in (1,2):
   if p[axis]<=box["min"][axis]-.15:faces.append((box["min"][axis]-.15-p[axis],axis,-1))
   if p[axis]>=box["max"][axis]+.15:faces.append((p[axis]-box["max"][axis]-.15,axis,1))
  if not faces:return None
  _,axis,side=max(faces);result.append([box["name"],axis,side])
 return json.dumps(result if result else [["DIRECT_NO_INTERIOR"]],separators=(",",":"))
def evaluate_set(model,items,out,dataset,step,initial_targets=None):
 out.mkdir(parents=True,exist_ok=True);csvpath=out/(dataset+f"_step{step:04d}.csv");before=state_sha(model);was=model.training;model.eval()
 rows=[];raws=[];coeffs=[];durations=[];new_targets={};task_rows=[];start=time.monotonic()
 with torch.no_grad():
  for si,(sample,field,meta) in enumerate(items):
   raw=model(depth_input=sample.depth_input[None],state=sample.state[None]);pred=TrajectoryPostprocessor()(raw,sample.state[None])
   require(bool(torch.isfinite(raw).all()),"FULL_EVAL_RAW")
   costs,_=cost_matrix(pred,sample);_,qq,tt=match(costs,"B_set");matches={int(q):int(t) for q,t in zip(qq,tt)};matched=set(matches)
   scene=meta["scene"];group=[];modes=set()
   for q in range(6):
    c=pred.trajectory.coeffs[0,q].numpy();d=pred.parameters.durations[0,q].numpy()
    hard=validate_teacher(c,d,pred.parameters.head_pva[0,q].numpy(),pred.parameters.tail_pva[0,q].numpy(),pred.parameters.inner_pts[0,q].numpy(),scene)
    goal=task_metrics(evaluate(c[-1],d[-1]),evaluate(c[-1],d[-1],1),sample.state[6:9].numpy())
    signed=fine_geometry(c,d,scene,4097) if scene["obstacles"] else None
    distances=[float((pred.parameters.inner_pts[0,q]-torch.tensor(t["inner"],dtype=torch.float32)).square().sum()+(pred.parameters.tail_pva[0,q,0]-torch.tensor(t["tail"][0],dtype=torch.float32)).square().sum()) for t in meta["teachers"]]
    nearest=int(np.argmin(distances));key=dataset+"/"+sample.sample_id+"/"+str(q);fixed=int(initial_targets[key]) if initial_targets is not None else nearest;new_targets[key]=fixed
    target=meta["teachers"][fixed];bn,bdiag=bottleneck_loss(pred.trajectory.coeffs[0,q],pred.parameters.durations[0,q],target["bottlenecks"])
    mode,_=mode_query(pred.parameters.inner_pts[0,q],pred.parameters.tail_pva[0,q],pred.parameters.durations[0,q],target,recipe())
    route=signature(c,d,scene) if hard["passed"] and goal["passed"] else None
    if route is not None:modes.add(route)
    row=dict(dataset=dataset,step=step,sample_id=sample.sample_id,family=meta["family"],kind=scene["kind"],regime=meta["regime"],M=len(sample.teacher_coeffs),query=q,matched=q in matched,matched_teacher=matches.get(q),hard_pass=hard["passed"],hard_reasons=";".join(hard["reasons"]),goal_pass=goal["passed"],complete=hard["passed"] and goal["passed"],goal_mode=goal["mode"],goal_progress=goal["projected_progress_m"],goal_endpoint_error=goal["endpoint_goal_error_m"],goal_lateral=goal["terminal_ray_offset_m"],signed_clearance=signed["minimum_signed_surface_clearance_m"] if signed else None,closest_time=signed["closest_time_s"] if signed else None,closest_box=signed["closest_box"] if signed else None,
     vmax=hard["max_speed_mps"],amax=hard["max_acceleration_mps2"],jmax=hard["max_jerk_mps3"],qa=integral(c,d,2),qj=integral(c,d,3),time=float(d.sum()),route_signature=route,fixed_initial_teacher=fixed,nearest_teacher=nearest,bottleneck_error=float(np.mean(bdiag["lateral_error_m"])) if bdiag["count"] else None,bottleneck_count=bdiag["count"],mode_loss_potential=float(mode),inner=json.dumps(pred.parameters.inner_pts[0,q].tolist()),raw19=json.dumps(raw[0,q,:19].tolist()))
    rows.append(row);group.append(row)
   task_rows.append(dict(sample_id=sample.sample_id,oracle=any(x["complete"] for x in group),known_geometric_routes=len(modes)))
   raws.append(raw.numpy()[0]);coeffs.append(pred.trajectory.coeffs.numpy()[0]);durations.append(pred.parameters.durations.numpy()[0])
   if (si+1)%64==0:print("FULL_EVAL",dataset,step,si+1,len(items),flush=True)
 with csvpath.open("x",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with (out/(dataset+f"_step{step:04d}.npz")).open("xb") as f:np.savez_compressed(f,raw=np.stack(raws),coeffs=np.stack(coeffs),durations=np.stack(durations),ids=np.array([s.sample_id for s,f,m in items]))
 strata={}
 for M in (1,4):
  for matched in (True,False):
   rr=[r for r in rows if r["M"]==M and r["matched"]==matched]
   strata[str(M)+"/"+str(matched)]=dict(queries=len(rr),physics=sum(r["hard_pass"] for r in rr),goal=sum(r["goal_pass"] for r in rr),complete=sum(r["complete"] for r in rr))
 clearance=[r["signed_clearance"] for r in rows if r["signed_clearance"] is not None]
 result=dict(dataset=dataset,step=step,tasks=len(items),queries=len(rows),physics=sum(r["hard_pass"] for r in rows),goal=sum(r["goal_pass"] for r in rows),complete=sum(r["complete"] for r in rows),oracle=sum(t["oracle"] for t in task_rows),geometric_route_count=sum(t["known_geometric_routes"] for t in task_rows),route_semantics="original-hard+Goal candidates with complete internal-AABB section face signatures, NOT homotopy",selected_at6=None,selected_reason="ranking disabled; Oracle is not online selection",
  strata=strata,mean_qa=float(np.mean([r["qa"] for r in rows])),mean_qj=float(np.mean([r["qj"] for r in rows])),mean_time=float(np.mean([r["time"] for r in rows])),max_v=max(r["vmax"] for r in rows),max_a=max(r["amax"] for r in rows),max_j=max(r["jmax"] for r in rows),signed_clearance_quantiles=np.quantile(clearance,[0,.1,.5,.9,1]).tolist() if clearance else None,wall_s=time.monotonic()-start,weights_sha256=before,csv=dict(path=str(csvpath),sha256=sha(csvpath)),timing_scope="CPU evaluation; hard verifier separate from online inference")
 write(out/(dataset+f"_step{step:04d}_SUMMARY.json"),result)
 require(state_sha(model)==before,"FULL_EVAL_CHANGED_WEIGHTS");model.train(was)
 return result,new_targets
def inference_measure(model,sample):
 was=model.training;model.eval();network=[];decode=[]
 with torch.no_grad():
  for i in range(35):
   t=time.perf_counter();raw=model(depth_input=sample.depth_input[None],state=sample.state[None]);u=time.perf_counter();TrajectoryPostprocessor()(raw,sample.state[None]);v=time.perf_counter()
   if i>=5:network.append((u-t)*1000);decode.append((v-t)*1000)
 model.train(was)
 return dict(parameters=sum(p.numel() for p in model.parameters()),threads=torch.get_num_threads(),device="cpu",batch=1,samples=30,network_ms=dict(zip(["p50","p95","max"],np.quantile(network,[.5,.95,1]).tolist())),network_and_minco_ms=dict(zip(["p50","p95","max"],np.quantile(decode,[.5,.95,1]).tolist())),scope="CPU inference, not ROS realtime qualification")
