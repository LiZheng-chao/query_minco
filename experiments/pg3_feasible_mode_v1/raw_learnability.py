"""Bounded single-Query raw diagnosis, not network training or generalization."""
import time,traceback,json
from pathlib import Path
import numpy as np
import torch
from query_minco.training.io import read,write,sha,digest,require
from query_minco.training.pg1.spec import qualify_bindings
from query_minco.training.pilot_evaluation import load_rows
from query_minco.inference.identity import state_sha
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance
from query_minco.training.pg3_feasible_mode_v1.spec import PREFIX,recipe
from query_minco.training.pg3_feasible_mode_v1.teacher_labels import TeacherBank,inverse_raw
from query_minco.training.pg3_feasible_mode_v1.objective import augment
from query_minco.trajectory import TrajectoryPostprocessor
from experiments.pg3_feasible_mode_v1.common import ROOT,OUT,model_from,dev_field,raw_base,query_metrics

def run():
 prep=OUT/"preparation001";cal=read(prep/"CALIBRATION.json");gate=read(prep/"CPU0.json")
 require(gate["passed"],"PG3_RAW_CPU0");ds,geo,init,q=qualify_bindings(cal["bindings"],42)
 model=model_from(init).eval();before=state_sha(model)
 out=OUT/"raw_learnability001";out.mkdir(parents=True,exist_ok=False)
 old=ROOT/"research_outputs/pg2_guided_v1/runs/A42_C1_PG2Frozen384_CPU0_run001/summary/DEEP_QUERY_REPLAY.json"
 original=read(old);deep=sorted({(r["sample_id"],int(r["query"])) for r in original["rows"] if r.get("geometry") and r["geometry"]["minimum_signed_surface_clearance_m"]<-.120001})
 require(len(deep)==14,"PG3_DEEP14")
 dev={s.sample_id:(s,m) for s,m in load_rows(read(cal["bindings"]["path"])["development"])}
 bank=TeacherBank(recipe());context=TeacherGuidance();diagnostics=[]
 for sid,query in deep:
  sample,meta=dev[sid];field=dev_field(meta);raw=model(depth_input=sample.depth_input[None],state=sample.state[None])
  base=raw_base(raw,sample,field,cal["old_weights"],context)
  z=augment(base,[sample],bank,PREFIX+"_route",cal["new"]["weights"],preferred={0:query})
  mode=z["per_query"]["mode"][0,query];bneck=z["per_query"]["bottleneck"][0,query]
  mg=torch.autograd.grad(mode,raw,retain_graph=True)[0]
  bg=torch.autograd.grad(bneck,raw,retain_graph=True)[0]
  total=mode+bneck+z["per_query"]["convex_corridor"][0,query]
  cg,dg=torch.autograd.grad(total,(z["output"].trajectory.coeffs,z["output"].parameters.durations),retain_graph=True,allow_unused=True)
  layers={}
  gg=torch.autograd.grad(total,list(model.parameters()),retain_graph=True,allow_unused=True)
  for (n,p),g in zip(model.named_parameters(),gg):
   if g is not None:layers[n.split(".")[0]]=layers.get(n.split(".")[0],0.)+float(g.detach().square().sum())
  assignment=z["assignments"][0];m=assignment["route_teacher_id"][query]
  detail=dict(sample_id=sid,query=query,M=len(sample.teacher_coeffs),matched=bool(z["matched"][0,query]),assignment=assignment,assigned_teacher=m,
   metrics=query_metrics(z["output"],sample,field,query),route_record=z["records"][0][query],mode_raw_waypoint_gradient=mg[0,query,:6].tolist(),bottleneck_raw_waypoint_gradient=bg[0,query,:6].tolist(),
   mode_raw_norm=float(mg[0,query].norm()),bottleneck_raw_norm=float(bg[0,query].norm()),route_coefficient_norm=float(cg[0,query].norm()) if cg is not None else None,
   route_duration_norm=float(dg[0,query].norm()) if dg is not None else None,layer_norms={k:v**.5 for k,v in layers.items()},
   duration_raw_jacobian=([float(torch.autograd.grad(z["output"].parameters.total_time[0,query],raw,retain_graph=True)[0][0,query,15])]),
   mode_coefficient_gradient="STRUCTURALLY_NONE_DIRECT_PARAMETER_FORMULA")
  if m>=0:
   target=z["banks"][0]["teachers"][m]
   encoded=raw.detach().clone();encoded[0,query]=encoded.new_tensor(inverse_raw(target["inner"],target["tail"],target["durations"],scores=encoded[0,query,19:].tolist()))
   reconstructed=TrajectoryPostprocessor()(encoded,sample.state[None])
   detail["inverse_teacher_reconstruction"]=query_metrics(reconstructed,sample,field,query)
  diagnostics.append(detail)
 write(out/"DEEP14_INITIAL_GRADIENTS.json",diagnostics)
 representatives=[]
 for M in (1,4):
  representatives.extend([(r["sample_id"],r["query"]) for r in diagnostics if r["M"]==M][:2])
 representatives=list(dict.fromkeys(representatives))
 for key in deep:
  if len(representatives)>=4:break
  if key not in representatives:representatives.append(key)
 spec=dict(schema="pg3_raw_diagnostic_v1",queries=representatives,source_deep14=dict(path=str(old),sha256=sha(old)),max_updates_per_case=128,first_window=32,
  methods=["safe","mode","bottleneck","convex_corridor"],optimizer=dict(name="AdamW",lr=.03,weight_decay=0.,clip_norm=10.),
  early_stop="minimum32 updates and two successive8-update hard+Goal checks PASS; same rule all arms",
  frozen_network_weights=before,raw19_20_frozen=True,network_optimizer_steps=0,
  warning="No imitation-ratio guard applies to standalone raw19 diagnosis with zero unmatched imitation; network training retains original >10/two-batch guard")
 write(out/"RUN_SPEC.json",spec);write(out/"AUTHORIZATION.json",dict(approved=True,authorized_by="LiZheng-chao",spec_sha256=digest(spec),scope="USER_AUTHORIZED_BOUNDED_RAW_LEARNABILITY_ONLY"))
 results=[];started=time.monotonic()
 for sid,query in representatives:
  sample,meta=dev[sid];field=dev_field(meta)
  with torch.no_grad():initial=model(depth_input=sample.depth_input[None],state=sample.state[None]).detach()
  for method in spec["methods"]:
   case=out/(sid+"_Q"+str(query)+"_"+method);case.mkdir(exist_ok=False);context=TeacherGuidance()
   variable=torch.nn.Parameter(initial[0,query,:19].clone());opt=torch.optim.AdamW([variable],lr=.03,weight_decay=0.)
   snapshots=[];success_streak=0;completed=0
   try:
    for step in range(129):
     require(time.monotonic()-started<1800,"PG3_RAW_WALL")
     raw=initial.clone();raw[0,query,:19]=variable
     base=raw_base(raw,sample,field,cal["old_weights"],context)
     weights=dict(cal["new"]["weights"])
     if method=="bottleneck":weights["convex_corridor"]=0.
     z=augment(base,[sample],bank,PREFIX+("_mode" if method=="mode" else "_route"),weights,preferred={0:query}) if method!="safe" else None
     loss=z["loss"] if z is not None else base["loss"]
     if step%8==0 or step==128:
      output=base["output"];metrics=query_metrics(output,sample,field,query)
      assigned=z["assignments"][0] if z is not None else None
      snapshot=dict(step=step,loss=float(loss.detach()),metrics=metrics,assignment=assigned,raw19=variable.detach().tolist(),raw_l2=float((variable.detach()-initial[0,query,:19]).norm()))
      if z is not None:snapshot["route"]=z["records"][0][query]
      snapshots.append(snapshot)
      if step in (0,32,64,128):
       with (case/f"raw_step{step:04d}.npz").open("xb") as f:np.savez(f,raw=raw.detach().numpy(),coeffs=output.trajectory.coeffs.detach().numpy(),durations=output.parameters.durations.detach().numpy())
      success_streak=success_streak+1 if metrics["hard_pass"] and metrics["goal"]["passed"] else 0
      if step>=32 and success_streak>=2:break
     if step==128:break
     opt.zero_grad(set_to_none=True);loss.backward()
     require(variable.grad is not None and bool(torch.isfinite(variable.grad).all()),"PG3_RAW_GRAD")
     torch.nn.utils.clip_grad_norm_([variable],10.);opt.step();completed+=1
    result=dict(status="COMPLETED_DIAGNOSTIC",sample_id=sid,query=query,method=method,updates=completed,network_updates=0,snapshots=snapshots,success=snapshots[-1]["metrics"]["hard_pass"] and snapshots[-1]["metrics"]["goal"]["passed"])
   except Exception:
    result=dict(status="FAIL_DIAGNOSTIC",sample_id=sid,query=query,method=method,updates=completed,error=traceback.format_exc(),snapshots=snapshots,network_updates=0)
   write(case/"RESULT.json",result);results.append(result);print("PG3_RAW_CASE",sid,query,method,completed,result.get("success"),flush=True)
 require(state_sha(model)==before,"PG3_RAW_NETWORK_MUTATION")
 write(out/"RESULTS.json",dict(results=results,network_weights_before=before,network_weights_after=state_sha(model),network_optimizer_steps=0,raw_optimizer_steps=sum(r["updates"] for r in results),wall_s=time.monotonic()-started))
 write(out/"RUN_SEAL.json",dict(files={str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.is_file()}))
if __name__=="__main__":run()
