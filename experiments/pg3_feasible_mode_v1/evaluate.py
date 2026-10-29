"""Read-only train-overfit and mechanism-dev evaluation of actual PG3 checkpoints."""
import csv,json
from pathlib import Path
import numpy as np
import torch
from query_minco.training.io import read,write,sha,require
from query_minco.training.pg1.spec import qualify_bindings
from query_minco.training.pilot_evaluation import load_rows
from query_minco.training.pg1_v2_r12.evaluation import evaluate
from query_minco.training.pg3_feasible_mode_v1.spec import PREFIX,recipe
from query_minco.training.pg3_feasible_mode_v1.teacher_labels import TeacherBank
from query_minco.training.pg3_feasible_mode_v1.objective import augment
from query_minco.training.pg3_feasible_mode_v1.bottleneck import bottleneck_loss
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance
from query_minco.models import QueryMincoPlanner
from query_minco.inference.identity import state_sha
from experiments.pg3_feasible_mode_v1.common import ROOT,OUT,dev_field,raw_base,query_metrics

def run():
 torch.set_num_threads(1);prep=OUT/"preparation002";cal=read(prep/"CALIBRATION.json")
 ds,geo,init,q=qualify_bindings(cal["bindings"],42)
 dev=load_rows(read(cal["bindings"]["path"])["development"])
 train=ds.load(cal["selected_training_tasks"]);fields=geo.load(train)
 datasets=dict(development=[(s,dev_field(m)) for s,m in dev],overfit_train=list(zip(train,fields)))
 out=OUT/"evaluation001";out.mkdir(exist_ok=False)
 rawdiag=read(OUT/"raw_learnability001/DEEP14_INITIAL_GRADIENTS.json") if (OUT/"raw_learnability001/DEEP14_INITIAL_GRADIENTS.json").exists() else []
 deep={(r["sample_id"],r["query"]):r for r in rawdiag};fixed={};initial={};records=[];summaries=[];states={}
 paths=[("overfit",OUT/"runs/A42_PG3_Overfit128_run001")]+[(a,OUT/f"runs/A42_{a}_PG3_CPU_Pilot32_run001") for a in "ABCD"]
 for arm,rp in paths:
  if not rp.exists():states[arm]="NOT_RUN";continue
  states[arm]=read(rp/"RESULT.json") if (rp/"RESULT.json").exists() else read(rp/"FAILURE_RECORD.json")
  target=out/arm;target.mkdir()
  for step in (0,8,16,32,64,128):
   cp=rp/"checkpoints"/f"checkpoint_step{step:04d}.pt"
   if not cp.exists():continue
   receipt=read(str(cp)+".json");require(sha(cp)==receipt["sha256"],"PG3_EVAL_CHECKPOINT_SHA")
   blob=torch.load(cp,map_location="cpu",weights_only=True);model=QueryMincoPlanner();model.load_state_dict(blob["state_dict"],strict=True);model.eval();before=state_sha(model)
   require(before==blob["weights_sha256"],"PG3_EVAL_WEIGHTS")
   ev=evaluate(model,read(cal["bindings"]["path"])["development"]);write(target/f"DEV_step{step:04d}.json",ev)
   bank=TeacherBank(recipe());context=TeacherGuidance()
   for kind,items in datasets.items():
    group=[];arrays={}
    for si,(sample,field) in enumerate(items):
     raw=model(depth_input=sample.depth_input[None],state=sample.state[None])
     base=raw_base(raw,sample,field,cal["old_weights"],context)
     z=augment(base,[sample],bank,PREFIX+"_control",cal["new"]["weights"],step=0)
     a=z["assignments"][0];teachers=z["banks"][0]["teachers"]
     key=(kind,sample.sample_id)
     if key not in initial:initial[key]=dict(raw=raw.detach().clone(),inner=z["output"].parameters.inner_pts.detach().clone())
     arrays[f"raw_{si}"]=raw.detach().numpy();arrays[f"coeffs_{si}"]=z["output"].trajectory.coeffs.detach().numpy();arrays[f"durations_{si}"]=z["output"].parameters.durations.detach().numpy()
     for query in range(6):
      qkey=(kind,sample.sample_id,query)
      if qkey not in fixed:fixed[qkey]=deep.get((sample.sample_id,query),{}).get("assigned_teacher",a["route_teacher_id"][query])
      m=fixed[qkey];metrics=query_metrics(z["output"],sample,field,query);signed=metrics["signed"]
      bd=None;way=None
      if m>=0 and m<len(teachers):
       t=teachers[m];_,bd=bottleneck_loss(z["output"].trajectory.coeffs[0,query],z["output"].parameters.durations[0,query],t["bottlenecks"])
       way=float((z["output"].parameters.inner_pts[0,query].detach()-torch.tensor(t["inner"])).norm())
      row=dict(arm=arm,step=step,dataset=kind,sample_id=sample.sample_id,query=query,M=len(sample.teacher_coeffs),matched=bool(z["matched"][0,query]),historical_deep14=(sample.sample_id,query) in deep,
       hard_pass=metrics["hard_pass"],hard_reasons="|".join(metrics["hard_reasons"]),goal_pass=metrics["goal"]["passed"],goal_progress=metrics["goal"]["projected_progress_m"],complete_pass=bool(metrics["hard_pass"] and metrics["goal"]["passed"]),
       signed_clearance=signed["minimum_signed_surface_clearance_m"] if signed else None,closest_time=signed["closest_time_s"] if signed else None,closest_position=json.dumps(signed.get("closest_position_B0_m")) if signed else None,
       qa_energy=metrics["acceleration_energy"],qj_energy=metrics["jerk_energy"],time_s=metrics["time_s"],vmax=metrics["vmax"],amax=metrics["amax"],jmax=metrics["jmax"],
       route_teacher=a["route_teacher_id"][query],initial_teacher=m,assignment_reason=a["reasons"][query],
       bottleneck_lateral_error_mean=float(np.mean(bd["lateral_error_m"])) if bd and bd["count"] else None,waypoint_error_to_fixed_teacher=way,
       raw_l2_delta=float((raw.detach()-initial[key]["raw"])[0,query].norm()),waypoint_l2_delta=float((z["output"].parameters.inner_pts.detach()-initial[key]["inner"])[0,query].norm()),
       route_record=json.dumps(z["records"][0][query],allow_nan=False),crossings=json.dumps(metrics["crossings"],allow_nan=False))
      group.append(row);records.append(row)
     del z,base,raw
    with (target/f"{kind}_step{step:04d}_TRAJECTORIES.npz").open("xb") as f:np.savez_compressed(f,**arrays)
    oracle=sum(any(r["complete_pass"] for r in group if r["sample_id"]==s.sample_id) for s,f in items)
    summary=dict(arm=arm,step=step,dataset=kind,queries=len(group),physics=sum(r["hard_pass"] for r in group),goal=sum(r["goal_pass"] for r in group),complete=sum(r["complete_pass"] for r in group),oracle=oracle,tasks=len(items),known_feasible_modes=ev["feasible_mode_count"] if kind=="development" else None,
     deep14_queries=sum(r["historical_deep14"] for r in group),deep14_pass=sum(r["historical_deep14"] and r["hard_pass"] for r in group),
     mean_qa=float(np.mean([r["qa_energy"] for r in group])),mean_qj=float(np.mean([r["qj_energy"] for r in group])),mean_time=float(np.mean([r["time_s"] for r in group])),max_v=max(r["vmax"] for r in group),max_a=max(r["amax"] for r in group),max_j=max(r["jmax"] for r in group))
    summaries.append(summary);print("PG3_FROZEN_EVAL",summary,flush=True)
   require(state_sha(model)==before,"PG3_EVAL_READONLY")
 with (out/"QUERY_RESULTS.csv").open("x",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 write(out/"SUMMARY.json",dict(states=states,summaries=summaries,signed_semantics="4097 per segment numeric signed SDF, not certificate",hard_semantics="unchanged original continuous verifier"))
 write(out/"RUN_SEAL.json",dict(files={str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.is_file()}))
if __name__=="__main__":run()
