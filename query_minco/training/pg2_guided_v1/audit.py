"""Bounded full Frozen384 CPU0. No optimizer is instantiated."""
import os,json,time,resource,traceback,subprocess
from pathlib import Path
import numpy as np
import torch
from query_minco.models import QueryMincoPlanner
from query_minco.inference.identity import state_sha
from query_minco.training.io import read,sha,digest,artifact,require
from query_minco.training.run_spec import ROOT,source_files
from query_minco.training.pg1.spec import qualify_bindings
from query_minco.training.pg1_v2_r12.objective import forward as r12_forward
from query_minco.training.initialization import configure
from query_minco.training.pilot_evaluation import load_rows
from query_minco.evaluation.signed_geometry import fine_geometry
from .spec import FULL,recipe
from .geometry import TeacherGuidance
from .objective import forward
from .gradients import audit_gradients

OUTPUT=ROOT/"research_outputs/pg2_guided_v1"
def strict(x):
 if isinstance(x,dict):return {str(k):strict(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [strict(v) for v in x]
 if isinstance(x,(np.integer,np.bool_)):return x.item()
 if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
 return x
def write(path,obj):
 with Path(path).open("x") as f:json.dump(strict(obj),f,sort_keys=True,indent=2,allow_nan=False)
def resource_gate(start,cap):
 require(time.monotonic()-start<cap,"PG2_CPU0_WALL")
 require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<=4096,"PG2_CPU0_RSS")
 mem=int(next(l.split()[1] for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:")))/1024
 require(mem>=3072,"PG2_CPU0_MEM_AVAILABLE")
def run(output,wall_cap_s=1800):
 output=Path(output).resolve()
 require(output.parent==OUTPUT/"runs" and output.name.startswith("A42_"),"PG2_OUTPUT_SCOPE")
 require(type(wall_cap_s) is int and 0<wall_cap_s<=1800,"PG2_CPU0_WALL_CONTRACT")
 output.mkdir(parents=True,exist_ok=False)
 for n in ("audit","arrays","summary"):(output/n).mkdir()
 start=time.monotonic();weights=None;model=None
 try:
  require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","PG2_CPU_HIDDEN")
  configure(dict(threads=1,seed=42,device="cpu"))
  original=ROOT/"research_outputs/pg1_v2_r12/cpu0_clean002/_preparation/CPU0_ONLY_BINDINGS.json"
  raw=read(original);basehead=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
  raw["code"]=dict(git_commit=basehead,files=source_files(),allow_uncommitted_sources=True)
  write(output/"audit/CPU0_BINDINGS.json",raw)
  binding=dict(path=str(output/"audit/CPU0_BINDINGS.json"),sha256=sha(output/"audit/CPU0_BINDINGS.json"))
  ds,geo,init,qualification=qualify_bindings(binding,42)
  write(output/"audit/DATA_QUALIFICATION.json",qualification)
  model=QueryMincoPlanner()
  model.load_state_dict(torch.load(str(artifact(init["checkpoint"])),map_location="cpu",weights_only=True)["state_dict"],strict=True)
  model.train();weights=state_sha(model)
  require(weights==init["weights_sha256"],"PG2_C1_WEIGHTS")
  context=TeacherGuidance();covered=set();exposures=0;records=[];streak=0;maxstreak=0
  write(output/"RUN_SPEC.json",dict(schema="pg2_guided_cpu0_v1",method=FULL,recipe=recipe(),
   binding=binding,initialization=init,optimizer_created=False,optimizer_steps=0,wall_cap_s=wall_cap_s,
   source_clean=False,source_files=raw["code"]["files"],original_binding_sha256=sha(original)))
  with (output/"BATCHES.jsonl").open("x") as ledger:
   for i in range(len(geo.orders[42].ids)):
    resource_gate(start,wall_cap_s)
    ids,roles=geo.orders[42].batch(i);samples=ds.load(ids)
    z=forward(model,samples,geo,context=context)
    rec,rg,pg,total=audit_gradients(model,z,groups=True)
    # Independent old-R0/Rtail arithmetic remains exact, including old gradients.
    old=r12_forward(model,samples,geo,"query_minco_pg1_v2_r12_r0_v1")
    require(torch.equal(old["raw"],z["raw"]) and torch.equal(old["loss"],z["base"]["loss"]),"PG2_OLD_R0_REGRESSION")
    require(all(torch.equal(old["terms"][k],z["base"]["terms"][k]) for k in z["base"]["terms"]),"PG2_OLD_TERMS_REGRESSION")
    ratio=rec["physical_imitation_ratio"];streak=streak+1 if ratio is not None and ratio>10 else 0;maxstreak=max(maxstreak,streak)
    rec.update(batch=i,ids=ids,roles=roles,M=[len(s.teacher_coeffs) for s in samples],
     regime=[ds.rows[s.sample_id]["regime"] for s in samples],
     losses={k:float(v.detach()) for k,v in z["terms"].items()},
     matched=z["matched"].tolist(),hard_pass=z["labels"]["passed"].tolist(),
     teacher_assignments=z["route"]["assignments"],teacher_records=z["route"]["records"],
     retreat_reasons=z["safe"]["reasons"],ratio_alarm_streak=streak,
     original_r0_bitwise=True,weights_unchanged=state_sha(model)==weights,
     score_argmax_hard_pass=[bool(z["labels"]["passed"][b,int(z["raw"][b,:,19].detach().argmax())]) for b in range(8)])
    arrays=dict(raw=z["raw"].detach().numpy(),coeffs=z["output"].trajectory.coeffs.detach().numpy(),
     durations=z["output"].parameters.durations.detach().numpy(),matched=z["matched"].numpy(),
     hard_pass=z["labels"]["passed"].numpy(),teacher_targets=z["route"]["target"].numpy())
    arrays.update({"grad_"+k:g.numpy() for k,g in rg.items()})
    arrays.update({"loss_"+k:v.detach().numpy() for k,v in z["per_query"].items()})
    with (output/"arrays"/("batch%03d.npz"%i)).open("xb") as f:np.savez_compressed(f,**arrays)
    write(output/"audit"/("HARD_BATCH_%03d.json"%i),dict(ids=ids,binding=z["labels"]["binding"],checks=z["labels"]["checks"]))
    ledger.write(json.dumps(strict(rec),allow_nan=False)+"\n");ledger.flush();records.append(rec)
    covered.update(ids);exposures+=8
    if i%8==0:print("PG2_CPU0",i+1,"unique",len(covered),"physical/imitation",round(ratio,4) if ratio else ratio,flush=True)
    del z,old,pg,total
    if len(covered)==384:break
  require(len(covered)==384,"PG2_FULL384_INCOMPLETE")
  require(source_files()==raw["code"]["files"],"PG2_SOURCE_DRIFT")
  require(state_sha(model)==weights,"PG2_CPU0_WEIGHT_MUTATION")
  write(output/"audit/COVERAGE.json",dict(unique_tasks=384,task_ids=sorted(covered),
   ids_sha256=digest(sorted(covered)),task_exposures=exposures,batches=len(records),
   order=next(o for o in raw["orders"] if o["seed"]==42)))
  # Mechanism-only readonly replay of the sealed Rtail step32, never for tuning.
  replay_development(model,raw,geo,output,context,start,wall_cap_s)
  result=dict(status="CPU0_COMPLETED",passed=True,optimizer_created=False,optimizer_steps=0,
   unique_tasks=384,task_exposures=exposures,batches=len(records),weights_before=weights,weights_after=state_sha(model),
   training_qualified=False,training_guard_max_consecutive=maxstreak,training_guard_passed=maxstreak<2,
   source_files=source_files(),wall_s=time.monotonic()-start,
   rss_peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
   limitations=["CPU0 is not training benefit","research source is not clean/frozen","open-dev8 is mechanism diagnostic only"])
  require(result["weights_after"]==weights,"PG2_FINAL_WEIGHTS")
  write(output/"RESULT.json",result)
  files={str(p.relative_to(output)):sha(p) for p in output.rglob("*") if p.is_file()}
  write(output/"RUN_SEAL.json",dict(schema="pg2_cpu0_evidence_seal_v1",files=files,
   purpose="CPU0_ONLY_NOT_TRAINING_AUTHORIZATION",optimizer_steps=0))
  print("PG2_CPU0_DONE",result,flush=True)
 except BaseException:
  write(output/"FAILURE_RECORD.json",dict(status="FAIL",exception=traceback.format_exc(),
   optimizer_created=False,optimizer_steps=0,weights_before=weights,
   weights_after=state_sha(model) if model is not None else None))
  raise

def replay_development(model,bindings,geo,output,context,start,cap):
 from types import SimpleNamespace
 from query_minco.trajectory import TrajectoryPostprocessor
 from query_minco.training.losses.set_physical_risk import segment_samples
 from query_minco.training.losses.imitation import cost_matrix
 from query_minco.training.matching.hungarian import match
 from .geometry import hard_labels
 from .losses import safety_loss,corridor_loss
 checkpoint=ROOT/"research_outputs/pg1_v2_r12/pilot_runs/A42_Rtail_R12_CPU_Pilot32_run002/checkpoints/checkpoint_step0032.pt"
 blob=torch.load(str(checkpoint),map_location="cpu",weights_only=True)
 replica=QueryMincoPlanner();replica.load_state_dict(blob["state_dict"],strict=True);replica.eval()
 require(state_sha(replica)==blob["weights_sha256"],"PG2_REPLAY_CHECKPOINT")
 rows=[]
 for sample,meta in load_rows(bindings["development"]):
  resource_gate(start,cap)
  # Original dev metadata remains DEV; no fabricated train row or train adapter.
  field=SimpleNamespace(row=dict(obstacles=meta["scene"]["obstacles"],
   flight_center_domain_m=meta["scene"]["flight_center_domain_m"],empty_certified=not meta["scene"]["obstacles"]),
   boxes=torch.tensor([[b["min"],b["max"]] for b in meta["scene"]["obstacles"]],dtype=torch.float32).reshape(-1,2,3))
  raw=replica(depth_input=sample.depth_input[None],state=sample.state[None])
  out=TrajectoryPostprocessor()(raw,sample.state[None]);p,v,w=segment_samples(out.trajectory,64)
  costs,_=cost_matrix(out,sample,0);_,qq,tt=match(costs,"B_set")
  matches=[dict(queries=qq.tolist(),teachers=tt.tolist())]
  route=context.build([sample],[field],matches,p,w)
  safe=safety_loss(p,v,w,[field],recipe())
  corridor=corridor_loss(p,w,route)
  labels=hard_labels(out,[sample],[field])
  grads={name:torch.autograd.grad(value,p,retain_graph=True,allow_unused=True)[0] for name,value in
   [("clearance",safe["clearance"].mean()),("corridor",corridor.mean()),("retreat",safe["retreat"].mean())]}
  for q in range(6):
   geom=fine_geometry(out.trajectory.coeffs[0,q].detach().numpy(),
    out.parameters.durations[0,q].detach().numpy(),meta["scene"],points=2049) if meta["scene"]["obstacles"] else None
   row=dict(sample_id=sample.sample_id,query=q,matched=q in qq.tolist(),hard_pass=bool(labels["passed"][0,q]),
    hard_reasons=labels["checks"][0][q]["reasons"],geometry=geom,assigned_teacher=route["assignments"][0][q],
    losses=dict(clearance=float(safe["clearance"][0,q]),retreat=float(safe["retreat"][0,q]),corridor=float(corridor[0,q])),
    position_gradient_norms={k:float(g[0,q].norm()) if g is not None else None for k,g in grads.items()})
   if grads["corridor"] is not None:
    g=grads["corridor"][0,q];direction=route["target"][0,q]-p[0,q].detach()
    row["negative_corridor_gradient_dot_teacher_route"]=float((-g*direction).sum())
   rows.append(row)
 write(output/"summary/DEEP_QUERY_REPLAY.json",dict(mode="OFFLINE_DIAGNOSTIC_NOT_TRAINING",
  checkpoint=dict(path=str(checkpoint),sha256=sha(checkpoint),weights_sha256=blob["weights_sha256"]),
  rows=rows,weights_unchanged=state_sha(replica)==blob["weights_sha256"]))
