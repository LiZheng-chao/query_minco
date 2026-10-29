"""Full training-set frozen audit and fixed calibration. No optimizer."""
import os,time,json,traceback,resource,subprocess
from pathlib import Path
import numpy as np
import torch
from query_minco.models import QueryMincoPlanner
from query_minco.training.io import read,artifact,sha,require,digest
from query_minco.training.run_spec import ROOT,source_files
from query_minco.training.initialization import configure
from query_minco.training.pg1.spec import qualify_bindings
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance,tensor_digest,scene_from_field
from query_minco.training.pg2_guided_v1.audit import write,resource_gate
from query_minco.inference.identity import state_sha
from query_minco.evaluation.signed_geometry import fine_geometry
from .objective import forward
from .gradients import measure,calibrate,budget_record,total_gradient
from .spec import COMPONENTS,FULL,weights_for_method,METHODS
ROOTOUT=ROOT/"research_outputs/pg2_guided_v2"

def run(output,clean=False):
 output=Path(output).resolve();require(output.parent==ROOTOUT/"runs" and output.name.startswith("A42_"),"V2_CPU0_OUTPUT")
 if clean:require(not subprocess.check_output(["git","status","--porcelain"],text=True).strip(),"V2_CLEAN_REQUIRED")
 output.mkdir(parents=True,exist_ok=False)
 for n in ("audit","arrays","summary"):(output/n).mkdir()
 started=time.monotonic();model=None;before=None
 try:
  require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","V2_CPU_HIDDEN")
  configure(dict(threads=1,seed=42,device="cpu"))
  old_binding=ROOT/"research_outputs/pg1_v2_r12/cpu0_clean002/_preparation/CPU0_ONLY_BINDINGS.json"
  binding=read(old_binding);binding["code"]=dict(git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),files=source_files(),allow_uncommitted_sources=not clean)
  write(output/"audit/BINDINGS.json",binding)
  ref=dict(path=str(output/"audit/BINDINGS.json"),sha256=sha(output/"audit/BINDINGS.json"))
  ds,geo,init,qualification=qualify_bindings(ref,42);write(output/"audit/QUALIFICATION.json",qualification)
  model=QueryMincoPlanner();model.load_state_dict(torch.load(str(artifact(init["checkpoint"])),map_location="cpu",weights_only=True)["state_dict"],strict=True);model.train()
  before=state_sha(model);require(before==init["weights_sha256"],"V2_INIT")
  context=TeacherGuidance();rows=[];covered=set();cache={}
  v1root=ROOT/"research_outputs/pg2_guided_v1/runs/A42_C1_PG2Frozen384_CPU0_run001"
  v1rows=[json.loads(l) for l in (v1root/"BATCHES.jsonl").read_text().splitlines()]
  with (output/"BATCHES.jsonl").open("x") as ledger:
   for i in range(len(geo.orders[42].ids)):
    resource_gate(started,1800)
    ids,roles=geo.orders[42].batch(i);samples=ds.load(ids)
    z=forward(model,samples,geo,context=context);rec,pg,rg=measure(model,z,groups=True)
    require(torch.equal(z["goal_diag"]["original_unmodified"],z["base"]["per_query"]["goal"]),"V2_ORIGINAL_TASK_VALUES")
    for k,v in z["feasibility_diag"]["original_dynamic_terms"].items():require(torch.equal(v,z["base"]["dynamic_terms"][k]),"V2_ORIGINAL_DYNAMIC")
    require(ids==v1rows[i]["ids"],"V2_V1_ORDER")
    with np.load(v1root/"arrays"/("batch%03d.npz"%i),allow_pickle=False) as old:
     require(np.array_equal(z["raw"].detach().numpy(),old["raw"]) and np.array_equal(z["labels"]["passed"].numpy(),old["hard_pass"]),"V2_FROZEN_V1_REGRESSION")
    signed=[];details=[]
    for b,s in enumerate(samples):
     signed_row=[]
     for q in range(6):
      key=(s.sample_id,q,tensor_digest(z["output"].trajectory.coeffs[b,q],z["output"].parameters.durations[b,q]))
      if key not in cache:
       cache[key]=fine_geometry(z["output"].trajectory.coeffs[b,q].detach().numpy(),z["output"].parameters.durations[b,q].detach().numpy(),scene_from_field(z["fields"][b]),points=2049) if len(z["fields"][b].boxes) else None
      gg=cache[key];signed_row.append(gg["minimum_signed_surface_clearance_m"] if gg else None)
     signed.append(signed_row)
    rec.update(batch=i,ids=ids,roles=roles,M=[len(s.teacher_coeffs) for s in samples],
     regime=[ds.rows[s.sample_id]["regime"] for s in samples],matched=z["matched"].tolist(),
     hard_pass=z["labels"]["passed"].tolist(),goal_pass=z["goal_labels"].tolist(),
     signed_clearance=signed,safety_records=z["safe"]["records"],teacher_assignments=z["route"]["assignments"],
     original_task_bitwise=True,original_dynamic_bitwise=True,v1_raw_hard_bitwise=True,
     ranking=__import__("query_minco.training.pg2_guided_v2.ranking",fromlist=["metrics"]).metrics(z["raw"],z["labels"]["passed"],z["goal_labels"]))
    arrays=dict(raw=z["raw"].detach().numpy(),coeffs=z["output"].trajectory.coeffs.detach().numpy(),durations=z["output"].parameters.durations.detach().numpy(),
     matched=z["matched"].numpy(),hard_pass=z["labels"]["passed"].numpy())
    arrays.update({"raw_grad_"+k:g.numpy() for k,g in rg.items()})
    arrays.update({"loss_"+k:v.detach().numpy() for k,v in z["per_query"].items()})
    for k,v in z["quality"]["integrals_si"].items():arrays["energy_"+k]=v.detach().numpy()
    with (output/"arrays"/("batch%03d.npz"%i)).open("xb") as f:np.savez_compressed(f,**arrays)
    write(output/"audit"/("HARD_%03d.json"%i),dict(ids=ids,checks=z["labels"]["checks"],binding=z["labels"]["binding"]))
    ledger.write(json.dumps(rec,allow_nan=False)+"\n");ledger.flush();rows.append(rec);covered.update(ids)
    del z,pg,rg
    if i%8==0:print("V2_MEASURE",i+1,"unique",len(covered),flush=True)
    if len(covered)==384:break
  require(len(covered)==384 and len(rows)==96,"V2_COMPLETE384")
  cal=calibrate(rows);cal.update(code=binding["code"],bindings=ref,weights_sha256=before,
   task_ids_sha256=digest(sorted(covered)),unique_tasks=384,exposures=768)
  # Exact recombination checks at fixed training indices; no optimizer created.
  checks=[]
  for i in (0,47,95):
   resource_gate(started,1800)
   ids,_=geo.orders[42].batch(i);z=forward(model,ds.load(ids),geo,context=context,weights=cal["weights"])
   rr,pg,rg=measure(model,z,groups=False)
   actual,coord=total_gradient(pg,cal["weights"],False)
   physical=actual-cal["weights"]["imitation"]*pg["imitation"]
   ratio=float(physical.norm())/float(pg["imitation"].norm())
   expected=budget_record(rows[i]["gram"],cal["weights"])
   require(np.isclose(ratio,expected["ratio"],rtol=1e-4,atol=1e-6),"V2_GRAM_RECONSTRUCTION")
   projected,projection=total_gradient(pg,cal["weights"],True)
   direct=torch.autograd.grad(z["loss"],list(model.parameters()),retain_graph=True,allow_unused=True)
   from query_minco.training.pg1_v2_r12.diagnostics import flat
   require(torch.allclose(actual,flat(direct,list(model.parameters())),rtol=5e-4,atol=3e-6),"V2_TOTAL_GRAD_RECONSTRUCTION")
   checks.append(dict(batch=i,actual_ratio=ratio,gram_ratio=expected["ratio"],projection=projection))
  cal["recombination_checks"]=checks
  write(output/"CALIBRATION.json",cal)
  write(output/"audit/COVERAGE.json",dict(ids=sorted(covered),ids_sha256=digest(sorted(covered)),batches=96,exposures=768))
  require(source_files()==binding["code"]["files"] and state_sha(model)==before,"V2_IDENTITY_DRIFT")
  result=dict(numerical_passed=True,training_gradient_guard_passed=cal["gradient_guard_passed"],unique_tasks=384,batches=96,
   task_exposures=768,query_exposures=4608,weights_before=before,weights_after=state_sha(model),
   optimizer_created=False,optimizer_steps=0,wall_s=time.monotonic()-started,
   peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,training_qualified=False,
   limitations=["unclean research source","server preflight NOT_RUN","no training efficacy evidence"])
  write(output/"RESULT.json",result)
  write(output/"RUN_SEAL.json",dict(schema="pg2_v2_cpu0_seal_v1",files={str(p.relative_to(output)):sha(p) for p in output.rglob("*") if p.is_file()},purpose="CPU0_ONLY_NOT_TRAINING_AUTHORIZATION"))
  print("V2_CPU0_DONE",result,flush=True)
 except BaseException:
  write(output/"FAILURE_RECORD.json",dict(error=traceback.format_exc(),optimizer_steps=0,weights_before=before,weights_after=state_sha(model) if model else None));raise
