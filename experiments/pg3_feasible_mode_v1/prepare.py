"""Training-only PG3 calibration/CPU0. Fixed first eight FrozenOrder batches."""
import os,time,subprocess
from pathlib import Path
import numpy as np
import torch
from query_minco.training.io import read,write,sha,digest,artifact,require
from query_minco.training.run_spec import ROOT,source_files
from query_minco.training.pg1.spec import qualify_bindings
from query_minco.training.initialization import configure
from query_minco.inference.identity import state_sha
from query_minco.training.pg2_guided_v2.spec import PREFIX as OLD,weights_for_method
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance
from query_minco.training.pg3_feasible_mode_v1.spec import recipe,PREFIX,EXTRA
from query_minco.training.pg3_feasible_mode_v1.teacher_labels import TeacherBank
from query_minco.training.pg3_feasible_mode_v1.objective import forward
from query_minco.training.pg3_feasible_mode_v1.gradients import audit,calibrate
from experiments.pg3_feasible_mode_v1.common import OUT,model_from

def run():
 out=OUT/"preparation001";out.mkdir(parents=True,exist_ok=False)
 require(not subprocess.check_output(["git","status","--porcelain","--untracked-files=all"],text=True).strip(),"PG3_CLEAN_CPU0")
 configure(dict(threads=1,seed=42,device="cpu"))
 old=ROOT/"research_outputs/pg2_guided_v2/runs/A42_C1_V2_CleanCPU0_run003"
 binding=read(old/"audit/BINDINGS.json");binding["code"]=dict(git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),files=source_files(),allow_uncommitted_sources=False)
 write(out/"BINDINGS.json",binding);ref=dict(path=str(out/"BINDINGS.json"),sha256=sha(out/"BINDINGS.json"))
 ds,geo,init,q=qualify_bindings(ref,42);write(out/"QUALIFICATION.json",q)
 model=model_from(init);model.train();before=state_sha(model)
 oldcal=dict(path=str(old/"CALIBRATION.json"),sha256=sha(old/"CALIBRATION.json"));old_weights=weights_for_method(read(artifact(oldcal))["weights"],OLD+"safe_v1")
 context=TeacherGuidance();bank=TeacherBank(recipe());records=[];candidates=[];seen=set();all_arrays={}
 for i in range(8):
  ids,_=geo.orders[42].batch(i);samples=ds.load(ids)
  z=forward(model,samples,geo,PREFIX+"_control",old_weights,{k:1. for k in EXTRA},bank,context,0)
  rec,vectors,total=audit(model,z,details=True);rec.update(batch=i,ids=ids,M=[len(s.teacher_coeffs) for s in samples],regime=[ds.rows[s.sample_id]["regime"] for s in samples]);records.append(rec)
  for b,sample in enumerate(samples):
   if sample.sample_id in seen:continue
   seen.add(sample.sample_id)
   signed=z["base"]["safe"]["signed_samples"][b]
   bad=(~z["matched"][b])&(~z["labels"]["passed"][b])
   minima=signed.detach().min(-1).values if signed is not None else torch.ones(6)
   qids=torch.nonzero(bad&(minima<-.12),as_tuple=False).flatten().tolist()
   candidates.append(dict(id=sample.sample_id,M=len(sample.teacher_coeffs),regime=ds.rows[sample.sample_id]["regime"],deep_unmatched=qids,minimum=float(minima.min())))
  require(torch.equal(z["base"]["goal_diag"]["original_unmodified"],z["base"]["base"]["per_query"]["goal"]),"PG3_OLD_GOAL")
  print("PG3_CPU0_TRAIN_BATCH",i,rec["physical_imitation_ratio"],flush=True)
 new=calibrate(records)
 ratios={}
 from query_minco.training.pg2_guided_v2.gradients import guard_pass
 # Exact initial weighted recombination on the same fixed audit batches.
 for method in (PREFIX+"_mode",PREFIX+"_route",PREFIX+"_control"):
  rr=[]
  for i in range(8):
   ids,_=geo.orders[42].batch(i)
   z=forward(model,ds.load(ids),geo,method,old_weights,new["weights"],bank,context,0)
   rec,_,_=audit(model,z,details=False);rr.append(rec["physical_imitation_ratio"])
  require(guard_pass(rr),"PG3_INITIAL_GRADIENT_GUARD_"+method);ratios[method]=rr
 selected=[]
 for M in (1,4):
  options=[r for r in candidates if r["M"]==M and r["deep_unmatched"]]
  if len(options)<2:options=options+[r for r in candidates if r["M"]==M and r not in options]
  require(len(options)>=2,"PG3_NO_M1_M4_OVERFIT_COVERAGE")
  selected += [r["id"] for r in options[:2]]
 require(len(set(selected))==4,"PG3_OVERFIT_DISTINCT")
 selected_ids=selected*2
 model.eval()
 probe=forward(model,ds.load(selected_ids),geo,PREFIX+"_route",old_weights,new["weights"],bank,context,0)
 initial_route=float((probe["terms"]["mode"]+probe["terms"]["bottleneck"]).detach())
 cal=dict(schema="pg3_training_only_calibration_v1",code=binding["code"],bindings=ref,uses_development=False,new=new,old_calibration=oldcal,old_weights=old_weights,initial_guard_ratios=ratios,selected_training_tasks=selected,overfit_ids=selected_ids,overfit_initial_route_error=initial_route,audit_indices=list(range(8)),unique_tasks=len(seen))
 write(out/"CALIBRATION.json",cal);write(out/"CPU0_RECORDS.json",records);write(out/"TRAINING_TASK_SELECTION.json",dict(all_screened=candidates,selected=selected,rule="first two initial-deep unmatched M1 and M4 from fixed first8 FrozenOrder; fallback first M-group if fewer; no dev selection"))
 write(out/"TEACHER_LABELS.json",bank.cache)
 require(state_sha(model)==before==init["weights_sha256"],"PG3_CPU0_WEIGHT_UPDATE")
 gate=dict(schema="pg3_cpu0_v1",passed=True,code=binding["code"],bindings=ref,weights_before=before,weights_after=state_sha(model),optimizer_created=False,optimizer_steps=0,audit_batches=8,unique_tasks=len(seen),test_log=dict(path=str(OUT/"implementation001/TESTS_final.log"),sha256=sha(OUT/"implementation001/TESTS_final.log")),calibration=dict(path=str(out/"CALIBRATION.json"),sha256=sha(out/"CALIBRATION.json")))
 write(out/"CPU0.json",gate);write(out/"RUN_SEAL.json",dict(files={str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.is_file()},purpose="PG3_CPU0_NO_UPDATES"))
 print("PG3_CPU0_DONE",gate,"OVERFIT_TASKS",selected,flush=True)
if __name__=="__main__":run()
