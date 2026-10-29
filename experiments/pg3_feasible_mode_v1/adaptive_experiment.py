"""Source-bound one-batch CPU0 and exact frozen A/B equivalence."""
import os,subprocess,types
from pathlib import Path
import torch
from query_minco.training.io import read,write,sha,artifact,require
from query_minco.training.run_spec import source_files,ROOT
from query_minco.training.pg1.spec import qualify_bindings
from query_minco.training.initialization import configure
from query_minco.training.pg3_feasible_mode_v1.spec import PREFIX,recipe
from query_minco.training.pg3_feasible_mode_v1.teacher_labels import TeacherBank
from query_minco.training.pg3_feasible_mode_v1.adaptive_assignment import AdaptiveAssignment,configuration
from query_minco.training.pg3_feasible_mode_v1.objective import forward
from query_minco.training.pg3_feasible_mode_v1.gradients import audit
from query_minco.training.pg1_v2_r12.diagnostics import flat
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance
from query_minco.inference.identity import state_sha
from experiments.pg3_feasible_mode_v1.common import OUT,model_from
R=OUT/"adaptive_assignment001";OLD="0c75b3afe1b7b5e3d26fa033f84efda1afb356f9"
def run():
 require(not subprocess.check_output(["git","status","--porcelain"],text=True).strip(),"ADAPTIVE_CLEAN")
 p=R/"preparation002";p.mkdir(exist_ok=False)
 old=read(OUT/"learnability_repair001/preparation002/CPU0.json");b=read(artifact(old["bindings"]))
 b["code"]=dict(git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),files=source_files(),allow_uncommitted_sources=False)
 write(p/"BINDINGS.json",b);ref=dict(path=str(p/"BINDINGS.json"),sha256=sha(p/"BINDINGS.json"))
 ds,geo,init,qual=qualify_bindings(ref,42);write(p/"QUALIFICATION.json",qual)
 configure(dict(seed=42,threads=1,device="cpu"));model=model_from(init);model.train();before=state_sha(model)
 samples=ds.load(old["ids"]);pp=list(model.parameters())
 mod=types.ModuleType("legacy_pg3_objective");mod.__package__="query_minco.training.pg3_feasible_mode_v1"
 exec(compile(subprocess.check_output(["git","show",OLD+":query_minco/training/pg3_feasible_mode_v1/objective.py"],text=True),"<sealed PG3 objective>","exec"),mod.__dict__)
 eq={}
 for arm in ("A","B"):
  rng=torch.get_rng_state();kwargs=dict(fixed_priorities=old["fixed_priority"] if arm=="B" else None)
  a=mod.forward(model,samples,geo,PREFIX+"_route",old["old_weights"],old["new_weights"],TeacherBank(recipe()),TeacherGuidance(),0,**kwargs)
  ag=flat(torch.autograd.grad(a["loss"],pp),pp)
  torch.set_rng_state(rng)
  z=forward(model,samples,geo,PREFIX+"_route",old["old_weights"],old["new_weights"],TeacherBank(recipe()),TeacherGuidance(),0,**kwargs)
  zg=flat(torch.autograd.grad(z["loss"],pp),pp)
  require(torch.equal(a["loss"],z["loss"]) and torch.equal(ag,zg) and a["assignments"]==z["assignments"],"ADAPTIVE_CHANGED_LEGACY")
  eq[arm]=dict(loss_bitwise_equal=True,full_parameter_gradient_bitwise_equal=True,assignments_equal=True,loss=float(z["loss"]),gradient_norm=float(zg.norm()))
 policy=AdaptiveAssignment();z=forward(model,samples,geo,PREFIX+"_route",old["old_weights"],old["new_weights"],TeacherBank(recipe()),TeacherGuidance(),0,assignment_policy=policy)
 rec,_,_=audit(model,z,details=True);require(rec["physical_imitation_ratio"]<=10,"ADAPTIVE_INITIAL_GUARD")
 require(before==state_sha(model)==old["weights_before"],"ADAPTIVE_CPU0_WEIGHT_DRIFT")
 cal=dict(old,bindings=ref,code=b["code"],weights_before=before,weights_after=state_sha(model),adaptive_config=configuration(),baseline_equivalence=True,baseline_equivalence_records=eq,arms=dict(C=rec),passed=True)
 write(p/"CPU0.json",cal);write(p/"CONFIG_PREREG.json",configuration())
 for rp in (OUT/"learnability_repair001/runs").iterdir():
  sp=read(rp/"RUN_SPEC.json");require(sp["selected_ids"]==old["ids"] and sp["old_weights"]==old["old_weights"] and sp["new_weights"]==old["new_weights"] and sp["initialization"]["weights_sha256"]==before,"ADAPTIVE_BASELINE_REUSE")
  seal=read(rp/"RUN_SEAL.json");require(all(sha(rp/n)==h for n,h in seal["files"].items()),"ADAPTIVE_BASELINE_SEAL")
 write(p/"RUN_SEAL.json",dict(files={q.name:sha(q) for q in p.iterdir() if q.is_file()},optimizer_created=False,optimizer_steps=0))
 print("ADAPTIVE_CPU0_PASS",eq,rec["physical_imitation_ratio"],flush=True)
if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser();p.add_argument("command",choices=["prepare","train"]);a=p.parse_args()
 if a.command=="prepare":run()
 else:
  from .learnability_repair import run as train
  train("C",root=R)
