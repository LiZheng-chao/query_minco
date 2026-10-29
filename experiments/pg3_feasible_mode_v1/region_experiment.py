"""Frozen Region CPU0 and source-bound point-control reuse."""
import subprocess,types,importlib.util
import torch
from query_minco.training.io import read,write,sha,artifact,require
from query_minco.training.run_spec import ROOT,source_files
from query_minco.training.pg1.spec import qualify_bindings
from query_minco.training.initialization import configure
from query_minco.training.pg3_feasible_mode_v1.objective import forward
from query_minco.training.pg3_feasible_mode_v1.feasible_section import RegionBottleneck,configuration
from query_minco.training.pg3_feasible_mode_v1.spec import PREFIX,recipe
from query_minco.training.pg3_feasible_mode_v1.teacher_labels import TeacherBank
from query_minco.training.pg3_feasible_mode_v1.adaptive_assignment import AdaptiveAssignment
from query_minco.training.pg3_feasible_mode_v1.gradients import audit
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance
from query_minco.training.pg1_v2_r12.diagnostics import flat
from query_minco.inference.identity import state_sha
from query_minco.trajectory import TrajectoryPostprocessor
from experiments.pg3_feasible_mode_v1.common import OUT,model_from
R=OUT/"feasible_section001";P=OUT/"adaptive_assignment001"
def prepare():
 require(not subprocess.check_output(["git","status","--porcelain"],text=True).strip(),"REGION_CPU0_CLEAN")
 p=R/"preparation002";p.mkdir(exist_ok=False);old=read(P/"preparation002/CPU0.json");b=read(artifact(old["bindings"]))
 b["code"]=dict(git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),files=source_files(),allow_uncommitted_sources=False)
 write(p/"BINDINGS.json",b);ref=dict(path=str(p/"BINDINGS.json"),sha256=sha(p/"BINDINGS.json"));ds,geo,init,q=qualify_bindings(ref,42);write(p/"QUALIFICATION.json",q)
 configure(dict(seed=42,threads=1,device="cpu"));m=model_from(init);m.train();before=state_sha(m);samples=ds.load(old["ids"]);pp=list(m.parameters());bank=TeacherBank(recipe())
 historical=types.ModuleType("sealed_point_objective");historical.__package__="query_minco.training.pg3_feasible_mode_v1"
 exec(compile(subprocess.check_output(["git","show","f57d7e6988a9b06aece0b705ea8171cb74f6d7f6:query_minco/training/pg3_feasible_mode_v1/objective.py"],text=True),"<sealed point objective>","exec"),historical.__dict__)
 rng=torch.get_rng_state();a=historical.forward(m,samples,geo,PREFIX+"_route",old["old_weights"],old["new_weights"],TeacherBank(recipe()),TeacherGuidance(),0,assignment_policy=AdaptiveAssignment())
 ag=flat(torch.autograd.grad(a["loss"],pp),pp)
 torch.set_rng_state(rng);point=forward(m,samples,geo,PREFIX+"_route",old["old_weights"],old["new_weights"],bank,TeacherGuidance(),0,assignment_policy=AdaptiveAssignment());pg=flat(torch.autograd.grad(point["loss"],pp),pp)
 require(torch.equal(a["loss"],point["loss"]) and torch.equal(ag,pg) and a["assignments"]==point["assignments"],"REGION_POINT_CHANGED")
 torch.set_rng_state(rng);policy=RegionBottleneck();region=forward(m,samples,geo,PREFIX+"_route",old["old_weights"],old["new_weights"],bank,TeacherGuidance(),0,assignment_policy=AdaptiveAssignment(),bottleneck_policy=policy)
 rec,_,_=audit(m,region,details=True);require(rec["physical_imitation_ratio"]<=10,"REGION_INITIAL_GUARD");require(point["assignments"]==region["assignments"],"REGION_CHANGED_ASSIGNMENT")
 require(all(torch.equal(point["base"]["terms"][k],region["base"]["terms"][k]) for k in point["base"]["terms"]),"REGION_CHANGED_OTHER_TERMS")
 for k in ("mode","convex_corridor","qa","qj","qt","lateral","diversity"):require(torch.equal(point["terms"][k],region["terms"][k]),"REGION_CHANGED_"+k)
 targets=[]
 for key,data in bank.cache.items():
  for teacher in data["teachers"]:
   for label in teacher["bottlenecks"]:
    if label["valid"]:
     from query_minco.training.pg3_feasible_mode_v1.feasible_section import build_region
     proof,status=build_region(teacher,label,data["scene"],data["empty_certified"]);targets.append(dict(teacher_identity=teacher["cache_identity"],teacher=teacher["teacher"],landmark=label,proof=proof,status=status))
 write(p/"REGION_CERTIFICATES.json",targets)
 # Q0/Q2/Q3 independent frozen trajectory probes; no target tuning.
 dsunique=ds.load(list(dict.fromkeys(old["ids"])));i=next(i for i,s in enumerate(dsunique) if s.sample_id=="f01_down_only_g0_v0");sample=dsunique[i];field=geo.load([sample])[0];teacher=bank.get(sample,field)["teachers"][0]
 hdsp=importlib.util.spec_from_file_location("section_diag",OUT/"section_alignment001/diagnose_readonly.py");hd=importlib.util.module_from_spec(hdsp);hdsp.loader.exec_module(hd)
 probes=[]
 for step in (0,32):
  import numpy as np
  ar=np.load(P/"evaluation"/f"C_training_step{step:04d}.npz");raw=torch.tensor(ar[f"raw_{i}"],requires_grad=True);out=TrajectoryPostprocessor()(raw,sample.state[None])
  for qid in (0,2,3):
   c=out.trajectory.coeffs[0,qid];dt=out.parameters.durations[0,qid];v,diag=policy(c,dt,teacher,bank.get(sample,field));g=torch.autograd.grad(v,raw,retain_graph=True)[0][0,qid]
   pointpos,_=hd.true_point(c,dt,teacher["bottlenecks"][0]);record=dict(step=step,query=qid,region_loss=float(v.detach()),region_raw_gradient_norm=float(g.norm()),point_loss=diag["original_teacher_point_loss"],point_error=diag["lateral_error_m"],valid_regions=diag["count"],z_before=float(pointpos[2].detach()),raw_score_grad_max=float(g[19:21].abs().max()))
   require(diag["count"]>0 and record["raw_score_grad_max"]==0.,"REGION_TARGET_OR_SCORE")
   if step==0 and qid==0:
    require(float(g.norm())>0,"REGION_Q0_ZERO")
    trial=raw.detach().clone();trial[0,qid]-=.01*g/g.norm();after=TrajectoryPostprocessor()(trial,sample.state[None]);pos,_=hd.true_point(after.trajectory.coeffs[0,qid],after.parameters.durations[0,qid],teacher["bottlenecks"][0])
    record["z_after_negative_raw_probe"]=float(pos[2]);require(float(pos[2])<record["z_before"],"REGION_Q0_WRONG_DIRECTION")
   if step==32 and qid in (2,3):require(float(v.detach())==0. and float(g.norm())==0.,"REGION_PULLS_ALREADY_INSIDE")
   probes.append(record)
 write(p/"Q0_Q2_Q3_PROBES.json",probes)
 require(state_sha(m)==before==old["weights_before"],"REGION_CPU0_WEIGHTS")
 cal=dict(old,bindings=ref,code=b["code"],weights_before=before,weights_after=state_sha(m),region_cpu0_passed=True,region_config=configuration(),region_gradient_record=rec,baseline_equivalence=True,passed=True)
 write(p/"CPU0.json",cal);write(p/"RUN_SEAL.json",dict(files={x.name:sha(x) for x in p.iterdir() if x.is_file()},optimizer_steps=0));print("REGION_CPU0_PASS",rec["physical_imitation_ratio"],len(targets),sum(x["proof"] is not None for x in targets),probes,flush=True)
if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser();p.add_argument("command",choices=["prepare","counterfactual","train"]);a=p.parse_args()
 if a.command=="prepare":prepare()
 elif a.command=="counterfactual":
  from .region_counterfactual import run
  run()
 else:
  from .learnability_repair import run
  run("R",root=R)
