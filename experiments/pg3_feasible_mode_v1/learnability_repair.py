"""PG3 repair factorial A/B; no unsupported Task gate, no automatic extension."""
import os,time,resource,subprocess,traceback,csv,math,signal
from pathlib import Path
import numpy as np
import torch
from query_minco.training.io import read,write,sha,artifact,digest,require
from query_minco.training.run_spec import source_files,verify_code,ROOT
from query_minco.training.pg1.spec import qualify_bindings
from query_minco.training.initialization import configure,head_rows,rows_unchanged
from query_minco.training.pg1.state_io import presets,atomic_json,inspect_commit
from query_minco.training.pg1_v2_r12.evaluation import evaluate
from query_minco.training.pilot_evaluation import metric_alarm
from query_minco.training.pg3_feasible_mode_v1.state_io import save_commit,restore,lr_at
from query_minco.training.pg3_feasible_mode_v1.objective import forward
from query_minco.training.pg3_feasible_mode_v1.teacher_labels import TeacherBank
from query_minco.training.pg3_feasible_mode_v1.spec import PREFIX,recipe
from query_minco.training.pg3_feasible_mode_v1.gradients import audit
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance
from query_minco.inference.identity import state_sha
from experiments.pg3_feasible_mode_v1.common import OUT,model_from
R=OUT/"learnability_repair001"
def prepare():
 p=R/"preparation002";p.mkdir(exist_ok=False)
 require(not subprocess.check_output(["git","status","--porcelain"],text=True).strip(),"PG3_REPAIR_CLEAN")
 old=read(OUT/"preparation002/CALIBRATION.json");b=read(artifact(old["bindings"]))
 b["code"]=dict(git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),files=source_files(),allow_uncommitted_sources=False)
 write(p/"BINDINGS.json",b);binding=dict(path=str(p/"BINDINGS.json"),sha256=sha(p/"BINDINGS.json"))
 ds,geo,init,q=qualify_bindings(binding,42);write(p/"QUALIFICATION.json",q)
 configure(dict(seed=42,threads=1,device="cpu"));model=model_from(init);model.train();before=state_sha(model)
 bank=TeacherBank(recipe());ctx=TeacherGuidance();samples=ds.load(old["overfit_ids"])
 z=forward(model,samples,geo,PREFIX+"_route",old["old_weights"],old["new"]["weights"],bank,ctx,0)
 priority={}
 for b,s in enumerate(samples):
  if s.sample_id in priority:continue
  bad=[q for q in range(6) if not bool(z["matched"][b,q]) and not bool(z["labels"]["passed"][b,q])]
  signed=z["base"]["safe"]["signed_samples"][b]
  priority[s.sample_id]=sorted(bad,key=lambda q:(float(signed[q].detach().min()) if signed is not None else 1.,q))
 records={}
 for arm in ("A","B"):
  zz=z if arm=="A" else forward(model,samples,geo,PREFIX+"_route",old["old_weights"],old["new"]["weights"],bank,ctx,0,fixed_priorities=priority)
  rec,_,_=audit(model,zz,details=True);require(rec["physical_imitation_ratio"]<=10,"PG3_REPAIR_INITIAL_GUARD");records[arm]=rec
 require(before==state_sha(model),"PG3_REPAIR_CPU0_UPDATED")
 cal=dict(bindings=binding,code=read(artifact(binding))["code"],weights_before=before,weights_after=state_sha(model),optimizer_created=False,optimizer_steps=0,old_weights=old["old_weights"],new_weights=old["new"]["weights"],ids=old["overfit_ids"],fixed_priority=priority,arms=records,passed=True)
 write(p/"CPU0.json",cal);write(p/"RUN_SEAL.json",dict(files={str(x.relative_to(p)):sha(x) for x in p.rglob("*") if x.is_file()}))
 print("REPAIR_CPU0",priority,flush=True)
def run(arm,root=None):
 R=OUT/"learnability_repair001" if root is None else Path(root)
 require(arm in ("A","B","C","R"),"NO_UNSUPPORTED_TASK_GATE")
 policy=None;region=None
 if arm in ("C","R"):
  from query_minco.training.pg3_feasible_mode_v1.adaptive_assignment import AdaptiveAssignment
  policy=AdaptiveAssignment()
 if arm=="R":
  from query_minco.training.pg3_feasible_mode_v1.feasible_section import RegionBottleneck
  region=RegionBottleneck()
 cal=read(R/"preparation002/CPU0.json");verify_code(cal["code"]);ds,geo,init,qual=qualify_bindings(cal["bindings"],42)
 rid="A42_PG3_Learn_"+("Rotation" if arm=="A" else ("Fixed" if arm=="B" else ("Adaptive" if arm=="C" else "FeasibleRegion")))+"_CPU32_run001";rp=R/"runs"/rid;rp.mkdir(parents=True,exist_ok=False)
 spec=dict(schema="pg3_repair_cpu32_v1",method_id=PREFIX+("_learn_rotation_v1" if arm=="A" else ("_learn_fixed_v1" if arm=="B" else ("_m1_adaptive_assignment_v1" if arm=="C" else "_feasible_section_region_v1"))),run_id=rid,
  source=cal["code"],cpu0=dict(path=str(R/"preparation002/CPU0.json"),sha256=sha(R/"preparation002/CPU0.json")),bindings=cal["bindings"],
  initialization=init,selected_ids=cal["ids"],fixed_priority=cal["fixed_priority"],arm=arm,seed=42,device="cpu",precision="FP32",threads=1,batch_size=8,
  max_updates=32,checkpoint_steps=[0,8,16,32],schedule="original_cosine_2e-5_1e-6_32",optimizer=dict(name="AdamW",betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_norm=10.),
  old_weights=cal["old_weights"],new_weights=cal["new_weights"],task_strategy="UNCHANGED_PG2_DISCRETE_BLOCK_MASK",quality_enabled=False,ranking=False,
  resources=dict(wall_s=1800,rss_mib=4096,min_available_mib=3072,disk_gib=4),extension="NOT_REGISTERED_NO_AUTOMATIC_EXTENSION")
 if policy is not None:
  spec["adaptive_assignment"]=policy.cfg
  require(root is not None and cal["adaptive_config"]==policy.cfg and cal["baseline_equivalence"] is True,"ADAPTIVE_PREREQUISITES")
 if region is not None:
  gatepath=R/"COUNTERFACTUAL_GATE.json";require(read(gatepath)["passed"] is True and cal["region_cpu0_passed"] is True,"REGION_ADMISSION")
  spec["bottleneck_target"]=region.cfg;spec["region_gate"]=dict(path=str(gatepath),sha256=sha(gatepath));spec["head_projection"]=False
 auth=dict(approved=True,authorized_by="LiZheng-chao",spec_sha256=digest(spec),run_id=rid,max_optimizer_updates=32,device="cpu")
 write(rp/"RUN_SPEC.json",spec);write(rp/"AUTHORIZATION.json",auth);write(rp/"QUALIFICATION.json",qual)
 configure(dict(seed=42,threads=1,device="cpu"));model=model_from(init);before=state_sha(model);require(before==init["weights_sha256"],"REPAIR_INIT")
 opt=torch.optim.AdamW(model.parameters(),lr=lr_at(spec["schedule"],0),betas=(.9,.999),eps=1e-8,weight_decay=0.)
 score=head_rows(model);bank=TeacherBank(recipe());ctx=TeacherGuidance();samples=ds.load(cal["ids"]);start=time.monotonic();streak=0;history=[];last=None;stop=False
 def request(*a):
  nonlocal stop
  stop=True
 oldsig=signal.signal(signal.SIGTERM,request)
 try:
  baseline=evaluate(model,read(artifact(cal["bindings"]))["development"]);write(rp/"DEV_STEP0000.json",baseline)
  last=save_commit(rp,model,opt,spec,0,dict(completed_optimizer_updates=0,optimizer_step_executed=False));presets(rp,last,0)
  for step in range(32):
   available=int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")))/1024
   require(not stop and time.monotonic()-start<1800 and available>=3072 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<=4096,"REPAIR_RESOURCE")
   require(sum(p.stat().st_size for p in rp.rglob("*") if p.is_file())<4*1024**3,"REPAIR_DISK")
   require(source_files()==spec["source"]["files"] and os.environ.get("CUDA_VISIBLE_DEVICES")=="" and torch.get_num_threads()==1,"REPAIR_RUNTIME_SOURCE")
   model.train();opt.zero_grad(set_to_none=True)
   z=forward(model,samples,geo,PREFIX+"_route",cal["old_weights"],cal["new_weights"],bank,ctx,step,fixed_priorities=cal["fixed_priority"] if arm=="B" else None,assignment_policy=policy,bottleneck_policy=region)
   rec,vec,total=audit(model,z,details=True);ratio=rec["physical_imitation_ratio"]
   require(bool(torch.isfinite(total).all()) and math.isfinite(ratio),"REPAIR_NONFINITE")
   streak=streak+1 if ratio>10 else 0
   if streak>=2:
    write(rp/"REJECTED_UPDATE.json",dict(attempt=step+1,ratio=ratio,streak=streak,optimizer_step_executed=False,components=rec));raise ValueError("TWO_STEP_GRADIENT_RATIO_GT10")
   off=0
   for n,p in model.named_parameters():
    p.grad=total[off:off+p.numel()].reshape_as(p).clone();off+=p.numel()
    if n.startswith("trajectory_head.") and p.shape[0]==21:require(not torch.count_nonzero(p.grad[19:21]),"REPAIR_SCORE_ROWS")
   gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.));require(math.isfinite(gn),"REPAIR_CLIP")
   for g in opt.param_groups:g["lr"]=lr_at(spec["schedule"],step)
   rec.update(completed_optimizer_updates=step+1,ids=cal["ids"],assignments=z["assignments"],route_records=z["records"],matched=z["matched"].tolist(),
    hard_pass=z["labels"]["passed"].tolist(),goal_pass=z["goal_labels"].tolist(),M=[len(s.teacher_coeffs) for s in samples],regime=[ds.rows[s.sample_id]["regime"] for s in samples],
    weighted_losses={k:float(v.detach())*z["weights"][k] for k,v in z["terms"].items()},loss=float(z["loss"].detach()),ratio_alarm_streak=streak,lr=opt.param_groups[0]["lr"],gradnorm=gn)
   arrays=dict(raw=z["raw"].detach().numpy(),inner=z["output"].parameters.inner_pts.detach().numpy(),coeffs=z["output"].trajectory.coeffs.detach().numpy(),durations=z["output"].parameters.durations.detach().numpy(),
    mode=z["per_query"]["mode"].detach().numpy(),bottleneck=z["per_query"]["bottleneck"].detach().numpy(),convex_corridor=z["per_query"]["convex_corridor"].detach().numpy())
   folder=rp/"intents";folder.mkdir(exist_ok=True)
   with (folder/f"step{step+1:04d}_ARRAYS.npz").open("xb") as f:np.savez_compressed(f,**arrays)
   atomic_json(folder/f"step{step+1:04d}_STARTED.json",dict(parent=sha(last/"COMMIT.json"),arrays_sha256=sha(folder/f"step{step+1:04d}_ARRAYS.npz")))
   if policy is not None:
    from query_minco.training.pg1_v2_r12.diagnostics import flat
    qgrad=[]
    for b,sample in enumerate(samples):
     selected=z["assignments"][b]["selected"];scale=8*max(len(selected),1)
     for q in selected:
      value=z["per_query"]["mode"][b,q]*z["weights"]["mode"]/scale
      gg=flat(torch.autograd.grad(value,list(model.parameters()),retain_graph=True,allow_unused=True),list(model.parameters()))
      qgrad.append(dict(batch_row=b,sample_id=sample.sample_id,query=q,weighted_mode_parameter_norm=float(gg.norm())))
    rec["adaptive_selected_gradient_norms"]=qgrad
    rec["adaptive_state_before"]=__import__("copy").deepcopy(policy.state)
   opt.step();require(rows_unchanged(model,score) and all(bool(torch.isfinite(p).all()) for p in model.parameters()),"REPAIR_PARAMETERS")
   if step+1 in spec["checkpoint_steps"]:
    ev=evaluate(model,read(artifact(cal["bindings"]))["development"]);metric_alarm(history,ev,baseline,{})
    rec["development"]=ev
   if policy is not None:rec["adaptive_state_after"]=policy.commit()
   last=save_commit(rp,model,opt,spec,step+1,rec,parent_hash=sha(last/"COMMIT.json"))
   if step+1 in spec["checkpoint_steps"]:presets(rp,last,step+1)
   print("REPAIR_UPDATE",arm,step+1,ratio,flush=True)
  blob,c=restore(last,model,opt,spec,False);require(c["step"]==32 and state_sha(model)==blob["weights_sha256"],"REPAIR_FINAL")
  result=dict(status="COMPLETED",optimizer_updates=32,wall_s=time.monotonic()-start,weights_before=before,weights_after=state_sha(model),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
  write(rp/"RESULT.json",result)
 except BaseException:
  write(rp/"FAILURE_RECORD.json",dict(error=traceback.format_exc(),last_committed=str(last),retry=False));raise
 finally:
  signal.signal(signal.SIGTERM,oldsig)
  write(rp/"RUN_SEAL.json",dict(files={str(p.relative_to(rp)):sha(p) for p in rp.rglob("*") if p.is_file()},status="COMPLETED" if (rp/"RESULT.json").exists() else "FAIL"))
if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser();p.add_argument("command",choices=["prepare","A","B"]);a=p.parse_args()
 prepare() if a.command=="prepare" else run(a.command)
