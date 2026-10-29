"""V2 score-row policy and bounded Pilot adapter. CPU0 never invokes training.

Persistence/checkpoints reuse existing primitives. PG1 run_engine and its frozen
score assertions are NEVER patched or monkeypatched.
"""
import os,time,resource,signal
from pathlib import Path
import numpy as np
import torch
from query_minco.training.io import require,read,artifact,digest,sha,write
from query_minco.training.run_spec import ROOT,verify_code,source_files
from query_minco.training.initialization import head_rows,rows_unchanged,configure
from query_minco.training.pg1.spec import qualify_bindings,lr_at
from query_minco.training.pg1.state_io import atomic_json,save_commit,presets
from query_minco.training.pilot_evaluation import metric_alarm
from query_minco.training.pg1_v2_r12.evaluation import evaluate
from query_minco.models import QueryMincoPlanner
from query_minco.inference.identity import state_sha
from .spec import METHODS,BASELINE,score_rows_allowed,weights_for_method,COMPONENTS
from .objective import forward
from .gradients import measure,total_gradient

def score_guard(model,initial,method):
 allowed=score_rows_allowed(method)
 if not allowed:
  require(rows_unchanged(model,initial),"V2_FROZEN_SCORE_ROWS")
 else:
  require(method in METHODS and len(initial)==2,"V2_SCORE_METHOD")
  for n,p in model.named_parameters():
   if n in initial:
    for row in (19,20):
     if row not in allowed:require(torch.equal(p.detach()[row],initial[n][row-19]),"V2_UNAUTHORIZED_SCORE_ROW")
 require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),"V2_FINITE_WEIGHTS")
 return True

def gradient_score_guard(model,method):
 allowed=score_rows_allowed(method)
 for n,p in model.named_parameters():
  if n.startswith("trajectory_head.") and p.shape[0]==21:
   require(p.grad is not None and bool(torch.isfinite(p.grad).all()),"V2_HEAD_GRAD_FINITE")
   for row in (19,20):
    if row not in allowed:require(not torch.count_nonzero(p.grad[row]),"V2_UNAUTHORIZED_SCORE_GRAD")
 return True

def preflight_identity(s):
 return digest({k:v for k,v in s.items() if k!="preflight"})

def validate_training_spec(s,authorization):
 require(s.get("schema")=="pg2_v2_pilot32_spec_v1" and s.get("status")=="READY_FOR_MANUAL_PILOT","V2_DRAFT_NOT_RUNNABLE")
 from .spec import draft
 expected=draft(s["bindings"],s["code"],s["calibration"],s["method"],s["run_id"],device=s["device"])
 expected.update(schema="pg2_v2_pilot32_spec_v1",status="READY_FOR_MANUAL_PILOT",preflight=s["preflight"],cpu0=s["cpu0"],output_root=s["output_root"],initialization=s["initialization"])
 require(s==expected,"V2_EXACT_SPEC_FIELDS")
 require((s["method"] in METHODS or s["method"]==BASELINE) and s["updates"]==32 and s["checkpoints"]==[0,8,16,32] and s["seed"]==42 and s["batch_size"]==8 and s["precision"]=="FP32","V2_FIXED_PILOT")
 require(s["score_contract"]==dict(component_id="pg2_v2_score_row_policy_v1",allowed_rows=list(score_rows_allowed(s["method"]))),"V2_SCORE_CONTRACT")
 require(s["optimizer"]==dict(name="AdamW",betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_norm=10.) and s["schedule"]=="cosine_2e-5_to_1e-6_endpoint_inclusive","V2_OPTIMIZER")
 require(s["code"]["allow_uncommitted_sources"] is False,"V2_CLEAN_TRAINING_SOURCE")
 verify_code(s["code"])
 require(authorization==dict(schema="pg2_v2_single_run_authorization_v1",run_id=s["run_id"],spec_sha256=digest(s),
  max_optimizer_updates=32,approved=True,authorized_by=authorization.get("authorized_by")) and isinstance(authorization["authorized_by"],str) and bool(authorization["authorized_by"].strip()),"V2_SCOPE_AUTH")
 pf=read(artifact(s["preflight"]))
 require(pf.get("schema")=="pg2_v2_preflight_v1" and pf.get("optimizer_created") is False and pf.get("deterministic_repeat") is True and pf.get("passed") is True and pf.get("device")==s["device"] and pf.get("planning_spec_identity")==preflight_identity(s) and pf.get("optimizer_steps")==0,"V2_DEPLOYMENT_PREFLIGHT")
 require(set(pf["files"])=={"PREFLIGHT_ARRAYS.npz","QUALIFICATION.json"} and pf.get("code")==s["code"] and pf.get("bindings")==s["bindings"] and pf.get("initial_weights_sha256")==s["initialization"]["weights_sha256"],"V2_PREFLIGHT_COMPLETE")
 for n,h in pf["files"].items():
  require(not Path(n).is_absolute() and ".." not in Path(n).parts and sha(artifact(s["preflight"]).parent/n)==h,"V2_PREFLIGHT_FILES")
 cal=read(artifact(s["calibration"]))
 cp=artifact(s["cpu0"]);seal=read(cp)
 require(seal.get("purpose")=="CPU0_ONLY_NOT_TRAINING_AUTHORIZATION","V2_CPU0_SEAL")
 for n,h in seal["files"].items():
  require(not Path(n).is_absolute() and ".." not in Path(n).parts and sha(cp.parent/n)==h,"V2_CPU0_FILES")
 require(seal["files"].get("CALIBRATION.json")==s["calibration"]["sha256"] and cal.get("unique_tasks")==384 and cal.get("exposures")==768,"V2_FULL_CALIBRATION")
 require(cal["gradient_guard_passed"] is True and cal["code"]==s["code"] and cal["bindings"]==s["bindings"],"V2_CALIBRATION_IDENTITY")
 expected_resources=(dict(wall_s=3600,rss_cap_mib=4096,min_available_mib=3072,threads=1,disk_cap_gib=4)
  if s["device"]=="cpu" else
  dict(wall_s=3600,rss_cap_mib=4096,min_available_mib=3072,gpu_mib=6144,disk_cap_gib=4))
 require(s["device"] in ("cpu","cuda") and s["resources"]==expected_resources,"V2_RESOURCES")
 return cal

def resources(s,start,run):
 require(time.monotonic()-start<3600,"V2_WALL")
 available=int(next(l.split()[1] for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:")))/1024
 require(available>=3072 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<=4096,"V2_HOST_MEMORY")
 require(source_files()==s["code"]["files"],"V2_SOURCE_DRIFT")
 require(sum(p.stat().st_size for p in run.rglob("*") if p.is_file())<4*1024**3,"V2_DISK")
 if s["device"]=="cuda":
  require(torch.cuda.is_available() and torch.cuda.max_memory_reserved()<=6144*1024**2,"V2_GPU")
 else:
  require(s["device"]=="cpu" and os.environ.get("CUDA_VISIBLE_DEVICES")=="" and
   torch.get_num_threads()==1 and s["resources"]["threads"]==1,"V2_CPU")

def _execute(spec_path,authorization_path,owner):
 s=read(spec_path);auth=read(authorization_path);cal=validate_training_spec(s,auth)
 root=(ROOT/"research_outputs/pg2_guided_v2/runs").resolve()
 require(Path(s["output_root"]).resolve()==root and s["run_id"].startswith("A42_") and all(c.isalnum() or c=="_" for c in s["run_id"]),"V2_OUTPUT_SCOPE")
 require(not (root/s["run_id"]).exists(),"V2_NO_REUSE")
 ds,geo,init,q=qualify_bindings(s["bindings"],42)
 require(init==s["initialization"],"V2_INITIALIZATION_BINDING")
 # No optimizer exists above this point.
 run=root/s["run_id"];run.mkdir(parents=True,exist_ok=False)
 write(run/"RUNTIME_OWNER.json",dict(owner=owner,pid=os.getpid()))
 claims=root.parent/"_authorization_claims";claims.mkdir(exist_ok=True)
 atomic_json(claims/(digest(s)+".json"),dict(authorization=auth))
 for n,v in [("RUN_SPEC.json",s),("ROOT_SPEC.json",s),("AUTHORIZATION.json",auth),("QUALIFICATION.json",q)]:write(run/n,v)
 configure(dict(threads=1,seed=42,device=s["device"]))
 model=QueryMincoPlanner();model.load_state_dict(torch.load(str(artifact(init["checkpoint"])),map_location="cpu",weights_only=True)["state_dict"],strict=True);model.to(s["device"])
 require(state_sha(model)==init["weights_sha256"],"V2_WEIGHTS_INIT")
 opt=torch.optim.AdamW(model.parameters(),lr=2e-5,betas=(.9,.999),eps=1e-8,weight_decay=0.)
 initial=head_rows(model);start=time.monotonic();streak=0;history=[]
 from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance
 context=TeacherGuidance();weights=weights_for_method(cal["weights"],s["method"]) if s["method"]!=BASELINE else None;stop={"value":False}
 old=signal.signal(signal.SIGTERM,lambda *a:stop.update(value=True))
 (run/"intents").mkdir();baseline=evaluate(model,read(artifact(s["bindings"]))["development"])
 committed=save_commit(run,model,opt,s,0,dict(completed_optimizer_updates=0,optimizer_step_executed=False))
 presets(run,committed,0)
 try:
  for i in range(32):
   require(not stop["value"],"V2_STOP_REQUESTED");resources(s,start,run)
   ids,_=geo.orders[42].batch(i);samples=ds.load(ids);model.train();opt.zero_grad(set_to_none=True)
   if s["method"]==BASELINE:
    from query_minco.training.pg1.execution import update as original_update
    rec,arrays=original_update(model,opt,samples,geo,BASELINE,"pg1_pilot32_v1",i,streak)
    ratio=rec["physical_imitation_gradient_ratio"];streak=rec["ratio_alarm_streak"];norm=rec["global_grad_norm_before_clip"]
    coord=dict(projection_applied=False,baseline_original_update=True);loss_value=rec["loss"]
   else:
    z=forward(model,samples,geo,s["method"],context,weights)
    rec,pg,rg=measure(model,z,groups=True);g,coord=total_gradient(pg,weights,projection=False)
    physical=g-weights["imitation"]*pg["imitation"];im=float(pg["imitation"].norm());ratio=float(physical.norm())/im if im else (0. if not float(physical.norm()) else 1e30)
    streak=streak+1 if ratio>10 else 0;require(streak<2,"TWO_STEP_GRADIENT_RATIO_GT10")
    offset=0
    for p in model.parameters():p.grad=g[offset:offset+p.numel()].reshape_as(p).clone();offset+=p.numel()
    gradient_score_guard(model,s["method"]);norm=float(torch.nn.utils.clip_grad_norm_(list(model.parameters()),10.));require(np.isfinite(norm),"V2_CLIP")
    arrays=dict(raw=z["raw"].detach().cpu().numpy(),coeffs=z["output"].trajectory.coeffs.detach().cpu().numpy(),
     durations=z["output"].parameters.durations.detach().cpu().numpy(),matched=z["matched"].cpu().numpy())
    for k,v in rg.items():arrays["raw_gradient_"+k]=v.cpu().numpy()
    loss_value=float(z["loss"].detach())
   for group in opt.param_groups:group["lr"]=lr_at("pg1_pilot32_v1",i)
   resources(s,start,run)
   with (run/"intents"/("step%04d_ARRAYS.npz"%(i+1))).open("xb") as f:np.savez_compressed(f,**arrays)
   atomic_json(run/"intents"/("step%04d_STARTED.json"%(i+1)),dict(step=i+1,parent_sha256=sha(committed/"COMMIT.json")))
   opt.step()
   score_guard(model,initial,s["method"])
   rec.update(ids=ids,loss=loss_value,lr=lr_at("pg1_pilot32_v1",i),ratio_alarm_streak=streak,physical_imitation_ratio=ratio,
    gradient_clip_before=norm,completed_optimizer_updates=i+1,arrays_sha256=sha(run/"intents"/("step%04d_ARRAYS.npz"%(i+1))),coordination=coord)
   if i+1 in (8,16,32):
    ev=evaluate(model,read(artifact(s["bindings"]))["development"]);metric_alarm(history,ev,baseline,{})
    rec["development"]=ev
   committed=save_commit(run,model,opt,s,i+1,rec,parent_hash=sha(committed/"COMMIT.json"))
   if i+1 in (8,16,32):presets(run,committed,i+1)
   print("PG2_V2_UPDATE",i+1,rec["loss"],flush=True)
  write(run/"RESULT.json",dict(status="COMPLETED_AWAITING_AUDIT",optimizer_updates=32,final_weights_sha256=state_sha(model)))
 except BaseException as exc:
  write(run/"FAILURE_RECORD.json",dict(error=repr(exc),last_committed=str(committed),retry=False));raise
 finally:signal.signal(signal.SIGTERM,old)

def preflight(spec_path,output):
 """Manual deployment zero-update check. Never creates an optimizer."""
 import subprocess
 s=read(spec_path)
 require(s["schema"]=="pg2_v2_pilot32_spec_v1" and s["status"]=="READY_FOR_MANUAL_PILOT" and s["preflight"] is None,"V2_PREFLIGHT_SPEC")
 require(s["code"]["allow_uncommitted_sources"] is False,"V2_PREFLIGHT_CLEAN")
 verify_code(s["code"])
 cal=read(artifact(s["calibration"]))
 require(cal["gradient_guard_passed"] and cal["code"]==s["code"] and cal["bindings"]==s["bindings"],"V2_PREFLIGHT_CALIBRATION")
 output=Path(output).resolve()
 require(output.parent==ROOT/"research_outputs/pg2_guided_v2/_preparation","V2_PREFLIGHT_PATH")
 output.mkdir(parents=True,exist_ok=False)
 ds,geo,init,q=qualify_bindings(s["bindings"],42)
 require(init==s["initialization"],"V2_PREFLIGHT_INITIALIZATION")
 configure(dict(threads=1,seed=42,device=s["device"]))
 if s["device"]=="cuda":
  require(str(ROOT.resolve()).startswith("/data/") and torch.cuda.device_count()==1,"V2_SERVER_ROOT_AND_GPU")
  free,total=torch.cuda.mem_get_info();cap=min(6144*1024**2,int(total*.75))
  require(free>=cap+512*1024**2,"V2_GPU_FREE")
  torch.cuda.set_per_process_memory_fraction(max(0,cap-512*1024**2)/total)
 else:
  require(s["device"]=="cpu" and os.environ.get("CUDA_VISIBLE_DEVICES")=="" and
   torch.get_num_threads()==1 and s["resources"]==dict(wall_s=3600,rss_cap_mib=4096,
   min_available_mib=3072,threads=1,disk_cap_gib=4),"V2_PREFLIGHT_CPU")
 model=QueryMincoPlanner();model.load_state_dict(torch.load(str(artifact(init["checkpoint"])),map_location="cpu",weights_only=True)["state_dict"],strict=True);model.to(s["device"]);model.train()
 before=state_sha(model);ids,_=geo.orders[42].batch(0);samples=ds.load(ids);arrays={}
 for repeat in (0,1):
  if s["method"]==BASELINE:
   from query_minco.training.pg1.objective import forward as old_forward
   z=old_forward(model,samples,geo,BASELINE)
   gg=torch.autograd.grad(z["loss"],list(model.parameters()),allow_unused=True)
   require(all(g is not None and bool(torch.isfinite(g).all()) for g in gg),"V2_PREFLIGHT_BASELINE_GRAD")
  else:
   z=forward(model,samples,geo,s["method"],weights=weights_for_method(cal["weights"],s["method"]))
   _,pg,rg=measure(model,z,groups=False)
   g,_=total_gradient(pg,weights_for_method(cal["weights"],s["method"]))
   require(bool(torch.isfinite(g).all()),"V2_PREFLIGHT_GRAD")
  for name,value in [("raw",z["raw"]),("coeffs",z["output"].trajectory.coeffs),("durations",z["output"].parameters.durations),("loss",z["loss"])]:
   arrays[name+str(repeat)]=value.detach().cpu().numpy()
 require(all(np.array_equal(arrays[k+"0"],arrays[k+"1"]) for k in ("raw","coeffs","durations","loss")),"V2_PREFLIGHT_REPEAT")
 require(state_sha(model)==before==init["weights_sha256"],"V2_PREFLIGHT_WEIGHTS")
 with (output/"PREFLIGHT_ARRAYS.npz").open("xb") as f:np.savez_compressed(f,**arrays)
 write(output/"QUALIFICATION.json",q)
 receipt=dict(schema="pg2_v2_preflight_v1",passed=True,planning_spec_identity=preflight_identity(s),device=s["device"],
  optimizer_created=False,optimizer_steps=0,deterministic_repeat=True,code=s["code"],bindings=s["bindings"],initial_weights_sha256=before,
  files={n:sha(output/n) for n in ("PREFLIGHT_ARRAYS.npz","QUALIFICATION.json")})
 write(output/"PREFLIGHT.json",receipt)
 complete=dict(s,preflight=dict(path=str(output/"PREFLIGHT.json"),sha256=sha(output/"PREFLIGHT.json")))
 write(output/"RUN_SPEC_WITH_PREFLIGHT.json",complete)
 print("PREFLIGHT_ONLY",output,flush=True)


def execute(spec_path,authorization_path):
 import uuid,traceback
 owner=uuid.uuid4().hex
 try:return _execute(spec_path,authorization_path,owner)
 except BaseException:
  try:
   s=read(spec_path);run=(ROOT/"research_outputs/pg2_guided_v2/runs"/s["run_id"]).resolve()
   if run.parent==ROOT/"research_outputs/pg2_guided_v2/runs" and (run/"RUNTIME_OWNER.json").is_file() and read(run/"RUNTIME_OWNER.json").get("owner")==owner and not (run/"FAILURE_RECORD.json").exists():
    write(run/"FAILURE_RECORD.json",dict(error=traceback.format_exc(),retry=False))
  except Exception:pass
  raise
