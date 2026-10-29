"""Exclusive CPU PG3 executor; original immutable checkpoints and stops are reused."""
import os,time,resource,signal,traceback
from pathlib import Path
import numpy as np
import torch
from query_minco.training.io import read,write,sha,digest,artifact,require
from query_minco.training.run_spec import ROOT,verify_code,source_files
from query_minco.training.initialization import configure,head_rows,rows_unchanged
from query_minco.training.pg1.spec import qualify_bindings,lr_at
from query_minco.training.pg1.state_io import atomic_json,presets
from .state_io import save_commit,lr_at as pg3_lr_at
from query_minco.training.pilot_evaluation import metric_alarm
from query_minco.training.pg1_v2_r12.evaluation import evaluate
from query_minco.models import QueryMincoPlanner
from query_minco.inference.identity import state_sha
from query_minco.training.pg2_guided_v2.objective import forward as old_forward
from query_minco.training.pg2_guided_v2.spec import PREFIX as OLD_PREFIX
from query_minco.training.pg1_v2_r12.diagnostics import flat
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance
from .spec import METHODS,recipe,EXTRA
from .teacher_labels import TeacherBank
from .objective import forward
from .gradients import audit
BASELINE=OLD_PREFIX+"safe_v1"
OUT=ROOT/"research_outputs/pg3_feasible_mode_v1"

def make_spec(binding,calibration,cpu0,method,run_id,kind="paired32",selected_ids=None):
 require(method in METHODS or method==BASELINE,"PG3_EXEC_METHOD")
 require(kind in ("paired32","overfit128"),"PG3_EXEC_PROFILE")
 b=read(artifact(binding))
 return dict(schema="pg3_cpu_experiment_v1",method=method,run_id=run_id,kind=kind,
  code=b["code"],bindings=binding,calibration=calibration,cpu0=cpu0,seed=42,device="cpu",precision="FP32",threads=1,batch_size=8,
  updates=32 if kind=="paired32" else 128,checkpoint_steps=[0,8,16,32] if kind=="paired32" else [0,32,64,128],
  optimizer=dict(name="AdamW",betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_norm=10.),
  schedule="original_cosine_2e-5_1e-6_32" if kind=="paired32" else "preregistered_cosine_2e-4_1e-5_128",
  selected_ids=selected_ids,recipe=recipe(),output_root=str(OUT/"runs"),
  resources=dict(wall_s=1800 if kind=="paired32" else 2400,rss_mib=4096,min_available_mib=3072,disk_gib=4),
  ranking=False,network_architecture="UNCHANGED_C1_QUERY_DECODER_21D_MINCO")

def authorization(s,approved=False,authorized_by=""):
 return dict(schema="pg3_single_run_authorization_v1",spec_sha256=digest(s),run_id=s["run_id"],max_updates=s["updates"],device="cpu",approved=approved,authorized_by=authorized_by)

def validate(s,a):
 expected=make_spec(s["bindings"],s["calibration"],s["cpu0"],s["method"],s["run_id"],s["kind"],s["selected_ids"])
 require(s==expected and s["code"]["allow_uncommitted_sources"] is False,"PG3_EXACT_FROZEN_SPEC")
 verify_code(s["code"])
 require(a==authorization(s,True,a.get("authorized_by","")) and bool(a["authorized_by"].strip()),"PG3_EXACT_AUTHORIZATION")
 cal=read(artifact(s["calibration"]));gate=read(artifact(s["cpu0"]))
 require(cal["code"]==s["code"] and cal["bindings"]==s["bindings"] and cal["uses_development"] is False,"PG3_CALIBRATION_SOURCE")
 require(gate["passed"] is True and gate["optimizer_steps"]==0 and gate["weights_before"]==gate["weights_after"] and gate["code"]==s["code"],"PG3_CPU0")
 require(all(v>=0 and np.isfinite(v) for v in cal["new"]["weights"].values()),"PG3_WEIGHTS")
 require(cal["old_calibration"]["sha256"]==sha(artifact(cal["old_calibration"])),"PG3_OLD_CALIBRATION")
 require(s["selected_ids"] is None if s["kind"]=="paired32" else isinstance(s["selected_ids"],list) and len(s["selected_ids"])==8,"PG3_ORDER_CONTRACT")
 return cal

def resource_gate(s,start,run):
 available=int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")))/1024
 require(time.monotonic()-start<s["resources"]["wall_s"] and available>=3072 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<=4096,"PG3_RESOURCE")
 require(sum(p.stat().st_size for p in run.rglob("*") if p.is_file())<4*1024**3,"PG3_DISK")
 require(source_files()==s["code"]["files"] and os.environ.get("CUDA_VISIBLE_DEVICES")=="" and torch.get_num_threads()==1,"PG3_RUNTIME_IDENTITY")

def run(spec_path,auth_path):
 s=read(spec_path);a=read(auth_path);cal=validate(s,a)
 root=OUT/"runs";require(Path(s["output_root"])==root and s["run_id"].startswith("A42_") and all(x.isalnum() or x=="_" for x in s["run_id"]),"PG3_RUN_SCOPE")
 run=root/s["run_id"];require(not run.exists(),"PG3_RUN_EXCLUSIVE")
 ds,geo,init,q=qualify_bindings(s["bindings"],42)
 run.mkdir(parents=True,exist_ok=False);(run/"intents").mkdir()
 claim=OUT/"authorization_claims";claim.mkdir(exist_ok=True)
 atomic_json(claim/(digest(s)+".json"),dict(authorization=a))
 for name,value in (("RUN_SPEC.json",s),("AUTHORIZATION.json",a),("QUALIFICATION.json",q)):write(run/name,value)
 configure(dict(seed=42,threads=1,device="cpu"))
 model=QueryMincoPlanner();model.load_state_dict(torch.load(artifact(init["checkpoint"]),map_location="cpu",weights_only=True)["state_dict"],strict=True)
 require(state_sha(model)==init["weights_sha256"],"PG3_INITIAL_WEIGHTS")
 opt=torch.optim.AdamW(model.parameters(),lr=pg3_lr_at(s["schedule"],0),betas=(.9,.999),eps=1e-8,weight_decay=0.)
 initial=head_rows(model);context=TeacherGuidance();bank=TeacherBank(s["recipe"]);start=time.monotonic();streak=0;history=[];stopping=False;committed=None
 def stop(*args):
  nonlocal stopping
  stopping=True
 old=signal.signal(signal.SIGTERM,stop)
 try:
  baseline=evaluate(model,read(artifact(s["bindings"]))["development"]);write(run/"DEV_STEP0000.json",baseline)
  committed=save_commit(run,model,opt,s,0,dict(completed_optimizer_updates=0,optimizer_step_executed=False));presets(run,committed,0)
  overfit_baseline=None;route_error_before=None;extend=None
  for index in range(s["updates"]):
   require(not stopping,"PG3_STOP_REQUESTED");resource_gate(s,start,run)
   ids=geo.orders[42].batch(index)[0] if s["kind"]=="paired32" else s["selected_ids"]
   samples=ds.load(ids);model.train();opt.zero_grad(set_to_none=True)
   if s["method"]==BASELINE:
    z=old_forward(model,samples,geo,BASELINE,context,cal["old_weights"])
    from query_minco.training.pg2_guided_v2.gradients import measure,total_gradient
    rec,pg,rg=measure(model,z,groups=True)
    total,coord=total_gradient(pg,cal["old_weights"],projection=False)
    im=pg["imitation"]
    ratio=float((total-im).norm())/float(im.norm()) if float(im.norm()) else 1e30
    rec.update(losses={k:float(v.detach()) for k,v in z["terms"].items()},assignments=None)
   else:
    z=forward(model,samples,geo,s["method"],cal["old_weights"],cal["new"]["weights"],bank,context,index)
    rec,_,total=audit(model,z,details=True);ratio=rec["physical_imitation_ratio"]
    rec.update(assignments=z["assignments"],route_records=z["records"],quality_gate=z["quality"]["gate"].tolist(),losses={k:float(v.detach()) for k,v in z["terms"].items()})
   require(bool(torch.isfinite(total).all()) and np.isfinite(ratio),"PG3_FINITE_GRADIENT")
   streak=streak+1 if ratio>10 else 0;require(streak<2,"TWO_STEP_GRADIENT_RATIO_GT10")
   offset=0
   for p in model.parameters():p.grad=total[offset:offset+p.numel()].reshape_as(p).clone();offset+=p.numel()
   for n,p in model.named_parameters():
    if n.startswith("trajectory_head.") and p.shape[0]==21:require(not torch.count_nonzero(p.grad[19:21]),"PG3_SCORE_GRAD")
   gradnorm=float(torch.nn.utils.clip_grad_norm_(list(model.parameters()),10.));require(np.isfinite(gradnorm),"PG3_CLIP")
   lr=pg3_lr_at(s["schedule"],index)
   for group in opt.param_groups:group["lr"]=float(lr)
   rec.update(ids=ids,index=index,completed_optimizer_updates=index+1,loss=float(z["loss"].detach()),lr=float(lr),physical_imitation_ratio=ratio,ratio_alarm_streak=streak,global_grad_norm_before_clip=gradnorm,
    matched=z["matched"].tolist(),hard_pass=z["labels"]["passed"].tolist(),goal_pass=z["goal_labels"].tolist(),M=[len(x.teacher_coeffs) for x in samples],regime=[ds.rows[x.sample_id]["regime"] for x in samples])
   arrays=dict(raw=z["raw"].detach().numpy(),coeffs=z["output"].trajectory.coeffs.detach().numpy(),durations=z["output"].parameters.durations.detach().numpy(),matched=z["matched"].numpy())
   if s["method"]!=BASELINE:
    arrays.update({k:z["per_query"][k].detach().numpy() for k in ("mode","bottleneck","convex_corridor")})
   with (run/"intents"/f"step{index+1:04d}_ARRAYS.npz").open("xb") as f:np.savez_compressed(f,**arrays)
   rec["arrays_sha256"]=sha(run/"intents"/f"step{index+1:04d}_ARRAYS.npz")
   atomic_json(run/"intents"/f"step{index+1:04d}_STARTED.json",dict(parent=sha(committed/"COMMIT.json"),step=index+1))
   opt.step();require(rows_unchanged(model,initial),"PG3_FROZEN_SCORE_ROWS");require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),"PG3_FINITE_WEIGHTS")
   if index+1 in s["checkpoint_steps"]:
    ev=evaluate(model,read(artifact(s["bindings"]))["development"]);metric_alarm(history,ev,baseline,{})
    rec["development"]=ev
   committed=save_commit(run,model,opt,s,index+1,rec,parent_hash=sha(committed/"COMMIT.json"))
   if index+1 in s["checkpoint_steps"]:presets(run,committed,index+1)
   print("PG3_UPDATE",s["run_id"],index+1,rec["loss"],ratio,flush=True)
   if s["kind"]=="overfit128" and index+1==32:
    # Predetermined extension rule uses training-set error and fresh hard success, never dev tuning.
    model.eval()
    with torch.enable_grad():
     probe=forward(model,ds.load(s["selected_ids"]),geo,s["method"],cal["old_weights"],cal["new"]["weights"],bank,context,0)
    score=float(probe["terms"]["mode"].detach()+probe["terms"]["bottleneck"].detach())
    initial_score=cal["overfit_initial_route_error"]
    extend=bool(score<.95*initial_score)
    write(run/"EXTENSION_DECISION.json",dict(allowed=extend,rule="continue to preregistered128 only if same training route loss falls>=5% by32; no guard failures",before=initial_score,after=score))
    if not extend:break
  completed=index+1
  write(run/"RESULT.json",dict(status="COMPLETED",optimizer_updates=completed,authorized_max=s["updates"],final_weights_sha256=state_sha(model),wall_s=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,extension=extend))
 except BaseException:
  write(run/"FAILURE_RECORD.json",dict(error=traceback.format_exc(),last_committed=str(committed),retry=False));raise
 finally:signal.signal(signal.SIGTERM,old)
