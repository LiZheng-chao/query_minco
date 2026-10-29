"""PG3 FullFrozen384 CPU128: new execution profile, unchanged method mathematics."""
import os,time,math,csv,json,copy,uuid,signal,traceback,subprocess,resource
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from query_minco.training.io import read,write,sha,digest,artifact,require
from query_minco.training.run_spec import ROOT,source_files,verify_code
from query_minco.training.pg1.spec import qualify_bindings
from query_minco.training.initialization import configure,head_rows,rows_unchanged
from query_minco.training.pg1.state_io import atomic_json,presets,publish,sync_dir,inspect_commit
from query_minco.training.checkpoints import save_checkpoint,load_checkpoint
from query_minco.training.pg2_guided_v2.objective import forward as safe_forward
from query_minco.training.pg2_guided_v2.gradients import measure,total_gradient
from query_minco.training.pg3_feasible_mode_v1.objective import forward as pg3_forward
from query_minco.training.pg3_feasible_mode_v1.spec import PREFIX,recipe,effective
from query_minco.training.pg3_feasible_mode_v1.teacher_labels import TeacherBank,landmarks,representability
from query_minco.training.pg3_feasible_mode_v1.gradients import audit
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance
from query_minco.training.pg1_v2_r12.evaluation import evaluate as open8
from query_minco.training.pilot_evaluation import metric_alarm,_certified_empty
from query_minco.evaluation.validation import validate_teacher,evaluate
from query_minco.inference.identity import state_sha
from experiments.pg3_feasible_mode_v1.common import OUT,model_from,dev_field
from .fulltrain128_evaluation import evaluate_set,inference_measure
R=OUT/"fulltrain128_001";SCHEDULE="fulltrain128_cosine_2e-5_1e-6_endpoint_v1"
METHODS=dict(A="query_minco_pg2_guided_v2_safe_v1",B=PREFIX+"_mode",C=PREFIX+"_route")
def lr_at(index):
 require(type(index) is int and 0<=index<128,"FULL_LR_INDEX")
 return float(1e-6+.5*(2e-5-1e-6)*(1+math.cos(math.pi*index/127)))
def code_scripts():
 return {str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name("fulltrain128_evaluation.py"),Path(__file__).with_name("test_fulltrain128.py")]}
def seal(folder,**kw):
 write(folder/"RUN_SEAL.json",dict(files={str(p.relative_to(folder)):sha(p) for p in folder.rglob("*") if p.is_file() and p.name!="RUN_SEAL.json"},**kw))
def committed(run,model,opt,spec,step,record,parent=None):
 require(all(type(g["lr"]) is float and g["lr"]==lr_at(max(step-1,0)) for g in opt.param_groups),"FULL_COMMIT_LR")
 directory=run/"commits";directory.mkdir(exist_ok=True);stage=directory/(".stage_%04d_"%step+uuid.uuid4().hex);stage.mkdir()
 receipt=save_checkpoint(stage,model,opt,spec,step,scheduler_id=SCHEDULE);write(stage/"UPDATE.json",record)
 write(stage/"COMMIT.json",dict(schema="pg1_commit_v1",step=step,parent_hash=parent,root_spec_sha256=digest(spec),files={p.name:sha(p) for p in stage.iterdir()},weights_sha256=receipt["weights_sha256"]))
 sync_dir(stage);target=directory/("step%04d"%step);publish(stage,target);return target
def teacher_targets(sample,scene,arrays):
 result=[]
 for i,(c,d) in enumerate(zip(sample.teacher_coeffs.numpy(),sample.teacher_durations.numpy())):
  check=validate_teacher(c,d,arrays["teacher_head_pva"][i],arrays["teacher_tail_pva"][i],arrays["teacher_inner_pts"][i],scene);require(check["passed"],"FULL_TEACHER_HARD_"+sample.sample_id)
  inner=np.stack([evaluate(c[j],d[j]) for j in (0,1)]);tail=np.stack([evaluate(c[-1],d[-1],j) for j in (0,1,2)]);flags=representability(inner,tail,d)
  require(flags["positions"] and flags["velocity"],"FULL_TEACHER_UNREPRESENTABLE")
  labels,events=landmarks(c,d,scene)
  result.append(dict(teacher=i,certified=True,representable=True,representability=flags,inner=inner.tolist(),tail=tail.tolist(),durations=d.tolist(),bottlenecks=labels,mode_signature=[(x["box"],*x["side"]) for x in labels if x["side"]],check_passed=True))
 return result
def stored_checks(row):
 checks=[]
 for ref in row["certificates"]:
  c=read(artifact(ref));key="teachers" if "teachers" in c else "experts";require(isinstance(c.get(key),list),"FULL_CERT_SCHEMA");checks+=c[key]
 require(len(checks)==row["M"] and all(c["verification"]["passed"] is True for c in checks),"FULL_STORED_CERT")
 return checks
def tensor_sample(sid,ref):
 with np.load(artifact(ref),allow_pickle=False) as z:
  arrays={k:z[k].copy() for k in z.files}
 fields={k:torch.from_numpy(arrays[k]) for k in ("depth_input","state","teacher_coeffs","teacher_durations")}
 require(all(v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in fields.values()),"FULL_ARRAYS")
 return SimpleNamespace(sample_id=sid,**fields),arrays
def load_items(cal):
 ds,geo,init,q=qualify_bindings(cal["bindings"],42)
 meta=read(artifact(cal["train_index"]));samples=ds.load(list(ds.rows));fields=geo.load(samples);train=[(s,f,meta[s.sample_id]) for s,f in zip(samples,fields)]
 dev=[]
 for row in read(artifact(cal["development_index"]))["rows"]:
  s,a=tensor_sample(row["id"],row["tensors"]);dev.append((s,dev_field(row),row))
 return ds,geo,init,q,train,dev
def prepare():
 require(not subprocess.check_output(["git","status","--porcelain"],text=True).strip(),"FULL_CLEAN")
 p=R/"preparation";p.mkdir(exist_ok=False);old=read(OUT/"preparation002/CALIBRATION.json");binding=read(artifact(old["bindings"]))
 binding["code"]=dict(git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),files=source_files(),allow_uncommitted_sources=False)
 write(p/"BINDINGS.json",binding);ref=dict(path=str(p/"BINDINGS.json"),sha256=sha(p/"BINDINGS.json"));ds,geo,init,q=qualify_bindings(ref,42);write(p/"QUALIFICATION.json",q)
 ids=list(ds.rows);require(len(ids)==384,"FULL384");order=geo.orders[42];prefix=order.ids[:128]
 require(prefix.shape==(128,8) and set(prefix.ravel())==set(ids),"FULL_ORDER_COVERAGE")
 train_index={};cache={};teacher_count=0;train_scenes=set()
 for n,sample in enumerate(ds.load(ids)):
  row=ds.rows[sample.sample_id];field=geo.load([sample])[0];g=field.row;src=g["scene_source"];path=artifact(src)
  if str(path) not in cache:cache[str(path)]=read(path)
  obj=cache[str(path)];scene=obj
  for part in g["scene_pointer"].strip("/").split("/"):scene=scene[int(part)] if isinstance(scene,list) else scene[part]
  require(scene["obstacles"]==g["obstacles"] and scene["flight_center_domain_m"]==g["flight_center_domain_m"],"FULL_GEOMETRY_SOURCE")
  checks=stored_checks(row);_,arrays=tensor_sample(sample.sample_id,row["artifact"]);targets=teacher_targets(sample,scene,arrays);teacher_count+=len(targets)
  train_scenes.add(digest(dict(obstacles=scene["obstacles"],domain=scene["flight_center_domain_m"])))
  train_index[sample.sample_id]=dict(id=sample.sample_id,family=g["family"],regime=row["regime"],M=row["M"],scene=scene,teachers=targets,teacher_actual_modes=[x["actual_mode"] for x in checks])
  if (n+1)%64==0:print("QUALIFY_TRAIN",n+1,"teachers",teacher_count,flush=True)
 require(teacher_count==672,"FULL672")
 devprotocol=read(artifact(binding["development"]));sourcepath=artifact(devprotocol["source"]);original=read(sourcepath);devrows=[];unknown=[]
 for row in original["rows"]:
  if row["split"]!="development":continue
  if row["M"]==0:unknown.append(row);continue
  refs={key:dict(path=str(sourcepath.parent/row[src]),sha256=row[src+"_sha256"]) for key,src in (("tensors","npz"),("metadata","metadata"),("checks","checks"))}
  meta=read(artifact(refs["metadata"]));checks=read(artifact(refs["checks"]));cc=checks.get("teachers",checks.get("experts"))
  require(meta["id"]==row["id"] and meta["split"]=="development" and row["id"] not in ds.rows,"FULL_DEV_LEAK_ID")
  scene=meta["scene"];require(meta["family_id"] not in ("family00","family01") and digest(dict(obstacles=scene["obstacles"],domain=scene["flight_center_domain_m"])) not in train_scenes,"FULL_DEV_LEAK_SCENE")
  require(len(cc)==row["M"] and all(x["verification"]["passed"] is True for x in cc),"FULL_DEV_CERT")
  sample,arrays=tensor_sample(row["id"],refs["tensors"]);require(len(sample.teacher_coeffs)==row["M"],"FULL_DEV_M")
  targets=teacher_targets(sample,scene,arrays);_certified_empty(scene)
  devrows.append(dict(id=row["id"],tensors=refs["tensors"],metadata=refs["metadata"],checks=refs["checks"],family=meta["family_id"],regime="near" if float(sample.state[6:9].double().norm())<=3.4 else "far",M=row["M"],scene=scene,teachers=targets))
 require(len(devrows)==192 and len(unknown)==24,"FULL_DEV_EXPECTED_COUNTS")
 write(p/"TRAIN_INDEX.json",train_index);write(p/"DEVELOPMENT_INDEX.json",dict(schema="pg3_full_existing_development192_v1",source=devprotocol["source"],rows=devrows,excluded_zero_teacher=unknown,qualification="family02/03 existing development, not blind test"))
 write(p/"ORDER128.json",dict(seed=42,order_source=next(x["artifact"] for x in binding["orders"] if x["seed"]==42),batches=prefix.tolist(),roles=order.roles[:128].tolist(),unique_tasks=384,exposures=1024,no_reordering=True))
 configure(dict(seed=42,threads=1,device="cpu"));m=model_from(init);m.train();before=state_sha(m);sample=ds.load(order.batch(0)[0]);bank=TeacherBank(recipe());ctx=TeacherGuidance();cpu={}
 for arm,method in METHODS.items():
  if arm=="A":
   z=safe_forward(m,sample,geo,method,ctx,old["old_weights"]);rec,pg,raw=measure(m,z,groups=True);total,_=total_gradient(pg,old["old_weights"],projection=False);ratio=float((total-pg["imitation"]).norm()/pg["imitation"].norm())
  else:
   z=pg3_forward(m,sample,geo,method,old["old_weights"],old["new"]["weights"],bank,ctx,0);rec,v,total=audit(m,z,details=True);ratio=rec["physical_imitation_ratio"]
  require(bool(torch.isfinite(total).all()) and ratio<=10,"FULL_INITIAL_GUARD");cpu[arm]=dict(loss=float(z["loss"].detach()),ratio=ratio,method=method)
 require(state_sha(m)==before==init["weights_sha256"],"FULL_CPU0_UPDATED")
 cal=dict(schema="pg3_fulltrain128_preparation_v1",bindings=ref,code=binding["code"],scripts=code_scripts(),initialization=init,old_weights=old["old_weights"],new_weights=old["new"]["weights"],legacy_calibration=dict(path=str(OUT/"preparation002/CALIBRATION.json"),sha256=sha(OUT/"preparation002/CALIBRATION.json")),train_index=dict(path=str(p/"TRAIN_INDEX.json"),sha256=sha(p/"TRAIN_INDEX.json")),development_index=dict(path=str(p/"DEVELOPMENT_INDEX.json"),sha256=sha(p/"DEVELOPMENT_INDEX.json")),order128=dict(path=str(p/"ORDER128.json"),sha256=sha(p/"ORDER128.json")),cpu0=cpu,weights_before=before,weights_after=state_sha(m),optimizer_created=False,optimizer_steps=0,passed=True,train_tasks=384,train_teachers=672,development_tasks=192,development_unknown_excluded=24)
 write(p/"CPU0.json",cal);seal(p,purpose="FULL_DATA_AND_CPU0_QUALIFIED")
 print("FULL_READINESS_PASS",cpu,flush=True)
def baseline():
 cal=read(R/"preparation/CPU0.json");verify_code(cal["code"]);require(code_scripts()==cal["scripts"],"FULL_SCRIPT_SHA")
 ds,geo,init,q,train,dev=load_items(cal);configure(dict(seed=42,threads=1,device="cpu"));m=model_from(init);folder=R/"baseline";folder.mkdir(exist_ok=False)
 targets={};results={}
 for name,items in (("training384",train),("development192",dev)):
  result,new=evaluate_set(m,items,folder,name,0);results[name]=result;targets.update(new)
 write(folder/"INITIAL_TARGETS.json",targets);write(folder/"OPEN_DEV8.json",open8(m,read(artifact(cal["bindings"]))["development"]));write(folder/"INFERENCE.json",inference_measure(m,train[0][0]));write(folder/"BASELINE.json",dict(weights_sha256=state_sha(m),code=cal["code"],scripts=cal["scripts"],results=results));seal(folder,purpose="SHARED_C1_STEP0_EVALUATION_NOT_TRAINING")
def make_spec(arm,cal):
 require(arm in METHODS,"FULL_ARM")
 return dict(schema="pg3_fulltrain128_cpu_v1",run_id="A42_"+arm+"_FullFrozen384_CPU128_run001",method_id=METHODS[arm],arm=arm,code=cal["code"],experiment_sources=cal["scripts"],preparation=dict(path=str(R/"preparation/CPU0.json"),sha256=sha(R/"preparation/CPU0.json")),bindings=cal["bindings"],initialization=cal["initialization"],initialization_prior_updates=2880,
  device="cpu",precision="FP32",threads=1,seed=42,batch_size=8,updates=128,checkpoint_steps=[0,32,64,96,128],schedule=SCHEDULE,optimizer=dict(name="AdamW",betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_norm=10.),
  old_weights=cal["old_weights"],new_weights={} if arm=="A" else effective(cal["new_weights"],METHODS[arm]),assignment="NONE" if arm=="A" else "ORIGINAL_ROTATING_CAPACITY2",point_bottleneck=arm=="C",convex_corridor=arm=="C",conditional_control=False,ranking=False,head_projection=False,region=False,adaptive=False,
  order128=cal["order128"],train_index=cal["train_index"],development_index=cal["development_index"],resources=dict(wall_s=1800,rss_mib=4096,min_available_mib=3072,disk_gib=4),stage_authorization="single user authorization covers128 in one uninterrupted run; no reset at32",output_root=str(R/"runs"))
def train(arm):
 cal=read(R/"preparation/CPU0.json");require(cal["passed"] and cal["optimizer_steps"]==0,"FULL_CPU0");verify_code(cal["code"]);require(code_scripts()==cal["scripts"],"FULL_SOURCES")
 spec=make_spec(arm,cal);require(spec["method_id"]==METHODS[arm] and not any(spec[k] for k in ("conditional_control","ranking","head_projection","region","adaptive")),"FULL_DISABLED")
 ds,geo,init,q,train_items,dev=load_items(cal);order=read(artifact(cal["order128"]));require(set(sum(order["batches"],[]))==set(ds.rows),"FULL_RUNTIME_ORDER")
 root=R/"runs";rp=root/spec["run_id"];rp.mkdir(parents=True,exist_ok=False)
 auth=dict(approved=True,authorized_by="LiZheng-chao",spec_sha256=digest(spec),run_id=spec["run_id"],device="cpu",max_optimizer_updates=128)
 for n,v in (("RUN_SPEC.json",spec),("AUTHORIZATION.json",auth),("QUALIFICATION.json",q)):write(rp/n,v)
 configure(dict(seed=42,threads=1,device="cpu"));m=model_from(init);require(state_sha(m)==init["weights_sha256"],"FULL_INIT")
 opt=torch.optim.AdamW(m.parameters(),lr=lr_at(0),betas=(.9,.999),eps=1e-8,weight_decay=0.);score=head_rows(m);bank=TeacherBank(recipe());ctx=TeacherGuidance();streak=0;history=[];steps=0;last=None;stop=False;start=time.monotonic()
 ref=read(R/"baseline/BASELINE.json");require(ref["weights_sha256"]==state_sha(m) and ref["code"]==spec["code"] and ref["scripts"]==spec["experiment_sources"],"FULL_BASELINE_ID")
 targets=read(R/"baseline/INITIAL_TARGETS.json");guardbase=read(R/"baseline/OPEN_DEV8.json")
 def request(*a):
  nonlocal stop
  stop=True
 oldsig=signal.signal(signal.SIGTERM,request)
 try:
  last=committed(rp,m,opt,spec,0,dict(completed_optimizer_updates=0,baseline=dict(path=str(R/"baseline/BASELINE.json"),sha256=sha(R/"baseline/BASELINE.json"))));presets(rp,last,0)
  for i in range(128):
   avail=int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")))/1024
   require(not stop and time.monotonic()-start<1800 and avail>=3072 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<=4096,"FULL_RESOURCE")
   require(sum(p.stat().st_size for p in rp.rglob("*") if p.is_file())<4*1024**3,"FULL_DISK")
   require(source_files()==spec["code"]["files"] and code_scripts()==spec["experiment_sources"] and os.environ.get("CUDA_VISIBLE_DEVICES")=="" and torch.get_num_threads()==1,"FULL_RUNTIME_SOURCE")
   ids,roles=geo.orders[42].batch(i);require(ids==order["batches"][i],"FULL_ORDER_DRIFT");samples=ds.load(ids);m.train();opt.zero_grad(set_to_none=True)
   if arm=="A":
    z=safe_forward(m,samples,geo,METHODS[arm],ctx,cal["old_weights"]);rec,vectors,raw=measure(m,z,groups=True);total,_=total_gradient(vectors,cal["old_weights"],projection=False);im=vectors["imitation"];ratio=float((total-im).norm()/im.norm());losses={k:float(v.detach()) for k,v in z["terms"].items()};assignments=None
   else:
    z=pg3_forward(m,samples,geo,METHODS[arm],cal["old_weights"],cal["new_weights"],bank,ctx,i);rec,vectors,total=audit(m,z,details=True);im=vectors["imitation"];ratio=rec["physical_imitation_ratio"];losses={k:float(v.detach()) for k,v in z["terms"].items()};assignments=z["assignments"]
   require(bool(torch.isfinite(total).all()) and np.isfinite(ratio),"FULL_NONFINITE")
   streak=streak+1 if ratio>10 else 0
   if streak>=2:write(rp/"REJECTED_UPDATE.json",dict(attempt=i+1,ratio=ratio,ids=ids,rec=rec,optimizer_step_executed=False));raise ValueError("TWO_STEP_GRADIENT_RATIO_GT10")
   oldflat=torch.cat([p.detach().flatten().clone() for p in m.parameters()]);offset=0
   for name,p in m.named_parameters():
    p.grad=total[offset:offset+p.numel()].reshape_as(p).clone();offset+=p.numel()
    if name.startswith("trajectory_head.") and p.shape[0]==21:require(not torch.count_nonzero(p.grad[19:21]),"FULL_SCORE_GRAD")
   gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),10.));require(np.isfinite(gn),"FULL_CLIP")
   for group in opt.param_groups:group["lr"]=lr_at(i)
   rec.update(index=i,completed_optimizer_updates=i+1,ids=ids,roles=roles,losses=losses,loss=float(z["loss"].detach()),lr=lr_at(i),physical_imitation_ratio=ratio,ratio_alarm_streak=streak,gradnorm=gn,assignments=assignments,matched=z["matched"].tolist(),hard_pass=z["labels"]["passed"].tolist(),goal_pass=z["goal_labels"].tolist(),M=[len(s.teacher_coeffs) for s in samples],regime=[ds.rows[s.sample_id]["regime"] for s in samples])
   if arm!="A":rec.update(route_records=z["records"],weighted_component_norms=rec["weighted_component_norms"])
   a=dict(raw=z["raw"].detach().numpy(),inner=z["output"].parameters.inner_pts.detach().numpy(),coeffs=z["output"].trajectory.coeffs.detach().numpy(),durations=z["output"].parameters.durations.detach().numpy(),matched=z["matched"].numpy())
   if arm!="A":a.update({k:z["per_query"][k].detach().numpy() for k in ("mode","bottleneck","convex_corridor")})
   folder=rp/"intents";folder.mkdir(exist_ok=True)
   with (folder/f"step{i+1:04d}_ARRAYS.npz").open("xb") as f:np.savez_compressed(f,**a)
   atomic_json(folder/f"step{i+1:04d}_STARTED.json",dict(parent=sha(last/"COMMIT.json"),arrays_sha256=sha(folder/f"step{i+1:04d}_ARRAYS.npz")))
   opt.step();steps=i+1;require(rows_unchanged(m,score) and all(bool(torch.isfinite(p).all()) for p in m.parameters()),"FULL_PARAMETER_CONTRACT")
   delta=torch.cat([p.detach().flatten() for p in m.parameters()])-oldflat
   def cosine(a,b):
    a=a.double();b=b.double();den=float(a.norm()*b.norm());return float(a@b)/den if den else None
   rec["adam_delta_vs_negative_imitation"]=cosine(delta,-im);rec["adam_delta_vs_negative_total"]=cosine(delta,-total)
   if arm!="A":rec["adam_delta_vs_negative_mode"]=cosine(delta,-vectors["mode"]);rec["imitation_mode_cosine"]=cosine(im,vectors["mode"])
   alarm=None
   if steps in spec["checkpoint_steps"]:
    rec["open_dev8"]=open8(m,read(artifact(spec["bindings"]))["development"])
    try:metric_alarm(history,rec["open_dev8"],guardbase,{})
    except ValueError as e:alarm=str(e)
    rec["checkpoint_guard_failure"]=alarm
   last=committed(rp,m,opt,spec,steps,rec,parent=sha(last/"COMMIT.json"))
   if steps in spec["checkpoint_steps"]:
    presets(rp,last,steps)
    for name,items in (("training384",train_items),("development192",dev)):evaluate_set(m,items,rp/"evaluation",name,steps,targets)
    write(rp/"evaluation"/f"INFERENCE_step{steps:04d}.json",inference_measure(m,train_items[0][0]))
   print("FULL_UPDATE",arm,steps,ratio,flush=True)
   if alarm:raise ValueError(alarm)
  blob=load_checkpoint(last/"checkpoint_step0128.pt",m,opt,spec,False);require(blob["scheduler"]==SCHEDULE and all(int(s["step"])==128 for s in opt.state.values()),"FULL_FINAL_STATE")
  write(rp/"RESULT.json",dict(status="COMPLETED",optimizer_updates=128,unique_tasks=len(set(sum(order["batches"],[]))),final_weights_sha256=state_sha(m),wall_s=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
 except BaseException:
  write(rp/"FAILURE_RECORD.json",dict(error=traceback.format_exc(),actual_optimizer_updates=steps,last_committed=str(last),retry=False))
  # A committed partial model may be evaluated read-only; never fabricate missing checkpoints.
  if last is not None and (rp/"commits"/f"step{steps:04d}").exists() and steps not in spec["checkpoint_steps"]:
   for name,items in (("training384",train_items),("development192",dev)):evaluate_set(m,items,rp/"partial_evaluation",name,steps,targets)
  raise
 finally:
  signal.signal(signal.SIGTERM,oldsig);seal(rp,status="COMPLETED" if (rp/"RESULT.json").exists() else "FAIL",actual_optimizer_updates=steps)
if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser();p.add_argument("command",choices=["prepare","baseline","A","B","C"]);a=p.parse_args()
 prepare() if a.command=="prepare" else baseline() if a.command=="baseline" else train(a.command)
