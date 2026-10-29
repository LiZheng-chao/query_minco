"""PG3 schedule-bound immutable commits; old serializers/checkpoints untouched."""
import uuid,math
from pathlib import Path
from query_minco.training.io import write,read,sha,digest,require
from query_minco.training.checkpoints import save_checkpoint,load_checkpoint
from query_minco.training.pg1.state_io import sync_dir,publish,inspect_commit
def lr_at(schedule,index):
 require(schedule in ("original_cosine_2e-5_1e-6_32","preregistered_cosine_2e-4_1e-5_128"),"PG3_SCHEDULE")
 n,hi,lo=(32,2e-5,1e-6) if schedule.startswith("original") else (128,2e-4,1e-5)
 require(0<=index<n,"PG3_LR_INDEX")
 return float(lo+.5*(hi-lo)*(1+math.cos(math.pi*index/(n-1))))
def save_commit(run,model,opt,root_spec,step,record,parent_hash=None,fault=None):
 require(all(type(g["lr"]) is float and math.isfinite(g["lr"]) for g in opt.param_groups),"PG3_STANDARD_LR")
 schedule=root_spec["schedule"];lr_at(schedule,0)
 directory=Path(run)/"commits";directory.mkdir(exist_ok=True);sync_dir(run)
 stage=directory/(".stage_%04d_"%step+uuid.uuid4().hex);stage.mkdir()
 receipt=save_checkpoint(stage,model,opt,root_spec,step,scheduler_id=schedule)
 write(stage/"UPDATE.json",record)
 write(stage/"COMMIT.json",dict(schema="pg1_commit_v1",step=step,parent_hash=parent_hash,root_spec_sha256=digest(root_spec),
  files={p.name:sha(p) for p in stage.iterdir()},weights_sha256=receipt["weights_sha256"]))
 sync_dir(stage)
 if fault:fault("staged",step)
 target=directory/("step%04d"%step);publish(stage,target);return target
def restore(path,model,opt,spec,restore_rng=True):
 c=inspect_commit(path,spec)
 # Inspect identity/schedule before restoring any caller state.
 import torch
 p=Path(path)/("checkpoint_step%04d.pt"%c["step"])
 receipt=read(str(p)+".json");require(sha(p)==receipt["sha256"],"PG3_CHECKPOINT_SHA")
 blob=torch.load(p,map_location="cpu",weights_only=True)
 require(blob["scheduler"]==spec["schedule"] and blob["next_lr_index"]==c["step"],"PG3_SCHEDULER_BINDING")
 require(all(type(g["lr"]) is float for g in blob["optimizer_state_dict"]["param_groups"]),"PG3_LOADED_LR_TYPE")
 expected=lr_at(spec["schedule"],max(c["step"]-1,0))
 require(all(g["lr"]==expected for g in blob["optimizer_state_dict"]["param_groups"]),"PG3_LOADED_LR_SEQUENCE")
 blob=load_checkpoint(p,model,opt,spec,restore_rng=restore_rng)
 require(blob["completed_optimizer_updates"]==c["step"] and all(int(v["step"])==c["step"] for v in opt.state.values()),"PG3_RESTORED_STEPS")
 return blob,c
