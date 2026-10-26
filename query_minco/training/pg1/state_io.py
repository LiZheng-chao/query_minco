"""Durable immutable commits. Reuses existing full-state serialization; no overwrite."""
import ctypes,errno,os,uuid
from pathlib import Path
from query_minco.training.io import read,write,sha,require,digest
from query_minco.training.checkpoints import save_checkpoint,load_checkpoint

def sync_dir(path):
 fd=os.open(str(path),os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)

def publish(stage,target):
 """Linux atomic no-replace rename; unsupported hosts fail closed."""
 libc=ctypes.CDLL(None,use_errno=True)
 fn=getattr(libc,'renameat2',None);require(fn is not None,'RENAME_NOREPLACE_UNAVAILABLE')
 if fn(-100,os.fsencode(stage),-100,os.fsencode(target),1)!=0:
  e=ctypes.get_errno();raise OSError(e,os.strerror(e),str(target))
 sync_dir(Path(target).parent)

def atomic_json(path,value):
 path=Path(path);tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
 write(tmp,value)
 try:os.link(str(tmp),str(path));sync_dir(path.parent)
 except BaseException:raise # failed files remain evidence
 else:tmp.unlink();sync_dir(path.parent)

def save_commit(run,model,opt,root_spec,step,record,parent_hash=None,fault=None):
 run=Path(run);directory=run/'commits';directory.mkdir(exist_ok=True);sync_dir(run)
 stage=directory/('.stage_%04d_'%step+uuid.uuid4().hex);stage.mkdir()
 receipt=save_checkpoint(stage,model,opt,root_spec,step,scheduler_id='pg1.spec.lr_at_v1')
 write(stage/'UPDATE.json',record)
 files={p.name:sha(p) for p in stage.iterdir()}
 write(stage/'COMMIT.json',dict(schema='pg1_commit_v1',step=step,parent_hash=parent_hash,
      root_spec_sha256=digest(root_spec),files=files,weights_sha256=receipt['weights_sha256']))
 sync_dir(stage)
 if fault:fault('staged',step)
 target=directory/('step%04d'%step);publish(stage,target)
 return target

def inspect_commit(path,root_spec):
 path=Path(path);c=read(path/'COMMIT.json')
 require(c['schema']=='pg1_commit_v1' and c['root_spec_sha256']==digest(root_spec),'COMMIT_SPEC')
 require(set(c['files'])=={'checkpoint_step%04d.pt'%c['step'],'checkpoint_step%04d.pt.json'%c['step'],'UPDATE.json'},'COMMIT_FILES')
 require(all('/' not in n and sha(path/n)==h for n,h in c['files'].items()),'COMMIT_SHA')
 return c

def restore(path,model,opt,root_spec,restore_rng=True):
 c=inspect_commit(path,root_spec)
 blob=load_checkpoint(Path(path)/('checkpoint_step%04d.pt'%c['step']),model,opt,root_spec,restore_rng=restore_rng)
 require(blob['completed_optimizer_updates']==c['step'] and blob['weights_sha256']==c['weights_sha256'],'COMMIT_STATE_CURSOR')
 require(all(int(v['step'])==c['step'] for v in opt.state.values()),'OPTIMIZER_EXACT_STEPS')
 require(c['step']==0 or len(opt.state)==len(list(model.parameters())),'OPTIMIZER_STATE_COMPLETENESS')
 return blob,c

def recovery_point(parent,root_spec):
 """Read only. Complete staged state may be used; ambiguous update is NOT replayed."""
 parent=Path(parent);commits=parent/'commits';valid=[]
 for path in sorted(commits.glob('step[0-9]*')):
  c=inspect_commit(path,root_spec);valid.append((c['step'],path,c))
 require(valid,'NO_RECOVERY_STATE')
 for i,(step,path,c) in enumerate(valid):
  if i:require(step==valid[i-1][0]+1 and c['parent_hash']==sha(valid[i-1][1]/'COMMIT.json'),'COMMIT_CHAIN')
 last=valid[-1];pending=last[0]+1
 if (parent/'intents'/('step%04d_STARTED.json'%pending)).exists():
  staged=[]
  for path in commits.glob('.stage_%04d_*'%pending):
   try:
    c=inspect_commit(path,root_spec)
    require(c['step']==pending and c['parent_hash']==sha(last[1]/'COMMIT.json'),'STAGED_CHAIN')
    staged.append((pending,path,c))
   except (ValueError,OSError):pass
  require(len(staged)==1,'AMBIGUOUS_UPDATE_BLOCKED_NO_REPLAY')
  return staged[0][1]
 return last[1]

def presets(run,commit,step):
 """Separate research checkpoint namespace; durable journal states aren't extra eval points."""
 run=Path(run);target=run/'checkpoints';target.mkdir(exist_ok=True)
 for name in ['checkpoint_step%04d.pt'%step,'checkpoint_step%04d.pt.json'%step]:
  os.link(str(Path(commit)/name),str(target/name))
 sync_dir(target)
