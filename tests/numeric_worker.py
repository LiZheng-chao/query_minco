"""Finite synthetic CPU checks. Always preserve failures; no optimizer updates."""
import sys,os,json,traceback,hashlib,subprocess
from pathlib import Path
N=Path(__file__).resolve().parents[1];R=Path('/home/lizheng/reach_planner')
sys.path.insert(0,str(N/'tools'))
from validation_guard import install,sha
action=sys.argv[1];run=Path(sys.argv[2]);cfg=json.loads((run/'SOURCE_FREEZE.json').read_text())
def write(name,z):
 with (run/name).open('x') as f:json.dump(z,f,indent=2,allow_nan=False)
def save(name,**arrays):
 with (run/(name+'.npz')).open('xb') as f:np.savez(f,**arrays)
def verify(p,h):
 if sha(p)!=h:raise ValueError('DEPENDENCY_SHA '+str(p))
def reject(label,fn):
 try:fn()
 except (ValueError,RuntimeError,TypeError,FileNotFoundError) as e:return dict(label=label,rejected=True,error=str(e))
 raise AssertionError('negative accepted '+label)
try:
 for n,h in cfg['sources'].items():verify(N/n,h)
 for n,h in cfg['legacy'].items():verify(R/n,h)
 for n,h in cfg['inputs'].items():verify(Path(n),h)
 allowed={str(R/n) for n in cfg['legacy']}
 events=install(run,allowed,cfg['runtime']['uname_sha256'],cfg['runtime']['platform_sha256'])
 import numpy as np
 import torch
 torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 assert os.statvfs(str(R)).f_flag & os.ST_RDONLY
 assert os.statvfs(str(N)).f_flag & os.ST_RDONLY
 assert not(os.statvfs(str(run)).f_flag & os.ST_RDONLY)
 negatives=[reject('reject_shell',lambda:subprocess.run(['/bin/sh','-c','true'])),
 reject('reject_uname_wrong_source',lambda:subprocess.check_output(('uname','-p'))),
 reject('reject_os_system',lambda:os.system('true')),
 reject('wrong_dependency_sha',lambda:verify(N/'query_minco/models/planner.py','0'*64)),
 reject('missing_dependency',lambda:verify(R/'UNAUTHORIZED.py','0'*64)),
 reject('hidden_teacher_data',lambda:(R/'query_minco/outputs/p2c2c1a_teacher_v1/dataset/manifest.json').read_bytes())]
 for p in [N/'query_minco/models/planner.py',R/'query_minco/planner.py',run.parent/'outside_run']:
  try:
   fd=os.open(str(p),os.O_WRONLY|os.O_CREAT);os.close(fd)
  except OSError as e:assert e.errno==30,(p,e)
  else:raise AssertionError('write escaped '+str(p))
 if action=='import':
  assert events,'real numpy startup must use narrowly permitted uname'
  write('RESULT.json',dict(passed=True,numpy=np.__version__,torch=torch.__version__,
   python=sys.version,uname_events=events,negatives=negatives,model_forward_calls=0,
   root_readonly=True,only_current_run_writable=True,GT_teacher_hidden=True))
 else:
  side='old' if action.endswith('old') or action=='old' else 'new'
  sys.path.insert(0,str(R if side=='old' else N))
  cp='query_minco/outputs/b2c1/train_v3/A42_C1_orthogonal/checkpoint_step2880.pt'
  binding=dict(checkpoint=cp,sha256='2352de6dd1e4a839b78ec12c115b5319b34da3c61cada105dbdeb24a4faec6d1',
   weights_sha256='68570b63e15f21ae9e9d512393391c9a4f3cf8569dfc730cfbeb9cb8f436d63b',seed=42,arm='C1_orthogonal')
  if action.startswith('training_'):
   from training_worker import execute
   result=execute(side,run)
  else:
   if side=='old':
    from query_minco.experiments.closed_loop_v1.runtime_adapter import RuntimeAdapter
    from query_minco.experiments.b2c1.common import state_sha
    from query_minco.experiments.b2c1 import execution_revision as rev
    adapter=RuntimeAdapter(binding,device='cpu')
    from query_minco.trajectory.types import MincoTrajectory
    import query_minco.p2c2a.objective as obj
    import query_minco.p2b.imitation as im
    targets=[(obj,'TrajectoryPostprocessor',rev.TrajectoryPostprocessor),(obj,'TrajectorySampler',rev.TrajectorySampler),
     (obj,'teacher_evaluate',rev.teacher_evaluate),(im,'TrajectorySampler',rev.TrajectorySampler),(im,'teacher_evaluate',rev.teacher_evaluate)]
    originals=[getattr(m,k) for m,k,v in targets]
    with rev.enabled():
     with rev.enabled():assert all(getattr(m,k) is v for m,k,v in targets)
    assert all(getattr(m,k) is v for (m,k,_),v in zip(targets,originals))
    try:
     with rev.enabled():raise ValueError('scope sentinel')
    except ValueError:pass
    assert all(getattr(m,k) is v for (m,k,_),v in zip(targets,originals))
   else:
    from query_minco.inference import load_planner,InferenceAdapter
    from query_minco.inference.identity import state_sha
    from query_minco.trajectory.types import MincoTrajectory
    negatives+=[reject('wrong_checkpoint_sha',lambda:InferenceAdapter(R/cp,expected_checkpoint_sha256='0'*64)),
     reject('unknown_method',lambda:load_planner('latest',R/cp))]
    adapter=load_planner('query_minco_b2c1_c1_e1_v1',R/cp)
    assert adapter.strict_load==dict(missing_keys=[],unexpected_keys=[])
   initial=state_sha(adapter.model);assert initial==binding['weights_sha256']
   param_keys=list(adapter.model.state_dict());save('state_dict',**{k:v.numpy() for k,v in adapter.model.state_dict().items()})
   calls=[0];traces={}
   def hook(*args):calls[0]+=1
   adapter.model.register_forward_pre_hook(hook)
   def export(out,raw,s):
    p=out.parameters;u=out.samples
    fields=dict(raw=raw,head_pva=p.head_pva,tail_pva=p.tail_pva,inner_pts=p.inner_pts,total_time=p.total_time,
     durations=p.durations,coefficients=out.trajectory.coeffs,solve_valid=out.trajectory.solve_valid,
     uniform_times=u.times,uniform_p=u.p,uniform_v=u.v,uniform_a=u.a,uniform_jerk=u.jerk,
     requested_times=s.times,p=s.p,v=s.v,a=s.a,jerk=s.jerk,segment_indices=s.segment_indices,local_times=s.local_times)
    return {k:v.detach().cpu().numpy().copy() for k,v in fields.items()}
   def original(d,s,t):
    got=adapter.forward(d[0],s[0])
    with torch.inference_mode():
     raw=torch.from_numpy(got['raw'])[None];out=adapter.post(raw,torch.from_numpy(s),num_samples=65)
     assert out.trajectory.coeffs[0].numpy().tobytes()==got['coeffs'].tobytes()
     assert out.parameters.durations[0].numpy().tobytes()==got['durations'].tobytes()
     ss=adapter.post.sampler(out.trajectory,times=torch.from_numpy(t))
     return export(out,raw,ss)
   def profile(f,event,arg):
    if event=='call' and ('sampler_e1.py' in f.f_code.co_filename or 'sampler_exec.py' in f.f_code.co_filename or 'cumulative_time.py' in f.f_code.co_filename):
     key=f.f_code.co_filename+':'+f.f_code.co_name;traces[key]=traces.get(key,0)+1
   sys.setprofile(profile)
   for name in ['near','far','nonzero_state']:
    with np.load(N/'research_outputs/migration_v1/fixtures'/(name+'.npz'),allow_pickle=False) as z:
     d=z['depth_input'].copy();s=z['state'].copy();t=z['times'].copy()
    fn=original if side=='old' else adapter.forward
    a=fn(d,s,t);b=fn(d,s,t)
    save(name,**a);save(name+'_repeat',**b)
    assert all(a[k].tobytes()==b[k].tobytes() for k in a)
    dt=a['durations'];x=dt[...,0];y=x+dt[...,1]
    bt=np.stack([np.zeros_like(x),np.nextafter(x,np.float32(-np.inf)),x,np.nextafter(x,np.float32(np.inf)),
     np.nextafter(y,np.float32(-np.inf)),y,np.nextafter(y,np.float32(np.inf)),dt.sum(-1)],-1)
    with torch.inference_mode():
     tr=MincoTrajectory(torch.from_numpy(a['coefficients']),torch.from_numpy(dt),torch.from_numpy(a['solve_valid']),{})
     ss=adapter.post.sampler(tr,times=torch.from_numpy(bt))
     save(name+'_boundary',**{k:getattr(ss,k).numpy() for k in ['times','p','v','a','jerk','segment_indices','local_times']})
    if side=='old':
     negatives.append(reject('bad_shape',lambda:adapter.forward(d[0,:,:127,:],s[0])))
     dd=d[0].copy();dd[0,0,0]=np.nan
     negatives.append(reject('NaN_input',lambda:adapter.forward(dd,s[0])))
    else:
     negatives.append(reject('bad_shape',lambda:adapter.forward(d[:,:,:127,:],s,t)))
     dd=d.copy();dd[0,0,0,0]=np.nan
     negatives.append(reject('NaN_input',lambda:adapter.forward(dd,s,t)))
   sys.setprofile(None)
   assert calls[0]==6 and state_sha(adapter.model)==initial
   assert all(p.grad is None for p in adapter.model.parameters())
   assert any(k.endswith('cumulative_time.py:cumulative_three') for k in traces)
   result=dict(passed=True,side=side,forward_calls=calls[0],parameter_keys=param_keys,
    initial_state_sha256=initial,final_state_sha256=state_sha(adapter.model),e1_trace=traces,
    scope_restore='normal/nested/exception checked' if side=='old' else 'explicit E1, no global binding',
    negatives=negatives,checkpoint_sha256=sha(R/cp),numpy=np.__version__,torch=torch.__version__,
    deterministic=torch.are_deterministic_algorithms_enabled(),precision='FP32',training_updates=0)
  for name,mod in list(sys.modules.items()):
   f=getattr(mod,'__file__',None)
   if f and R in Path(f).parents:
    rel=str(Path(f).relative_to(R))
    assert rel in cfg['legacy'],('unlisted dependency',rel)
    verify(Path(f),cfg['legacy'][rel])
  write('RESULT.json',result)
 print('PASSED',action,flush=True)
except BaseException:
 write('STOP.json',dict(error=traceback.format_exc(),action=action))
 traceback.print_exc();raise SystemExit(1)
