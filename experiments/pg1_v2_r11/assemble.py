"""Verify/preserve raw streams after diagnostic O_EXCL failure; no model updates."""
import json,hashlib,shutil,re,sys
from pathlib import Path
import numpy as np
R=Path(sys.argv[1]);base=R.parent;A=base/'audit001';B=base/'audit002';C=R/'combined'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def copy(src,dst):
 with dst.open('xb') as f:f.write(src.read_bytes())
pre=read(R/'_preparation/RECOVERY_PREREG.json')
assert all(sha(Path(p))==h for p,h in pre['source_files'].items())
r=read(R/'audit/CPU0_RESULT.json');assert r['status']=='COMPLETE' and r['mode']=='DIRECTION_REPLAY_ONLY'
C.mkdir(exist_ok=False)
for d in ['audit','runs','summary','_preparation']:(C/d).mkdir()
models=[read(A/'audit/PROGRESS.json')['models'][0],read(B/'audit/PROGRESS.json')['models'][0],r['models'][0]]
assert [m['model'] for m in models]==['C1_step0','E2_step32','PG1_step32']
batches=read(R/'_preparation/COVERAGE_CLARIFICATION.json')['ordered_batches']
assert len(batches)==96 and len(set(sum(batches,[])))==384
raws=[];tasks=[];directions=[];cases=[];inputs={};coverage={}
for name,src,dirsrc in [('C1_step0',A,A),('E2_step32',B,B),('PG1_step32',B,R)]:
 gf=src/'runs'/(name+'_GRADIENTS.jsonl');vf=src/'runs'/(name+'_VALUES.jsonl')
 gs=[json.loads(l) for l in gf.read_text().splitlines()];vv=[json.loads(l) for l in vf.read_text().splitlines()]
 assert len(vv)==768 and len({v['sample_id'] for v in vv})==384
 assert set(g['batch'] for g in gs)==set(range(96))
 assert all(sum(g['group']=='all' and g['batch']==i for g in gs)==16 for i in range(96))
 for f in [gf,vf]:inputs[str(f)]=sha(f);copy(f,C/'runs'/f.name)
 tasks.extend(v for v in vv if v['first_visit'])
 directions.extend(d for d in read(dirsrc/'runs/DIRECTIONS.json')['records'] if d['model']==name)
 cases.extend(d for d in read(dirsrc/'runs/CASES.json')['records'] if d['model']==name)
 for bi,ids in enumerate(batches):
  path=src/'runs'/('%s_batch%03d_RAW.npz'%(name,bi));inputs[str(path)]=sha(path);copy(path,C/'runs'/path.name)
  rows=vv[bi*8:(bi+1)*8];assert [v['sample_id'] for v in rows]==ids
  with np.load(path,allow_pickle=False) as z:
   assert z['raw'].shape==(8,6,21) and np.isfinite(z['raw']).all()
   rg={k[2:]:z[k] for k in z.files if k.startswith('g_')}
   assert len(rg)==16
   rr={}
   for term,g in rg.items():
    assert np.isfinite(g).all() and not np.count_nonzero(g[...,19:21])
    st=dict(norm=np.linalg.norm(g,axis=-1).tolist(),time15_19=[float(np.linalg.norm(g[...,j])) for j in range(15,19)],raw19_21_zero=True)
    for ref in ['imitation','obstacle']:
     dot=(g*rg[ref]).sum(-1);den=np.linalg.norm(g,axis=-1)*np.linalg.norm(rg[ref],axis=-1)
     st['cos_'+ref]=[[float(dot[i,q]/den[i,q]) if den[i,q]>1e-20 else None for q in range(6)] for i in range(8)]
    rr[term]=st
   raws.append(dict(model=name,batch=bi,ids=ids,first_visit=z['first_visit'].tolist(),raw=rr,
      budget=dict(status='see_postaudit_joint_budget; original last-model cumulative budget not persisted')))
 coverage[name]=dict(complete_batches=96,exposures=768,unique_tasks=384,gradient_terms=16,raw19_21_zero=True)
assert len(directions)==1989 and len(cases)==153
for key,data in [('BATCH_RAW_AND_BUDGET',raws),('UNIQUE_TASKS',tasks),('DIRECTIONS',directions),('CASES',cases)]:
 put(C/'runs'/(key+'.json'),dict(records=data))
for name in ['CPU0_ONLY_BINDINGS.json','COVERAGE_CLARIFICATION.json','PREREG.json']:
 copy(R/'_preparation'/name,C/'_preparation'/name)
last=max(float(x) for x in re.findall(r'wall ([0-9.]+)',(B/'audit/STDOUT.log').read_text()))
wall=read(A/'audit/PROGRESS.json')['wall_s']+last+r['wall_s']
result=dict(status='COMPLETE_FROM_VERIFIED_STREAMS_AND_REPLAY',coverage_not_process_status=True,
 models=models,batches=288,unique_task_models=1152,unique_candidate_models=6912,candidate_visits=13824,direction_records=1989,
 optimizer_created=False,optimizer_steps=0,GPU_jobs=0,training_qualified=False,wall_s=wall,
 wall_s_semantics='lower bound: completed C1 PROGRESS + last failed-stage wall log + exact direction replay. Not total CPU time.',
 thread_cpu_s=None,max_rss_mib=None,direction_replay_rss_mib=r['max_rss_mib'],
 source_processes=dict(audit001='STOPPED_AFTER_C1_SIGTERM_143',audit002='FAIL_EXCLUSIVE_AGGREGATE_WRITE_1',audit003='PASS_DIRECTION_REPLAY_ONLY_0'),
 missing_peak_and_total_thread_cpu='NOT_RECORDED for failed process; no fabricated values')
put(C/'audit/CPU0_RESULT.json',result)
put(C/'audit/SOURCE_ASSEMBLY.json',dict(coverage=coverage,files=inputs,direction_sources=[str(A),str(B),str(R)],
 aggregate_is_not_original_process_PASS=True,original_files_retained=True))
print('ASSEMBLY_VERIFIED',coverage,'directions',len(directions))
