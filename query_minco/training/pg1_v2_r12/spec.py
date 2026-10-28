"""Fixed R12 CPU pilot identity; draft is not authorization."""
import copy
from pathlib import Path
from query_minco.training.pg1.spec import recipe as base_recipe
from query_minco.training.run_spec import ROOT,verify_code
from query_minco.training.io import require,digest,read,artifact,sha
R0='query_minco_pg1_v2_r12_r0_v1'
RTAIL='query_minco_pg1_v2_r12_tail_v1'
METHODS=(R0,RTAIL)
OUTPUT=ROOT/'research_outputs/pg1_v2_r12'
def external_sources():
 return {str(ROOT/p):sha(ROOT/p) for p in ['experiments/pg1_v2_r1/cpu0_readonly.py','research_outputs/pg1_v2_diagnostics_v1/_preparation/diagnose.py']}
def recipe(method):
 require(method in METHODS,'R12_METHOD')
 base=base_recipe('query_minco_pg1_task_complete_v1')
 return dict(base,schema='pg1_v2_r12_recipe_v1',method_id=method,
  base_method_id=base['method_id'],base_recipe_sha256=digest(base),
  tail_scale=.0025 if method==RTAIL else 0.,tail_fraction=.1,tail_rho=False,
  tail_source='query_minco.training.pg1_v2_r1.losses.bottleneck_safety_loss.losses.tail',
  quality_training_terms=[],projection=False,external_sources=external_sources())
def make_spec(binding,method,run_id):
 m=read(artifact(binding));cfg=recipe(method)
 require(run_id.startswith('A42_') and all(x.isalnum() or x=='_' for x in run_id),'R12_RUN_ID')
 return dict(schema='pg1_v2_r12_cpu_pilot_spec_v1',component_id='pg1_v2_r12_cpu_pilot_v1',method=method,recipe=cfg,
  recipe_sha256=digest(cfg),run_id=run_id,seed=42,device='cpu',threads=1,batch_size=8,precision='FP32',
  profile='pg1_pilot32_v1',optimizer=dict(name='AdamW',betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_norm=10.),
  schedule='cosine_2e-5_to_1e-6_endpoint_inclusive',bindings=copy.deepcopy(binding),code=m['code'],
  output_root=str(OUTPUT/'pilot_runs'),parent=None,cpu0=None,
  initialization_accounting=dict(kind='C1_step2880',C1_stage_updates=2880,additional_updates=32,prior_shared_history_updates=None,full_lifetime_updates=None),
  resources=dict(wall_s=3600,rss_cap_mib=4096,min_available_mib=3072,disk_cap_gib=4),
  checkpoints=[0,8,16,32],gradient_guard=dict(ratio=10.,consecutive=2,includes_weighted_tail=True),
  scope='CPU_PILOT32_ONLY_NO_CUDA_FORMAL_OR_RESUME')
def validate(s,training=False):
 expected=make_spec(s['bindings'],s['method'],s['run_id']);expected['cpu0']=s['cpu0']
 require(s==expected,'R12_FIXED_CONTRACT')
 if training:
  require(s['code']['allow_uncommitted_sources'] is False,'R12_CLEAN_SOURCE_REQUIRED')
  verify_code(s['code'])
  require(s['cpu0'] is not None,'R12_CPU0_REQUIRED')
 return s
def authorization(s):
 validate(s)
 return dict(schema='pg1_v2_r12_single_run_authorization_v1',run_id=s['run_id'],method=s['method'],
  spec_sha256=digest(s),max_optimizer_updates=32,device='cpu',approved=False,authorized_by='',
  scope='ONE_R12_CPU_PILOT32_NOT_CUDA_NOT_FORMAL')
def check_authorization(s,a):
 validate(s,training=True)
 require(a==dict(authorization(s),approved=True,authorized_by=a.get('authorized_by')) and
  type(a['approved']) is bool and isinstance(a['authorized_by'],str) and bool(a['authorized_by'].strip()),'R12_AUTHORIZATION')

def runtime_identity():
 import sys,numpy as np,torch
 return dict(python=sys.version,executable=sys.executable,torch=torch.__version__,numpy=np.__version__,device='cpu')

def cpu0_contract():
 return dict(schema='r12_cpu0_contract_v2',methods=list(METHODS),seed=42,initialization='C1_step2880',
  batch_size=8,batches_per_method=96,unique_tasks_per_method=384,exposures_per_method=768,
  precision='FP32',lambda_tail=.0025,tail_fraction=.1,tail_rho=False,radius=.12,margin=.03,
  dynamic_limits=[3.,8.,100.],time_contract=[.8,2.2,.12],gradient_ratio=10.,consecutive=2,
  recipe_shas={m:digest(recipe(m)) for m in METHODS},quality_training_terms=[])

def validate_cpu0_metadata(s,pf,initial_weights):
 require(pf.get('schema')=='pg1_v2_r12_cpu0_seal_v2','R12_CPU0_SCHEMA_V2_REQUIRED')
 require(pf.get('passed') is True and pf.get('optimizer_created') is False and type(pf.get('optimizer_steps')) is int and pf['optimizer_steps']==0,'R12_CPU0_PASS')
 require(pf.get('contract')==cpu0_contract(),'R12_CPU0_COMPLETE_CONTRACT')
 require(pf.get('runtime')==runtime_identity(),'R12_CPU0_RUNTIME')
 require(pf.get('code')==s['code'] and pf.get('bindings')==s['bindings'],'R12_CPU0_IDENTITY')
 require(pf.get('external_sources')==external_sources(),'R12_EXTERNAL_CPU0_BINDING')
 require(pf.get('model_weights_before')==pf.get('model_weights_after')==initial_weights,'R12_CPU0_INITIAL_WEIGHTS')

def validate_cpu0_seal(s,path,initial_weights):
 import json,math
 path=Path(path);pf=read(path);validate_cpu0_metadata(s,pf,initial_weights)
 files=pf.get('files');require(isinstance(files,dict),'R12_CPU0_FILES')
 for n,h in files.items():
  pp=Path(n)
  require(not pp.is_absolute() and '..' not in pp.parts and bool(pp.parts),'R12_CPU0_FILE_SCOPE')
  try:(path.parent/pp).resolve().relative_to(path.parent.resolve())
  except ValueError:raise ValueError('R12_CPU0_FILE_SCOPE')
  require(sha(path.parent/pp)==h,'R12_CPU0_SEAL')
 mandatory={'audit/CPU0_RESULT.json','_preparation/COVERAGE.json','runs/BATCHES.jsonl'}
 mandatory.update('runs/%s_batch%03d.npz'%(arm,i) for arm in ['R0','Rtail'] for i in range(96))
 require(mandatory<=set(files),'R12_CPU0_MISSING_BATCH_EVIDENCE')
 result=read(path.parent/'audit/CPU0_RESULT.json')
 require(result.get('schema')=='pg1_v2_r12_cpu0_v2' and result.get('source_bundle_clean') is True,'R12_CPU0_RESULT_CLEAN_V2')
 for key in ['code','bindings','contract','runtime','external_sources','model_weights_before','model_weights_after','optimizer_created','optimizer_steps','passed']:
  require(result.get(key)==pf.get(key),'R12_CPU0_RESULT_SEAL_METADATA_'+key)
 require(result.get('methods')==list(METHODS) and result.get('batches_per_method')==96 and result.get('unique_tasks_per_method')==384 and result.get('exposures_per_method')==768,'R12_CPU0_RESULT_COUNTS')
 coverage=read(path.parent/'_preparation/COVERAGE.json');batches=coverage['batches']
 require(len(batches)==96 and all(len(b)==8 for b in batches) and len(set(sid for b in batches for sid in b))==384,'R12_CPU0_COVERAGE')
 require(coverage['covered_ids']==sorted(set(sid for b in batches for sid in b)) and coverage['covered_ids_sha256']==digest(coverage['covered_ids']),'R12_CPU0_UNIQUE_IDS')
 rows=[json.loads(l) for l in (path.parent/'runs/BATCHES.jsonl').read_text().splitlines()]
 require(len(rows)==192,'R12_CPU0_RECORD_COUNT')
 for method in METHODS:
  rr=[r for r in rows if r['method']==method];streak=0
  require([r['batch'] for r in rr]==list(range(96)),'R12_CPU0_BATCH_SEQUENCE')
  for i,r in enumerate(rr):
   require(r['ids']==batches[i] and r['weights_before']==r['weights_after']==initial_weights,'R12_CPU0_ROW_BINDING')
   require(all(r.get(k) is True for k in ['numpy_reconstruction','raw19_21_zero','score_rows_zero','original_loss_and_dynamic_regression','paired_forward_same_raw','all_original_numeric_guards']),'R12_CPU0_ROW_GATES')
   ratio=r['physical_imitation_gradient_ratio']
   require(ratio is None or (isinstance(ratio,(float,int)) and not isinstance(ratio,bool) and math.isfinite(ratio) and ratio>=0),'R12_CPU0_ROW_RATIO')
   streak=streak+1 if ratio is not None and ratio>10 else 0
   require(streak<2 and r['ratio_alarm_streak']==streak,'R12_CPU0_ROW_GUARD')
 return pf

def validate_checkpoint_blob(blob,root_spec,step,weights_sha256):
 require(type(step) is int and all(type(blob.get(k)) is int for k in ['completed_optimizer_updates','order_cursor','next_lr_index']),'R12_CHECKPOINT_CURSOR_TYPE')
 require(blob.get('spec')==root_spec and blob.get('spec_sha256')==digest(root_spec),'R12_CHECKPOINT_SPEC')
 require(blob.get('completed_optimizer_updates')==blob.get('order_cursor')==blob.get('next_lr_index')==step,'R12_CHECKPOINT_CURSOR')
 require(blob.get('weights_sha256')==weights_sha256,'R12_CHECKPOINT_WEIGHT_ID')
