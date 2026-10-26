"""New PG1 identities and fail-closed specs. No training executor or optimizer."""
import copy,math
from pathlib import Path
import torch
from query_minco.training.io import require,exact,artifact,read,digest,sha
from query_minco.training.run_spec import source_files,verify_code,ROOT
from query_minco.evaluation.goal_config import CONFIG
METHODS={
 'query_minco_pg1_e2_paired_v1':(0.,0.,0.),
 'query_minco_pg1_goal_v1':(1.,0.,0.),
 'query_minco_pg1_goal_dyn_v1':(1.,.1,0.),
 'query_minco_pg1_task_complete_v1':(1.,.1,1.),
}
PROFILES={
 'pg1_cpu_zero_update_v1':dict(updates=0,seeds=[42,43,44],checkpoints=[0],wall_cap_s=900),
 'pg1_pilot32_v1':dict(updates=32,seeds=[42],checkpoints=[0,8,16,32],wall_cap_s=900),
 'pg1_screen3seed_v1':dict(updates=32,seeds=[42,43,44],checkpoints=[0,8,16,32],wall_cap_s=900),
 'pg1_formal2880_v1':dict(updates=2880,seeds=[42,43,44],checkpoints=[0,960,1920,2880],wall_cap_s=21600),
}
OUTPUT=ROOT/'research_outputs/pg1_task_complete_v1'
def recipe(method):
 require(method in METHODS,'PG1_UNKNOWN_METHOD')
 g,d,b=METHODS[method]
 return dict(schema='pg1_recipe_v1',method_id=method,base='unchanged_E2_imitation_and_risk',points_per_segment=64,
  radius=.12,margin=.03,eps=.05,beta=1.,v_ref=1.,w_max=3.,goal_scale=g,dynamic_scale=d,domain_scale=b,
  dynamic_limits=[3.,8.,100.],goal_config_sha256=sha(ROOT/'query_minco/evaluation/goal_config.py'),
  goal_proof_sha256=sha(ROOT/'query_minco/evaluation/goal.py'),goal_mask='trusted_separating_slab_only',
  reduction='batch_query_mean_including_zero_masked_rows',raw19_21='UNSUPERVISED_ZERO_GRADIENT',
  sampler='fixed64_trapezoid_proxy_NOT_CONTINUOUS_PROOF',imitation_no_teacher='zero_only_if_proven_infeasible',
  time_contract=[.8,2.2,.12])
def profile(name):
 require(name in PROFILES,'PG1_UNKNOWN_PROFILE')
 return copy.deepcopy(PROFILES[name])
def lr_at(profile_id,index):
 n=profile(profile_id)['updates'];require(type(index) is int and n>0 and 0<=index<n,'PG1_LR_INDEX')
 return 1e-6+.5*(2e-5-1e-6)*(1+math.cos(math.pi*index/(n-1)))
def draft(method,profile_id,seed,code):
 p=profile(profile_id);require(seed in p['seeds'],'PG1_SEED')
 return dict(schema='pg1_run_spec_v1',run_id='A%d_%s_%s_PENDING'%(seed,method,profile_id),
  training_method_id=method,profile=profile_id,seed=seed,device='cpu' if p['updates']==0 else 'cuda',
  batch_size=8,output_root=str(OUTPUT),initialization_protocol='paired_C1_finetune_v1',
  initialization_prior_updates=2880,additional_updates=p['updates'],total_lineage_updates=2880+p['updates'],
  schedule='PG1_C1_FINETUNE_cosine_2e-5_to_1e-6_v1',optimizer=dict(name='AdamW',betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_norm=10.),
  recipe=recipe(method),recipe_sha256=digest(recipe(method)),code=copy.deepcopy(code),
  bindings=None,binding_status='NOT_QUALIFIED',qualification_reasons=['USER_DID_NOT_SUPPLY_G2_BINDING'],
  resource_budget=dict(wall_cap_s=p['wall_cap_s'],gpu_max_fraction=.75,gpu_max_mib=6144,min_free_gpu_mib=512,
                       formal_disk_cap_gib=8,limits_status='PROPOSAL_NOT_AUTHORIZED'))
def validate_spec(s,training=False):
 exact(s,['schema','run_id','training_method_id','profile','seed','device','batch_size','output_root',
  'initialization_protocol','initialization_prior_updates','additional_updates','total_lineage_updates','schedule',
  'optimizer','recipe','recipe_sha256','code','bindings','binding_status','qualification_reasons','resource_budget'],'PG1_SPEC')
 require(s['schema']=='pg1_run_spec_v1' and type(s['seed']) is int and type(s['additional_updates']) is int,'PG1_SPEC_SCHEMA')
 cfg=recipe(s['training_method_id']);p=profile(s['profile'])
 require(s['seed'] in p['seeds'] and s['batch_size']==8 and s['additional_updates']==p['updates'] and
         s['initialization_protocol']=='paired_C1_finetune_v1' and s['initialization_prior_updates']==2880 and
         s['total_lineage_updates']==2880+p['updates'],'PG1_BUDGET_LINEAGE')
 require(s['device']==('cpu' if p['updates']==0 else 'cuda'),'PG1_DEVICE_PROFILE')
 require(s['recipe']==cfg and s['recipe_sha256']==digest(cfg),'PG1_RECIPE_BINDING')
 require(s['schedule']=='PG1_C1_FINETUNE_cosine_2e-5_to_1e-6_v1' and
         s['optimizer']==dict(name='AdamW',betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_norm=10.),'PG1_OPTIMIZATION_CONTRACT')
 require(Path(s['output_root'])==OUTPUT and s['run_id'] and all(c.isalnum() or c in '_-' for c in s['run_id']),'PG1_RUN_OUTPUT')
 require(s['resource_budget']==dict(wall_cap_s=p['wall_cap_s'],gpu_max_fraction=.75,gpu_max_mib=6144,min_free_gpu_mib=512,
         formal_disk_cap_gib=8,limits_status='PROPOSAL_NOT_AUTHORIZED'),'PG1_RESOURCES')
 if training:
  require(p['updates']>0 and s['device']=='cuda' and s['code']['allow_uncommitted_sources'] is False,'PG1_CLEAN_CUDA_ONLY')
 verify_code(s['code'])
 require(s['bindings'] is not None,'PG1_NOT_QUALIFIED_MISSING_BINDINGS')
 require(s['binding_status']=='QUALIFIED' and s['qualification_reasons']==[],'PG1_NOT_QUALIFIED')
 return s

def qualify_bindings(binding,seed):
 """PG1-only v2 identity/data qualification and strict legacy evidence adapter."""
 from .qualification import qualify_bindings as implementation
 return implementation(binding,seed)

def authorization_template(s):
 return dict(schema='pg1_human_authorization_v1',run_id=s['run_id'],spec_sha256=digest(s),
  profile=s['profile'],method_id=s['training_method_id'],max_updates=profile(s['profile'])['updates'],
  initialization_protocol=s['initialization_protocol'],scope='PG1_SINGLE_RUN_ONLY',approved=False,authorized_by='')
def authorize(s,a):
 validate_spec(s,training=True)
 require(type(a.get('approved')) is bool and a['approved'] is True and a==dict(authorization_template(s),approved=True,authorized_by=a.get('authorized_by')) and
         isinstance(a['authorized_by'],str) and bool(a['authorized_by'].strip()),'PG1_AUTHORIZATION')
 require(read(artifact(s['bindings']))['code']==s['code'],'PG1_AUTH_CODE_BINDING')
 qualify_bindings(s['bindings'],s['seed'])
 return dict(validated=True,run_id=s['run_id'],optimizer_steps_executed=0)

def audit_schedule(s,updates,checkpoint_steps):
 p=profile(s['profile']);require(len(updates)==p['updates'],'PG1_EXACT_UPDATE_COUNT')
 for i,row in enumerate(updates):
  require(row['completed_optimizer_updates']==i+1 and row['order_cursor']==i+1 and
          row['lr']==lr_at(s['profile'],i) and row['spec_sha256']==digest(s),'PG1_LEDGER_SEQUENCE_LR')
 require(sorted(checkpoint_steps)==p['checkpoints'],'PG1_CHECKPOINT_SET')
 return dict(passed=True,updates=p['updates'],sample_visits=p['updates']*8)
