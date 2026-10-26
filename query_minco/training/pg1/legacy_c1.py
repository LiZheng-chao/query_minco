"""Read-only, exactly versioned legacy C1 evidence adapter. Never rewrites a blob."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from query_minco.training.io import artifact,read,require,exact,sha,digest
from query_minco.inference.identity import state_sha

# Actual archived identities, independently rehashed in g2_discovery001.
PINS={
 42:('2352de6dd1e4a839b78ec12c115b5319b34da3c61cada105dbdeb24a4faec6d1','68570b63e15f21ae9e9d512393391c9a4f3cf8569dfc730cfbeb9cb8f436d63b','6054f48ce5acc1328831e2e47b108351ed612e4f9eb06aa7402cba7b06b2bfd0'),
 43:('5c2728103ccf66254a2eb9ef7fab6f6608e7b369fa440eb171705da39e16e9f5','ebdf05484b8d857c0a61eae22a73ba4e599ccdaafdb256f3c9b2ecb73e76e0f8','cac00e7e41f495327b3f9e95689e0e5f9fcea623af28cd53a1a2a77fd376fe58'),
 44:('0195ccb75b13a5c903cec9d49db0361ad9a7192bdcf3d0c5a52f2f28dedd00c0','aa6eed04ae6de696a5a9361b94bf241bc2de27f09367754fb491ca4f0edfcea7','51a34382983b1ba59e614140474f027d2846b09d5126971d1565633cd32c8e1d'),
}
FREEZE_SHA='d55550563478c07665e71592bb51b8ac05433dd6161812279ff1d98a4c7b9bb6'
IMMUTABLE_SHA='f12ce130669674e8d913ec058c97e0c0a99b9ed96809d911d988bc05fe5bf06b'
REGISTRY_SHA='964dfef0f0149f90a7fdc49b4887a08a714b5dce1fe046204f6e0a3f551fb448'
EVIDENCE_KEYS=['run_config','training_seal','receipt','step0','step0_receipt','source_freeze','initialization_registry','parent_checkpoint','content_validation','process_exit','immutable_inputs']
def weights_sha(blob):
 require(isinstance(blob.get('state_dict'),dict),'LEGACY_STATE_DICT')
 require(all(isinstance(x,torch.Tensor) and x.dtype==torch.float32 and torch.isfinite(x).all() for x in blob['state_dict'].values()),'LEGACY_WEIGHTS_FINITE_FP32')
 return state_sha(SimpleNamespace(state_dict=lambda:blob['state_dict']))

def metadata(blob,seed,receipt,config,order,freeze,registry):
 """No guessed/default fields. This projection is not a synthetic checkpoint."""
 require(type(seed) is int and seed in PINS,'LEGACY_SEED')
 require(type(blob['seed']) is int and blob['seed']==receipt['seed']==config['spec']['seed']==seed,'LEGACY_SEED')
 require(blob['arm']==blob['spec']['arm']==receipt['arm']==config['spec']['arm']=='C1_orthogonal','LEGACY_ARM')
 require(all(type(x) is int and x==2880 for x in [blob['step'],blob['optimizer_updates_completed'],receipt['step']]),'LEGACY_UPDATES')
 require(blob['spec']==config['spec'] and blob['protocol']==config['protocol'],'LEGACY_CONFIG')
 require(blob['revision']==config['execution_revision']==blob['spec']['revision']=='E1','LEGACY_REVISION')
 require(blob['source_freeze_sha256']==receipt['source_freeze_sha256']==config['source_freeze_sha256']==sha(freeze)==FREEZE_SHA,'LEGACY_SOURCE_FREEZE')
 require(blob['protocol']['initialization_registry_sha256']==sha(registry)==REGISTRY_SHA,'LEGACY_INIT_REGISTRY')
 require(blob['spec']['order_sha256']==order['sha256']==sha(artifact(order)),'LEGACY_ORDER')
 require(blob['spec']['order'].endswith('/C1_orthogonal_seed%d.npz'%seed),'LEGACY_ORDER_SEED')
 require(blob['weights_sha256']==receipt['weights_sha256']==weights_sha(blob)==PINS[seed][1],'LEGACY_WEIGHTS')
 protocol=blob['protocol']
 require(protocol['steps']==2880 and protocol['batch_size']==8 and protocol['weight_decay']==0. and protocol['clip_norm']==10. and protocol['lr_start']==2e-4 and protocol['lr_end']==1e-5 and protocol['checkpoints']==[0,960,1920,2880],'LEGACY_TRAINING_PROTOCOL')
 return dict(seed=seed,arm=blob['arm'],stage_updates=blob['optimizer_updates_completed'],revision=blob['revision'],source_freeze_sha256=blob['source_freeze_sha256'],weights_sha256=blob['weights_sha256'],reference_mapping='legacy_C1_E1_to_frozen_reference_v1')

def load_legacy(init,seed,order):
 exact(init,['seed','checkpoint','weights_sha256','lineage'],'LEGACY_INITIALIZATION')
 require(type(init['seed']) is int and init['seed']==seed and seed in PINS,'LEGACY_INIT_SEED')
 cp=artifact(init['checkpoint'])
 require(sha(cp)==PINS[seed][0] and init['weights_sha256']==PINS[seed][1],'LEGACY_PINNED_CHECKPOINT')
 desc=read(artifact(init['lineage']))
 exact(desc,['schema','seed','evidence'],'LEGACY_DESCRIPTOR')
 require(desc['schema']=='pg1_legacy_c1_evidence_v1' and type(desc['seed']) is int and desc['seed']==seed,'LEGACY_DESCRIPTOR_TYPE')
 exact(desc['evidence'],EVIDENCE_KEYS,'LEGACY_EVIDENCE')
 paths={k:artifact(b) for k,b in desc['evidence'].items()}
 require(sha(paths['training_seal'])==PINS[seed][2],'LEGACY_PINNED_SEAL')
 require(sha(paths['source_freeze'])==FREEZE_SHA and sha(paths['initialization_registry'])==REGISTRY_SHA,'LEGACY_PINNED_SOURCE')
 require(sha(paths['immutable_inputs'])==IMMUTABLE_SHA,'LEGACY_IMMUTABLE_INPUTS')
 from query_minco.registry import MANIFEST_SHA
 from query_minco.training.run_spec import ROOT
 refpath=ROOT/'query_minco/methods/reference.json'
 require(sha(refpath)==MANIFEST_SHA,'LEGACY_REFERENCE_MAPPING')
 immutable=read(paths['immutable_inputs']);ref=read(refpath);checked=[]
 for name,component in ref['components'].items():
  if component['path'] in immutable:
   require(component['sha256']==immutable[component['path']],'LEGACY_CORE_MAPPING '+name)
   checked.append(name)
 require(len(checked)>=10,'LEGACY_CORE_MAPPING_COVERAGE')
 freeze=read(paths['source_freeze'])
 require(freeze['original_engine_metrics_byte_identical'] is True and freeze['protection_predicates_unchanged'] is True,'LEGACY_SOURCE_REVISION')
 seal=read(paths['training_seal'])
 names=dict(run_config='RUN_CONFIG.json',receipt='checkpoint_step2880_SEALED.json',step0='checkpoint_step0000.pt',step0_receipt='checkpoint_step0000_SEALED.json',content_validation='CONTENT_VALIDATION.json',process_exit='PROCESS_EXIT.json')
 require(paths['training_seal'].parent==cp.parent and paths['training_seal'].name=='TRAINING_SEAL.json','LEGACY_SEAL_PATH')
 for key,name in names.items():
  require(paths[key].parent==cp.parent and paths[key].name==name and sha(paths[key])==seal['files'][name],'LEGACY_SEAL_MEMBER '+key)
 require(cp.name=='checkpoint_step2880.pt' and sha(cp)==seal['files'][cp.name],'LEGACY_SEALED_CHECKPOINT')
 config=read(paths['run_config']);receipt=read(paths['receipt']);reg=read(paths['initialization_registry'])
 require(receipt['sha256']==sha(cp) and receipt['score_feasibility_rows_unchanged'] is True and receipt['sealed_before_formal_evaluation'] is True,'LEGACY_RECEIPT')
 blob=torch.load(str(cp),map_location='cpu',weights_only=True)
 projection=metadata(blob,seed,receipt,config,order,paths['source_freeze'],paths['initialization_registry'])
 parent=reg['A%d'%seed]
 require(parent['path']==config['spec']['checkpoint'] and parent['sha256']==config['spec']['checkpoint_sha256']==sha(paths['parent_checkpoint']) and str(paths['parent_checkpoint']).endswith('/'+parent['path']),'LEGACY_PARENT')
 audit=read(paths['content_validation'])
 require(audit['passed'] is True and audit['returncode']==0 and read(paths['process_exit'])['returncode']==0,'LEGACY_AUDIT_EXIT')
 content=json.loads(audit['stdout'])
 require(content['passed'] is True and content['seed']==seed and content['arm']=='C1_orthogonal' and content['updates']==2880 and content['task_exposures']==23040 and content['teacher_exposures']==40320,'LEGACY_AUDIT_CONTENT')
 zero=torch.load(str(paths['step0']),map_location='cpu',weights_only=True);zr=read(paths['step0_receipt'])
 require(zr['sha256']==sha(paths['step0']) and zr['seed']==zero['seed']==seed and zr['arm']==zero['arm']=='C1_orthogonal' and zr['step']==zero['step']==zero['optimizer_updates_completed']==0,'LEGACY_STEP0')
 require(weights_sha(zero)==zr['weights_sha256']==content['initial_weights_sha256'],'LEGACY_INITIAL_WEIGHTS')
 head=[k for k,v in blob['state_dict'].items() if k.startswith('trajectory_head.') and v.shape[0]==21]
 require(len(head)==2 and all(torch.equal(blob['state_dict'][k][19:21],zero['state_dict'][k][19:21]) for k in head),'LEGACY_SCORE_ROWS')
 return blob,dict(passed=True,adapter='legacy_C1_e1_v1',checkpoint=init['checkpoint'],descriptor=init['lineage'],evidence=desc['evidence'],projection=projection,parent_tensor_comparison='NOT_RUN_NO_UNSAFE_NUMPY_RNG_LOAD',checkpoint_modified=False,immutable_legacy_components_checked=checked,stage_local_updates=2880,full_lifetime_updates=None)
