"""PG1 data authority and current-code identity, separate from legacy C1 runtime."""
from pathlib import Path
from collections import Counter
from query_minco.registry import MANIFEST_SHA
from query_minco.training.io import artifact,read,require,exact,sha
from query_minco.training.run_spec import ROOT,verify_code,source_files
from query_minco.training.datasets.manifest import FrozenDataset
from query_minco.training.datasets.order import FrozenOrder
from query_minco.training.datasets.geometry_sidecar import GeometrySidecar
from query_minco.training.pilot_evaluation import qualify_development
from .geometry import authenticated_task_gates
from .legacy_c1 import load_legacy
P0_DIAGNOSTIC_SHA='ccfa44f4129d6be811f040b270ac0bf6885b6a795c8a1fbbe3c674b3112d9abf'
P0_PREVIOUS_SHA='3cf7aeea4d8bf581fe9947a01ddb8fe9bf5c805a4f308312522ceb225bd71703'

def current_identity(code,historical_reference):
 require(type(code.get('allow_uncommitted_sources')) is bool,'PG1_CODE_FLAG')
 verify_code(code) # Exact current HEAD and every current source file; CPU may explicitly use dirty sources.
 refpath=artifact(historical_reference)
 require(refpath==ROOT/'query_minco/methods/reference.json' and sha(refpath)==MANIFEST_SHA,'PG1_DATA_AUTHORITY')
 ref=read(refpath);differences=[]
 for name,expected in ref['portable_source_sha256'].items():
  actual=sha(ROOT/name)
  if actual!=expected:
   require(name=='query_minco/evaluation/validation.py' and expected==P0_PREVIOUS_SHA and actual==P0_DIAGNOSTIC_SHA,'PG1_UNAPPROVED_LEGACY_SOURCE '+name)
   differences.append(dict(path=name,historical_sha256=expected,current_sha256=actual,reason='exact authorized P0 empty diagnostic reduction; NOT old C1 runtime identity'))
 return ref,dict(passed=True,current_code=code,historical_reference=historical_reference,historical_runtime_identity_claimed=False,exact_approved_differences=differences)

def qualify_data(m,store=None):
 resolve=artifact if store is None else store.artifact
 if store is not None:
  from .transport import MappedDataset,MappedGeometry,mapped_development
 ref,identity=current_identity(m['code'],m['historical_reference'])
 ds=FrozenDataset(m['dataset'],'b2c1_reference_v1') if store is None else MappedDataset(m['dataset'],store)
 require(len(ds.rows)==384,'PG1_FROZEN384');ds.load(list(ds.rows))
 required=('TEACHER_DATA_GATE.json','DATA_CONTRACT_GATE.json','DATASET_MANIFEST.json','REPLAY_MANIFEST.json','PROTOCOL_SEAL.json','DATA_BINDINGS.json')
 evidence={}
 for name in required:
  bs=[b for b in ds.manifest['evidence'] if Path(b['path']).name==name]
  require(len(bs)==1,'PG1_DATA_EVIDENCE '+name);evidence[name]=read(resolve(bs[0]))
 for original,h in ref['dataset_manifest_bindings'].items():
  bs=[b for b in ds.manifest['evidence'] if Path(b['path']).name==Path(original).name]
  require(len(bs)==1 and bs[0]['sha256']==h,'PG1_FROZEN_MANIFEST '+original)
 for name in ('TEACHER_DATA_GATE.json','DATA_CONTRACT_GATE.json'):require(evidence[name]['passed'] is True,'PG1_ORIGINAL_DATA_GATE')
 seal=evidence['PROTOCOL_SEAL.json']['files']
 for b in ds.manifest['evidence']:
  p=resolve(b)
  if p.name in seal:require(b['sha256']==seal[p.name],'PG1_ORIGINAL_DATA_SEAL')
 require(evidence['TEACHER_DATA_GATE.json']['dataset_manifest_sha']==seal['DATASET_MANIFEST.json'],'PG1_TEACHER_MANIFEST')
 original={}
 for row in evidence['DATASET_MANIFEST.json']['rows']:
  if row['block']=='train_orth':original[row['id']]=(row['M'],row['npz_sha256'],'teachers')
 for row in evidence['REPLAY_MANIFEST.json']:original[row['id']]=(row['M'],row['npz_sha'],'experts')
 for sid,row in ds.rows.items():
  require(sid in original and original[sid][:2]==(row['M'],row['artifact']['sha256']),'PG1_ORIGINAL_TASK')
  require(all(Path(c['path']).resolve().parent==Path(row['artifact']['path']).resolve().parent for c in row['certificates']),'PG1_CERT_TASK_PATH')
  key=original[sid][2];certs=[read(resolve(c)) for c in row['certificates']]
  require(any(key in c and len(c[key])==row['M'] and all(t.get('verification',{}).get('passed') is True and t.get('task',{}).get('passed') is True for t in c[key]) for c in certs),'PG1_POSITIVE_TEACHER')
 require(len(m['orders'])==3 and all(type(x['seed']) is int for x in m['orders']) and {x['seed'] for x in m['orders']}=={42,43,44},'PG1_ALL_ORDERS')
 expected={Path(x['path']).name:x['sha256'] for x in evidence['DATA_BINDINGS.json']['orders']}
 orders={}
 for row in m['orders']:
  exact(row,['seed','artifact'],'PG1_ORDER');o=FrozenOrder(row['artifact'] if store is None else store.binding(row['artifact']),ds,'b2c1_reference_v1')
  require(o.path.name=='C1_orthogonal_seed%d.npz'%row['seed'] and sha(o.path)==expected[o.path.name]==seal['orders/'+o.path.name],'PG1_ORIGINAL_ORDER')
  orders[row['seed']]=o
 geo=GeometrySidecar(m['sidecar'],ds) if store is None else MappedGeometry(m['sidecar'],ds,store)
 require((geo.manifest['orders'] if store is None else [dict(seed=x['seed'],artifact=store.binding(x['artifact'])) for x in geo.manifest['orders']])==m['orders'],'PG1_SIDECAR_ALL_ORDERS')
 paths=qualify_development(m['development'],set(ds.rows),{r['scene_id'] for r in geo.rows.values()}) if store is None else mapped_development(m['development'],set(ds.rows),{r['scene_id'] for r in geo.rows.values()},store)
 masked=0;ids=list(ds.rows)
 for i in range(0,384,8):
  _,gates=authenticated_task_gates(ds.load(ids[i:i+8]),geo)
  masked+=sum(not g['goal_enabled'] for g in gates)
 require(masked==0,'PG1_POSITIVE_TEACHER_PROOF_CONTRADICTION')
 return ds,geo,dict(passed=True,identity=identity,rows=384,dataset=m['dataset'],sidecar=m['sidecar'],development=m['development'],orders={str(s):dict(artifact=m['orders'][i]['artifact'],summary=orders[s].summary) for i,s in enumerate(x['seed'] for x in m['orders'])},development_paths_verified=len(paths),train_dev_id_overlap=0,train_dev_scene_overlap=0,goal_masked_count=masked,teacher_counts=dict(Counter(r['M'] for r in ds.rows.values())),regimes=dict(Counter(r['regime'] for r in ds.rows.values())),provenance=geo.manifest['provenance'])

def qualify_bindings(binding,seed):
 m=read(artifact(binding))
 portable=m.get('schema')=='pg1_portable_data_bindings_v1'
 store=None
 if portable:
  from .transport import Store
  store=Store(m['transport']);store.verify()
  require(m['origin_binding_sha256']==store.manifest['original_binding']['sha256'],'TRANSPORT_ORIGIN')
 exact(m,['schema','code','historical_reference','dataset','sidecar','orders','development','initializations']+(['transport','origin_binding_sha256'] if portable else []),'PG1_BINDINGS_V2')
 require(m['schema'] in ('pg1_data_bindings_v2','pg1_portable_data_bindings_v1') and type(seed) is int and seed in (42,43,44),'PG1_BINDINGS_SCHEMA_SEED')
 ds,geo,gate=qualify_data(m,store)
 require(len(m['initializations'])==3 and all(type(x['seed']) is int for x in m['initializations']) and {x['seed'] for x in m['initializations']}=={42,43,44},'PG1_INITIALIZATIONS')
 audits={}
 for init in m['initializations']:
  order=next(x['artifact'] for x in m['orders'] if x['seed']==init['seed'])
  _,audits[str(init['seed'])]=load_legacy(init,init['seed'],order)
 require(source_files()==m['code']['files'],'PG1_QUALIFICATION_SOURCE_DRIFT')
 init=next(x for x in m['initializations'] if x['seed']==seed)
 return ds,geo,init,dict(passed=True,schema='pg1_g2_qualification_v2',binding=binding,current_code=m['code'],data=gate,initializations=audits,selected_seed=seed)
