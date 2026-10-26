"""Explicit immutable artifact transport. Never rewrites historical JSON or fakes old roots."""
import copy,os,shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from query_minco.training.io import artifact,read,write,require,exact,sha,digest
from query_minco.training.run_spec import ROOT,source_files,verify_code
from query_minco.training.datasets.manifest import FrozenDataset
from query_minco.training.datasets.geometry_sidecar import GeometrySidecar,validate_semantics
from query_minco.training.datasets.order import FrozenOrder
from query_minco.training.pilot_evaluation import _certified_empty

def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))

class Store:
 def __init__(self,binding):
  self.path=artifact(binding);self.root=self.path.parent;self.manifest=read(self.path)
  require(self.manifest['schema']=='pg1_transfer_v1','TRANSFER_SCHEMA')
  self.rows={r['original_path']:r for r in self.manifest['files']}
  require(len(self.rows)==len(self.manifest['files']),'TRANSFER_DUPLICATES')
  self.reverse={}
  for logical,r in self.rows.items():
   require(Path(logical).is_absolute() and r['relative_path']=='tree/'+logical.lstrip('/'),'TRANSFER_LAYOUT')
   p=self.root/r['relative_path'];require(p.resolve()==p,'TRANSFER_ESCAPE_OR_SYMLINK')
   require(str(p) not in self.reverse,'TRANSFER_ALIAS')
   self.reverse[str(p)]=logical
 def artifact(self,b):
  exact(b,['path','sha256'],'transport_artifact')
  logical=b['path'] if b['path'] in self.rows else self.reverse.get(b['path'])
  require(logical in self.rows,'TRANSFER_UNKNOWN_PATH '+str(b['path']))
  r=self.rows[logical];require(r['sha256']==b['sha256'],'TRANSFER_EXPECTED_SHA')
  p=self.root/r['relative_path'];require(p.resolve()==p and p.is_file() and sha(p)==r['sha256'],'TRANSFER_BYTES '+str(p))
  return p
 def binding(self,b):return dict(path=str(self.artifact(b)),sha256=b['sha256'])
 def read(self,b):return read(self.artifact(b))
 def verify(self):
  for logical,r in self.rows.items():self.artifact(dict(path=logical,sha256=r['sha256']))
  return dict(passed=True,files=len(self.rows),bytes=sum(r['size'] for r in self.rows.values()))

class MappedDataset(FrozenDataset):
 """Reuses original tensor loader/checks; constructor validates unchanged raw manifest."""
 def __init__(self,binding,store):
  self.store=store;self.path=store.artifact(binding);self.manifest=store.read(binding);m=self.manifest
  exact(m,['schema','qualification','roots','rows','evidence'],'cache_manifest')
  require(m['schema']==1 and m['qualification']=='TRAIN','MAPPED_TRAIN_ONLY')
  self.roots=[Path(p) for p in m['roots']]
  require(self.roots and all(p.is_absolute() for p in self.roots),'MAPPED_ORIGINAL_ROOTS')
  self.rows={};self.files=[self.path]
  for r in m['rows']:
   exact(r,['id','artifact','split','regime','M','roles','teacher_status','certificates'],'cache_row')
   require(r['id'] not in self.rows and r['split']=='train' and r['regime'] in ('near','far') and r['M'] in (1,4) and r['teacher_status']=='verified_positive','MAPPED_ROW')
   require(r['roles'] and len(set(r['roles']))==len(r['roles']) and r['certificates'],'MAPPED_ROLES_CERTS')
   self.files.append(self.check_artifact(r['artifact']))
   for c in r['certificates']:self.files.append(self.check_artifact(c))
   self.rows[r['id']]=r
  require(len(self.rows)==384,'MAPPED384')
  for b in m['evidence']:self.files.append(self.check_artifact(b))
 def check_artifact(self,b):
  logical=self.store.reverse.get(b['path'],b['path']);p=Path(logical)
  require(p.is_absolute() and '..' not in p.parts and any(p==r or r in p.parents for r in self.roots),'MAPPED_CACHE_ROOT')
  require(not any('blind' in part.lower() for part in p.parts),'BLIND_PATH')
  return self.store.artifact(b)

class MappedGeometry(GeometrySidecar):
 """Original per-row scene/anchor/contract and tensor predicates run unchanged."""
 def __init__(self,binding,ds,store):
  self.store=store;self.path=store.artifact(binding);self.binding=store.binding(binding)
  self.dataset=ds;self.manifest=store.read(binding);self._json_cache={}
  m=self.manifest
  exact(m,['schema','qualification','dataset','orders','provenance','rows'],'geometry_sidecar')
  require(m['schema']=='rsps_geometry_sidecar_v1' and m['qualification']=='TRAIN_GEOMETRY_ONLY_NOT_TEACHER_REGENERATION','MAPPED_GEO_SCHEMA')
  require(store.artifact(m['dataset'])==ds.path and m['provenance']['status']=='still_open' and m['provenance']['blocks_geometry_qualification'] is False,'MAPPED_GEO_DATA')
  require(len(m['rows'])==len(ds.rows)==384,'MAPPED_GEO384')
  self.rows={}
  for row in m['rows']:
   require(row['sample_id'] not in self.rows,'MAPPED_GEO_DUP')
   self._validate_row(row);self.rows[row['sample_id']]=copy.deepcopy(row)
  require(set(self.rows)==set(ds.rows),'MAPPED_GEO_IDS')
  require(len(m['orders'])==3 and {x['seed'] for x in m['orders']}=={42,43,44},'MAPPED_SEEDS')
  self.orders={x['seed']:FrozenOrder(store.binding(x['artifact']),ds,'b2c1_reference_v1') for x in m['orders']}
 def _read(self,b):return self.store.read(b)

def mapped_development(binding,train_ids,train_scenes,store):
 m=store.read(binding);require(m['schema']=='rsps_open_dev8_v1' and len(m['rows'])==8,'DEV_SCHEMA')
 original=store.read(m['source']);original={r['id']:r for r in original['rows']}
 require(len({r['id'] for r in m['rows']})==8,'DEV_DUPLICATE')
 paths=[store.artifact(binding),store.artifact(m['source'])]
 for r in m['rows']:
  src=original[r['id']]
  require(r['id'] not in train_ids and src['split']=='development' and src['M']==r['M'] and r['scene_sha256'] not in train_scenes,'DEV_TRAIN_LEAKAGE')
  for k,s in [('tensors','npz'),('metadata','metadata'),('certificates','checks')]:
   require(r[k]['sha256']==src[s+'_sha256'] and store.artifact(r[k]).name==src[s].split('/')[-1],'DEV_ORIGINAL_SHA');paths.append(store.artifact(r[k]))
  meta=store.read(r['metadata'])
  require(meta['id']==r['id'] and meta['split']=='development' and digest(dict(obstacles=meta['scene']['obstacles'],domain=meta['scene']['flight_center_domain_m']))==r['scene_sha256'],'DEV_SCENE')
  _certified_empty(meta['scene'])
 return paths

def asset_bindings(m):
 """Explicit train/dev/lineage closure only; never walks a user's home."""
 items=[m['dataset'],m['sidecar'],m['development']]+[x['artifact'] for x in m['orders']]
 ds=read(artifact(m['dataset']));geo=read(artifact(m['sidecar']));dev=read(artifact(m['development']))
 items+=ds['evidence']+[dev['source']]
 for row in ds['rows']:items+=[row['artifact']]+row['certificates']
 for row in geo['rows']:
  items+=[row['scene_source'],row['contract_source'],row['anchor']['artifact']]
  if row['anchor'].get('seal'):items.append(row['anchor']['seal'])
 for row in dev['rows']:items +=[row[k] for k in ('tensors','metadata','certificates')]
 for init in m['initializations']:
  items +=[init['checkpoint'],init['lineage']]
  items +=list(read(artifact(init['lineage']))['evidence'].values())
 return items

def export_package(binding,out,copy_files=False):
 m=read(artifact(binding));verify_code(m['code'])
 require(m['schema']=='pg1_data_bindings_v2','EXPORT_NATIVE_ONLY')
 out=Path(out);out.mkdir(parents=True,exist_ok=False);rows={}
 for b in asset_bindings(m)+[binding]:
  p=artifact(b);logical=str(p)
  if logical in rows:require(rows[logical]['sha256']==b['sha256'],'EXPORT_CONFLICT');continue
  relative='tree/'+logical.lstrip('/')
  rows[logical]=dict(original_path=logical,relative_path=relative,sha256=b['sha256'],size=p.stat().st_size)
  if copy_files:
   target=out/relative;target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('xb') as f,p.open('rb') as source:shutil.copyfileobj(source,f)
   require(sha(target)==b['sha256'],'EXPORT_COPY_SHA')
 write(out/'TRANSFER.json',dict(schema='pg1_transfer_v1',files=list(rows.values()),original_binding=binding,source_code=m['code'],materialized=copy_files))
 return bind(out/'TRANSFER.json')

def import_binding(package,out):
 store=Store(package);store.verify();p=store.manifest
 require(p['materialized'] is True,'TRANSFER_NOT_MATERIALIZED')
 verify_code(p['source_code']) # Relocated code must have exactly the same HEAD/files.
 origin=store.read(p['original_binding']);m=copy.deepcopy(origin);out=Path(out);out.mkdir(parents=True,exist_ok=False)
 for k in ('dataset','sidecar','development'):m[k]=store.binding(origin[k])
 m['orders']=[dict(seed=x['seed'],artifact=store.binding(x['artifact'])) for x in origin['orders']]
 for init in m['initializations']:
  desc=store.read(init['lineage']);desc['evidence']={k:store.binding(v) for k,v in desc['evidence'].items()}
  target=out/('A%d_EVIDENCE.json'%init['seed']);write(target,desc)
  init.update(checkpoint=store.binding(init['checkpoint']),lineage=bind(target))
 m.update(schema='pg1_portable_data_bindings_v1',transport=package,origin_binding_sha256=p['original_binding']['sha256'],historical_reference=bind(ROOT/'query_minco/methods/reference.json'))
 write(out/'BINDINGS.json',m);return bind(out/'BINDINGS.json')
