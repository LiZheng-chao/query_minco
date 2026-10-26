"""Synthetic CPU-only P0 regressions. No optimizer, frozen assets, CUDA or ROS.
RowSidecar exercises real per-row source/anchor/tensor authentication; it is NOT
the 384-row/three-seed real-dataset qualification.
"""
import copy,json,os,subprocess,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from query_minco.models import QueryMincoPlanner
from query_minco.trajectory import TrajectoryPostprocessor
from query_minco.training.io import artifact,digest,sha
from query_minco.training.datasets.geometry_sidecar import GeometrySidecar,geometry_digest
from query_minco.training.pilot_spec import recipe as pilot_recipe,IDS
from query_minco.training.pilot_worker import batch
from query_minco.training.pilot_audit import audit_batch
from query_minco.training.pilot_evaluation import evaluate_model
from query_minco.training.objective_rsps import cpu_dry_run,recipe as dry_recipe
from query_minco.training.audit_rsps import numpy_physical,audit_record
from query_minco.training.run_spec import source_files
from query_minco.training.losses.set_physical_risk import physical_loss,box_sdf
from query_minco.inference.identity import state_sha

ROOT=Path(__file__).resolve().parents[1]
def binding(p):return dict(path=str(p.resolve()),sha256=sha(p))
def dump(p,value):p.write_text(json.dumps(value,allow_nan=False))
def code_binding():
 return dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip(),files=source_files(),allow_uncommitted_sources=True)
def fixture_raw():
 # A global quintic 0 ->1m in2s, v0=.8, vT=0, a0=aT=0.
 A=np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,2,0,0,0],
             [1,2,4,8,16,32],[0,1,4,12,32,80],[0,0,2,12,48,160]],float)
 c=np.linalg.solve(A,[0,.8,0,1,0,0])
 x=np.polynomial.polynomial.polyval(np.array([2/3,4/3,2]),c)
 points=np.stack([x,np.zeros(3),np.zeros(3)],-1)
 low=np.array([0,-4,-2]);high=np.array([10,4,2])
 raw=np.zeros((6,21),np.float32)
 raw[:,:9]=np.arctanh(2*(points-low)/(high-low)-1).reshape(9)
 raw[:,15]=np.log(6.)
 return torch.tensor(raw)
class RawModel(torch.nn.Module):
 def __init__(self):super().__init__();self.raw=torch.nn.Parameter(fixture_raw())
 def forward(self,depth_input,state):return self.raw[None].expand(len(state),-1,-1)
class RowSidecar(GeometrySidecar):
 def __init__(self,path,samples,rows):
  self.path=path;self.binding=binding(path);self.rows={r['sample_id']:r for r in rows};self._json_cache={}
  original={x.sample_id:copy.deepcopy(x) for x in samples}
  drows={r['sample_id']:dict(split='train',artifact=r['sample'],certificates=r['certificates']) for r in rows}
  self.dataset=SimpleNamespace(rows=drows,load=lambda ids:[copy.deepcopy(original[i]) for i in ids],check_artifact=artifact)
  ids=[x.sample_id for x in samples]
  self.orders={s:SimpleNamespace(ids=[ids],batch=lambda i:(ids,['synthetic']*len(ids))) for s in (42,43,44)}
class Fixtures:
 def __init__(self,path,kinds,split='train',domain=None,goal=1.):
  self.path=path;path.mkdir();self.samples=[];self.rows=[];self.metadata=[];self.original=[]
  self.domain=domain or [[-3.,-3.,-3.],[6.,3.,3.]]
  state=torch.tensor([.8,0,0,0,0,0,goal,0,0,0,0,-1],dtype=torch.float32)
  with torch.no_grad():teacher=TrajectoryPostprocessor()(fixture_raw()[None],state[None],num_samples=32)
  for i,kind in enumerate(kinds):
   sid='synthetic_%d'%i;M=1 if i%2==0 else 4
   sample=SimpleNamespace(sample_id=sid,depth_input=torch.stack([torch.full((128,160),.5),torch.ones(128,160)]),
    state=state.clone(),teacher_coeffs=teacher.trajectory.coeffs[0,:M].clone(),teacher_durations=teacher.parameters.durations[0,:M].clone())
   self.samples.append(sample)
   obs=[] if kind=='free' else [dict(name='box',min=[.4,-.25,-.25],max=[.7,.25,.25])]
   scene=dict(kind=kind,obstacles=obs,flight_center_domain_m=self.domain,frame='fixed B0 FLU',family_id='synthetic',crossing_x_m=.5)
   if kind=='free':scene['certified_empty']=True
   meta=dict(id=sid,split=split,state=state.tolist(),scene=scene,teacher_modes=['direct'])
   mp=path/(sid+'.json');dump(mp,meta);self.metadata.append(mp)
   npz=path/(sid+'.npz')
   with npz.open('xb') as f:np.savez(f,**{k:getattr(sample,k).numpy() for k in ['depth_input','state','teacher_coeffs','teacher_durations']})
   cp=path/(sid+'_checks.json');dump(cp,dict(scope='SYNTHETIC_FIXTURE_NOT_A_SAFETY_CERTIFICATE'))
   self.original.append(dict(id=sid,split=split,M=M,metadata=str(mp),metadata_sha256=sha(mp),npz=str(npz),npz_sha256=sha(npz),checks=str(cp),checks_sha256=sha(cp)))
   contract=path/(sid+'_contract.json');dump(contract,dict(split='train',scene=scene,body_radius_m=.12,margin_m=.03,camera=dict(optical_to_B0_rotation=[[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]],origin_B0=[0.,0.,0.])))
   self.rows.append(dict(sample_id=sid,split='train',scene_id=geometry_digest(scene),family='synthetic',sample=binding(npz),certificates=[binding(cp)],
    scene_source=binding(mp),scene_pointer='/scene',anchor=None,contract_source=binding(contract),frame='fixed B0 FLU',units='m_s',body_radius_m=.12,safety_margin_m=.03,obstacles=obs,
    flight_center_domain_m=self.domain,domain_semantics='center_bounds_not_occupied_solids',empty_certified=kind=='free'))
  self.anchor=path/'original.json';dump(self.anchor,dict(rows=self.original))
  for row in self.rows:row['anchor']=dict(kind='metadata',artifact=binding(self.anchor),seal=None)
  self.manifest=path/'geometry.json';dump(self.manifest,dict(scope='SYNTHETIC_ROW_AUTH_ONLY',rows=self.rows))
  self.geo=RowSidecar(self.manifest,self.samples,self.rows)
 def dev(self):
  assert len(self.samples)==8
  rows=[]
  for sample,row,src in zip(self.samples,self.rows,self.original):
   rows.append(dict(id=sample.sample_id,M=src['M'],scene_sha256=row['scene_id'],metadata=row['scene_source'],tensors=row['sample'],certificates=row['certificates'][0]))
  p=self.path/'development.json';dump(p,dict(schema='rsps_open_dev8_v1',source=binding(self.anchor),rows=rows))
  return binding(p)
 def revoke_source_empty(self):
  mp=self.metadata[0];meta=json.loads(mp.read_text());meta['scene'].pop('certified_empty');dump(mp,meta)
  self.rows[0]['scene_source']=binding(mp);self.original[0]['metadata_sha256']=sha(mp);dump(self.anchor,dict(rows=self.original))
  for row in self.rows:row['anchor']['artifact']=binding(self.anchor)
  dump(self.manifest,dict(scope='SYNTHETIC_ROW_AUTH_ONLY',rows=self.rows));self.geo.binding=binding(self.manifest)
  self.geo._json_cache.clear()

class CertifiedEmptyP0(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
  cls.old={}
  for name in ['pilot_worker','audit_rsps','pilot_evaluation']:
   namespace=dict(__package__='query_minco.training')
   source=subprocess.check_output(['git','show','HEAD:query_minco/training/'+name+'.py'],cwd=str(ROOT),text=True)
   exec(compile(source,'HEAD_BEFORE_P0_'+name,'exec'),namespace);cls.old[name]=namespace
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.counter=0
  self.step_guard=patch('torch.optim.Adam.step',side_effect=AssertionError('optimizer.step forbidden'));self.step_guard.start()
 def tearDown(self):self.step_guard.stop();self.temp.cleanup()
 def fixture(self,kinds,**kw):
  self.counter+=1;return Fixtures(self.root/str(self.counter),kinds,**kw)
 def spec(self,arm):
  cfg=pilot_recipe(arm)
  return dict(device='cpu',objective_arm=arm,recipe=cfg,recipe_sha256=digest(cfg),training_method_id=IDS[arm],code=code_binding())
 def test_four_arms_free_obstacle_mixed_and_pilot_numpy_audit(self):
  for kinds in [['free']*8,['obstacle']*8,['free','obstacle']*4]:
   f=self.fixture(kinds)
   for arm in ['E0','E1','E2','E3']:
    with self.subTest(kinds=kinds,arm=arm):
     m=RawModel();before=state_sha(m);spec=self.spec(arm);rec,a=batch(m,f.samples,f.geo,spec)
     self.assertEqual(rec['certified_empty'],[k=='free' for k in kinds]);self.assertEqual(a['per_query'].shape,(8,6))
     self.assertEqual(state_sha(m),before);self.assertFalse(np.count_nonzero(a['raw_grad'][...,19:21]))
     self.assertTrue(audit_batch(spec,rec,a,f.samples,f.geo,0)['passed'])
     for i,kind in enumerate(kinds):
      if kind=='free':self.assertEqual(rec['sampled_collision'][i],[False]*6);self.assertFalse(np.count_nonzero(a['per_query'][i]));self.assertFalse(np.count_nonzero(a['physical_raw_grad'][i]))
     if arm=='E0':
      self.assertFalse(np.count_nonzero(a['physical_raw_grad']));self.assertFalse(np.count_nonzero(a['raw_grad'][~a['mask']]))
     for kind in ['loss','raw19']:
      badrec=copy.deepcopy(rec);bada=copy.deepcopy(a)
      if kind=='loss':badrec['loss']+=.1
      else:bada['raw_grad'][0,0,19]=1
      with self.assertRaises(ValueError):audit_batch(spec,badrec,bada,f.samples,f.geo,0)
 def test_real_backbone_mixed_cpu_no_update(self):
  f=self.fixture(['free','obstacle'])
  for arm in ['E0','E1','E2','E3']:
   with self.subTest(arm=arm):
    torch.manual_seed(42);m=QueryMincoPlanner();before=state_sha(m)
    rec,a=batch(m,f.samples,f.geo,self.spec(arm));self.assertEqual(before,state_sha(m));self.assertEqual(a['raw'].shape,(2,6,21))
    self.assertTrue(np.isfinite(a['raw_grad']).all());self.assertFalse(np.count_nonzero(a['raw_grad'][...,19:21]))
 def test_original_nonempty_four_arm_arrays_bitwise_unchanged(self):
  f=self.fixture(['obstacle']*8)
  for arm in ['E0','E1','E2','E3']:
   with self.subTest(arm=arm):
    spec=self.spec(arm);a,aa=self.old['pilot_worker']['batch'](RawModel(),f.samples,f.geo,spec)
    b,bb=batch(RawModel(),f.samples,f.geo,spec)
    self.assertEqual(set(aa),set(bb))
    for key in aa:self.assertTrue(np.array_equal(aa[key],bb[key]),key)
    for key in ['loss','imitation','physical','matches','physical_imitation_gradient_ratio','gradient_cosine']:self.assertEqual(a[key],b[key])
 def test_e0_matches_original_imitation_gradient_exactly(self):
  from query_minco.training.objective import forward
  f=self.fixture(['free','obstacle']*4);m=RawModel();rec,a=batch(m,f.samples,f.geo,self.spec('E0'))
  m.zero_grad(set_to_none=True);loss,_,raw,_,_=forward(m,f.samples,'B_set','cpu');loss.backward()
  self.assertEqual(rec['loss'],float(loss));self.assertTrue(np.array_equal(a['raw_grad'],raw.grad.numpy()))
 def test_untrusted_train_geometry_all_four_arms_fail_before_network(self):
  for label in ['missing','false','string','source_certificate','source_hash','contradiction']:
   f=self.fixture(['free','obstacle']*4)
   if label=='missing':f.geo.rows['synthetic_0'].pop('empty_certified')
   elif label=='false':f.geo.rows['synthetic_0']['empty_certified']=False
   elif label=='string':f.geo.rows['synthetic_0']['empty_certified']='true'
   elif label=='source_certificate':f.revoke_source_empty()
   elif label=='source_hash':f.geo.rows['synthetic_0']['scene_source']['sha256']='0'*64
   else:f.geo.rows['synthetic_1']['empty_certified']=True
   for arm in ['E0','E1','E2','E3']:
    with self.subTest(label=label,arm=arm),patch('query_minco.training.pilot_worker.objective.forward',side_effect=AssertionError('untrusted must reject first')) as network:
     with self.assertRaises((ValueError,KeyError)):batch(RawModel(),f.samples,f.geo,self.spec(arm))
     network.assert_not_called()
 def test_forged_empty_boxes_with_nonempty_row_rejected(self):
  f=self.fixture(['obstacle']*8);fields=f.geo.load(f.samples);fields[0].boxes=torch.empty(0,2,3)
  fake=SimpleNamespace(load=lambda samples:fields)
  for arm in ['E0','E1','E2','E3']:
   with self.assertRaisesRegex(ValueError,'GEOMETRY_BOX_BINDING'):batch(RawModel(),f.samples,fake,self.spec(arm))
 def test_box_sdf_direct_empty_contract_unchanged(self):
  with self.assertRaises(ValueError):box_sdf(torch.zeros(6,10,3),torch.empty(0,2,3))
 def dry(self,f,arm):
  m=RawModel();cp=f.path/('weights%d.pt'%arm);torch.save(dict(state_dict=m.state_dict()),cp)
  method='query_minco_rsps_e%d_cpu_dry_v1'%arm
  spec=dict(schema='rsps_cpu_zero_update_v1',training_method_id=method,seed=42,batch_index=0,checkpoint=binding(cp),weights_sha256=state_sha(m),sidecar=f.geo.binding,code=code_binding(),recipe_sha256=digest(dry_recipe(method)))
  z=cpu_dry_run(m,f.samples,f.geo,method,spec)
  return z,spec,method
 def test_independent_full_audit_free_obstacle_mixed_and_strict_json(self):
  for kinds in [['free']*8,['obstacle']*8,['free','obstacle']*4]:
   for arm in [1,2,3]:
    f=self.fixture(kinds);z,spec,method=self.dry(f,arm)
    r=audit_record(z['record'],z['arrays'],f.samples,f.geo,42,0,method,spec)
    json.dumps(r,allow_nan=False)
    self.assertTrue(r['passed']);self.assertEqual(np.asarray(r['sampled_collision']).shape,(8,6))
    for i,kind in enumerate(kinds):
     if kind=='free':
      self.assertEqual(r['minimum_sampled_surface_clearance'][i],[None]*6)
      self.assertEqual(r['sampled_collision'][i],[False]*6);self.assertEqual(r['sampled_margin_violation'][i],[False]*6)
    if 'free' in kinds:
     with self.assertRaises(ValueError):self.old['audit_rsps']['audit_record'](z['record'],z['arrays'],f.samples,f.geo,42,0,method,spec)
 def test_independent_audit_negative_records_gradients_and_geometry(self):
  f=self.fixture(['free','obstacle']*4);z,spec,method=self.dry(f,3)
  for key in ['loss','score_gradient','per_query']:
   rec=copy.deepcopy(z['record']);a=copy.deepcopy(z['arrays'])
   if key=='loss':rec['loss']+=.1
   elif key=='score_gradient':a['raw_grad'][0,0,19]=1
   else:a['per_query'][0,0]=1
   with self.subTest(key=key),self.assertRaises(ValueError):audit_record(rec,a,f.samples,f.geo,42,0,method,spec)
  f.geo.rows['synthetic_0']['empty_certified']=False
  with self.assertRaises(ValueError):audit_record(z['record'],z['arrays'],f.samples,f.geo,42,0,method,spec)
 def test_nonempty_hard_verifier_result_unchanged(self):
  import query_minco.evaluation.validation as current
  namespace=dict(__package__='query_minco.evaluation')
  old=subprocess.check_output(['git','show','HEAD:query_minco/evaluation/validation.py'],cwd=str(ROOT),text=True)
  exec(compile(old,'ORIGINAL_HARD_VERIFIER','exec'),namespace)
  f=self.fixture(['obstacle']*8);sample=f.samples[0]
  out=TrajectoryPostprocessor()(fixture_raw()[None],sample.state[None])
  args=[out.trajectory.coeffs[0,0].numpy(),out.parameters.durations[0,0].numpy(),out.parameters.head_pva[0,0].numpy(),out.parameters.tail_pva[0,0].numpy(),out.parameters.inner_pts[0,0].numpy(),json.loads(f.metadata[0].read_text())['scene']]
  self.assertEqual(namespace['validate_teacher'](*args),current.validate_teacher(*args))
 def test_numpy_physical_matches_torch_free_and_mixed(self):
  f=self.fixture(['free','obstacle']*4);_,a=batch(RawModel(),f.samples,f.geo,self.spec('E2'));cfg=pilot_recipe('E2')
  boxes=[x.boxes for x in f.geo.load(f.samples)]
  p,v,w=[torch.from_numpy(a[k]) for k in ['positions','velocities','time_weights']]
  pq,_=physical_loss(p,v,w,boxes,geometry_bound=[True]*8,radius=.12,margin=.03,eps=.05,beta=1.)
  actual=numpy_physical(p.numpy(),v.numpy(),w.numpy(),[b.numpy() for b in boxes],cfg)
  self.assertTrue(np.allclose(actual,pq.numpy(),rtol=2e-5,atol=2e-7))
 def test_full_evaluation_free_obstacle_mixed(self):
  for kinds in [['free']*8,['obstacle']*8,['free','obstacle']*4]:
   f=self.fixture(kinds,split='development');binding_=f.dev();m=RawModel();result=evaluate_model(m,binding_)
   self.assertTrue(m.training);json.dumps(result,allow_nan=False);self.assertEqual(len(result['rows']),8)
   for row,kind in zip(result['rows'],kinds):
    self.assertEqual(row['certified_empty'],kind=='free')
    if kind=='free':
     self.assertEqual(row['minimum_obstacle_clearance'],[None]*6);self.assertEqual(row['sampled_margin_violation'],[False]*6)
     self.assertTrue(all(row['physics']));self.assertTrue(all(row['goal']));self.assertTrue(row['complete_oracle'])
    else:self.assertTrue(any(row['conservative_collision']))
   if 'free' in kinds:
    with self.assertRaises(ValueError):self.old['pilot_evaluation']['evaluate_model'](RawModel(),binding_)
   else:
    old=self.old['pilot_evaluation']['evaluate_model'](RawModel(),binding_)
    stripped=copy.deepcopy(result)
    for x in stripped['rows']:
     for key in ['certified_empty','minimum_obstacle_clearance','sampled_collision','sampled_margin_violation']:x.pop(key)
    self.assertEqual(old,stripped)
 def test_certified_empty_does_not_override_domain_goal_or_dynamic_checks(self):
  for case in ['domain','goal','dynamics']:
   f=self.fixture(['free']*8,split='development',domain=[[-.01]*3,[.01]*3] if case=='domain' else None,goal=2.4 if case=='goal' else 1.)
   m=RawModel()
   if case=='dynamics':
    with torch.no_grad():m.raw[:,9]=2.
   result=evaluate_model(m,f.dev());json.dumps(result,allow_nan=False)
   for row in result['rows']:
    self.assertFalse(row['complete_oracle']);self.assertEqual(row['sampled_margin_violation'],[False]*6)
    if case=='goal':self.assertFalse(any(row['goal']))
    else:self.assertFalse(any(row['physics']))
    if case=='domain':self.assertTrue(all(row['domain_violation']))
 def test_dev_missing_certification_and_provenance_fail_closed(self):
  for label in ['missing','false','not_free','source_hash','metadata_hash','scene_hash','contradiction']:
   f=self.fixture(['free','obstacle']*4,split='development');b=f.dev()
   m=json.loads(Path(b['path']).read_text())
   if label in ['missing','false','not_free','contradiction']:
    i=1 if label=='contradiction' else 0;mp=f.metadata[i];meta=json.loads(mp.read_text())
    if label=='missing':meta['scene'].pop('certified_empty')
    elif label=='false':meta['scene']['certified_empty']=False
    elif label=='not_free':meta['scene']['kind']='unknown'
    else:meta['scene']['certified_empty']=True
    dump(mp,meta);m['rows'][i]['metadata']=binding(mp)
   elif label=='source_hash':m['source']['sha256']='0'*64
   elif label=='metadata_hash':m['rows'][0]['metadata']['sha256']='0'*64
   else:m['rows'][0]['scene_sha256']='0'*64
   dump(Path(b['path']),m);b=binding(Path(b['path']))
   with self.subTest(label=label),self.assertRaises((ValueError,KeyError)):evaluate_model(RawModel(),b)
if __name__=='__main__':unittest.main(verbosity=2)
