"""Frozen-model paired native-forest snapshots. No training, ROS or control publisher."""
from pathlib import Path
import sys,os,json,time,math,hashlib,argparse,importlib.util,ast,subprocess,traceback,csv
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import query_minco
from query_minco.models import QueryMincoPlanner
from query_minco.trajectory import TrajectoryPostprocessor
from query_minco.inference.identity import state_sha
from query_minco.inference.planner_adapter import InferenceAdapter
from query_minco.evaluation.validation import bernstein_controls,evaluate,axis_range
from query_minco.evaluation.goal import task_metrics
OUT=ROOT/'research_outputs/rsps_forest15m_v1'
PREP=OUT/'preparation001'
HIST=Path('/home/lizheng/reach_planner/query_minco')
NATIVE=HIST/'f0plus_20261021_v1'
PY='/home/lizheng/anaconda3/envs/yopo/bin/python'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def clean(v):
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,(np.bool_,)):return bool(v)
 if isinstance(v,(np.integer,)):return int(v)
 if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
 return v
def write(p,v):
 with Path(p).open('x') as f:json.dump(clean(v),f,indent=2,allow_nan=False)
def require(ok,why):
 if not ok:raise ValueError(why)
def load_external(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
def rotation(q):
 x,y,z,w=np.asarray(q,float);require(abs(np.linalg.norm(q)-1)<1e-6,'QUATERNION')
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def legacy():
 voxel=load_external('forest_voxel_frozen',HIST/'scripts/f0_continuous_swept_voxel_v1.py')
 dyn=load_external('forest_dynamics_frozen',NATIVE/'dynamics.py')
 # Reuse exact pure historical functions without importing its common.py/sys.path mutations.
 source=ast.parse((NATIVE/'sensor.py').read_text())
 names={'encode','rays'};nodes=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names]
 require({n.name for n in nodes}==names,'SENSOR_FUNCTION_CLOSURE')
 scope=dict(np=np,rotation=rotation)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(NATIVE/'sensor.py'),'exec'),scope)
 return SimpleNamespace(voxel=voxel,dynamics=dyn,encode=scope['encode'],rays=scope['rays'])

def local_state(global_goal,position,R):
 relative=R.T@(np.asarray(global_goal)-position);distance=float(np.linalg.norm(relative))
 require(np.isfinite(relative).all() and distance>=2.4-2e-6,'LOCAL_GOAL_BELOW_SUPPORTED_RANGE')
 goal=relative*min(1.,2.4/distance);goal[np.abs(goal)<1e-6]=0
 require(goal[0]>0 and abs(goal[1])<1e-5 and abs(goal[2])<1e-5,'UNSUPPORTED_LOCAL_GOAL_DIRECTION')
 state=np.r_[.8,0.,0.,0.,0.,0.,goal,R.T@np.array([0.,0.,-1.])].astype(np.float32)
 return state,dict(global_goal_W=np.asarray(global_goal).tolist(),local_goal_B0=goal.tolist(),global_distance_m=distance,
                   clipped=distance>2.4,local_policy='straight_ray_2p4m_no_GT_no_global_search')
def load_world(folder,lib):
 m=read(folder/'occupancy_metadata.json');p=folder/m['voxel_file'];shape=tuple(m['shape_xyz'])
 require(m['schema']=='F0_OFFLINE_NATIVE_FOREST_VOXEL_REPLICA_V1' and m['skipped_out_of_bounds_points']==0 and abs(m['resolution_m']-.1)<1e-6,'MAP_SCHEMA')
 require(p.stat().st_size==math.prod(shape),'MAP_LENGTH')
 grid=np.memmap(p,np.uint8,'r',shape=shape,order='C')
 return lib.voxel.VoxelWorld(grid,m['origin_m'],m['resolution_m'],sha(p)),m
def frozen(binding):
 require(sha(binding['checkpoint']['path'])==binding['checkpoint']['sha256'],'CHECKPOINT_SHA')
 blob=torch.load(binding['checkpoint']['path'],map_location='cpu',weights_only=True)
 if binding['label']=='C1':
  require(blob['step']==2880 and blob['seed']==42 and blob['arm']=='C1_orthogonal','C1_IDENTITY')
 else:
  require(blob['completed_optimizer_updates']==blob['order_cursor']==32 and blob['spec']['seed']==42 and
          blob['spec']['training_method_id']==binding['method_id'],'PILOT_IDENTITY')
 model=QueryMincoPlanner();model.load_state_dict(blob['state_dict'],strict=True)
 require(state_sha(model)==binding['weights_sha256'],'WEIGHTS_SHA')
 model.eval()
 for p in model.parameters():p.requires_grad_(False)
 return model
def predict(model,depth,state):
 InferenceAdapter.inputs(depth[None],state[None],None)
 start=time.perf_counter()
 with torch.inference_mode():
  raw=model(depth_input=torch.from_numpy(depth[None].copy()),state=torch.from_numpy(state[None].copy()))
  net=time.perf_counter();out=TrajectoryPostprocessor()(raw,torch.from_numpy(state[None].copy()),num_samples=65)
 return raw,out,dict(network_s=net-start,minco_and_65sample_s=time.perf_counter()-net,candidate_generation_s=time.perf_counter()-start)

def command(run,name,argv):
 # Native binaries are read-only; only current exclusive Run is writable. No ROS network.
 require(Path('/usr/bin/bwrap').is_file(),'BWRAP_MISSING')
 env=dict(PATH='/usr/bin:/bin',HOME=str(run/'runtime'),TMPDIR=str(run/'runtime'),PYTHONDONTWRITEBYTECODE='1',
          LD_LIBRARY_PATH='/opt/ros/noetic/lib:/usr/local/cuda-11.8/lib64:/usr/lib/wsl/lib',CUDA_VISIBLE_DEVICES='0',LC_ALL='C.UTF-8')
 cmd=['/usr/bin/bwrap','--die-with-parent','--unshare-net','--ro-bind','/','/','--proc','/proc','--dev-bind','/dev','/dev',
      '--bind',str(run),str(run),'--chdir',str(run)]+[str(x) for x in argv]
 write(run/'logs'/(name+'.COMMAND.json'),dict(argv=cmd,env=env,timeout_s=300))
 start=time.perf_counter()
 with (run/'logs'/(name+'.stdout.log')).open('xb') as out,(run/'logs'/(name+'.stderr.log')).open('xb') as err:
  result=subprocess.run(cmd,env=env,stdout=out,stderr=err,timeout=300)
 write(run/'logs'/(name+'.EXIT.json'),dict(returncode=result.returncode,elapsed_s=time.perf_counter()-start))
 require(result.returncode==0,'NATIVE_COMMAND_FAILED '+name)
 return time.perf_counter()-start

def bind_check():
 m=read(PREP/'configs/EXECUTION_BINDING.json')
 actual=dict(python=sys.version,executable=sys.executable,torch=torch.__version__,numpy=np.__version__,scipy=sys.modules['scipy'].__version__,yaml=yaml.__version__,query_minco_file=query_minco.__file__)
 require(m['runtime']==actual,'RUNTIME_IDENTITY')
 for p,h in m['files'].items():require(sha(p)==h,'BINDING_DRIFT '+p)
 require(str(Path(query_minco.__file__).resolve())==str(ROOT/'query_minco/__init__.py'),'IMPORT_PATH')
 for key,path in [('query',ROOT),('yopo',Path('/home/lizheng/YOPO')),('history',Path('/home/lizheng/reach_planner'))]:
  head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()
  require(head==m['heads'][key],'GIT_HEAD '+key)
 return m

def capture(run,kind,lib):
 plans=read(PREP/'configs/SCENE_PLAN.json')
 tasks=[];maprecords=[]
 selected=plans['smoke'] if kind=='smoke' else plans['forest']
 seen=plans['known_map_hashes'];newhashes=[]
 for item in selected:
  seed=item['seed'];folder=run/'maps'/('seed%d'%seed)
  if item.get('existing_map'):
   folder=Path(item['existing_map']);world,meta=load_world(folder,lib);export_s=0.
  else:
   available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
   require(available>=3*2**30,'NATIVE_EXPORT_REQUIRES_3GIB_AVAILABLE_RAM')
   config=yaml.safe_load((NATIVE/'configs/seed11.yaml').read_text())
   config.update(seed=seed,tree_dist=item['tree_dist'],tree_file='/home/lizheng/YOPO/Simulator/src/pointcloud/tree.ply')
   cp=run/'configs'/('seed%d.yaml'%seed)
   with cp.open('x') as f:yaml.safe_dump(config,f)
   export_s=command(run,'export_%d'%seed,[NATIVE/'build/native_export',cp,folder])
   world,meta=load_world(folder,lib)
   require(world.hash not in seen+newhashes,'MAP_LEAKAGE_OR_DUPLICATE')
   newhashes.append(world.hash)
  config=Path(meta['config_path'])
  require(int(meta['seed'])==seed and sha(meta['tree_path'])==sha('/home/lizheng/YOPO/Simulator/src/pointcloud/tree.ply'),'MAP_IDENTITY')
  rng=np.random.default_rng(seed+20261024);poses=[];accepted=[];admissions=[]
  for anchor,yaw in enumerate([0.,.5,-.5] if kind=='forest' else [0.]):
   R=rotation([0,0,math.sin(yaw/2),math.cos(yaw/2)]);start=None
   for attempt in range(600):
    p=rng.uniform([-12,-12,1.8],[12,12,2.6])
    inside=world.inside_envelope(p,p,.6);blocked=world.occupancy_near_bbox(p,p,.6) if inside else True
    admissions.append(dict(anchor=anchor,attempt=attempt,position=p.tolist(),inside=inside,blocked=blocked))
    if inside and not blocked:start=p;break
   if start is None:
    for frame,t in enumerate([0.,.2,.4] if kind=='forest' else [0.]):
     tasks.append(dict(task_id='seed%d_a%d_f%d'%(seed,anchor,frame),map_seed=seed,status='INVALID_NO_ADMITTED_START',map_sha256=world.hash))
    continue
   global_goal=start+R@np.array([15.,0,0])
   for frame,t in enumerate([0.,.2,.4] if kind=='forest' else [0.]):
    pos=start+R@np.array([.8*t,0,0]);tid='seed%d_a%d_f%d'%(seed,anchor,frame)
    state,goals=local_state(global_goal,pos,R)
    pose=dict(id=tid,position=pos.tolist(),q_xyzw=[0.,0.,math.sin(yaw/2),math.cos(yaw/2)])
    row=dict(task_id=tid,map_seed=seed,tree_dist=item['tree_dist'],anchor=anchor,frame=frame,timestamp_s=t,
       position_world_m=pos.tolist(),q_xyzw=pose['q_xyzw'],state=state.tolist(),goals=goals,synthetic_snapshot_not_measured_odom=True,
       map_dir=str(folder),map_sha256=world.hash,status='VALID')
    poses.append(pose);accepted.append(row)
  write(run/'admission'/('seed%d.json'%seed),dict(attempts=admissions,model_predictions_used=False))
  posepath=run/'configs'/('poses%d.yaml'%seed)
  with posepath.open('x') as f:yaml.safe_dump(dict(poses=poses),f)
  renderdir=run/'observations'/('seed%d'%seed)
  render_s=command(run,'render_%d'%seed,[NATIVE/'build/native_render',config,posepath,renderdir]) if poses else 0.
  for row in accepted:
   tid=row['task_id'];rawpath=renderdir/(tid+'_depth_f32.bin')
   raw=np.fromfile(rawpath,np.float32).reshape(128,160);started=time.perf_counter()
   ray=lib.rays(raw,world,row['position_world_m'],row['q_xyzw'])
   write(run/'ray_audits'/(tid+'.json'),ray)
   require(ray['passed'],'SENSOR_GT_RAY_MISMATCH '+tid)
   ray_done=time.perf_counter();enc=lib.encode(raw);encode_s=time.perf_counter();state=np.asarray(row['state'],np.float32)
   with (run/'inputs'/(tid+'.npz')).open('xb') as f:np.savez(f,depth_input=enc,state=state,depth_m=raw)
   R=rotation(row['q_xyzw']);g=R@state[6:9]+row['position_world_m']
   row.update(input_path=str(run/'inputs'/(tid+'.npz')),input_sha256=sha(run/'inputs'/(tid+'.npz')),
      raw_depth_sha256=sha(rawpath),depth_input_sha256=hashlib.sha256(enc.tobytes()).hexdigest(),
      state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),ray_audit_sha256=sha(run/'ray_audits'/(tid+'.json')),
      input_and_ray_audit_s=encode_s-started,input_encode_s=encode_s-ray_done,ray_audit_s=ray_done-started,
      global_goal_occupied=world.point_witness(np.asarray(row['goals']['global_goal_W']),.15) is not None,goal_occupied=world.point_witness(g,.15) is not None,
      local_goal_outside_map=not world.inside_envelope(g,g,.15),renderer_s_per_frame=render_s/max(1,len(poses)))
   tasks.append(row)
  maprecords.append(dict(seed=seed,tree_dist=item['tree_dist'],map_sha256=world.hash,metadata_sha256=sha(folder/'occupancy_metadata.json'),
       config_sha256=sha(config),map_dir=str(folder),export_s=export_s,render_s=render_s,shape=world.shape.tolist(),occupied_voxels=meta['occupied_voxels']))
 write(run/'TASK_MANIFEST.json',dict(schema='forest15m_shared_snapshots_v1',kind=kind,maps=maprecords,tasks=tasks,
       model_forward_calls_at_freeze=0,task_count=len(tasks),online_selector='N/A',global_goal_navigation_success='NOT_MEASURED'))
 write(run/'TASK_MANIFEST.json.seal.json',dict(sha256=sha(run/'TASK_MANIFEST.json')))
 return tasks

def probe_geometry(world,points,dense):
 # Exact min over nearby occupied voxel AABB SDFs, capped at .5m.
 # Far truncation raises soft penalty: report as diagnostic upper bound only.
 half=world.res/2;pad=.5+math.sqrt(3)*half
 lo=np.minimum(points.min(0),dense.min(0))-pad;hi=np.maximum(points.max(0),dense.max(0))+pad
 i0=np.minimum(np.maximum(np.floor((lo-world.origin)/world.res).astype(int),0),world.shape)
 i1=np.maximum(np.minimum(np.ceil((hi-world.origin)/world.res).astype(int)+1,world.shape),0)
 sub=world.grid[i0[0]:i1[0],i0[1]:i1[1],i0[2]:i1[2]]
 ii=np.argwhere(sub!=0)
 if not len(ii):return np.full(len(points),.5),None
 centers=world.origin+(ii+i0+.5)*world.res;tree=cKDTree(centers)
 ds=[]
 for p,ids in zip(points,tree.query_ball_point(points,pad)):
  if not ids:ds.append(.5);continue
  q=np.abs(p-centers[ids])-half
  sdf=np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(q.max(-1),0)
  ds.append(min(.5,float(sdf.min())))
 # A voxel center is occupied; nearest-center distance is an UPPER bound on clearance.
 upper=float(np.min(tree.query(dense,k=1)[0])-.12)
 return np.asarray(ds),upper

def connected_groups(world,curves,lib):
 parent=list(range(len(curves)));unknown=0;proven=0
 def find(x):
  while parent[x]!=x:x=parent[x]
  return x
 for i in range(len(curves)):
  for j in range(i):
   stack=[(a,b,0) for a,b in zip(curves[i],curves[j])];nodes=0;ok=True
   while stack:
    a,b,depth=stack.pop();nodes+=1;cp=np.concatenate([a,b])
    lo,hi=cp.min(0),cp.max(0)
    if world.inside_envelope(lo,hi,.15) and not world.occupancy_near_bbox(lo,hi,.15):continue
    if depth>=10 or nodes>4096:ok=False;break
    al,ar=lib.voxel.split_bezier(a);bl,br=lib.voxel.split_bezier(b)
    stack.extend([(al,bl,depth+1),(ar,br,depth+1)])
   if ok:parent[find(i)]=find(j);proven+=1
   else:unknown+=1
 return dict(geometric_connected_group_count=len({find(i) for i in range(len(curves))}),
             certified_connection_pairs=proven,unresolved_pair_relations=unknown,known_homotopy_count=None,
             semantics='Certified same-corridor connections only; unresolved groups are not proven distinct homotopy classes.')

def assess(out,state,row,world,lib):
 R=rotation(row['q_xyzw']);pos=np.asarray(row['position_world_m']);records=[];valid_curves=[]
 c=out.trajectory.coeffs[0].numpy();dt=out.parameters.durations[0].numpy();tail=out.parameters.tail_pva[0].numpy()
 for q in range(6):
  local=c[q];dur=dt[q]
  cp=np.array([bernstein_controls(a,t) for a,t in zip(local,dur)])
  worldcp=cp@R.T+pos
  structural=lib.dynamics.structural(local,dur,state,tail[q])
  dynamics=lib.dynamics.trajectory_dynamics(local,dur)
  geometry=lib.voxel.classify(world,list(worldcp),.12,.03,16,50000)
  goal=task_metrics(evaluate(local[-1],dur[-1]),evaluate(local[-1],dur[-1],1),state[6:9])
  dynfail=any(x['status']=='WITNESS_VIOLATION' for x in dynamics.values())
  unknown=geometry['status']=='UNKNOWN' or any(x['status']=='UNKNOWN' for x in dynamics.values())
  physical='FAIL' if not structural['passed'] or dynfail or geometry['status'] in ['BODY_COLLISION','MARGIN_ONLY'] else ('UNKNOWN' if unknown else 'PASS')
  complete='FAIL' if physical=='FAIL' or not goal['passed'] else ('UNKNOWN' if physical=='UNKNOWN' else 'PASS')
  dense=np.concatenate([evaluate(a,np.linspace(0,t,1025)) for a,t in zip(local,dur)])@R.T+pos
  sample=np.concatenate([evaluate(a,np.linspace(0,t,64)) for a,t in zip(local,dur)])@R.T+pos
  distances,upper=probe_geometry(world,sample,dense)
  trapezoid=np.ones(64);trapezoid[[0,-1]]=.5
  alpha=np.concatenate([t*trapezoid/63/dur.sum() for t in dur])
  soft=float(np.sum(alpha*.05*np.logaddexp(0,-(distances-.15)/.05)))
  no64=bool(np.all(distances>=.15))
  outside=not world.inside_envelope(dense.min(0),dense.max(0),.15)
  record=dict(query=q,physics=physical,goal=goal,complete=complete,structural=structural,dynamics=dynamics,geometry=geometry,terminal_state_B0=tail[q].tolist(),trajectory_horizon_s=float(dur.sum()),
       surface_clearance_lower_bound_m=.03 if geometry['status']=='CLEAR_CERTIFIED' else None,
       dense_surface_clearance_upper_bound_m=upper,domain_outside_witness=outside,
       sampled64_missed_witness_violation=no64 and geometry['status'] in ['BODY_COLLISION','MARGIN_ONLY'],
       sampled64_no_violation_but_unknown=no64 and geometry['status']=='UNKNOWN',
       uniform64_soft_penalty_truncated_voxel_SDF_upper_bound=soft,unmatched=None,
       online_safe_status='UNKNOWN_NO_CONTINUOUS_OBSERVABILITY',selector='N/A')
  records.append(record)
  if complete=='PASS':valid_curves.append(worldcp)
 modes=connected_groups(world,valid_curves,lib)
 return records,modes

def summary(run,rows):
 maps=[]
 for seed in sorted({x['map_seed'] for x in rows}):
  for label in ['C1','E0','E1','E2','E3']:
   group=[x for x in rows if x['map_seed']==seed and x['model']==label]
   candidates=[q for x in group for q in x['candidates']]
   maps.append(dict(map_seed=seed,model=label,tasks=len(group),physics_pass=float(np.mean([q['physics']=='PASS' for q in candidates])),
      goal_pass=float(np.mean([q['goal']['passed'] for q in candidates])),complete_pass=float(np.mean([q['complete']=='PASS' for q in candidates])),
      oracle_pass=float(np.mean([any(q['complete']=='PASS' for q in x['candidates']) for x in group])),
      body_collision_rate=float(np.mean([q['geometry']['status']=='BODY_COLLISION' for q in candidates])),
      conservative_nonclear_rate=float(np.mean([q['geometry']['status']!='CLEAR_CERTIFIED' for q in candidates])),
      unknown_rate=float(np.mean([q['physics']=='UNKNOWN' for q in candidates])),
      numerical_anomaly_rate=float(np.mean([q.get('numerical_status')=='FAIL' for q in candidates])),
      domain_outside_witness_rate=float(np.mean([q.get('domain_outside_witness',False) for q in candidates])),
      unmatched_collision_rate=None,unmatched_status='N/A_NO_FROZEN_TEACHER'))
 write(run/'PER_MAP.json',maps)
 with (run/'PER_MAP.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(maps[0]));w.writeheader();w.writerows(maps)
 deltas=[]
 for base in ['E0','C1']:
  for arm in ['E1','E2','E3']:
   for metric in ['physics_pass','goal_pass','complete_pass','oracle_pass']:
    by={(x['map_seed'],x['model']):x for x in maps};seeds=sorted({x['map_seed'] for x in maps})
    d=np.array([by[(s,arm)][metric]-by[(s,base)][metric] for s in seeds]);rng=np.random.default_rng(20261024)
    boot=d[rng.integers(0,len(d),size=(10000,len(d)))].mean(-1)
    deltas.append(dict(arm=arm,baseline=base,metric=metric,map_count=len(d),mean_delta=float(d.mean()),bootstrap95=np.quantile(boot,[.025,.975]).tolist(),exploratory_only=True))
 timing={}
 for label in ['C1','E0','E1','E2','E3']:
  grp=[x for x in rows if x['model']==label and x.get('timing')]
  timing[label]={k:dict(p50_s=float(np.percentile([x['timing'][k] for x in grp],50)),p95_s=float(np.percentile([x['timing'][k] for x in grp],95))) for k in grp[0]['timing']} if grp else {}
 write(run/'SUMMARY.json',dict(status='OFFLINE_SNAPSHOT_EVALUATION_COMPLETE',models=read(PREP/'configs/FROZEN_MODELS.json'),
     per_map=maps,paired_deltas=deltas,timings=timing,training=False,ROS=False,closed_loop=False,READY_FOR_15M_CLOSED_LOOP=False,
     selector='N/A',unmatched='N/A_NO_FROZEN_TEACHER',mode_semantics='geometric connectivity only, not homotopy count'))


def known_controls(run,bindings):
 from query_minco.training.pilot_evaluation import evaluate_model
 from query_minco.evaluation.validation import validate_teacher
 from query_minco.evaluation.goal import separating_slab_proof
 dev=read(ROOT/'research_outputs/rsps_gate_e_v1/configs/DEVELOPMENT_PROTOCOL.json')
 source=Path(dev['source']['path']);manifest=read(source)
 negatives=[x for x in manifest['rows'] if x['id'] in ['f02_blocked_g0_v0','f02_blocked_g3_v0']]
 require(len(negatives)==2 and all(x['split']=='development' and x['status']=='proven_infeasible' for x in negatives),'NEGATIVE_CONTROLS')
 results=[]
 for binding in bindings:
  model=frozen(binding)
  dev_result=evaluate_model(model,dict(path=str(ROOT/'research_outputs/rsps_gate_e_v1/configs/DEVELOPMENT_PROTOCOL.json'),
                                     sha256=sha(ROOT/'research_outputs/rsps_gate_e_v1/configs/DEVELOPMENT_PROTOCOL.json')))
  negative_results=[]
  for row in negatives:
   data=source.parent/row['npz'];meta=source.parent/row['metadata']
   require(sha(data)==row['npz_sha256'] and sha(meta)==row['metadata_sha256'],'CONTROL_SHA')
   with np.load(data,allow_pickle=False) as z:depth=z['depth_input'].copy();state=z['state'].copy()
   scene=read(meta)['scene'];proof=separating_slab_proof(scene,state);require(proof and proof['proven_infeasible'],'NEGATIVE_PROOF')
   raw,out,timing=predict(model,depth,state);p=out.parameters
   checks=[]
   for q in range(6):
    ck=validate_teacher(out.trajectory.coeffs[0,q].numpy(),p.durations[0,q].numpy(),
         p.head_pva[0,q].numpy(),p.tail_pva[0,q].numpy(),p.inner_pts[0,q].numpy(),scene)
    goal=task_metrics(p.tail_pva[0,q,0].numpy(),p.tail_pva[0,q,1].numpy(),state[6:9])
    checks.append(dict(physics=ck,goal=goal,complete=ck['passed'] and goal['passed']))
   require(not any(x['complete'] for x in checks),'PROVEN_NEGATIVE_FALSE_POSITIVE')
   negative_results.append(dict(id=row['id'],proof=proof,candidates=checks,timing=timing))
  results.append(dict(model=binding['label'],known_dev8=dev_result,known_proven_negative_controls=negative_results,independent_forest=False))
 write(run/'KNOWN_CONTROLS.json',results)

def evaluate_tasks(run,tasks,lib):
 bindings=read(PREP/'configs/FROZEN_MODELS.json');models={b['label']:frozen(b) for b in bindings};rows=[];cached={}
 for task in tasks:
  tid=task['task_id']
  if task['status']!='VALID':
   for b in bindings:
    row=dict(task_id=tid,map_seed=task['map_seed'],model=b['label'],failure=task['status'],
       candidates=[dict(query=q,physics='UNKNOWN',goal=dict(passed=False,reasons=[task['status']]),complete='FAIL',geometry=dict(status='UNKNOWN'),numerical_status='NOT_EVALUATED_INVALID_INPUT') for q in range(6)],timing=None)
    rows.append(row);write(run/'results'/(tid+'_'+b['label']+'.json'),row)
   continue
  require(sha(task['input_path'])==task['input_sha256'],'SHARED_INPUT_CHANGED')
  with np.load(task['input_path'],allow_pickle=False) as z:depth=z['depth_input'].copy();state=z['state'].copy()
  key=task['map_dir']
  if key not in cached:
   # Only hold one memmap map at a time.
   cached.clear();cached[key]=load_world(Path(key),lib)[0]
  world=cached[key];require(world.hash==task['map_sha256'],'MAP_CHANGED')
  InferenceAdapter.inputs(depth[None],state[None],None)
  for b in bindings:
   try:
    raw,out,timing=predict(models[b['label']],depth,state)
    require(out.trajectory.solve_valid.all().item() and torch.isfinite(raw).all().item(),'NUMERICAL_OUTPUT')
   except (ValueError,RuntimeError) as error:
    row=dict(task_id=tid,map_seed=task['map_seed'],model=b['label'],failure='NUMERICAL_EXCEPTION',error=repr(error),traceback=traceback.format_exc(),input_sha256=task['input_sha256'],
       candidates=[dict(query=q,physics='FAIL',goal=dict(passed=False,reasons=['NUMERICAL_EXCEPTION']),complete='FAIL',geometry=dict(status='UNKNOWN'),numerical_status='FAIL') for q in range(6)],timing=None)
    rows.append(row);write(run/'results'/(tid+'_'+b['label']+'.json'),row)
    continue
   started=time.perf_counter()
   candidates,modes=assess(out,state,task,world,lib)
   timing.update(geometry_evaluation_s=time.perf_counter()-started,
                 input_and_ray_audit_s=task['input_and_ray_audit_s'],input_encode_s=task['input_encode_s'],ray_audit_s=task['ray_audit_s'],native_render_process_amortized_s=task['renderer_s_per_frame'])
   row=dict(task_id=tid,map_seed=task['map_seed'],model=b['label'],method_id=b['method_id'],
      checkpoint_sha256=b['checkpoint']['sha256'],input_sha256=task['input_sha256'],goals=task['goals'],
      goal_occupied=task['goal_occupied'],global_goal_occupied=task['global_goal_occupied'],goal_outside_map=task['local_goal_outside_map'],candidates=candidates,modes=modes,timing=timing)
   with (run/'results'/(tid+'_'+b['label']+'.npz')).open('xb') as f:
    np.savez(f,raw=raw.numpy(),coefficients=out.trajectory.coeffs.numpy(),durations=out.parameters.durations.numpy(),state=state,head_pva=out.parameters.head_pva.numpy(),tail_pva=out.parameters.tail_pva.numpy(),inner_pts=out.parameters.inner_pts.numpy())
   rows.append(row);write(run/'results'/(tid+'_'+b['label']+'.json'),row)
   print(tid,b['label'],'complete',sum(x['complete']=='PASS' for x in candidates),flush=True)
 require(all(state_sha(models[b['label']])==b['weights_sha256'] for b in bindings),'MODEL_CHANGED')
 write(run/'PER_SCENE.json',rows)
 flat=[]
 for row in rows:
  for q in row['candidates']:
   flat.append(dict(task_id=row['task_id'],map_seed=row['map_seed'],model=row['model'],query=q['query'],physics=q['physics'],goal_pass=q['goal']['passed'],complete=q['complete'],
       geometry=q['geometry']['status'],numerical_failure=q.get('numerical_status')=='FAIL',
       surface_clearance_lower_bound_m=q.get('surface_clearance_lower_bound_m'),dense_surface_clearance_upper_bound_m=q.get('dense_surface_clearance_upper_bound_m'),
       sampled64_missed_witness_violation=q.get('sampled64_missed_witness_violation'),input_sha256=row.get('input_sha256'),unmatched='N/A_NO_FROZEN_TEACHER'))
 with (run/'PER_CANDIDATE.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
 summary(run,rows)

def run(kind,run_id):
 bind=bind_check()
 require(run_id in ['smoke001','forest_paired001'] and run_id==('smoke001' if kind=='smoke' else 'forest_paired001'),'RUN_ID')
 if kind=='forest':
  prior=read(OUT/'smoke001/RESULT.json')
  require(all(x['numerical_anomaly_rate']==0 for x in read(OUT/'smoke001/SUMMARY.json')['per_map']),'SMOKE_NUMERICAL_FAILURE')
  priorrun=OUT/'smoke001';seal=read(priorrun/'RUN_SEAL.json')['files']
  require({str(p.relative_to(priorrun)):sha(p) for p in priorrun.rglob('*') if p.is_file() and p.name!='RUN_SEAL.json'}==seal,'SMOKE_CONTENT_SEAL')
  require(prior['status']=='COMPLETED_OFFLINE_ONLY' and prior['binding_sha256']==sha(PREP/'configs/EXECUTION_BINDING.json'),'SMOKE_REQUIRED')
 require(os.statvfs(OUT).f_bavail*os.statvfs(OUT).f_frsize>=4*2**30,'REQUIRE_4GIB_FREE_OUTPUT_STORAGE')
 dest=OUT/run_id;require(not dest.exists(),'EXCLUSIVE_RUN');dest.mkdir()
 for n in ['runtime','maps','configs','logs','inputs','observations','admission','ray_audits','results']:(dest/n).mkdir()
 gpu_info=OUT/('command_'+run_id)/'GPU_INFO.log';require(gpu_info.is_file(),'USER_GPU_INFO_LOG_REQUIRED')
 write(dest/'COMMAND.json',dict(argv=sys.argv,python=sys.executable,runtime=bind['runtime'],gpu_info_log=str(gpu_info),gpu_info_sha256=sha(gpu_info),env={k:os.environ.get(k) for k in ['PYTHONPATH','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','LD_LIBRARY_PATH']},binding=bind))
 before={p:sha(p) for p in bind['files']};write(dest/'PROTECTION_BEFORE.json',before)
 try:
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
  lib=legacy();tasks=capture(dest,kind,lib)
  require(read(dest/'TASK_MANIFEST.json.seal.json')['sha256']==sha(dest/'TASK_MANIFEST.json'),'TASK_MANIFEST_SEAL')
  if kind=='smoke':known_controls(dest,read(PREP/'configs/FROZEN_MODELS.json'))
  evaluate_tasks(dest,tasks,lib)
  require(all(sha(p)==h for p,h in before.items()),'EXTERNAL_INPUT_CHANGED')
  write(dest/'PROTECTION_AFTER.json',dict(passed=True,files=before))
  write(dest/'RESULT.json',dict(status='COMPLETED_OFFLINE_ONLY',binding_sha256=sha(PREP/'configs/EXECUTION_BINDING.json'),
                               no_training=True,no_control=True,closed_loop_qualified=False))
  write(dest/'RUN_SEAL.json',dict(files={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}))
 except BaseException as e:
  write(dest/'STOP.json',dict(status='STOP',error=repr(e),traceback=traceback.format_exc(),no_resume=True))
  raise

def check_run(kind,run_id):
 bind_check();run=OUT/run_id
 require(run_id==('smoke001' if kind=='smoke' else 'forest_paired001'),'CHECK_RUN_ID')
 seal=read(run/'RUN_SEAL.json')['files']
 require({str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file() and p.name!='RUN_SEAL.json'}==seal,'RUN_CONTENT_SEAL')
 require(read(run/'RESULT.json')['status']=='COMPLETED_OFFLINE_ONLY','INCOMPLETE_RUN')
 cmd=read(run/'COMMAND.json');require(sha(cmd['gpu_info_log'])==cmd['gpu_info_sha256'],'RECORDED_GPU_INFO')
 manifest=read(run/'TASK_MANIFEST.json')
 require(read(run/'TASK_MANIFEST.json.seal.json')['sha256']==sha(run/'TASK_MANIFEST.json'),'SHARED_TASK_SEAL')
 rows=read(run/'PER_SCENE.json');tasks=manifest['tasks'];summary_data=read(run/'SUMMARY.json')
 require(len(rows)==len(tasks)*5,'PAIRED_EXPOSURES')
 require(len(manifest['maps'])==(1 if kind=='smoke' else 21),'MAP_COUNT')
 for task in tasks:
  group=[x for x in rows if x['task_id']==task['task_id']]
  require({x['model'] for x in group}=={'C1','E0','E1','E2','E3'} and len(group)==5,'PAIRED_METHODS')
  if task['status']=='VALID':
   require(sha(task['input_path'])==task['input_sha256'],'INPUT_SHA')
   with np.load(task['input_path'],allow_pickle=False) as z:
    InferenceAdapter.inputs(z['depth_input'][None],z['state'][None],None)
    require(np.allclose(z['state'][6:9],[2.4,0,0],rtol=0,atol=2e-6),'LOCAL_GLOBAL_GOAL_SEPARATION')
   require(all(x.get('input_sha256')==task['input_sha256'] for x in group),'SAME_INPUT')
 for item in summary_data['per_map']:
  group=[x for x in rows if x['map_seed']==item['map_seed'] and x['model']==item['model']]
  cand=[q for x in group for q in x['candidates']]
  for key,value in [('physics_pass',np.mean([q['physics']=='PASS' for q in cand])),
                    ('complete_pass',np.mean([q['complete']=='PASS' for q in cand])),
                    ('goal_pass',np.mean([q['goal']['passed'] for q in cand])),
                    ('oracle_pass',np.mean([any(q['complete']=='PASS' for q in x['candidates']) for x in group]))]:
   require(abs(item[key]-value)<1e-12,'REBUILT_METRIC '+key)
 numerical=sum(q.get('numerical_status')=='FAIL' for row in rows for q in row['candidates'])
 require(numerical==0,'NUMERICAL_MODEL_FAILURES_STOP')
 return dict(passed=True,kind=kind,tasks=len(tasks),map_count=len(manifest['maps']),candidates=len(rows)*6,
             numerical_failures=numerical,closed_loop=False,READY_FOR_15M_CLOSED_LOOP=False,summary=str(run/'SUMMARY.json'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['smoke','forest'],required=True);p.add_argument('--run-id',required=True);p.add_argument('--check',action='store_true')
 a=p.parse_args()
 if a.check:print(json.dumps(check_run(a.kind,a.run_id),indent=2))
 else:run(a.kind,a.run_id)
