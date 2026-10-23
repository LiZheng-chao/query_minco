"""Installed-wheel import/plan without any legacy project code."""
import sys,os,json,subprocess,hashlib,importlib,pkgutil
from pathlib import Path
N=Path(__file__).resolve().parents[1]
run=Path(sys.argv[1]);site=run/'site';sys.path.insert(0,str(N/'tools'))
from validation_guard import install,sha
cp=Path('/tmp/reference_checkpoint.pt')
install(run,{str(cp)},sha('/usr/bin/uname'),sha('/home/lizheng/anaconda3/envs/yopo/lib/python3.8/platform.py'))
sys.path.insert(0,str(site))
import numpy as np,torch,query_minco
from query_minco.inference import load_planner
assert site in Path(query_minco.__file__).parents
assert not Path('/home/lizheng/reach_planner/query_minco/planner.py').exists()
modules=[]
for m in pkgutil.walk_packages(query_minco.__path__,query_minco.__name__+'.'):
 importlib.import_module(m.name);modules.append(m.name)
f=N/'research_outputs/migration_v1/fixtures/near.npz'
with np.load(f,allow_pickle=False) as z:d=z['depth_input'].copy();s=z['state'].copy();t=z['times'].copy()
p=load_planner('query_minco_b2c1_c1_e1_v1',cp)
got=p.plan(d,s,t)
reference=N/'research_outputs/migration_v1/runs/inference_new001/near.npz'
with np.load(reference,allow_pickle=False) as z:
 assert set(got)==set(z.files)
 for k in got:assert got[k].tobytes()==z[k].tobytes(),k
for name,module in list(sys.modules.items()):
 file=getattr(module,'__file__',None)
 if file:assert '/home/lizheng/reach_planner/' not in file
sys.path.insert(0,str(N/'tests'))
import unittest,test_reference_contracts
suite=unittest.defaultTestLoader.loadTestsFromModule(test_reference_contracts)
tests=unittest.TextTestRunner(verbosity=2).run(suite)
assert tests.wasSuccessful() and tests.testsRun==8
result=dict(unit_tests=tests.testsRun,passed=True,installed_package=str(query_minco.__file__),module_imports=modules,
 model_forward_calls=1,isolated_old_tree=True,outputs_bitwise_equal_to_verified_package=True,
 checkpoint_sha256=sha(cp),torch=torch.__version__,numpy=np.__version__,device='cpu',optimizer_updates=0)
with (run/'RESULT.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(dict(passed=True,imported_modules=len(modules),forward_calls=1)))
