import sys,json,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,torch
from unittest.mock import patch
from query_minco.training.k1_physics_trainer_v1 import trainer as T
from query_minco.training.k1_physics_trainer_v1.optimizer import DeadRowGuard
from query_minco.training.physics_only_v1_2_gate1.k1_interface import make_planner
r=Path.cwd();o=r/'research_outputs/g2b_k1_visual_audit_20261110_v1';run=r/'research_outputs/g2b_k1_pilot_candidate_20261110_v1/A_reasonable_seed20261104'
e=json.loads((o/'FORMAL_128STEP_EVIDENCE.json').read_text())
logs=[json.loads(s) for s in (run/'STEPS.jsonl').read_text().splitlines()]
assert all(x['dead_row_isolation']['dead_sha_before']==x['dead_row_isolation']['dead_sha_after']==e['dead_sha256'] for x in logs)
assert all(all(v['nonzero']==0 for v in x['dead_row_isolation']['state_after'].values()) for x in logs)
reports=[]
with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('NO_UPDATES_ALLOWED')):
 for step in [0,8,32,64,128]:
  path=run/('step%03d.pt'%step);b=torch.load(path,map_location='cpu',weights_only=True)
  torch.set_rng_state(b['rng']['torch']);random.setstate(b['rng']['python']);n=b['rng']['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
  expected=(torch.rand(5),[random.random() for _ in range(5)],np.random.rand(5))
  m=make_planner();opt=T.optimizer_for(m,b['metadata']['optimizer']);guard=DeadRowGuard(m)
  cursor,dirty=T.load_checkpoint(path,T.sha(path),m,opt,guard,b['metadata'])
  actual=(torch.rand(5),[random.random() for _ in range(5)],np.random.rand(5))
  assert torch.equal(expected[0],actual[0]) and expected[1]==actual[1] and np.array_equal(expected[2],actual[2])
  assert cursor==step and T.state_sha(m)==json.loads(Path(str(path)+'.json').read_text())['model_sha256']
  restored=opt.state_dict()
  for k,vs in b['optimizer']['state'].items():
   for name,v in vs.items():
    assert torch.equal(v,restored['state'][k][name]) if isinstance(v,torch.Tensor) else v==restored['state'][k][name]
  reports.append(dict(step=step,model_exact=True,optimizer_state_exact=True,cursor_exact=True,next_rng_draws_torch_python_numpy_exact=True,dead_reference_exact=True))
with (o/'CHECKPOINT_ZERO_STEP_RESTORE.json').open('x') as f:json.dump(dict(status='PASS',all_128_dead_logs_exact=True,restores=reports,optimizer_step_calls=0,limits='Fresh AdamW instantiated only for state loading; no future optimization step, no automatic training resume.'),f,indent=2)
print('5 CHECKPOINT_RESTORE_PASS; 128 DEAD_LOGS_PASS; NEW_NETWORK_UPDATES=0')
