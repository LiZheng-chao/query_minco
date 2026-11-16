"""Read frozen G1 current-K6 numerical baseline;no registry bypass/no updates."""
from pathlib import Path
import sys,json,numpy as np,torch
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.models import QueryMincoPlanner
from query_minco.training.physics_only_v1_2.set_objective_v12 import PhysicsV12Objective
from query_minco.training.physics_only_v1.costs import SyntheticGeometry
from query_minco.trajectory.postprocessor import TrajectoryPostprocessor
from query_minco.training.physics_only_v1_2_gate1.roi_safe_training import RoiSafeTrainingObjective
from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
from query_minco.training.physics_only_v1_2_gate1.e1_dataset_binding import load_e1_bound
from query_minco.training.k1_physics_trainer_v1.trainer import state_sha
R=Path.cwd();O=R/'research_outputs/g4c_startup_and_dynamics_20261111_v1'
torch.set_num_threads(1)
old=R/'research_outputs/g1_k1_current_baseline_20261110_v2'
a=np.load(old/'K6_BEFORE.npz');rows=json.loads((old/'BINDING_INDEX.json').read_text())
row=rows[0];batch=load_e1_bound(row['manifest'],row['manifest_sha256'])
assert np.array_equal(batch.state.numpy(),a['state_input'])
fixture_depth=torch.from_numpy(a['depth'].copy())  # Frozen synthetic network probe,not a dataset replay
model=QueryMincoPlanner().eval();model.load_state_dict({k[7:]:torch.from_numpy(a[k].copy()) for k in a.files if k.startswith('state::')},strict=True);before=state_sha(model)
with patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('NO_OPTIMIZER')):
 raw=model(fixture_depth,batch.state);raw.retain_grad();geo=[SyntheticGeometry(torch.tensor([[[.9,-.3,-.3],[1.2,.3,.3]]]),torch.tensor([[0.,-4.,-2.],[10.,4.,2.]]),False)]
 out=PhysicsV12Objective(PhysicsV12Spec(phase='A',aggregation='foundation'))(raw,batch.state,geo,goal_enabled=torch.ones(1,dtype=torch.bool),task_types=('normal',));out.loss.backward()
 actual=dict(raw=raw,raw_grad=raw.grad,coeffs=out.trajectory.coeffs,durations=out.trajectory.durations,loss=out.loss)
 actual.update({'grad::'+n:p.grad for n,p in model.named_parameters()})
 actual.update({'metric::'+n:v for n,v in out.metrics.items() if isinstance(v,torch.Tensor)})
 cp=R.parent/'reach_planner/query_minco/outputs/b2c1/train_v3/A42_C1_orthogonal/checkpoint_step2880.pt'
 c1=QueryMincoPlanner().eval();c1.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['state_dict'],strict=True)
 cr=c1(fixture_depth,batch.state);co=TrajectoryPostprocessor()(cr,batch.state,num_samples=65)
 actual.update({'c1::raw':cr,'c1::coeffs':co.trajectory.coeffs,'c1::durations':co.trajectory.durations,'c1::samples':co.samples.p})
 e0=QueryMincoPlanner().eval();e0.load_state_dict(torch.load(R/'research_outputs/physics_only_v1_2_first_short_train_v1/runs/phase_a_seed20261104_train001/checkpoints/step_032.pt',map_location='cpu',weights_only=True)['model'],strict=True)
 actual['e0::raw']=e0(fixture_depth,batch.state)
 checks={k:dict(exact=bool(np.array_equal(v.detach().numpy(),a[k])),max_abs=float(np.max(np.abs(v.detach().numpy().astype(float)-a[k].astype(float))))) for k,v in actual.items() if k in a.files}
 assert state_sha(model)==before
 result=dict(status='PASS' if all(v['exact'] for v in checks.values()) else 'FAIL',checks=checks,arrays_checked=len(checks),model_sha_before=before,model_sha_after=state_sha(model),strict_load=True,optimizer_steps=0,legacy_C1_registry='BLOCKED_UNCHANGED')
 with (O/'K6_DIRECT_REGRESSION_CORRECT_FIXTURE.json').open('x') as f:json.dump(result,f,indent=2)
 print('K6_DIRECT_NUMERIC',result['status'],len(checks));assert result['status']=='PASS'
