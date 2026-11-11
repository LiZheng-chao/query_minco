"""G3-C contracts. All test physics is TRAIN-only; no optimizer/update is permitted."""
from pathlib import Path
import json,unittest,tempfile,copy
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from query_minco.audit.nonlocal_signal_v1 import physical_lateral_shift,lateral_tangent,SymmetricNonlocalCost
from query_minco.audit.k1_signal_diagnostics_v1 import crossing_functional,seal_train_decision,dataset
from query_minco.evaluation.avoidance_families_v1 import load_family_batch
from query_minco.training.physics_only_v1_2_gate1.k1_interface import make_objective
from query_minco.training.physics_only_v1_2_gate1.binding import BoundBatch
from query_minco.evaluation.mirror_holdout_v1 import sha
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g3_c_nonlocal_signal_20261110_v1'
def read(p):return json.loads(Path(p).read_text())
class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.plan=read(O/'PREREGISTRATION.json')
        rows=dataset(cls.plan);row=next(x for x in rows if x['split']=='train')
        cls.batch=load_family_batch(row['manifest'],row['manifest_sha256'],'train')
        cls.results=read(O/'train_diagnostics/TRAIN_SIGNALS.json')
        cls.raw=torch.tensor(cls.results['records'][0]['raw21'],dtype=torch.float32)
    def setUp(self):
        self.forbid=patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('NO_UPDATES'));self.forbid.start()
    def tearDown(self):self.forbid.stop()
    def test_01_physical_units_and_abi(self):
        raw=self.raw.clone()
        for delta in [.1,-.1,.7,-.7]:
            shifted=physical_lateral_shift(raw,delta)
            torch.testing.assert_close(4*torch.tanh(shifted[..., [1,4]]),4*torch.tanh(raw[..., [1,4]])+delta,atol=2e-6,rtol=0)
            for i in [0,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:self.assertTrue(torch.equal(shifted[...,i],raw[...,i]))
    def test_02_shift_graph_chain_rule(self):
        raw=self.raw.clone().requires_grad_(True);shifted=physical_lateral_shift(raw,.5)
        physical=4*torch.tanh(shifted[..., [1,4]])
        grad=torch.autograd.grad(physical.sum(),raw)[0]
        torch.testing.assert_close(grad[..., [1,4]],4*(1-torch.tanh(raw[..., [1,4]]).square()),atol=3e-6,rtol=3e-6)
        self.assertIsNone(raw.grad)
    def test_03_inverse_support_fails_closed(self):
        raw=self.raw.clone();raw[...,1]=torch.atanh(torch.tensor(3.9/4))
        with self.assertRaises(ValueError):physical_lateral_shift(raw,.7)
        with self.assertRaises(ValueError):physical_lateral_shift(self.raw,float('nan'))
        with self.assertRaises(ValueError):physical_lateral_shift(torch.zeros(1,6,21),.1)
        with self.assertRaises(ValueError):SymmetricNonlocalCost([.1,.1])
    def test_04_mean_is_exact_original_cost_mean(self):
        raw=self.raw.clone().requires_grad_(True);obj=make_objective();cost=SymmetricNonlocalCost(objective=obj);b=cost(raw,self.batch)
        direct=torch.stack([obj(physical_lateral_shift(raw,s*d),self.batch).loss for d in [.1,.3,.5,.7] for s in [-1,1]]).mean()
        torch.testing.assert_close(b.loss,direct,atol=1e-7,rtol=1e-7)
        gb=torch.autograd.grad(b.loss,raw,retain_graph=True)[0];gd=torch.autograd.grad(direct,raw)[0]
        torch.testing.assert_close(gb,gd,atol=2e-5,rtol=2e-5);self.assertEqual(int(gb[...,15:].count_nonzero()),0)
    def test_05_side_metadata_not_a_cost_input(self):
        b=self.batch;manifest=copy.deepcopy(b.manifest);manifest['side']=-manifest['side']
        other=BoundBatch(b.depth,b.state,b.geometry,manifest,b.verify);other._depth=b._depth;other._state=b._state
        cost=SymmetricNonlocalCost()
        with torch.no_grad():a=cost(self.raw,b).loss;c=cost(self.raw,other).loss
        self.assertTrue(torch.equal(a,c))
    def test_06_implicit_crossing_time_derivative(self):
        velocity=torch.tensor(2.,requires_grad=True);coeff=torch.zeros(1,1,3,3,6)
        coeff[0,0,0,0,1]=velocity;coeff[0,0,0,1,2]=1.
        result=SimpleNamespace(trajectory=SimpleNamespace(coeffs=coeff,durations=torch.full((1,1,3),2/3)))
        report={'required_crossings':[{'planes':[{}, {'forward_hits':[{'piece':0,'time_s':.5}]}]}]}
        q={'sections':[{'planes_B0':[.9,1.,1.1]}]}
        value=crossing_functional(result,report,q);g=torch.autograd.grad(value,velocity)[0]
        self.assertAlmostEqual(float(value),.25);self.assertAlmostEqual(float(g),-.25)
    def test_07_loss_gradient_records_finite_full_path(self):
        self.assertEqual(len(self.results['records']),8)
        for x in self.results['records']:
            self.assertEqual(x['raw15_21_grad_nonzero'],0);self.assertTrue(x['parameters_grad_buffers_unchanged'])
            for name in ['A','B']:
                self.assertTrue(np.isfinite(x[name]['raw15_gradient']).all())
                self.assertTrue(all(v['all_finite'] and v['norm']>0 for v in x[name]['module_gradients'].values()))
            self.assertEqual(len(x['candidates']),8)
            self.assertAlmostEqual(x['B']['loss'],np.mean([v['loss'] for v in x['candidates']]),places=6)
    def test_08_finite_difference_modes_and_candidate_failures_retained(self):
        for x in self.results['records']:
            self.assertEqual([v['epsilon_m'] for v in x['finite_differences']],[.001,.005,.01])
            for v in x['finite_differences']:self.assertTrue(np.isfinite(v['A']) and np.isfinite(v['B']))
            self.assertEqual([v['delta_m'] for v in x['scales']],[.1,.3,.5,.7])
            self.assertFalse(x['nominal']['hard_task_and_safe_crossing'])
            self.assertTrue(x['any_perturbed_safe'])
            self.assertTrue(x['nominal_unsafe_despite_safe_perturbation'])
    def test_09_parameter_projection_matches_saved_jacobian(self):
        data=np.load(O/'train_diagnostics/TRAIN_SIGNAL_ARRAYS.npz')
        for x in self.results['records']:
            key='step%s_%s'%(x['step'],x['scene_id'])
            q=data[key+'_crossing_y_param_jac'].astype(float)
            for name in ['A','B']:
                g=data[key+'_'+name+'_param_grad'].astype(float)
                expected=-float(np.dot(q,g));actual=x[name]['parameter_projection']['predicted_dy_per_infinitesimal_sgd_step']
                self.assertAlmostEqual(expected,actual,delta=1e-6*max(1,abs(expected)))
            self.assertEqual(len(q),469821)
    def test_10_shared_gradient_is_mean_of_one_model_samples(self):
        data=np.load(O/'train_diagnostics/TRAIN_SIGNAL_ARRAYS.npz')
        for step in [0,128]:
            for name in ['A','B']:
                vals=[data['step%s_%s_%s_param_grad'%(step,x['scene_id'],name)] for x in self.results['records'] if x['step']==step]
                np.testing.assert_allclose(np.mean(vals,axis=0),data['step%s_shared_%s_grad'%(step,name)],atol=1e-6,rtol=1e-5)
    def test_11_seal_uses_no_dev_or_data_loader(self):
        with tempfile.TemporaryDirectory() as tmp,patch('query_minco.audit.k1_signal_diagnostics_v1.dataset',side_effect=AssertionError('NO_DATA_READ_DURING_DECISION')):
            decision=seal_train_decision(O/'PREREGISTRATION.json',O/'train_diagnostics',Path(tmp)/'decision.json')
            expected=read(O/'TRAIN_ONLY_DECISION.json')
            self.assertEqual(json.loads(json.dumps(decision)),expected)
            self.assertFalse(decision['dev_used_for_decision']);self.assertFalse(decision['delta_tuned'])
    def test_12_dev_is_forward_only_after_seal(self):
        d=read(O/'dev_forward/DEV_FORWARD.json');self.assertEqual(len(d['records']),4);self.assertEqual(d['dev_backward_calls'],0)
        for row in d['records']:
            self.assertEqual(row['Raw15_gradient'],'NOT_RUN_DEV_FORWARD_ONLY')
            self.assertEqual(row['module_gradients'],'NOT_RUN_DEV_FORWARD_ONLY')
            self.assertEqual(row['nonlocal_cost_or_offsets'],'NOT_RUN_TRAIN_ONLY')
        p=read(O/'logs/dev_forward_process/PROCESS_START.json')
        self.assertLess((O/'TRAIN_ONLY_DECISION.json').stat().st_mtime,p['started_unix'])
        self.assertEqual(d['train_decision_sha256'],sha(O/'TRAIN_ONLY_DECISION.json'))
    def test_13_source_data_and_checkpoint_hashes(self):
        freeze=read(O/'EXPERIMENT_SOURCE_FREEZE.json')
        for n,h in freeze.items():self.assertEqual(sha(n),h)
        for c in self.results['checkpoint_identity']:self.assertEqual(sha(c['path']),c['file_sha256'])
        self.assertEqual(len(dataset(self.plan)),6)
if __name__=='__main__':unittest.main(verbosity=2)
