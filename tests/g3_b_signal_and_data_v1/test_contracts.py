"""G3-B zero-update mathematical, data identity and split contracts."""
from pathlib import Path
import json,unittest
from unittest.mock import patch
import numpy as np
import torch
from query_minco.models import QueryMincoPlanner
from query_minco.evaluation.avoidance_families_v1 import load_family_batch,obstacle_mask
from query_minco.evaluation.mirror_holdout_v1 import sha
from query_minco.evaluation.validation import evaluate
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g3_b_signal_and_data_20261110_v1'
def read(p):return json.loads(Path(p).read_text())
class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.plan=read(O/'PREREGISTRATION.json')
        cls.grad=read(O/'gradient/LOSS_GRADIENT_RESULTS.json')
        cls.scan=read(O/'gradient/LATERAL_COST_HARD_SCAN.json')['records']
        cls.data=read(O/'dataset/DATASET_QUALIFICATION.json')
        cls.index=read(O/'dataset/DATASET_INDEX.json')
    def setUp(self):
        self.patches=[patch.object(QueryMincoPlanner,'__init__',side_effect=AssertionError('NO_MODEL')),patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('NO_OPTIMIZER')),patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('NO_UPDATES'))]
        for p in self.patches:p.start()
    def tearDown(self):
        for p in self.patches[::-1]:p.stop()
    def test_01_original_33_sources_unchanged(self):
        s=read(R/'research_outputs/g2a_k1_trainer_qualification_20261110_v1/K1_TRAINING_RUNSPEC_DRAFT.json')
        for name,h in s['source_sha256'].items():self.assertEqual(sha(name),h)
    def test_02_common_trajectory_and_state_control(self):
        rows=self.grad['records'];self.assertEqual(len(rows),8)
        for start in ['fixed_forward','common_step128_negative_bias']:
            xs=[x for x in rows if x['start']==start]
            self.assertEqual(len({x['common_raw_sha256'] for x in xs}),1)
            self.assertEqual(len({x['state_sha256'] for x in xs}),1)
        self.assertEqual({x['raw_leaf_updates'] for x in rows},{0})
    def test_03_l0_gradient_reconstructs_original_weights(self):
        weights=self.grad['weights']
        self.assertEqual(weights,dict(C_viol=1.,F=1.,G_task_train=1.,C_shape=.1,P=.025,S=.1))
        for x in self.grad['records']:
            grads=[np.asarray(x['gradients'][n]['raw0_15'])*w for n,w in weights.items()]
            np.testing.assert_allclose(np.sum(grads,axis=0),x['total_raw0_15'],atol=3e-5,rtol=3e-5)
            self.assertTrue(np.isfinite(x['total_raw0_15']).all());self.assertEqual(x['raw15_21_nonzero'],0)
    def test_04_finite_difference_diagnostics_preserved(self):
        for row in self.grad['records']:
            self.assertEqual([x['epsilon_m'] for x in row['finite_difference']],[.001,.005,.01])
            for x in row['finite_difference']:
                self.assertTrue(np.isfinite(x['finite_difference_total']))
                self.assertAlmostEqual((x['loss_plus']-x['loss_minus'])/(2*x['epsilon_m']),x['finite_difference_total'])
        for row in self.grad['records']:
            if row['start']=='fixed_forward':
                self.assertEqual(row['total_lateral_derivative'],0.)
                self.assertTrue(all(x['finite_difference_total']==0 for x in row['finite_difference']))
    def test_05_large_correct_excursion_identified_without_optimization(self):
        scenes={x['scene_id'] for x in self.scan}
        for scene in scenes:
            xs=[x for x in self.scan if x['scene_id']==scene and x['start']=='fixed_forward'];side=xs[0]['side']
            good=next(x for x in xs if x['physical_inner_y_shift_m']==.5*side)
            wrong=next(x for x in xs if x['physical_inner_y_shift_m']==-.5*side)
            self.assertTrue(good['hard']['passed']);self.assertFalse(wrong['hard']['passed'])
            self.assertLess(good['loss'],wrong['loss'])
    def test_06_no_outside_projection_in_base_cases(self):
        self.assertTrue(all(x['values']['outside_proxy_points']==0 for x in self.grad['records']))
        probes=read(O/'gradient/LOCAL_P0C_POINT_PROBES.json')['records']
        self.assertEqual(len(probes),20)
        for x in probes:self.assertTrue(np.isfinite(x['gradient_xyz']).all());self.assertFalse(x['outside_proxy'])
    def test_07_three_family_qualification_target(self):
        self.assertEqual(self.data['status'],'QUALIFIED');self.assertEqual((self.data['train_scenes'],self.data['dev_scenes']),(4,2))
        self.assertTrue(all(x['pair_qualified'] and x['exact_occupancy_reflection'] for x in self.data['families']))
        self.assertFalse(self.data['network_inference_on_new_data'])
    def test_08_family_splits_and_geometry_hashes(self):
        train={x['family_id'] for x in self.index if x['split']=='train'};dev={x['family_id'] for x in self.index if x['split']=='dev'}
        self.assertEqual(len(train),2);self.assertEqual(len(dev),1);self.assertFalse(train&dev)
        for k in ['scene_source_sha256','spatial_occupancy_sha256','canonical_occupied_geometry_sha256']:
            self.assertEqual(len({x['identity'][k] for x in self.data['scenes']}),6)
        hold=read(R/'research_outputs/g3_a_mirror_holdout_20261110_v1/scenes/QUALIFICATION_RESULT.json')
        self.assertFalse({x['identity']['canonical_occupied_geometry_sha256'] for x in self.data['scenes']}&{x['identity']['canonical_occupied_geometry_sha256'] for x in hold['scenes']})
    def test_09_strict_loader_shapes_mask_split_and_sha(self):
        for row in self.index:
            b=load_family_batch(row['manifest'],row['manifest_sha256'],row['split'])
            self.assertEqual(tuple(b.depth.shape),(1,2,128,160));self.assertEqual(tuple(b.state.shape),(1,12))
            self.assertTrue(torch.isfinite(b.depth).all());self.assertTrue(torch.all((b.depth[:,1]==0)|(b.depth[:,1]==1)))
            self.assertEqual(int(b.depth[:,:, :19].count_nonzero()),0);self.assertEqual(int(b.depth[:,:,109:].count_nonzero()),0)
            self.assertEqual(b.geometry[0].identity['split'],row['split']);self.assertFalse(b.manifest['teacher_loaded'])
            with self.assertRaises(ValueError):load_family_batch(row['manifest'],row['manifest_sha256'],'dev' if row['split']=='train' else 'train')
            with self.assertRaises(ValueError):load_family_batch(row['manifest'],'0'*64,row['split'])
    def test_10_obstacle_families_are_structurally_distinct(self):
        masks=[]
        for spec in self.plan['data']['families']:
            a,_=obstacle_mask(spec,1);b,_=obstacle_mask(spec,-1)
            self.assertTrue(np.array_equal(a[:,::-1,:],b));masks.append(a)
        self.assertTrue(all(not np.array_equal(a,b) for i,a in enumerate(masks) for b in masks[:i]))
        # DEV decoy gaps cannot fit the .30m diameter safety tube.
        spec=self.plan['data']['families'][2];a,_=obstacle_mask(spec,1)
        row=a[spec['column_x'][0],:,18];lo,hi=spec['wide_gap_y']
        free=[j for j in range(121) if not row[j] and j not in range(lo,hi)]
        self.assertTrue(all((j+1 not in free) for j in free));self.assertGreater((hi-lo)*.1,.3)
    def test_11_all_native_witnesses_and_visible_sections(self):
        for x in self.data['scenes']:
            self.assertTrue(x['witness_hard']['passed']);self.assertFalse(x['wrong_side_hard']['passed'])
            self.assertTrue(all(y['passed'] for y in x['witness_crossings']))
            self.assertTrue(all(y['gap_pixels']>=3 and y['wall_pixels']>=3 for y in x['observability']))
            self.assertLessEqual(x['depth_replay_max_error_m'],2e-5);self.assertEqual(x['renderer_exit'],0)
    def test_12_witness_pva_c4_and_limits(self):
        state=np.array(self.plan['data']['state_B0'])
        for x in self.data['scenes']:
            c=np.array(x['witness_coefficients']);d=np.array(x['durations'])
            for order in range(5):
                for i in [0,1]:np.testing.assert_allclose(evaluate(c[i],d[i],order),evaluate(c[i+1],0,order),atol=1e-7)
            np.testing.assert_allclose(evaluate(c[0],0,1),state[:3],atol=1e-8)
            np.testing.assert_allclose(evaluate(c[0],0,2),state[3:6],atol=1e-8)
            np.testing.assert_allclose(evaluate(c[-1],d[-1]),state[6:9],atol=1e-8)
            self.assertTrue(all(v<=m for v,m in zip(x['witness_hard']['dynamic_polynomial_maxima'],[3,8,100])))
    def test_13_true_native_process_logs_and_new_qualification_identity(self):
        for row in self.index:
            m=read(row['manifest']);self.assertEqual(m['schema'],'qualified_avoidance_family_scene_v1')
            p=read(Path(m['folder'])/'renderer_process/PROCESS_RESULT.json')
            self.assertEqual(p['exit_code'],0);self.assertEqual(p['argv'][1],'depth')
            self.assertEqual(p['inputs']['RUNSPEC']['sha256'],sha(O/'PREREGISTRATION.json'))
if __name__=='__main__':unittest.main(verbosity=2)
