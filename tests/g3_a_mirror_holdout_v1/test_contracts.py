"""G3 zero-update tests: candidate-axis parity, crossing and real subprocess provenance."""
from pathlib import Path
import sys,json,copy,tempfile,unittest,subprocess,multiprocessing,os,time
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from query_minco.training.physics_only_v1_2_gate1.hard_candidates_v3 import validate_result_candidates
from query_minco.training.physics_only_v1_2_gate1.hard_validator import validate_result,validate_polynomial
from query_minco.training.physics_only_v1_2_gate1.roi_safe_training import RoiSafeTrainingObjective
from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
from query_minco.training.physics_only_v1.decoder import FixedTimeDecoder
from query_minco.trajectory.minco_adapter import MincoAdapter
from query_minco.models import QueryMincoPlanner
from query_minco.models.config import QueryMincoConfig
from query_minco.evaluation.mirror_holdout_v1 import load_scene,sha
from query_minco.evaluation.crossing_v1 import crossing_metric
from query_minco.evaluation.validation import evaluate,numerical_reference
from query_minco.audit.run_provenance_v1 import run_logged,AtomicAttemptLedger
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g3_a_mirror_holdout_20261110_v1'
def reserve_worker(path,queue):
    try:
        ledger=AtomicAttemptLedger(path,3);token=ledger.reserve();ledger.complete(token);queue.put('ok')
    except PermissionError:queue.put('rejected')
def claim_worker(path,queue):
    try:
        result=run_logged([sys.executable,'-B','-c','print("CHILD_ONCE")'],path,cwd=R);queue.put(result['exit_code'])
    except FileExistsError:queue.put('occupied')
class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        q=json.loads((O/'scenes/QUALIFICATION_RESULT.json').read_text());assert q['status']=='QUALIFIED'
        cls.scenes=q
        s=q['scenes'][0];cls.depth,cls.state,cls.g,cls.m,cls.cert=load_scene(s['manifest'],s['manifest_sha256'])
        cls.batch=SimpleNamespace(geometry=(cls.g,),state=cls.state)
        cls.outputs={}
        with torch.no_grad():
            for k in [1,6]:
                torch.manual_seed(20261104);model=QueryMincoPlanner(QueryMincoConfig(num_queries=k)).eval();raw=model(cls.depth,cls.state)
                params=FixedTimeDecoder(PhysicsV12Spec(phase='A',aggregation='foundation'),num_candidates=k)(raw,cls.state)
                tr=MincoAdapter(num_candidates=k)(params.head_pva,params.tail_pva,params.inner_pts,params.durations)
                cls.outputs[k]=SimpleNamespace(parameters=params,trajectory=tr)
    def setUp(self):
        self.guard=patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('NEW_NETWORK_UPDATES_FORBIDDEN'));self.guard.start()
    def tearDown(self):self.guard.stop()
    def modified(self,k,field,value):
        old=self.outputs[k]
        p=SimpleNamespace(head_pva=old.parameters.head_pva,tail_pva=old.parameters.tail_pva,inner_pts=old.parameters.inner_pts)
        tr=SimpleNamespace(coeffs=old.trajectory.coeffs,durations=old.trajectory.durations)
        setattr(tr if field in ['coeffs','durations'] else p,field,value)
        return SimpleNamespace(parameters=p,trajectory=tr)
    def test_01_k1_one_row_and_legacy_failure(self):
        r=validate_result_candidates(self.outputs[1],self.batch);self.assertEqual(len(r),1)
        with self.assertRaises(IndexError):validate_result(self.outputs[1],self.batch)
        self.assertIn('independent_minco_max_abs_pva',r[0])
    def test_02_k6_full_result_exact(self):
        self.assertEqual(validate_result(self.outputs[6],self.batch),validate_result_candidates(self.outputs[6],self.batch))
    def test_03_positive_hard_direct_exact(self):
        for k in [1,6]:
            out=self.outputs[k];rows=validate_result_candidates(out,self.batch)
            for i,row in enumerate(rows):
                direct=validate_polynomial(out.trajectory.coeffs[0,i].numpy(),out.trajectory.durations[0,i].numpy(),self.g,self.state[0,6:9].numpy())
                for name,value in direct.items():self.assertEqual(row[name],value)
                self.assertTrue(all(v<t for v,t in zip(row['independent_minco_max_abs_pva'],[.002,.01,.05])))
    def test_04_illegal_times_all_candidates(self):
        for k in [1,6]:
            for v in [0.,-1.,float('nan'),float('inf'),float('-inf')]:
                with self.subTest(k=k,v=v):
                    out=self.modified(k,'durations',torch.full((1,k,3),v))
                    rows=validate_result_candidates(out,self.batch)
                    self.assertEqual(len(rows),k);self.assertTrue(all(not x['passed'] for x in rows))
                    self.assertTrue(all(x['reasons'][0].startswith(('INVALID_DURATION','NONPOSITIVE')) for x in rows))
    def test_05_wrong_duration_shapes(self):
        for k in [1,6]:
            for shape in [(1,k,2),(1,k,4),(1,k,1,3),(k,3)]:
                with self.subTest(k=k,shape=shape):
                    with self.assertRaises(ValueError):validate_result_candidates(self.modified(k,'durations',torch.ones(shape)),self.batch)
    def test_06_wrong_coeff_shapes_and_dtype(self):
        for c in [torch.zeros(1,2,3,3,6),torch.zeros(2,1,3,3,6),torch.zeros(1,1,3,3,5),torch.zeros(1,1,3,3,6,dtype=torch.int64)]:
            with self.assertRaises(ValueError):validate_result_candidates(self.modified(1,'coeffs',c),self.batch)
    def test_07_wrong_and_nonfinite_pva(self):
        for field in ['head_pva','tail_pva','inner_pts']:
            with self.assertRaises(ValueError):validate_result_candidates(self.modified(1,field,torch.zeros(1)),self.batch)
            x=getattr(self.outputs[1].parameters,field).clone();x.flatten()[0]=float('nan')
            with self.assertRaises(ValueError):validate_result_candidates(self.modified(1,field,x),self.batch)
    def test_08_nonfinite_coeff_rejected(self):
        c=self.outputs[1].trajectory.coeffs.clone();c.flatten()[0]=float('nan')
        row=validate_result_candidates(self.modified(1,'coeffs',c),self.batch)[0]
        self.assertEqual(row['reasons'],['INVALID_POLYNOMIAL'])
    def test_09_independent_minco_violation(self):
        c=self.outputs[1].trajectory.coeffs.clone();c[0,0,0,0,0]+=.1
        row=validate_result_candidates(self.modified(1,'coeffs',c),self.batch)[0]
        self.assertFalse(row['passed']);self.assertIn('INDEPENDENT_MINCO',row['reasons'])
    def test_10_unknown_not_promoted(self):
        c=torch.zeros(1,1,3,3,6);c[0,0,:,0,0]=30
        row=validate_result_candidates(self.modified(1,'coeffs',c),self.batch)[0]
        self.assertFalse(row['passed'])
        self.assertTrue(any(x['reason']=='UNKNOWN_OR_ROI_MARGIN' for x in row['certificates']))
    def test_11_qualified_witness_and_crossing_all_four(self):
        for s in self.scenes['scenes']:
            self.assertTrue(s['witness_hard']['passed'] and s['witness_crossing']['passed'])
            self.assertFalse(s['wrong_sign_witness_hard']['passed'])
            self.assertGreaterEqual(s['observable_gap_pixels_center_row'],3)
            self.assertGreaterEqual(s['observable_wall_pixels_center_row'],3)
        self.assertTrue(all(x['exact_counts_reflection'] for x in self.scenes['pairs']))
    def test_12_pre_gate_stop_fails_crossing(self):
        c=np.zeros((3,3,6));c[:,0,0]=.2
        x=crossing_metric(c,np.full(3,2/3),self.cert['planes_B0'],self.cert['gap_B0'],1)
        self.assertFalse(x['passed']);self.assertFalse(x['crossed_all_planes'])
    def test_13_wrong_sign_crossing_fails(self):
        c=np.array(self.cert['witness_coefficients']);c[:,1]*=-1
        x=crossing_metric(c,np.full(3,2/3),self.cert['planes_B0'],self.cert['gap_B0'],1)
        self.assertTrue(x['crossed_all_planes']);self.assertFalse(x['passed']);self.assertFalse(x['correct_direction'])
    def test_14_witness_minco_pva_c4(self):
        for s in self.scenes['scenes']:
            c=np.array(s['witness_coefficients']);d=np.array(s['durations'])
            for order in range(5):
                for i in [0,1]:np.testing.assert_allclose(evaluate(c[i],d[i],order),evaluate(c[i+1],0,order),atol=1e-7)
    def test_15_provenance_true_exit_stdout_stderr_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=root/'input.json';spec.write_text('{"test_only":true}')
            env=dict(os.environ,FAKE_SECRET='redact_this_only_for_test')
            p=run_logged([sys.executable,'-B','-c','import sys; print("OUT"); print("ERR",file=sys.stderr); sys.exit(7)'],root/'run',cwd=R,env=env,runspec=spec,approval=spec)
            self.assertEqual(p['exit_code'],7)
            self.assertEqual((root/'run/stdout.log').read_text(),'OUT\n')
            self.assertEqual((root/'run/stderr.log').read_text(),'ERR\n')
            self.assertEqual(p['inputs']['RUNSPEC']['sha256'],sha(spec))
            self.assertEqual((root/'run/APPROVAL.json').read_bytes(),spec.read_bytes())
            self.assertTrue(p['environment']['FAKE_SECRET']['redacted'])
            with self.assertRaises(FileExistsError):run_logged([sys.executable,'-c','raise Exception("MUST_NOT_START")'],root/'run',cwd=R)
    def test_16_real_process_tree_and_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            code='import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c","import time; a=bytearray(1000000); time.sleep(.3)"]); p.wait()'
            p=run_logged([sys.executable,'-B','-c',code],Path(tmp)/'tree',cwd=R)
            self.assertEqual(p['exit_code'],0);self.assertGreaterEqual(p['max_sampled_processes'],2);self.assertGreater(p['sampled_process_tree_peak_rss_bytes'],0)
            p=run_logged([sys.executable,'-B','-c','import time;time.sleep(2)'],Path(tmp)/'timeout',cwd=R,timeout_s=.05)
            self.assertTrue(p['timed_out']);self.assertLess(p['exit_code'],0)
    def test_17_atomic_ledger_concurrency_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=str(Path(tmp)/'ledger.json');queue=multiprocessing.Queue()
            ps=[multiprocessing.Process(target=reserve_worker,args=(path,queue)) for _ in range(8)]
            for p in ps:p.start()
            for p in ps:p.join(10);self.assertEqual(p.exitcode,0)
            xs=[queue.get(timeout=1) for _ in ps]
            self.assertEqual(xs.count('ok'),3);self.assertEqual(xs.count('rejected'),5)
            state=json.loads(Path(path).read_text());self.assertEqual((state['attempted'],state['completed']),(3,3));self.assertEqual(state['pending'],[])
            with self.assertRaises(ValueError):AtomicAttemptLedger(path,3).complete(state['done'][0])
            with self.assertRaises(ValueError):AtomicAttemptLedger(path,4).reserve()
    def test_18_abandoned_reservation_not_refunded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'ledger.json';l=AtomicAttemptLedger(path,1);token=l.reserve()
            with self.assertRaises(PermissionError):AtomicAttemptLedger(path,1).reserve()
            state=json.loads(path.read_text());self.assertEqual(state['completed'],0);self.assertEqual(state['pending'],[token])
    def test_19_concurrent_run_claim_launch_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=str(Path(tmp)/'exclusive');queue=multiprocessing.Queue()
            ps=[multiprocessing.Process(target=claim_worker,args=(path,queue)) for _ in range(4)]
            for p in ps:p.start()
            for p in ps:p.join(10);self.assertEqual(p.exitcode,0)
            xs=[queue.get(timeout=1) for _ in ps]
            self.assertEqual(xs.count(0),1);self.assertEqual(xs.count('occupied'),3)
    def test_20_frozen_model_source_protection(self):
        e=json.loads((O/'ENTRY_PROTECTION.json').read_text())
        for n in ['query_minco/training/physics_only_v1_2_gate1/hard_validator.py','query_minco/training/k1_physics_trainer_v1/trainer.py']:
            path=R/n;self.assertEqual(sha(path),e['source_before'][str(path)])
if __name__=='__main__':unittest.main(verbosity=2)
