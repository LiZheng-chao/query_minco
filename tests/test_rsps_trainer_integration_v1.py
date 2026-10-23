"""CPU zero-update integration; real data require explicit qualified binding."""
import copy,json,os,pathlib,subprocess,unittest
import numpy as np
import torch
from query_minco.models import QueryMincoPlanner
from query_minco.inference.identity import state_sha
from query_minco.training.io import sha,artifact,digest
from query_minco.training.run_spec import source_files,components as reference_components
from query_minco.training.datasets.manifest import FrozenDataset
from query_minco.training.datasets.geometry_sidecar import GeometrySidecar
from query_minco.training.objective_rsps import recipe,cpu_dry_run
from query_minco.training.audit_rsps import audit_record

class Isolation(unittest.TestCase):
    def test_independent_method_dispatch(self):
        for arm in [1,2,3]:
            name='query_minco_rsps_e%d_cpu_dry_v1'%arm
            self.assertEqual(recipe(name)['arm'],arm)
            with self.assertRaises(ValueError):reference_components(name)
        with self.assertRaises(ValueError):recipe('query_minco_imitation_e1_trainer_v1')
        with self.assertRaises(ValueError):recipe('unregistered_rsps')

@unittest.skipUnless(os.environ.get('RSPS_SIDECAR'),'explicit qualified real input only')
class RealIntegration(unittest.TestCase):
    def test_three_seeds_three_arms_and_independent_audit(self):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        r=pathlib.Path(__file__).resolve().parents[1]
        output=pathlib.Path(os.environ['RSPS_AUDIT_OUTPUT'])
        gate=json.loads((output.parents[1]/'summary/GATE_C.json').read_text())
        self.assertEqual(gate['GATE_C'],'PASS')
        mpath=pathlib.Path(os.environ['RSPS_SIDECAR'])
        binding=dict(path=str(mpath),sha256=sha(mpath));m=json.loads(mpath.read_text())
        ds=FrozenDataset(m['dataset'],'b2c1_reference_v1');sidecar=GeometrySidecar(binding,ds)
        summaries=[];init_receipts=[]
        for seed in [42,43,44]:
            # Actual existing paths checked/hash bound, never presumed published weights.
            cp=pathlib.Path('/home/lizheng/reach_planner/query_minco/outputs/b2c1/train_v3')/('A%d_C1_orthogonal'%seed)/'checkpoint_step2880.pt'
            checkpoint=dict(path=str(cp),sha256=sha(cp));artifact(checkpoint)
            state=torch.load(str(cp),map_location='cpu',weights_only=True)['state_dict']
            ids,_=sidecar.orders[seed].batch(0);samples=ds.load(ids)
            results={}
            for arm in [1,2,3]:
                method='query_minco_rsps_e%d_cpu_dry_v1'%arm
                model=QueryMincoPlanner();model.load_state_dict(state,strict=True);model.train();before=state_sha(model)
                spec=dict(schema='rsps_cpu_zero_update_v1',training_method_id=method,seed=seed,batch_index=0,
                          checkpoint=checkpoint,weights_sha256=before,sidecar=binding,
                          code=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(r),text=True).strip(),
                                    files=source_files(),allow_uncommitted_sources=True),
                          recipe_sha256=digest(recipe(method)))
                if seed==42 and arm==1:
                    wrong=QueryMincoPlanner();wrong.train()
                    bad_spec=copy.deepcopy(spec);bad_spec['weights_sha256']=state_sha(wrong)
                    with self.assertRaises(ValueError):cpu_dry_run(wrong,samples,sidecar,method,bad_spec)
                z=cpu_dry_run(model,samples,sidecar,method,spec)
                record=z['record'];arrays=z['arrays']
                result=audit_record(record,arrays,samples,sidecar,seed,0,method,spec)
                self.assertTrue(result['passed']);self.assertEqual(state_sha(model),before)
                self.assertGreater(result['matched_physical_raw_grad_norm'],0)
                self.assertGreater(result['unmatched_physical_raw_grad_norm'],0)
                self.assertGreater(result['duration_gradient_max'],0)
                self.assertEqual(np.count_nonzero(arrays['raw_grad'][...,19:21]),0)
                label='A%d_E%d'%(seed,arm)
                with (output/(label+'_record.json')).open('x') as f:json.dump(record,f,indent=2,allow_nan=False)
                with (output/(label+'_arrays.npz')).open('xb') as f:np.savez(f,**arrays)
                with (output/(label+'_audit.json')).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
                summaries.append(dict(seed=seed,arm=arm,method=method,**result))
                results[arm]=(record,arrays,spec)
                print(label,'loss',record['loss'],'physical',record['physical'],'matched/unmatched',result['matched_physical_raw_grad_norm'],result['unmatched_physical_raw_grad_norm'])
            # Pairing: same network output, same matching/imitation, different physics allocation only.
            for arm in [2,3]:
                self.assertTrue(np.array_equal(results[1][1]['raw'],results[arm][1]['raw']))
                self.assertEqual(results[1][0]['imitation'],results[arm][0]['imitation'])
                self.assertEqual(results[1][0]['matches'],results[arm][0]['matches'])
            record,arrays,spec=results[3]
            negative=[]
            for label,mutate in [
                ('loss',lambda z:z.update(loss=z['loss']+.1)),
                ('method',lambda z:z['recipe'].update(training_method_id='query_minco_rsps_e1_cpu_dry_v1')),
                ('order',lambda z:z['ids'].reverse()),
                ('gradient_norm',lambda z:z['raw_gradient_norms'][0].__setitem__(0,999.)),
                ('spec',lambda z:z['run_spec'].update(seed=99)),
                ('source',lambda z:z.update(package_sources={}))]:
                bad=copy.deepcopy(record);mutate(bad)
                with self.assertRaises(ValueError):audit_record(bad,arrays,samples,sidecar,seed,0,'query_minco_rsps_e3_cpu_dry_v1',spec)
                negative.append(label)
            bad=copy.deepcopy(arrays);bad['positions'][0,0,0,0]=float('nan')
            with self.assertRaises(ValueError):audit_record(record,bad,samples,sidecar,seed,0,'query_minco_rsps_e3_cpu_dry_v1',spec)
            with (output/('A%d_negative_cases.json'%seed)).open('x') as f:json.dump(negative+['nonfinite_arrays'],f)
            init_receipts.append(dict(seed=seed,checkpoint=checkpoint,weights_sha256=before))
        with (output/'INITIALIZATION_RECEIPTS.json').open('x') as f:json.dump(init_receipts,f,indent=2)
        with (output/'INTEGRATION_SUMMARY.json').open('x') as f:json.dump(summaries,f,indent=2,allow_nan=False)
if __name__=='__main__':unittest.main(verbosity=2)
