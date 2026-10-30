"""Fixed component selection, no-fake exposure and identity tests."""
import copy,unittest
from unittest.mock import patch
import torch
from query_minco.training.io import digest
from query_minco.training.pg3_v2_no_corridor.ablation_v21 import *
from query_minco.training.pg3_v2_no_corridor.executor_v21 import SCHEDULE
from experiments.pg3_v2_no_corridor.attribution_v21 import taxonomy
class AblationTests(unittest.TestCase):
    def test_fixed_sequential_weights(self):
        for a,keys in ACTIVE.items():
            self.assertEqual({k for k,v in weights_for(a).items() if v},set(keys))
            for k,v in weights_for(a).items():self.assertEqual(v,FROZEN[k] if k in keys else 0.)
        self.assertEqual(weights_for('K'),FROZEN)
    def test_unknown_arm_rejected(self):
        with self.assertRaises(ValueError):weights_for('D')
    def test_stop_has_no_fake_exposure(self):
        a=NoDirectExposure();r=a.commit_after_optimizer_step(0,optimizer_succeeded=True,finite=True,scores_unchanged=True)
        self.assertEqual(a.history,{});self.assertFalse(r['direct_supervision']);self.assertEqual(a.last_step,0)
        b=NoDirectExposure();b.load_state_dict(a.state_dict());self.assertEqual(a.state_dict(),b.state_dict())
        with self.assertRaises(ValueError):b.commit_after_optimizer_step(0,optimizer_succeeded=True,finite=True,scores_unchanged=True)
    def test_unsuccessful_stop_cannot_commit(self):
        a=NoDirectExposure()
        with self.assertRaises(ValueError):a.commit_after_optimizer_step(0,optimizer_succeeded=False,finite=True,scores_unchanged=True)
        self.assertEqual(a.last_step,-1)
    def test_masked_terms_are_graph_zero(self):
        for arm in ('M','G','K'):
            raw=torch.randn(6,21,requires_grad=True);zero=raw.sum()*0
            values={k:raw[...,:19].square().sum()*(i+1) for i,k in enumerate(KEYS_V21)}
            fake=dict(terms=values,branch_terms={k:values['geom']/3 for k in ('SECTION','BNECK_MATCHED','BNECK_FALLBACK')},geometry=[{'source':'test'}],diagnostics={'keep':{}},base_loss=raw[...,:19].sum())
            z=dict(raw=raw,output=type('O',(),dict(trajectory=type('T',(),dict(coeffs=raw))(),parameters=type('P',(),dict(durations=raw))()))())
            obj=object.__new__(AblationObjective);obj.arm=arm;obj.weights=weights_for(arm)
            with patch.object(ObjectiveV21,'augment',return_value=fake):r=obj.augment(z,[],0)
            expected=fake['base_loss']+sum(weights_for(arm)[k]*values[k] for k in KEYS_V21)
            self.assertTrue(torch.equal(r['loss'],expected))
            for k in KEYS_V21:
                if k not in ACTIVE[arm]:self.assertEqual(torch.count_nonzero(torch.autograd.grad(r['terms'][k],raw,retain_graph=True)[0]),0)
    def test_protocol_authority_and_weight_drift(self):
        s=dict(arm='S',seed=42,threads=1,batch_size=8,updates=128,device='cpu',precision='FP32',schedule=SCHEDULE,method_id=METHOD['S'],weights=weights_for('S'),ablation=dict(active=list(ACTIVE['S']),frozen_weights=FROZEN,semantics='sequential_conditional_increment_v1'),stop_mode='replace_full_stop',optimizer=dict(name='AdamW',betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_norm=10.),execution=dict(limit=128,checkpoint_every=1,purpose='PG3_V21_ABLATION128',wall_s=1800,rss_mib=4096,min_available_mib=3072,disk_gib=4))
        a=dict(approved=True,spec_sha256=digest(s),scope=s['execution']['purpose'],max_updates=128,authorized_by='test')
        authorize_ablation(s,a);bad=copy.deepcopy(s);bad['weights']['stop']*=.99
        with self.assertRaises(ValueError):authorize_ablation(bad,a)
        bad=copy.deepcopy(s);bad['execution']['wall_s']=3600
        with self.assertRaises(ValueError):validate_ablation_spec(bad)
        a['approved']=False
        with self.assertRaises(ValueError):authorize_ablation(s,a)
    def test_collision_vs_margin_and_unknown(self):
        for d,label in [(-.2,'Collision'),(.029,'Margin')]:self.assertEqual(taxonomy(dict(hard_reasons='collision_or_required_margin_violation',signed_clearance=str(d))),[label])
        self.assertEqual(taxonomy(dict(hard_reasons='continuous_clearance_unproven',signed_clearance='.031')),['Other'])
    def test_stop_exact_existing_formula(self):
        from test_pg3_v2_1_no_corridor import RawModel,base_forward
        m=RawModel();z,s,f=base_forward(m);obj=AblationObjective(m,state_sha(m),'S');r=obj.augment(z,[s],0)
        old=z['base'];stop=full_stopping_risk_v21(old['positions'],old['velocities'],old['time_weights'],z['fields'])
        expected=z['loss']-z['weights']['braking']*z['terms']['braking']+FROZEN['stop']*stop
        self.assertTrue(torch.equal(r['loss'],expected));self.assertEqual(obj.assignment.history,{})
        self.assertIsNone(obj.assignment.pending_step)
