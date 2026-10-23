"""SYNTHETIC_ONLY sampling/certificate/gradient audit, no optimization."""
import json,os,time,pathlib,unittest
import numpy as np
import torch
from query_minco.trajectory import MincoTrajectory
from query_minco.training.losses.set_physical_risk import segment_samples,refined_segment_samples,physical_loss,box_sdf
from query_minco.evaluation.validation import certify_piece

class ThinAudit(unittest.TestCase):
    def test_sampling_gradient_cost_and_conservative_reference(self):
        torch.set_num_threads(1)
        cases={'thin':(0.,1.,0.),'tangent':(1.15,0.,0.),'corner':(1.10,0.,1.10),'inside':(0.,0.,0.),'safe':(2.,0.,0.)}
        results=[]
        for case,(offset,speed,z) in cases.items():
            base=torch.zeros(1,6,3,3,6);base[:,:,:,0,0]=offset;base[:,:,:,0,1]=speed;base[:,:,:,2,0]=z
            if case=='thin':box=torch.tensor([[[.5001,-1.,-1.],[.5003,1.,1.]]]);radius=0.
            else:box=torch.tensor([[[-1.,-1.,-1.],[1.,1.,1.]]]);radius=.12
            records=[]
            for strategy,n in [('fixed',32),('fixed',64),('fixed',128),('local',32),('geometry',32),('fixed',4097)]:
                c=base.clone().requires_grad_();dt=torch.ones(1,6,3,requires_grad=True)
                tr=MincoTrajectory(c,dt,torch.ones(1,6,dtype=torch.bool),{})
                start=time.perf_counter()
                if strategy=='fixed':p,v,a=segment_samples(tr,n);selected=p.shape[2]
                else:
                    p,v,a,diag=refined_segment_samples(tr,[box],strategy=strategy,base_points=n)
                    selected=diag['actual_selected_points']/6
                q,_=physical_loss(p,v,a,[box],geometry_bound=[True],radius=radius,eps=.01)
                value=q.mean();g=torch.autograd.grad(value,[c,dt])
                elapsed=time.perf_counter()-start
                self.assertTrue(all(torch.isfinite(x).all() for x in g))
                torch.testing.assert_close(a.sum(-1),torch.ones(1,6))
                hit=bool((box_sdf(p,box)-radius<0).any())
                records.append(dict(strategy=strategy,base_points_per_segment=n,evaluated_points_per_query=p.shape[2],selected_points_per_query=selected,hit=hit,loss=value.item(),coeff_grad_norm=g[0].norm().item(),duration_grad_norm=g[1].norm().item(),elapsed_s=elapsed))
            dense=records[-1]['loss']
            for x in records:x['loss_abs_error_to_dense']=abs(x['loss']-dense)
            if os.environ.get('RSPS_AUDIT_OUTPUT'):
                with (pathlib.Path(os.environ['RSPS_AUDIT_OUTPUT'])/('PARTIAL_'+case+'.json')).open('x') as f:json.dump(records,f,indent=2)
            if case=='thin':
                self.assertFalse(records[0]['hit']);self.assertFalse(records[2]['hit'])
                self.assertFalse(records[3]['hit']);self.assertTrue(records[4]['hit'])
                self.assertLess(records[4]['loss_abs_error_to_dense'],records[0]['loss_abs_error_to_dense'])
            # Reuse original sphere-body + .03 margin Bernstein numerical check.
            boxes=[dict(min=box[0,0].tolist(),max=box[0,1].tolist())]
            cert=certify_piece(base[0,0,0].numpy(),1.,boxes)
            if case=='safe':self.assertTrue(cert['passed'])
            results.append(dict(case=case,measurements=records,independent_original_bernstein=cert))
        out=os.environ.get('RSPS_AUDIT_OUTPUT')
        if out:
            with (pathlib.Path(out)/'THIN_RESULTS.json').open('x') as f:json.dump(results,f,indent=2)
        print(json.dumps(results))
    def test_ties_and_invalid_refinement(self):
        c=torch.zeros(1,6,3,3,6);dt=torch.ones(1,6,3)
        tr=MincoTrajectory(c,dt,torch.ones(1,6,dtype=torch.bool),{})
        box=torch.tensor([[[-1.,-1.,-1.],[1.,1.,1.]]])
        for strategy in ['local','geometry']:
            p,v,a,_=refined_segment_samples(tr,[box],strategy=strategy)
            q,_=physical_loss(p,v,a,[box],geometry_bound=[True])
            self.assertTrue(torch.isfinite(q).all())
        with self.assertRaises(ValueError):refined_segment_samples(tr,[box],strategy='unknown')
    def test_frozen_grid_finite_difference(self):
        # The selection itself is detached; validate precisely that surrogate.
        from query_minco.trajectory.sampler_e1 import evaluate_piece
        base=torch.zeros(1,6,3,3,6,dtype=torch.double);base[:,:,:,0,1]=1
        dt=torch.ones(1,6,3,dtype=torch.double)
        box=torch.tensor([[[.5001,-1.,-1.],[.5003,1.,1.]]],dtype=torch.double)
        for strategy in ['local','geometry']:
            c=base.clone().requires_grad_();d=dt.clone().requires_grad_()
            tr=MincoTrajectory(c,d,torch.ones(1,6,dtype=torch.bool),{})
            p,v,a,_=refined_segment_samples(tr,[box],strategy=strategy)
            u=p[...,0].reshape(1,6,3,-1).detach()
            gaps=u[...,1:]-u[...,:-1]
            trap=torch.cat([gaps[...,:1]/2,(gaps[...,:-1]+gaps[...,1:])/2,gaps[...,-1:]/2],-1)
            def frozen(cc,dd):
                pos,vel=(evaluate_piece(cc,u*dd[...,None],k).flatten(2,3) for k in (0,1))
                weights=(trap*dd[...,None]/dd.sum(-1)[...,None,None]).flatten(2,3)
                return physical_loss(pos,vel,weights,[box],geometry_bound=[True],radius=0.,eps=.01)[0].mean()
            actual=physical_loss(p,v,a,[box],geometry_bound=[True],radius=0.,eps=.01)[0].mean()
            ga=torch.autograd.grad(actual,(c,d))
            gf=torch.autograd.grad(frozen(c,d),(c,d))
            for x,y in zip(ga,gf):torch.testing.assert_close(x,y)
            self.assertTrue(torch.autograd.gradcheck(frozen,(c,d),fast_mode=True,atol=2e-5,rtol=1e-3))

if __name__=='__main__':unittest.main(verbosity=2)
