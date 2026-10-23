"""Portable SYNTHETIC_ONLY RSPS tests; CPU forward/backward, no optimizer."""
import unittest
import numpy as np
import torch
from query_minco.training.losses.set_physical_risk import box_sdf, segment_samples, physical_loss, reduce_set
from query_minco.evaluation.signed_geometry import numpy_box_sdf
from query_minco.trajectory import TrajectoryPostprocessor, MincoTrajectory
from query_minco.training.matching.hungarian import match


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.box = torch.tensor([[[-1.,-1.,-1.],[1.,1.,1.]]])

    def samples(self, dtype=torch.float32):
        x = torch.linspace(-1.5, 2., 17, dtype=dtype)
        p = torch.zeros(2,6,17,3,dtype=dtype)
        p[...,0] = x
        p[...,1] = torch.arange(6,dtype=dtype)[None,:,None] * .31 + .17
        v = torch.ones_like(p); v[...,1:] = 0
        a = torch.ones_like(p[...,0])
        return p,v,a

    def loss(self,p,v,a,beta=0,**kw):
        return physical_loss(p,v,a,[self.box.to(p),self.box.to(p)],geometry_bound=[True,True],beta=beta,**kw)

    def test_sdf_reference_faces_corners_penetration(self):
        p=torch.tensor([[0.,0.,0.],[.99,.2,.3],[1.,.2,.3],[1.001,.2,.3],[2.,2.,2.],[-2.,.2,.3]])
        want=numpy_box_sdf(p.numpy(),dict(min=[-1]*3,max=[1]*3))
        np.testing.assert_allclose(box_sdf(p,self.box).numpy(),want,atol=1e-6)
        g=torch.autograd.grad(box_sdf(p.requires_grad_(),self.box).sum(),p)[0]
        self.assertTrue(torch.isfinite(g).all())
        self.assertGreater(g[-1,0].neg().item(),0)

    def test_gradcheck_smooth_sdf_and_loss(self):
        p=torch.tensor([[1.4,.2,.3],[.3,.6,.1],[1.6,1.4,.2]],dtype=torch.double,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda z:box_sdf(z,self.box.double()),(p,)))
        p,v,a=self.samples(torch.double)
        p=(p+.013).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda z:self.loss(z,v,a)[0],(p,),fast_mode=True))

    def test_approach_direction_and_detach(self):
        p,v,a=self.samples()
        _,d=self.loss(p,v,a,beta=2)
        self.assertLess(d['clearance_rate'][0,0,0],0)
        self.assertGreater(d['clearance_rate'][0,0,-1],0)
        self.assertGreater(d['unnormalized_risk'][0,0,0],d['unnormalized_risk'][0,0,-1])
        self.assertFalse(d['risk_weights'].requires_grad)
        torch.testing.assert_close((d['risk_weights']*a/a.sum(-1,keepdim=True)).sum(-1),torch.ones(2,6))
        # Check grad SDF dot v against displacement in physical seconds.
        h=1e-4; point=torch.tensor([[1.5,.2,.3]]); vel=torch.tensor([[-.7,.1,.1]])
        fd=(box_sdf(point+h*vel,self.box)-box_sdf(point-h*vel,self.box))/(2*h)
        torch.testing.assert_close(fd,torch.tensor([-.7]),atol=.001,rtol=.001)

    def test_beta_zero_and_count_corrected_equal_strength(self):
        p,v,a=self.samples()
        q,_=self.loss(p,v,a)
        r,_=self.loss(p,v,a,beta=0)
        self.assertTrue(torch.equal(q,r))
        mask=torch.arange(6)[None,:]<torch.tensor([1,4])[:,None]
        torch.testing.assert_close(reduce_set(q,mask),reduce_set(q))
        for m in [1,4]:
            z=torch.arange(6,dtype=torch.float32)[None].requires_grad_()
            mask=torch.arange(6)[None]<m
            reduce_set(z,mask,1,2).backward()
            expected=torch.where(mask,1.,2.)/(m+2*(6-m))
            torch.testing.assert_close(z.grad,expected)

    def test_nonfinite_and_shapes(self):
        p,v,a=self.samples()
        for x in [float('nan'),float('inf')]:
            bad=p.clone();bad[0,0,0,0]=x
            with self.assertRaises(ValueError):self.loss(bad,v,a)
            with self.assertRaises(ValueError):self.loss(p,v,a,eps=x)
        for aa in [a*0,-a,a[:,:1]]:
            with self.assertRaises(ValueError):self.loss(p,v,aa)
        with self.assertRaises(ValueError):box_sdf(p,torch.zeros_like(self.box))
        with self.assertRaises(ValueError):self.loss(p,v[:,:1],a)

    def test_missing_vs_certified_empty(self):
        p,v,a=self.samples();p.requires_grad_()
        for b,g in [(None,[True,True]),([self.box]*2,[False,True]),([None]*2,[True]*2)]:
            with self.assertRaises(ValueError):physical_loss(p,v,a,b,geometry_bound=g)
        q,_=physical_loss(p,v,a,[torch.empty(0,2,3)]*2,geometry_bound=[True]*2)
        q.sum().backward();self.assertEqual(torch.count_nonzero(p.grad).item(),0)
        self.assertEqual(q.sum().item(),0)

    def test_ties_zero_axis_sign_and_frame(self):
        p=torch.tensor([[0.,0.,0.],[1.,1.,1.],[.2,.2,.2]],requires_grad=True)
        d=box_sdf(p,self.box.repeat(2,1,1));d.sum().backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        for axis in range(3):
            x=torch.zeros(1,3);x[0,axis]=2
            y=-x
            torch.testing.assert_close(box_sdf(x,self.box),box_sdf(y,self.box))

    def test_all_collision_partial_safe_and_near_far(self):
        p,v,a=self.samples()
        for x in [0.,1.,20.]:
            pp=torch.full_like(p,x)
            q,_=self.loss(pp,v,a)
            self.assertTrue(torch.isfinite(q).all())
            if x==0: self.assertTrue((q>1).all())
            if x==20: self.assertTrue((q<1e-10).all())
        q,_=self.loss(p,v,a)
        self.assertGreater(torch.unique(q).numel(),1)

    def test_minco_m1_m4_unmatched_raw_gradient(self):
        torch.manual_seed(17)
        for beta in [0.,1.]:
            raw=(torch.randn(2,6,21)*.08).requires_grad_()
            state=torch.zeros(2,12);state[:,0]=.8;state[:,11]=-1;state[:,6]=torch.tensor([2.8,5.8])
            o=TrajectoryPostprocessor()(raw,state)
            p,v,a=segment_samples(o.trajectory,32)
            boxes=torch.tensor([[[.8,-.4,-.4],[1.5,.4,.4]]])
            q,_=physical_loss(p,v,a,[boxes]*2,geometry_bound=[True]*2,beta=beta)
            masks=[]
            for m in [1,4]:
                _,idx,_=match((torch.arange(6)[:,None]-torch.arange(m)[None]).float().square(),'B_set')
                mask=torch.zeros(6,dtype=torch.bool);mask[idx]=True;masks.append(mask)
            mask=torch.stack(masks)
            for group in [None,mask]:
                g=torch.autograd.grad(reduce_set(q,group,1,2),raw,retain_graph=True)[0]
                self.assertTrue(torch.isfinite(g).all())
                self.assertTrue((g[:,:,:19].norm(dim=-1)[~mask]>0).all())
                self.assertEqual(torch.count_nonzero(g[:,:,19:21]).item(),0)
                self.assertTrue((g[:,:,:19].norm(dim=-1)[mask]>0).all())
            self.assertGreater(torch.unique(q.detach()).numel(),6)
            print('MINCO',beta,'per_query',q.detach().tolist(),'matched_norm',g[mask].norm().item(),'unmatched_norm',g[~mask].norm().item())

    def test_quadrature_short_segments_and_duration_normalization(self):
        c=torch.zeros(1,6,3,3,6);c[:,:,:,0,0]=2.
        dt=torch.tensor([[[1e-5,.2,1.], [.1,.2,.3], [.3,.2,.1], [.4,.4,.4], [.2,.7,.4], [.6,.2,.1]]])
        tr=MincoTrajectory(c,dt,torch.ones(1,6,dtype=torch.bool),{})
        p,v,a=segment_samples(tr,32)
        torch.testing.assert_close(a.sum(-1),torch.ones(1,6))
        q,_=physical_loss(p,v,a,[self.box],geometry_bound=[True])
        torch.testing.assert_close(q,q[:,0:1].expand_as(q))
        dt[0,0,0]=0
        with self.assertRaises(ValueError):segment_samples(tr)

    def test_sampling_density_and_missed_thin_obstacle(self):
        # Linear polynomial segment crossing a thin slab. Dense reference is diagnostic.
        c=torch.zeros(1,6,3,3,6);c[:,:,:,0,1]=1
        dt=torch.ones(1,6,3)
        tr=MincoTrajectory(c,dt,torch.ones(1,6,dtype=torch.bool),{})
        box=torch.tensor([[[.5001,-1.,-1.],[.5003,1.,1.]]])
        result=[]
        for n in [32,64,128,4097]:
            p,v,a=segment_samples(tr,n)
            q,_=physical_loss(p,v,a,[box],geometry_bound=[True],radius=0,eps=.01)
            hit=(box_sdf(p,box)<0).any().item()
            result.append((n,q.mean().item(),hit))
        self.assertLess(abs(result[2][1]-result[3][1]),abs(result[0][1]-result[3][1]))
        self.assertFalse(result[0][2]);self.assertTrue(result[-1][2])
        print('DENSITY_SYNTHETIC_ONLY',result)

    def test_risk_finite_difference_and_minco_raw_finite_difference(self):
        # Plane regions keep detached normal/velocity weights fixed under p perturbations.
        p,v,a=self.samples(torch.double)
        p[...,0]=torch.linspace(1.3,1.7,17);p[...,1:]=.1
        v[...,0]=torch.linspace(-1.,.7,17)
        p.requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda z:self.loss(z,v,a,beta=1.)[0],(p,),fast_mode=True))
        torch.manual_seed(9)
        raw=(torch.randn(1,6,21)*.03).requires_grad_()
        state=torch.zeros(1,12);state[:,0]=.8;state[:,6]=2.8;state[:,11]=-1
        box=torch.tensor([[[.8,-.4,-.4],[1.5,.4,.4]]])
        def f(z):
            o=TrajectoryPostprocessor()(z,state)
            p,v,a=segment_samples(o.trajectory,64)
            return physical_loss(p,v,a,[box],geometry_bound=[True],beta=0)[0].mean()
        value=f(raw);g=torch.autograd.grad(value,raw)[0]
        comparisons=[]
        for k in [0,3,6,15,16]:
            h=.002;plus=raw.detach().clone();minus=plus.clone()
            plus[0,2,k]+=h;minus[0,2,k]-=h
            fd=((f(plus)-f(minus))/(2*h)).item();ad=g[0,2,k].item()
            self.assertAlmostEqual(fd,ad,delta=max(2e-5,.06*abs(ad)))
            comparisons.append((k,ad,fd))
        print('MINCO_RAW_FINITE_DIFFERENCE',comparisons)

if __name__=='__main__':unittest.main(verbosity=2)
