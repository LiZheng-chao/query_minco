import unittest
import torch
from query_minco import QueryMincoPlanner
from query_minco.trajectory import TrajectoryPostprocessor
from query_minco.p2b.imitation import (assign_costs,NoTeacherError,forward_batch,teacher_evaluate)
from query_minco.p2b.toy_data import sample_for_seed


class P2BImitationTests(unittest.TestCase):
    def test_zero_one_and_more_than_six_teachers(self):
        with self.assertRaises(NoTeacherError):assign_costs(torch.zeros(6,0))
        costs=torch.tensor([[4.],[3.],[2.],[1.],[2.],[5.]],requires_grad=True)
        loss,queries,teachers,unmatched=assign_costs(costs)
        self.assertEqual(queries.tolist(),[3]);self.assertEqual(unmatched,0)
        loss.backward()
        self.assertEqual(float(costs.grad.sum()),1.)
        many=torch.rand(6,8,requires_grad=True)
        loss,q,t,unmatched=assign_costs(many);loss.backward()
        self.assertEqual(len(q),6);self.assertEqual(unmatched,2)
        self.assertEqual(len(set(t.tolist())),6)
        self.assertTrue(bool(torch.isfinite(many.grad).all()))

    def test_teacher_uses_physical_seconds_not_normalized_progress(self):
        s=sample_for_seed(32000,"train")
        times=torch.tensor([[0.,.1,.3]]).expand(6,-1)
        expected=times[...,None]*s.state[:3]
        actual=teacher_evaluate(s.teacher_coeffs[0],s.teacher_durations[0],times,0)
        torch.testing.assert_close(actual,expected,atol=2e-5,rtol=0)

    def test_training_loss_reaches_network_and_scores_remain_unused(self):
        torch.set_num_threads(1);torch.manual_seed(42)
        samples=[sample_for_seed(32000,"train"),sample_for_seed(32001,"train")]
        model=QueryMincoPlanner()
        loss,_,_=forward_batch(model,TrajectoryPostprocessor(),samples,"cpu")
        loss.backward()
        for module in (model.depth_encoder,model.state_encoder,model.query_decoder,model.trajectory_head):
            grads=[p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and bool(torch.isfinite(g).all()) for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads),0)
        final=model.trajectory_head.mlp[-1]
        self.assertTrue(bool((final.weight.grad[19:]==0).all()))
        self.assertTrue(bool((final.bias.grad[19:]==0).all()))
