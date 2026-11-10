"""Checkpoint bundles retain assignment counts and reject missing proof/state."""
import unittest,tempfile
from pathlib import Path
import torch
from .exposure_experiment import committed,restore
from .fulltrain128 import lr_at,SCHEDULE
from query_minco.training.pg3_feasible_mode_v1.exposure_assignment import ExposureRotation
from query_minco.training.pg3_feasible_mode_v1.test_exposure_assignment import choose
class ExposureCheckpointTests(unittest.TestCase):
    def test_exact_checkpoint_restore_and_next_assignment(self):
        with tempfile.TemporaryDirectory() as d:
            m=torch.nn.Linear(2,1);o=torch.optim.AdamW(m.parameters(),lr=lr_at(0),weight_decay=0.)
            p=ExposureRotation();choose(p);m(torch.ones(1,2)).sum().backward();o.step();p.commit(0)
            spec=dict(device="cpu",schedule=SCHEDULE);folder=committed(Path(d),m,o,spec,1,dict(step=1),p)
            n=torch.nn.Linear(2,1);v=torch.optim.AdamW(n.parameters(),lr=.1);q=ExposureRotation()
            blob=restore(folder,n,v,spec,q,False)
            self.assertEqual(q.state_dict(),p.state_dict());self.assertEqual(choose(p,step=1),choose(q,step=1))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(m.parameters(),n.parameters())))
            self.assertTrue(all(int(s["step"])==1 for s in v.state.values()))
    def test_missing_or_tampered_state_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            m=torch.nn.Linear(2,1);o=torch.optim.AdamW(m.parameters(),lr=lr_at(0));p=ExposureRotation()
            spec=dict(device="cpu",schedule=SCHEDULE);folder=committed(Path(d),m,o,spec,0,{},p)
            (folder/"ASSIGNMENT_STATE.json").write_text("{}")
            with self.assertRaises(ValueError):restore(folder,m,o,spec,ExposureRotation(),False)
    def test_pending_or_wrong_cursor_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            m=torch.nn.Linear(2,1);o=torch.optim.AdamW(m.parameters(),lr=lr_at(0));p=ExposureRotation();choose(p)
            with self.assertRaises(ValueError):committed(Path(d),m,o,dict(device="cpu"),1,{},p)
if __name__=="__main__":unittest.main()
