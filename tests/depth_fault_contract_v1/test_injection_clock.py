"""Exercise the real fault-writer AST after a delayed owner identity check. No ROS."""
import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from query_minco.integration.yopo_execution_v4.evidence import depth_loss_causality

ROOT = Path(__file__).resolve().parents[2]

class InjectionClock(unittest.TestCase):
    def execute(self, fresh, wall, delayed_now=10.0, fail_write=False):
        tree = ast.parse((ROOT/'experiments/stop_latency_fix_v1/manual.py').read_text())
        episode = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'episode')
        branch = next(n for n in ast.walk(episode) if isinstance(n, ast.If) and n.body and isinstance(n.body[0], ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'injected' for t in n.body[0].targets))
        rows = []
        def write(path, data):
            if fail_write:
                raise FileExistsError('preserve prior evidence')
            rows.append(data)
        ns = dict(now=delayed_now, armed_at=8.0, injected=None, stage='C', case='depth_loss', time=SimpleNamespace(monotonic=lambda:fresh, time=lambda:wall), out=Path('/not-created'), write=write, h={'plant':{'integration_steps':1000}})
        exec(compile(ast.Module(body=branch.body[:2], type_ignores=[]), '<actual fault writer>', 'exec'), ns)
        return ns, rows

    def fixture(self, fault):
        events = [dict(kind='plant', accepted=True, mode='ARMED', monotonic=10.003, payload=dict(integration_steps=1000, state_source='DYNAMICS_INTEGRATED')), dict(kind='depth', accepted=True, stamp=99.998, monotonic=10.006, payload=dict(source_odom_stamp=99.998)), dict(kind='fault', monotonic=10.208, ros_now=100.208, payload=dict(reason='depth_STALE'))]
        terminal = dict(reason='depth_STALE', terminal_state='OWNED_PROCESS_TERMINATED', actual_exit_code=-9, pid=12, start_ticks='abc')
        return events, fault, terminal, 10.220

    def test_delayed_identity_check_uses_fresh_monotonic(self):
        ns, rows = self.execute(10.009, 100.009)
        self.assertEqual(ns['injected'],10.009)
        self.assertEqual(rows[0]['monotonic'],10.009)
        self.assertEqual(rows[0]['ros_time'],100.009)
        self.assertTrue(depth_loss_causality(*self.fixture(rows[0])))
        old = copy.deepcopy(rows[0]); old['monotonic']=10.0
        self.assertFalse(depth_loss_causality(*self.fixture(old)))

    def test_monotonic_is_not_replaced_with_wall_clock(self):
        ns, rows = self.execute(20.5, 1794989383.6)
        self.assertEqual(rows[0]['monotonic'],20.5)
        self.assertEqual(rows[0]['ros_time'],1794989383.6)

    def test_post_injection_depth_is_still_rejected(self):
        ns, rows = self.execute(10.009,100.009)
        ev,f,t,stop=self.fixture(rows[0]);ev.insert(2,dict(kind='depth',accepted=True,stamp=100.02,monotonic=10.02,payload=dict(source_odom_stamp=100.02)))
        self.assertFalse(depth_loss_causality(ev,f,t,stop))

    def test_stop_deadline_not_extended(self):
        ns,rows=self.execute(10.009,100.009)
        ev,f,t,stop=self.fixture(rows[0])
        self.assertFalse(depth_loss_causality(ev,f,t,10.360))

    def test_existing_fault_file_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            self.execute(10.009,100.009,fail_write=True)

if __name__ == '__main__':
    unittest.main()
