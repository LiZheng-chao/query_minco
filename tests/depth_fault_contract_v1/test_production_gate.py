import ast,json,tempfile,unittest,importlib.util
from pathlib import Path
from query_minco.integration.yopo_execution_v4.evidence import expected_fault,depth_loss_causality
spec=importlib.util.spec_from_file_location('fixture',Path(__file__).with_name('test_contract.py'))
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
class Production(unittest.TestCase):
 def gate(self,change=None,reason="REQUEST ValueError('F0_DEPTH_STALE')"):
  events,fault,terminal,stop=fixture.Contract().fixture(reason)
  tree=ast.parse(Path('experiments/stop_latency_fix_v1/manual.py').read_text())
  ep=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='episode')
  start=next(i for i,n in enumerate(ep.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected' for t in n.targets))
  nodes=ep.body[start:start+3]
  self.assertTrue(any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='depth_loss_causality' for n in nodes for x in ast.walk(n)))
  with tempfile.TemporaryDirectory() as td:
   out=Path(td);(out/'FAULT.json').write_text(json.dumps(fault))
   ns=dict(enough_feedback=True,stage='C',injected=fault['monotonic'],reason=reason,case='depth_loss',stop_time=stop,accepted=[True],qualification=True,expected_fault=expected_fault,depth_loss_causality=depth_loss_causality,events=events,terminal=terminal,out=out,json=json,write=lambda p,v:p.write_text(json.dumps(v)))
   if change:change(ns,out)
   exec(compile(ast.Module(body=nodes,type_ignores=[]),'production_gate','exec'),ns)
   return ns['expected'],ns['qualification']
 def test_old_positive(self):self.assertEqual(self.gate(reason='depth_STALE'),(True,True))
 def test_new_positive(self):self.assertEqual(self.gate(),(True,True))
 def test_missing_injection_file(self):self.assertEqual(self.gate(lambda n,p:(p/'FAULT.json').unlink()),(False,False))
 def test_no_injection(self):self.assertEqual(self.gate(lambda n,p:n.update(injected=None)),(False,False))
 def test_no_physics(self):self.assertEqual(self.gate(lambda n,p:n.update(accepted=[])),(False,False))
 def test_false_pair_despite_valid_string(self):self.assertEqual(self.gate(lambda n,p:n['events'][1]['payload'].update(source_odom_stamp=1.)),(False,False))
 def test_fresh_depth(self):self.assertEqual(self.gate(lambda n,p:n['events'][2].update(ros_now=100.1)),(False,False))
 def test_late_stop(self):self.assertEqual(self.gate(lambda n,p:n.update(stop_time=10.351)),(False,False))
 def test_wrong_reason(self):self.assertEqual(self.gate(reason="REQUEST ValueError('BIRTH_EXPIRED')"),(False,False))
