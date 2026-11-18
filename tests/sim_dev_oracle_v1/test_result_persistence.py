"""Execute the REAL generated persistence/aggregation AST only; no ROS/Plant."""
import ast,json,tempfile,types,hashlib,unittest
from pathlib import Path
P=Path('research_outputs/sim_dev_oracle_20261118_v1/generated_v2/manual.py')
def write(path,value):
 with Path(path).open('x') as f:json.dump(value,f)
class Persistence(unittest.TestCase):
 def run_case(self,status):
  tree=ast.parse(P.read_text());episode=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='episode');start=next(i for i,x in enumerate(episode.body) if isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Attribute) and isinstance(x.value.func.value,ast.Name) and x.value.func.value.id=='outcome' and x.value.func.attr=='update' and any(k.arg=='experiment_mode' for k in x.value.keywords));fn=ast.FunctionDef(name='persist',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=episode.body[start:],decorator_list=[])
  main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main');index=next(i for i,x in enumerate(main.body) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in x.targets));tail=main.body[index:]
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);out=root/'none';out.mkdir();ns=dict(out=out,outcome=dict(status=status,stage='S'),owner_events=[],write=write,json=json,Path=Path)
   exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'generated_persist','exec'),ns);result=ns['persist']();self.assertEqual(json.loads((out/'RESULT.json').read_text()),result);self.assertFalse(result['strict_depth_qualified'])
   # Existing-file protection is deliberately retained.
   before=(out/'RESULT.json').read_bytes()
   with self.assertRaises(FileExistsError):ns['persist']()
   self.assertEqual((out/'RESULT.json').read_bytes(),before)
   ns.update(out=root,a=types.SimpleNamespace(stage='S'),master_codes=[0],results=[dict(result_path=str(out/'RESULT.json'))],sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest())
   code=compile(ast.fix_missing_locations(ast.Module(body=tail,type_ignores=[])),'generated_aggregate','exec')
   if status=='FAIL':
    with self.assertRaises(SystemExit):exec(code,ns)
   else:exec(code,ns)
   self.assertEqual(json.loads((root/'RESULT.json').read_text())['status'],status)
 def test_success_writes_child_once_then_summary(self):self.run_case('PASS')
 def test_failure_stays_failure_in_both_results(self):self.run_case('FAIL')
 def test_no_outer_duplicate_child_write(self):
  self.assertNotIn("write(out/case/'RESULT.json',result)",P.read_text());self.assertIn("outcome.update(experiment_mode='SIM_DEV_ORACLE'",P.read_text())
if __name__=='__main__':unittest.main(verbosity=2)
