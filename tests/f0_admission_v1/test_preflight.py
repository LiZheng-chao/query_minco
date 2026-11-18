import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from experiments.f0_admission_v1.prepare import binding_with_additions
from query_minco.integration.birth_execution_v1.evidence import predecessor

class Preflight(unittest.TestCase):
    def test_add_dependency_and_detect_drift(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'dependency.py';p.write_text('x=1')
            old={str(p):hashlib.sha256(p.read_bytes()).hexdigest()}
            self.assertEqual(binding_with_additions(old,[]),old)
            p.write_text('x=2')
            with self.assertRaisesRegex(ValueError,'BASELINE_DRIFT'):binding_with_additions(old,[])
    def test_old_c7_cannot_qualify_extended_binding(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'RESULT.json'
            p.write_text(json.dumps(dict(stage='C_CAMPAIGN',status='PASS',source_binding={},episodes=[])))
            with self.assertRaisesRegex(PermissionError,'SEVEN_CASE_BINDING'):predecessor(p,'C_CAMPAIGN',{'new':'hash'})
    def test_physical_entry_blocked_before_importing_launcher(self):
        p=subprocess.run([sys.executable,'-B','experiments/f0_admission_v1/manual.py',
                          '--stage','F0'],capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0)
        self.assertIn('PHYSICAL_EXECUTION_NOT_AUTHORIZED',p.stderr)
    def test_check_only_returns_before_output_claim_or_ros(self):
        tree=ast.parse(Path('experiments/stop_latency_fix_v1/manual.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        check=next(n for n in main.body if isinstance(n,ast.If) and isinstance(n.test,ast.Attribute) and isinstance(n.test.value,ast.Name) and n.test.value.id=='a' and n.test.attr=='check_only')
        self.assertIsInstance(check.body[-1],ast.Return)
        self.assertLess(check.lineno,next(n.lineno for n in main.body
                        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='out' for t in n.targets)))
