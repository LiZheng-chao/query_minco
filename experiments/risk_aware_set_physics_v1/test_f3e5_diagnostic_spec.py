"""Specification/static tests only. Does NOT compile or apply Simulator patch."""
import errno,json,unittest
from pathlib import Path
ROOT=Path('/home/lizheng/query_minco_mainline');OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution009'
def flush_spec(payload,writer,fsync,close):
 offset=0;ok=True
 while offset<len(payload):
  try:n=writer(payload[offset:])
  except InterruptedError:continue
  except OSError:ok=False;break
  if n<=0:ok=False;break
  offset+=n
 if ok:ok=bool(fsync())
 if not close():ok=False
 return ok and offset==len(payload)
def metrology_spec(previous,current):
 steps0,wall0,mono0,dt0=previous;steps,wall,mono,dt=current
 return steps>steps0 and wall>wall0 and mono>mono0 and dt==dt0 and abs((steps-steps0)*dt-(wall-wall0))<=.002 and abs((wall-wall0)-(mono-mono0))<=.005
class Tests(unittest.TestCase):
 def test_partial_writes_and_EINTR(self):
  collected=bytearray();calls=[0]
  def writer(data):
   calls[0]+=1
   if calls[0]==1:raise InterruptedError(errno.EINTR,'unit')
   n=min(3,len(data));collected.extend(data[:n]);return n
  self.assertTrue(flush_spec(b'complete diagnostic text',writer,lambda:True,lambda:True))
  self.assertEqual(collected,b'complete diagnostic text')
 def test_zero_write_fails(self):
  self.assertFalse(flush_spec(b'row',lambda _:0,lambda:True,lambda:True))
 def test_io_error_fails(self):
  def writer(_):raise OSError(errno.EIO,'unit')
  self.assertFalse(flush_spec(b'row',writer,lambda:True,lambda:True))
 def test_fsync_error_fails(self):
  self.assertFalse(flush_spec(b'row',lambda x:len(x),lambda:False,lambda:True))
 def test_close_error_fails(self):
  self.assertFalse(flush_spec(b'row',lambda x:len(x),lambda:True,lambda:False))
 def test_clock_mismatch_fails_without_removing_row(self):
  before=(10,1.,2.,.001);bad=(12,1.018,2.018,.001)
  self.assertFalse(metrology_spec(before,bad));rows=[before,bad];self.assertEqual(len(rows),2)
  self.assertTrue(metrology_spec(before,(20,1.01,2.01,.001)))
 def test_overflow_is_failure(self):
  rows=[];overflow=False
  for i in range(3):
   if len(rows)==2:overflow=True;continue
   rows.append(i)
  self.assertTrue(overflow);self.assertEqual(rows,[0,1])
 def test_final_diff_is_only_additions_and_old_draft_missing_flush(self):
  text=(OUT/'patches/PRIVATE_DIAGNOSTIC_PATCH_PROPOSAL.diff').read_text()
  self.assertFalse([x for x in text.splitlines() if x.startswith('-') and not x.startswith('---')])
  for token in ('step_diagnostic_enabled,false','O_EXCL|O_NOFOLLOW|O_CLOEXEC','write(fd_','errno==EINTR','fsync(fd_)','close(fd_)','diagnostics.finish()'):
   self.assertIn(token,text)
  review=json.loads((OUT/'DIAGNOSTIC_STATIC_REVIEW.json').read_text())
  self.assertFalse(review['Simulator_patch_compiled']);self.assertFalse(review['Simulator_patch_applied'])
  self.assertIn('never written or closed',review['old_draft_verdict'])
if __name__=='__main__':
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 report=dict(mode='PYTHON_SPEC_AND_STATIC_ONLY_NOT_CPP_PATCH_REGRESSION',tests=r.testsRun,passed=r.wasSuccessful(),patch_applied=False,patch_compiled=False,patch_run=False,actual_fixed_input_PVA_regression='NOT_RUN_NEEDS_AUTHORIZATION')
 (OUT/'DIAGNOSTIC_SPEC_CPU_RESULTS.json').write_text(json.dumps(report,indent=2));assert r.wasSuccessful()
