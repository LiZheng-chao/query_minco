import json,time,tempfile,threading,os,types,sys,unittest
from pathlib import Path
from unittest.mock import patch
from query_minco.integration.depth_minco_t2_v1.space import BirthGrant
class ClockTests(unittest.TestCase):
 def fixture(self,folder):
  self.selected=dict(map_id='map',selected=dict(position=[.25,-.25,2.],clearance_lower_bound_m=.62146))
  self.cert=dict(scope='ENGINEERING_BIRTH_BALL_ONLY',center_world=[.25,-.25,2.],radius_m=.6,map_id='map',run_nonce='owned',issued=990.,expires=1020.)
  self.truth=dict(status='PASS',map_id='map',monotonic=100.)
  self.args=['_run_nonce:=owned'];self.folder=Path(folder)
  (self.folder/'truth_data').mkdir();(self.folder/'plant').mkdir()
  self.save()
 def save(self):
  (self.folder/'BIRTH_CERTIFICATE.json').write_text(json.dumps(self.cert))
  (self.folder/'truth_data/HEARTBEAT.json').write_text(json.dumps(self.truth))
  (self.folder/'plant/PROCESS_START.json').write_text(json.dumps(dict(argv=self.args)))
 def load(self,now=1000.,mono=100.):
  return BirthGrant.load(self.folder,self.selected,clock=lambda:now,monotonic_clock=lambda:mono)
 def test_old_fails_new_passes_same_read_order_race(self):
  # Exact old load body (from_records math remains unchanged), no history-file dependency.
  def legacy_load(run,selected,now,mono):
   raw=(run/'BIRTH_CERTIFICATE.json').read_bytes()
   truth=json.loads((run/'truth_data/HEARTBEAT.json').read_text())
   args=json.loads((run/'plant/PROCESS_START.json').read_text())['argv']
   return BirthGrant.from_records(raw,truth,args,selected,now,mono)
  with tempfile.TemporaryDirectory() as td:
   self.fixture(td);pre_read_mono=100.
   # Producer publishes between caller sampling and consumer reading.
   self.truth['monotonic']=100.001;self.save()
   with self.assertRaisesRegex(ValueError,'BIRTH_TRUTH_INVALID'):
    legacy_load(self.folder,self.selected,1000.,pre_read_mono)
   g=self.load(1000.002,100.002)
   self.assertEqual(g.radius,.6);self.assertEqual(g.center,(.25,-.25,2.))
   self.assertAlmostEqual(g.expires,1000.201)
 def test_future_stale_fail_map_certificate(self):
  with tempfile.TemporaryDirectory() as td:
   self.fixture(td)
   for update in [dict(monotonic=100.001),dict(monotonic=99.799),dict(status='FAIL'),dict(map_id='wrong')]:
    self.truth=dict(status='PASS',map_id='map',monotonic=100.);self.truth.update(update);self.save()
    with self.subTest(update=update),self.assertRaises(ValueError):self.load()
   self.truth=dict(status='PASS',map_id='map',monotonic=100.);self.cert['expires']=999.;self.save()
   with self.assertRaises(ValueError):self.load()
 def test_finish_clock_cannot_extend_grant(self):
  with tempfile.TemporaryDirectory() as td:
   self.fixture(td);ticks=iter([1000.,1000.201])
   with self.assertRaisesRegex(ValueError,'CLOCK_OR_EXPIRY'):
    BirthGrant.load(self.folder,self.selected,clock=lambda:next(ticks),monotonic_clock=lambda:100.)
 def test_ros_rollback_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   self.fixture(td);ticks=iter([1000.,999.99])
   with self.assertRaises(ValueError):BirthGrant.load(self.folder,self.selected,clock=lambda:next(ticks),monotonic_clock=lambda:100.)
 def test_certificate_expires_during_check(self):
  with tempfile.TemporaryDirectory() as td:
   self.fixture(td);self.cert['expires']=1000.01;self.save();ticks=iter([1000.,1000.02])
   with self.assertRaises(ValueError):BirthGrant.load(self.folder,self.selected,clock=lambda:next(ticks),monotonic_clock=lambda:100.)
 def test_clock_calls_after_all_reads(self):
  with tempfile.TemporaryDirectory() as td:
   self.fixture(td);seen=[];read=Path.read_text
   def trace(p,*a,**k):
    result=read(p,*a,**k);seen.append(p.name);return result
   def clock():self.assertIn('HEARTBEAT.json',seen);self.assertIn('PROCESS_START.json',seen);return 1000.
   with patch.object(Path,'read_text',trace):BirthGrant.load(self.folder,self.selected,clock=clock,monotonic_clock=lambda:100.)
 def test_100hz_writer_1000hz_checks(self):
  with tempfile.TemporaryDirectory() as td:
   self.fixture(td);now=time.time();self.cert.update(issued=now,expires=now+30);self.truth['monotonic']=time.monotonic();self.save()
   stop=threading.Event();writes=[];errors=[]
   def producer():
    next_tick=time.monotonic()
    while not stop.is_set():
     try:
      stamp=time.monotonic();p=self.folder/'truth_data/HEARTBEAT.tmp'
      p.write_text(json.dumps(dict(status='PASS',map_id='map',monotonic=stamp)));os.replace(p,p.with_suffix('.json'));writes.append(stamp)
     except Exception as e:errors.append(repr(e))
     next_tick+=.01;stop.wait(max(0,next_tick-time.monotonic()))
   thread=threading.Thread(target=producer);thread.start();checks=0;failures=[]
   start=time.monotonic();deadline=start+2.
   try:
    while time.monotonic()<deadline:
     try:BirthGrant.load(self.folder,self.selected);checks+=1
     except Exception as e:failures.append(repr(e))
     time.sleep(.001)
   finally:stop.set();thread.join(1)
   result=dict(scope='NONPHYSICAL_OWNED_TEMPFILE',seconds=time.monotonic()-start,writes=len(writes),checks=checks,failures=failures,writer_errors=errors)
   Path(os.environ.get('BIRTH_CLOCK_AUDIT_OUTPUT',str(self.folder/'HIGH_RATE.json'))).write_text(json.dumps(result,indent=2))
   self.assertGreaterEqual(len(writes),100);self.assertGreater(checks,500);self.assertFalse(failures);self.assertFalse(errors)

 def test_birth_radius_and_clearance_remain_strict(self):
  with tempfile.TemporaryDirectory() as td:
   self.fixture(td);self.selected['selected']['clearance_lower_bound_m']=.599
   with self.assertRaises(ValueError):self.load()
   self.selected['selected']['clearance_lower_bound_m']=.62146;self.cert['radius_m']=.61;self.save()
   with self.assertRaises(ValueError):self.load()
