"""Targeted nonphysical checks of the generated 10m software."""
import inspect
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from threading import RLock
from tempfile import TemporaryDirectory
import unittest
from .startup import Startup
from .audit import AsyncJournal
from .prepare import O
from .reference import source, config

class Tests(unittest.TestCase):
    def fixture(self):
        mono=[0.]
        engine=SimpleNamespace(lock=RLock(),arm_sent=False,committed='old',arm_request_stamp=99.)
        r=Startup(20.,engine,lambda:mono[0],lambda *a:None)
        r.mode='ACK'
        r.last_sent=99.9
        r.sent={99.9}
        r.acks={99.9}
        p=dict(phase='ACK',integration_steps=0,final_command_stamp=99.9,final_count=8,command_valid=True)
        r.samples['plant']=(100.,100.,p)
        r.samples['imu']=(99.7,99.7,{})
        return r,engine,mono,p

    def test_short_gap_needs_new_plant_ack(self):
        r,e,mono,p=self.fixture()
        self.assertEqual(r.tick(100.),'PREPARE')
        self.assertIsNotNone(r.wait)
        self.assertIsNone(e.committed)
        self.assertIsNone(e.arm_request_stamp)
        self.assertFalse(r.forward_allowed(100.))
        for k in ['odom','imu','depth','request','raw','plant']:
            r.counts[k]=5
            r.samples[k]=(100.01,100.01,dict(pva=[[.25,-.25,2.],[0,0,0],[0,0,0]]) if k=='request' else p)
        self.assertEqual(r.tick(100.02),'PREPARE')
        self.assertFalse(r.acks)
        for i in range(5):
            stamp=100.03+i*.01
            r.forwarded(stamp)
            p['final_command_stamp']=stamp
            r.samples['plant']=(stamp,stamp,p)
            r.tick(stamp)
        self.assertEqual(r.mode,'ACK')
        self.assertEqual(len(r.acks),5)

    def test_timeout_and_armed_staleness(self):
        r,e,mono,p=self.fixture()
        r.tick(100.)
        mono[0]=20.
        self.assertEqual(r.tick(100.),'ABORT')
        self.assertEqual(r.reason,'STARTUP_SENSOR_WAIT_TIMEOUT')
        r,e,mono,p=self.fixture()
        r.mode='ARMED'
        self.assertEqual(r.tick(100.),'ABORT')
        self.assertEqual(r.reason,'imu_STALE')
        r,e,mono,p=self.fixture()
        e.arm_sent=True
        self.assertEqual(r.tick(100.),'ABORT')

    def test_bound_goal_in_reference_and_engine(self):
        from query_minco.integration.depth_minco_t2_v1.ros_bridge import reference_main
        cfg=json.loads((O/'runtime_bundle_v2/runtime_v2/SCENES.json').read_text())['forest']['forest10m']
        text=source(inspect.getsource(reference_main))
        compile(text,'<reference>','exec')
        self.assertIn("target=config(run)['goal']",text)
        with TemporaryDirectory() as tmp:
            run=Path(tmp)/'none';run.mkdir()
            (run.parent/'SCENE_IDENTITY.json').write_text(json.dumps(dict(scene=dict(forest10m=cfg))))
            goal=config(run)['goal']
            self.assertEqual(goal,[10.5,.75,2.])
        file=O/'runtime_bundle_v2/generated_v2/bridge.py'
        spec=importlib.util.spec_from_file_location('forest10m_generated_bridge',file)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        e=m.Engine.__new__(m.Engine)
        e.clock=lambda:100.;e.goal_config=goal;e.lock=RLock();e.record=lambda *a,**kw:None
        e.set_goal(goal,100.)
        self.assertEqual(e.goal,(goal,100.))
        with self.assertRaisesRegex(ValueError,'SIM_GOAL_COORDINATES'):
            e.set_goal([3.25,-.25,2.],100.)

    def test_async_write_and_display_namespace(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'audit.jsonl'
            j=AsyncJournal(path)
            with j.lock:
                self.assertTrue(j.append(dict(kind='imu',stamp=100.)))
                self.assertTrue(j.append(dict(kind='ARM',stamp=101.)))
            self.assertEqual(j.close(),[])
            self.assertEqual([json.loads(x)['kind'] for x in path.read_text().splitlines()],['imu','ARM','JOURNAL_CLOSED'])
        display=Path(__file__).with_name('display.py').read_text()
        self.assertIn("prefix='/forest10m/display/'",display)
        self.assertNotIn('/p2/',display)
