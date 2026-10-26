"""Finite CPU contracts, real frozen weights, and fake-transport ROS integration."""
import ast
import contextlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from query_minco.inference import shadow as core
from query_minco.inference import ros_shadow as ros

class ShadowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pair=core.FrozenPair(core.sha(core.FROZEN))
        f=ROOT/'research_outputs/rsps_forest15m_v1/forest_paired001'
        with np.load(f/'inputs/seed31001_a0_f0.npz') as z:
            cls.depth=z['depth_input'].copy()
            cls.state=z['state'].copy()
        cls.output=cls.pair.forward(cls.depth,cls.state)

    def test_frozen_pair_original_C1_and_repeat(self):
        from query_minco.inference.planner_adapter import InferenceAdapter
        old=InferenceAdapter(self.pair.bindings[0]['checkpoint']['path'])
        ref=old.forward(self.depth[None],self.state[None])
        again=self.pair.forward(self.depth,self.state)
        for k in ref:
            np.testing.assert_array_equal(ref[k][0],self.output['C1'][k])
        for label in ['C1','E3']:
            for k in again[label]:
                np.testing.assert_array_equal(again[label][k],self.output[label][k])
            m=self.pair.models[label]
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in m.parameters()))
            self.assertEqual(core.state_sha(m),next(b['weights_sha256'] for b in self.pair.bindings if b['label']==label))
        self.assertRaisesRegex(ValueError,'FROZEN_BINDING_SHA',core.FrozenPair,'0'*64)

    def test_depth_encoding_stride_mask_and_camera(self):
        z=np.full((128,164),5,dtype='>f4')
        z[0,:4]=[0,20,np.nan,np.inf]
        d=core.decode_image(z.tobytes(),'32FC1',160,128,164*4,True)
        self.assertEqual(d.shape,(2,128,160))
        np.testing.assert_array_equal(d[:,0,:4],0)
        np.testing.assert_allclose(d[:,1,1],[.25,1])
        mm=np.full((128,160),5000,dtype='<u2')
        np.testing.assert_allclose(core.decode_image(mm.tobytes(),'16UC1',160,128,320,False)[:,1,1],[.25,1])
        self.assertRaises(ValueError,core.decode_image,z.tobytes(),'32FC1',160,90,656,True)
        core.camera_check(160,128,core.K,np.zeros(5),np.eye(3),np.c_[core.K,np.zeros(3)])
        self.assertRaises(ValueError,core.camera_check,160,90,core.K,[],np.eye(3),np.c_[core.K,np.zeros(3)])

    def test_goal_coordinates_terminal_and_unknown(self):
        R=core.rotation([0,0,np.sin(.4),np.cos(.4)])
        pos=np.array([3.,-2,2])
        state,g=core.make_state(pos,R,R@[.8,0,0],np.zeros(3),pos+R@[15,0,0])
        np.testing.assert_allclose(state,[.8,0,0,0,0,0,2.4,0,0,0,0,-1],atol=1e-6)
        self.assertEqual(g['ood'],[])
        self.assertRaisesRegex(ValueError,'DIRECTION_ENVELOPE',core.make_state,pos,R,np.zeros(3),np.zeros(3),pos+R@[15.,5.,0])
        self.assertRaisesRegex(ValueError,'TERMINAL',core.make_state,pos,R,np.zeros(3),np.zeros(3),pos+R@[2.,0,0])
        self.assertEqual(core.point_depth_label(np.zeros((2,128,160)),[2.4,0,0]),'UNKNOWN_NO_HIT')
        d=np.stack([np.ones((128,160))*.05,np.ones((128,160))])
        self.assertEqual(core.point_depth_label(d,[2.4,0,0]),'OCCLUDED_GOAL_OCCUPANCY_UNKNOWN')
        self.assertEqual(core.point_depth_label(d,[-1,0,0]),'UNKNOWN_BEHIND_CAMERA')
        self.assertRaises(ValueError,core.rotation,[0,0,0,0])

    def test_real_forest_inputs_and_nonzero_PVA_diagnostics(self):
        f=ROOT/'research_outputs/rsps_forest15m_v1/forest_paired001/inputs'
        for tid in ['seed31018_a0_f1','seed31020_a0_f1','seed31010_a0_f2']:
            with np.load(f/(tid+'.npz')) as z:
                d=z['depth_input'].copy();s=z['state'].copy()
            s[:6]=[1.1,.1,-.05,.2,-.1,.05]
            out=self.pair.forward(d,s)
            for label,data in out.items():
                self.assertTrue(data['solve_valid'].all())
                np.testing.assert_allclose(data['head_pva'][:,1],np.tile(s[:3],(6,1)),atol=1e-6)
                np.testing.assert_allclose(data['head_pva'][:,2],np.tile(s[3:6],(6,1)),atol=1e-6)
        for b in self.pair.bindings:
            self.assertEqual(core.state_sha(self.pair.models[b['label']]),b['weights_sha256'])

    def test_publisher_topology_rejects_takeover(self):
        good=[('/sim/odom',['/quadrotor_simulator_so3']),('/so3_control/pos_cmd',['/yopo_net']),('/query_shadow/candidates',['/query_minco_shadow'])]
        ros.topology_check(good)
        self.assertRaises(ValueError,ros.topology_check,[('/so3_control/pos_cmd',['/query_minco_shadow'])])
        self.assertRaises(ValueError,ros.topology_check,good+[('/evil_control',['/query_minco_shadow'])])
        tree=ast.parse(Path(ros.__file__).read_text())
        calls=[x for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='Publisher']
        self.assertEqual(len(calls),2)
        self.assertEqual({x.args[1].id for x in calls},{'String','MarkerArray'})

    def test_checker_rebuilds_actual_outputs_and_rejects_seal_addition(self):
        import importlib.util, contextlib, io
        path=ROOT/'experiments/risk_aware_set_physics_v1/forest_shadow_v1.py'
        spec=importlib.util.spec_from_file_location('checker_test',path)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        parent=ROOT/'research_outputs/rsps_forest_shadow_v1/preparation001/runtime'
        run=Path(tempfile.mkdtemp(prefix='CPU_CHECK_FIXTURE_',dir=parent))
        sensor=run/'sensor.yaml';sensor.write_text('CPU_ONLY_SYNTHETIC_TRANSPORT: true\n')
        binding=json.loads((mod.PREP/'configs/EXECUTION_BINDING.json').read_text())
        ros.dump(run/'BINDING.json',binding)
        ros.dump(run/'RUN_SPEC.json',dict(frozen_models_sha=core.sha(core.FROZEN),sensor_config_sha=core.sha(sensor),CPU_ONLY_SYNTHETIC_TRANSPORT=True))
        ros.dump(run/'START.json',dict(mode='SHADOW_ONLY',models=self.pair.bindings,control=False,training=False,CPU_ONLY_SYNTHETIC_TRANSPORT=True))
        ros.dump(run/'LAUNCHER_EXIT.json',dict(returncode=0))
        ros.dump(run/'PROCESS_EXIT.json',dict(bridge_exit=0,sensor_exited_early=False,resource_violation=False))
        ros.dump(run/'RESULT.json',dict(status='RUNTIME_COMPLETE',control=False,counters=dict(accepted=50)))
        top=dict(type='TOPOLOGY',shadow_publishers=list(ros.TOPICS.values()),control_publishers={'/so3_control/pos_cmd':['/yopo_net']},odom_publishers=['/quadrotor_simulator_so3'])
        events=[top,top];cache=[]
        for j in range(3):
            d=self.depth.copy();d[0][d[1]>0]*=(1-j*.001)
            cache.append((d,self.pair.forward(d,self.state)))
        for i in range(50):
            d,out=cache[i%3];stamp_ns=100_000_000_000+i*200_000_000
            arrays=dict(depth_input=d,state=self.state,stamp=np.array(stamp_ns/1e9),stamp_ns=np.array(stamp_ns))
            for label,data in out.items():arrays.update({label+'_'+k:v for k,v in data.items()})
            with (run/('frame_%06d.npz'%i)).open('xb') as f:np.savez_compressed(f,**arrays)
            events.append(dict(type='FRAME',index=i,stamp=stamp_ns/1e9,stamp_ns=stamp_ns,input_age_s=.02,output_age_s=.04,generation_s=.01,selected_query=None,online_safe_status='UNKNOWN_CPU_FIXTURE_ONLY'))
        (run/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
        with patch.object(mod,'OUTROOT',parent),contextlib.redirect_stdout(io.StringIO()):mod.check(run)
        self.assertTrue((run/'SHADOW_RUN_SEAL.json').is_file())
        (run/'ADDED_AFTER_SEAL.txt').write_text('Intentional CPU-only tamper fixture. Not a live ROS run.\n')
        with patch.object(mod,'OUTROOT',parent):
            self.assertRaisesRegex(ValueError,'RUN_SEAL_MEMBERSHIP',mod.check,run)

    def test_fake_transport_real_outputs_timestamps_and_readonly_publishers(self):
        # No ROS master/socket/node is created. Real model outputs from setUpClass.
        ns=types.SimpleNamespace
        now=[100.]
        class Stamp:
            def __init__(self,t):self.t=t
            def to_sec(self):return self.t
            def to_nsec(self):return int(round(self.t*1e9))
        messages=[]
        for i in range(3):
            h=ns(stamp=Stamp(100+i*.2),frame_id='camera_link')
            d=ns(header=h,data=np.full((128,160),5+i*.01,dtype='<f4').tobytes(),
                 encoding='32FC1',height=128,width=160,step=640,is_bigendian=False)
            c=ns(header=h,width=160,height=128,K=core.K.reshape(-1),D=[0]*5,R=np.eye(3).reshape(-1),
                 P=np.c_[core.K,np.zeros(3)].reshape(-1),binning_x=0,binning_y=0,
                 roi=ns(x_offset=0,y_offset=0,height=0,width=0))
            o=ns(header=ns(stamp=h.stamp,frame_id='world'),child_frame_id='/quadrotor',
                 pose=ns(pose=ns(position=ns(x=i*.16,y=0,z=2),orientation=ns(x=0,y=0,z=0,w=1))),
                 twist=ns(twist=ns(linear=ns(x=.8,y=0,z=0))))
            messages.append((d,c,o))
        class Frames:
            def __init__(self,**kw):pass
            def get(self,**kw):
                m=messages.pop(0);now[0]=m[0].header.stamp.to_sec()
                return (*m,ros.time.monotonic())
        pubs={}
        class Pub:
            def __init__(self,t,*a,**kw):self.t=t;pubs[t]=[]
            def publish(self,x):pubs[self.t].append(x)
        class Marker:
            LINE_STRIP=4;ADD=0
            def __init__(self):
                self.header=ns();self.pose=ns(orientation=ns());self.scale=ns();self.color=ns()
        class MA:
            def __init__(self):self.markers=[]
        rospy=ns(init_node=lambda *a,**kw:None,get_name=lambda:'/query_minco_shadow',get_param=lambda n:2. if n.endswith('_z') else 0.,Publisher=Pub,is_shutdown=lambda:False,
                 Time=ns(now=lambda:Stamp(now[0])),Duration=lambda x:x,signal_shutdown=lambda x:None)
        mf=ns(Subscriber=lambda *a,**kw:ns(sub=ns(unregister=lambda:None)),
              TimeSynchronizer=lambda *a:ns(registerCallback=lambda f:None))
        modules={'rospy':rospy,'message_filters':mf,'rosgraph':ns(Master=lambda n:ns(getSystemState=lambda:([('/so3_control/pos_cmd',['/yopo_net']),('/sim/odom',['/quadrotor_simulator_so3'])],[],[]))),
                 'sensor_msgs.msg':ns(Image=object,CameraInfo=object),'nav_msgs.msg':ns(Odometry=object),
                 'std_msgs.msg':ns(String=lambda **kw:kw),'visualization_msgs.msg':ns(Marker=Marker,MarkerArray=MA),
                 'geometry_msgs.msg':ns(Point=lambda *xyz:xyz)}
        fake=ns(bindings=self.pair.bindings,forward=lambda d,s:self.output)
        with contextlib.nullcontext(tempfile.mkdtemp(dir=ROOT/'research_outputs/rsps_forest_shadow_v1/preparation001/runtime')) as tmp:
            run=Path(tmp)
            with patch.dict(sys.modules,modules),patch.object(ros,'OUTROOT',run.parent),patch.object(ros,'FrozenPair',return_value=fake),patch.object(ros.queue,'Queue',Frames),patch.object(sys,'argv',['shadow','--run',str(run),'--binding-sha','unused','--max-frames','2']):
                ros.main()
            result=json.loads((run/'RESULT.json').read_text())
            self.assertEqual(result['counters']['accepted'],2)
            self.assertEqual(result['counters']['rejected'],1) # acceleration warmup is preserved
            self.assertEqual(set(pubs),set(ros.TOPICS.values()))
            self.assertEqual(len(pubs[ros.TOPICS['candidates']][0].markers),12)
            with np.load(run/'frame_000000.npz') as z:
                self.assertAlmostEqual(float(z['stamp']),100.2)
                np.testing.assert_allclose(z['state'][:6],[.8,0,0,0,0,0],atol=1e-6)

if __name__=='__main__':
    unittest.main(verbosity=2)
