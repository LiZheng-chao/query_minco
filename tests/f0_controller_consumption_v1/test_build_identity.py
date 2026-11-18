"""Static and isolated ROS diagnostics. Never launches a Plant or ARM publisher."""
from pathlib import Path
import sys,os,json,subprocess,time,signal,socket,xml.etree.ElementTree as ET,unittest,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.f0_controller_consumption_v1.prepare import OUT,UP,PKG,environment,sha

def function(text,name):
    start=text.index('::'+name+'(');start=text.index('{',start);depth=1;i=start+1
    while depth:
        if text[i]=='{':depth+=1
        if text[i]=='}':depth-=1
        i+=1
    return text[start:i]
class Build(unittest.TestCase):
    def test_distinct_plugin(self):
        xml=ET.parse(PKG/'nodelet_plugin.xml').getroot();c=xml.find('class')
        self.assertEqual(c.attrib['type'],'F0AuditSO3ControlNodelet')
        self.assertEqual(c.attrib['name'],'f0_controller_audit_pkg/AuditSO3ControlNodelet')
        self.assertEqual(xml.attrib['path'],'lib/libf0_audit_nodelet')
    def test_control_and_callback_body_preserved(self):
        original=(UP/'src/so3_control_nodelet.cpp').read_text();generated=(OUT/'generated/audit_nodelet_v2.cpp').read_text()
        for name in ['publishSO3Command','position_cmd_callback','odom_callback','imu_callback','corrections_callback','enable_motors_callback']:
            old=function(original,name);new=function(generated,name)
            new=new.replace('  const double audit_compute_begin = f0audit::mono();\n','').replace('  audit_.output(*so3_command, audit_compute_begin);\n','').replace('  audit_.consume(*cmd, kx_, kv_);\n','')
            self.assertEqual(old,new,name)
    def test_core_numerical_reference(self):
        env=environment()
        a=subprocess.check_output([str(OUT/'build_v2/original_control_equivalence')],env=env)
        b=subprocess.check_output([str(OUT/'devel_v2/lib/f0_controller_audit_pkg/f0_control_equivalence')],env=env)
        (OUT/'audit/original_core.txt').write_bytes(a);(OUT/'audit/diagnostic_core.txt').write_bytes(b)
        self.assertEqual(a,b,'512 inputs: exact force/quaternion output equality required')
    def test_build_identity(self):
        d=json.loads((OUT/'BUILD_IDENTITY_v2.json').read_text());self.assertEqual(sha(Path(d['library'])),d['library_sha256'])

def bench():
    env=environment()
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    env.update(ROS_MASTER_URI='http://127.0.0.1:%d'%port,ROS_IP='127.0.0.1')
    os.environ.update(env)
    bench_id=sys.argv[sys.argv.index('--bench-id')+1] if '--bench-id' in sys.argv else 'ros_bench'
    if not bench_id.replace('_','').isalnum():raise ValueError('BENCH_ID')
    logs=OUT/'audit'/bench_id;logs.mkdir()
    children=[];commands=[]
    def spawn(label,argv):
        so=(logs/(label+'.stdout.log')).open('x');se=(logs/(label+'.stderr.log')).open('x')
        p=subprocess.Popen(argv,env=env,cwd=ROOT,stdout=so,stderr=se,start_new_session=True)
        children.append((p,so,se));commands.append(dict(label=label,argv=argv,pid=p.pid));return p
    def stop(p):
        if p.poll() is None:os.killpg(p.pid,signal.SIGINT)
        try:return p.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);return p.wait(timeout=3)
    master=spawn('master',['/opt/ros/noetic/bin/roscore','-p',str(port)])
    import xmlrpc.client
    rpc=xmlrpc.client.ServerProxy(env['ROS_MASTER_URI'])
    try:
        for _ in range(100):
            try:
                if rpc.getPid('/f0_bench')[0]==1:break
            except OSError:pass
            time.sleep(.05)
        else:raise RuntimeError('ISOLATED_MASTER_TIMEOUT')
        import rospy
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Imu
        from quadrotor_msgs.msg import PositionCommand,SO3Command
        rospy.init_node('p2_execution_supervisor',disable_signals=True)
        pubs=dict(odom=rospy.Publisher('/p2/odom',Odometry,queue_size=1),
                  imu=rospy.Publisher('/p2/imu',Imu,queue_size=1),
                  cmd=rospy.Publisher('/p2/position_cmd',PositionCommand,queue_size=1),
                  final=rospy.Publisher('/p2/so3_cmd',SO3Command,queue_size=1))
        outputs=[]
        def received(m):
            outputs.append(dict(stamp_ns=m.header.stamp.to_nsec(),force=[m.force.x,m.force.y,m.force.z],
                orientation=[m.orientation.x,m.orientation.y,m.orientation.z,m.orientation.w],kr=list(m.kR),kom=list(m.kOm),
                aux=[m.aux.current_yaw,m.aux.kf_correction,*m.aux.angle_corrections,float(m.aux.enable_motors),float(m.aux.use_external_yaw),0.]))
        sub=rospy.Subscriber('/p2/so3_raw',SO3Command,received,queue_size=1000)
        results={};nonce='a'*32
        from experiments.f0_controller_consumption_v1.manual import verify_controller
        for variant in ['original','diagnostic_off','diagnostic_on']:
            name='f0_baseline_controller' if variant=='original' else 'f0_audit_position_controller'
            plugin='so3_control/SO3ControlNodelet' if variant=='original' else 'f0_controller_audit_pkg/AuditSO3ControlNodelet'
            rospy.set_param('/'+name,dict(mass=.98,record_log=variant=='diagnostic_on',
                PID_logger_file_name=str(logs/'CONSUMPTION.jsonl'),audit_run_nonce=nonce,
                so3_control=dict(init_state_x=.25,init_state_y=-.25,init_state_z=-10000.),
                use_external_yaw=False,gains=dict(rot=dict(x=1.5,y=1.5,z=1.),ang=dict(x=.13,y=.13,z=.1),kx=dict(x=5.7,y=5.7,z=6.2),kv=dict(x=3.4,y=3.4,z=4.))))
            p=spawn(variant,['/opt/ros/noetic/lib/nodelet/nodelet','standalone',plugin,'__name:='+name,
                '~odom:=/p2/odom','~imu:=/p2/imu','~position_cmd:=/p2/position_cmd','~so3_cmd:=/p2/so3_raw'])
            for _ in range(100):
                if pubs['cmd'].get_num_connections()==1 and pubs['odom'].get_num_connections()==1 and sub.get_num_connections()==1:break
                if p.poll() is not None:raise RuntimeError('NODELET_EXIT '+variant)
                time.sleep(.05)
            else:raise RuntimeError('CONTROLLER_CONNECTION_TIMEOUT '+variant)
            if variant!='original':
                ident=verify_controller(env['ROS_MASTER_URI'],nonce)
                (logs/(variant+'.LOADED_IDENTITY.json')).write_text(json.dumps(ident,indent=2)+'\n')
            maps=Path('/proc/%d/maps'%p.pid).read_text();(logs/(variant+'.maps')).write_text(maps)
            outputs.clear();inputs=[];latencies=[];counts=[]
            imu=Imu();imu.linear_acceleration.z=0.;pubs['imu'].publish(imu)
            odom=Odometry();odom.header.frame_id='world';odom.pose.pose.position.x=.25;odom.pose.pose.position.y=-.25;odom.pose.pose.position.z=2.;odom.pose.pose.orientation.w=1.
            odom.header.stamp=rospy.Time.now();pubs['odom'].publish(odom);time.sleep(.03)
            if outputs:raise AssertionError('STARTUP_WITH_DISABLED_HOLD_OUTPUT')
            def await_count(n):
                until=time.monotonic()+2
                while len(outputs)<n and time.monotonic()<until:time.sleep(.0005)
                if len(outputs)!=n:raise AssertionError('OUTPUT_COUNT %s expected %d got %d'%(variant,n,len(outputs)))
            for i in range(60):
                c=PositionCommand();c.header.frame_id='world';c.header.stamp=rospy.Time.now();c.trajectory_id=i+1;c.trajectory_flag=1
                c.position.x=.25+.28*(i/59.);c.position.y=-.25;c.position.z=2.
                c.velocity.x=.03;c.acceleration.x=.01;c.yaw=.1*(i/59.)
                # Zero overrides must retain identical effective gains in both implementations.
                inputs.append(dict(trajectory_id=i+1,input_ns=c.header.stamp.to_nsec(),pva=[c.position.x,c.position.y,c.position.z,c.velocity.x,0,0,c.acceleration.x,0,0]))
                t=time.perf_counter();pubs['cmd'].publish(c);await_count(i*2+1);latencies.append(time.perf_counter()-t)
                odom.header.stamp=rospy.Time.now();pubs['odom'].publish(odom);time.sleep(.003)
                if len(outputs)!=i*2+1:raise AssertionError('FIRST_ODOM_CHANGED_TRIGGER')
                odom.header.stamp=rospy.Time.now();pubs['odom'].publish(odom);await_count(i*2+2)
                counts.append(len(outputs));time.sleep(.004)
            code=stop(p);time.sleep(.05)
            results[variant]=dict(output=list(outputs),inputs=inputs,latency_s=latencies,counts=counts,exit_code=code)
            (logs/(variant+'.REPLAY.json')).write_text(json.dumps(results[variant],indent=2)+'\n')
            for _ in range(100):
                if pubs['cmd'].get_num_connections()==0:break
                time.sleep(.02)
        # Every dynamic field, callback-triggered publication count and order must match exactly.
        normalize=lambda rows:[{k:v for k,v in x.items() if k!='stamp_ns'} for x in rows]
        for variant in ['diagnostic_off','diagnostic_on']:
            assert normalize(results['original']['output'])==normalize(results[variant]['output']),variant
            assert results['original']['counts']==results[variant]['counts'],variant
        rows=[json.loads(x) for x in (logs/'CONSUMPTION.jsonl').read_text().splitlines()]
        assert rows[-1]['kind']=='AUDIT_CLOSE' and rows[-1]['lost']==0 and rows[-1]['write_failed'] is False,rows[-1]
        assert rows[-1]['records']==len(rows)-2
        assert [x['index'] for x in rows[1:-1]]==list(range(1,len(rows)-1))
        consumed=[x for x in rows if x['kind']=='COMMAND_CONSUMED'];computed=[x for x in rows if x['kind']=='CONTROL_OUTPUT']
        assert len(consumed)==60 and len(computed)==120
        for i,c in enumerate(consumed):
            assert c['trajectory_id']==i+1 and c['pva']==results['diagnostic_on']['inputs'][i]['pva']
            pair=computed[2*i:2*i+2];assert all(x['command_index']==c['command_index'] for x in pair)
        assert [x['output_ns'] for x in computed]==[x['stamp_ns'] for x in results['diagnostic_on']['output']]
        stats=lambda values:dict(n=len(values),mean_s=sum(values)/len(values),max_s=max(values),p95_s=sorted(values)[int(.95*(len(values)-1))])
        report=dict(status='PASS_NONDRIVING_ROS_NO_PLANT_NO_ARM',same_inputs=60,outputs_each=120,
            force_orientation_gain_aux_exact_equal=True,publication_order_equal=True,
            latency={k:stats(v['latency_s']) for k,v in results.items()},
            compute=stats([x['compute_s'] for x in computed]),enqueue=stats([x['enqueue_s'] for x in computed+consumed]),
            log_close=rows[-1],physical_feedback='NOT_RUN',physical_frequency_qualification='NOT_RUN')
        (logs/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
        sub.unregister();rospy.signal_shutdown('bench complete')
    finally:
        for child,so,se in reversed(children):
            code=stop(child);so.close();se.close()
            for row in commands:
                if row['pid']==child.pid:row['exit_code']=code
        (logs/'COMMANDS.json').write_text(json.dumps(commands,indent=2)+'\n')
if __name__=='__main__':
    if '--ros-bench' in sys.argv:bench()
    else:unittest.main()
