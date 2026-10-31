"""Explicit opt-in native plant/Controller rig; no original source or Query control modification."""
from pathlib import Path
import argparse,os,sys,json,subprocess,time,socket,signal,hashlib
ROOT=Path('/home/lizheng/query_minco_mainline');BASE=ROOT/'research_outputs/pg2_stagec_execution_v2';PY='/home/lizheng/anaconda3/envs/yopo/bin/python'
FAULTS=('publisher_exit','computation_timeout','stale','duplicate','future','owner_conflict','clock_reset_adapter','expired')
CASES={('v%03d_'%int(speed*1000)+fault):(11511+si*8+i,speed,fault,0,False) for si,speed in enumerate((.4,.8)) for i,fault in enumerate(FAULTS)}
CASES['v400_publisher_exit_r2']=(11540,.4,'publisher_exit',0,False)
CASES['v400_publisher_exit_r3']=(11541,.4,'publisher_exit',0,False)
CASES['v400_publisher_exit_r4']=(11542,.4,'publisher_exit',0,False)
CASES['v400_publisher_exit_r5']=(11543,.4,'publisher_exit',0,False)
CASES['v400_publisher_exit_r6']=(11544,.4,'publisher_exit',0,False)
CASES['domain_shadow']=(11533,.8,'publisher_exit',0,False)
CASES['domain_shadow_r2']=(11545,.8,'publisher_exit',0,False)
CASES.update(semantics_ready=(11531,.0,'publisher_exit',1,True),semantics_feedback=(11532,.0,'publisher_exit',0,True))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inventory(out):
    result=[];prefix='__log:='+str(out/'roslog')+'/'
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x];st=(p/'stat').read_text().split();uid=(p/'status').stat().st_uid
            if uid==os.getuid() and st[2]!='Z' and any(x.startswith(prefix) for x in argv):
                if not any(x in ' '.join(argv[:3]) for x in ['rosmaster','rosout','quadrotor_simulator_so3','network_control_node']):raise ValueError('UNEXPECTED_OWNED_ROS_CHILD')
                result.append(dict(pid=int(p.name),argv=argv,start=st[21]))
        except (FileNotFoundError,ProcessLookupError):pass
    return result

def cleanup(out,port):
    assert out.resolve().parent==BASE/'SYSTEM' and out.name in CASES
    actions=[]
    for sig,delay in [(signal.SIGINT,1.),(signal.SIGTERM,.5),(signal.SIGKILL,.2)]:
        for p in inventory(out):
            now={x['pid']:x for x in inventory(out)}.get(p['pid'])
            if now==p:os.kill(p['pid'],sig);actions.append(dict(pid=p['pid'],signal=int(sig),start=p['start']))
        end=time.monotonic()+delay
        while inventory(out) and time.monotonic()<end:time.sleep(.02)
        if not inventory(out):break
    s=socket.socket();s.settimeout(.2);closed=s.connect_ex(('127.0.0.1',port))!=0;s.close();r=dict(remaining=inventory(out),port_closed=closed,actions=actions);r['status']='PASS' if not r['remaining'] and closed else 'FAILED';(out/'CLEANUP.json').write_text(json.dumps(r,indent=2));assert r['status']=='PASS'

def run(name):
    source_identity=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),executed_files={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name('actor.py'),ROOT/'query_minco/integration/stagec_execution_v2/reference.py']})
    port,speed,fault,flag,semantics=CASES[name];folder=BASE/'SYSTEM';folder.mkdir(exist_ok=True);out=folder/name;out.mkdir(exist_ok=False)
    sock=socket.socket();sock.bind(('127.0.0.1',port));sock.close();protocol=json.loads((BASE/'PREREGISTRATION.json').read_text());(out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2));case=dict(case_id=name,scope=protocol['scope'],port=port,speed=speed,fault=fault,controller_flag=flag,semantics_probe=semantics);(out/'CASE.json').write_text(json.dumps(case,indent=2))
    (out/'EXECUTION_IDENTITY.json').write_text(json.dumps(source_identity,indent=2))
    # Only remap the independent downstream input; retain all original Controller equations/parameters.
    original=Path('/home/lizheng/YOPO/Controller/src/so3_quadrotor_simulator/launch/simulator_attitude_control.launch');launch=original.read_text().replace('to="/so3_control/pos_cmd"','to="/stagec/controller_ref"');(out/'RIG.launch').write_text(launch)
    identity=json.loads((ROOT/'research_outputs/pg3_minimal_system_v1/SYSTEM/BINARY_IDENTITIES.json').read_text());assert all(sha(p)==v for p,v in identity.items())
    raw=subprocess.check_output(['bash','-c','source /opt/ros/noetic/setup.bash && source /home/lizheng/YOPO/Controller/devel/setup.bash && source /home/lizheng/YOPO/Simulator/devel/setup.bash && env -0']);env=dict(x.decode().split('=',1) for x in raw.split(b'\0') if b'=' in x)
    env.update(ROS_MASTER_URI='http://127.0.0.1:'+str(port),ROS_IP='127.0.0.1',ROS_LOG_DIR=str(out/'roslog'),ROS_HOME=str(out/'roshome'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    env['CMAKE_PREFIX_PATH']='/home/lizheng/YOPO/Controller/devel:/home/lizheng/YOPO/Simulator/devel:/opt/ros/noetic';env['ROS_PACKAGE_PATH']='/home/lizheng/YOPO/Controller/src:/home/lizheng/YOPO/Simulator/src:/opt/ros/noetic/share';env['LD_LIBRARY_PATH']='/home/lizheng/YOPO/Controller/devel/lib:/home/lizheng/YOPO/Simulator/devel/lib:'+env.get('LD_LIBRARY_PATH','');env['PYTHONPATH']='/home/lizheng/YOPO/Controller/devel/lib/python3/dist-packages:'+env.get('PYTHONPATH','')
    prefix=['/usr/bin/bwrap','--die-with-parent','--ro-bind','/','/','--proc','/proc','--dev-bind','/dev','/dev','--bind',str(out),str(out),'--chdir',str(ROOT),'--']
    actor=str(ROOT/'experiments/pg2_stagec_execution_v2/actor.py');commands=[('master',['/opt/ros/noetic/bin/roscore','-p',str(port)]),('controller',['/opt/ros/noetic/bin/roslaunch',str(out/'RIG.launch')]),('gateway',[PY,'-B','-s','-u',actor,'gateway',str(out),'--enable-isolated-dynamics-test']),('source',[PY,'-B','-s','-u',actor,'source',str(out),'--enable-isolated-dynamics-test'])]
    if name.startswith('domain_shadow'):
        import yaml
        old=ROOT/'research_outputs/pg3_minimal_stageabc_v1/SYSTEM';sensor=yaml.safe_load((old/'moving_sensor.yaml').read_text());sensor.update(maze_type=6,room_number=1,x_length=12,y_length=12,z_length=6,add_ceiling=1,render_lidar=False)
        (out/'SENSOR.yaml').write_text(yaml.safe_dump(sensor,sort_keys=False))
        shadow=(old/'moving_pg2_shadow.py').read_text().replace('time.monotonic()-start<31','time.monotonic()-start<12');(out/'PG2_SHADOW_EXECUTED.py').write_text(shadow)
        commands += [('sensor',['/home/lizheng/YOPO/Simulator/devel/lib/sensor_simulator/sensor_simulator_cuda']),('camera',[PY,'-B','-s','-u','-m','query_minco.inference.shadow_camera30','--host',str(out)]),('shadow',[PY,'-B','-s','-u',str(out/'PG2_SHADOW_EXECUTED.py'),str(out)]),('volume',[PY,'-B','-s','-u',str(ROOT/'experiments/pg2_stagec_execution_v2/full_volume_shadow.py'),str(out)])]
    trace=[];files=[];ps=[];start=time.monotonic();error=None
    try:
        for kind,cmd in commands:
            f=(out/(kind+'.log')).open('x');files.append(f)
            if kind=='sensor':argv=prefix[:-3]+['--ro-bind',str(out/'SENSOR.yaml'),'/home/lizheng/YOPO/Simulator/src/config/config.yaml','--chdir','/home/lizheng/YOPO/Simulator','--']+cmd
            else:argv=prefix+cmd
            p=subprocess.Popen(argv,env=dict(env,CUDA_VISIBLE_DEVICES='0') if kind=='sensor' else env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);ps.append((kind,p));trace.append(dict(name=kind,pid=p.pid,argv=argv));time.sleep(1.5 if kind=='master' else .6)
            if p.poll() is not None:raise ValueError('EARLY_START_EXIT:'+kind)
        while time.monotonic()-start<45:
            if (out/'GATEWAY_RESULT.json').exists() and (not name.startswith('domain_shadow') or ((out/'MOVING_SHADOW_RESULT.json').exists() and (out/'FULL_VOLUME_RESULT.json').exists())):time.sleep(.1);break
            if any(p.poll() is not None and not (kind in ('source','shadow','gateway','volume') and p.returncode==0) for kind,p in ps):raise ValueError('EARLY_CHILD_EXIT')
            available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))/1024
            if available<3072:raise ValueError('AVAILABLE_MEMORY_GATE')
            time.sleep(.05)
        else:raise ValueError('WALL_BUDGET')
    except Exception as ex:error=repr(ex)
    finally:
        for kind,p in reversed(ps):
            if p.poll() is None:os.killpg(p.pid,signal.SIGINT)
        for kind,p in reversed(ps):
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=3)
        cleanup(out,port)
        for f in files:f.close()
        result=dict(status='EXECUTED' if error is None else 'FAILED',error=error,processes=trace,wall_s=time.monotonic()-start,original_launch_sha256=sha(original),binary_inputs=identity,Query_control=False,scope=case['scope'],source_identity=source_identity);(out/'RUN.json').write_text(json.dumps(result,indent=2))
    if error:raise ValueError(error)
    return json.loads((out/'GATEWAY_RESULT.json').read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--enable-isolated-dynamics-test',action='store_true');ap.add_argument('--case',choices=list(CASES));args=ap.parse_args()
    if not args.enable_isolated_dynamics_test:raise ValueError('DEFAULT_DISABLED')
    print(json.dumps(run(args.case or 'v400_publisher_exit')),flush=True)
if __name__=='__main__':main()
