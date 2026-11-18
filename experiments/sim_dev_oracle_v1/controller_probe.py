"""Real independent SO3 nodelet consumption probe: NO Plant, ARM or motors topic."""
from pathlib import Path
import sys,os,json,time,socket,signal,subprocess,hashlib,xmlrpc.client
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));O=R/'research_outputs/sim_dev_oracle_20261118_v1/controller_probe_v3'

def main():
 O.mkdir();from experiments.f0_controller_consumption_v1.prepare import environment
 env=environment()
 with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
 env.update(ROS_DISTRO='noetic',ROS_VERSION='1',ROS_PYTHON_VERSION='3',ROS_ROOT='/opt/ros/noetic/share/ros',ROS_MASTER_URI='http://127.0.0.1:%d'%port,ROS_IP='127.0.0.1',ROS_HOME=str(O/'ros_home'),ROS_LOG_DIR=str(O/'ros_logs'),PYTHONPATH=str(R)+':'+env['PYTHONPATH']);os.environ.update(env);sys.path[:0]=env['PYTHONPATH'].split(':');owned=[];commands=[]
 def spawn(name,argv):
  so=(O/(name+'.stdout.log')).open('x');se=(O/(name+'.stderr.log')).open('x');p=subprocess.Popen(argv,cwd=R,env=env,stdout=so,stderr=se,start_new_session=True);owned.append((p,so,se));commands.append(dict(name=name,argv=argv,pid=p.pid));return p
 def stop(p):
  if p.poll() is None:os.killpg(p.pid,signal.SIGINT)
  try:return p.wait(timeout=8)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait();raise RuntimeError('PROBE_FORCED_CLEANUP')
 try:
  master=spawn('master',['/opt/ros/noetic/bin/roscore','-p',str(port)]);rpc=xmlrpc.client.ServerProxy(env['ROS_MASTER_URI'])
  for _ in range(120):
   try:
    if rpc.getPid('/sim_dev_probe')[0]==1:break
   except OSError:pass
   time.sleep(.05)
  else:raise RuntimeError('PROBE_MASTER_TIMEOUT')
  import rospy,numpy as np,torch
  from nav_msgs.msg import Odometry
  from sensor_msgs.msg import Imu
  from quadrotor_msgs.msg import PositionCommand,SO3Command
  from query_minco.integration.yopo_execution_v4.trajectory import build
  from query_minco.integration.yopo_shadow_v1.command_contract import position_command
  from experiments.f0_controller_consumption_v1.manual import verify_controller
  torch.set_num_threads(1);rospy.init_node('p2_execution_supervisor',disable_signals=True);nonce='abcdef0123456789abcdef0123456789'
  odompub=rospy.Publisher('/p2/odom',Odometry,queue_size=1);imupub=rospy.Publisher('/p2/imu',Imu,queue_size=1);cmdpub=rospy.Publisher('/p2/position_cmd',PositionCommand,queue_size=1);unusedfinal=rospy.Publisher('/p2/so3_cmd',SO3Command,queue_size=1);outputs=[]
  def output(m):outputs.append(dict(stamp_ns=m.header.stamp.to_nsec(),force=[m.force.x,m.force.y,m.force.z],q=[m.orientation.x,m.orientation.y,m.orientation.z,m.orientation.w],received_monotonic=time.monotonic()))
  sub=rospy.Subscriber('/p2/so3_raw',SO3Command,output,queue_size=1000)
  name='f0_audit_position_controller';rospy.set_param('/'+name,dict(mass=.98,record_log=True,PID_logger_file_name=str(O/'CONSUMPTION.jsonl'),audit_run_nonce=nonce,so3_control=dict(init_state_x=.25,init_state_y=-.25,init_state_z=-10000.),use_external_yaw=False,gains=dict(rot=dict(x=1.5,y=1.5,z=1.),ang=dict(x=.13,y=.13,z=.1),kx=dict(x=5.7,y=5.7,z=6.2),kv=dict(x=3.4,y=3.4,z=4.))))
  controller=spawn('controller',['/opt/ros/noetic/lib/nodelet/nodelet','standalone','f0_controller_audit_pkg/AuditSO3ControlNodelet','__name:='+name,'~odom:=/p2/odom','~imu:=/p2/imu','~position_cmd:=/p2/position_cmd','~so3_cmd:=/p2/so3_raw'])
  for _ in range(120):
   if cmdpub.get_num_connections()==1 and odompub.get_num_connections()==1 and sub.get_num_connections()==1:break
   if controller.poll() is not None:raise RuntimeError('PROBE_CONTROLLER_EXIT')
   time.sleep(.05)
  else:raise RuntimeError('PROBE_CONTROLLER_CONNECT_TIMEOUT')
  identity=verify_controller(env['ROS_MASTER_URI'],nonce);(O/'LOADED_IDENTITY.json').write_text(json.dumps(identity,indent=2));(O/'PROCESS_MAPS.txt').write_text(Path('/proc/%d/maps'%controller.pid).read_text())
  base=rospy.Time.now().to_sec()+.25;anchor=np.array([.25,-.25,2.]);A=build(anchor,np.zeros(3),np.zeros(3),[0,0,0,1],anchor+[.6,0,0],base);head=A.sample(.2);B=build(*head,[0,0,0,1],head[0]+[.6,0,0],base+.2);plans={A.id:A.record,B.id:B.record};inputs=[]
  while rospy.Time.now().to_sec()<base:time.sleep(.001)
  until=base+.45;counter=0
  while rospy.Time.now().to_sec()<until:
   now=rospy.Time.now().to_sec();plan=A if now<base+.2 else B;pva=plan.at(now);odom=Odometry();odom.header.frame_id='world';odom.header.stamp=rospy.Time.from_sec(now);odom.pose.pose.position.x,odom.pose.pose.position.y,odom.pose.pose.position.z=pva[0];odom.pose.pose.orientation.w=1.;odom.twist.twist.linear.x,odom.twist.twist.linear.y,odom.twist.twist.linear.z=pva[1];imu=Imu();imu.header.stamp=odom.header.stamp;imu.linear_acceleration.x,imu.linear_acceleration.y,imu.linear_acceleration.z=pva[2]
   odompub.publish(odom);imupub.publish(imu);counter+=1;msg=position_command(PositionCommand,pva,odom.header.stamp,0.,counter);msg.kx=[0,0,0];msg.kv=[0,0,0];cmdpub.publish(msg);inputs.append(dict(trajectory_id=counter,input_ns=msg.header.stamp.to_nsec(),plan_id=plan.id,pva=pva.reshape(-1).tolist(),nonce=nonce));time.sleep(.01)
  time.sleep(.1);graph=rpc.getSystemState('/sim_dev_probe');stop(controller);time.sleep(.05)
  rows=[json.loads(x) for x in (O/'CONSUMPTION.jsonl').read_text().splitlines()];consumed=[x for x in rows if x.get('kind')=='COMMAND_CONSUMED'];computed=[x for x in rows if x.get('kind')=='CONTROL_OUTPUT'];mapping={x['trajectory_id']:x for x in inputs};seen=[]
  for c in consumed:
   item=mapping[c['trajectory_id']];assert rows[0]['nonce']==nonce and c['input_ns']==item['input_ns'] and np.allclose(c['pva'],item['pva'],atol=1e-9,rtol=0);seen.append(item['plan_id'])
  assert len(consumed)==len(inputs) and len(set(seen))==2 and any(a!=b for a,b in zip(seen,seen[1:]));assert rows[-1]['kind']=='AUDIT_CLOSE' and rows[-1]['lost']==0 and rows[-1]['write_failed'] is False
  output_ns={x['stamp_ns'] for x in outputs};assert all(x['output_ns'] in output_ns for x in computed)
  (O/'INPUT_COMMANDS.json').write_text(json.dumps(dict(plans=plans,inputs=inputs),indent=2));(O/'OUTPUT_SO3.json').write_text(json.dumps(outputs,indent=2));result=dict(status='PASS_NONDRIVING_CONTROLLER_ONLY',mode='SIM_DEV_ORACLE',commands=len(inputs),consumed=len(consumed),control_outputs=len(computed),distinct_consumed_plan_ids=2,switch_proven=True,controller_identity=identity,physical_Plant_steps=0,ARM_publishers=0,final_SO3_publications=0,Plant_causal_evidence='NOT_RUN',live_graph=graph)
  (O/'RESULT.json').write_text(json.dumps(result,indent=2));print('CONTROLLER_PROBE',result['status'],len(inputs),len(computed));sub.unregister();rospy.signal_shutdown('probe done')
 finally:
  for child,so,se in reversed(owned):
   code=stop(child);so.close();se.close()
   for row in commands:
    if row['pid']==child.pid:row['exit_code']=code
  (O/'COMMANDS.json').write_text(json.dumps(commands,indent=2))
if __name__=='__main__':main()
