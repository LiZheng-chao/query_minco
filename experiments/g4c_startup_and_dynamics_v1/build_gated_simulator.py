"""Deterministic minimal overlay of read-only native simulator;compile only."""
from pathlib import Path
import sys,json,hashlib,difflib,shlex,os
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.audit.run_provenance_v1 import run_logged
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g4c_startup_and_dynamics_20261111_v1';Y=R.parent/'YOPO/Controller'
def main():
 b=O/'startup_build_v2';b.mkdir();src=Y/'src/so3_quadrotor_simulator/src/quadrotor_simulator_so3.cpp';old=src.read_text();s=old
 insert='#include <std_msgs/String.h>\n#include <sstream>\n#include "plant_gate.hpp"\nstatic g4c::PlantGate startup_gate;\nstatic void start_permit_cb(const std_msgs::String::ConstPtr& m){startup_gate.permit(m->data,ros::Time::now().toSec());}\n'
 s=s.replace('typedef struct _Control',insert+'\ntypedef struct _Control',1)
 anchor='  command.force[0]         = cmd->force.x;'
 hook='  startup_gate.accept(cmd->header.stamp.toSec(),ros::Time::now().toSec(),cmd->force.x,cmd->force.y,cmd->force.z,std::sqrt(cmd->orientation.x*cmd->orientation.x+cmd->orientation.y*cmd->orientation.y+cmd->orientation.z*cmd->orientation.z+cmd->orientation.w*cmd->orientation.w));\n'
 assert s.count(anchor)==1;s=s.replace(anchor,hook+anchor)
 anchor='  ros::NodeHandle n("~");';s=s.replace(anchor,anchor+'\n  n.param("run_nonce",startup_gate.nonce,std::string(""));\n  if(startup_gate.nonce.empty()){ROS_ERROR("EXPLICIT_RUN_NONCE_REQUIRED");return 2;}\n  auto start_sub=n.subscribe("start_permit",1,&start_permit_cb);\n  auto phase_pub=n.advertise<std_msgs::String>("startup_state",10);')
 s=s.replace('  QuadrotorSimulator::Quadrotor quad;','  QuadrotorSimulator::Quadrotor quad;\n  startup_gate.hover=quad.getMass()*quad.getGravity();')
 s=s.replace('  Control control;','  Control control = {};')
 first=s.index('    auto last = control;');last=s.index('    ros::Time tnow =',first)
 original_block=s[first:last]
 newblock='''    const double control_now=ros::Time::now().toSec();
    if(startup_gate.step_allowed(control_now)){
      control=getControl(quad,command);
      bool rpm_ok=true;for(int i=0;i<4;++i)rpm_ok=rpm_ok&&std::isfinite(control.rpm[i])&&control.rpm[i]>=0;
      if(!rpm_ok){startup_gate.aborted=true;}
      else {
        quad.setInput(control.rpm[0],control.rpm[1],control.rpm[2],control.rpm[3]);
        quad.setExternalForce(disturbance.f);quad.setExternalMoment(disturbance.m);
        quad.step(dt);++startup_gate.steps;
      }
    }
    if(startup_gate.aborted){ROS_ERROR("STARTUP_OR_COMMAND_ABORT_LATCHED");return 3;}
'''
 s=s[:first]+newblock+s[last:]
 s=s.replace('      quadToImuMsg(quad, imu);','''      if(startup_gate.steps>0)quadToImuMsg(quad, imu);
      else {imu.orientation=odom_msg.pose.pose.orientation;imu.angular_velocity.x=imu.angular_velocity.y=imu.angular_velocity.z=0;imu.linear_acceleration.x=imu.linear_acceleration.y=imu.linear_acceleration.z=0;}
      imu.header.stamp=tnow;  // Same actual publication/state sample as odometry,not receipt-time restamping.
      std_msgs::String phase;std::ostringstream state_text;
      state_text<<"{\\"phase\\":\\""<<(startup_gate.armed?"INTEGRATING":"HELD_BEFORE_INTEGRATION")<<"\\",\\"integration_steps\\":"<<startup_gate.steps
        <<",\\"stamp\\":"<<std::setprecision(17)<<tnow.toSec()<<",\\"final_command_stamp\\":"<<startup_gate.last_stamp
        <<",\\"command_valid\\":"<<(startup_gate.valid?"true":"false")<<",\\"mass\\":"<<quad.getMass()<<",\\"gravity\\":"<<quad.getGravity()
        <<",\\"force_world\\":["<<command.force[0]<<","<<command.force[1]<<","<<command.force[2]<<"]}";
      phase.data=state_text.str();phase_pub.publish(phase);''')
 s=s.replace('#include <sstream>','#include <sstream>\n#include <iomanip>')
 generated=b/'gated_quadrotor_simulator.cpp';generated.write_text(s)
 (b/'plant_gate.hpp').write_bytes((R/'query_minco/integration/yopo_startup_v2/plant_gate.hpp').read_bytes())
 write(b/'DERIVATION.json',dict(component_id='native_held_start_common_imu_stamp_v2',baseline=str(src),baseline_sha256=sha(src),generated_sha256=sha(generated),
  changed=['stamp actual IMU sample','held-state0 acceleration explicitly tagged,not uninitialized native getAcc','require5 valid final hover commands+nonce before any step','failclosed on invalid actuators,no staleRPM fallback','phase/force/step telemetry'],
  unchanged=['dynamics library,mass,gravity,inertia,motorlag,control math,integration dt,gains,Hard thresholds'],native_execution='NOT_RUN'))
 with (b/'BASELINE_OVERLAY.patch').open('x') as f:f.writelines(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile=str(src),tofile=str(generated)))
 make=Y/'build/so3_quadrotor_simulator/CMakeFiles/quadrotor_simulator_so3.dir';flags=(make/'flags.make').read_text().splitlines()
 includes=shlex.split(next(x.split('=',1)[1] for x in flags if x.startswith('CXX_INCLUDES =')))
 obj=b/'gated.o';binary=b/'gated_quadrotor_simulator'
 compile=['/usr/bin/g++-9','-std=c++11','-O2','-I'+str(R/'query_minco/integration/yopo_startup_v2')]+includes+['-c',str(generated),'-o',str(obj)]
 link=shlex.split((make/'link.txt').read_text());link=[str(obj) if x.endswith('quadrotor_simulator_so3.cpp.o') else x for x in link];link[link.index('-o')+1]=str(binary)
 env=dict(os.environ,LD_LIBRARY_PATH=str(Y/'devel/lib')+':/opt/ros/noetic/lib')
 for name,argv,cwd in [('compile',compile,R),('link',link,make.parent.parent)]:
  p=run_logged(argv,b/name,cwd=cwd,env=env,runspec=O/'PREREGISTRATION.json',timeout_s=180)
  if p['exit_code']:raise RuntimeError('BUILD_'+name)
 write(b/'BUILD_IDENTITY.json',dict(binary=str(binary),sha256=sha(binary),generated_sha256=sha(generated),baseline_sha256=sha(src),
  dynamics_library_sha256=sha(Y/'devel/lib/libquadrotor_dynamics.so'),header_sha256=sha(R/'query_minco/integration/yopo_startup_v2/plant_gate.hpp'),compiled=True,executed=False))
 print('GATED_SIMULATOR_BUILD_PASS_NO_EXECUTION')
if __name__=='__main__':main()
