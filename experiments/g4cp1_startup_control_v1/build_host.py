"""Versioned overlay of frozen G4C host;compile only,never execute the simulator."""
from pathlib import Path
import sys,json,difflib,shlex,os
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.audit.run_provenance_v1 import run_logged
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path.cwd();O=R/'research_outputs/g4cp1_startup_control_20261111_v1';Y=R.parent/'YOPO/Controller'
def main():
 b=O/'build_equilibrium_v3';b.mkdir();oldroot=R/'research_outputs/g4c_startup_and_dynamics_20261111_v1/startup_build_v2'
 meta=json.loads((oldroot/'BUILD_IDENTITY.json').read_text());src=oldroot/'gated_quadrotor_simulator.cpp'
 assert sha(src)==meta['generated_sha256']
 old=src.read_text();s=old.replace('g4c::PlantGate','g4cp1::PlantGate')
 line=next(x for x in s.splitlines() if 'startup_gate.accept(' in x)
 s=s.replace(line,'  if(cmd->header.frame_id!="/quadrotor"||!cmd->aux.enable_motors){startup_gate.trip();return;}\n'+
  '  for(int i=0;i<3;++i)if(!std::isfinite(cmd->kR[i])||!std::isfinite(cmd->kOm[i])||cmd->kR[i]<0||cmd->kOm[i]<0){startup_gate.trip();return;}\n'+
  '  if(!std::isfinite(cmd->aux.kf_correction)||cmd->aux.kf_correction>=8.98132e-9||!std::isfinite(cmd->aux.current_yaw)||!std::isfinite(cmd->aux.angle_corrections[0])||!std::isfinite(cmd->aux.angle_corrections[1])){startup_gate.trip();return;}\n'+line.replace('startup_gate.accept(', 'if(!startup_gate.accept(').replace('));', ')))return;'))
 s=s.replace('static void start_permit_cb', 'static void prepare_cb(const std_msgs::String::ConstPtr& m){startup_gate.prepare(m->data,ros::Time::now().toSec());}\nstatic void start_permit_cb')
 s=s.replace('  auto start_sub=', '  auto prepare_sub=n.subscribe("prepare_permit",1,&prepare_cb);\n  auto start_sub=')
 s=s.replace('  quad.setStatePos(position);', '  quad.setStatePos(position);\n  std::string motor_initial_state; n.param("motor_initial_state",motor_initial_state,std::string("zero"));\n  if(motor_initial_state=="equilibrium_hover"){auto initial=quad.getState();initial.motor_rpm.setConstant(std::sqrt(quad.getMass()*quad.getGravity()/(4*quad.getPropellerThrustCoefficient())));quad.setState(initial);}\n  else if(motor_initial_state!="zero"){ROS_ERROR("INVALID_MOTOR_INITIAL_STATE");return 2;}')
 s=s.replace('startup_gate.aborted=true;','startup_gate.trip();')
 s=s.replace('startup_gate.armed?"INTEGRATING":"HELD_BEFORE_INTEGRATION"','startup_gate.name()')
 extra=r'''std::string base=state_text.str();base.pop_back();std::ostringstream extra;
      extra<<",\"frame\":\"world\",\"motor_initial_state\":\""<<motor_initial_state<<"\",\"final_count\":"<<startup_gate.count
        <<",\"kf\":"<<std::setprecision(17)<<quad.getPropellerThrustCoefficient()
        <<",\"motor_tau_s\":"<<quad.getMotorTimeConstant()<<",\"dt\":"<<dt
        <<",\"rpm_actual\":["<<quad.getState().motor_rpm(0)<<","<<quad.getState().motor_rpm(1)<<","<<quad.getState().motor_rpm(2)<<","<<quad.getState().motor_rpm(3)<<"]"
        <<",\"rpm_input\":["<<control.rpm[0]<<","<<control.rpm[1]<<","<<control.rpm[2]<<","<<control.rpm[3]<<"]"
        <<",\"thrust_actual_N\":"<<quad.getPropellerThrustCoefficient()*quad.getState().motor_rpm.square().sum()
        <<",\"state_source\":\""<<(startup_gate.steps?"DYNAMICS_INTEGRATED":"HELD_NOT_PHYSICAL")<<"\"}";
      phase.data=base+extra.str();phase_pub.publish(phase);'''
 s=s.replace('phase.data=state_text.str();phase_pub.publish(phase);',extra)
 assert 'motor_initial_state' in s and 'motor_initial_state' in extra
 target=b/'gated_host.cpp';target.write_text(s);(b/'plant_gate.hpp').write_bytes((R/'query_minco/integration/yopo_startup_v3/plant_gate.hpp').read_bytes())
 with (b/'G4C_V2_TO_P1_V3.patch').open('x') as f:f.writelines(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile=str(src),tofile=str(target)))
 write(b/'DERIVATION.json',dict(component_id='g4cp1_explicit_held_prepare_ack_arm_native_host_v3',frozen_g4c_source=str(src),frozen_g4c_sha256=sha(src),original_yopo_sha256=meta['baseline_sha256'],new_source_sha256=sha(target),
  math_unchanged=['native SO3 force/moment conversion','mass/gravity/inertia','motorlag;zero cold-start retained;equilibrium initial state separately explicit','dt1000Hz,odom100Hz','original dynamics library'],executed=False))
 make=Y/'build/so3_quadrotor_simulator/CMakeFiles/quadrotor_simulator_so3.dir';flags=(make/'flags.make').read_text().splitlines()
 includes=shlex.split(next(x.split('=',1)[1] for x in flags if x.startswith('CXX_INCLUDES =')))
 obj=b/'gated.o';binary=b/'gated_host'
 compile=['/usr/bin/g++-9','-std=c++11','-O2']+includes+['-c',str(target),'-o',str(obj)]
 link=shlex.split((make/'link.txt').read_text());link=[str(obj) if x.endswith('quadrotor_simulator_so3.cpp.o') else x for x in link];link[link.index('-o')+1]=str(binary)
 for name,argv,cwd in [('compile',compile,R),('link',link,make.parent.parent)]:
  result=run_logged(argv,b/name,cwd=cwd,env=dict(os.environ,LD_LIBRARY_PATH=str(Y/'devel/lib')+':/opt/ros/noetic/lib'),timeout_s=180)
  assert result['exit_code']==0,name
 write(b/'BUILD_IDENTITY.json',dict(binary=str(binary),binary_sha256=sha(binary),header_sha256=sha(b/'plant_gate.hpp'),source_sha256=sha(target),dynamics_library_sha256=sha(Y/'devel/lib/libquadrotor_dynamics.so'),compiled=True,executed=False))
 print('V3_HOST_COMPILE_PASS_NOT_EXECUTED')
if __name__=='__main__':main()
