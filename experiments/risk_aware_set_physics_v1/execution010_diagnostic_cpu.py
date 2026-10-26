"""Exact reviewed private patch CPU tests. No roscore, ROS node, sensor or C1."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/home/lizheng/query_minco_mainline')
OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution010/native_diag001'
SRC=OUT/'runtime/quadrotor_simulator_diagnostic.cpp'
BASE=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution004/runtime/quadrotor_simulator_startup_candidate.cpp'
DATA=OUT/'STEP_DIAGNOSTIC.csv'
os.environ['LD_LIBRARY_PATH']='/home/lizheng/YOPO/Controller/devel/lib:/opt/ros/noetic/lib'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
HARNESS=r'''
#include <atomic>
#include <new>
#include <cstdlib>
#include <cstdarg>
#include <cerrno>
#include <fstream>
#include <fcntl.h>
#include <iostream>
#include <iomanip>
#include <cstring>
static bool count_alloc=false;
static size_t collected_allocs=0;
void* operator new(std::size_t n){if(count_alloc)++collected_allocs;if(void*p=std::malloc(n))return p;throw std::bad_alloc();}
void* operator new[](std::size_t n){return ::operator new(n);}
void operator delete(void*p)noexcept{std::free(p);}
void operator delete[](void*p)noexcept{std::free(p);}
#define main uncalled_simulator_main
#ifdef QM_BASELINE
#include "BASE_INCLUDE"
#else
#include "PATCH_INCLUDE"
#endif
#undef main
static std::string test_mode;
static int diag_fd=-1;
static size_t writes=0,syncs=0,closes=0;
static bool interrupted=false;
extern "C" int __real_open(const char*,int,...);
extern "C" ssize_t __real_write(int,const void*,size_t);
extern "C" int __real_fsync(int);
extern "C" int __real_close(int);
extern "C" int __wrap_open(const char*p,int f,...){
 mode_t mode=0;if(f&O_CREAT){va_list v;va_start(v,f);mode=va_arg(v,int);va_end(v);}
 int fd=__real_open(p,f,mode);
 if(fd>=0&&std::strstr(p,"/native_diag001/STEP_DIAGNOSTIC.csv"))diag_fd=fd;
 return fd;
}
extern "C" ssize_t __wrap_write(int f,const void*p,size_t n){
 if(f==diag_fd){++writes;
  if(test_mode=="eintr"&&!interrupted){interrupted=true;errno=EINTR;return -1;}
  if(test_mode=="zero_write")return 0;
  if(test_mode=="write_error"){errno=ENOSPC;return -1;}
  if(test_mode=="partial")n=std::min(n,size_t(17));
 }
 return __real_write(f,p,n);
}
extern "C" int __wrap_fsync(int f){
 if(f==diag_fd){++syncs;if(test_mode=="fsync_error"){errno=EIO;return -1;}}
 return __real_fsync(f);
}
extern "C" int __wrap_close(int f){
 if(f==diag_fd){++closes;int r=__real_close(f);diag_fd=-1;
  if(test_mode=="close_error"){errno=EIO;return -1;}return r;}
 return __real_close(f);
}
#ifndef QM_BASELINE
static E5Row row_for(unsigned long long step){
 E5Row r{};r.steps=step;r.dt=.001;r.stamp=100.+step*.001;r.mono=200.+step*.001;
 r.so3_stamp=100.;r.so3_recv_mono=200.;r.so3_seq=step/20;
 r.p={{0,0,2}};r.so3_force={{0,0,9.6138}};
 r.request_rpm.fill(15000);r.state_rpm.fill(15000);return r;
}
#endif
int main(int argc,char**argv){
 if(argc!=3)return 90;
 test_mode=argv[1];ros::Time::init();
 bool equivalent=test_mode=="equiv_off"||test_mode=="equiv_on";
 bool enabled=test_mode!="disabled"&&test_mode!="equiv_off";
 try{
#ifndef QM_BASELINE
 E5Diagnostics diag(enabled);e5_diag_enabled=enabled;e5_so3_seen=true;
#endif
 size_t before=writes+syncs+closes;
 if(equivalent){
  QuadrotorSimulator::Quadrotor q;q.setStatePos(Eigen::Vector3d(-1.6,0,2));
  auto initial=q.getState();double hover=std::sqrt(q.getMass()*q.getGravity()/(4*q.getPropellerThrustCoefficient()));
  initial.motor_rpm.setConstant(hover);q.setState(initial);
  std::ofstream statefile(argv[2]);statefile<<std::setprecision(17);
  statefile<<"steps,p0,p1,p2,v0,v1,v2,rhs_a0,rhs_a1,rhs_a2\n";
  for(unsigned long long i=1;i<=12000;++i){
   auto msg=boost::make_shared<quadrotor_msgs::SO3Command>();
   msg->header.stamp=ros::Time(100.+i*.001);msg->header.seq=i/20;
   msg->force.z=q.getMass()*q.getGravity()+.02*std::sin(i*.001);
   msg->orientation.w=1;msg->aux.enable_motors=true;
   for(int j=0;j<3;++j){msg->kR[j]=2;msg->kOm[j]=.15;}
   cmd_callback(msg);Control c=getControl(q,command);
   q.setInput(c.rpm[0],c.rpm[1],c.rpm[2],c.rpm[3]);
   q.setExternalForce(Eigen::Vector3d::Zero());q.setExternalMoment(Eigen::Vector3d::Zero());
   q.step(.001);
   auto state=q.getState();auto acc=q.getAcc();statefile<<i;
   for(int j=0;j<3;++j)statefile<<','<<state.x[j];
   for(int j=0;j<3;++j)statefile<<','<<state.v[j];
   for(int j=0;j<3;++j)statefile<<','<<acc[j];
   statefile<<'\n';
#ifndef QM_BASELINE
   if(i%10==0){
    E5Row r=row_for(i);
    for(int j=0;j<3;++j){r.p[j]=state.x[j];r.v[j]=state.v[j];r.rhs_a[j]=acc[j];r.so3_force[j]=command.force[j];}
    for(int j=0;j<4;++j){r.request_rpm[j]=c.rpm[j];r.state_rpm[j]=state.motor_rpm[j];}
    count_alloc=true;diag.add(r);count_alloc=false;
   }
#endif
  }
  statefile.close();
 }else{
#ifndef QM_BASELINE
  int n=test_mode=="capacity"||test_mode=="overflow"?12000:3;
  if(test_mode=="empty")n=0;if(test_mode=="overflow")n=12001;
  for(int i=1;i<=n;++i){
   E5Row r=row_for(i*10);
   if(i==2){
    if(test_mode=="bad_step")r.steps=10;
    if(test_mode=="bad_wall")r.stamp-=.02;
    if(test_mode=="bad_mono")r.mono-=.02;
    if(test_mode=="bad_dt")r.dt=.002;
    if(test_mode=="metrology")r.steps+=10;
    if(test_mode=="nan_state")r.p[0]=std::nan("");
    if(test_mode=="nan_rpm")r.state_rpm[0]=std::nan("");
    if(test_mode=="nan_rhs")r.rhs_a[0]=std::nan("");
    if(test_mode=="no_so3")e5_so3_seen=false;
   }
   count_alloc=true;diag.add(r);count_alloc=false;
  }
#endif
 }
 bool no_io=(writes+syncs+closes)==before;
 bool good=true;
#ifndef QM_BASELINE
 good=diag.finish();
#endif
 std::cout<<"HARNESS_RESULT {\"good\":"<<(good?"true":"false")<<",\"collection_allocations\":"<<collected_allocs
          <<",\"no_collection_io\":"<<(no_io?"true":"false")<<",\"writes\":"<<writes
          <<",\"fsyncs\":"<<syncs<<",\"closes\":"<<closes<<"}\n";
 if(collected_allocs||!no_io)return 91;
 return good?0:3;
 }catch(const std::exception&e){std::cout<<"HARNESS_EXCEPTION "<<e.what()<<'\n';return 2;}
}
'''
def build():
 text=HARNESS.replace('BASE_INCLUDE',str(BASE)).replace('PATCH_INCLUDE',str(SRC))
 harness=OUT/'cpu/diagnostic_harness.cpp';harness.write_text(text)
 cmds=json.loads((OUT/'BUILD_COMMANDS.json').read_text());records=[]
 for baseline in (False,True):
  name='baseline_harness' if baseline else 'diagnostic_harness'
  obj=OUT/'cpu'/(name+'.o');exe=OUT/'cpu'/name
  cc=cmds[0].copy();cc[cc.index('-c')+1]=str(harness);cc[cc.index('-o')+1]=str(obj)
  if baseline:cc.insert(1,'-DQM_BASELINE')
  ld=cmds[1].copy();ld[ld.index('-rdynamic')+1]=str(obj);ld[ld.index('-o')+1]=str(exe)
  ld+=['-Wl,--wrap=open','-Wl,--wrap=write','-Wl,--wrap=fsync','-Wl,--wrap=close']
  for k,cmd in enumerate((cc,ld)):
   records.append(cmd)
   with (OUT/'logs'/f'{name}_build{k}.log').open('xb') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
   assert r.returncode==0,('CPU_HARNESS_BUILD_FAIL',name,k)
 (OUT/'cpu/BUILD_COMMANDS.json').write_text(json.dumps(records,indent=2))
def test_case(mode,expected,baseline=False):
 case=OUT/'cpu'/mode;case.mkdir(exist_ok=False)
 assert not DATA.exists(),'UNEXPECTED_EXISTING_STEP_FILE'
 if mode=='exclusive':DATA.write_text('CPU_EXCLUSIVE_SENTINEL\n')
 exe=OUT/'cpu'/('baseline_harness' if baseline else 'diagnostic_harness')
 args=[str(exe),'equiv_off' if baseline else mode,str(case/'PVA.csv')]
 start=time.monotonic()
 r=subprocess.run(args,capture_output=True,text=True,timeout=30)
 (case/'stdout.log').write_text(r.stdout);(case/'stderr.log').write_text(r.stderr)
 output=None
 if DATA.exists():
  output=case/'STEP_DIAGNOSTIC_CPU_ONLY.csv'
  DATA.rename(output) # Preserve each newly created CPU test artifact, never use as native data.
 assert r.returncode==expected,(mode,r.returncode,r.stdout,r.stderr)
 info=None
 for line in r.stdout.splitlines():
  if line.startswith('HARNESS_RESULT '):info=json.loads(line[len('HARNESS_RESULT '):])
 if mode=='exclusive':assert output.read_text()=='CPU_EXCLUSIVE_SENTINEL\n'
 elif mode=='disabled' or baseline or mode=='equiv_off':assert output is None
 else:
  assert output is not None
  if mode not in ('zero_write','write_error'):
   lines=output.read_text().splitlines();assert lines[-1].startswith('#COMPLETE')
   if mode in ('capacity','overflow'):assert len(lines)==12002
  if info:assert info['collection_allocations']==0 and info['no_collection_io']
 return dict(case=mode,expected_exit=expected,actual_exit=r.returncode,passed=True,wall_s=time.monotonic()-start,compiled_cpp_result=info,diagnostic_cpu_sha256=sha(output) if output else None)
def main():
 assert sha(SRC)=='7a029bc6c21bed95b5ae5d6901f7dac9817838ceea7b2c3ddd27b27806592ce4'
 # Static default-off/guard and unchanged core statements supplement compiled tests.
 text=SRC.read_text()
 assert 'bool step_diagnostic_enabled=false;' in text
 assert 'n.param("step_diagnostic_enabled",step_diagnostic_enabled,false);' in text
 assert 'startup_session!="f3_execution010_native_diag001"' in text
 body=text[text.index('  void add('):text.index('  bool finish()')]
 assert not any(t in body for t in ('write(','fsync(','close(','reserve(','resize(','mutex','sleep(','ostringstream'))
 build();records=[]
 specs=[('disabled',0),('equiv_off',0),('equiv_on',0),('equiv_baseline',0),('capacity',0),('overflow',3),('exclusive',2),('partial',0),('eintr',0),('zero_write',3),('write_error',3),('fsync_error',3),('close_error',3),('empty',3),('bad_step',3),('bad_wall',3),('bad_mono',3),('bad_dt',3),('metrology',3),('nan_state',3),('nan_rpm',3),('nan_rhs',3),('no_so3',3)]
 try:
  for mode,expect in specs:records.append(test_case(mode,expect,mode=='equiv_baseline'))
  states=[OUT/'cpu'/n/'PVA.csv' for n in ('equiv_baseline','equiv_off','equiv_on')]
  hashes=[sha(p) for p in states];assert len(set(hashes))==1,'FIXED_INPUT_PVA_STEPS_NOT_IDENTICAL'
  assert all(sum(1 for _ in p.open())==12001 for p in states)
  report=dict(status='PASS',tests=len(records),all_passed=True,actual_exact_compiled_cpp=True,
   fixed_input=dict(steps=12000,fixed_dt=.001,PVA_RHS_stepwise_bitwise_identical=True,sha256=hashes[0],paths=[str(p) for p in states],Controller_command_input='deterministic fixed SO3 input, CPU only, no Controller modification'),
   cases=records,ROS_started=False,C1_inference=False,raw_native_diagnostic_file_exists=DATA.exists(),
   limitation='No wall-paced ROS loop in CPU equality. Real native tracking/latency remains separate.')
 except Exception as e:
  report=dict(status='FAIL',all_passed=False,completed=records,error=repr(e),ROS_allowed=False)
  (OUT/'CPU_DIAGNOSTIC_RESULTS.json').write_text(json.dumps(report,indent=2));raise
 (OUT/'CPU_DIAGNOSTIC_RESULTS.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
