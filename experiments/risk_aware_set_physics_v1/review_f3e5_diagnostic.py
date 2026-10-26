"""Emit diagnostic DIFF TEXT ONLY. Never apply, compile or run Simulator code."""
import difflib,hashlib,json,re
from pathlib import Path
ROOT=Path('/home/lizheng/query_minco_mainline');OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution009'
def candidate_text():
 base=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution004/runtime/quadrotor_simulator_startup_candidate.cpp';old=base.read_text()
 helper=r'''
// E5 DRAFT ONLY: default-off, bounded readonly diagnostics.
#include <array>
#include <vector>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cerrno>
#include <limits.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <ctime>
static bool e5_diag_enabled=false;
static bool e5_so3_seen=false;
static double e5_so3_stamp=0,e5_so3_recv_mono=0;
static unsigned long long e5_so3_seq=0;
static double e5_mono() {
  timespec t;return clock_gettime(CLOCK_MONOTONIC,&t)==0 ?
    t.tv_sec+t.tv_nsec*1e-9 : std::nan("");
}
struct E5Row {
  unsigned long long steps,so3_seq;
  double stamp,mono,dt,so3_stamp,so3_recv_mono;
  std::array<double,3> p,v,rhs_a,so3_force;
  std::array<double,4> request_rpm,state_rpm;
};
class E5Diagnostics {
  static constexpr size_t capacity=12000;
  static constexpr double step_wall_tolerance=.002;
  static constexpr double paired_clock_tolerance=.005;
  std::vector<E5Row> rows_;int fd_=-1;
  bool enabled_=false,overflow_=false,metrology_bad_=false,finished_=false,success_=false;
 public:
  explicit E5Diagnostics(bool enabled):enabled_(enabled) {
    if(!enabled_)return;
    const char* root="/home/lizheng/query_minco_mainline/research_outputs/rsps_f3_low_speed_v1/f3_execution010/native_diag001";
    char canonical[PATH_MAX];
    if(!realpath(root,canonical)||std::string(canonical)!=root)
      throw std::runtime_error("DIAG_EXACT_NEW_RUN_ROOT");
    rows_.reserve(capacity); // All allocation before integration begins.
    std::string path=std::string(root)+"/STEP_DIAGNOSTIC.csv";
    fd_=open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);
    if(fd_<0)throw std::runtime_error("DIAG_EXCLUSIVE_FILE");
  }
  ~E5Diagnostics(){if(fd_>=0)close(fd_);} // Missing finish => no COMPLETE footer.
  void add(const E5Row& r) {
    if(!enabled_)return;
    if(rows_.size()==capacity){overflow_=true;return;}
    bool valid=std::isfinite(r.stamp)&&std::isfinite(r.mono)&&
      std::isfinite(r.dt)&&r.dt>0&&e5_so3_seen&&
      std::isfinite(r.so3_stamp)&&std::isfinite(r.so3_recv_mono);
    for(const auto* group:{&r.p,&r.v,&r.rhs_a,&r.so3_force})
      for(double x:*group)valid=valid&&std::isfinite(x);
    for(double x:r.request_rpm)valid=valid&&std::isfinite(x);
    for(double x:r.state_rpm)valid=valid&&std::isfinite(x);
    if(!rows_.empty()) {
      const E5Row& last=rows_.back();
      double wall=r.stamp-last.stamp,mono=r.mono-last.mono;
      valid=valid&&r.steps>last.steps&&wall>0&&mono>0&&r.dt==last.dt;
      if(valid) {
        double physics=(r.steps-last.steps)*r.dt;
        valid=std::fabs(physics-wall)<=step_wall_tolerance&&
              std::fabs(wall-mono)<=paired_clock_tolerance;
      }
    }
    metrology_bad_=metrology_bad_||!valid;
    rows_.push_back(r); // Finite preallocated copy. Never write/format/flush here.
  }
  bool finish() {
    if(!enabled_)return true;
    if(finished_)return success_;
    finished_=true;
    bool data_ok=!rows_.empty()&&!overflow_&&!metrology_bad_;
    std::ostringstream out;out<<std::setprecision(17);
    out<<"steps,stamp,mono,fixed_dt,so3_stamp,so3_recv_mono,so3_seq,p0,p1,p2,v0,v1,v2,rhs_a0,rhs_a1,rhs_a2,so3_f0,so3_f1,so3_f2,request_rpm0,request_rpm1,request_rpm2,request_rpm3,state_rpm0,state_rpm1,state_rpm2,state_rpm3\n";
    for(const auto&r:rows_) {
      out<<r.steps<<','<<r.stamp<<','<<r.mono<<','<<r.dt<<','<<r.so3_stamp<<','<<r.so3_recv_mono<<','<<r.so3_seq;
      for(const auto*group:{&r.p,&r.v,&r.rhs_a,&r.so3_force})
        for(double x:*group)out<<','<<x;
      for(double x:r.request_rpm)out<<','<<x;
      for(double x:r.state_rpm)out<<','<<x;
      out<<'\n';
    }
    out<<"#COMPLETE rows="<<rows_.size()<<" overflow="<<overflow_
       <<" metrology_bad="<<metrology_bad_<<" data_ok="<<data_ok<<"\n";
    const std::string text=out.str();size_t offset=0;bool io_ok=true;
    while(offset<text.size()) {
      ssize_t n=write(fd_,text.data()+offset,text.size()-offset);
      if(n<0&&errno==EINTR)continue;
      if(n<=0){io_ok=false;break;}
      offset+=static_cast<size_t>(n);
    }
    if(io_ok&&fsync(fd_)!=0)io_ok=false;
    if(close(fd_)!=0)io_ok=false;fd_=-1;
    success_=data_ok&&io_ok&&offset==text.size();
    if(success_)ROS_INFO("DIAG_FLUSH_COMPLETE rows=%zu bytes=%zu fsync=1",rows_.size(),offset);
    else ROS_ERROR("DIAG_FAIL rows=%zu overflow=%d metrology=%d io=%d",rows_.size(),overflow_,metrology_bad_,io_ok);
    return success_;
  }
};
'''
 new=old.replace('typedef struct _Control',helper+'\ntypedef struct _Control',1)
 anchor='  startup_command_seen = true;\n'
 so3=r'''  if(e5_diag_enabled) {
    e5_so3_seen=true;e5_so3_stamp=cmd->header.stamp.toSec();
    e5_so3_recv_mono=e5_mono();e5_so3_seq=cmd->header.seq;
  }
'''
 assert old.count(anchor)==1;new=new.replace(anchor,so3+anchor,1)
 anchor='  const double dt = 1 / simulation_rate;\n'
 setup=r'''  bool step_diagnostic_enabled=false;
  n.param("step_diagnostic_enabled",step_diagnostic_enabled,false);
  if(step_diagnostic_enabled&&(!startup_hold||
     startup_session!="f3_execution010_native_diag001")) {
    ROS_FATAL("E5 diagnostics require the independently authorized private run");
    return 2;
  }
  E5Diagnostics diagnostics(step_diagnostic_enabled);
  e5_diag_enabled=step_diagnostic_enabled;
'''
 assert old.count(anchor)==1;new=new.replace(anchor,anchor+setup,1)
 anchor='      stateToOdomMsg(state, odom_msg);\n'
 record=r'''      if(e5_diag_enabled) {
        const Eigen::Vector3d rhs=quad.getAcc();
        E5Row row{};row.steps=integration_steps;row.stamp=tnow.toSec();
        row.mono=e5_mono();row.dt=dt;row.so3_stamp=e5_so3_stamp;
        row.so3_recv_mono=e5_so3_recv_mono;row.so3_seq=e5_so3_seq;
        row.p={{odom_msg.pose.pose.position.x,odom_msg.pose.pose.position.y,odom_msg.pose.pose.position.z}};
        row.v={{odom_msg.twist.twist.linear.x,odom_msg.twist.twist.linear.y,odom_msg.twist.twist.linear.z}};
        for(int j=0;j<3;j++){row.rhs_a[j]=rhs[j];row.so3_force[j]=command.force[j];}
        for(int j=0;j<4;j++){row.request_rpm[j]=control.rpm[j];row.state_rpm[j]=state.motor_rpm[j];}
        diagnostics.add(row);
      }
'''
 assert old.count(anchor)==1;new=new.replace(anchor,anchor+record,1)
 anchor='  return 0;\n}\n'
 assert old.count(anchor)==1;new=new.replace(anchor,'  if(!diagnostics.finish())return 3;\n'+anchor,1)
 return base,old,new
def main():
 base,old,new=candidate_text()
 oldpatch=OUT.parent/'f3_execution008/patches/private_startup_step_diagnostic.diff';prior=oldpatch.read_text()
 lines=list(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/PRIVATE_STARTUP_BASE_ONLY.cpp',tofile='b/FUTURE_PRIVATE_E5_DIAGNOSTIC.cpp'))
 diff=''.join(lines)
 # TEXT ONLY: no patched source file is written and no compiler is invoked.
 (OUT/'patches/PRIVATE_DIAGNOSTIC_PATCH_PROPOSAL.diff').write_text(diff)
 removed=[s for s in lines if s.startswith('-') and not s.startswith('---')]
 assert not removed
 for token in ('quad.step(dt);','odom_msg.header.stamp = tnow;','quad.setInput(control.rpm[0], control.rpm[1], control.rpm[2],','const double dt = 1 / simulation_rate;'):
  assert new.count(token)==old.count(token)==1
 add=new[new.index('  void add('):new.index('  bool finish()')]
 assert not any(token in add for token in ('open(','write(','fsync(','close(','ostringstream','reserve(','resize('))
 assert 'write(fd,' not in prior and 'write(fd_' not in prior
 assert 'close(fd)' not in prior and 'close(fd_)' not in prior
 sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
 report=dict(status='STATIC_REVIEW_COMPLETE_PENDING_AUTHORIZED_BUILD',Simulator_patch_applied=False,Simulator_patch_compiled=False,Simulator_patch_run=False,
 baseline_source=str(base),baseline_sha256=sha(old),proposed_in_memory_source_sha256=sha(new),final_diff_sha256=sha(diff),future_binary_sha256=None,
 old_draft_sha256=sha(prior),old_draft_verdict='REJECTED: buffer formatted but never written or closed; no reliable flush/overflow/io-failure receipt; broad output path',
 final_changes=['default-off explicit switch','pinned future exclusive canonical run path; O_EXCL/O_NOFOLLOW/CLOEXEC','bounded preallocated100Hz rows; no IO/formatting in add','actual write loop handles partial/EINTR; fsync+close checked; explicit finish exit3 on failure','missing finish/footer/SIGKILL treated as log loss','step/wall2ms and wall/mono5ms mismatch flags FAIL; never replace original tracking measurement','SO3 source stamp/receive mono/seq plus force and motor/RHS fields'],
 original_physics_lines_unchanged=True,removed_original_lines=0,
 future_target_run='/home/lizheng/query_minco_mainline/research_outputs/rsps_f3_low_speed_v1/f3_execution010/native_diag001',
 CPU_scope='Static invariants and separate Python logging specification only. No claim of compiled patch or actual PVA-equivalence test.',
 unresolved_build_tests=['C++ syntax/build/link after independent authorization','diagnostic off/on actual fixed-input PVA bitwise comparison','preallocated collection overhead and actual flush/overflow/failure behavior','metrology consistency versus true model RHS, whole native tracking'])
 (OUT/'DIAGNOSTIC_STATIC_REVIEW.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
