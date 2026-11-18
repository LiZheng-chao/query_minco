#pragma once
#include <quadrotor_msgs/PositionCommand.h>
#include <quadrotor_msgs/SO3Command.h>
#include <Eigen/Core>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
namespace f0audit {
inline double mono() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
struct Record {
 int kind=0; uint64_t index=0, command_index=0, output_index=0, input_ns=0, output_ns=0;
 uint32_t trajectory=0; double monotonic=0, received_ros=0, compute_s=0, enqueue_s=0;
 std::array<double,9> pva{}; std::array<double,3> kx{},kv{},force{},kr{},kom{};
 std::array<double,4> orientation{}; std::array<double,7> aux{};
 double yaw=0,yaw_dot=0;
};
class Audit {
 std::array<Record,8192> queue_; std::atomic<size_t> head_{0},tail_{0};
 std::mutex mutex_; std::condition_variable cv_; std::thread writer_;
 std::atomic<bool> stopped_{false},failed_{false}; std::atomic<uint64_t> lost_{0};
 int fd_=-1; uint64_t serial_=0,command_serial_=0,output_serial_=0; Record active_;
 std::string nonce_;
 void bytes(const std::string& value) {
  const char* p=value.data();size_t left=value.size();
  while(left) {ssize_t n=::write(fd_,p,left);if(n<0&&errno==EINTR)continue;
   if(n<=0){failed_=true;return;}p+=n;left-=n;}
 }
 template<size_t N> static void array(std::ostream& s,const std::array<double,N>& a){s<<'[';for(size_t i=0;i<N;++i){if(i)s<<',';s<<a[i];}s<<']';}
 void emit(const Record& r) {
  std::ostringstream s;s<<std::setprecision(17);
  s<<"{\"kind\":\""<<(r.kind==1?"COMMAND_CONSUMED":"CONTROL_OUTPUT")<<"\",\"index\":"<<r.index
   <<",\"command_index\":"<<r.command_index<<",\"output_index\":"<<r.output_index
   <<",\"trajectory_id\":"<<r.trajectory<<",\"input_ns\":"<<r.input_ns<<",\"output_ns\":"<<r.output_ns
   <<",\"monotonic\":"<<r.monotonic<<",\"received_ros\":"<<r.received_ros<<",\"compute_s\":"<<r.compute_s
   <<",\"enqueue_s\":"<<r.enqueue_s<<",\"pva\":";array(s,r.pva);
  s<<",\"kx\":";array(s,r.kx);s<<",\"kv\":";array(s,r.kv);s<<",\"yaw\":"<<r.yaw<<",\"yaw_dot\":"<<r.yaw_dot;
  s<<",\"force\":";array(s,r.force);s<<",\"orientation\":";array(s,r.orientation);
  s<<",\"kr\":";array(s,r.kr);s<<",\"kom\":";array(s,r.kom);s<<",\"aux\":";array(s,r.aux);s<<"}\n";bytes(s.str());
 }
 void push(Record r) {
  if(fd_<0)return;
  const double began=mono();r.index=++serial_;r.monotonic=began;
  const size_t tail=tail_.load(std::memory_order_relaxed),next=(tail+1)%queue_.size();
  if(next==head_.load(std::memory_order_acquire)){++lost_;return;}
  r.enqueue_s=mono()-began;queue_[tail]=r;tail_.store(next,std::memory_order_release);
 }
 void drain() {
  while(true){
   const size_t head=head_.load(std::memory_order_relaxed);
   if(head==tail_.load(std::memory_order_acquire)){
    if(stopped_.load(std::memory_order_acquire))break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));continue;
   }
   Record r=queue_[head];head_.store((head+1)%queue_.size(),std::memory_order_release);emit(r);
  }
 }

public:
 void start(const std::string& path,const std::string& nonce){
  if(nonce.empty()||nonce.find_first_not_of("0123456789abcdef")!=std::string::npos)throw std::runtime_error("AUDIT_NONCE");
  fd_=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);if(fd_<0)throw std::runtime_error("AUDIT_EXCLUSIVE_OPEN");
  nonce_=nonce;bytes("{\"kind\":\"AUDIT_OPEN\",\"schema\":\"f0_controller_consumption_v1\",\"nonce\":\""+nonce+"\",\"pid\":"+std::to_string(getpid())+"}\n");
  writer_=std::thread([this]{drain();});
 }
 void consume(const quadrotor_msgs::PositionCommand& c,const Eigen::Vector3d& kx,const Eigen::Vector3d& kv){
  active_=Record();active_.kind=1;active_.command_index=++command_serial_;active_.trajectory=c.trajectory_id;
  active_.input_ns=c.header.stamp.toNSec();active_.received_ros=ros::Time::now().toSec();
  active_.pva={{c.position.x,c.position.y,c.position.z,c.velocity.x,c.velocity.y,c.velocity.z,c.acceleration.x,c.acceleration.y,c.acceleration.z}};
  active_.yaw=c.yaw;active_.yaw_dot=c.yaw_dot;for(int i=0;i<3;++i){active_.kx[i]=kx[i];active_.kv[i]=kv[i];}push(active_);
 }
 void output(const quadrotor_msgs::SO3Command& c,double began){
  Record r=active_;r.kind=2;r.output_index=++output_serial_;r.output_ns=c.header.stamp.toNSec();r.compute_s=mono()-began;
  r.force={{c.force.x,c.force.y,c.force.z}};r.orientation={{c.orientation.x,c.orientation.y,c.orientation.z,c.orientation.w}};
  for(int i=0;i<3;++i){r.kr[i]=c.kR[i];r.kom[i]=c.kOm[i];}
  r.aux={{c.aux.current_yaw,c.aux.kf_correction,c.aux.angle_corrections[0],c.aux.angle_corrections[1],double(c.aux.enable_motors),double(c.aux.use_external_yaw),0}};push(r);
 }
 ~Audit(){if(fd_<0)return;stopped_=true;cv_.notify_one();if(writer_.joinable())writer_.join();
  bytes("{\"kind\":\"AUDIT_CLOSE\",\"records\":"+std::to_string(serial_)+",\"lost\":"+std::to_string(lost_.load())+",\"write_failed\":"+(failed_?"true":"false")+"}\n");
  ::fsync(fd_);::close(fd_);}
};
}
