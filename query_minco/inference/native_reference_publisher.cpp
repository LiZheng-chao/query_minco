// Reference-only transport of Python-certified immutable MINCO/quintic polynomials.
// NO network, selector, optimizer, planning, geometry certification, JSON or disk IO in tick.
#include <ros/ros.h>
#include <ros/master.h>
#include <nav_msgs/Odometry.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <std_msgs/String.h>
#include <array>
#include <atomic>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <thread>
#include <cmath>
#include <ctime>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <string>
using A9=std::array<double,9>;
double mono(){timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
double tcpu(){timespec t;clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return t.tv_sec+t.tv_nsec*1e-9;}
double dist(const A9&a,const A9&b,int d){double v=0;for(int j=0;j<3;j++)v+=std::pow(a[d*3+j]-b[d*3+j],2);return std::sqrt(v);}
double norm(const A9&a,int d){A9 z{};return dist(a,z,d);}
struct Piece{double T;std::array<double,18> c;};
struct Poly{
 std::vector<Piece> pieces;double duration=0;
 A9 at(double t)const{
  t=std::max(0.,std::min(t,duration));const Piece* p=&pieces.back();
  for(const auto&v:pieces){p=&v;if(t<=v.T+1e-10)break;t-=v.T;}
  A9 a{};for(int d=0;d<3;d++)for(int j=0;j<3;j++)
   for(int k=5;k>=d;k--){double f=1;for(int x=0;x<d;x++)f*=k-x;a[d*3+j]=a[d*3+j]*t+p->c[j*6+k]*f;}
  return a;
 }
};
struct Stop{double phase;Poly p;};
struct Bundle{int id;double preparedR,preparedM,validR,validM;Poly pre;std::vector<Stop> stops;};
template<class T>T read(std::ifstream&f){T x;f.read(reinterpret_cast<char*>(&x),sizeof x);if(!f)throw std::runtime_error("TRUNCATED");return x;}
Poly loadpoly(std::ifstream&f){Poly p;int n=read<int>(f);if(n<1||n>16)throw std::runtime_error("POLY_BOUND");
 for(int i=0;i<n;i++){Piece v;v.T=read<double>(f);if(!(v.T>0&&v.T<100))throw std::runtime_error("DURATION");for(auto&x:v.c){x=read<double>(f);if(!std::isfinite(x))throw std::runtime_error("COEFF");}p.duration+=v.T;p.pieces.push_back(v);}return p;}
std::vector<Bundle> load(const std::string&path){
 std::ifstream f(path,std::ios::binary);char magic[8];f.read(magic,8);if(std::memcmp(magic,"QMRF005",8))throw std::runtime_error("MAGIC");
 int n=read<int>(f);if(n<3||n>12)throw std::runtime_error("BUNDLE_BOUND");std::vector<Bundle>b;
 for(int i=0;i<n;i++){Bundle x;x.id=read<int>(f);x.preparedR=read<double>(f);x.preparedM=read<double>(f);x.validR=read<double>(f);x.validM=read<double>(f);x.pre=loadpoly(f);int k=read<int>(f);if(k<1||k>1025)throw std::runtime_error("STOP_BOUND");
 for(int j=0;j<k;j++){Stop s;s.phase=read<double>(f);s.p=loadpoly(f);x.stops.push_back(s);}b.push_back(std::move(x));}return b;
}
struct State{double stamp=0,recv=0,entrycpu=0,enqueue=0;A9 pva{};std::array<double,4>q{};bool valid=false;};
std::atomic<uint64_t> sequence{0};std::array<std::atomic<double>,21> snapshot;
std::atomic<bool> released{false},finish{false},authority{false},holdready{false};
std::atomic<double> authorityTime{0};std::atomic<int> liveStage{-1};
struct ORec{double stamp,recv,cpu,done;A9 a;std::array<double,4>q;};
std::vector<ORec> raw(20000);size_t nr=0;double prevStamp=0;std::array<double,3>prevV{};
std::atomic<bool> overflow{false};
void odom(const nav_msgs::Odometry::ConstPtr&m){
 double entry=mono(),cpu=tcpu(),t=m->header.stamp.toSec();A9 a{};
 const auto&p=m->pose.pose;const auto&v=m->twist.twist.linear;
 a[0]=p.position.x;a[1]=p.position.y;a[2]=p.position.z;a[3]=v.x;a[4]=v.y;a[5]=v.z;
 double dt=t-prevStamp;if(prevStamp==0){prevStamp=t;for(int j=0;j<3;j++)prevV[j]=a[3+j];return;}if(dt<.005)return;
 bool valid=dt<=.05&&dt>0;for(int j=0;j<3;j++)a[6+j]=(a[3+j]-prevV[j])/dt;
 prevStamp=t;for(int j=0;j<3;j++)prevV[j]=a[3+j];
 std::array<double,4>q{{p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w}};
 sequence.fetch_add(1,std::memory_order_acq_rel);
 snapshot[0]=t;snapshot[1]=entry;snapshot[2]=cpu;snapshot[3]=valid?1:0;
 for(int j=0;j<9;j++)snapshot[4+j]=a[j];for(int j=0;j<4;j++)snapshot[13+j]=q[j];snapshot[17]=mono();
 sequence.fetch_add(1,std::memory_order_release);
 if(nr<raw.size())raw[nr++]={t,entry,cpu,mono(),a,q};else overflow=true;
}
State state(){
 static State cached;State s;for(int tries=0;tries<3;tries++){auto q=sequence.load(std::memory_order_acquire);if(q&1)continue;
 s.stamp=snapshot[0];s.recv=snapshot[1];s.entrycpu=snapshot[2];s.valid=snapshot[3]>.5;
 for(int j=0;j<9;j++)s.pva[j]=snapshot[4+j];for(int j=0;j<4;j++)s.q[j]=snapshot[13+j];s.enqueue=snapshot[17];
 if(q==sequence.load(std::memory_order_acquire)){cached=s;return s;}}return cached;
}
struct Event{int code,stage,bundle;double r,m,cpu,x,y,z;A9 actual,reference;};
struct Cmd{double stamp,begin,end,m,wake,cpu0,cpu1,tick,odomstamp,odomrecv;int stage,bundle,backup;A9 pva;};
std::vector<Event>events(30000);size_t ne=0;std::vector<Cmd>commands(10000);size_t nc=0;
int stage=0;bool failed=false;const Bundle*active=nullptr;const Stop*backup=nullptr;
double startR=0,startM=0,backupM=0;bool backupReceipt=false;
void event(int code,double r,double m,double x=0,double y=0,double z=0,const State&s=State{},const A9&ref=A9{}){
 if(ne<events.size())events[ne++]={code,stage,active?active->id:-1,r,m,tcpu(),x,y,z,s.pva,ref};else{overflow=true;failed=true;}
}
A9 reference(double m){if(!active)return A9{};double t=m-startM;if(backup&&m>=backupM)return backup->p.at(m-backupM);if(t>=active->pre.duration)return active->stops.back().p.at(t-active->pre.duration);return active->pre.at(t);}
bool lifetime(const Bundle*b,double r,double m,double remaining){return b&&r>=b->preparedR&&m>=b->preparedM&&r+remaining<=b->validR&&m+remaining<=b->validM;}
bool gate(const State&s,const A9&expected,double r,double m){
 if(!s.valid||r-s.stamp<0||r-s.stamp>.05||m-s.recv<0||m-s.recv>.05)return false;
 A9 a=s.pva;double dt=r-s.stamp;for(int j=0;j<3;j++){a[j]+=a[3+j]*dt+.5*a[6+j]*dt*dt;a[3+j]+=a[6+j]*dt;}
 return dist(a,expected,0)<=.02&&dist(a,expected,1)<=.05&&dist(a,expected,2)<=.1;
}
bool stoprequest(int reason,double r,double m,const State&s){
 event(30,r,m,reason,0,0,s,reference(m));if(!active)return false;if(backup)return true;
 double target=m-startM+.06;const Stop*chosen=&active->stops.back();
 for(const auto&x:active->stops)if(x.phase>=target-1e-9){chosen=&x;break;}
 double am=startM+chosen->phase;double remaining=std::max(0.,am-m)+chosen->p.duration;
 if(!lifetime(active,r,m,remaining)){event(34,r,m,reason,0,0,s);failed=true;return false;}
 backup=chosen;backupM=am;backupReceipt=false;
 event(31,r,m,chosen->phase,am,chosen->p.duration,s,chosen->p.at(0));return true;
}
void graph(const std::string&run){
 bool readyWritten=false;int oldstage=-99;
 while(!finish){
  XmlRpc::XmlRpcValue args,result,payload;args[0]=ros::this_node::getName();bool ok=false;
  if(ros::master::execute("getSystemState",args,result,payload,true)){
   auto pubs=payload[0];for(int i=0;i<pubs.size();i++)if(std::string(pubs[i][0])=="/so3_control/pos_cmd"){
    auto nodes=pubs[i][1];ok=nodes.size()==1&&std::string(nodes[0])=="/f3_reference_arbiter";
   }
  }
  authority=ok;authorityTime=mono();
  if(holdready&&!readyWritten){std::ofstream f(run+"/HOLD_READY.json");f<<"{\"reference_only\":true,\"Query_control\":false}";readyWritten=true;}
  int st=liveStage.load();if(st!=oldstage){std::ofstream f(run+"/CONTROL_STAGE.jsonl",std::ios::app);f<<std::setprecision(17)<<"{\"stage\":"<<st<<",\"mono\":"<<mono()<<"}\n";oldstage=st;}
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
 }
}
void waituntil(double x){timespec t{time_t(x),long((x-std::floor(x))*1e9)};while(clock_nanosleep(CLOCK_MONOTONIC,TIMER_ABSTIME,&t,nullptr)==EINTR){}}
bool clock_ok(double lr,double lm,double r,double m){return m-lm<=.04&&std::fabs((r-lr)-(m-lm))<=.005;}
bool activation_time_ok(double ar,double am,double r,double m){return r>=ar&&r-ar<=.01&&m>=am&&m-am<=.01;}
void phase_cb(const std_msgs::String::ConstPtr&m){if(m->data=="RELEASED_BEFORE_FIRST_INTEGRATION")released=true;}
int main(int argc,char**argv){
 if(argc==3&&std::string(argv[1])=="--cpu-test"){
  auto bs=load(argv[2]);size_t checked=0;
  for(const auto&b:bs)for(const auto&st:b.stops){
   auto a=b.pre.at(st.phase),h=st.p.at(0),end=st.p.at(st.p.duration);
   if(dist(a,h,0)>.0002||dist(a,h,1)>.002||dist(a,h,2)>.01||norm(end,1)>1e-7||norm(end,2)>1e-7)return 3;
   checked++;
  }
  if(activation_time_ok(1,1,.999,.999)||activation_time_ok(1,1,1.02,1.02)||!activation_time_ok(1,1,1.005,1.005))return 4;
  State sample;sample.valid=true;sample.stamp=1;sample.recv=1;A9 origin{};
  if(!gate(sample,origin,1.005,1.005))return 6;
  sample.recv=.9;if(gate(sample,origin,1.005,1.005))return 7;
  if(clock_ok(1,1,1.075,1.075)||clock_ok(1,1,1.22,1.02))return 8;
  std::cout<<"CPU transport PVA/stop checks "<<checked<<" passed\\n";return 0;
 }

 if(argc==4&&std::string(argv[1])=="--cpu-timer"){
  auto bs=load(argv[2]);std::vector<std::array<double,4>> samples(251);
  double start=mono();
  for(int i=0;i<251;i++){double d=start+i*.02;waituntil(d);double m=mono(),c=tcpu();auto p=bs[1].pre.at(std::min(.02*i,bs[1].pre.duration));samples[i]={{m,d,c,tcpu()}};if(!std::isfinite(p[0]))return 5;}
  std::ofstream f(argv[3]);f<<std::setprecision(17)<<"mono,due,cpu0,cpu1\n";for(auto&a:samples)f<<a[0]<<','<<a[1]<<','<<a[2]<<','<<a[3]<<'\n';return 0;
 }

 ros::init(argc,argv,"f3_reference_arbiter");ros::V_string args;ros::removeROSArgs(argc,argv,args);
 if(args.size()!=4||args[3]!="--reference-only")return 2;
 std::string file=args[1],run=args[2];auto bundles=load(file);
 auto find=[&](int id)->const Bundle*{for(auto&b:bundles)if(b.id==id)return &b;return nullptr;};
 const Bundle*pending=nullptr;double activationR=0,activationM=0;
 ros::NodeHandle n;auto pub=n.advertise<quadrotor_msgs::PositionCommand>("/so3_control/pos_cmd",1);
 auto sub=n.subscribe<nav_msgs::Odometry>("/sim/odom",5,odom,ros::TransportHints().tcpNoDelay());
 auto phase=n.subscribe<std_msgs::String>("/quadrotor_simulator_so3/startup_phase",2,phase_cb);
 ros::AsyncSpinner spinner(1);spinner.start();std::thread monitor(graph,run);
 {std::ofstream f(run+"/ARBITER_READY.json");f<<"{\"reference_only\":true,\"Query_control\":false}";}
 auto schedule=[&](const Bundle*b,double r,double m){pending=b;activationR=r+.06;activationM=m+.06;event(10,r,m,activationR,activationM,b?b->id:-1);};
 double begin=mono(),next=begin,lastM=0,lastR=0,lastPub=0,stable=-1,heartbeat=0;
 bool wd=false,rejection=false,completed=false;int abortId=-1;
 while(ros::ok()&&mono()-begin<72){
  double wake=next;
  auto align=[&](double d){if(d>=mono()){if(next<=d&&d<=next+.005)wake=d;else wake=std::min(wake,d);}};
  if(pending)align(activationM);if(active){if(backup&&!backupReceipt)align(backupM);else if(!backup)align(startM+active->pre.duration);}
  waituntil(wake);double m=mono(),r=ros::Time::now().toSec(),cpu0=tcpu();State s=state();bool newAct=false,newBackup=false;
  if(lastM&&!clock_ok(lastR,lastM,r,m)){
   failed=true;event(40,r,m,m-lastM,r-lastR,0,s);pending=nullptr;stoprequest(40,r,m,s);
  }lastM=m;lastR=r;
  if(!authority||m-authorityTime>.5){if(active){failed=true;stoprequest(41,r,m,s);}pending=nullptr;}
  if(!active&&!pending&&authority&&s.valid)schedule(find(0),r,m);
  if(pending&&m>=activationM){
   bool ok=activation_time_ok(activationR,activationM,r,m);
   double horizon=0;for(const auto&x:pending->stops)horizon=std::max(horizon,x.phase+x.p.duration);
   ok=ok&&lifetime(pending,r,m,horizon);
   A9 head=pending->pre.at(0),old=reference(activationM);
   if(active)ok=ok&&dist(head,old,0)<=.0002&&dist(head,old,1)<=.002&&dist(head,old,2)<=.01;
   ok=ok&&gate(s,pending->pre.at(m-activationM),r,m);
   if(ok){active=pending;startR=activationR;startM=activationM;backup=nullptr;backupReceipt=false;newAct=true;event(11,r,m,r-activationR,activationR,activationM,s,head);}
   else event(12,r,m,r-activationR,activationR,activationM,s,head);pending=nullptr;
  }
  if(active&&!backup&&m-startM>=active->pre.duration){
   backup=&active->stops.back();backupM=startM+backup->phase;backupReceipt=false;event(31,r,m,backup->phase,backupM,backup->p.duration,s,backup->p.at(0));
  }
  if(active){
   A9 ref=reference(m);bool valid=gate(s,ref,r,m)&&lifetime(active,r,m,0)&&norm(s.pva,1)<=1&&norm(s.pva,2)<=1.5;
   if(!valid){if(!failed)event(42,r,m,r-s.stamp,m-s.recv,0,s,ref);failed=true;stoprequest(42,r,m,s);}
   if(backup&&m>=backupM&&!backupReceipt){
    bool stateok=gate(s,backup->p.at(m-backupM),r,m);
    event(32,r,m,m-backupM,stateok?1:0,backupM,s,backup->p.at(m-backupM));
    if(m-backupM>.01||!stateok)failed=true;
    backupReceipt=true;newBackup=true;
   }
   if(failed&&stage!=6){stage=6;stable=-1;event(43,r,m,0,0,0,s,ref);}
   if(stage==0&&released&&active->id==0){
    bool ok=norm(s.pva,1)<=.03&&norm(s.pva,2)<=.05&&dist(s.pva,active->pre.at(0),0)<=.02;
    if(ok){if(stable<0)stable=s.stamp;if(s.stamp-stable>=.5){schedule(find(1),r,m);stage=1;stable=-1;}}else stable=-1;
   }
   if(stage==1&&active->id==1){
    if(!rejection&&m-startM>2){if(activation_time_ok(r-.02,m-.02,r,m)){failed=true;event(45,r,m);}else event(12,r,m,.02,r-.02,m-.02,s,ref);rejection=true;} // rejected lease never replaces commitment
    if(m-startM>=active->pre.duration){
     bool ok=norm(s.pva,1)<=.03&&norm(s.pva,2)<=.05&&dist(s.pva,active->pre.at(active->pre.duration),0)<=.1;
     if(ok){if(stable<0)stable=s.stamp;if(s.stamp-stable>=.5){event(50,r,m,0,0,0,s,ref);schedule(find(2),r,m);stage=2;stable=-1;}}else stable=-1;
    }
   }
   if(stage==2&&active->id==2){
    double elapsed=m-startM;if(elapsed<2.4)heartbeat=m;
    if(!wd&&m-heartbeat>=.1){event(51,r,m,m-heartbeat,0,0,s,ref);stoprequest(51,r,m,s);wd=true;stage=3;}
   }
   if(stage==4&&active->id>=3&&m-startM>=2.4){event(52,r,m,0,0,0,s,ref);stoprequest(52,r,m,s);stage=5;}
   if((stage==3||stage==5||stage==6)&&backup&&m>=backupM+backup->p.duration){
    bool ok=norm(s.pva,1)<=.03&&norm(s.pva,2)<=.05;
    if(ok){if(stable<0)stable=s.stamp;
     if(s.stamp-stable>=.5){event(53,r,m,stage,0,0,s,ref);
      if(stage==3){int key=int(std::llround(backup->phase*100));abortId=(key==256?3:key==258?4:key==260?5:key==262?6:-1);
       if(abortId<0||!find(abortId)){failed=true;stage=6;}else{schedule(find(abortId),r,m);stage=4;stable=-1;}
      }else{completed=!failed&&stage==5;break;}
     }
    }else stable=-1;
   }
   // No file, JSON, RPC or planning on the physical publish path.
   quadrotor_msgs::PositionCommand msg;msg.header.stamp=ros::Time::now();msg.header.frame_id="world";msg.header.seq=nc;
   msg.trajectory_id=nc;msg.trajectory_flag=quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_EMPTY;
   msg.position.x=ref[0];msg.position.y=ref[1];msg.position.z=ref[2];msg.velocity.x=ref[3];msg.velocity.y=ref[4];msg.velocity.z=ref[5];
   msg.acceleration.x=ref[6];msg.acceleration.y=ref[7];msg.acceleration.z=ref[8];msg.yaw=0;
   double pbegin=ros::Time::now().toSec();pub.publish(msg);double pend=ros::Time::now().toSec();
   if(newAct){event(20,pend,mono(),pbegin-startR,pend-startR,startR,s,ref);if(pend-startR>.01)failed=true;}
   if(newBackup){event(33,pend,mono(),pbegin-(startR+backup->phase),pend-(startR+backup->phase),startR+backup->phase,s,ref);if(pend-(startR+backup->phase)>.01)failed=true;}
   if(active->id==0)holdready=true;
   if(lastPub&&pbegin-lastPub>.04){failed=true;event(44,r,m,pbegin-lastPub,0,0,s,ref);}lastPub=pbegin;
   if(nc<commands.size())commands[nc++]={msg.header.stamp.toSec(),pbegin,pend,m,wake,cpu0,tcpu(),m,s.stamp,s.recv,stage,active->id,backup?int(std::llround(backup->phase*100)):-1,ref};else{failed=true;overflow=true;}
  }
  liveStage=stage;if(wake>=next-.001)next+=.02;
 }
 finish=true;monitor.join();spinner.stop();sub.shutdown();phase.shutdown();pub.shutdown();
 std::ofstream c(run+"/COMMAND.csv");c<<std::setprecision(17)<<"stamp,begin,end,mono,wake,cpu0,cpu1,tick,odom_stamp,odom_recv,stage,bundle,backup,p0,p1,p2,v0,v1,v2,a0,a1,a2\n";
 for(size_t i=0;i<nc;i++){auto&x=commands[i];c<<x.stamp<<','<<x.begin<<','<<x.end<<','<<x.m<<','<<x.wake<<','<<x.cpu0<<','<<x.cpu1<<','<<x.tick<<','<<x.odomstamp<<','<<x.odomrecv<<','<<x.stage<<','<<x.bundle<<','<<x.backup;for(double v:x.pva)c<<','<<v;c<<'\n';}
 std::ofstream o(run+"/ODOM.csv");o<<std::setprecision(17)<<"stamp,recv,cpu,done,p0,p1,p2,v0,v1,v2,a0,a1,a2,qx,qy,qz,qw\n";
 for(size_t i=0;i<nr;i++){auto&x=raw[i];o<<x.stamp<<','<<x.recv<<','<<x.cpu<<','<<x.done;for(double v:x.a)o<<','<<v;for(double v:x.q)o<<','<<v;o<<'\n';}
 std::ofstream e(run+"/EVENT.csv");e<<std::setprecision(17)<<"code,stage,bundle,ros,mono,cpu,x,y,z,p0,p1,p2,v0,v1,v2,a0,a1,a2,rp0,rp1,rp2,rv0,rv1,rv2,ra0,ra1,ra2\n";
 for(size_t i=0;i<ne;i++){auto&x=events[i];e<<x.code<<','<<x.stage<<','<<x.bundle<<','<<x.r<<','<<x.m<<','<<x.cpu<<','<<x.x<<','<<x.y<<','<<x.z;for(double v:x.actual)e<<','<<v;for(double v:x.reference)e<<','<<v;e<<'\n';}
 std::ofstream result(run+"/ARBITER_RESULT.json");
 result<<"{\"status\":\""<<(completed&&!overflow?"REFERENCE_TRIALS_COMPLETE":"FAIL_NATIVE_CONTRACT")<<"\",\"commands\":"<<nc<<",\"Query_control\":false,\"qualification_failed\":"<<(failed?"true":"false")<<",\"buffer_overflow\":"<<(overflow?"true":"false")<<"}";
 ros::shutdown();return 0;
}
