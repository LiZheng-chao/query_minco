// Reuse the exact previously qualified reference-only core. This wrapper adds
// a READONLY worker inbox. No write to physical active/pending/backup anywhere.
#define main qm_reference_only_main
#include "/home/lizheng/query_minco_mainline/research_outputs/rsps_f3_low_speed_v1/f3_execution005/runtime/native_reference_shadow.cpp"
#undef main
#include "dynamic_proposal_gate.hpp"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/Time.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <poll.h>
#include <deque>
#include <mutex>
#include <cstdlib>
namespace dyn {
using qm::Tree;
std::vector<unsigned char> filebytes(const std::string&p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("READ_FILE");return std::vector<unsigned char>((std::istreambuf_iterator<char>(f)),{});}
qm::Curve curve(const Poly&p){qm::Curve q;for(auto&x:p.pieces)q.p.push_back({x.T,x.c});return q;}
qm::Context context(const Tree&t){
 qm::Context c;c.nowR=t.get<double>("now_ros");c.nowM=t.get<double>("now_mono");c.stamp=t.get<double>("state_stamp");c.recv=t.get<double>("state_recv_mono");c.parentStartR=t.get<double>("parent_start_ros");
 c.cellExpiryR=t.get<double>("cell_expiry_ros");c.cellExpiryM=t.get<double>("cell_expiry_mono");c.epoch=t.get<std::string>("owner_epoch");c.geometry=t.get<std::string>("geometry");c.calibration=t.get<std::string>("calibration");c.parentDigest=t.get<std::string>("parent_digest");c.token=t.get<unsigned long long>("snapshot_token");
 auto pv=qm::values(t,"planning_pva"),pos=qm::values(t,"source_position"),R=qm::values(t,"source_rotation");std::copy(pv.begin(),pv.end(),c.pva.begin());std::copy(pos.begin(),pos.end(),c.sourcePosition.begin());std::copy(R.begin(),R.end(),c.sourceRotation.begin());
 c.publisherUnique=t.get<bool>("publisher_unique",true);c.clockValid=t.get<bool>("clock_valid",true);c.sourceSeen=t.get<bool>("source_seen",true);c.calibrationValid=t.get<bool>("calibration_valid",true);
 auto b=load(t.get<std::string>("reference_file"));c.parent=curve(b[1].pre);return c;
}
void packet(const std::vector<unsigned char>&in,Tree&m,std::vector<unsigned char>&body){
 qm::Reader r{in.data(),in.size()};int n=r.get<int>();if(n<2||n>16384||r.left<(size_t)n)throw std::runtime_error("HEADER_BOUND");
 m=qm::json(std::string((const char*)r.p,n));r.p+=n;r.left-=n;body.assign(r.p,r.p+r.left);
}
bool receive(int fd,void*p,size_t n){char*a=(char*)p;while(n){ssize_t k=recv(fd,a,n,0);if(k<=0)return false;a+=k;n-=k;}return true;}
void sendtree(int fd,const Tree&t){auto s=qm::dump(t);int n=s.size();send(fd,&n,4,MSG_NOSIGNAL);send(fd,s.data(),s.size(),MSG_NOSIGNAL);}
struct Observed{qm::PVA pva{};std::array<double,9>R{};double stamp=0,recv=0;bool valid=false;};
std::mutex sensorMutex;std::deque<Observed>observations;std::deque<double>depths,infos;double nominalStart=0;bool cameraValid=false;qm::PVA lastpv{};double laststamp=0;
void observedOdom(const nav_msgs::Odometry::ConstPtr&m){
 Observed o;auto&p=m->pose.pose.position;auto&v=m->twist.twist.linear;o.pva={{p.x,p.y,p.z,v.x,v.y,v.z,0,0,0}};o.stamp=m->header.stamp.toSec();o.recv=mono();
 auto&q=m->pose.pose.orientation;double x=q.x,y=q.y,z=q.z,w=q.w;
 o.R={{1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)}};
 std::lock_guard<std::mutex>l(sensorMutex);double dt=o.stamp-laststamp;
 if(dt>=.005){o.valid=laststamp>0&&dt<=.05;for(int j=0;j<3;j++)o.pva[6+j]=(o.pva[3+j]-lastpv[3+j])/dt;lastpv=o.pva;laststamp=o.stamp;if(o.valid){observations.push_back(o);if(observations.size()>128)observations.pop_front();}}
}
void image(const sensor_msgs::Image::ConstPtr&m){std::lock_guard<std::mutex>l(sensorMutex);if(m->width==160&&m->height==128&&m->encoding=="32FC1"&&m->step==640&&!m->is_bigendian){depths.push_back(m->header.stamp.toSec());if(depths.size()>64)depths.pop_front();}}
void camera(const sensor_msgs::CameraInfo::ConstPtr&m){
 const double K[9]={100,0,79.5,0,100,63.5,0,0,1};const double R[9]={1,0,0,0,1,0,0,0,1};const double P[12]={100,0,79.5,0,0,100,63.5,0,0,0,1,0};bool ok=m->width==160&&m->height==128;for(int i=0;i<9;i++)ok=ok&&m->K[i]==K[i];for(auto d:m->D)ok=ok&&d==0;for(int i=0;i<9;i++)ok=ok&&m->R[i]==R[i];for(int i=0;i<12;i++)ok=ok&&m->P[i]==P[i];
 std::lock_guard<std::mutex>l(sensorMutex);cameraValid=ok;if(ok){infos.push_back(m->header.stamp.toSec());if(infos.size()>64)infos.pop_front();}
}
void epochCallback(const std_msgs::Time::ConstPtr&m){std::lock_guard<std::mutex>l(sensorMutex);nominalStart=m->data.toSec();}
struct Snapshot{unsigned long long token;Observed observed;double issuedR,issuedM;};
void worker(std::string run,std::string refs,std::string config){
 while(!ros::isStarted()&&!finish)std::this_thread::sleep_for(std::chrono::milliseconds(2));
 if(finish)return;
 auto configBytes=filebytes(config);auto cfg=qm::json(std::string(configBytes.begin(),configBytes.end()));
 auto bundles=load(refs);qm::Curve parent=curve(bundles[1].pre);qm::Verifier verifier;
 ros::NodeHandle n;auto os=n.subscribe("/sim/odom",100,observedOdom);auto ds=n.subscribe("/query_shadow/depth",4,image);auto cs=n.subscribe("/query_shadow/camera_info",4,camera);
 auto es=n.subscribe<std_msgs::Time>("/reference/nominal_start",1,epochCallback);
 int server=socket(AF_UNIX,SOCK_STREAM,0);sockaddr_un addr{};addr.sun_family=AF_UNIX;std::string address="qm_f3e007_"+std::to_string(getpid());memcpy(addr.sun_path+1,address.data(),address.size());
 if(bind(server,(sockaddr*)&addr,offsetof(sockaddr_un,sun_path)+1+address.size())||listen(server,8))throw std::runtime_error("SOCKET_BIND");
 {std::ofstream f(run+"/DYNAMIC_READY.json");Tree t;t.put("address","@"+address);t.put("pid",getpid());t.put("control",false);f<<qm::dump(t);}
 std::deque<Snapshot>snaps;unsigned long long token=0;std::vector<Tree>log;std::vector<std::shared_ptr<const qm::Verified>>pending;
 int allowedPid=0;double previousR=0,previousM=0;bool clockValid=true;
 while(!finish){
  double r=ros::Time::now().toSec(),m=mono();
  if(previousM&&std::fabs((r-previousR)-(m-previousM))>.005)clockValid=false;previousR=r;previousM=m;
  if(!allowedPid){std::ifstream f(cfg.get<std::string>("worker_pid_file"));f>>allowedPid;}
  for(auto it=pending.begin();it!=pending.end();){
   auto v=*it;double ar=v->meta.get<double>("activation_ros");if(r<ar){++it;continue;}
   Tree row;row.put("type","ACTUAL_READONLY_GATE");row.put("sequence",v->sequence);row.put("actual_ros",r);row.put("actual_mono",m);row.put("control",false);
   Observed state;{std::lock_guard<std::mutex>l(sensorMutex);if(!observations.empty())state=observations.back();}
   bool ok=clockValid&&authority&&m-authorityTime.load()<=.5&&state.valid&&r-state.stamp>=0&&r-state.stamp<=.05&&m-state.recv<=.05&&r-ar<=.01&&r+v->bank.horizon<=v->meta.get<double>("valid_until_ros")&&m+v->bank.horizon<=v->meta.get<double>("valid_until_mono");
   qm::PVA predicted=state.pva;double dt=r-state.stamp;for(int j=0;j<3;j++){predicted[j]+=predicted[3+j]*dt+.5*predicted[6+j]*dt*dt;predicted[3+j]+=predicted[6+j]*dt;}
   auto expected=v->bank.prefix.at(std::max(0.,std::min(r-ar,v->bank.prefix.duration())));
   for(int d=0;d<3;d++){double err=qm::gap(predicted,expected,d);row.put("error"+std::to_string(d),err);ok=ok&&err<=(d==0?.02:d==1?.05:.1);}
   row.put("measurement_stamp",state.stamp);row.put("measurement_recv_mono",state.recv);row.put("passed",ok);log.push_back(row);it=pending.erase(it);
  }
  pollfd pf{server,POLLIN,0};if(poll(&pf,1,1)<=0)continue;
  int fd=accept(server,nullptr,nullptr);if(fd<0)continue;timeval to{0,20000};setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&to,sizeof to);
  Tree answer;auto start=qm::mark();try{
   ucred peer{};socklen_t len=sizeof peer;getsockopt(fd,SOL_SOCKET,SO_PEERCRED,&peer,&len);
   if(allowedPid<=0||peer.pid!=allowedPid||peer.uid!=getuid())throw std::runtime_error("UNTRUSTED_WORKER_PID");
   int size=0;if(!receive(fd,&size,4)||size<1||size>262144)throw std::runtime_error("MESSAGE_BOUND");
   std::vector<unsigned char>input(size);if(!receive(fd,input.data(),input.size()))throw std::runtime_error("SHORT_MESSAGE");
   if(input[0]=='S'){
    Observed o;double epoch;{std::lock_guard<std::mutex>l(sensorMutex);if(!observations.empty())o=observations.back();epoch=nominalStart;}
    if(!o.valid||epoch==0||liveStage.load()!=1)throw std::runtime_error("NO_CURRENT_NOMINAL_STATE");
    Snapshot snap{++token,o,ros::Time::now().toSec(),mono()};snaps.push_back(snap);if(snaps.size()>64)snaps.pop_front();
    answer.put("passed",true);answer.put("token",snap.token);answer.put("stamp",o.stamp);answer.put("recv_mono",o.recv);answer.put("epoch_ros",epoch);answer.put("cell_expiry_ros",bundles[0].validR);answer.put("cell_expiry_mono",bundles[0].validM);
    Tree pv;for(auto x:o.pva){Tree v;v.put("",x);pv.push_back({"",v});}answer.add_child("pva",pv);
   }else if(input[0]=='P'){
    Tree meta;std::vector<unsigned char>body;packet(std::vector<unsigned char>(input.begin()+1,input.end()),meta,body);
    auto tk=meta.get<unsigned long long>("snapshot_token");auto sit=std::find_if(snaps.begin(),snaps.end(),[&](auto&s){return s.token==tk;});if(sit==snaps.end())throw std::runtime_error("UNKNOWN_SNAPSHOT_TOKEN");
    qm::Context c;c.nowR=ros::Time::now().toSec();c.nowM=mono();c.stamp=sit->observed.stamp;c.recv=sit->observed.recv;c.pva=sit->observed.pva;c.token=tk;c.parent=parent;
    c.cellExpiryR=bundles[0].validR;c.cellExpiryM=bundles[0].validM;c.epoch=cfg.get<std::string>("owner_epoch");c.geometry=cfg.get<std::string>("geometry");c.calibration=cfg.get<std::string>("calibration");c.parentDigest=cfg.get<std::string>("parent_digest");
    c.publisherUnique=authority&&c.nowM-authorityTime.load()<=.5;c.clockValid=clockValid;c.sourceSeen=false;
    {std::lock_guard<std::mutex>l(sensorMutex);c.parentStartR=nominalStart;c.calibrationValid=cameraValid;double source=meta.get<double>("depth_stamp");
     auto found=std::find_if(observations.begin(),observations.end(),[&](auto&o){return o.stamp==source;});
     if(found!=observations.end()&&std::find(depths.begin(),depths.end(),source)!=depths.end()&&std::find(infos.begin(),infos.end(),source)!=infos.end()){c.sourceSeen=true;c.sourcePosition=qm::vector(found->pva,0);c.sourceRotation=found->R;}
    }
    auto previous=verifier.readonlyPending;auto oldHighwater=verifier.highwater;
    auto verified=verifier.check(meta,body,c);double readyR=ros::Time::now().toSec(),readyM=mono();
    if(readyR>meta.get<double>("activation_ros")-.02||readyM>meta.get<double>("activation_mono")-.02){verifier.readonlyPending=previous;verifier.highwater=oldHighwater;throw std::runtime_error("CPP_COMPLETE_READY_DEADLINE");}
    pending.push_back(verified);answer.put("passed",true);answer.put("sequence",verified->sequence);answer.put("receipt",verified->receipt);answer.put("scope","READONLY_LATCH_NOT_PHYSICAL");answer.put("verified_ros",readyR);answer.put("verified_mono",readyM);
   }else throw std::runtime_error("MESSAGE_KIND");
  }catch(const std::exception&e){answer.put("passed",false);answer.put("reason",e.what());}
  auto end=qm::mark();answer.put("wall_ms",(end.wall-start.wall)*1000);answer.put("thread_cpu_ms",(end.cpu-start.cpu)*1000);answer.put("runqueue_wait_ms",(end.rq-start.rq)*1000);answer.put("control",false);
  sendtree(fd,answer);close(fd);answer.put("type","SUBMISSION_OR_SNAPSHOT");answer.put("ros",ros::Time::now().toSec());log.push_back(answer);
 }
 close(server);std::ofstream f(run+"/DYNAMIC_GATE.jsonl");for(auto&t:log)f<<qm::dump(t);
}
int cpu(const std::string&ctxfile,const std::string&packetfile){
 auto cb=filebytes(ctxfile);auto t=qm::json(std::string(cb.begin(),cb.end()));auto c=context(t);Tree meta;std::vector<unsigned char>body;packet(filebytes(packetfile),meta,body);
 qm::Verifier verifier;verifier.highwater=t.get<unsigned long long>("highwater",0);
 auto sentinel=std::make_shared<const qm::Verified>(qm::Verified{});verifier.readonlyPending=sentinel;
 auto bs=load(t.get<std::string>("reference_file"));active=&bs[1];backup=&bs[1].stops[0];auto oldActive=active;auto oldBackup=backup;Tree answer;
 try{auto v=verifier.check(meta,body,c);answer.put("passed",true);answer.put("receipt",v->receipt);}catch(const std::exception&e){answer.put("passed",false);answer.put("reason",e.what());answer.put("readonly_latch_preserved",verifier.readonlyPending==sentinel);}
 answer.put("physical_reference_and_backup_preserved",active==oldActive&&backup==oldBackup);answer.put("control",false);std::cout<<qm::dump(answer);active=nullptr;backup=nullptr;return 0;
}
}
int main(int argc,char**argv){
 if(argc==4&&std::string(argv[1])=="--gate-cpu")return dyn::cpu(argv[2],argv[3]);
 if(argc==4&&std::string(argv[3])=="--reference-only"){
  const char*cfg=std::getenv("QM_DYNAMIC_CONFIG");if(!cfg)return 9;
  std::thread worker([&](){try{dyn::worker(argv[2],argv[1],cfg);}catch(const std::exception&e){std::ofstream f(std::string(argv[2])+"/DYNAMIC_FATAL.txt");f<<e.what();}});
  int result=qm_reference_only_main(argc,argv);finish=true;worker.join();return result;
 }
 return qm_reference_only_main(argc,argv);
}
