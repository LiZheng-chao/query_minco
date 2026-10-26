#pragma once
// Independent worker-side verification. No control authority, no arbitrary
// CERTIFIED_FREE field is consumed. Geometry/time/identity are checked anew.
#include "proposal_numerics.hpp"
#include <openssl/sha.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <memory>
#include <iomanip>
namespace qm {
using Tree=boost::property_tree::ptree;
std::string sha(const unsigned char*p,size_t n){unsigned char d[32];SHA256(p,n,d);std::ostringstream o;for(auto x:d)o<<std::hex<<std::setfill('0')<<std::setw(2)<<int(x);return o.str();}
std::string sha(const std::vector<unsigned char>&b){return sha(b.data(),b.size());}
Tree json(const std::string&s){std::istringstream in(s);Tree t;boost::property_tree::read_json(in,t);return t;}
std::string dump(const Tree&t){std::ostringstream o;boost::property_tree::write_json(o,t,false);return o.str();}
std::vector<double> values(const Tree&t,const std::string&name){std::vector<double>v;for(auto&x:t.get_child(name)){double a=x.second.get_value<double>();if(!std::isfinite(a))throw std::runtime_error("NONFINITE");v.push_back(a);}return v;}
struct Reader{
 const unsigned char*p;size_t left;
 template<class T>T get(){if(left<sizeof(T))throw std::runtime_error("TRUNCATED");T x;memcpy(&x,p,sizeof x);p+=sizeof x;left-=sizeof x;return x;}
 Curve curve(){
  int n=get<int>();if(n<1||n>16)throw std::runtime_error("PIECE_BOUND");Curve c;
  for(int i=0;i<n;i++){Piece a;a.T=get<double>();for(auto&x:a.c)x=get<double>();c.p.push_back(a);}return c;
 }
 Bank bank(){Bank b;b.prefix=curve();int n=get<int>();if(n<2||n>1025)throw std::runtime_error("PHASE_BOUND");for(int i=0;i<n;i++){double t=get<double>();b.stops.push_back({t,curve()});}if(left)throw std::runtime_error("TRAILING_BYTES");return b;}
};
struct Context{
 double nowR=0,nowM=0,stamp=0,recv=0,parentStartR=0,cellExpiryR=0,cellExpiryM=0;
 PVA pva{};Vec sourcePosition{};std::array<double,9> sourceRotation{{1,0,0,0,1,0,0,0,1}};
 std::string epoch,geometry,calibration,parentDigest;unsigned long long parentVersion=1,token=0;
 bool publisherUnique=true,clockValid=true,sourceSeen=true,calibrationValid=true;
 Curve parent;
};
struct Verified{
 unsigned long long sequence;Tree meta;Bank bank;std::string bodyDigest,receipt;
};
struct Verifier{
 unsigned long long highwater=0;std::shared_ptr<const Verified> readonlyPending;
 std::shared_ptr<const Verified> check(const Tree&m,const std::vector<unsigned char>&bytes,const Context&c){
  auto require=[](bool b,const char*s){if(!b)throw std::runtime_error(s);};
  require(m.get<int>("stop_bank_version")==1,"STOP_BANK_VERSION");require(c.clockValid,"CLOCK_JUMP");require(c.publisherUnique,"PUBLISHER_CONFLICT");require(c.sourceSeen&&c.calibrationValid,"UNOBSERVED_SOURCE_OR_CALIBRATION");
  require(m.get<std::string>("method")=="query_minco_b2c1_c1_e1_v1","METHOD");
  require(m.get<std::string>("checkpoint")=="2352de6dd1e4a839b78ec12c115b5319b34da3c61cada105dbdeb24a4faec6d1","CHECKPOINT");
  require(m.get<std::string>("frame")=="world:B0:optical_depth_128x160","FRAME");
  require(m.get<std::string>("calibration")==c.calibration,"CALIBRATION");
  require(m.get<std::string>("owner_epoch")==c.epoch,"OWNER_EPOCH");
  require(m.get<unsigned long long>("parent_version")==c.parentVersion&&m.get<std::string>("parent_digest")==c.parentDigest,"PARENT_VERSION");
  require(m.get<unsigned long long>("snapshot_token")==c.token,"CAUSAL_SNAPSHOT_TOKEN");
  unsigned long long seq=m.get<unsigned long long>("sequence");require(seq>highwater,"DUPLICATE_OR_OUT_OF_ORDER");
  double depth=m.get<double>("depth_stamp"),odom=m.get<double>("source_odom_stamp"),pr=m.get<double>("prepared_ros"),pm=m.get<double>("prepared_mono"),ar=m.get<double>("activation_ros"),am=m.get<double>("activation_mono"),vr=m.get<double>("valid_until_ros"),vm=m.get<double>("valid_until_mono");
  for(double x:{depth,odom,pr,pm,ar,am,vr,vm,c.nowR,c.nowM})require(std::isfinite(x),"NONFINITE");
  require(depth==odom&&c.nowR-depth>=0&&c.nowR-depth<=.25,"SOURCE_TIME");
  require(c.nowR-c.stamp>=0&&c.nowR-c.stamp<=.05&&c.nowM-c.recv>=0&&c.nowM-c.recv<=.05,"STALE_STATE");
  require(std::fabs(m.get<double>("state_stamp")-c.stamp)<1e-7&&std::fabs(m.get<double>("state_recv_mono")-c.recv)<1e-7,"STATE_RECEIPT");
  auto state=values(m,"planning_pva");require(state.size()==9,"STATE_SHAPE");for(int i=0;i<9;i++)require(std::fabs(state[i]-c.pva[i])<1e-8,"CAUSAL_STATE_MISMATCH");
  auto position=values(m,"source_position"),rotation=values(m,"source_rotation");
  require(position.size()==3&&rotation.size()==9,"SOURCE_POSE_SHAPE");
  for(int i=0;i<3;i++)require(std::fabs(position[i]-c.sourcePosition[i])<1e-7,"COORDINATE_TRANSFORM");
  for(int i=0;i<9;i++)require(std::fabs(rotation[i]-c.sourceRotation[i])<1e-7,"COORDINATE_TRANSFORM");
  require(pr<=c.nowR&&pm<=c.nowM&&c.nowR-pr<=.04&&c.nowM-pm<=.04,"PREPARATION_TIME");
  require(am-c.nowM>=.02-1e-7&&ar-c.nowR>=.02-1e-7&&ar-pr>=.02&&ar-pr<=.1+1e-7,"READY_DEADLINE");
  require(std::fabs((ar-pr)-(am-pm))<=.005&&std::fabs((c.nowR-pr)-(c.nowM-pm))<=.005,"CLOCK_OFFSET");
  require(m.get<std::string>("geometry")==c.geometry,"GEOMETRY_ID");
  require(vr<=c.cellExpiryR-.000999&&vm<=c.cellExpiryM-.000999&&vr>c.nowR&&vm>c.nowM,"EXPIRED_OR_EXTENDED_CERTIFICATE");
  Cell cell;cell.expiry=c.cellExpiryR;auto errors=values(m,"error_budget");require(errors.size()==8,"ERROR_BUDGET");
  for(int i=0;i<8;i++)require(errors[i]==cell.errors[i],"ERROR_BUDGET");
  auto digest=sha(bytes);require(m.get<std::string>("body_sha256")==digest,"TRAJECTORY_DIGEST");
  Reader reader{bytes.data(),bytes.size()};auto b=reader.bank();
  validate(b.prefix);int count=int(std::llround(b.prefix.duration()/.02));
  require(std::fabs(count*.02-b.prefix.duration())<1e-9&&(int)b.stops.size()==count+1,"MISSING_STOP_PHASE");
  int nodes=0;double slack=sweep(b.prefix,cell,nodes);
  for(int i=0;i<=count;i++){
   const auto&x=b.stops[i];require(std::fabs(x.phase-i*.02)<1e-9,"STOP_PHASE_ORDER");
   validate(x.stop,.35);auto h=b.prefix.at(x.phase),s=x.stop.at(0),tail=x.stop.at(x.stop.duration());
   require(gap(h,s,0)<=.0002&&gap(h,s,1)<=.002&&gap(h,s,2)<=.01,"STOP_HEAD_PVA");
   require(norm(vector(tail,1))<1e-8&&norm(vector(tail,2))<1e-8,"STOP_TERMINAL_VA");
   slack=std::min(slack,sweep(x.stop,cell,nodes));b.horizon=std::max(b.horizon,x.phase+x.stop.duration());
  }
  require(ar+b.horizon<=vr&&am+b.horizon<=vm,"COMPLETE_STOP_EXPIRY");
  auto expected=c.parent.at(ar-c.parentStartR),head=b.prefix.at(0);
  require(gap(expected,head,0)<=.0002&&gap(expected,head,1)<=.002&&gap(expected,head,2)<=.01,"PARENT_HEAD_PVA");
  PVA predicted=c.pva;double dt=ar-c.stamp;
  require(dt>=0&&dt<=.15,"PREDICTION_TIME");
  for(int j=0;j<3;j++){predicted[j]+=predicted[3+j]*dt+.5*predicted[6+j]*dt*dt;predicted[3+j]+=predicted[6+j]*dt;}
  require(gap(predicted,head,0)<=.02&&gap(predicted,head,1)<=.05&&gap(predicted,head,2)<=.1,"ACTIVATION_PREDICTED_TUBE");
  b.slack=slack;b.nodes=nodes;
  std::vector<unsigned char>bound(bytes);auto identity=dump(m);bound.insert(bound.end(),identity.begin(),identity.end());
  auto verified=std::make_shared<const Verified>(Verified{seq,m,b,digest,sha(bound)});
  // Readonly latch only. Physical active reference and backup are never touched.
  highwater=seq;readonlyPending=verified;return verified;
 }
};
}
