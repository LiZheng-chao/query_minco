#pragma once
// Worker-only continuous bounds and same soft-stop family. No ROS or controls.
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <chrono>
#include <ctime>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <fcntl.h>
namespace qm {
using Vec=std::array<double,3>;using PVA=std::array<double,9>;
struct Piece{double T=0;std::array<double,18> c{};};
struct Curve{
 std::vector<Piece> p;double duration()const{double t=0;for(auto&x:p)t+=x.T;return t;}
 PVA at(double t)const{
  if(t< -1e-9||t>duration()+1e-8)throw std::runtime_error("EXTRAPOLATION");
  const Piece*a=&p.back();for(auto&v:p){a=&v;if(t<=v.T+1e-10)break;t-=v.T;}
  t=std::max(0.,std::min(t,a->T));PVA out{};
  for(int d=0;d<3;d++)for(int j=0;j<3;j++)for(int k=5;k>=d;k--){double f=1;for(int l=0;l<d;l++)f*=k-l;out[d*3+j]=out[d*3+j]*t+a->c[j*6+k]*f;}return out;
 }
};
double norm(const Vec&a){double v=0;for(double x:a)v+=x*x;return std::sqrt(v);}
Vec vector(const PVA&a,int d){return Vec{{a[3*d],a[3*d+1],a[3*d+2]}};}
double gap(const PVA&a,const PVA&b,int d){Vec x{};for(int j=0;j<3;j++)x[j]=a[d*3+j]-b[d*3+j];return norm(x);}
double choose(int n,int k){double a=1;for(int i=1;i<=k;i++)a*=double(n+1-i)/i;return a;}
using Hull=std::vector<Vec>;
Hull controls(const Piece&p,int derivative){
 int n=5-derivative;Hull h(n+1);
 for(int i=0;i<=n;i++)for(int k=0;k<=i;k++){
  double factor=1;for(int l=0;l<derivative;l++)factor*=k+derivative-l;
  double f=std::pow(p.T,k)*factor*choose(i,k)/choose(n,k);
  for(int j=0;j<3;j++)h[i][j]+=p.c[j*6+k+derivative]*f;
 }return h;
}
std::pair<Hull,Hull> split(Hull h){
 Hull left{h.front()},right{h.back()};
 while(h.size()>1){Hull next(h.size()-1);for(size_t i=0;i<next.size();i++)for(int j=0;j<3;j++)next[i][j]=(h[i][j]+h[i+1][j])*.5;h=next;left.push_back(h.front());right.push_back(h.back());}
 std::reverse(right.begin(),right.end());return {left,right};
}
double motionHull(const Hull&h,double limit,int depth=0){
 double upper=0;for(auto&v:h)upper=std::max(upper,norm(v));
 if(upper<=limit+1e-10)return upper;
 if(norm(h.front())>limit+1e-8||norm(h.back())>limit+1e-8||depth==12)throw std::runtime_error("MOTION_BOUND");
 auto halves=split(h);return std::max(motionHull(halves.first,limit,depth+1),motionHull(halves.second,limit,depth+1));
}
void validate(const Curve&c,double jerk=10.){
 if(c.p.empty()||c.p.size()>16)throw std::runtime_error("PIECES");
 for(size_t i=0;i<c.p.size();i++){
  auto&p=c.p[i];if(!(p.T>0&&p.T<100))throw std::runtime_error("DURATION");
  for(auto x:p.c)if(!std::isfinite(x))throw std::runtime_error("COEFF");
  motionHull(controls(p,1),1);motionHull(controls(p,2),1.5);motionHull(controls(p,3),jerk);
  if(i){Curve a{{c.p[i-1]}},b{{p}};auto x=a.at(a.duration()),y=b.at(0);
   if(gap(x,y,0)>.0002||gap(x,y,1)>.002||gap(x,y,2)>.01)throw std::runtime_error("PVA_GAP");}
 }
}
struct Cell{Vec lo{{-2.8,-2.8,.2}},hi{{6.8,2.8,3.8}};double expiry=0;std::array<double,8> errors{{.12,.03,.02,.02,.005,.008726646259971648,.15,.000001}};};
double radius(const Cell&c,double distance){return c.errors[0]+c.errors[1]+c.errors[2]+c.errors[3]+c.errors[4]+2*distance*std::sin(c.errors[5]/2);}
double sweepHull(const Hull&h,const Cell&c,int&nodes,int depth=0){
 if(++nodes>4096)throw std::runtime_error("SWEEP_BUDGET");
 double distance=0;for(auto&v:h)distance=std::max(distance,norm(v));
 double r=radius(c,distance)+c.errors[7];double slack=1e30;
 for(int j=0;j<3;j++){double lo=1e30,hi=-1e30;for(auto&v:h){lo=std::min(lo,v[j]);hi=std::max(hi,v[j]);}slack=std::min(slack,std::min(lo-c.lo[j],c.hi[j]-hi)-r);}
 if(slack>=0)return slack;
 if(depth==12)throw std::runtime_error("GEOMETRY_UNPROVEN");
 auto halves=split(h);return std::min(sweepHull(halves.first,c,nodes,depth+1),sweepHull(halves.second,c,nodes,depth+1));
}
double sweep(const Curve&c,const Cell&cell,int&nodes){double slack=1e30;for(auto&p:c.p)slack=std::min(slack,sweepHull(controls(p,0),cell,nodes));return slack;}
Curve softStop(const PVA&h){
 Vec v=vector(h,1),a=vector(h,2),pos=vector(h,0);double an=norm(a),dot=0;for(int j=0;j<3;j++)dot+=v[j]*a[j];
 Curve trans;
 if(dot>0&&an>.02){
  double T=an/.35;Piece tr;tr.T=T;
  for(int j=0;j<3;j++){tr.c[j*6]=pos[j];tr.c[j*6+1]=v[j];tr.c[j*6+2]=a[j]/2;tr.c[j*6+3]=-a[j]/(6*T);}
  trans.p.push_back(tr);validate(trans,.35);auto end=trans.at(T);pos=vector(end,0);v=vector(end,1);a=vector(end,2);
 }
 for(int k=0;k<=40;k++){
  double T=2.+.1*k;Vec j0{},j1{};
  for(int j=0;j<3;j++){j0[j]=-6*v[j]/(T*T)-4*a[j]/T;j1[j]=6*v[j]/(T*T)+2*a[j]/T;}
  if(std::max(norm(j0),norm(j1))>.35+1e-12)continue;
  Piece p;p.T=T;for(int j=0;j<3;j++){p.c[j*6]=pos[j];p.c[j*6+1]=v[j];p.c[j*6+2]=a[j]/2;p.c[j*6+3]=-v[j]/(T*T)-2*a[j]/(3*T);p.c[j*6+4]=v[j]/(2*T*T*T)+a[j]/(4*T*T);}
  Curve stop{{p}};
  try{validate(stop,.35);}catch(const std::exception&){continue;}
  trans.p.push_back(p);return trans;
 }throw std::runtime_error("NO_SOFT_STOP");
}
struct Reserve{double phase;Curve stop;};
struct Bank{Curve prefix;std::vector<Reserve> stops;double horizon=0,slack=0;int nodes=0;};
struct Mark{double wall,cpu,rq;};
Mark mark(){
 timespec a,b;clock_gettime(CLOCK_MONOTONIC,&a);clock_gettime(CLOCK_THREAD_CPUTIME_ID,&b);
 char buf[160];double rq=0;int fd=open("/proc/thread-self/schedstat",O_RDONLY);
 if(fd>=0){int n=read(fd,buf,159);close(fd);if(n>0){buf[n]=0;unsigned long long x,y,z;if(sscanf(buf,"%llu %llu %llu",&x,&y,&z)==3)rq=y*1e-9;}}
 return {a.tv_sec+a.tv_nsec*1e-9,b.tv_sec+b.tv_nsec*1e-9,rq};
}
void delta(std::vector<std::array<double,3>>&m,Mark&a){auto b=mark();m.push_back({{b.wall-a.wall,b.cpu-a.cpu,b.rq-a.rq}});a=b;}
Bank prepare(const Curve&prefix,const Cell&cell,double now,std::vector<std::array<double,3>>&timing){
 auto t=mark();Bank b;b.prefix=prefix;validate(prefix);delta(timing,t);
 int count=int(std::llround(prefix.duration()/.02));if(count<1||count>1024||std::fabs(count*.02-prefix.duration())>1e-9)throw std::runtime_error("PHASE_GRID");
 std::vector<PVA>heads;for(int k=0;k<=count;k++)heads.push_back(prefix.at(k*.02));delta(timing,t);
 for(int k=0;k<=count;k++){auto stop=softStop(heads[k]);b.horizon=std::max(b.horizon,k*.02+stop.duration());b.stops.push_back({k*.02,stop});}
 delta(timing,t);
 for(size_t k=0;k<b.stops.size();k++){
  auto&s=b.stops[k];auto head=s.stop.at(0),tail=s.stop.at(s.stop.duration());
  if(gap(head,heads[k],0)>.0002||gap(head,heads[k],1)>.002||gap(head,heads[k],2)>.01||norm(vector(tail,1))>1e-8||norm(vector(tail,2))>1e-8)throw std::runtime_error("RESERVE_PVA");
 }
 if(b.horizon+.06>std::min(120.,cell.expiry-now-.001))throw std::runtime_error("CELL_EXPIRY");
 delta(timing,t);
 b.slack=sweep(prefix,cell,b.nodes);
 for(auto&s:b.stops)b.slack=std::min(b.slack,sweep(s.stop,cell,b.nodes));
 delta(timing,t);return b;
}
template<class T>void put(std::vector<unsigned char>&out,T x){auto p=(unsigned char*)&x;out.insert(out.end(),p,p+sizeof(T));}
void putCurve(std::vector<unsigned char>&out,const Curve&p){put<int>(out,p.p.size());for(auto&a:p.p){put<double>(out,a.T);for(auto x:a.c)put<double>(out,x);}}
std::vector<unsigned char> encode(const Bank&b){std::vector<unsigned char>out;putCurve(out,b.prefix);put<int>(out,b.stops.size());for(auto&s:b.stops){put<double>(out,s.phase);putCurve(out,s.stop);}return out;}
}
