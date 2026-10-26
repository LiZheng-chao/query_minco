#include "proposal_numerics.hpp"
extern "C" int qm_prepare(const double*data,int n,double now,double expiry,unsigned char*out,int cap,double*metrics,char*error,int errcap){
 try{
  qm::Curve p;for(int i=0;i<n;i++){qm::Piece x;x.T=data[i*19];for(int j=0;j<18;j++)x.c[j]=data[i*19+1+j];p.p.push_back(x);}
  qm::Cell cell;cell.expiry=expiry;std::vector<std::array<double,3>>timing;auto b=qm::prepare(p,cell,now,timing);
  auto t=qm::mark();auto bytes=qm::encode(b);qm::delta(timing,t);
  if((int)bytes.size()>cap)throw std::runtime_error("BUFFER_CAP");
  std::memcpy(out,bytes.data(),bytes.size());qm::delta(timing,t);
  for(size_t i=0;i<timing.size();i++)for(int k=0;k<3;k++)metrics[i*3+k]=timing[i][k]*1000;
  metrics[30]=b.horizon;metrics[31]=b.slack;metrics[32]=b.nodes;metrics[33]=b.stops.size();
  return bytes.size();
 }catch(const std::exception&e){snprintf(error,errcap,"%s",e.what());return -1;}
}
