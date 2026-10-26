#pragma once
#include "dynamic_proposal_gate.hpp"
namespace qm {
// Additive v2 verifier: independently bind the actual12D neural input to trusted
// source-time Odom PVA, not merely to the later execution-head snapshot.
void bindSource(const Tree&m,const PVA&source,const std::array<double,9>&R){
 auto require=[](bool ok,const char*s){if(!ok)throw std::runtime_error(s);};
 require(m.get<int>("source_state_version",0)==2,"SOURCE_STATE_BINDING_REQUIRED");
 auto supplied=values(m,"source_pva"),state=values(m,"network_state");
 require(supplied.size()==9&&state.size()==12,"SOURCE_STATE_SHAPE");
 for(int i=0;i<9;i++)require(std::fabs(supplied[i]-source[i])<1e-7,"SOURCE_PVA_MISMATCH");
 std::array<double,12>expected{};Vec delta{{3-source[0],-source[1],2-source[2]}};
 double distance=norm(delta);require(distance>=2.4,"TERMINAL_SOURCE_GOAL");
 for(int j=0;j<3;j++)for(int k=0;k<3;k++){
  expected[j]+=R[k*3+j]*source[3+k];expected[3+j]+=R[k*3+j]*source[6+k];
  expected[6+j]+=R[k*3+j]*delta[k]*2.4/distance;
 }
 for(int j=0;j<3;j++)expected[9+j]=-R[6+j];
 for(int j=0;j<12;j++)require(std::fabs(expected[j]-state[j])<2e-6,"NEURAL_INPUT_SOURCE_STATE_MISMATCH");
 std::vector<unsigned char>bytes;for(double x:state){float f=x;put<float>(bytes,f);}
 require(m.get<std::string>("network_state_sha256")==sha(bytes),"NEURAL_STATE_DIGEST");
}
}
