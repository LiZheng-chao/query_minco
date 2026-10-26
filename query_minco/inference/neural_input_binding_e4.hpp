#pragma once
#include "source_state_binding.hpp"
namespace qm {
Tree bindInputE4(const Tree&m,const PVA&source,const std::array<double,9>&R,
 const std::string&imageSha,double previousStamp,const Vec&previousVelocity){
 bindSource(m,source,R);
 if(m.get<std::string>("input_depth_mask_sha256")!=imageSha)throw std::runtime_error("DEPTH_MASK_DIGEST");
 double stamp=m.get<double>("depth_stamp");
 if(m.get<double>("camera_info_stamp")!=stamp||m.get<std::string>("camera_frame")!="camera_link")throw std::runtime_error("CAMERA_STAMP_OR_FRAME");
 if(std::fabs(m.get<double>("source_previous_stamp")-previousStamp)>1e-7)throw std::runtime_error("SOURCE_PREVIOUS_STAMP");
 auto pv=values(m,"source_previous_velocity");if(pv.size()!=3)throw std::runtime_error("SOURCE_PREVIOUS_SHAPE");
 for(int j=0;j<3;j++)if(std::fabs(pv[j]-previousVelocity[j])>1e-8)throw std::runtime_error("SOURCE_PREVIOUS_VELOCITY");
 double dt=stamp-previousStamp;if(dt<.005||dt>.05)throw std::runtime_error("SOURCE_HISTORY_TIME");
 for(int j=0;j<3;j++)if(std::fabs((source[3+j]-previousVelocity[j])/dt-source[6+j])>1e-7)throw std::runtime_error("SOURCE_ACCELERATION_HISTORY");
 Vec velocityError=vector(source,1);velocityError[0]-=.8;
 if(norm(velocityError)>.03||norm(vector(source,2))>.05||std::acos(std::max(-1.,std::min(1.,R[8])))>.008726646259971648)throw std::runtime_error("SOURCE_JOINT_SUPPORT");
 Tree audit,expectedTree,errorTree;
 Vec delta{{3-source[0],-source[1],2-source[2]}};double distance=norm(delta);
 auto supplied=values(m,"network_state");
 for(int d=0;d<4;d++)for(int j=0;j<3;j++){
  double expected=0;
  for(int k=0;k<3;k++)expected+=R[k*3+j]*(d==0?source[3+k]:d==1?source[6+k]:d==2?delta[k]*2.4/distance:(k==2?-1.:0.));
  Tree a,b;a.put("",expected);b.put("",std::fabs(expected-supplied[d*3+j]));expectedTree.push_back({"",a});errorTree.push_back({"",b});
 }
 audit.add_child("expected_state12",expectedTree);audit.add_child("absolute_field_errors12",errorTree);
 audit.put("depth_mask_sha256",imageSha);audit.put("source_stamp",stamp);audit.put("source_previous_stamp",previousStamp);audit.put("derivative_dt",dt);
 return audit;
}
}
