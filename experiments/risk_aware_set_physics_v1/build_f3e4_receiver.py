"""Add source/image binding + isolated ROS queue to exact existing readonly core."""
import json,subprocess
from pathlib import Path
ROOT=Path('/home/lizheng/query_minco_mainline');OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution008'
def render():
 base=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution007/runtime/native_reference_dynamic_source_bound.cpp'
 s=base.read_text().replace('#include "source_state_binding.hpp"','#include "source_state_binding.hpp"\n#include "neural_input_binding_e4.hpp"\n#include <ros/callback_queue.h>')
 s=s.replace('double stamp=0,recv=0;bool valid=false;','double stamp=0,recv=0,previousStamp=0;qm::Vec previousVelocity{};bool valid=false;')
 s=s.replace('std::deque<double>depths,infos;','std::deque<double>depths,infos;std::deque<std::pair<double,std::string>>depthHashes;')
 s=s.replace('double dt=o.stamp-laststamp;','double dt=o.stamp-laststamp;o.previousStamp=laststamp;o.previousVelocity=qm::vector(lastpv,1);')
 # Same real PVA calculation, strict older history and no output on reset.
 s=s.replace('if(dt>=.005){o.valid=', 'if(dt<=0){observations.clear();laststamp=0;return;}if(dt>=.005){o.valid=')
 old='void image(const sensor_msgs::Image::ConstPtr&m){std::lock_guard<std::mutex>l(sensorMutex);if(m->width==160&&m->height==128&&m->encoding=="32FC1"&&m->step==640&&!m->is_bigendian){depths.push_back(m->header.stamp.toSec());if(depths.size()>64)depths.pop_front();}}'
 new='''std::string encodedHash(const sensor_msgs::Image&m){
 if(m.data.size()!=128*160*4)throw std::runtime_error("IMAGE_BYTES");
 std::vector<float>encoded(2*128*160);for(int i=0;i<128*160;i++){float v;memcpy(&v,m.data.data()+4*i,4);bool valid=std::isfinite(v)&&v>0&&v<20;encoded[i]=valid?v/20.f:0.f;encoded[128*160+i]=valid?1.f:0.f;}
 return qm::sha((const unsigned char*)encoded.data(),encoded.size()*4);
}
void image(const sensor_msgs::Image::ConstPtr&m){if(m->width!=160||m->height!=128||m->encoding!="32FC1"||m->step!=640||m->is_bigendian||m->header.frame_id!="camera_link")return;
 auto digest=encodedHash(*m);std::lock_guard<std::mutex>l(sensorMutex);depths.push_back(m->header.stamp.toSec());depthHashes.push_back({m->header.stamp.toSec(),digest});if(depths.size()>64)depths.pop_front();if(depthHashes.size()>64)depthHashes.pop_front();}'''
 assert old in s;s=s.replace(old,new)
 s=s.replace('ros::NodeHandle n;auto os=', 'ros::CallbackQueue workerQueue;ros::NodeHandle n;n.setCallbackQueue(&workerQueue);ros::AsyncSpinner workerSpinner(1,&workerQueue);workerSpinner.start();auto os=')
 s=s.replace('"qm_f3e007_"','"qm_f3e008_"')
 # Expose actual previous sample identity from the I-source RPC.
 needle='answer.add_child("pva",pv);answer.add_child("rotation",rotation);'
 replacement=needle+'''answer.put("previous_stamp",observed.previousStamp);Tree previousVelocity;for(auto x:observed.previousVelocity){Tree v;v.put("",x);previousVelocity.push_back({"",v});}answer.add_child("previous_velocity",previousVelocity);
    std::string imageSha;{std::lock_guard<std::mutex>l(sensorMutex);auto found=std::find_if(depthHashes.begin(),depthHashes.end(),[&](auto&h){return h.first==sourceStamp;});if(found==depthHashes.end())throw std::runtime_error("SOURCE_IMAGE_NOT_OBSERVED");imageSha=found->second;
    if(std::find(infos.begin(),infos.end(),sourceStamp)==infos.end()||!cameraValid)throw std::runtime_error("SOURCE_CAMERA_NOT_OBSERVED");}
    answer.put("input_depth_mask_sha256",imageSha);'''
 assert needle in s;s=s.replace(needle,replacement)
 s=s.replace('qm::PVA trustedSource{};qm::Context c;', 'qm::PVA trustedSource{};double previousSourceStamp=0;qm::Vec previousSourceVelocity{};std::string trustedDepthSha;qm::Context c;')
 s=s.replace('trustedSource=found->pva;c.sourceSeen=true;', 'trustedSource=found->pva;previousSourceStamp=found->previousStamp;previousSourceVelocity=found->previousVelocity;auto dh=std::find_if(depthHashes.begin(),depthHashes.end(),[&](auto&h){return h.first==source;});if(dh!=depthHashes.end())trustedDepthSha=dh->second;c.sourceSeen=true;')
 s=s.replace('qm::bindSource(meta,trustedSource,c.sourceRotation);auto verified=', 'auto inputAudit=qm::bindInputE4(meta,trustedSource,c.sourceRotation,trustedDepthSha,previousSourceStamp,previousSourceVelocity);auto verified=')
 s=s.replace('answer.put("scope","READONLY_LATCH_NOT_PHYSICAL");','answer.put("scope","READONLY_LATCH_NOT_PHYSICAL");answer.add_child("input_audit",inputAudit);')
 s=s.replace('qm::bindSource(meta,trustedSource,c.sourceRotation);auto v=', 'qm::Vec previousVelocity{};auto pp=qm::values(t,"source_previous_velocity");std::copy(pp.begin(),pp.end(),previousVelocity.begin());auto audit=qm::bindInputE4(meta,trustedSource,c.sourceRotation,t.get<std::string>("input_depth_mask_sha256"),t.get<double>("source_previous_stamp"),previousVelocity);auto v=')
 s=s.replace('answer.put("receipt",v->receipt);','answer.put("receipt",v->receipt);answer.add_child("input_audit",audit);')
 s=s.replace('close(server);std::ofstream f(', 'workerSpinner.stop();close(server);std::ofstream f(')
 s=s.replace('int main(int argc,char**argv){', 'int main(int argc,char**argv){\\n if(argc==3&&std::string(argv[1])=="--input-image-sha"){sensor_msgs::Image image;image.data=dyn::filebytes(argv[2]);std::cout<<dyn::encodedHash(image);return 0;}')
 return s.replace(chr(92)+'n if(argc',chr(10)+' if(argc')
def main():
 cpp=OUT/'runtime/native_reference_dynamic_e4.cpp';cpp.write_text(render())
 command=['g++','-O3','-std=c++17','-pthread','-I'+str(ROOT/'query_minco/inference'),'-I/opt/ros/noetic/include','-I/home/lizheng/YOPO/Controller/devel/include',str(cpp),'-o',str(OUT/'runtime/native_reference_dynamic_e4'),'-L/opt/ros/noetic/lib','-Wl,-rpath,/opt/ros/noetic/lib','-lroscpp','-lrosconsole','-lroscpp_serialization','-lrostime','-lxmlrpcpp','-lcrypto']
 r=subprocess.run(command,capture_output=True,text=True);(OUT/'logs/build_receiver.log').write_text(r.stdout+r.stderr);(OUT/'BUILD_COMMAND.json').write_text(json.dumps(command,indent=2))
 if r.returncode:raise RuntimeError(r.stderr)
 print('source-bound/image-bound readonly receiver compiled, own ROS callback queue')
if __name__=='__main__':main()
