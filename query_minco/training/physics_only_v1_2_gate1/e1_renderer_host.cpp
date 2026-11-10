// CPU generation/PLY/maps stay in the ordinary C++ translation unit.
#include "maps.hpp"
#include <pcl/common/common.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
extern "C" int e1_export(const float*,size_t,const char*,int,const float*);
namespace fs=std::filesystem;
int main(int argc,char**argv){
 try{
 if(argc!=5)throw std::runtime_error("usage: bridge counts yopo_root scene_yaml out OR bridge depth cloud pose_yaml out");
 std::string mode=argv[1];fs::path out=argv[4];
 if(!fs::is_directory(out)||!fs::is_empty(out))throw std::runtime_error("EXCLUSIVE_EMPTY_OUT_REQUIRED");
 std::vector<float> xyz;float pos[]={0,0,0};
 if(mode=="counts"){
  auto spec=YAML::LoadFile(argv[3]);std::string id=spec["scene_id"].as<std::string>();
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  if(id.find("_sparse_")!=std::string::npos){
   fs::path root=argv[2];auto config=YAML::LoadFile((root/"Simulator/src/config/config.yaml").string());
   config["tree_file"]=(root/"Simulator/src/pointcloud/tree.ply").string();config["tree_dist"]=16.;
   mocka::Maps::BasicInfo info;info.sizeX=160;info.sizeY=160;info.sizeZ=150;info.seed=spec["seed"].as<int>();info.scale=10.;info.cloud=cloud;
   mocka::Maps maps;maps.setParam(config);maps.setInfo(info);maps.generate(5);
  }else{
   // Frozen world floor and retained far boundary corners establish native top.
   for(int i=0;i<=140;i++)for(int j=0;j<=120;j++)cloud->push_back(pcl::PointXYZ(-2.f+i*.1f,-6.f+j*.1f,0.f));
   for(float x:{-2.f,12.f})for(float y:{-6.f,6.f})cloud->push_back(pcl::PointXYZ(x,y,6.f));
   bool train=id.rfind("train_",0)==0;bool slit=id.find("single_slit")!=std::string::npos;
   if(slit||id.find("two_portals")!=std::string::npos){
    auto layout=spec["layout"];auto box=layout[slit?"slab_box_m":"block_box_m"];
    float lo[3],hi[3];for(int k=0;k<3;k++){lo[k]=box[0][k].as<float>();hi[k]=box[1][k].as<float>();}
    for(int i=0;i<static_cast<int>(std::round((hi[0]-lo[0])/.1f));i++)
     for(int j=0;j<static_cast<int>(std::round((hi[1]-lo[1])/.1f));j++)
      for(int k=0;k<static_cast<int>(std::round((hi[2]-lo[2])/.1f));k++){
       float x=lo[0]+(i+.5f)*.1f,y=lo[1]+(j+.5f)*.1f,z=lo[2]+(k+.5f)*.1f;
       if(slit && y>=layout["remove_y_interval_m"][0].as<float>() && y<=layout["remove_y_interval_m"][1].as<float>())continue;
       cloud->push_back(pcl::PointXYZ(x,y,z));
      }
   }
  }
  for(const auto&p:*cloud){xyz.push_back(p.x);xyz.push_back(p.y);xyz.push_back(p.z);}
 }else if(mode=="depth"){
  auto cert=YAML::LoadFile(argv[3]);
  if(cert["schema"].as<std::string>()!="e1_r2_pose_v1" || !cert["passed"].as<bool>() || cert["clearance_lower_bound_m"].as<double>()<.150001)
   throw std::runtime_error("POSE_CERTIFICATE");
  for(int i=0;i<3;i++){pos[i]=cert["position_world"][i].as<float>();if(!std::isfinite(pos[i]))throw std::runtime_error("POSE_NONFINITE");
   for(int j=0;j<3;j++)if(cert["R_WB0"][i][j].as<double>()!=(i==j?1.:0.))throw std::runtime_error("POSE_ROTATION");}
  std::ifstream f(argv[2],std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("CLOUD_OPEN");
  auto n=f.tellg();if(n<=0||static_cast<size_t>(n)%12)throw std::runtime_error("CLOUD_SIZE");
  xyz.resize(static_cast<size_t>(n)/4);f.seekg(0);f.read(reinterpret_cast<char*>(xyz.data()),n);if(!f)throw std::runtime_error("CLOUD_READ");
 }else throw std::runtime_error("MODE");
 return e1_export(xyz.data(),xyz.size()/3,out.string().c_str(),mode=="depth",pos);
 }catch(const std::exception&e){std::cerr<<"STOP "<<e.what()<<"\n";return 1;}
}
