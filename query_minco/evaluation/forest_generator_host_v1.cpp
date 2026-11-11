// Read-only YOPO generator bridge: no ROS node, collector, cleanup or original output write.
#include "maps.hpp"
#include <pcl/common/common.h>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
#include <cmath>
#include <vector>
namespace fs=std::filesystem;
int main(int argc,char**argv){
 try {
  if(argc!=4)throw std::runtime_error("usage: forest_host yopo_root spec_json exclusive_out");
  fs::path root=argv[1],out=argv[3];
  if(!fs::is_directory(out)||!fs::is_empty(out))throw std::runtime_error("EXCLUSIVE_EMPTY_OUT_REQUIRED");
  auto spec=YAML::LoadFile(argv[2]);auto config=YAML::LoadFile((root/"Simulator/src/config/config.yaml").string());
  if(config["maze_type"].as<int>()!=5||std::abs(config["resolution"].as<double>()-.1)>1e-8)throw std::runtime_error("SOURCE_PROFILE");
  config["tree_file"]=(root/"Simulator/src/pointcloud/tree.ply").string();
  config["tree_dist"]=spec["tree_dist_m"].as<double>();
  double width=spec["map_width_m"].as<double>();
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  mocka::Maps::BasicInfo info;info.sizeX=static_cast<int>(std::round(width*10));info.sizeY=info.sizeX;info.sizeZ=150;
  info.seed=spec["seed"].as<int>();info.scale=10.;info.cloud=cloud;
  mocka::Maps maps;maps.setParam(config);maps.setInfo(info);maps.generate(5);
  if(cloud->empty()||cloud->size()>6000000)throw std::runtime_error("SOURCE_POINT_BUDGET");
  int fd=::open((out/"source_cloud.f32").c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
  if(fd<0)throw std::runtime_error("EXCLUSIVE_FILE_CREATE");
  std::vector<float> xyz;xyz.reserve(cloud->size()*3);
  for(const auto&p:*cloud){
   if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z)){::close(fd);throw std::runtime_error("NONFINITE_POINT");}
   xyz.push_back(p.x);xyz.push_back(p.y);xyz.push_back(p.z);
  }
  const char* bytes=reinterpret_cast<const char*>(xyz.data());size_t remaining=xyz.size()*sizeof(float);
  while(remaining){auto written=::write(fd,bytes,remaining);if(written<=0){::close(fd);throw std::runtime_error("SOURCE_WRITE");}bytes+=written;remaining-=written;}
  ::fsync(fd);::close(fd);
  std::cout<<"GENERATED seed="<<info.seed<<" tree_dist="<<spec["tree_dist_m"].as<double>()<<" points="<<cloud->size()<<"\n";
  return 0;
 } catch(const std::exception&e){std::cerr<<"STOP "<<e.what()<<"\n";return 1;}
}
