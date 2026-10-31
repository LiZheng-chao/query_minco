// Calls the unchanged native CUDA shared library. GT clouds are offline fixtures only.
#include "sensor_simulator.cuh"
#include <fstream>
#include <string>
using namespace raycast;
void add(pcl::PointCloud<pcl::PointXYZ>::Ptr c,float x,float y,float z,int n=2){for(int i=0;i<n;i++)c->push_back(pcl::PointXYZ(x,y,z));}
int main(int argc,char**argv){
 if(argc!=2)return 2; std::string dir=argv[1];
 for(int scenario=0;scenario<3;scenario++){
  pcl::PointCloud<pcl::PointXYZ>::Ptr c(new pcl::PointCloud<pcl::PointXYZ>);
  add(c,-1.05f,-20.05f,-14.05f);add(c,21.05f,20.05f,18.05f);
  for(int yi=-200;yi<=200;yi++){
   for(int zi=0;zi<=40;zi++)add(c,18.f,yi*.1f,zi*.1f);
   for(int xi=-10;xi<=180;xi++){add(c,xi*.1f,yi*.1f,0.f);add(c,xi*.1f,yi*.1f,4.f);}
  }
  if(scenario)add(c,15.f,0.f,2.f,scenario==1?2:1);
  GridMap g(c,.1f,1);CameraParams k;k.fx=400;k.fy=400;k.cx=400;k.cy=254;k.image_width=800;k.image_height=512;k.max_depth_dist=20;k.normalize_depth=false;
  cudaMat::SE3<float>T(1,0,0,0,0,0,2);cv::Mat depth;renderDepthImage(&g,&k,T,depth);
  cudaError_t err=cudaGetLastError();if(err!=cudaSuccess){std::cerr<<"CUDA_ERROR "<<cudaGetErrorString(err)<<std::endl;return 3;}
  std::ofstream f(dir+"/native"+std::to_string(scenario)+".f32",std::ios::binary);f.write((char*)depth.data,800*512*sizeof(float));g.freeGridMap();
 }
 return 0;
}
