// Thin ABI only. GridMap and renderDepthImage are linked from frozen native object.
#include <cuda_runtime.h>
#include "cuda_toolkit/se3.cuh"
#include <opencv2/opencv.hpp>
#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <chrono>
#include <vector>
#include <iostream>
#include <cmath>
#define private public
#include "sensor_simulator.cuh"
#undef private
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
namespace fs=std::filesystem;
static void ok(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> static void save(const fs::path &p,const T* x,size_t n){
 if(fs::exists(p))throw std::runtime_error("OUTPUT_EXISTS");
 std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(x),n*sizeof(T));
 if(!f)throw std::runtime_error("OUTPUT_WRITE");
}
extern "C" int e1_export(const float* xyz,size_t n,const char* dest,int render,const float* position){
 try{
 if(!n || n>3000000)throw std::runtime_error("POINT_BUDGET");
 pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
 cloud->reserve(n);
 for(size_t i=0;i<n;i++){
  if(!std::isfinite(xyz[3*i])||!std::isfinite(xyz[3*i+1])||!std::isfinite(xyz[3*i+2]))throw std::runtime_error("NONFINITE");
  cloud->push_back(pcl::PointXYZ(xyz[3*i],xyz[3*i+1],xyz[3*i+2]));
 }
 Eigen::Vector4f lo,hi;pcl::getMinMax3D(*cloud,lo,hi);
 size_t cells=1;for(int k=0;k<3;k++)cells*=static_cast<size_t>(ceil((hi[k]-lo[k]+.002f)/.1f));
 if(cells>30000000)throw std::runtime_error("GRID_BUDGET");
 ok(cudaSetDevice(0));size_t free=0,total=0;ok(cudaMemGetInfo(&free,&total));
 // Bound all allocations in this ABI plus native monocular path before allocation.
 size_t request=cells*sizeof(int)+(render?160*90*sizeof(float):0);
 if(request>1610612736ULL || total-free+request>1610612736ULL)throw std::runtime_error("GPU_MEMORY_BUDGET");
 raycast::GridMap grid(cloud,.1f,0);ok(cudaGetLastError());
 size_t actual=static_cast<size_t>(grid.grid_size_x_)*grid.grid_size_y_*grid.grid_size_z_;
 if(actual!=cells)throw std::runtime_error("GRID_SIZE");
 std::vector<int> counts(cells);
 ok(cudaMemcpy(counts.data(),grid.map_cuda_,cells*sizeof(int),cudaMemcpyDeviceToHost));
 fs::path out=dest;
 save(out/"occupancy_counts.i32",counts.data(),counts.size());
 if(!render)save(out/"cloud_xyz.f32",xyz,n*3);
 if(render){
  raycast::CameraParams c;c.fx=80;c.fy=80;c.cx=80;c.cy=45;c.image_width=160;c.image_height=90;c.max_depth_dist=20;c.normalize_depth=false;
  cudaMat::SE3<float> pose(1.f,0.f,0.f,0.f,position[0],position[1],position[2]);
  cv::Mat depth;raycast::renderDepthImage(&grid,&c,pose,depth);ok(cudaGetLastError());ok(cudaDeviceSynchronize());
  if(depth.type()!=CV_32FC1||!depth.isContinuous())throw std::runtime_error("DEPTH_LAYOUT");
  save(out/"depth_z.f32",depth.ptr<float>(),depth.total());
 }
 ok(cudaMemGetInfo(&free,&total));
 if(total-free>1610612736ULL)throw std::runtime_error("GPU_MEMORY_BUDGET_AFTER");
 std::ofstream f(out/(render?"renderer.json":"grid.json"));
 f<<std::setprecision(9)<<"{\n";
 f<<"\"schema\":\"e1_native_cuda_two_stage_v1\",\"mode\":\""<<(render?"depth":"counts_only")<<"\",";
 f<<"\"grid_shape\":["<<grid.grid_size_x_<<","<<grid.grid_size_y_<<","<<grid.grid_size_z_<<"],";
 f<<"\"grid_origin\":["<<grid.origin_x_<<","<<grid.origin_y_<<","<<grid.origin_z_<<"],";
 f<<"\"resolution_m\":"<<grid.resolution_<<",\"occupy_threshold\":0,\"point_count\":"<<n<<",";
 f<<"\"position_world\":["<<position[0]<<","<<position[1]<<","<<position[2]<<"],";
 f<<"\"body_quaternion_wxyz\":[1,0,0,0],\"camera_pitch_deg\":0,";
 f<<"\"camera\":{\"width\":160,\"height\":90,\"fx\":80,\"fy\":80,\"cx\":80,\"cy\":45,\"max_depth_m\":20,\"normalized\":false},";
 f<<"\"cuda_total_bytes\":"<<total<<",\"cuda_used_bytes_after\":"<<total-free<<",\"bounded_explicit_allocations_bytes\":"<<request<<"}\n";
 if(!f)throw std::runtime_error("METADATA_WRITE");
 grid.freeGridMap();std::cout<<"EXPORTED "<<(render?"depth":"counts_only")<<" cells="<<cells<<"\n";return 0;
 }catch(const std::exception&e){std::cerr<<"STOP "<<e.what()<<"\n";return 1;}
}
