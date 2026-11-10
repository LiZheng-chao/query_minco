// Offline qualification build of the original YOPO renderer, without ROS nodes.
// Access visibility is opened ONLY in this translation unit to export the exact
// allocated GridMap counts/origin/dimensions. No kernel or mapping formula changes.
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
#include "sensor_simulator.cu"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace fs = std::filesystem;
void cuda_ok(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
template<class T> void binary(const fs::path &p, const T *data, size_t n) {
    if (fs::exists(p)) throw std::runtime_error("output exists");
    std::ofstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("output open failed");
    f.write(reinterpret_cast<const char*>(data), n*sizeof(T));
    if (!f) throw std::runtime_error("output write failed");
}
extern "C" int gate1_render_export(
    const float *points_xyz, size_t point_count, const char *output_path,
    const char *case_kind, int seed, const float *camera_values, const float *body_position) {
  try {
    fs::path output=output_path;
    std::string kind=case_kind;
    int devices=0; cuda_ok(cudaGetDeviceCount(&devices));
    if (!devices) throw std::runtime_error("no CUDA device");
    cuda_ok(cudaSetDevice(0));
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->reserve(point_count);
    for(size_t i=0;i<point_count;i++)
        cloud->push_back(pcl::PointXYZ(points_xyz[3*i],points_xyz[3*i+1],points_xyz[3*i+2]));
    if (cloud->empty()) throw std::runtime_error("empty source cloud");
    raycast::GridMap grid(cloud,.1f,0);
    cuda_ok(cudaGetLastError());
    size_t total=static_cast<size_t>(grid.grid_size_x_)*grid.grid_size_y_*grid.grid_size_z_;
    if (total==0 || total>30000000) throw std::runtime_error("grid budget");
    std::vector<int> counts(total);
    cuda_ok(cudaMemcpy(counts.data(),grid.map_cuda_,total*sizeof(int),cudaMemcpyDeviceToHost));
    std::vector<float> xyz; xyz.reserve(cloud->size()*3);
    for (const auto &p:*cloud) {xyz.push_back(p.x);xyz.push_back(p.y);xyz.push_back(p.z);}
    raycast::CameraParams camera;
    camera.fx=camera_values[0]; camera.fy=camera_values[1];
    camera.cx=camera_values[2]; camera.cy=camera_values[3];
    camera.image_width=static_cast<int>(camera_values[4]);
    camera.image_height=static_cast<int>(camera_values[5]);
    camera.max_depth_dist=camera_values[6]; camera.normalize_depth=false;
    if (camera.image_width!=160 || camera.image_height!=90 || camera.max_depth_dist!=20.f)
        throw std::runtime_error("unsupported camera profile");
    float px=body_position[0],py=body_position[1],pz=body_position[2];
    double t0=29.5*(2./3.)/63.;
    if(kind=="grazing"){
        auto v=grid.Pos2Vox(raycast::Vector3f(2.051f,.051f,1.551f));
        auto center=grid.Vox2Pos(v);
        px=center.x-static_cast<float>(2.5*t0);
        float y_lower=grid.origin_y_+v.y*grid.resolution_;
        py=y_lower-static_cast<float>(.15+.1*t0*t0-5e-7);
        pz=center.z;
    }
    cudaMat::SE3<float> pose(1.f,0.f,0.f,0.f,px,py,pz);
    cv::Mat depth; raycast::renderDepthImage(&grid,&camera,pose,depth);
    cuda_ok(cudaGetLastError());cuda_ok(cudaDeviceSynchronize());
    if(depth.type()!=CV_32FC1 || !depth.isContinuous())throw std::runtime_error("depth layout");
    binary(output/"cloud_xyz.f32",xyz.data(),xyz.size());
    binary(output/"occupancy_counts.i32",counts.data(),counts.size());
    binary(output/"depth_z.f32",depth.ptr<float>(),depth.total());
    std::ofstream f(output/"renderer.json");
    f<<std::setprecision(9);
    f<<"{\n\"schema\":\"yopo_original_renderer_export_v1\",\"case\":\""<<kind<<"\",\"seed\":"<<seed<<",";
    f<<"\"grid_shape\":["<<grid.grid_size_x_<<","<<grid.grid_size_y_<<","<<grid.grid_size_z_<<"],";
    f<<"\"grid_origin\":["<<grid.origin_x_<<","<<grid.origin_y_<<","<<grid.origin_z_<<"],";
    f<<"\"resolution_m\":"<<grid.resolution_<<",\"occupy_threshold\":"<<grid.occupy_threshold_<<",";
    f<<"\"point_count\":"<<cloud->size()<<",\"position_world\":["<<px<<","<<py<<","<<pz<<"],";
    f<<"\"body_quaternion_wxyz\":[1,0,0,0],\"camera_pitch_deg\":0,";
    f<<"\"camera\":{\"width\":160,\"height\":90,\"fx\":"<<camera.fx<<",\"fy\":"<<camera.fy
      <<",\"cx\":"<<camera.cx<<",\"cy\":"<<camera.cy<<",\"max_depth_m\":20,\"normalized\":false},";
    f<<"\"canary_t0\":"<<t0<<",\"scene_profile\":\""
      <<((kind=="sparse"||kind=="occluded")?"original_maps_forest_16m_one_tree":"rendered_floor_diagnostic")<<"\"}\n";
    if(!f)throw std::runtime_error("metadata write failed");
    grid.freeGridMap();
    std::cout<<"EXPORTED "<<kind<<" points="<<cloud->size()<<" cells="<<total<<"\n";
    return 0;
  } catch(const std::exception &e) {std::cerr<<"STOP "<<e.what()<<"\n";return 1;}
}
