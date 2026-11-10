// Ordinary C++ translation unit: PLY/Boost and forest generation never enter NVCC.
#include "maps.hpp"
#include <pcl/common/common.h>
#include <filesystem>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
extern "C" int gate1_render_export(
    const float *points_xyz, size_t point_count, const char *output_path,
    const char *case_kind, int seed, const float *camera_values, const float *body_position);
namespace fs = std::filesystem;
int main(int argc, char **argv) {
  try {
    if (argc != 5 && argc != 6) throw std::runtime_error("usage: bridge yopo_root output_dir case seed [pose_certificate]");
    fs::path source=fs::canonical(argv[1]), output=argv[2];
    std::string kind=argv[3]; int seed=std::stoi(argv[4]);
    if (kind!="free" && kind!="sparse" && kind!="occluded" && kind!="grazing")
        throw std::runtime_error("unsupported fixed diagnostic");
    if (!fs::create_directory(output)) throw std::runtime_error("output directory exists");
    auto config=YAML::LoadFile((source/"Simulator/src/config/config.yaml").string());
    if (config["camera"]["normalize_depth"].as<bool>() ||
        std::abs(config["camera"]["pitch"].as<float>())>1e-7)
        throw std::runtime_error("this offline contract requires unnormalized zero-pitch camera");
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    if (kind=="sparse" || kind=="occluded") {
        config["tree_file"]=(source/"Simulator/src/pointcloud/tree.ply").string();
        config["tree_dist"]=16.;
        mocka::Maps::BasicInfo info;
        info.sizeX=160; info.sizeY=160; info.sizeZ=150;
        info.seed=seed; info.scale=10.; info.cloud=cloud;
        mocka::Maps maps; maps.setParam(config); maps.setInfo(info); maps.generate(5);
    } else {
        // Diagnostic obstacle-free environment includes a real rendered floor.
        for (int x=0;x<=480;x++) for (int y=0;y<=480;y++)
            cloud->push_back(pcl::PointXYZ(-24.f+x*.1f,-24.f+y*.1f,0.f));
        if (kind=="grazing") cloud->push_back(pcl::PointXYZ(2.05f,.05f,1.55f));
    }
    if (cloud->empty()) throw std::runtime_error("empty source cloud");
    std::vector<float> xyz; xyz.reserve(cloud->size()*3);
    for (const auto &p:*cloud) {xyz.push_back(p.x);xyz.push_back(p.y);xyz.push_back(p.z);}
    float px=0.f,py=0.f,pz=1.8f;
    double t0=29.5*(2./3.)/63.;
    if (kind=="sparse" || kind=="occluded") {
        Eigen::Vector4f lo,hi;pcl::getMinMax3D(*cloud,lo,hi);
        px=lo.x()-(kind=="sparse"?2.f:1.5f);
        double y_sum=0;size_t n=0;
        for(const auto &p:*cloud) if(p.z>.7f && p.z<2.3f){y_sum+=p.y;n++;}
        py=n?static_cast<float>(y_sum/n):(lo.y()+hi.y())/2;
    }
    if (argc==6) {
        auto pose=YAML::LoadFile(argv[5]);
        if (kind!="sparse" || seed!=1001 ||
            pose["schema"].as<std::string>()!="gate1_pose_certificate_v1" ||
            !pose["passed"].as<bool>() || pose["scene_seed"].as<int>()!=seed ||
            pose["clearance_lower_bound_m"].as<double>()<.150001 ||
            pose["source_occupancy_sha256"].as<std::string>()!="869c4ceb3413d7b5bd8452ffda2c55f3225a0baece592bcb344b1664504af3bf")
            throw std::runtime_error("invalid independent pose certificate");
        for(int i=0;i<3;i++) for(int j=0;j<3;j++)
            if(pose["R_WB0"][i][j].as<double>()!=(i==j?1.:0.))
                throw std::runtime_error("renderer supports identity pose rotation only");
        px=pose["position_world"][0].as<float>();
        py=pose["position_world"][1].as<float>();
        pz=pose["position_world"][2].as<float>();
        if(!std::isfinite(px)||!std::isfinite(py)||!std::isfinite(pz))
            throw std::runtime_error("nonfinite pose");
    }
    float camera_values[] = {
        config["camera"]["fx"].as<float>(),config["camera"]["fy"].as<float>(),
        config["camera"]["cx"].as<float>(),config["camera"]["cy"].as<float>(),
        static_cast<float>(config["camera"]["image_width"].as<int>()),
        static_cast<float>(config["camera"]["image_height"].as<int>()),
        config["camera"]["max_depth_dist"].as<float>()};
    float body_position[]={px,py,pz};
    return gate1_render_export(xyz.data(),cloud->size(),output.string().c_str(),
                              kind.c_str(),seed,camera_values,body_position);
  } catch(const std::exception &e) {std::cerr<<"STOP "<<e.what()<<"\n";return 1;}
}
