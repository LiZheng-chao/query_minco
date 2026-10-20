// F0 experimental offline forest voxel exporter.
// IMPORTANT: standalone executable; DOES NOT edit Simulator or talk to ROS.
// Compile by linking the EXISTING Simulator/src/src/maps.cpp and perlinnoise.cpp.
// Run only after stopping live forest simulator due to ~52M source points memory footprint.
#include "maps.hpp"
#include <yaml-cpp/yaml.h>
#include <pcl/common/common.h>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <cerrno>
#include <cstring>
#include <sys/stat.h>
#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// C++11/POSIX path operations: avoid std::filesystem and an extra
// C++17 standard library dependency on older ROS Noetic toolchains.
static bool regularFile(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

static bool existsOrFail(const std::string& path) {
    struct stat st;
    if (::lstat(path.c_str(), &st) == 0) return true;
    if (errno == ENOENT) return false;
    throw std::runtime_error("Cannot stat output path: " + path + ": " + std::strerror(errno));
}

static void mkdirOutputWithoutOverwrite(const std::string& path) {
    if (path.size() < 2 || path[0] != '/' || path.back() == '/')
        throw std::runtime_error("Output directory must be an absolute path without trailing slash");
    if (existsOrFail(path))
        throw std::runtime_error("Output directory already exists (refusing overwrite): " + path);

    // Create missing parents while refusing to reuse an existing FINAL folder.
    for (std::size_t i = 1; i <= path.size(); ++i) {
        if (i != path.size() && path[i] != '/') continue;
        const std::string subpath = path.substr(0, i);
        if (subpath.empty()) continue;
        if (::mkdir(subpath.c_str(), 0755) == 0) continue;
        const int error = errno;
        if (error == EEXIST && i < path.size()) {
            struct stat st;
            if (::stat(subpath.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) continue;
        }
        throw std::runtime_error("Cannot create directory: " + subpath + ": " + std::strerror(error));
    }
}

static std::string escaped(const std::string &s) {
    std::string t;
    for (unsigned char c : s) {
        switch (c) {
        case '\\': t += "\\\\"; break;
        case '"': t += "\\\""; break;
        case '\n': t += "\\n"; break;
        default: t += static_cast<char>(c);
        }
    }
    return t;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: f0_export_forest_voxels /abs/path/f0_config.yaml /abs/path/NEW_output_directory\n";
        return 2;
    }
    try {
        const std::string config_file = argv[1];
        const std::string out_dir = argv[2];
        if (config_file.empty() || config_file[0] != '/')
            throw std::runtime_error("Config path must be absolute");
        if (!regularFile(config_file)) throw std::runtime_error("Config not found");
        if (out_dir.size() < 2 || out_dir[0] != '/' || out_dir.back() == '/')
            throw std::runtime_error("Output directory must be an absolute path without trailing slash");
        if (existsOrFail(out_dir)) throw std::runtime_error("Output directory must be NEW (refusing overwrite)");
        const YAML::Node c = YAML::LoadFile(config_file);
        if (!c["random_map"].as<bool>() || c["maze_type"].as<int>()!=5)
            throw std::runtime_error("F0 exporter only supports native random forest maze_type=5");
        const float resolution=c["resolution"].as<float>();
        if (!std::isfinite(resolution) || resolution<=0 || resolution>1)
            throw std::runtime_error("Invalid resolution");
        const int threshold=c["occupy_threshold"].as<int>();
        if (threshold<0 || threshold>=65535) throw std::runtime_error("Unsupported occupy_threshold");
        const std::string tree_file=c["tree_file"].as<std::string>();
        if (tree_file.empty() || tree_file[0] != '/' || !regularFile(tree_file))
            throw std::runtime_error("Tree PLY not found (requires absolute path)");
        const int seed=c["seed"].as<int>();
        if (seed<0) throw std::runtime_error("Seed must be nonnegative");
        // IMPORTANT: reproduce the exact C++ expression in test_simulator_cuda.cpp.
        // There, `1 / resolution` divides in float precision before converting
        // to double.  1.0/double(resolution) instead gives 9.99999985 for 0.1f,
        // truncating int(60 * scale) from 600 to 599, changing the forest layout.
        const double scale=1 / resolution;
        const int x_len=c["x_length"].as<int>();
        const int y_len=c["y_length"].as<int>();
        const int z_len=c["z_length"].as<int>();
        if (x_len<1 || y_len<1 || z_len<1) throw std::runtime_error("Invalid forest dimensions");

        auto cloud=pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
        mocka::Maps::BasicInfo info;
        info.sizeX=static_cast<int>(x_len*scale); // matches truncation in test_simulator_cuda.cpp
        info.sizeY=static_cast<int>(y_len*scale);
        info.sizeZ=static_cast<int>(z_len*scale);
        info.seed=seed;
        info.scale=scale;
        info.cloud=cloud;
        mocka::Maps map;
        map.setParam(c);
        map.setInfo(info);
        std::cout << std::setprecision(17)
                  << "Simulator expression 1 / resolution: resolution=" << resolution
                  << ", scale=" << scale
                  << ", BasicInfo sizeXYZ=" << info.sizeX << "x" << info.sizeY << "x" << info.sizeZ
                  << std::endl;
        std::cout << "Generating native Maps::forest(), seed=" << seed << std::endl;
        map.generate(5); // exact existing YOPO generator implementation
        const uint64_t npts=cloud->points.size();
        if (npts==0) throw std::runtime_error("No points generated");
        std::cout << "Generated points=" << npts << std::endl;

        Eigen::Vector4f lo,hi;
        pcl::getMinMax3D(*cloud,lo,hi);
        constexpr float eps=0.001f;
        const float origin[3]={lo.x(),lo.y(),lo.z()};
        // GridMap constructor computes a float span and divides it by a float
        // resolution before ceil; preserve those float semantics here.
        const float span[3]={hi.x()-lo.x()+2*eps,
                             hi.y()-lo.y()+2*eps,
                             hi.z()-lo.z()+2*eps};
        int64_t dim[3];
        for(int k=0;k<3;++k) {
            dim[k]=static_cast<int64_t>(std::ceil(span[k]/resolution));
            if(dim[k]<2) throw std::runtime_error("Degenerate grid dimension");
        }
        // With 0.1 m resolution, nominal 60x60x15 m already requires ~54M voxels.
        // A conservative 100M cap permits this forest without unbounded allocation.
        // Caller additionally checks >=3GiB available RAM before running.
        const int64_t nv=dim[0]*dim[1]*dim[2];
        std::cout << std::setprecision(17)
                  << "GridMap source-compatible scale=" << scale
                  << ", BasicInfo sizeXYZ=" << info.sizeX << "x" << info.sizeY << "x" << info.sizeZ
                  << ", shapeXYZ=" << dim[0] << "x" << dim[1] << "x" << dim[2]
                  << ", voxel_count=" << nv
                  << ", counts+occupied_bytes_approx=" << nv*3 << std::endl;
        if (nv<=0 || nv>100000000) throw std::runtime_error("Grid exceeds 100M voxel safety cap");
        std::vector<uint16_t> counts(static_cast<size_t>(nv),0);
        uint64_t skipped=0;
        const uint16_t sat=static_cast<uint16_t>(threshold+1);
        for(const auto& pt:cloud->points) {
            int64_t v[3];
            const float p[3]={pt.x,pt.y,pt.z};
            for(int k=0;k<3;++k) {
                // mimic CUDA GridMap constructor's float coordinate math
                const float pos=p[k]+eps;
                v[k]=static_cast<int64_t>(std::floor((pos-origin[k])/resolution));
            }
            if(v[0]<0 || v[0]>=dim[0] || v[1]<0 || v[1]>=dim[1] || v[2]<0 || v[2]>=dim[2]) {
                ++skipped; continue;
            }
            size_t idx=static_cast<size_t>((v[0]*dim[1]+v[1])*dim[2]+v[2]);
            if(counts[idx]<sat) ++counts[idx];
        }
        if(skipped) throw std::runtime_error("Out-of-bounds points="+std::to_string(skipped)+"; STOP: voxel replica not proven equal");
        std::vector<uint8_t> occupied(static_cast<size_t>(nv),0);
        uint64_t n_occ=0;
        for(size_t i=0;i<occupied.size();++i) {
            occupied[i]=(counts[i]>threshold)?1:0;
            n_occ+=occupied[i];
        }
        // Free the expensive generated point cloud before writing results.
        cloud->points.clear(); cloud->points.shrink_to_fit();
        cloud.reset(); counts.clear(); counts.shrink_to_fit();
        mkdirOutputWithoutOverwrite(out_dir);
        const std::string binary=out_dir+"/occupancy_xyz_u8.bin";
        const std::string meta=out_dir+"/occupancy_metadata.json";
        {
            std::ofstream f(binary, std::ios::binary);
            if(!f) throw std::runtime_error("Cannot create occupancy binary");
            f.write(reinterpret_cast<const char*>(occupied.data()),static_cast<std::streamsize>(occupied.size()));
            if(!f) throw std::runtime_error("Cannot write occupancy binary");
        }
        {
            std::ofstream f(meta);
            if(!f) throw std::runtime_error("Cannot create metadata");
            f << std::setprecision(17);
            f << "{\n  \"schema\": \"F0_OFFLINE_NATIVE_FOREST_VOXEL_REPLICA_V1\",\n";
            f << "  \"not_live_simulator_map\": true,\n";
            f << "  \"config_path\": \"" << escaped(config_file) << "\",\n";
            f << "  \"tree_path\": \"" << escaped(tree_file) << "\",\n";
            f << "  \"seed\": " << seed << ",\n";
            f << "  \"maze_type\": 5,\n";
            f << "  \"resolution_m\": " << resolution << ",\n";
            f << "  \"origin_m\": [" << origin[0] << ", " << origin[1] << ", " << origin[2] << "],\n";
            f << "  \"shape_xyz\": [" << dim[0] << ", " << dim[1] << ", " << dim[2] << "],\n";
            f << "  \"occupy_threshold\": " << threshold << ",\n";
            f << "  \"point_count\": " << npts << ",\n";
            f << "  \"occupied_voxels\": " << n_occ << ",\n";
            f << "  \"skipped_out_of_bounds_points\": " << skipped << ",\n";
            f << "  \"voxel_file\": \"occupancy_xyz_u8.bin\",\n";
            f << "  \"map_query_xy\": \"symmetric mirror (period 2*n-2)\",\n";
            f << "  \"map_query_z\": \"z<=0 occupied; z>=nz free\",\n";
            f << "  \"assessment\": \"Separate process regenerated same seed with original Maps::forest; compare against saved depth before trusting as GT\"\n}\n";
            if(!f) throw std::runtime_error("Cannot write metadata");
        }
        std::cout << "SUCCESS voxel grid: " << dim[0] << "x" << dim[1] << "x" << dim[2]
                  << ", occupied=" << n_occ << " / " << nv << ", skipped=" << skipped << "\n";
        std::cout << "Metadata: " << meta << "\nBinary: " << binary << "\n";
        return 0;
    } catch(const std::exception &e) {
        std::cerr << "F0 GT EXPORT BLOCKED: " << e.what() << std::endl;
        return 1;
    }
}
