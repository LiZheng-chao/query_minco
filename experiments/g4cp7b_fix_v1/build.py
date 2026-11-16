"""Compile isolated native renderer I/O/telemetry overlay;never execute ROS/CUDA simulation."""
from pathlib import Path
import sys,json,os,difflib
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.audit.run_provenance_v1 import run_logged
from query_minco.evaluation.mirror_holdout_v1 import sha
R=Path.cwd();O=R/'research_outputs/g4cp7b_fix_20261112_v1';B=O/'build_v2';B.mkdir();Y=R.parent/'YOPO/Simulator'
header=(Y/'src/include/sensor_simulator.cuh').read_text()
method=r'''
            void exportTruth(const std::string& base) const {
                const size_t n=size_t(grid_size_x_)*grid_size_y_*grid_size_z_;
                if(n==0||n>128000000)throw std::runtime_error("TRUTH_GRID_SIZE");
                std::ofstream out(base+".u8",std::ios::binary|std::ios::out);
                std::vector<int> values(1048576);std::vector<unsigned char> bytes(1048576);
                for(size_t start=0;start<n;start+=values.size()){
                    size_t count=std::min(values.size(),n-start);
                    if(cudaMemcpy(values.data(),map_cuda_+start,count*sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess)throw std::runtime_error("TRUTH_GRID_COPY");
                    for(size_t j=0;j<count;++j)bytes[j]=values[j]>occupy_threshold_?1:0;
                    out.write(reinterpret_cast<const char*>(bytes.data()),count);
                }
                out.close();if(!out)throw std::runtime_error("TRUTH_GRID_WRITE");
                std::ofstream meta(base+".json");meta.precision(17);
                meta<<"{\"shape\":["<<grid_size_x_<<","<<grid_size_y_<<","<<grid_size_z_<<"],\"origin\":["<<origin_x_<<","<<origin_y_<<","<<origin_z_<<"],\"resolution\":"<<resolution_<<",\"threshold\":"<<occupy_threshold_<<",\"cells\":"<<n<<"}";
                meta.close();if(!meta)throw std::runtime_error("TRUTH_META_WRITE");
            }
'''
header=header.replace('            float raycast_step_;',method+'\n            float raycast_step_;')
(B/'sensor_simulator.cuh').write_text('#include <fstream>\n#include <stdexcept>\n'+header)
old=(Y/'src/src/test_simulator_cuda.cpp').read_text();s=old
s=s.replace('#include <chrono>','#include <chrono>\n#include <std_msgs/String.h>\n#include <sstream>')
s=s.replace('        pcl::toROSMsg(*cloud, output);','        // Huge PointCloud2 export removed;full native GPU grid exported below.')
s=s.replace('        timer_map_   = nh_.createTimer(ros::Duration(1), &SensorSimulator::timerMapCallback, this);','        timing_pub_=nh_.advertise<std_msgs::String>("/query_minco/renderer_timing",10);')
s=s.replace('        grid_map = new GridMap(cloud, resolution, occupy_threshold);','''        grid_map = new GridMap(cloud, resolution, occupy_threshold);
        std::string truth_path;if(!nh_.getParam("truth_export_path",truth_path)||truth_path.empty())throw std::runtime_error("TRUTH_EXPORT_REQUIRED");
        grid_map->exportTruth(truth_path);
        pcl::PointCloud<pcl::PointXYZ>().swap(*cloud);''')
s=s.replace('    std::string camera_info_topic_', '    ros::Publisher timing_pub_;double odom_receipt_=0;\n    std::string camera_info_topic_')
s=s.replace('    quat.x() = msg->pose.pose.orientation.x;', '    odom_receipt_=ros::Time::now().toSec();\n    quat.x() = msg->pose.pose.orientation.x;')
s=s.replace('        renderDepthImage(grid_map, camera, T_wc, depth_image);','        double render_start=ros::Time::now().toSec();\n        renderDepthImage(grid_map, camera, T_wc, depth_image);\n        double render_end=ros::Time::now().toSec();')
s=s.replace('        camera_info_pub_.publish(camera_info_);',r'''        camera_info_pub_.publish(camera_info_);
        std_msgs::String timing;std::ostringstream ts;ts.precision(17);
        ts<<"{\"source_stamp\":"<<stamp.toSec()<<",\"odom_receipt\":"<<odom_receipt_<<",\"render_start\":"<<render_start<<",\"render_end\":"<<render_end<<",\"publish_ros\":"<<ros::Time::now().toSec()<<"}";
        timing.data=ts.str();timing_pub_.publish(timing);''')
start=s.index('        std_msgs::String timing;');end=s.index('        timing.data=ts.str();timing_pub_.publish(timing);',start)+len('        timing.data=ts.str();timing_pub_.publish(timing);')
s=s[:start]+s[end:]
src=B/'renderer.cpp';src.write_text(s)
(O/'RENDERER_DIFF_v2.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True))))
oldbuild=R/'research_outputs/g4a_forest_shadow_20261111_v1/ros_build'
cmd=json.loads((oldbuild/'compile/PROCESS_START.json').read_text())['argv']
cmd=[str(src) if t==str(Y/'src/src/test_simulator_cuda.cpp') else str(B/'renderer.o') if t==str(oldbuild/'native_sensor.o') else '-DCONFIG_FILE_PATH="'+str(Y/'src/config/config.yaml')+'"' if t.startswith('-DCONFIG_FILE_PATH=') else t for t in cmd]
cmd.insert(3,'-I'+str(B))
link=json.loads((oldbuild/'link/PROCESS_START.json').read_text())['argv']
link=[str(B/'renderer.o') if t==str(oldbuild/'native_sensor.o') else str(B/'native_renderer') if t==str(oldbuild/'native_sensor_shadow') else t for t in link]
env=dict(os.environ,LD_LIBRARY_PATH='/usr/local/cuda-11.8/lib64:/opt/ros/noetic/lib:'+str(Y/'devel/lib'))
for name,argv,cwd in [('compile',cmd,R),('link',link,Y/'build')]:
 result=run_logged(argv,B/name,cwd=cwd,env=env,timeout_s=180)
 if result['exit_code']:raise RuntimeError(name)
(B/'BUILD_IDENTITY.json').write_text(json.dumps(dict(source_sha=sha(src),header_sha=sha(B/'sensor_simulator.cuh'),binary_sha=sha(B/'native_renderer'),native_library_sha=sha(Y/'devel/lib/libraycast_cuda.so'),config_sha=sha(Y/'src/config/config.yaml'),native_cuda_unchanged=True),indent=2))
print('COMPILE_LINK_PASS_NO_RENDERER_EXECUTION')
