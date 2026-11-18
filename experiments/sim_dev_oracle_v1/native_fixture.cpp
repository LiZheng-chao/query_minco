#include "sensor_simulator.cuh"
#include <dlfcn.h>
#include <set>
#include <tuple>
#include <iomanip>
using Cell=std::tuple<int,int,int>;
Cell cell(float x,float y,float z){return Cell(int(floor((x+2)/.1f)),int(floor((y+25)/.1f)),int(floor((z+1)/.1f)));}
int main(int argc,char**argv){
 Dl_info info{};dladdr(reinterpret_cast<void*>(&raycast::renderDepthImage),&info);
 std::cout<<"{\"loaded_library\":\""<<info.dli_fname<<"\",\"resolved_symbol\":\""<<info.dli_sname<<"\",\"cases\":[";
 std::set<Cell> samples;float px=.013f,py=.017f,pz=2.023f;
 for(int k=1;k<=405;k++){px+=.05f;py+=-.9875f*.05f;samples.insert(cell(px,py,pz));}
 Cell missed;bool found=false;float missed_t=0;
 for(int k=2000;k<30000;k++){float x=k*.0001f;Cell c=cell(.013f+x,.017f-.9875f*x,2.023f);if(!samples.count(c)){missed=c;found=true;missed_t=x;break;}}
 if(!found)return 10;
 for(int test=0;test<9;test++){
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->push_back(pcl::PointXYZ(-2,-25,-1));cloud->push_back(pcl::PointXYZ(25,25,6));
  float ox=0,oy=0,oz=2;int u=80,v=45;std::string name="empty_ray";float hx=0,hy=0,hz=0;int duplicates=3;
  if(test==1){name="near_tree_occupied_voxel";hx=.45f;hz=2;}
  if(test==2){name="single_return_below_occupancy_threshold";hx=.45f;hz=2;duplicates=1;}
  if(test==3){name="thin_grazing_occupied_voxel_missed_by_sampling";ox=.013f;oy=.017f;oz=2.023f;u=159;hx=-2+(std::get<0>(missed)+.5f)*.1f;hy=-25+(std::get<1>(missed)+.5f)*.1f;hz=-1+(std::get<2>(missed)+.5f)*.1f;}
  if(test==4){name="oblique_hit";u=120;hx=2.05f;hy=-1.05f;hz=2;}
  if(test==5){name="upper_z_outside_map_returns_free_queries";oz=8;}
  if(test==6){name="exact_max_range_voxel_center_hit";ox=.05f;hx=20.05f;hz=2;}
  if(test==7){name="lower_z_boundary_is_occupied";oz=-.95f;}
  if(test==8){name="outside_x_uses_mirrored_occupancy";ox=28;hx=22.05f;hz=2;}
  if(test==1||test==2||test==3||test==4||test==6||test==8)for(int k=0;k<duplicates;k++)cloud->push_back(pcl::PointXYZ(hx,hy,hz));
  raycast::GridMap grid(cloud,.1f,1);if(test==6)ox=grid.Vox2Pos(grid.Pos2Vox(raycast::Vector3f(hx,hy,hz))).x-20.f;raycast::CameraParams camera;cudaMat::SE3<float> pose(1,0,0,0,ox,oy,oz);cv::Mat depth;raycast::renderDepthImage(&grid,&camera,pose,depth);
  if(cudaGetLastError()!=cudaSuccess)return 11;
  if(test)std::cout<<",";
  std::cout<<std::setprecision(10)<<"{\"name\":\""<<name<<"\",\"pixel\":["<<u<<","<<v<<"],\"depth\":"<<depth.at<float>(v,u)<<",\"camera\":["<<ox<<","<<oy<<","<<oz<<"],\"obstacle\":["<<hx<<","<<hy<<","<<hz<<"],\"duplicates\":"<<duplicates<<",\"thin_crossing_parameter\":"<<missed_t<<"}";
  if(test==3 && argc==2)grid.exportTruth(std::string(argv[1])+"/THIN_COUNTEREXAMPLE_GRID");
  grid.freeGridMap();
 }
 std::cout<<"]}"<<std::endl;return 0;
}
