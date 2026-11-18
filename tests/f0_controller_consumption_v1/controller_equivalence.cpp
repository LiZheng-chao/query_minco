#include <so3_control/SO3Control.h>
#include <iostream>
#include <iomanip>
#include <cmath>
int main(){SO3Control c;c.setMass(.98);c.setGravity(9.81);std::cout<<std::setprecision(17);
for(int i=0;i<512;++i){double u=i/511.;c.setPosition(Eigen::Vector3d(.25+u*.28,-.25,2));
c.setVelocity(Eigen::Vector3d(.1*u,.01,-.02));c.setAcc(Eigen::Vector3d(.02,0,.01));
c.calculateControl(Eigen::Vector3d(.3+u*.28,-.24,2.02),Eigen::Vector3d(.1,0,0),Eigen::Vector3d(.03,0,0),u*.2,0,Eigen::Vector3d(5.7,5.7,6.2),Eigen::Vector3d(3.4,3.4,4));
const auto& f=c.getComputedForce();const auto& q=c.getComputedOrientation();
std::cout<<i<<' '<<f.x()<<' '<<f.y()<<' '<<f.z()<<' '<<q.x()<<' '<<q.y()<<' '<<q.z()<<' '<<q.w()<<'\n';}
}
