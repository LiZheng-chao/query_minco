#pragma once
#include <cmath>
#include <string>
namespace g4c {
struct PlantGate {
 double hover=0,last_stamp=0,received=0;unsigned count=0,steps=0;bool armed=false,aborted=false,valid=false;
 std::string nonce;
 void accept(double stamp,double now,double fx,double fy,double fz,double qnorm) {
  valid=std::isfinite(stamp)&&stamp>0&&std::isfinite(now)&&now-stamp<=.2&&stamp-now<=.01&&
   std::isfinite(fx)&&std::isfinite(fy)&&std::isfinite(fz)&&std::isfinite(qnorm)&&std::abs(qnorm-1)<1e-3;
  if(!armed) valid=valid&&hover>0&&std::abs(fz-hover)<=.05*hover&&std::hypot(fx,fy)<=.05*hover;
  if(!valid){aborted=true;return;}
  last_stamp=stamp;received=now;++count;
 }
 bool permit(const std::string& token,double now) {
  if(!std::isfinite(now)||aborted||nonce.empty()||token!=nonce||!valid||count<5||now-last_stamp>.2||last_stamp-now>.01)return false;
  armed=true;return true;
 }
 bool step_allowed(double now) {
  if(!std::isfinite(now)){aborted=true;return false;}
  if(aborted)return false;
  if(!armed)return false;
  if(!valid||now-last_stamp>.2||last_stamp-now>.01){aborted=true;return false;}
  return true;
 }
};
}
