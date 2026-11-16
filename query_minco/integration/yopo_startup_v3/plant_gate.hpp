#pragma once
#include <cmath>
#include <string>
namespace g4cp1 {
struct PlantGate {
 enum Phase {HELD,PREPARE,ACK,ARMED,ABORT};
 Phase phase=HELD;
 double hover=0,last_stamp=0,received=0;unsigned count=0,steps=0;bool armed=false,aborted=false,valid=false;
 std::string nonce;
 const char* name()const {switch(phase){case HELD:return "HELD";case PREPARE:return "PREPARE";case ACK:return "ACK";case ARMED:return "ARMED";default:return "ABORT";}}
 void trip(){aborted=true;valid=false;phase=ABORT;}
 bool prepare(const std::string& token,double now){
  if(!std::isfinite(now)||now<=0||aborted||nonce.empty()||token!=nonce||phase!=HELD)return false;
  phase=PREPARE;return true;
 }
 bool accept(double stamp,double now,double fx,double fy,double fz,double qnorm){
  if(aborted)return false;
  if(phase==HELD){trip();return false;}
  if(!std::isfinite(stamp)||stamp<=0||!std::isfinite(now)||now<=0||now-stamp>.2||stamp-now>.01||
    !std::isfinite(fx)||!std::isfinite(fy)||!std::isfinite(fz)||fz<=0||!std::isfinite(qnorm)||std::abs(qnorm-1)>1e-3){trip();return false;}
  if(count&&stamp<last_stamp){trip();return false;}
  if(count&&stamp==last_stamp)return false; // Never refresh freshness or replace command.
  if(count&&(stamp-last_stamp>.2||now-received>.2)){trip();return false;}
  if(!armed&&(!std::isfinite(hover)||hover<=0||std::abs(fz-hover)>.05*hover||std::hypot(fx,fy)>.05*hover)){trip();return false;}
  valid=true;last_stamp=stamp;received=now;++count;
  if(phase==PREPARE&&count>=5)phase=ACK;
  return true;
 }
 bool permit(const std::string& token,double now){
  if(!std::isfinite(now)||now<=0||aborted||nonce.empty()||token!=nonce||phase!=ACK||!valid||count<5||now-last_stamp>.2||last_stamp-now>.01)return false;
  armed=true;phase=ARMED;return true;
 }
 bool step_allowed(double now){
  if(aborted)return false;
  if(!std::isfinite(now)||now<=0){trip();return false;}
  if(phase!=ARMED)return false;
  if(!valid||now-last_stamp>.2||now-received>.2||last_stamp-now>.01){trip();return false;}
  return true;
 }
};
}
