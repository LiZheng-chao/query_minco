#pragma once
// Nonphysical session guard. No reference/backup mutation or ROS dependency.
#include <atomic>
namespace qm {
struct SessionPermission {bool allowed;bool fault;const char*reason;};
class ReadonlySessionGate {
 std::atomic<bool> stopped_{false};
public:
 SessionPermission check(int stage,bool clock_ok,bool publisher_unique,bool finished) noexcept {
  if(stage==0&&!stopped_.load()&&clock_ok&&!finished)return {false,false,"WAIT_NOMINAL_SESSION"};
  if(finished)stopped_.store(true);
  if(stage==6||!clock_ok||!publisher_unique)stopped_.store(true);
  if(stopped_.load())return {false,true,finished?"SESSION_FINISHED":"SESSION_FAULT_LATCHED"};
  if(stage!=1)return {false,false,"WAIT_NOMINAL_SESSION"};
  return {true,false,"NOMINAL_READONLY"};
 }
 bool latched()const noexcept{return stopped_.load();}
};
}
