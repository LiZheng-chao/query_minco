"""Generate uncompiled future receiver interface diff; NOT Simulator patch."""
import difflib
from pathlib import Path
ROOT=Path('/home/lizheng/query_minco_mainline');OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution009'
def render():
 base=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution008/runtime/native_reference_dynamic_e4.cpp';old=base.read_text()
 s=old.replace('#include "neural_input_binding_e4.hpp"','#include "neural_input_binding_e4.hpp"\n#include "proposal_session_gate.hpp"')
 s=s.replace('qm::Verifier verifier;', 'qm::Verifier verifier;qm::ReadonlySessionGate sessionGate;')
 s=s.replace('bool ok=clockValid&&authority', 'auto actualPermission=sessionGate.check(liveStage.load(),clockValid,authority&&m-authorityTime.load()<=.5,finish.load());row.put("session_reason",actualPermission.reason);bool ok=actualPermission.allowed&&clockValid&&authority')
 needle="   if(input[0]=='I'){"
 replacement="""   auto permission=sessionGate.check(liveStage.load(),clockValid,authority&&mono()-authorityTime.load()<=.5,finish.load());
   if(!permission.allowed){answer.put("fault_latched",permission.fault);throw std::runtime_error(permission.reason);}
"""+needle
 assert needle in s;s=s.replace(needle,replacement)
 needle='if(readyR>meta.get<double>("activation_ros")-.02'
 replacement='auto finalPermission=sessionGate.check(liveStage.load(),clockValid,authority&&mono()-authorityTime.load()<=.5,finish.load());if(!finalPermission.allowed){verifier.readonlyPending=previous;verifier.highwater=oldHighwater;answer.put("fault_latched",finalPermission.fault);throw std::runtime_error(finalPermission.reason);}\n    '+needle
 assert needle in s;s=s.replace(needle,replacement)
 return old,s
def main():
 old,new=render();(OUT/'runtime/READONLY_RECEIVER_E5_REVIEW.cpp').write_text(new)
 (OUT/'patches/READONLY_RECEIVER_E5.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='existing_readonly_receiver',tofile='future_E5_readonly_receiver')))
 print('future readonly interface source rendered only; no ROS executable built')
if __name__=='__main__':main()
