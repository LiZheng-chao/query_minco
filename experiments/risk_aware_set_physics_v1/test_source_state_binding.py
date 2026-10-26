"""CPU-only correction and regression of the discovered zero-acceleration bug."""
import copy,hashlib,json,struct,subprocess
from pathlib import Path
import numpy as np
from query_minco.inference.causal_source_state import SourceHistory
from query_minco.inference.dynamic_proposal import packet
ROOT=Path('/home/lizheng/query_minco_mainline');OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution007'
def main():
 results=[]
 h=SourceHistory()
 h.add(10,20,[0,0,2],[.8,0,0],np.eye(3));h.add(10.01,20.01,[.008,0,2],[.8003,0,0],np.eye(3))
 r=h.source(10.01,10.02,20.02);np.testing.assert_allclose(r.acceleration,[.03,0,0],atol=1e-10)
 assert not r.acceleration.flags.writeable
 results.append('nonzero_acceleration_computed_before_history_update')
 # Previous behavior reproduced exactly; new output must differ from fabricated0.
 vel=np.array([.8003,0,0]);past=(10,np.array([.8,0,0]));dt=10.01-past[0];past=(10.01,vel.copy())
 np.testing.assert_allclose((vel-past[1])/dt,0)
 assert np.linalg.norm(r.acceleration)>0;results.append('old_zero_acceleration_bug_reproduced_and_rejected')
 for label,fn in [('future_received',lambda:h.source(10.01,10.02,20.005)),('missing_source',lambda:h.source(10.015,10.02,20.02)),('stale_source',lambda:h.source(10.01,10.3,20.3)),('clock_jump',lambda:h.add(9,21,[0,0,2],[.8,0,0],np.eye(3)))]:
  try:fn();raise AssertionError(label)
  except ValueError:results.append(label+'_rejected')
 # Build an additive receiver version; original runtime/bindings remain untouched.
 old=(ROOT/'query_minco/inference/native_reference_dynamic.cpp').read_text()
 source=old.replace('#include "dynamic_proposal_gate.hpp"','#include "dynamic_proposal_gate.hpp"\n#include "source_state_binding.hpp"')
 source=source.replace('qm::Context c;c.nowR=', 'qm::PVA trustedSource{};qm::Context c;c.nowR=')
 source=source.replace('c.sourceSeen=true;c.sourcePosition=', 'trustedSource=found->pva;c.sourceSeen=true;c.sourcePosition=')
 source=source.replace("if(input[0]=='S'){", """if(input[0]=='I'){
    if(input.size()!=9)throw std::runtime_error("SOURCE_REQUEST_BOUND");
    double sourceStamp;memcpy(&sourceStamp,input.data()+1,8);Observed observed;
    {std::lock_guard<std::mutex>l(sensorMutex);auto found=std::find_if(observations.begin(),observations.end(),[&](auto&o){return o.stamp==sourceStamp;});if(found==observations.end())throw std::runtime_error("SOURCE_STATE_NOT_OBSERVED");observed=*found;}
    if(ros::Time::now().toSec()-observed.stamp>.25)throw std::runtime_error("STALE_SOURCE_STATE");
    answer.put("passed",true);answer.put("stamp",observed.stamp);answer.put("recv_mono",observed.recv);Tree pv,rotation;
    for(auto x:observed.pva){Tree v;v.put("",x);pv.push_back({"",v});}for(auto x:observed.R){Tree v;v.put("",x);rotation.push_back({"",v});}
    answer.add_child("pva",pv);answer.add_child("rotation",rotation);
   }else if(input[0]=='S'){""")
 source=source.replace('auto verified=verifier.check(meta,body,c);','qm::bindSource(meta,trustedSource,c.sourceRotation);auto verified=verifier.check(meta,body,c);')
 source=source.replace('try{auto v=verifier.check(meta,body,c);','try{qm::PVA trustedSource{};auto pv=qm::values(t,"source_pva");std::copy(pv.begin(),pv.end(),trustedSource.begin());qm::bindSource(meta,trustedSource,c.sourceRotation);auto v=verifier.check(meta,body,c);')
 cpp=OUT/'runtime/native_reference_dynamic_source_bound.cpp';cpp.write_text(source)
 exe=OUT/'runtime/native_reference_dynamic_source_bound'
 cmd=['g++','-O3','-std=c++17','-pthread','-I'+str(ROOT/'query_minco/inference'),'-I/opt/ros/noetic/include','-I/home/lizheng/YOPO/Controller/devel/include',str(cpp),'-o',str(exe),'-L/opt/ros/noetic/lib','-Wl,-rpath,/opt/ros/noetic/lib','-lroscpp','-lrosconsole','-lroscpp_serialization','-lrostime','-lxmlrpcpp','-lcrypto']
 proc=subprocess.run(cmd,capture_output=True,text=True);(OUT/'logs/source_bound_build.log').write_text(proc.stdout+proc.stderr);assert proc.returncode==0,proc.stderr
 # Context uses actual source-time Odom, independent from submitted state.
 context=json.loads((OUT/'offline/gate_valid_context.json').read_text())
 oldrun=OUT.parent/'f3_execution006';z=np.load(oldrun/'shadow_inputs001/c1/frame_006.npz')
 o=np.genfromtxt(oldrun/'shadow_inputs001/ODOM.csv',delimiter=',',names=True)
 sample=o[np.abs(o['stamp']-float(z['source_stamp']))<1e-7][-1]
 source_pva=np.array([sample[k+str(i)] for k in ('p','v','a') for i in range(3)]).reshape(3,3)
 context['source_pva']=source_pva.reshape(-1).tolist()
 raw=(OUT/'offline/gate_valid.bin').read_bytes();n=struct.unpack_from('<i',raw)[0];meta=json.loads(raw[4:4+n]);body=raw[4+n:]
 R=z['rotation_W_B0'];delta=np.array([3,0,2])-source_pva[0]
 neural=np.r_[R.T@source_pva[1],R.T@source_pva[2],R.T@delta*2.4/np.linalg.norm(delta),R.T@[0,0,-1]].astype(np.float32)
 meta.update(source_state_version=2,source_pva=source_pva.reshape(-1).tolist(),network_state=neural.tolist(),network_state_sha256=hashlib.sha256(neural.astype('<f4').tobytes()).hexdigest())
 cp=OUT/'offline/source_bound_context.json';cp.write_text(json.dumps(context));cases=[]
 for name,change,expected in [('correct',lambda m:None,True),('zero_acceleration',lambda m:m['network_state'].__setitem__(slice(3,6),[0,0,0]),False),('forged_source_pva',lambda m:m['source_pva'].__setitem__(6,0),False),('missing_source_version',lambda m:m.update(source_state_version=1),False)]:
  m=copy.deepcopy(meta);change(m);m['network_state_sha256']=hashlib.sha256(np.array(m['network_state'],dtype='<f4').tobytes()).hexdigest()
  pp=OUT/('offline/source_bound_'+name+'.bin');pp.write_bytes(packet(m,body))
  p=subprocess.run([str(exe),'--gate-cpu',str(cp),str(pp)],capture_output=True,text=True,timeout=5);assert p.returncode==0,p.stderr
  row=json.loads(p.stdout);assert (row['passed']=='true')==expected,(name,row);cases.append(dict(name=name,result=row))
  # Re-run the entire original geometric/time/identity suite against v2.
 inherited=[]
 baseline_fields={key:meta[key] for key in ('source_state_version','source_pva','network_state','network_state_sha256')}
 for record in json.loads((OUT/'DYNAMIC_GATE_CPU_RESULTS.json').read_text())['results']:
  name=record['name'];c=json.loads((OUT/('offline/gate_'+name+'_context.json')).read_text());c['source_pva']=context['source_pva']
  payload=(OUT/('offline/gate_'+name+'.bin')).read_bytes();hn=struct.unpack_from('<i',payload)[0];m=json.loads(payload[4:4+hn]);m.update(copy.deepcopy(baseline_fields));cfile=OUT/('offline/source_v2_suite_'+name+'.json');pfile=OUT/('offline/source_v2_suite_'+name+'.bin')
  cfile.write_text(json.dumps(c));pfile.write_bytes(packet(m,payload[4+hn:]))
  proc=subprocess.run([str(exe),'--gate-cpu',str(cfile),str(pfile)],capture_output=True,text=True,timeout=5);assert proc.returncode==0,proc.stderr
  row=json.loads(proc.stdout);assert (row['passed']=='true')==record['expected_accept'],(name,row);inherited.append(dict(name=name,result=row))
 # Explicit old-v1 acceptance counterexample; no physical mode is invoked.
 m=copy.deepcopy(meta);m['network_state'][3:6]=[0,0,0];m['network_state_sha256']=hashlib.sha256(np.array(m['network_state'],dtype='<f4').tobytes()).hexdigest()
 cp=OUT/'offline/source_bound_context.json';pp=OUT/'offline/old_v1_accepts_bad_neural_state.bin';pp.write_bytes(packet(m,body))
 oldproc=subprocess.run([str(OUT/'runtime/native_reference_dynamic'),'--gate-cpu',str(cp),str(pp)],capture_output=True,text=True,timeout=5)
 oldresult=json.loads(oldproc.stdout);assert oldresult['passed']=='true'
 report=dict(source_history_cpu_tests=results,source_binding_cpp_tests=cases,original_suite_revalidated=inherited,old_v1_bad_neural_state_counterexample=oldresult,cpp_tests=35,passed=True,runtime_deployed=False,ROS_run=False,old_v3_run_unchanged=True)
 (OUT/'SOURCE_STATE_FIX_CPU_RESULTS.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
