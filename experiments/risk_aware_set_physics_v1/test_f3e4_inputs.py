"""Known refusal suite plus independent image and genuine history regressions."""
import copy,hashlib,json,struct,subprocess
from pathlib import Path
import numpy as np
from query_minco.inference.dynamic_proposal import packet
from query_minco.inference.shadow import decode_image
from query_minco.inference.causal_source_state import SourceHistory
from query_minco.inference.shadow_camera_cadence import output_stamps
ROOT=Path('/home/lizheng/query_minco_mainline');OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution008';OLD=OUT.parent/'f3_execution007'
def main():
 exe=OUT/'runtime/native_reference_dynamic_e4'
 valid=(OLD/'offline/source_bound_correct.bin').read_bytes();n=struct.unpack_from('<i',valid)[0];correct=json.loads(valid[4:4+n]);body=valid[4+n:]
 context=json.loads((OLD/'offline/source_bound_context.json').read_text())
 source=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution006/shadow_inputs001'
 z=np.load(source/'c1/frame_006.npz');t=float(z['source_stamp']);odom=np.genfromtxt(source/'ODOM.csv',delimiter=',',names=True)
 i=int(np.where(np.abs(odom['stamp']-t)<1e-7)[0][-1]);previous=odom[i-1];prevstamp=float(previous['stamp']);prevv=[float(previous[k]) for k in ('v0','v1','v2')]
 image=z['depth'];imageSha=hashlib.sha256(image.astype('<f4').tobytes()).hexdigest()
 additions=dict(input_depth_mask_sha256=imageSha,camera_info_stamp=t,camera_frame='camera_link',source_previous_stamp=prevstamp,source_previous_velocity=prevv)
 context.update(input_depth_mask_sha256=imageSha,source_previous_stamp=prevstamp,source_previous_velocity=prevv)
 base={k:correct[k] for k in ('source_state_version','source_pva','network_state','network_state_sha256')};base.update(additions)
 cases=[]
 # Keep all previous geometric/time/identity mutations; true source fields added.
 suite=json.loads((OLD/'DYNAMIC_GATE_CPU_RESULTS.json').read_text())['results']
 for row in suite:
  name=row['name'];raw=(OLD/('offline/gate_'+name+'.bin')).read_bytes();hn=struct.unpack_from('<i',raw)[0];m=json.loads(raw[4:4+hn]);m.update(copy.deepcopy(base))
  c=json.loads((OLD/('offline/gate_'+name+'_context.json')).read_text());c.update(source_pva=context['source_pva'],input_depth_mask_sha256=imageSha,source_previous_stamp=prevstamp,source_previous_velocity=prevv)
  cases.append((name,m,raw[4+hn:],c,row['expected_accept']))
 for name,fn in [
 ('wrong_acceleration',lambda m:m['network_state'].__setitem__(3,m['network_state'][3]+.2)),
 ('zero_acceleration_substitution',lambda m:m['network_state'].__setitem__(slice(3,6),[0,0,0])),
 ('wrong_history_stamp',lambda m:m.update(source_previous_stamp=m['depth_stamp'])),
 ('wrong_history_velocity',lambda m:m['source_previous_velocity'].__setitem__(0,0)),
 ('wrong_depth_mask',lambda m:m.update(input_depth_mask_sha256='0'*64)),
 ('wrong_camera_stamp',lambda m:m.update(camera_info_stamp=m['camera_info_stamp']-.01)),
 ('missing_source_schema',lambda m:m.update(source_state_version=1)),
 ('wrong_source_pva',lambda m:m['source_pva'].__setitem__(6,0))]:
  m=copy.deepcopy(correct);m.update(copy.deepcopy(additions));fn(m)
  m['network_state_sha256']=hashlib.sha256(np.array(m['network_state'],dtype='<f4').tobytes()).hexdigest()
  cases.append((name,m,body,copy.deepcopy(context),False))
 results=[]
 for name,m,b,c,expect in cases:
  cp=OUT/('offline/input_'+name+'.json');bp=OUT/('offline/input_'+name+'.bin');cp.write_text(json.dumps(c));bp.write_bytes(packet(m,b))
  r=subprocess.run([str(exe),'--gate-cpu',str(cp),str(bp)],capture_output=True,text=True,timeout=5)
  assert r.returncode==0,(name,r.stderr)
  result=json.loads(r.stdout);assert (result['passed']=='true')==expect,(name,result)
  assert result['physical_reference_and_backup_preserved']=='true'
  if not expect:assert result['readonly_latch_preserved']=='true'
  results.append(dict(name=name,expected_accept=expect,result=result,passed=True))
 # C++ and Python encode actual/synthetic optical depth identically.
 images=[]
 for name,raw in [('actual_saved',np.load(OLD/'shadow_inputs001/camera_proof_000.npz')['query']),('edge_cases',np.resize(np.array([0.,-1.,20.,19.9,np.nan,np.inf,1e-40,2.],np.float32),(128,160)))]:
  path=OUT/('offline/image_'+name+'.raw');path.write_bytes(raw.astype('<f4').tobytes())
  value=subprocess.check_output([str(exe),'--input-image-sha',str(path)],text=True).strip()
  expected=hashlib.sha256(decode_image(path.read_bytes(),'32FC1',160,128,640,False).astype('<f4').tobytes()).hexdigest()
  assert value==expected,(name,value,expected);images.append(dict(name=name,sha256=value,passed=True))
 h=SourceHistory();h.add(1,2,[0,0,2],[.8,0,0],np.eye(3));h.add(1.01,2.01,[.008,0,2],[.8003,0,0],np.eye(3))
 sourceState=h.source(1.01,1.02,2.02);np.testing.assert_allclose(sourceState.acceleration,[.03,0,0],atol=1e-10);assert not sourceState.acceleration.flags.writeable
 history=[]
 for name,fn in [('old_source',lambda:h.source(1.01,1.3,2.3)),('future_receipt',lambda:h.source(1.01,1.02,2.005)),('wrong_history',lambda:h.source(1.015,1.02,2.02)),('clock_reset',lambda:h.add(.5,3,[0,0,2],[.8,0,0],np.eye(3)))]:
  try:fn();raise AssertionError(name)
  except ValueError:history.append(dict(name=name,rejected=True))
 stamps=np.rint(np.arange(67)*1e9/33).astype(np.int64).tolist();out30=output_stamps(stamps,30);assert len(out30)>=60 and max(np.diff(out30))<=100_000_000
 report=dict(mode='CPU_ONLY_NOT_NATIVE',passed=True,cpp_cases=results,case_count=len(results),independent_image_tests=images,nonzero_history_test=True,history_refusals=history,cadence=dict(native_count=len(stamps),output_30Hz_count=len(out30),max_source_gap_ns=int(max(np.diff(out30)))),field12_errors=results[0]['result'].get('input_audit'),Query_control=False)
 (OUT/'CPU_INPUT_ACCEPTANCE.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('cpp_cases','field12_errors')},indent=2))
if __name__=='__main__':main()
