"""Historical-clock CPU gate tests, independent contexts; never ROS control."""
import copy,hashlib,json,pickle,struct,subprocess
from pathlib import Path
import numpy as np
from query_minco.inference.proposal_worker import NativeWorker,curve_bytes,decode_body
from query_minco.inference.dynamic_proposal import metadata,packet,CALIBRATION
from query_minco.inference.observed_space import FreeVolume,box_cell,Scope
from query_minco.inference.trajectory_execution import Program
ROOT=Path('/home/lizheng/query_minco_mainline');OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution007';OLD=OUT.parent/'f3_execution006'
def main():
 with (OLD/'shadow_inputs001/c1/banks_006.pkl').open('rb') as f:old=pickle.load(f)[5]
 with (OLD/'shadow_preparation/TRANSPORT_BUNDLES.pkl').open('rb') as f:refs=pickle.load(f)
 z=np.load(OLD/'shadow_inputs001/c1/frame_006.npz')
 q=json.loads((OLD/'E_REALTIME_RESULTS.json').read_text())['complete_proposals'][0]
 ev=[json.loads(l) for l in (OLD/'shadow_inputs001/c1/events.jsonl').read_text().splitlines()][6];row=next(r for r in ev['queries'] if r['passed'])
 p=Program(z['q5_prefix_coefficients'],z['q5_prefix_durations']);worker=NativeWorker(OUT/'runtime/libproposal_numerics.so')
 volume=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],refs[0].valid_until_ros,'VERIFIED_EMPTY_ROOM'),),Scope.EXTERNAL_SUPERVISOR)
 b=worker.prepare(p,volume,old.prepared_ros,old.prepared_mono,inherited_mono_expiry=refs[0].valid_until_mono)
 snapshot=dict(token=1,stamp=row['measurement_stamp'],recv_mono=row['measurement_received_mono'],pva=np.array(row['measurement_pva']).reshape(-1).tolist())
 config=dict(owner_epoch='CPU_OFFLINE_ONLY',geometry='VERIFIED_EMPTY_ROOM_SHA',parent_digest=hashlib.sha256(curve_bytes(Program(np.frombuffer(refs[1].prefix.coefficients,dtype='<f8').reshape(-1,3,6),refs[1].prefix.durations))).hexdigest())
 ar=ev['activation_ros'];am=old.prepared_mono+ar-old.prepared_ros
 meta=metadata(b,snapshot,1,ev['source_stamp'],z['position_W'],z['rotation_W_B0'],ar,am,config)
 import csv
 with (OLD/'shadow_inputs001/EVENT.csv').open() as f:epoch=next(r for r in csv.DictReader(f) if r['code']=='11' and r['bundle']=='1')
 ctx=dict(now_ros=old.prepared_ros+.008,now_mono=old.prepared_mono+.008,state_stamp=snapshot['stamp'],state_recv_mono=snapshot['recv_mono'],planning_pva=snapshot['pva'],source_position=z['position_W'].tolist(),source_rotation=z['rotation_W_B0'].reshape(-1).tolist(),parent_start_ros=float(epoch['y']),cell_expiry_ros=refs[0].valid_until_ros,cell_expiry_mono=refs[0].valid_until_mono,owner_epoch=config['owner_epoch'],geometry=config['geometry'],calibration=CALIBRATION,parent_digest=config['parent_digest'],snapshot_token=1,reference_file=str(OLD/'shadow_preparation/NATIVE_REFERENCES.bin'))
 cases=[('valid',meta,b.body,ctx,True)]
 def change(name,fn):
  m=copy.deepcopy(meta);c=copy.deepcopy(ctx);body=b.body;ret=fn(m,c,body);body=body if ret is None else ret;cases.append((name,m,body,c,False))
 change('wrong_stop_bank_version',lambda m,c,b:m.update(stop_bank_version=0))
 change('wrong_method',lambda m,c,b:m.update(method='old_S1'))
 change('wrong_checkpoint',lambda m,c,b:m.update(checkpoint='0'*64))
 change('wrong_frame',lambda m,c,b:m.update(frame='camera'))
 change('wrong_calibration',lambda m,c,b:m.update(calibration='wrong'))
 change('wrong_transform',lambda m,c,b:m['source_rotation'].__setitem__(0,m['source_rotation'][0]+.01))
 change('wrong_position',lambda m,c,b:m['source_position'].__setitem__(0,m['source_position'][0]+1))
 change('old_owner_epoch',lambda m,c,b:m.update(owner_epoch='OLD'))
 change('old_parent_version',lambda m,c,b:m.update(parent_version=0))
 change('wrong_parent_digest',lambda m,c,b:m.update(parent_digest='0'*64))
 change('publisher_conflict',lambda m,c,b:c.update(publisher_unique=False))
 change('clock_jump',lambda m,c,b:c.update(clock_valid=False))
 change('monotonic_offset_error',lambda m,c,b:m.update(activation_mono=m['activation_mono']+.1))
 change('expired_certificate',lambda m,c,b:m.update(valid_until_ros=c['now_ros']-.01))
 change('extended_certificate',lambda m,c,b:m.update(valid_until_ros=c['cell_expiry_ros']+1))
 change('old_state',lambda m,c,b:c.update(state_stamp=c['now_ros']-.051))
 change('old_state_receipt',lambda m,c,b:c.update(state_recv_mono=c['now_mono']-.051))
 change('forged_planning_state',lambda m,c,b:m['planning_pva'].__setitem__(0,m['planning_pva'][0]+.01))
 change('unknown_snapshot_token',lambda m,c,b:m.update(snapshot_token=2))
 change('old_depth',lambda m,c,b:m.update(depth_stamp=c['now_ros']-.251,source_odom_stamp=c['now_ros']-.251))
 change('future_depth',lambda m,c,b:m.update(depth_stamp=c['now_ros']+.001,source_odom_stamp=c['now_ros']+.001))
 change('unobserved_source',lambda m,c,b:c.update(source_seen=False))
 change('deadline_missed',lambda m,c,b:m.update(activation_ros=c['now_ros']+.019,activation_mono=c['now_mono']+.019))
 change('duplicate',lambda m,c,b:c.update(highwater=1))
 change('out_of_order',lambda m,c,b:c.update(highwater=2))
 change('weaker_error_budget',lambda m,c,b:m['error_budget'].__setitem__(0,0.))
 change('wrong_geometry_id',lambda m,c,b:m.update(geometry='DECLARED_CERTIFIED_FREE'))
 def translated(m,c,body,rehash):
  prefix,stops=decode_body(body)
  def shift(p):
   a=p.coefficients.copy();a[:,0,0]+=100;return Program(a,p.durations)
  encoded=curve_bytes(shift(prefix))+struct.pack('<i',len(stops))
  for phase,stop in stops:encoded+=struct.pack('<d',phase)+curve_bytes(shift(stop))
  if rehash:m['body_sha256']=hashlib.sha256(encoded).hexdigest()
  return encoded
 change('translated100m_old_proof_and_digest',lambda m,c,b:translated(m,c,b,False))
 change('translated100m_rehashed_same_old_cell_proof',lambda m,c,b:translated(m,c,b,True))
 def missing(m,c,body):
  pre,stops=decode_body(body);stops.pop(5);encoded=curve_bytes(pre)+struct.pack('<i',len(stops))
  for phase,stop in stops:encoded+=struct.pack('<d',phase)+curve_bytes(stop)
  m['body_sha256']=hashlib.sha256(encoded).hexdigest();return encoded
 change('missing_stop_phase_rehashed',missing)
 results=[]
 for name,m,body,c,expect in cases:
  cp=OUT/('offline/gate_'+name+'_context.json');pp=OUT/('offline/gate_'+name+'.bin');cp.write_text(json.dumps(c));pp.write_bytes(packet(m,body))
  proc=subprocess.run([str(OUT/'runtime/native_reference_dynamic'),'--gate-cpu',str(cp),str(pp)],capture_output=True,text=True,timeout=5)
  assert proc.returncode==0,(name,proc.stderr)
  result=json.loads(proc.stdout);passed=result['passed'] in (True,'true')
  assert passed==expect,(name,result)
  assert result['physical_reference_and_backup_preserved'] in (True,'true')
  if not expect:assert result['readonly_latch_preserved'] in (True,'true')
  results.append(dict(name=name,expected_accept=expect,gate=result,test_pass=True))
 report=dict(mode='HISTORICAL_OFFLINE_CPU_NO_AUTHORIZATION',tests=len(results),passed=len(results),invalid_rejected=len(results)-1,results=results,Query_control=False)
 (OUT/'DYNAMIC_GATE_CPU_RESULTS.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(tests=len(results),passed=len(results),invalid_rejected=len(results)-1),indent=2))
if __name__=='__main__':main()
