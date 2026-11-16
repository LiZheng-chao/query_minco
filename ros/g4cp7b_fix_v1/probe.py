"""Non-ARM native sensor/renderer latency probe;bounded samples,source and receipt separate."""
from pathlib import Path
import sys,argparse,time,json,collections,threading,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
 import rospy,numpy as np
 from sensor_msgs.msg import Image,CameraInfo
 from std_msgs.msg import String
 from query_minco.integration.yopo_execution_v4.lifecycle import Journal,atomic_json
 run=Path(a.run);out=run/'sensor_probe';out.mkdir();j=Journal(out/'EVENTS.jsonl');rospy.init_node('fixed_native_sensor_probe')
 gpu=[];gpu_stop=threading.Event()
 def gpu_sample():
  while not gpu_stop.is_set():
   try:
    p=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=1)
    gpu.append(dict(monotonic=time.monotonic(),exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr))
   except Exception as e:gpu.append(dict(error=repr(e)))
   gpu_stop.wait(1)
 gpu_worker=threading.Thread(target=gpu_sample,daemon=True);gpu_worker.start()
 depth=[];timing={};errors=[];began=time.monotonic();camera=None;steps=0
 def dep(m):
  with j.callback() as active:
   if not active:return
   start=time.perf_counter();now=rospy.Time.now().to_sec();s=m.header.stamp.to_sec()
   if m.header.frame_id!='camera_link' or m.encoding!='32FC1' or m.is_bigendian or m.step<640 or (m.height,m.width)!=(90,160):errors.append('DEPTH_CONTRACT')
   row=dict(source_stamp=s,callback_ros=now,receipt_monotonic=time.monotonic(),callback_s=time.perf_counter()-start)
   depth.append(row);j.append(dict(kind='depth',**row));row['callback_s']=time.perf_counter()-start
 def met(m):
  with j.callback() as active:
   if not active:return
   d=json.loads(m.data);d['metric_receipt_ros']=rospy.Time.now().to_sec();timing[d['source_stamp']]=d;j.append(dict(kind='renderer',**d))
   if len(timing)>3000:timing.pop(next(iter(timing)))
 def cam(m):
  nonlocal camera
  with j.callback() as active:
   if active:
    camera=list(m.K)
    if m.header.frame_id!='camera_link' or camera!=[80.,0.,80.,0.,80.,45.,0.,0.,1.]:errors.append('CAMERA_CONTRACT')
 def plant(m):
  nonlocal steps
  with j.callback() as active:
   if active:
    d=json.loads(m.data);steps=max(steps,d['integration_steps'])
    if d['phase']!='HELD' or steps:errors.append('UNAUTHORIZED_INTEGRATION')
 subs=[rospy.Subscriber('/p2/native/depth',Image,dep,queue_size=2),rospy.Subscriber('/query_minco/renderer_timing',String,met,queue_size=100),rospy.Subscriber('/p2/native/camera_info',CameraInfo,cam,queue_size=1),rospy.Subscriber('/p2/startup_state',String,plant,queue_size=10)]
 try:
  while not rospy.is_shutdown() and time.monotonic()-began<32:
   if errors:break
   if depth and time.monotonic()-depth[0]['receipt_monotonic']>=20:break
   rospy.sleep(.01)
 finally:
  gpu_stop.set();gpu_worker.join(2)
  atomic_json(out/'GPU_SAMPLES.json',gpu)
  cleanup=j.close([s.unregister for s in subs])
  src=[d['source_stamp'] for d in depth];receipt=[d['receipt_monotonic'] for d in depth]
  gap=max(np.diff(src)) if len(src)>1 else 999.;rgap=max(np.diff(receipt)) if len(src)>1 else 999.
  rate=(len(src)-1)/(src[-1]-src[0]) if len(src)>1 and src[-1]>src[0] else 0
  def stats(values):
   return dict(n=len(values),p50_ms=float(np.percentile(values,50)*1000),p95_ms=float(np.percentile(values,95)*1000),p99_ms=float(np.percentile(values,99)*1000),max_ms=float(max(values)*1000)) if values else dict(n=0)
  paired=[(d,timing[d['source_stamp']]) for d in depth if d['source_stamp'] in timing]
  phases={name:stats([fn(d,t) for d,t in paired]) for name,fn in [
   ('source_to_renderer_receipt',lambda d,t:t['odom_receipt']-t['source_stamp']),
   ('renderer_queue_to_render',lambda d,t:t['render_start']-t['odom_receipt']),
   ('render_and_readback',lambda d,t:t['render_end']-t['render_start']),
   ('pack_and_publish',lambda d,t:t['publish_ros']-t['render_end']),
   ('publish_to_depth_callback',lambda d,t:d['callback_ros']-t['publish_ros']),
   ('source_to_depth_callback',lambda d,t:d['callback_ros']-t['source_stamp'])]}
  passed=all(x>0 for x in np.diff(src)) and all(0<=d['callback_ros']-d['source_stamp']<=.2 for d in depth) and not errors and not cleanup and steps==0 and len(depth)>=500 and gap<=.2 and rgap<=.2 and rate>=25 and len(paired)>=.9*len(depth) and camera is not None
  atomic_json(out/'RESULT.json',dict(status='PASS' if passed else 'FAIL',frames=len(depth),paired=len(paired),source_max_gap_s=float(gap),receipt_max_gap_s=float(rgap),rate_hz=rate,latency=phases,callback_cost=stats([d['callback_s'] for d in depth]),errors=errors,cleanup=cleanup,integration_steps=steps,scope='HELD_SENSOR_STABILITY_ONLY'))
 if not passed:raise SystemExit(2)
if __name__=='__main__':main()
