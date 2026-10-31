"""Real native full-ray/depth+odom diagnostic. No controller Publisher, no GT input."""
from pathlib import Path
import sys,time,json,threading
sys.path.insert(0,'/home/lizheng/query_minco_mainline')
import numpy as np,rospy,message_filters
from sensor_msgs.msg import Image,CameraInfo
from nav_msgs.msg import Odometry
from query_minco.integration.stagec_execution_v2.renderer_volume import RendererHistory,NativeModel
from query_minco.inference.shadow import rotation
out=Path(sys.argv[1]);frames=out/'FULL_NATIVE_INPUTS';frames.mkdir(exist_ok=False);rospy.init_node('stagec_full_volume_shadow',disable_signals=True);h=RendererHistory();log=(out/'FULL_VOLUME.jsonl').open('x',buffering=1);counts={};errors=[];lock=threading.Lock();m=NativeModel()
def cb(d,k,o):
 with lock:
  row=dict(stamp_ns=d.header.stamp.to_nsec(),received_wall_ns=time.time_ns())
  try:
   assert d.header.stamp==k.header.stamp==o.header.stamp,'EXACT_SYNC'
   assert d.header.frame_id==k.header.frame_id=='camera_link' and o.header.frame_id=='world','FRAME'
   K=np.array(k.K).reshape(3,3);assert (k.width,k.height)==(800,512) and np.array_equal(K,np.array([[400.,0,400],[0,400,254],[0,0,1.]])),'ACTUAL_NATIVE_K'
   assert np.array_equal(np.array(k.R).reshape(3,3),np.eye(3)) and np.array_equal(np.asarray(k.P).reshape(3,4),np.c_[K,np.zeros(3)]) and np.all(np.asarray(k.D)==0),'RECTIFICATION'
   assert d.encoding=='32FC1' and d.width==800 and d.height==512 and d.step==3200 and not d.is_bigendian,'RAW_DEPTH_FORMAT'
   now=rospy.Time.now().to_sec();stamp=d.header.stamp.to_sec();assert 0<=now-stamp<=.25,'STALE_NATIVE'
   depth=np.frombuffer(d.data,dtype='<f4').reshape(512,800).copy();p=o.pose.pose;origin=np.array([p.position.x,p.position.y,p.position.z]);R=rotation([p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w]);h.add(stamp,depth,origin,R)
   c=np.zeros((3,3,6));c[:,:,0]=origin+R@np.array([1.,0,0]);cert=h.certify(c,np.ones(3)*.02,now)
   np.savez_compressed(str(frames/(str(d.header.stamp.to_nsec())+'.npz')),depth_m=depth,origin=origin,rotation=R,source_stamp_ns=d.header.stamp.to_nsec(),camera_K=K)
   row.update(status='RECORDED_DIAGNOSTIC_ONLY',forward_probe_certificate=cert,pose=origin.tolist(),age_s=now-stamp,execution_permission=False);counts[cert['status']]=counts.get(cert['status'],0)+1
  except Exception as ex:row.update(status='REJECTED',reason=str(ex));errors.append(str(ex))
  log.write(json.dumps(row)+'\n')
ds=message_filters.Subscriber('/forest30/native_depth',Image,queue_size=2,buff_size=8<<20,tcp_nodelay=True);ks=message_filters.Subscriber('/forest30/native_camera_info',CameraInfo,queue_size=2);osub=message_filters.Subscriber('/sim/odom',Odometry,queue_size=200,tcp_nodelay=True);sync=message_filters.TimeSynchronizer([ds,ks,osub],200);sync.registerCallback(cb);start=time.monotonic()
while time.monotonic()-start<12 and not rospy.is_shutdown():time.sleep(.02)
with lock:
 log.close();(out/'FULL_VOLUME_RESULT.json').write_text(json.dumps(dict(status='DIAGNOSTIC_COMPLETE',counts=counts,rejected=len(errors),reasons=sorted(set(errors)),online_releases=0,GT_inputs=False,forward_probe_scope='CONDITIONAL_VOXEL_MODEL_ONLY_NOT_ACTUAL_QUERY_PERMISSION'),indent=2))
rospy.signal_shutdown('bounded full-ray diagnostic finished')
