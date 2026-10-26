"""Exact integer ray extraction from one unchanged native CUDA renderer."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
FULL_K=np.array([[400.,0,400.],[0,400.,254.],[0,0,1.]])
QUERY_K=np.array([[100.,0,79.5],[0,100.,63.5],[0,0,1.]])
YOPO_K=np.array([[80.,0,80.],[0,80.,45.],[0,0,1.]])
def extract(z):
    if z.shape!=(512,800) or z.dtype!=np.float32:raise ValueError("NATIVE_SUPER_CAMERA")
    return np.ascontiguousarray(z[0:512:4,82:722:4]),np.ascontiguousarray(z[29:479:5,0:800:5])
def main():
    import rospy,message_filters
    from sensor_msgs.msg import Image,CameraInfo
    p=argparse.ArgumentParser();p.add_argument("--host",required=True);a=p.parse_args()
    host=Path(a.host);log=(host/"CAMERA_TRACE.jsonl").open("x",buffering=1)
    rospy.init_node("forest30_camera_lattice",anonymous=False)
    pubs={n:rospy.Publisher(t,typ,queue_size=1) for n,t,typ in [
        ("qd","/query_shadow/depth",Image),("qc","/query_shadow/camera_info",CameraInfo),
        ("yd","/depth_image",Image),("yc","/camera_info",CameraInfo)]}
    next_query=None;count=0;proofs=0
    def callback(image,info):
        nonlocal next_query,count,proofs
        try:
            assert image.header.stamp==info.header.stamp
            assert image.header.frame_id==info.header.frame_id=="camera_link"
            assert image.encoding=="32FC1" and not image.is_bigendian
            assert (image.height,image.width)==(512,800) and image.step>=3200
            assert len(image.data)==image.height*image.step
            assert np.array_equal(np.array(info.K).reshape(3,3),FULL_K)
            assert np.array_equal(np.array(info.R).reshape(3,3),np.eye(3))
            assert np.array_equal(np.array(info.P).reshape(3,4),np.c_[FULL_K,np.zeros(3)])
            assert np.all(np.array(info.D)==0) and info.binning_x in (0,1) and info.binning_y in (0,1)
            assert info.roi.x_offset==info.roi.y_offset==info.roi.width==info.roi.height==0
            full=np.ndarray((512,800),dtype="<f4",buffer=image.data,strides=(image.step,4))
            q,y=extract(full)
            def publish(data,K,dp,cp):
                msg=Image();msg.header=copy.deepcopy(image.header);msg.height,msg.width=data.shape
                msg.encoding="32FC1";msg.is_bigendian=0;msg.step=4*msg.width;msg.data=data.tobytes()
                c=copy.deepcopy(info);c.height,c.width=data.shape;c.K=K.reshape(-1).tolist();c.P=np.c_[K,np.zeros(3)].reshape(-1).tolist()
                cp.publish(c);dp.publish(msg)
            publish(y,YOPO_K,pubs["yd"],pubs["yc"])
            stamp=image.header.stamp.to_nsec()
            if next_query is None:next_query=stamp
            if stamp>=next_query:
                while next_query<=stamp:next_query+=200_000_000
                publish(q,QUERY_K,pubs["qd"],pubs["qc"])
                mask=np.isfinite(q)&(q>0)&(q<20)
                encoded=np.stack([np.where(mask,q/20,0),mask.astype(np.float32)]).astype(np.float32)
                rec=dict(stamp_ns=stamp,query_index=count,input_age_s=rospy.Time.now().to_sec()-image.header.stamp.to_sec(),
                    query_encoded_sha=hashlib.sha256(encoded.tobytes()).hexdigest(),
                    query_raw_sha=hashlib.sha256(q.tobytes()).hexdigest(),yopo_raw_sha=hashlib.sha256(y.tobytes()).hexdigest())
                if count%25==0 and proofs<5:
                    name="camera_proof_%03d.npz"%proofs
                    with (host/name).open("xb") as f:np.savez_compressed(f,full=full,query=q,yopo=y,stamp_ns=np.array(stamp,dtype=np.int64))
                    rec["proof"]=name;proofs+=1
                log.write(json.dumps(rec)+"\n");count+=1
        except Exception as e:
            log.write(json.dumps(dict(type="FATAL",error=repr(e)))+"\n");rospy.signal_shutdown("camera validation failed")
    subs=[message_filters.Subscriber("/forest30/native_depth",Image,queue_size=2,buff_size=8<<20,tcp_nodelay=True),
          message_filters.Subscriber("/forest30/native_camera_info",CameraInfo,queue_size=3)]
    sync=message_filters.TimeSynchronizer(subs,20);sync.registerCallback(callback)
    try:rospy.spin()
    finally:log.close()
if __name__=="__main__":main()
