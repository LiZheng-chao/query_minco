"""Native renderer acquisition only. Virtual orientations, no flight execution."""
from pathlib import Path
import os,sys,json,time,subprocess,signal,socket,math,hashlib
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'research_outputs/forest_f1_blockers_20261118_v3'
def acquire():
    OUT.mkdir();(OUT/'frames').mkdir();binary=ROOT/'research_outputs/g4cp7b_fix_20261112_v1/build_v2/native_renderer'
    expected='18f42b9c9da1893272b17667ec096460fa1bd9acc6a9e295f927dde8ebe836a6'
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=expected:raise ValueError('RENDERER_SHA')
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    socket.setdefaulttimeout(2.)
    env=dict(os.environ,CMAKE_PREFIX_PATH='/opt/ros/noetic',PATH='/opt/ros/noetic/bin:'+os.environ['PATH'],ROS_MASTER_URI='http://127.0.0.1:%d'%port,ROS_IP='127.0.0.1',ROS_HOME=str(OUT/'ros_home'),ROS_LOG_DIR=str(OUT/'ros_logs'),
        ROS_PACKAGE_PATH='/opt/ros/noetic/share:/home/lizheng/YOPO/Controller/src',
        LD_LIBRARY_PATH='/usr/local/cuda-11.8/lib64:/opt/ros/noetic/lib:/home/lizheng/YOPO/Simulator/devel/lib',
        PYTHONPATH='/opt/ros/noetic/lib/python3/dist-packages:/home/lizheng/YOPO/Controller/devel/lib/python3/dist-packages',PYTHONDONTWRITEBYTECODE='1')
    os.environ.update(env);sys.path[:0]=env["PYTHONPATH"].split(":");children=[];commands=[]
    def spawn(name,argv):
        so=(OUT/(name+'.stdout.log')).open('x');se=(OUT/(name+'.stderr.log')).open('x')
        p=subprocess.Popen(argv,cwd='/home/lizheng/YOPO/Simulator',env=env,stdout=so,stderr=se,start_new_session=True);children.append((p,so,se));commands.append(dict(name=name,argv=argv,pid=p.pid));return p
    try:
        master=spawn('master',['/opt/ros/noetic/bin/roscore','-p',str(port)])
        import xmlrpc.client
        rpc=xmlrpc.client.ServerProxy(env['ROS_MASTER_URI'])
        for _ in range(100):
            try:
                if rpc.getPid('/virtual_views')[0]==1:break
            except OSError:pass
            time.sleep(.05)
        else:raise RuntimeError('MASTER_TIMEOUT')
        import rospy,numpy as np
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Image,CameraInfo
        from scipy.spatial.transform import Rotation
        rospy.init_node('virtual_view_fixture',disable_signals=True)
        rospy.set_param('/truth_export_path',str(OUT/'native_grid_GT_NOT_USED'))
        poses={};images={};cameras={}
        def image(m):images[m.header.stamp.to_nsec()]=(m,time.time(),time.monotonic())
        def camera(m):cameras[m.header.stamp.to_nsec()]=m
        sub=rospy.Subscriber('/virtual/depth',Image,image,queue_size=100,buff_size=10485760)
        cam=rospy.Subscriber('/virtual/camera_info',CameraInfo,camera,queue_size=100)
        pub=rospy.Publisher('/virtual/odom',Odometry,queue_size=1)
        sensor=spawn('renderer',[str(binary),'__name:=virtual_renderer','/sim/odom:=/virtual/odom','/depth_image:=/virtual/depth','/camera_info:=/virtual/camera_info'])
        for _ in range(400):
            if pub.get_num_connections()==1:break
            if sensor.poll() is not None:raise RuntimeError('RENDERER_EXIT')
            time.sleep(.05)
        else:raise RuntimeError('RENDERER_CONNECTION_TIMEOUT')
        records=[]
        for index,(yaw,pitch) in enumerate([(y,p) for y in [0,45,90,135,180,225,270,315] for p in [-25,0,25]]):
            q=Rotation.from_euler('ZY',[yaw,pitch],degrees=True).as_quat().tolist();deadline=time.monotonic()+2.;found=None
            while time.monotonic()<deadline:
                m=Odometry();m.header.frame_id='world';m.header.stamp=rospy.Time.now();m.pose.pose.position.x=.25;m.pose.pose.position.y=-.25;m.pose.pose.position.z=2.;m.pose.pose.orientation.x,m.pose.pose.orientation.y,m.pose.pose.orientation.z,m.pose.pose.orientation.w=q
                ns=m.header.stamp.to_nsec();poses[ns]=dict(p=[.25,-.25,2.],q=q,frame='world',v=[0.,0.,0.])
                pub.publish(m);time.sleep(.012)
                matches=[n for n in images if n in cameras and n in poses and poses[n]['q']==q]
                if matches:found=max(matches);break
            if found is None:raise RuntimeError('MATCHED_FRAME_TIMEOUT')
            im,receipt,mono=images[found];ci=cameras[found]
            if im.encoding!='32FC1' or im.header.frame_id!='camera_link':raise ValueError('IMAGE_CONTRACT')
            dep=np.ndarray((im.height,im.width),dtype='<f4',buffer=im.data,strides=(im.step,4)).copy()
            path=OUT/'frames'/('view_%02d.npz'%index);np.savez_compressed(path,depth=dep,stamp=np.array(im.header.stamp.to_sec()),stamp_ns=np.array(found),K=np.array(ci.K),position=np.array(poses[found]['p']),quaternion=np.array(q))
            records.append(dict(index=index,path=str(path),stamp=im.header.stamp.to_sec(),stamp_ns=found,receipt_ros=receipt,receipt_monotonic=mono,pose=poses[found],yaw_deg=yaw,pitch_deg=pitch,K=list(ci.K),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        graph=rpc.getSystemState('/virtual_views')
        assert not any(t in row[0] for row in graph[2][0] for t in ['position_cmd','so3_cmd','start_permit','prepare_permit'])
        (OUT/'INPUTS.json').write_text(json.dumps(dict(scope='VIRTUAL_VIEW_GEOMETRY_ONLY',fixed_birth=[.25,-.25,2.],records=records,live_graph=graph,renderer=str(binary),renderer_sha=expected,GT_used=False,controller=False,plant=False,ARM=False),indent=2)+'\n')
        print('VIRTUAL_VIEWS',len(records),'SPAN_S',records[-1]['stamp']-records[0]['stamp'])
        sub.unregister();cam.unregister();pub.unregister();rospy.signal_shutdown('done')
    finally:
        for p,so,se in reversed(children):
            if p.poll() is None:os.killpg(p.pid,signal.SIGINT)
            try:code=p.wait(timeout=8)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
            so.close();se.close()
            for c in commands:
                if c['pid']==p.pid:c['exit_code']=code
        (OUT/'ACQUISITION_COMMANDS.json').write_text(json.dumps(commands,indent=2)+'\n')
if __name__=='__main__':acquire()
