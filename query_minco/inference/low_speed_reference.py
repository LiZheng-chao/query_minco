"""Isolated controlled-reference sensor experiment; never a Query planner."""
import argparse,json,time
from pathlib import Path
import numpy as np
def ramp(t,T=4.,v=.8):
    u=float(np.clip(t/T,0,1))
    return v*T*(2.5*u**4-3*u**5+u**6),v*(10*u**3-15*u**4+6*u**5),v/T*(30*u**2-60*u**3+30*u**4)
def reference(t):
    if t<4:return ramp(t)
    return 1.6+.8*(t-4),.8,0.
def braking(t,p0,v0,T=4.):
    p,v,a=ramp(t,T,v0)
    return p0+v0*min(t,T)-p,v0-v,-a
def main():
    import rospy,rosgraph
    from nav_msgs.msg import Odometry
    from quadrotor_msgs.msg import PositionCommand
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);a=p.parse_args()
    host=Path(a.host);rospy.init_node('low_speed_reference',disable_signals=True)
    master=rosgraph.Master('/low_speed_reference')
    sources=dict(master.getSystemState()[0]).get('/so3_control/pos_cmd',[])
    if sources:raise RuntimeError('PREEXISTING_CONTROL_PUBLISHER '+repr(sources))
    pub=rospy.Publisher('/so3_control/pos_cmd',PositionCommand,queue_size=1,tcp_nodelay=True)
    latest=[None]
    sub=rospy.Subscriber('/sim/odom',Odometry,lambda x:latest.__setitem__(0,x),queue_size=1)
    log=(host/'REFERENCE_TRACE.jsonl').open('x',buffering=1)
    time.sleep(.5);start=rospy.Time.now().to_sec()
    with (host/'REFERENCE_START.json').open('x') as f:json.dump(dict(stamp=start,wall_time_ns=time.time_ns(),mode='CONTROLLED_REFERENCE_SENSOR_INTEGRATION',query_control=False),f)
    brake_start=None;brake_p=brake_v=0.;last=0.
    try:
        rate=rospy.Rate(50)
        while not rospy.is_shutdown():
            now=rospy.Time.now().to_sec();t=now-start
            if (t>=30 or (host/'REFERENCE_STOP').exists()) and brake_start is None:
                brake_start=t;brake_p,brake_v,_=reference(t)
            if brake_start is None:x,v,acc=reference(t);phase='RAMP' if t<4 else 'CRUISE'
            else:x,v,acc=braking(t-brake_start,brake_p,brake_v);phase='BRAKE' if t-brake_start<4 else 'HOLD'
            msg=PositionCommand();msg.header.stamp=rospy.Time.now();msg.header.frame_id='world'
            msg.position.x=x;msg.position.z=2.;msg.velocity.x=v;msg.acceleration.x=acc
            # Native network_controller: READY uses acceleration-only branch.
            # EMPTY intentionally selects existing position/velocity feedback branch.
            msg.trajectory_flag=PositionCommand.TRAJECTORY_STATUS_EMPTY
            msg.yaw=0.;pub.publish(msg)
            if t-last>=.05:
                o=latest[0];actual=None
                if o is not None:
                    actual=dict(stamp=o.header.stamp.to_sec(),p=[o.pose.pose.position.x,o.pose.pose.position.y,o.pose.pose.position.z],
                        v=[o.twist.twist.linear.x,o.twist.twist.linear.y,o.twist.twist.linear.z])
                log.write(json.dumps(dict(stamp=now,t=t,phase=phase,desired_p=[x,0,2],desired_v=[v,0,0],desired_a=[acc,0,0],actual=actual))+'\n');last=t
            if brake_start is not None and t-brake_start>=6:break
            if t>38:raise RuntimeError('REFERENCE_WALL_LIMIT')
            rate.sleep()
    finally:
        sub.unregister();pub.unregister();log.close();rospy.signal_shutdown('bounded reference complete')
        with (host/'REFERENCE_EXIT.json').open('x') as f:json.dump(dict(stamp=rospy.Time.now().to_sec(),query_control=False),f)
if __name__=='__main__':main()
