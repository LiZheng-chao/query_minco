"""Manual read-only ROS shadow entry. Only MarkerArray and String publishers."""
import argparse
import json
import os
from pathlib import Path
import queue
import time
import traceback
import numpy as np
from query_minco.inference.shadow import (ROOT, FrozenPair, camera_check, decode_image,
    make_state, point_depth_label, require, rotation)

TOPICS = {'diagnostics': '/query_shadow/diagnostics', 'candidates': '/query_shadow/candidates'}
OUTROOT = ROOT/'research_outputs/rsps_forest_shadow_v1'

def dump(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)

def topology_check(publishers):
    own = {t for t, nodes in publishers if '/query_minco_shadow' in nodes}
    require(own <= set(TOPICS.values()) | {'/rosout'}, 'UNEXPECTED_SHADOW_PUBLISHER')
    control = {t: nodes for t, nodes in publishers if t in ('/so3_control/pos_cmd', '/so3_cmd', '/motors', '/motor_cmd')}
    require(control.get('/so3_control/pos_cmd') == ['/yopo_net'], 'YOPO_NOT_UNIQUE_POSITION_SOURCE')
    require(dict(publishers).get('/sim/odom') == ['/quadrotor_simulator_so3'], 'NATIVE_SIM_ODOM_REQUIRED')
    return dict(shadow_publishers=sorted(own), control_publishers=control, odom_publishers=dict(publishers)['/sim/odom'])

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--binding-sha', required=True)
    p.add_argument('--seconds', type=int, default=120)
    p.add_argument('--max-frames', type=int, default=300)
    args = p.parse_args()
    require(1 <= args.seconds <= 120 and 1 <= args.max_frames <= 300, 'RUN_LIMIT')
    run = Path(args.run).resolve()
    require(run.parent == OUTROOT and run.is_dir() and not (run/'START.json').exists(), 'EXCLUSIVE_RUN_PATH')
    import rospy
    import message_filters
    import rosgraph
    from sensor_msgs.msg import Image, CameraInfo
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    pair = FrozenPair(args.binding_sha)
    rospy.init_node('query_minco_shadow', anonymous=False)
    require(rospy.get_name() == '/query_minco_shadow', 'NODE_NAMESPACE')
    pub_diag = rospy.Publisher(TOPICS['diagnostics'], String, queue_size=1)
    pub_viz = rospy.Publisher(TOPICS['candidates'], MarkerArray, queue_size=1)
    master = rosgraph.Master('/query_minco_shadow')
    frames = queue.Queue(maxsize=1)
    counters = dict(synchronized=0, dropped=0, accepted=0, rejected=0, watchdog=0)
    previous = global_goal = last_stamp = child_frame = None
    start = last_received = last_watchdog = time.monotonic()
    fatal = None
    events = (run/'events.jsonl').open('x', buffering=1)
    dump(run/'START.json', dict(mode='SHADOW_ONLY', models=pair.bindings, pid=os.getpid(),
         argv=vars(args), wall_time_ns=time.time_ns(), training=False, control=False))

    def event(record):
        record.update(wall_time_ns=time.time_ns(), elapsed_s=time.monotonic()-start,
                      online_safe_status='UNKNOWN_NO_CONTINUOUS_BODY_OBSERVABILITY', selected_query=None)
        events.write(json.dumps(record, allow_nan=False)+'\n')
        pub_diag.publish(String(data=json.dumps(record, allow_nan=False)))

    def enqueue(depth, camera, odom):
        counters['synchronized'] += 1
        item = (depth, camera, odom, time.monotonic())
        try:
            frames.put_nowait(item)
        except queue.Full:
            try:
                frames.get_nowait()
            except queue.Empty:
                pass
            counters['dropped'] += 1
            try:
                frames.put_nowait(item)
            except queue.Full:
                counters['dropped'] += 1

    subscribers = [message_filters.Subscriber('/query_shadow/depth', Image, queue_size=2),
                   message_filters.Subscriber('/query_shadow/camera_info', CameraInfo, queue_size=2),
                   message_filters.Subscriber('/sim/odom', Odometry, queue_size=200)]
    sync = message_filters.TimeSynchronizer(subscribers, 200)
    sync.registerCallback(enqueue)
    try:
        topology_check(master.getSystemState()[0])
        while not rospy.is_shutdown() and time.monotonic()-start < args.seconds and counters['accepted'] < args.max_frames:
            if time.monotonic()-last_watchdog >= 1:
                event(dict(type='TOPOLOGY', **topology_check(master.getSystemState()[0])))
                if time.monotonic()-last_received > 1:
                    counters['watchdog'] += 1
                    event(dict(type='WATCHDOG_NO_FRESH_SYNC', age_wall_s=time.monotonic()-last_received))
                last_watchdog = time.monotonic()
            try:
                depth, camera, odom, received = frames.get(timeout=.1)
            except queue.Empty:
                continue
            stamp = depth.header.stamp.to_sec()
            stamp_ns = depth.header.stamp.to_nsec()
            attempt = dict(depth_frame=depth.header.frame_id, odom_frame=odom.header.frame_id)
            try:
                require(stamp_ns == camera.header.stamp.to_nsec() == odom.header.stamp.to_nsec(), 'STAMP_MISMATCH')
                age = rospy.Time.now().to_sec()-stamp
                require(0 <= age <= .25 and time.monotonic()-received <= .25, 'STALE_OR_FUTURE_FRAME')
                require(last_stamp is None or stamp_ns > last_stamp, 'NONMONOTONIC_STAMP')
                last_stamp = stamp_ns
                last_received = time.monotonic()
                require(depth.header.frame_id == camera.header.frame_id == 'camera_link' and odom.header.frame_id == 'world', 'FRAME_CHANGED')
                require(camera.binning_x in (0, 1) and camera.binning_y in (0, 1) and camera.roi.x_offset == camera.roi.y_offset == camera.roi.height == camera.roi.width == 0, 'CAMERA_ROI_BINNING')
                camera_check(camera.width, camera.height, camera.K, camera.D, camera.R, camera.P)
                encoded = decode_image(depth.data, depth.encoding, depth.width, depth.height, depth.step, depth.is_bigendian)
                pose = odom.pose.pose
                pos = np.array([pose.position.x, pose.position.y, pose.position.z])
                q = pose.orientation
                R = rotation([q.x, q.y, q.z, q.w])
                v = odom.twist.twist.linear
                velocity = np.array([v.x, v.y, v.z])  # Verified native simulator WORLD velocity.
                require(np.isfinite(pos).all() and np.isfinite(velocity).all(), 'ODOM_NONFINITE')
                if child_frame is None:
                    child_frame = odom.child_frame_id
                require(odom.child_frame_id == child_frame, 'ODOM_CHILD_FRAME_CHANGED')
                if global_goal is None:
                    origin = np.array([rospy.get_param('/quadrotor_simulator_so3/simulator/init_state_'+axis) for axis in 'xyz'],dtype=float)
                    require(np.isfinite(origin).all(), 'SIM_INITIAL_ORIGIN')
                    global_goal = origin+np.array([15.,0,0])
                    dump(run/'GLOBAL_GOAL.json', dict(origin_W=origin.tolist(), goal_W=global_goal.tolist(), first_observation_W=pos.tolist(), stamp=stamp, stamp_ns=stamp_ns))
                attempt.update(position_W=pos.tolist(), velocity_W=velocity.tolist(), global_goal_W=global_goal.tolist(), global_distance_m=float(np.linalg.norm(global_goal-pos)), q_xyzw=[q.x,q.y,q.z,q.w])
                prev = previous
                previous = (stamp, pos.copy(), velocity.copy())
                require(prev is not None, 'ACCELERATION_WARMUP')
                dt = stamp-prev[0]
                require(.005 <= dt <= .3, 'ACCELERATION_DT_OR_RESET')
                require(np.linalg.norm(pos-prev[1]) <= max(1., 2*np.linalg.norm(velocity)*dt), 'ODOM_POSITION_JUMP')
                acceleration = (velocity-prev[2])/dt
                state, goal = make_state(pos, R, velocity, acceleration, global_goal)
                goal['depth_label'] = point_depth_label(encoded, state[6:9])
                t0 = time.monotonic()
                outputs = pair.forward(encoded, state)
                generation_s = time.monotonic()-t0
                require(0 <= rospy.Time.now().to_sec()-stamp <= .25, 'INFERENCE_RESULT_STALE')
                index = counters['accepted']
                arrays = dict(depth_input=encoded, state=state, position_W=pos, R_WB0=R,
                              global_goal_W=global_goal, stamp=np.array(stamp), stamp_ns=np.array(stamp_ns,dtype=np.int64), acceleration_dt=np.array(dt))
                viz = MarkerArray()
                diagnostics = {}
                for mi, (label, out) in enumerate(outputs.items()):
                    arrays.update({label+'_'+k: v for k, v in out.items()})
                    diagnostics[label] = dict(horizon_s=out['total_time'].reshape(-1).tolist(),
                        head_pva_residual=float(np.max(np.abs(out['head_pva']-np.stack([np.zeros(3), state[:3], state[3:6]])))),
                        endpoint_ray_labels=[point_depth_label(encoded, x) for x in out['uniform_p'][:, -1]],
                        endpoint_goal_error_m=np.linalg.norm(out['uniform_p'][:,-1]-state[6:9],axis=-1).tolist(),
                        terminal_speed_mps=np.linalg.norm(out['uniform_v'][:,-1],axis=-1).tolist(),
                        sampled_speed_max_mps=np.linalg.norm(out['uniform_v'],axis=-1).max(-1).tolist(),
                        sampled_acceleration_max_mps2=np.linalg.norm(out['uniform_a'],axis=-1).max(-1).tolist(),
                        sampled_jerk_max_mps3=np.linalg.norm(out['uniform_jerk'],axis=-1).max(-1).tolist(),
                        full_body_swept_status=['UNKNOWN']*6, selector='N/A_NOT_CALIBRATED')
                    for qi, points in enumerate(out['uniform_p']):
                        m = Marker()
                        m.header.stamp = depth.header.stamp
                        m.header.frame_id = 'world'
                        m.ns, m.id = label, qi
                        m.type, m.action = Marker.LINE_STRIP, Marker.ADD
                        m.pose.orientation.w = 1
                        m.scale.x = .025
                        m.color.r, m.color.g, m.color.b, m.color.a = (1., .5, .1, .7) if mi else (.1, .6, 1., .7)
                        m.lifetime = rospy.Duration(.3)
                        m.points = [Point(*map(float, pos+R@x)) for x in points]
                        viz.markers.append(m)
                with (run/('frame_%06d.npz' % index)).open('xb') as f:
                    np.savez_compressed(f, **arrays)
                event(dict(type='FRAME', index=index, stamp=stamp, stamp_ns=stamp_ns, odom_child_frame=odom.child_frame_id,
                           input_age_s=age, output_age_s=rospy.Time.now().to_sec()-stamp,
                           generation_s=generation_s, acceleration_source='past_synced_world_velocity_difference',
                           acceleration_dt_s=dt, global_goal_W=global_goal.tolist(), state=state.tolist(),
                           goal=goal, diagnostics=diagnostics))
                pub_viz.publish(viz)
                counters['accepted'] += 1
            except ValueError as exc:
                counters['rejected'] += 1
                event(dict(type='REJECT', stamp=stamp, stamp_ns=stamp_ns, reason=str(exc), attempt=attempt))
    except Exception:
        fatal = traceback.format_exc()
        event(dict(type='FATAL', traceback=fatal))
    finally:
        for s in subscribers:
            s.sub.unregister()
        event(dict(type='END', counters=counters))
        events.close()
        dump(run/'RESULT.json', dict(status='RUNTIME_COMPLETE' if not fatal else 'FAILED', counters=counters,
             fatal=fatal, elapsed_s=time.monotonic()-start, control=False, F3='BLOCKED'))
        if not rospy.is_shutdown():
            rospy.signal_shutdown('bounded shadow finished')
    require(fatal is None, 'RUNTIME_FATAL')

if __name__ == '__main__':
    main()
