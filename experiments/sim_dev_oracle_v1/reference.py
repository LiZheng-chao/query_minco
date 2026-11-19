"""SIM fixed-scene Goal; fresh stamp at publication, original request evidence."""
from pathlib import Path
import sys,argparse,inspect,json,math
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
def fixed_goal(run,position):
 scene=json.loads((Path(run).parent/'SCENE_IDENTITY.json').read_text())['scene']
 if (scene.get('seed'),scene.get('maze_type'),scene.get('tree_dist'),scene.get('spawn'),scene.get('goal'))!=(3,5,6,[.25,-.25,2.],[3.25,-.25,2.]):raise ValueError('SIM_FIXED_SCENE_GOAL')
 if len(position)!=3 or not all(math.isfinite(x) for x in position):raise ValueError('SIM_FIXED_SPAWN')
 if max(abs(x-y) for x,y in zip(position,scene['spawn']))>.005:raise ValueError('SIM_FIXED_SPAWN')
 return list(scene['goal'])
def source(code):
 from experiments.sim_dev_oracle_v1.prepare_runtime import replace_once
 code=replace_once(code,"p=m.pose.pose.position;target=[p.x+3.,p.y,p.z]","p=m.pose.pose.position;target=fixed_goal(run,[p.x,p.y,p.z])")
 code=replace_once(code,"pub.publish(String(data=json.dumps(dict(kind='forest_goal',frame='world',stamp=now,goal=target))))", "now=rospy.Time.now().to_sec()\n   envelope=dict(kind='forest_goal',frame='world',stamp=now,goal=target)\n   pub.publish(String(data=json.dumps(envelope)))\n   journal.append(dict(kind='GOAL_PUBLISHED',payload=envelope,published_ros=rospy.Time.now().to_sec(),monotonic=time.monotonic()))")
 return replace_once(code,'done=plans>=20 and p.x-anchor[0]>=2.','done=plans>=2 and p.x-anchor[0]>=.80')
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--stage',default='SIM_DEV_ORACLE');a=p.parse_args()
 from query_minco.integration.depth_minco_t2_v1 import ros_bridge
 code=source(inspect.getsource(ros_bridge.reference_main));namespace=dict(ros_bridge.__dict__,fixed_goal=fixed_goal)
 exec(compile(code,__file__,'exec'),namespace);namespace['reference_main'](a)
if __name__=='__main__':main()
