"""Run the existing depth goal publisher using the bound forest10m scene."""
import argparse
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.sim_dev_oracle_v1.prepare_runtime import replace_once as patch

def config(run):
    return json.loads((Path(run).parent/'SCENE_IDENTITY.json').read_text())['scene']['forest10m']

def source(code):
    code=patch(code,'target=[p.x+3.,p.y,p.z]',"target=config(run)['goal']")
    code=patch(code,'anchor=[target[0]-3.,target[1],target[2]]',"anchor=config(run)['spawn']")
    code=patch(code,'done=plans>=20 and p.x-anchor[0]>=2.',"""distance=math.dist([p.x,p.y,p.z],target)
   vel=m.twist.twist.linear
   speed=math.sqrt(vel.x**2+vel.y**2+vel.z**2)
   progress=p.x-anchor[0]
   peak=max(locals().get('peak',0.),progress)
   done=h.get('mode')=='ARMED' and plans>=2 and distance<=.15 and speed<=.10""")
    return patch(code,'replans=plans,monotonic=',"replans=plans,armed=h.get('mode')=='ARMED',progress_m=progress,peak_progress_m=peak,goal_error_m=distance,milestones_m=[x for x in config(run)['milestones_m'] if peak>=x],monotonic=")

def main():
    import math
    from query_minco.integration.depth_minco_t2_v1 import ros_bridge
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--stage',required=True)
    args=parser.parse_args()
    namespace=dict(ros_bridge.__dict__,config=config,math=math)
    exec(compile(source(inspect.getsource(ros_bridge.reference_main)),__file__,'exec'),namespace)
    namespace['reference_main'](args)

if __name__=='__main__':main()
