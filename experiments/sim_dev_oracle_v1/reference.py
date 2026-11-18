"""Fixed sparse-simulation goal only; no GT, no candidate routes."""
from pathlib import Path
import sys,argparse,types,inspect
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--stage',default='SIM_DEV_ORACLE');a=p.parse_args()
 from query_minco.integration.depth_minco_t2_v1 import ros_bridge
 code=inspect.getsource(ros_bridge.reference_main)
 code=code.replace("target=[p.x+3.,p.y,p.z]","\n    if max(abs(x-y) for x,y in zip([p.x,p.y,p.z],[.25,-.25,2.]))>.005:raise ValueError('SIM_FIXED_SPAWN')\n    target=[3.25,-.25,2.]")
 code=code.replace('done=plans>=20 and p.x-anchor[0]>=2.','done=plans>=2 and p.x-anchor[0]>=.80')
 namespace=dict(ros_bridge.__dict__);exec(compile(code,str(Path(__file__)),'exec'),namespace);namespace['reference_main'](a)
if __name__=='__main__':main()
