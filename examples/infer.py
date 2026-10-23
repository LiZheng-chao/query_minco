"""Explicit checkpoint, synthetic observation; never selects or executes a trajectory."""
import argparse
import numpy as np
from query_minco.inference import load_planner
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);a=p.parse_args()
 depth=np.zeros((1,2,128,160),np.float32);depth[:,0]=.25;depth[:,1]=1
 state=np.array([[.8,0,0,0,0,0,2.8,0,0,0,0,-1]],np.float32)
 candidates=load_planner('query_minco_b2c1_c1_e1_v1',a.checkpoint).plan(depth,state)
 print({k:list(v.shape) for k,v in candidates.items()})
if __name__=='__main__':main()
