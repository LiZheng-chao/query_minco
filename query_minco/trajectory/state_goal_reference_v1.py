"""Geometry-free K1-compatible references for offline comparison only."""
import numpy as np
import torch
COMPONENT_ID="fixed_forward_and_state_goal_quintic_raw21_reference_v1"
def reference_raw(state,kind):
    if state.shape!=(1,12) or state.dtype!=torch.float32 or not torch.isfinite(state).all():raise ValueError("REFERENCE_STATE")
    s=state[0].detach().double().numpy();T=2.;times=np.array([T/3,2*T/3,T])
    goal=s[6:9];length=float(np.linalg.norm(goal))
    if length<=1e-8:raise ValueError("REFERENCE_GOAL")
    if kind=="fixed_forward":
        u=times/T;points=(10*u**3-15*u**4+6*u**5)[:,None]*np.array([2.4,0,0])
    elif kind=="state_goal_quintic":
        end=goal*min(1,2.4/length);v=s[:3];a=s[3:6]
        matrix=np.array([[T**3,T**4,T**5],[3*T*T,4*T**3,5*T**4],[6*T,12*T*T,20*T**3]])
        high=np.linalg.solve(matrix,np.stack([end-v*T-.5*a*T*T,-v-a*T,-a]))
        points=np.array([v*t+.5*a*t*t+sum(high[i]*t**(i+3) for i in range(3)) for t in times])
    else:raise ValueError("REFERENCE_KIND")
    low=np.array([0.,-4.,-2.]);high=np.array([10.,4.,2.])
    unit=2*(points-low)/(high-low)-1
    if (np.abs(unit)>=1).any():raise ValueError("REFERENCE_OUTSIDE_ORIGINAL_A_SUPPORT")
    raw=torch.zeros(1,1,21);raw[0,0,:9]=torch.from_numpy(np.arctanh(unit).reshape(-1).astype(np.float32))
    return raw
