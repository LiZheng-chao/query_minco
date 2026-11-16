"""Supplement inactive real jerk cases with explicit synthetic component finite differences."""
from pathlib import Path
import sys,math,json,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.training.peak_dynamics_proxy_v1.proxy import peak_proxy
O=Path.cwd()/'research_outputs/g4c_startup_and_dynamics_20261111_v1'
records=[]
for order,key,L in [(1,'speed',3.),(2,'acceleration',8.),(3,'jerk',100.)]:
 c=torch.zeros((1,1,3,3,6),dtype=torch.float64)
 for seg,ratio in enumerate([1.2,1.1,1.05]):c[0,0,seg,0,order]=ratio*L/math.factorial(order)
 c.requires_grad_();d=torch.full((1,1,3),2/3,dtype=torch.float64)
 f=peak_proxy(c,d)['terms']['F_peak_'+key].sum();g=torch.autograd.grad(f,c)[0]
 direction=torch.zeros_like(c);direction[0,0,0,0,order]=1
 eps=1e-5
 values=[float(peak_proxy(c.detach()+s*eps*direction,d)['terms']['F_peak_'+key]) for s in [-1,1]]
 fd=(values[1]-values[0])/(2*eps);ad=float((g*direction).sum());assert abs(fd-ad)<1e-7
 records.append(dict(component=key,fixture='SYNTHETIC_ONLY_NOT_DATA',value=float(f),epsilon=eps,autograd=ad,finite_difference=fd,status='PASS'))
with (O/'SYNTHETIC_COMPONENT_FD.json').open('x') as f:json.dump(records,f,indent=2)
print('THREE_COMPONENT_FD_PASS_NO_UPDATES')
