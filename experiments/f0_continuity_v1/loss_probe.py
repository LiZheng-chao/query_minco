import sys,json
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]));torch.set_num_threads(1)
from query_minco.integration.yopo_execution_v4.trajectory import build
from query_minco.training.peak_dynamics_proxy_v1.proxy import peak_proxy
p=build(np.zeros(3),np.zeros(3),np.zeros(3),[0,0,0,1],[.28,0,0],100.)
results=[]
for scale in [1.,20.]:
 c=torch.tensor(p.c[None,None]*scale,dtype=torch.float64,requires_grad=True);d=torch.tensor(p.d[None,None],dtype=torch.float64)
 out=peak_proxy(c,d);loss=out['F_peak'].sum();g=torch.autograd.grad(loss,c)[0];idx=int(g.abs().argmax());flat=c.detach().clone().reshape(-1);h=1e-5
 plus=flat.clone();minus=flat.clone();plus[idx]+=h;minus[idx]-=h
 fd=float((peak_proxy(plus.reshape_as(c),d)['F_peak'].sum()-peak_proxy(minus.reshape_as(c),d)['F_peak'].sum())/(2*h))
 results.append(dict(scale=scale,F_peak=float(loss),gradient_norm=float(g.norm()),chosen_gradient=float(g.reshape(-1)[idx]),finite_difference=fd,finite=bool(torch.isfinite(g).all()),scope='synthetic_coefficients_only_not_network_training'))
Path('research_outputs/f0_continuity_20261116_v1/PEAK_READONLY_PROBE.json').write_text(json.dumps(results,indent=2));print(results)
