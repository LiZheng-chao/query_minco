"""Final source syntax/CLI and proxy replay;no ROS launch or optimization."""
from pathlib import Path
import sys,json,ast,subprocess,torch,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.training.peak_dynamics_proxy_v1.proxy import peak_proxy
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path.cwd();O=R/'research_outputs/g4c_startup_and_dynamics_20261111_v1'
dirs=['query_minco/integration/yopo_startup_v2','query_minco/training/peak_dynamics_proxy_v1','ros/g4c_startup_v2','experiments/g4c_startup_and_dynamics_v1','tests/g4c_startup_and_dynamics_v1']
parsed=[]
for d in dirs:
 for p in (R/d).rglob('*.py'):
  ast.parse(p.read_text(),filename=str(p));compile(p.read_text(),str(p),'exec');parsed.append(str(p))
summary=json.loads((O/'DYNAMICS_ZERO_UPDATE_SUMMARY.json').read_text());arrays=np.load(O/'DYNAMICS_FROZEN_ARRAYS.npz');checks=[]
for r in summary['all_records']:
 key='%03d__%s'%(r['step'],r['sample_id'])
 # Recover exactly stored FP32 coefficients/durations;the independent NumPy audit stored them promoted.
 c=torch.from_numpy(arrays[key+'_coeff'].astype(np.float32))[None,None];d=torch.from_numpy(arrays[key+'_durations'].astype(np.float32))[None,None]
 p=peak_proxy(c,d)
 checks.append(all(float(v)==r['upper_ratios'][k] for k,v in p['upper_ratios'].items()))
assert all(checks)
c=torch.zeros((1,1,3,3,6),dtype=torch.float64);c[:,:,:,0,1]=3.2;c.requires_grad_();d=torch.full((1,1,3),2/3,dtype=torch.float64)
f=peak_proxy(c,d)['F_peak'].sum();g=torch.autograd.grad(f,c)[0]
e=1e-5;v=torch.zeros_like(c);v[0,0,0,0,1]=1
fp=float(peak_proxy(c.detach()+e*v,d)['F_peak']);fm=float(peak_proxy(c.detach()-e*v,d)['F_peak']);f0=float(f)
tie=dict(case='three identical constant-speed pieces,tied maxima',autograd=float((g*v).sum()),forward_difference=(fp-f0)/e,backward_difference=(f0-fm)/e,central_difference=(fp-fm)/(2*e),
 interpretation='EXPECTED_NONSMOOTH_MAX_TIE_SUBGRADIENT;central FD need not equal autograd;not a smooth-point pass')
assert tie['backward_difference']-1e-10<=tie['autograd']<=tie['forward_difference']+1e-10
write(O/'NONSMOOTH_TIE_DIAGNOSIS.json',tie)
cmd=[sys.executable,'-B','experiments/g4c_startup_and_dynamics_v1/manual_startup.py','--help']
p=subprocess.run(cmd,capture_output=True,text=True);write(O/'MANUAL_CLI_HELP.json',dict(argv=cmd,stdout=p.stdout,stderr=p.stderr,exit_code=p.returncode,physics_execution=False))
assert p.returncode==0
build=json.loads((O/'startup_build_v2/BUILD_IDENTITY.json').read_text())
assert build['header_sha256']==sha(R/'query_minco/integration/yopo_startup_v2/plant_gate.hpp')
assert build['sha256']==sha(build['binary']) and build['executed'] is False
write(O/'FINAL_STATIC_AND_REPLAY.json',dict(status='PASS',syntax_files=parsed,proxy_exact_frozen_replays=len(checks),simulator='COMPILED_NOT_EXECUTED',ros_nodes='NOT_RUN',real_physics='NOT_RUN',engineering_steps_not_repeated=True))
print('FINAL_CHECKS_PASS',len(parsed),'SYNTAX',len(checks),'EXACT_PROXY_REPLAYS')
