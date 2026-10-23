"""Offline FP64 reference linear system, same frozen C4 pattern; no P2A or solver edits."""
import torch
from query_minco._vendor.minco.minco import MincoS3NU
def solve64(head,tail,inner,durations):
 n=inner.shape[0];dt=durations.double();pattern=MincoS3NU._constraint_pattern(3)
 idx=torch.tensor([r*18+c for r,c,_,_,_ in pattern]);ti=torch.tensor([p*6+k for _,_,p,k,_ in pattern])
 constants=torch.tensor([v for *_,v in pattern],dtype=torch.float64)
 powers=dt[...,None]**torch.arange(6,dtype=torch.float64)
 flat=torch.zeros(n,324,dtype=torch.float64);flat[:,idx]=powers.reshape(n,-1)[:,ti]*constants
 blocks=[head.double()[None].expand(n,-1,-1)]
 for i in range(2):blocks += [inner.double()[:,i:i+1],torch.zeros(n,5,3,dtype=torch.float64)]
 blocks += [tail.double()[None].expand(n,-1,-1)]
 coeff=torch.linalg.solve(flat.reshape(n,18,18),torch.cat(blocks,dim=1))
 return coeff.reshape(n,3,6,3).permute(0,1,3,2).contiguous()
