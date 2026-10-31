"""Frozen physical-time nearest-AABB proxy; not a safety certificate."""
import hashlib,json,math
import numpy as np
import torch
from query_minco.training.io import require
from query_minco.evaluation.validation import roots,evaluate
from query_minco.evaluation.signed_geometry import numpy_box_sdf
ID='pg3_next_frozen_clearance_proxy_v1'
DENSE_PER_PIECE=4097
class Parameters:
    def __init__(self,model):
        self.named=sorted([(n,p) for n,p in model.named_parameters() if p.requires_grad],key=lambda x:x[0]);self.parts=[]
        for n,p in self.named:
            mask=torch.ones_like(p,dtype=torch.bool)
            if n.startswith('trajectory_head.') and p.shape[0]==21:mask[19:21]=False
            self.parts.append(mask)
        self.order=[dict(name=n,shape=list(p.shape),included=int(m.sum())) for (n,p),m in zip(self.named,self.parts)]
    def values(self):return torch.cat([p.detach()[m].double() for (n,p),m in zip(self.named,self.parts)])
    def gradient(self,value,retain_graph=True):
        gg=torch.autograd.grad(value,[p for n,p in self.named],retain_graph=retain_graph,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g)[m].double() for (n,p),m,g in zip(self.named,self.parts,gg)])
    def set_disposable(self,values):
        offset=0
        with torch.no_grad():
            for (n,p),mask in zip(self.named,self.parts):
                k=int(mask.sum());p[mask]=values[offset:offset+k].to(p);offset+=k
        require(offset==len(values),'NEXT_PARAMETER_PACK')

def times_and_active(c,d,boxes):
    c=np.asarray(c,dtype=np.float64);d=np.asarray(d,dtype=np.float64)
    require(np.isfinite(c).all() and np.isfinite(d).all() and (d>0).all() and len(boxes)>0,'NEXT_PROXY_INPUT')
    times=[];start=0.;critical_count=0
    for cc,T in zip(c,d):
        local=list(np.linspace(0,float(T),DENSE_PER_PIECE));critical=[]
        for box in boxes:
            cuts=[0.,float(T)]
            for axis in range(3):
                for bound in (box['min'][axis],box['max'][axis]):
                    poly=cc[axis].copy();poly[0]-=bound;cuts+=roots(poly,float(T))
            cuts=sorted(set(cuts))
            for lo,hi in zip(cuts[:-1],cuts[1:]):
                point=evaluate(cc,(lo+hi)/2);sq=np.zeros(1)
                for axis in range(3):
                    side=box['min'][axis] if point[axis]<box['min'][axis] else box['max'][axis] if point[axis]>box['max'][axis] else None
                    if side is not None:
                        poly=cc[axis].copy();poly[0]-=side;sq=np.polynomial.polynomial.polyadd(sq,np.polynomial.polynomial.polymul(poly,poly))
                critical += [x for x in roots(np.polynomial.polynomial.polyder(sq),float(T)) if lo<x<hi]
            critical+=cuts
        critical_count+=len(critical);local+=critical;times.extend(start+np.array(local));start+=float(T)
    times=np.unique(np.array(times));ends=np.cumsum(d);pieces=np.minimum(np.searchsorted(ends,times,side='right'),2);local=times-np.r_[0,ends[:-1]][pieces]
    pp=np.empty((len(times),3))
    for k in range(3):
        mask=pieces==k;pp[mask]=evaluate(c[k],local[mask])
    sdfs=np.stack([numpy_box_sdf(pp,b) for b in boxes],axis=1);minimum=float(sdfs.min());ti,bi=np.unravel_index(sdfs.argmin(),sdfs.shape)
    v=(pp[ti]-(np.asarray(boxes[bi]['min'])+boxes[bi]['max'])/2);q=np.abs(v)-(np.asarray(boxes[bi]['max'])-boxes[bi]['min'])/2
    tied=np.argwhere(np.isclose(sdfs,minimum,atol=1e-10,rtol=0))
    meta=dict(physical_times_count=len(times),dense_per_piece=DENSE_PER_PIECE,critical_count=critical_count,active_minimum_index=int(ti),active_box_index=int(bi),active_box_name=boxes[bi]['name'],active_time_s=float(times[ti]),active_piece=int(pieces[ti]),minimum_sdf_m=minimum,h_m=minimum-.15,tied_minima_count=len(tied),tie_rule='time-major then original box order; abs(0) uses zero derivative; positive protected SDF uses exterior norm',active_face_tie_count=int(np.isclose(q,q.max(),atol=1e-10).sum()),maximum_time_gap_s=float(np.diff(times).max()),times_sha256=hashlib.sha256(times.tobytes()).hexdigest(),certificate=False)
    return float(times[ti]),int(pieces[ti]),int(bi),meta

def proxy(c,d,boxes):
    t,k,bi,info=times_and_active(c.detach().numpy(),d.detach().numpy(),boxes)
    # Times are fixed in physical seconds at the BEFORE state, including geometry-critical nodes.
    local=c.new_tensor(t)-d[:k].sum();point=(c[k]*local**torch.arange(6,dtype=c.dtype)).sum(-1)
    box=boxes[bi];center=point.new_tensor((np.array(box['min'])+box['max'])/2);half=point.new_tensor((np.array(box['max'])-box['min'])/2)
    q=(point-center).abs()-half
    require(float(q.detach().clamp_min(0).norm())>0,'NEXT_NONSMOOTH_INTERIOR_PROTECTED_PROXY')
    h=q.clamp_min(0).norm()-.15
    return h,info

def canonical_gradient_witness(record):
    r=record['audit'];return dict(component_order=r['component_order'],gram=r['gram'],gradient_norm_before_clip=record['gradient_norm_before_clip'],ratios=r['ratios'],loss=record['loss'])
def semantic_sha(value):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):h.update(str((tuple(x.shape),x.dtype)).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()
