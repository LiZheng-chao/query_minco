"""Training-only gradient norms, raw routes, finite checks and fixed-budget calibration."""
import numpy as np
import torch
from query_minco.training.io import require
from query_minco.training.pg1_v2_r12.diagnostics import flat
from .objective import KEYS,PILOT
BUDGETS=dict(mode=.5,bottleneck=.3,diversity=.05)

def audit(model,z):
    pp=list(model.parameters()); raw=z["raw"]
    terms=dict(imitation=z["base"]["base"]["terms"]["imitation"],base_safe=z["base"]["loss"],**z["terms"])
    vectors={}; rows={}
    for k,value in terms.items():
        v=flat(torch.autograd.grad(value,pp,retain_graph=True,allow_unused=True),pp)
        rg=torch.autograd.grad(value,raw,retain_graph=True,allow_unused=True)[0]
        cg,dg=torch.autograd.grad(value,(z["output"].trajectory.coeffs,z["output"].parameters.durations),retain_graph=True,allow_unused=True)
        require(bool(torch.isfinite(v).all()),"V2_PARAMETER_GRAD_"+k)
        if rg is not None:require(bool(torch.isfinite(rg).all()) and not torch.count_nonzero(rg[...,19:21]),"V2_SCORE_OR_NONFINITE_"+k)
        rows[k]=dict(value=float(value.detach()),parameter_norm=float(v.norm()),raw_norm=None if rg is None else float(rg.norm()),
            raw_query_norm=None if rg is None else rg.norm(dim=-1).tolist(),raw_time_norm=None if rg is None else rg[...,15:19].norm(dim=(0,1)).tolist(),
            coefficient_norm=None if cg is None else float(cg.norm()),duration_norm=None if dg is None else float(dg.norm()),
            strata={} if rg is None else {("matched" if m else "unmatched")+"_"+("safe" if s else "unsafe"):float(rg[(z["matched"]==m)&(z["labels"]["passed"]==s)].norm()) for m in (False,True) for s in (False,True)})
        vectors[k]=v
    total=flat(torch.autograd.grad(z["loss"],pp,retain_graph=True,allow_unused=True),pp)
    require(bool(torch.isfinite(total).all()),"V2_TOTAL_GRAD")
    direct=vectors["base_safe"]+sum(z["weights"][k]*vectors[k] for k in KEYS)
    require(torch.allclose(total,direct,atol=2e-5,rtol=2e-4),"V2_TOTAL_RECONSTRUCTION")
    im=vectors["imitation"]
    ratio=float((total-im).norm()/im.norm()) if float(im.norm()) else None
    names=list(vectors);stack=torch.stack([vectors[k].double() for k in names])
    gram=(stack@stack.T).tolist()
    return dict(components=rows,component_order=names,component_gram=gram,physical_imitation_ratio=ratio,weights=z["weights"],
                weighted_norms={k:z["weights"][k]*rows[k]["parameter_norm"] for k in KEYS}),vectors,total

def calibrate(records):
    require(bool(records) and all(r["components"]["imitation"]["parameter_norm"]>0 for r in records),"V2_CALIBRATION_IM")
    weights=dict(PILOT)
    for k,budget in BUDGETS.items():
        ratios=[r["components"][k]["parameter_norm"]/r["components"]["imitation"]["parameter_norm"] for r in records if r["components"][k]["parameter_norm"]>1e-12]
        weights[k]=float(budget/np.median(ratios)) if ratios else 0.
    upper=[sum(weights[k]*r["components"][k]["parameter_norm"]/r["components"]["imitation"]["parameter_norm"] for k in BUDGETS) for r in records]
    factor=min(1.,2./max(max(upper),1e-30))
    for k in BUDGETS:weights[k]*=factor
    ratios=[]
    for r in records:
        names=r["component_order"];w=np.array([1. if k=="base_safe" else -1. if k=="imitation" else weights[k] for k in names])
        g=np.asarray(r["component_gram"]);im=names.index("imitation")
        ratios.append(float(np.sqrt(max(0.,w@g@w)/g[im,im])))
    return dict(calibrated_physical_imitation_ratios=ratios,gradient_guard_passed=all(v<=10 for v in ratios),weights=weights,budgets=BUDGETS,scale=factor,uses_development=False,optimizer_steps=0,
                pilot_constants=PILOT,formula="median positive norm ratio; mode+bneck+div triangle upper bound <=2 on calibration batches; pilot three fixed")
