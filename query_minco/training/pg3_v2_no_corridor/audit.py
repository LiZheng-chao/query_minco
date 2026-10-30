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


# V2.1 reports use original PG2 components, never an opaque merged baseline.
from .objective import KEYS_V21

def audit_v21(model,z,strata_parameters=True):
    named=list(model.named_parameters());pp=[p for _,p in named];raw=z["raw"];pg2=z["pg2"]
    require(all(p.requires_grad and p.dtype==raw.dtype for p in pp),"V21_PARAMETER_ORDER_DTYPE")
    import inspect
    group_files={}
    for name,p in named:
        group=name.split(".")[0];module=model.get_submodule(group) if "." in name else model
        filename=inspect.getsourcefile(type(module))
        group_files[group]=filename or type(module).__module__
    terms={"pg2/"+k:v for k,v in pg2["terms"].items()}
    terms.update(z["terms"]);terms.update({"geom/"+k:v for k,v in z["branch_terms"].items()})
    terms.update(PG2_total=pg2["loss"],base_total=z["base_loss"])
    vectors={};records={}
    for name,value in terms.items():
        v=flat(torch.autograd.grad(value,pp,retain_graph=True,allow_unused=True),pp)
        rg=torch.autograd.grad(value,raw,retain_graph=True,allow_unused=True)[0]
        gc,gd=torch.autograd.grad(value,(z["output"].trajectory.coeffs,z["output"].parameters.durations),retain_graph=True,allow_unused=True)
        require(v.numel()==sum(p.numel() for p in pp) and v.dtype==raw.dtype and bool(torch.isfinite(v).all()),"V21_GRAD_FINITE_"+name)
        if rg is not None:require(bool(torch.isfinite(rg).all()) and not torch.count_nonzero(rg[...,19:21]),"V21_SCORE_GRAD_"+name)
        groups={};offset=0
        for param_name,p in named:
            part=v[offset:offset+p.numel()];offset+=p.numel();group=param_name.split(".")[0]
            groups[group]=groups.get(group,0.)+float(part.double().square().sum())
        strata={}
        for matched in (False,True):
            for safe in (False,True):
                mask=(z["matched"]==matched)&(z["labels"]["passed"]==safe)
                key=("matched" if matched else "unmatched")+"_"+("PASS" if safe else "FAIL")
                pv=None
                if rg is not None and strata_parameters:
                    pv=flat(torch.autograd.grad(raw,pp,grad_outputs=rg*mask[...,None],retain_graph=True,allow_unused=True),pp)
                strata[key]=dict(count=int(mask.sum()),raw_norm=None if rg is None else float(rg[mask].norm()),parameter_norm=None if pv is None else float(pv.norm()))
        weight=z["weights"].get(name,0.)
        old_weight=None
        if name.startswith("pg2/"):
            key=name.split("/",1)[1];old_weight=pg2["weights"][key]
            weight=0. if key=="braking" and z["stop_mode"]=="replace_full_stop" else old_weight
        records[name]=dict(value=float(value.detach()),parameter_norm=float(v.double().norm()),effective_weight=weight,
            weighted_norm=abs(weight)*float(v.double().norm()),pg2_weight=old_weight,
            raw_norm=None if rg is None else float(rg.norm()),raw_per_query=None if rg is None else rg.norm(dim=-1).tolist(),
            coefficient_norm=None if gc is None else float(gc.norm()),duration_norm=None if gd is None else float(gd.norm()),
            parameter_groups={k:float(np.sqrt(v)) for k,v in groups.items()},parameter_group_source_files=group_files,strata=strata)
        vectors[name]=v
    require(torch.allclose(z["terms"]["geom"],sum(z["branch_terms"].values()),atol=1e-7,rtol=1e-6),"V21_GEOM_BRANCH_VALUE")
    require(torch.allclose(vectors["geom"],sum(vectors["geom/"+k] for k in z["branch_terms"]),atol=3e-5,rtol=3e-4),"V21_GEOM_BRANCH_GRAD")
    total=flat(torch.autograd.grad(z["loss"],pp,retain_graph=True,allow_unused=True),pp)
    gpg2=vectors["PG2_total"];im=vectors["pg2/imitation"]
    require(float(im.double().norm())>0,"V21_ZERO_IMITATION_GRADIENT")
    brake=pg2["weights"]["braking"]*vectors["pg2/braking"]
    base=gpg2 if z["stop_mode"]=="legacy" else gpg2-brake
    reconstructed=base+sum(z["weights"][k]*vectors[k] for k in KEYS_V21)
    require(torch.allclose(total,reconstructed,atol=3e-5,rtol=3e-4),"V21_TOTAL_GRAD_RECONSTRUCTION")
    require(torch.allclose(vectors["base_total"],base,atol=3e-5,rtol=3e-4),"V21_BASE_GRAD_RECONSTRUCTION")
    pg2_expanded=sum(pg2["weights"][k]*vectors["pg2/"+k] for k in pg2["terms"])
    require(torch.allclose(gpg2,pg2_expanded,atol=3e-5,rtol=3e-4),"V21_PG2_GRAD_RECONSTRUCTION")
    expanded=pg2["loss"]-(pg2["weights"]["braking"]*pg2["terms"]["braking"] if z["stop_mode"]=="replace_full_stop" else 0.)+sum(z["weights"][k]*z["terms"][k] for k in KEYS_V21)
    require(torch.allclose(z["loss"],expanded,atol=1e-7,rtol=1e-6),"V21_LOSS_RECONSTRUCTION")
    from query_minco.training.pg1_v2_r12.diagnostics import cosine
    for k,v in vectors.items():records[k].update(cosine_imitation=cosine(v.double(),im.double()),cosine_PG2=cosine(v.double(),gpg2.double()))
    denom=im.double().norm()
    norm=lambda x:float(x.double().norm()/denom)
    ratios=dict(R_PG2=norm(gpg2-im),R_base=norm(base-im),R_D=norm(total-im),R_increment=norm(total-gpg2))
    names=list(vectors);stack=torch.stack([vectors[k].double() for k in names]);gram=stack@stack.T
    lengths=gram.diag().clamp_min(0).sqrt();den=lengths[:,None]*lengths[None,:]
    cos=torch.where(den>0,gram/den.clamp_min(1e-300),torch.zeros_like(gram))
    cosine_rows=[[float(cos[i,j]) if float(den[i,j])>0 else None for j in range(len(names))] for i in range(len(names))]
    return dict(components=records,component_order=names,gram=gram.tolist(),cosine=cosine_rows,
        conflicts=[dict(a=names[i],b=names[j],cosine=float(cos[i,j])) for i in range(len(names)) for j in range(i) if float(den[i,j])>0 and float(cos[i,j])<0],
        ratios=ratios,weights=dict(z["weights"]),stop_mode=z["stop_mode"],braking_weight=float(pg2["weights"]["braking"]),
        gradient_reconstruction_max_abs=float((total-reconstructed).abs().max()),loss_reconstruction_abs=float((z["loss"]-expanded).detach().abs()),
        geometry=z.get("geometry",[]),diagnostics=z.get("diagnostics",{}),
        parameter_order=[dict(name=n,shape=list(p.shape),dtype=str(p.dtype),module_file=type(model.get_submodule(n.rsplit(".",1)[0]) if "." in n else model).__module__) for n,p in named]),vectors,total


def ratios_from_gram_v21(record,weights):
    names=record["component_order"];g=np.asarray(record["gram"],dtype=np.float64)
    unit=lambda name:np.array([float(k==name) for k in names])
    im=unit("pg2/imitation");pg2=unit("PG2_total")
    base=pg2-record["braking_weight"]*unit("pg2/braking") if record["stop_mode"]=="replace_full_stop" else pg2
    total=base+sum(weights[k]*unit(k) for k in KEYS_V21)
    denominator=float(im@g@im);require(denominator>0,"V21_ZERO_IMITATION_GRAM")
    norm=lambda v:float(np.sqrt(max(0.,float(v@g@v))/denominator))
    return dict(R_PG2=norm(pg2-im),R_base=norm(base-im),R_D=norm(total-im),R_increment=norm(total-pg2))


def calibrate_v21(records,probes,config):
    require(len(records)==len(probes)==8,"V21_CALIBRATION_FIXED_EIGHT")
    budgets=config["budgets"];weights={};states={};blockers=[]
    for k in KEYS_V21:
        source=probes if k=="keep" else records
        ratios=[r["components"][k]["parameter_norm"]/r["components"]["pg2/imitation"]["parameter_norm"] for r in source if r["components"][k]["parameter_norm"]>1e-12]
        if not ratios:
            weights[k]=0.;states[k]=("INACTIVE_NO_PAIR" if not any(r.get("diagnostics",{}).get("div") for r in records) else "INACTIVE_ZERO_GRADIENT_WITH_PAIRS") if k=="div" else "KEEP_NOT_CALIBRATABLE" if k=="keep" else "NO_VALID_DATA"
            if k!="div":blockers.append(states[k]+":"+k)
        else:
            # A conservative active-probe maximum keeps every measured Keep budget <= .05.
            reference=max(ratios) if k=="keep" else float(np.median(ratios))
            weights[k]=float(budgets[k]/reference);states[k]="PROBE_ONLY_CALIBRATED" if k=="keep" else "CALIBRATED_TRAIN_ONLY"
    cap=config["increment_limit"];factor=1.
    for r in records+probes:
        im=r["components"]["pg2/imitation"]["parameter_norm"]
        removed=r["braking_weight"]*r["components"]["pg2/braking"]["parameter_norm"]
        new=sum(weights[k]*r["components"][k]["parameter_norm"] for k in KEYS_V21)
        if removed/im>=cap:blockers.append("REMOVED_BRAKING_EXCEEDS_INCREMENT_BUDGET")
        elif new>0:factor=min(factor,(cap*im-removed)/new)
    for k in weights:weights[k]*=factor
    joint=[ratios_from_gram_v21(r,weights) for r in records]
    probe_joint=[ratios_from_gram_v21(r,weights) for r in probes]
    for k in ("geom","stop"):
        median=float(np.median([weights[k]*r["components"][k]["parameter_norm"]/r["components"]["pg2/imitation"]["parameter_norm"] for r in records]))
        if weights[k]<=config["minimum_lambda"] or median<config["minimum_effective_ratio"]:states[k]="EFFECTIVELY_INACTIVE";blockers.append("EFFECTIVELY_INACTIVE:"+k)
    if any(r["R_base"]>10 for r in joint):blockers.append("BASELINE_GRADIENT_ALREADY_EXCEEDS_10")
    if any(r["R_D"]>10 for r in joint):blockers.append("FAILED_GRADIENT_GUARD")
    if any(r["R_increment"]>cap+1e-6 for r in joint+probe_joint):blockers.append("FAILED_INCREMENT_BUDGET")
    return dict(weights=weights,budgets=budgets,states=states,common_scale=factor,joint_ratios=joint,probe_joint_ratios=probe_joint,
        blockers=sorted(set(blockers)),passed=not blockers,uses_development=False,optimizer_steps=0,
        calibration_batches=8,keep_probe="PROBE_ONLY",method="median positive train norms; Keep=max positive shadow norm ratio; common triangle-budget scale; joint Gram verification")


def make_keep_probe_v21(initial_model,probe_config):
    """Deterministic clone only; no optimizer, source mutation or committed exposure."""
    import copy
    from query_minco.inference.identity import state_sha
    from query_minco.training.io import digest
    before=state_sha(initial_model);clone=copy.deepcopy(initial_model).train()
    biases=[p for n,p in clone.named_parameters() if n.startswith("trajectory_head.") and p.shape==(21,)]
    require(len(biases)==1,"V21_PROBE_HEAD_BIAS")
    with torch.no_grad():
        for index in probe_config["bias_indices"]:
            require(type(index) is int and 0<=index<19,"V21_PROBE_SCORE_FORBIDDEN")
            biases[0][index]+=probe_config["bias_offset"]
    require(state_sha(initial_model)==before and state_sha(clone)!=before,"V21_PROBE_ISOLATION")
    return clone,dict(status="PROBE_ONLY",initial_sha256=before,probe_sha256=state_sha(clone),
                      config_sha256=digest(probe_config),initial_unchanged=True,optimizer_created=False)


def step_guard_v21(record,config,*,streak=0,keep_previously_active=False):
    """Preclip guard for any future real step; diagnostic shadow uses it too."""
    import math
    require(set(KEYS_V21)<=set(record["components"]),"V21_INCOMPLETE_STEP_AUDIT")
    require(all(math.isfinite(v) for v in record["ratios"].values()),"V21_NONFINITE_STEP_RATIO")
    require(record["ratios"]["R_base"]<=10,"V21_BASE_ALREADY_EXCEEDS_GUARD")
    streak=streak+1 if record["ratios"]["R_D"]>10 else 0
    require(streak<2,"V21_TWO_STEP_GRADIENT_RATIO_GT10")
    keep=record["components"]["keep"];im=record["components"]["pg2/imitation"]["parameter_norm"]
    require(im>0,"V21_ZERO_IMITATION_STEP")
    active=keep["parameter_norm"]>0
    ratio=record["weights"]["keep"]*keep["parameter_norm"]/im
    if active:require(ratio<=config["keep_active_max_ratio"]+1e-8,"V21_KEEP_ACTIVE_BUDGET_EXCEEDED")
    return dict(ratio_alarm_streak=streak,first_keep_activation=active and not keep_previously_active,
        keep_ever_active=active or keep_previously_active,weighted_keep_imitation_ratio=ratio,
        increment_budget_passed=record["ratios"]["R_increment"]<=config["increment_limit"])
