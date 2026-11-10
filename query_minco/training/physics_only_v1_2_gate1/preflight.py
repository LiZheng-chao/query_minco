"""Gate1 CPU qualification only; rejected exports never reach a model or trainer."""
import argparse
import json
from pathlib import Path
from .qualification import qualify_export


def write(p,x):
    with p.open("x") as f:json.dump(x,f,indent=2,allow_nan=False);f.write("\n")


def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",required=True,choices=("qualify","cpu0"))
    p.add_argument("--sample",required=True);p.add_argument("--output-root",required=True)
    a=p.parse_args();out=Path(a.output_root)
    if a.stage!="qualify":raise ValueError("CPU0 requires qualified data and a separately validated bound objective interface")
    raw_binding=json.loads((out/"runs/render_sparse_1001_002/EXPORTED_SHA256.json").read_text())
    binary=json.loads((out/"runs/build_002/BINARY.json").read_text())
    result,_=qualify_export(out/"data"/a.sample,raw_binding,binary)
    dest=out/"runs/qualify_sparse_1001_001"
    write(dest/"QUALIFICATION.json",result)
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)
    return 0 if result["training_qualified"] else 2

legacy_main=main

# This section audits the existing production math. It does not alter its SDF/loss.
import hashlib
import numpy as np
import torch

OUT=Path("research_outputs/physics_only_v1_2_gate1_forest_v1")
PREP=OUT/"_preparation/sdf_zero_update_001"
MANIFEST=OUT/"runs/bind_posecert_001/BOUND_MANIFEST.json"
MANIFEST_SHA="fd8237790ec5bb2d7b33fce6f8b66a4398c321b0d4ee6a01a2284f449b68445f"


def complete_branches(g,world):
    """Independent clipping distance, with negative distance to closest face inside."""
    p=np.asarray(world,np.float64);lo=g.boxes[:,0];hi=g.boxes[:,1]
    nearest=np.minimum(np.maximum(p,lo),hi)
    delta=p-nearest
    distances=np.linalg.norm(delta,axis=1)
    inside=((p>=lo)&(p<=hi)).all(1)
    faces=np.concatenate([p-lo,hi-p],1)
    distances[inside]=-faces[inside].min(1)
    # Axis-specific ROI halfspaces are all min branches.
    return np.r_[distances,p[2]-g.floor,p-g.roi[0],g.roi[1]-p]


def compare_full(g,world_points):
    errors=[];included=[];values=[]
    for world in world_points:
        branches=complete_branches(g,world)
        local=(world-g.origin)@g.rotation
        actual=float(g.sdf(torch.tensor(local,dtype=torch.float64)))
        expected=float(branches.min());errors.append(abs(actual-expected));values.append(actual)
        center_distance,_=g.tree.query(world)
        candidate=g.tree.query_ball_point(world,center_distance+g.circumradius+1e-6,eps=0)
        ids=np.flatnonzero(branches[:len(g.boxes)]==branches[:len(g.boxes)].min())
        included.append(any(int(i) in candidate for i in ids))
    return dict(count=len(errors),max_abs_error_m=max(errors),kd_nearest_inclusion=all(included),
                negative_count=sum(v<0 for v in values),positive_count=sum(v>0 for v in values))


def branch_gradient(g,world,index):
    b=g.boxes[index]
    diff=world-np.minimum(np.maximum(world,b[0]),b[1])
    if np.linalg.norm(diff)>0:return diff/np.linalg.norm(diff)
    faces=np.r_[world-b[0],b[1]-world];i=int(np.argmin(faces))
    grad=np.zeros(3);grad[i%3]=-1 if i<3 else 1
    return grad


def old_counterexample(g):
    from torch.autograd.gradcheck import GradcheckError
    p=torch.zeros(3,dtype=torch.float64,requires_grad=True)
    original=torch.tensor([[[0.,0.,0.],[.1,.02,.03],[.2,-.02,.1]]],dtype=torch.float64,requires_grad=True)
    reproduced=False
    try:torch.autograd.gradcheck(lambda x:g.sdf(x),(original,),atol=1e-5)
    except GradcheckError:reproduced=True
    branches=complete_branches(g,g.origin);ids=np.argsort(branches)[:2]
    gap=float(branches[ids[1]]-branches[ids[0]])
    automatic=torch.autograd.grad(g.sdf(p),p)[0].detach().numpy()
    analytic=branch_gradient(g,g.origin,int(ids[0]))@g.rotation
    records=[]
    for axis in range(3):
        direction=np.eye(3)[axis];delta=1e-6
        plus=complete_branches(g,g.origin+direction*delta)
        minus=complete_branches(g,g.origin-direction*delta)
        records.append(dict(direction=direction.tolist(),epsilon_m=delta,
                            plus_branch=int(plus.argmin()),minus_branch=int(minus.argmin()),
                            central_difference=float((plus.min()-minus.min())/(2*delta)),
                            selected_branch_analytic=float(analytic@direction)))
    # All branches are 1-Lipschitz. Gap>2r certifies same global min within r.
    radius=gap/8
    if not radius>0:raise ValueError("OLD_FIXTURE_EXACT_TIE_REQUIRES_DIFFERENT_PROOF")
    one_sided=[]
    for axis in range(3):
        u=np.eye(3)[axis]
        moved=complete_branches(g,g.origin+radius*u)
        if moved.argmin()!=ids[0]:raise ValueError("STABLE_BRANCH_PROOF_FAILED")
        derivative=float((moved.min()-branches.min())/radius)
        if abs(derivative-analytic[axis])>1e-5:raise ValueError("FIXED_BRANCH_DIRECTION_MISMATCH")
        one_sided.append(derivative)
    repeat=[]
    for _ in range(3):
        x=p.detach().clone().requires_grad_();v=g.sdf(x);grad=torch.autograd.grad(v,x)[0]
        repeat.append((float(v),grad.tolist()))
    # Also an exact 2-box tie fixture: minimum of two smooth distances has
    # right directional derivative min(active gradients . u).
    grads=np.stack([branch_gradient(g,g.origin,int(i)) for i in ids])
    return dict(original_point_B0_m=[0,0,0],original_gradcheck_reproduced=reproduced,
       original_atol=1e-5,original_epsilon_m=1e-6,nearest_ids=ids.tolist(),
       nearest_distances_m=branches[ids].tolist(),nearest_gap_m=gap,
       analytic_selected_gradient=analytic.tolist(),active_nearby_gradients=grads.tolist(),
       stable_radius_m=radius,stable_branch_one_sided=one_sided,
       stable_branch_analytic_difference=float(np.max(np.abs(automatic-analytic))),
       central_difference_crosses_branch=records,repeatable=all(r==repeat[0] for r in repeat),
       distinction="FP64 original point has unique nearest branch but 1e-6 finite-difference ball crosses the switch; not an exact mathematical tie")


def smooth_certificate(g):
    rng=np.random.RandomState(20261104);selected=[]
    radius=1e-6
    for _ in range(256):
        local=rng.uniform([-.15,-.15,-.15],[.25,.15,.15]);world=g.origin+local
        distances=complete_branches(g,world);ids=np.argsort(distances)[:2]
        if ids[0]>=len(g.boxes) or distances[ids[1]]-distances[ids[0]]<=1e-3:continue
        b=g.boxes[ids[0]];center=b.mean(0);half=(b[1]-b[0])/2
        if np.min(np.abs(np.abs(world-center)-half))<=10*radius:continue
        if distances[ids[0]]<=.01:continue
        selected.append(dict(point_B0_m=local.tolist(),nearest_id=int(ids[0]),
            nearest_gap_m=float(distances[ids[1]]-distances[ids[0]]),difference_radius_m=radius,
            independent_analytic_gradient=(branch_gradient(g,world,int(ids[0]))@g.rotation).tolist()))
        if len(selected)==4:break
    if len(selected)<4:raise ValueError("NO_PROVEN_SMOOTH_POINTS")
    p=torch.tensor([x["point_B0_m"] for x in selected],dtype=torch.float64,requires_grad=True)
    ok=torch.autograd.gradcheck(lambda x:g.sdf(x),(p,),eps=radius,atol=1e-5)
    auto=torch.autograd.grad(g.sdf(p).sum(),p)[0]
    analytic=torch.tensor([x["independent_analytic_gradient"] for x in selected],dtype=torch.float64)
    torch.testing.assert_close(auto,analytic,atol=1e-10,rtol=0)
    fp32=p.detach().float().requires_grad_();d=g.sdf(fp32);grad=torch.autograd.grad(d.sum(),fp32)[0]
    if not torch.isfinite(grad).all():raise ValueError("FP32_NONFINITE")
    torch.testing.assert_close(grad.double(),analytic,atol=1e-5,rtol=1e-3)
    return dict(points=selected,fp64_gradcheck=bool(ok),fp64_autograd=auto.tolist(),
                fp32_finite=True,fp32_gradients=grad.tolist(),
                proof="every branch is 1-Lipschitz; nearest gap>2r and single-box feature gaps>10r")


def exact_tie_fixture():
    # Hand boxes, independent of real provenance. Their symmetric distances tie at zero.
    boxes=torch.tensor([[[-2.,-.5,-.5],[-1.,.5,.5]],[[1.,-.5,-.5],[2.,.5,.5]]],dtype=torch.float64)
    def branches(x):
        q=(x[None]-(boxes[:,0]+boxes[:,1])/2).abs()-(boxes[:,1]-boxes[:,0])/2
        return q.clamp_min(0).norm(dim=-1)+q.amax(-1).clamp_max(0)
    x=torch.zeros(3,dtype=torch.float64,requires_grad=True)
    ds=branches(x);index=int(ds.detach().argmin());v=ds[index]
    grad=torch.autograd.grad(v,x)[0]
    expected=torch.tensor([1.,0,0],dtype=torch.float64)
    torch.testing.assert_close(grad,expected,atol=1e-12,rtol=0)
    records=[]
    for u in (torch.tensor([1.,0,0],dtype=torch.float64),torch.tensor([-1.,0,0],dtype=torch.float64),
              torch.tensor([0.,1,0],dtype=torch.float64)):
        right=float((branches(x.detach()+1e-6*u).min()-ds.min())/1e-6)
        active=torch.tensor([[1.,0,0],[-1.,0,0]],dtype=torch.float64)
        expectation=float((active@u).min())
        if abs(right-expectation)>1e-5:raise ValueError("TIE_DIRECTIONAL_DERIVATIVE")
        records.append(dict(direction=u.tolist(),right_derivative=right,expected=expectation))
    return dict(selected_index=index,selected_gradient=grad.tolist(),directional_derivatives=records,
                classical_gradient_exists=False,selected_gradient_is_active_branch=True,
                caveat="a selected active gradient is not every directional derivative of the min")


def source_hashes():
    root=Path.cwd()
    files=list((root/"query_minco").rglob("*.py"))+list((root/"tests").glob("test_physics_v12_gate1_*.py"))
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def assert_frozen():
    entry=json.loads((PREP/"ENTRY.json").read_text())
    for p,h in entry["protected_sha256"].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError("HISTORICAL_SHA_DRIFT "+p)
    frozen=PREP/"SOURCE_FREEZE.json"
    if frozen.exists() and source_hashes()!=json.loads(frozen.read_text()):
        raise ValueError("SOURCE_DRIFT_SINCE_FREEZE")


def run_b(run):
    import sys
    torch.set_num_threads(1);assert_frozen()
    # pytest is pure Python appended after environment numerical dependencies.
    sys.path.append("/home/lizheng/anaconda3/lib/python3.11/site-packages")
    import pytest
    argv=["-q","-p","no:cacheprovider","--basetemp="+str(run/"pytest_tmp"),
          "--junitxml="+str(run/"TEST_RESULTS.xml"),
          "tests/test_physics_v12_gate1_sdf_nonsmooth.py",
          "tests/test_physics_v12_gate1_pose.py","tests/test_physics_v12_gate1_objective.py",
          "tests/test_physics_v12_gate1_geometry.py",
          "-k","not test_exact_distance_matches_independent_all_boxes"]
    write(run/"PYTEST_ARGS.json",argv)
    code=pytest.main(argv)
    if code:raise ValueError("GATE1_B_TESTS_FAILED exit="+str(code))
    from .binding import load_bound
    from .objective import collision_direction_probe,BoundPhysicsV12Objective
    b=load_bound(MANIFEST,MANIFEST_SHA);g=b.geometry[0]
    rng=np.random.RandomState(20261104)
    points=rng.uniform(g.roi[0]+.01,g.roi[1]-.01,size=(64,3))
    report=dict(status="GATE1_B_PASS",manifest_sha256=MANIFEST_SHA,
                old_counterexample=old_counterexample(g),exact_tie=exact_tie_fixture(),
                smooth=smooth_certificate(g),enumeration=compare_full(g,points),
                collision_direction_probe=collision_direction_probe(b),
                unknown_policy="outside ROI and above source top reject; no-hit invalid; occlusion known only from complete generated counts",
                samples=1,external_authority=False,source_sha256=source_hashes(),network_runs=0)
    assert_frozen()
    write(run/"B_RESULT.json",report)
    print(json.dumps(dict(status=report["status"],tests=0,manifest_sha256=MANIFEST_SHA)),flush=True)


def supervise(case,argv):
    """Fresh runs only, CPU-only, retained receipts and finite resource budget."""
    import os,sys,time,subprocess,signal,traceback,importlib.util
    run=OUT/"runs"/case;run.mkdir();tmp=run/"tmp";tmp.mkdir()
    env=dict(os.environ);env.update(PYTHONPATH=str(Path.cwd()),PYTHONDONTWRITEBYTECODE="1",
       CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
       NUMEXPR_NUM_THREADS="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",TMPDIR=str(tmp.resolve()),
       TEMP=str(tmp.resolve()),TMP=str(tmp.resolve()))
    sp=importlib.util.spec_from_file_location("budget_helpers",Path("experiments/physics_only_v1_2_math_contract_v1/run_math_acceptance.py"))
    helpers=importlib.util.module_from_spec(sp);sp.loader.exec_module(helpers)
    write(run/"COMMAND.json",dict(argv=argv,cwd=str(Path.cwd()),env={k:env[k] for k in (
        "PYTHONPATH","PYTHONDONTWRITEBYTECODE","CUDA_VISIBLE_DEVICES","OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","PYTEST_DISABLE_PLUGIN_AUTOLOAD","TMPDIR")},
        timeout_s=300,rss_limit_bytes=4*1024**3,new_output_limit_bytes=128*1024**2,
        source_sha256=source_hashes(),manifest_sha256=MANIFEST_SHA))
    peak=0;stop=None;error=None;child=None;code=None;start=time.monotonic()
    new_paths=json.loads((PREP/"ENTRY.json").read_text())["output_allowlist"]
    with (run/"stdout.log").open("xb") as stdout,(run/"stderr.log").open("xb") as stderr:
        try:
            child=subprocess.Popen(argv,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            while child.poll() is None:
                peak=max(peak,helpers.process_tree_rss(child.pid))
                disk=sum(helpers.disk_bytes(OUT/p) for p in new_paths if (OUT/p).exists())
                if time.monotonic()-start>300:stop="TIME_BUDGET"
                if peak>4*1024**3:stop="RSS_BUDGET"
                if disk>128*1024**2:stop="DISK_BUDGET"
                if stop:os.killpg(child.pid,signal.SIGKILL);break
                time.sleep(.1)
            code=child.wait()
        except BaseException:
            error=traceback.format_exc()
            if child and child.poll() is None:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
    result=dict(exit_code=code,stop=stop,supervisor_error=error,elapsed_s=time.monotonic()-start,
                sampled_peak_tree_RSS_bytes=peak,optimizer_updates=0,episodes=0)
    write(run/"PROCESS_EXIT.json",result);print(json.dumps(dict(case=case,**result)),flush=True)
    if code!=0 or stop or error:raise ValueError("GATE1_STOP "+case)
    return run


def stats(x):
    a=x.detach().double().reshape(-1)
    if not a.numel() or not bool(torch.isfinite(a).all()):raise ValueError("NONFINITE_DIAGNOSTIC")
    return dict(min=float(a.min()),median=float(a.quantile(.5)),p90=float(a.quantile(.9)),
        max=float(a.max()),mean=float(a.mean()),norm=float(a.norm()),
        rms=float(a.square().mean().sqrt()),zero_ratio=float((a==0).double().mean()),
        positive_ratio=float((a>0).double().mean()),negative_ratio=float((a<0).double().mean()),finite=True)


def sample_switch_risk(g,points):
    points=points.detach().numpy().reshape(-1,3)
    gaps=[];ids=[];branch_gradients=[];errors=[]
    # At most 1152 samples, complete enumeration independent of KD-tree.
    for p in points:
        world=p.astype(np.float64)@g.rotation.T+g.origin
        ds=complete_branches(g,world);i=np.argsort(ds)[:2]
        gaps.append(float(ds[i[1]]-ds[i[0]]));ids.append(i.tolist())
        branch_gradients.append(branch_gradient(g,world,int(i[0])).tolist() if i[0]<len(g.boxes) else None)
    tensor=torch.tensor(points,requires_grad=True);distance=g.sdf(tensor)
    grad=torch.autograd.grad(distance.sum(),tensor)[0]
    flat=grad.detach().numpy()
    gaps=np.asarray(gaps)
    return dict(sample_count=len(points),difference_probe_radius_m=1e-6,
        switch_region_count=int((gaps<=2e-6).sum()),switch_region_ratio=float((gaps<=2e-6).mean()),
        exact_double_ties_count=int((gaps==0).sum()),fp32_resolution_risk_count=int((gaps<=2e-5).sum()),
        nonfinite_gradient_ratio=float((~np.isfinite(flat)).any(1).mean()),
        all_gradients_B0=flat.tolist(),nearest_ids=ids,nearest_gap_m=gaps.tolist(),
        branch_gradients_world_reference=branch_gradients,
        note="gap<=2r is a risk flag, not proof of exact nondifferentiability; FP32 world-coordinate rounding adds uncertainty")


def run_c(arm,run):
    import sys,platform
    from dataclasses import asdict
    from unittest.mock import patch
    from query_minco.models import QueryMincoPlanner
    from query_minco.training.physics_only_v1.smoke import parameter_hash
    from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
    from query_minco.training.physics_only_v1_2.smoke import cosine
    from query_minco.training.pg1.losses import analytic_samples
    from .binding import load_bound
    from .objective import BoundPhysicsV12Objective,collision_direction_probe
    from .hard_validator import validate_result
    import scipy,query_minco
    assert_frozen()
    b_receipt=json.loads((OUT/"runs/sdf_b_001/PROCESS_EXIT.json").read_text())
    b_result=json.loads((OUT/"runs/sdf_b_001/B_RESULT.json").read_text())
    if b_receipt["exit_code"]!=0 or b_result["status"]!="GATE1_B_PASS":raise ValueError("B_NOT_PASSED")
    config=PhysicsV12Spec(phase=arm,aggregation="foundation" if arm=="A" else "softmin_guard")
    arm_binding=dict(spec=asdict(config),seed=20261104,checkpoint=None,manifest_sha256=MANIFEST_SHA,
                     b_result_sha256=hashlib.sha256((OUT/"runs/sdf_b_001/B_RESULT.json").read_bytes()).hexdigest(),
                     source_sha256=source_hashes(),device="cpu",dtype="float32",optimizer=None)
    write(run/"ARM_CONFIG.json",arm_binding)
    config_hash=hashlib.sha256(json.dumps(arm_binding,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    batch=load_bound(MANIFEST,MANIFEST_SHA);batch.validate()
    torch.set_num_threads(1);torch.manual_seed(20261104)
    # Any attempt to construct an optimizer is a task violation.
    with patch.object(torch.optim.Optimizer,"__init__",side_effect=RuntimeError("OPTIMIZER_FORBIDDEN")):
        model=QueryMincoPlanner().cpu().eval()
        before=parameter_hash(model)
        raw=model(batch.depth,batch.state)
        result=BoundPhysicsV12Objective(config)(raw,batch)
        write(run/"FORWARD.json",dict(loss=float(result.loss),raw=raw.detach().tolist(),
              per_query={k:v.detach().tolist() for k,v in result.metrics.items()},
              model_sha256_before=before,config_sha256=config_hash,manifest_sha256=MANIFEST_SHA))
        named=[(n,p) for n,p in model.named_parameters() if (
            n.startswith("depth_encoder.") or n.startswith("state_encoder.") or n.startswith("query_decoder.layers."))]
        shared=[p for _,p in named]
        group_indices={group:[i for i,(n,_) in enumerate(named) if n.startswith(prefix)] for group,prefix in (
            ("CNN","depth_encoder."),("State","state_encoder."),("Transformer","query_decoder.layers."))}
        def gradient(value):
            grad=torch.autograd.grad(value,(raw,*shared),retain_graph=True,allow_unused=True)
            rg=grad[0] if grad[0] is not None else torch.zeros_like(raw)
            arrays=[g if g is not None else torch.zeros_like(p) for g,p in zip(grad[1:],shared)]
            sg=torch.cat([g.reshape(-1) for g in arrays])
            if not torch.isfinite(rg).all() or not torch.isfinite(sg).all():raise ValueError("NONFINITE_GRADIENT")
            if rg[...,15:21].count_nonzero():raise ValueError("FORBIDDEN_RAW_TIME_SCORE_GRADIENT")
            groups={group:stats(torch.cat([arrays[i].reshape(-1) for i in indices])) for group,indices in group_indices.items()}
            return rg,sg,groups
        names=("C_viol","C_shape","G_N","G_F","G_task_train","P","S","F_dyn","F_domain","F","V","Q","J")
        weights=dict(C_viol=config.w_cv,C_shape=config.w_cc,G_task_train=config.w_g,P=config.w_p,
                     S=config.w_s,F=config.w_f,F_dyn=config.w_f,F_domain=config.w_f*config.eta_domain,
                     V=1.,Q=1.,J=1.,G_N=1.,G_F=1.)
        components={};stored={}
        for name in names:
            weight=weights[name];term=result.metrics[name]
            rg,sg,groups=gradient(weight*term.mean())
            if name in ("C_viol","C_shape","G_task_train","P","S","F","V","Q"):
                stored[name]=(rg.detach(),sg.detach())
            queries=[]
            for k in range(6):
                qr,qs,qgroups=gradient(weight*term[0,k])
                queries.append(dict(query=k,raw_gradient=qr.detach().tolist(),raw=stats(qr[0,k,:15]),
                    raw_position=stats(qr[0,k,:9]),raw_terminal_velocity=stats(qr[0,k,9:12]),
                    raw_terminal_acceleration=stats(qr[0,k,12:15]),shared=stats(qs),shared_groups=qgroups,
                    time_score_gradient_max=float(qr[...,15:21].abs().max())))
            components[name]=dict(cost=stats(term),weighted_cost=stats(weight*term),weight=weight,
                gradient_of="weight * component.mean over six queries",
                mean_weighted_raw=stats(rg[...,:15]),mean_weighted_shared=stats(sg),
                shared_groups=groups,per_query=queries,
                diagnostic_only=name in ("G_N","G_F"))
        primary=("C_viol","F","G_task_train","C_shape","P","S")
        effective={};sum_raw=torch.zeros_like(raw);sum_shared=None
        for name in primary:
            term=result.metrics[name]
            upstream=torch.autograd.grad(result.loss,term,retain_graph=True)[0].detach()
            rg,sg,groups=gradient((upstream*term).sum())
            effective[name]=dict(upstream_per_query=upstream.tolist(),raw=stats(rg[...,:15]),shared=stats(sg),
                per_query_raw=[stats(rg[0,k,:15]) for k in range(6)],shared_groups=groups)
            sum_raw+=rg;sum_shared=sg if sum_shared is None else sum_shared+sg
        rg,sg,groups=gradient(result.loss)
        torch.testing.assert_close(sum_raw,rg,atol=2e-5,rtol=2e-4)
        torch.testing.assert_close(sum_shared,sg,atol=2e-5,rtol=2e-4)
        stored["total"]=(rg.detach(),sg.detach())
        pairs={}
        keys=list(stored)
        for i,name in enumerate(keys):
            for other in keys[i+1:]:
                pairs[name+"__"+other]=dict(raw=cosine(stored[name][0][...,:15],stored[other][0][...,:15]),
                    shared=cosine(stored[name][1],stored[other][1]))
        # Hard results are independent of sampled C and never supplied to the model.
        hard=validate_result(result,batch)
        ana=analytic_samples(result.trajectory,config.points_per_segment)
        switch=sample_switch_risk(batch.geometry[0],ana["p"][0])
        after=parameter_hash(model)
        if before!=after:raise ValueError("MODEL_UPDATED")
        report=dict(status="GATE1_C_PASS",arm=arm,aggregation=config.aggregation,loss=float(result.loss),
            config_sha256=config_hash,manifest_sha256=MANIFEST_SHA,
            model_sha256_before=before,model_sha256_after=after,parameters_unchanged=True,
            checkpoint_before=None,checkpoint_after=None,optimizer_constructed=False,optimizer_updates=0,
            shape=dict(depth=list(batch.depth.shape),state=list(batch.state.shape),raw=list(raw.shape)),
            device="cpu",dtype="float32",seed=20261104,near_far="far; goal distance 6m; goal enabled",
            per_query={k:v.detach().tolist() for k,v in result.metrics.items()},
            components=components,effective_objective_gradients=effective,gradient_cosines=pairs,
            undefined_cosine="null / UNDEFINED for zero norm",total_raw_gradient=rg.detach().tolist(),
            total_raw_stats=stats(rg[...,:15]),total_shared_stats=stats(sg),total_shared_groups=groups,
            shared_parameters=[dict(name=n,shape=list(p.shape)) for n,p in named],
            direct_raw_time_score_gradient_max=float(rg[...,15:21].abs().max()),
            tanh_saturation_ratio=float((raw[...,:15].tanh().abs()>.99).double().mean()),
            pi=result.metrics["pi"].detach().tolist(),entropy=result.metrics["entropy"].detach().tolist(),
            K_eff=result.metrics["K_eff"].detach().tolist(),
            winner_counts=torch.bincount(result.metrics["J"].argmin(-1),minlength=6).tolist(),
            candidate_raw_pairwise_distance=torch.cdist(raw[0,:,:15],raw[0,:,:15]).detach().tolist(),
            independent_hard=hard,HardAllPass_count=sum(x["passed"] for x in hard),
            solver_residuals={k:v.detach().tolist() for k,v in result.trajectory.diagnostics.items()},
            durations_s=result.trajectory.durations.detach().tolist(),
            trajectory_coefficients_B0=result.trajectory.coeffs.detach().tolist(),
            trajectory_sampled_positions_B0=ana["p"].detach().tolist(),
            switch_risk=switch,collision_direction_probe=collision_direction_probe(batch),
            external_safety_certification=False,weights_tuned=False,episodes=0,
            environment=dict(python=sys.version,query_minco_file=query_minco.__file__,
                torch_version=torch.__version__,torch_file=torch.__file__,numpy_version=np.__version__,
                numpy_file=np.__file__,scipy_version=scipy.__version__,scipy_file=scipy.__file__),
            source_sha256=source_hashes())
        assert_frozen();batch.validate()
        write(run/"C_RESULT.json",report)
        print(json.dumps(dict(status=report["status"],arm=arm,loss=report["loss"],
            hard_pass_count=report["HardAllPass_count"],model_unchanged=True)),flush=True)


def run_regression(run):
    import sys,os
    assert_frozen();sys.path.append("/home/lizheng/anaconda3/lib/python3.11/site-packages")
    import pytest
    artifact=run/"pytest_artifacts";artifact.mkdir()
    os.environ["PG1_TEST_OUTPUT"]=str(artifact.resolve())
    argv=["-q","-p","no:cacheprovider","--basetemp="+str(run/"pytest_tmp"),
      "--junitxml="+str(run/"TEST_RESULTS.xml"),
      "query_minco/training/physics_only_v1_2/tests","query_minco/training/physics_only_v1/tests",
      "tests/test_pg1_phase1.py","tests/test_pg1_v2_r1.py","tests/test_set_physical_risk_v1.py","tests/test_reference_contracts.py"]
    write(run/"PYTEST_ARGS.json",argv)
    result=pytest.main(argv)
    if result:raise ValueError("REGRESSION_FAILED "+str(result))
    assert_frozen();write(run/"REGRESSION_RESULT.json",dict(status="PASS",optimizer_updates=0,source_sha256=source_hashes()))


def main():
    import sys
    if "--stage" in sys.argv and sys.argv[sys.argv.index("--stage")+1] in ("qualify","cpu0"):
        return legacy_main()
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("b","c","regression"),required=True)
    p.add_argument("--arm",choices=("A","B"));p.add_argument("--run",required=True)
    args=p.parse_args();run=Path(args.run)
    if args.stage=="b":run_b(run)
    elif args.stage=="c":run_c(args.arm,run)
    else:run_regression(run)
    return 0

if __name__=="__main__":raise SystemExit(main())
