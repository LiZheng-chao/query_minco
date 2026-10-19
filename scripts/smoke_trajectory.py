"""P2A reproducible numerical evidence; untrained/debug trajectories only."""
import argparse
import dataclasses
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from YOPO.minco.minco import MincoS3NU
from query_minco.trajectory import (PhysicalParameterConfig, MincoAdapter, NumericalSolveError,
                                   TrajectorySampler, TrajectoryPostprocessor, evaluate_piece)
from query_minco.tests.common import model_and_inputs, gradient_summary
from query_minco.tests.trajectory_common import fixture, raw_fixture, trajectory_loss
from query_minco.tests.trajectory_gradient_checks import GROUPS, directional_checks
from query_minco.tests.trajectory_reference import reference_solve, reference_evaluate
from query_minco.tests.test_minco_adapter import REFERENCE_ATOL, JOINT_ATOL


def maximum(value):
    return float(value.abs().max())


def boundary_details(solution, head, tail, inner):
    c,d=solution.coeffs,solution.durations
    zero=torch.zeros_like(d[...,0,None])
    result={}
    for k,name in enumerate(("p","v","a")):
        result["head_"+name]=maximum(evaluate_piece(c[...,0,:,:],zero,k).squeeze(-2)-head[...,k,:])
        result["tail_"+name]=maximum(evaluate_piece(c[...,2,:,:],d[...,2,None],k).squeeze(-2)-tail[...,k,:])
    for k,name in enumerate(("p","v","a","jerk","snap")):
        result["joint_"+name]=max(maximum(
            evaluate_piece(c[...,i,:,:],d[...,i,None],k)-evaluate_piece(c[...,i+1,:,:],zero,k))
            for i in range(2))
    result["waypoints"]=max(maximum(
        evaluate_piece(c[...,i,:,:],d[...,i,None],0).squeeze(-2)-inner[...,i,:]) for i in range(2))
    return result


def fixture_evidence(device):
    cases=[]
    for batch in (1,2):
        h,t,q,d=fixture(batch,device)
        solution=MincoAdapter()(h,t,q,d)
        sampled=TrajectorySampler()(solution,num_samples=65)
        reference_errors={name:0. for name in ("coeffs","p","v","a","jerk")}
        for b in range(batch):
            for cid in range(6):
                arrays=[x[b,cid].cpu().numpy() for x in (h,t,q,d)]
                ref=reference_solve(*arrays)
                reference_errors["coeffs"]=max(reference_errors["coeffs"],float(
                    np.abs(solution.coeffs[b,cid].cpu().numpy()-ref).max()))
                for k,name in enumerate(("p","v","a","jerk")):
                    evaluated=reference_evaluate(ref,arrays[-1],sampled.times[b,cid].cpu().numpy(),k)
                    error=float(np.abs(getattr(sampled,name)[b,cid].cpu().numpy()-evaluated).max())
                    reference_errors[name]=max(reference_errors[name],error)
        assert all(v<=REFERENCE_ATOL[k] for k,v in reference_errors.items())
        cases.append({"batch":batch,"candidates":batch*6,
                      "boundary_errors":boundary_details(solution,h,t,q),
                      "independent_fp64_reference_errors":reference_errors})
    # Short default-domain case is checked using the same runtime numerical limits;
    # report its larger FP32 high-order error separately from the regular fixture.
    h,t,q,d=fixture(1,device); d[:]=d.new_tensor([.12,.12,.56])
    short=MincoAdapter()(h,t,q,d)
    short_record={"durations_s":[.12,.12,.56],"boundary_errors":boundary_details(short,h,t,q)}
    d[:]=d.new_tensor([.001,10.,.001])
    try:
        MincoAdapter()(h,t,q,d)
    except NumericalSolveError as error:
        ill={"durations_s":[.001,10.,.001],"status":"NUMERICAL_REJECT",
             "reason":str(error),"residuals":{k:maximum(v) for k,v in
                        error.diagnostics.get("max_abs_residuals",{}).items()}}
    else:
        raise AssertionError("ill-conditioned fixture unexpectedly passed")
    return {"regular_cases":cases,"short_segment_case":short_record,"ill_conditioned_case":ill}


def network_evidence(device,amp=False):
    model,depth,state=model_and_inputs(device=device)
    with torch.autocast(device_type=device,dtype=torch.float16,enabled=amp):
        raw=model(depth,state); raw.retain_grad()
        output=TrajectoryPostprocessor()(raw,state,num_samples=64)
        loss=output.samples.p.square().mean()+.03*(output.samples.v-.2).square().mean()
    loss.backward()
    groups=gradient_summary(model)
    assert all(v["all_finite"] and v["l2_norm"]>0 for v in groups.values())
    raw_groups={name:float(raw.grad[...,lo:hi].float().norm()) for name,(lo,hi) in GROUPS.items()}
    assert all(np.isfinite(v) and v>0 for v in raw_groups.values())
    assert bool((raw.grad[...,19:21]==0).all())
    record={"raw_dtype":str(raw.dtype),"coefficient_dtype":str(output.trajectory.coeffs.dtype),
            "sample_dtype":str(output.samples.p.dtype),"raw_shape":list(raw.shape),
            "durations_shape":list(output.parameters.durations.shape),
            "coeffs_shape":list(output.trajectory.coeffs.shape),
            "times_shape":list(output.samples.times.shape),
            "pva_jerk_shape":list(output.samples.p.shape),
            "loss":float(loss.detach()),"network_gradients":groups,
            "raw_group_gradient_norms":raw_groups,"score_feas_gradient_max":maximum(raw.grad[...,19:21]),
            "time_sum_error_s":maximum(output.parameters.diagnostics["time_sum_error_s"]),
            "boundary_errors":boundary_details(output.trajectory,output.parameters.head_pva,
                                               output.parameters.tail_pva,output.parameters.inner_pts),
            "verification_scope":"NUMERICAL_ONLY_NO_COLLISION_OR_DYNAMICS_CERTIFICATE"}
    arrays={"raw":raw.detach().cpu().numpy(),"state":state.cpu().numpy(),
            "durations":output.parameters.durations.detach().cpu().numpy(),
            "coeffs":output.trajectory.coeffs.detach().cpu().numpy(),
            "times":output.samples.times.detach().cpu().numpy()}
    for name in ("p","v","a","jerk"):
        arrays[name]=getattr(output.samples,name).detach().cpu().numpy()
    return record,arrays


def run(device):
    result={"fixture":fixture_evidence(device)}
    raw,state=raw_fixture(device)
    grad,checks=directional_checks(raw,state)
    assert all(x["absolute_error"]<=.003+.03*abs(x["autograd"]) for x in checks)
    result["directional_differences"]=checks
    result["physical_raw_gradient_norms"]={name:float(grad[...,lo:hi].norm())
                                         for name,(lo,hi) in GROUPS.items()}
    result["network"],arrays=network_evidence(device)
    if device=="cuda":
        result["network_amp"],_=network_evidence(device,amp=True)
    return result,arrays


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    method=Path(__file__).resolve().parents[1]
    args.output_dir.resolve().relative_to(method)
    if args.output_dir.exists():
        parser.error("output directory exists; choose a new run rather than overwrite")
    torch.set_num_threads(1)
    torch.manual_seed(701)
    result={"label":"UNTRAINED_DEBUG_MATH_ONLY","frame":"B0 fixed execution-start local frame",
            "units":{"p":"m","v":"m/s","a":"m/s^2","jerk":"m/s^3","time":"s"},
            "torch":torch.__version__,"cuda_build":torch.version.cuda,"seed_fixture":701,"seed_network":42,
            "physical_config":dataclasses.asdict(PhysicalParameterConfig()),
            "baseline_solver_source":inspect.getfile(MincoS3NU),
            "reference":"Independent FP64 normalized-time equality-constrained minimum-jerk KKT QP",
            "reference_abs_tolerances":REFERENCE_ATOL,"joint_abs_tolerances":JOINT_ATOL}
    result["cpu"],arrays=run("cpu")
    if torch.cuda.is_available():
        result["gpu"]=torch.cuda.get_device_name(0)
        result["cuda"],_=run("cuda")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    with (args.output_dir/"summary.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2); f.write("\n")
    # Detach/NumPy only for artifact serialization, after all differentiable solves.
    with (args.output_dir/"untrained_debug_samples.npz").open("xb") as f:
        np.savez_compressed(f,**arrays)
    print("raw",result["cpu"]["network"]["raw_shape"],"durations",
          result["cpu"]["network"]["durations_shape"],"coeffs",result["cpu"]["network"]["coeffs_shape"],
          "p/v/a/jerk",result["cpu"]["network"]["pva_jerk_shape"])
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
