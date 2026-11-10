"""One pre-registered A bias initialization; no map, DEV, witness or runtime anchor."""
import torch
from query_minco.training.physics_only_v1_2_gate1.k1_interface import make_planner

COMPONENT_ID = "A_nominal_forward24_stop_bias_v1"
MODES = ("A_random", "A_reasonable")


def build(mode, seed):
    if mode not in MODES or type(seed) is not int:
        raise ValueError("INITIALIZATION_IDENTITY")
    torch.manual_seed(seed)
    model = make_planner().cpu()
    if mode == "A_reasonable":
        # Nominal distance = .4 * frozen vmax(3) * fixed T(2) = 2.4 m.
        u = torch.tensor([1/3, 2/3, 1.], dtype=torch.float64)
        points = torch.zeros(3, 3, dtype=torch.float64)
        points[:, 0] = 2.4*(10*u**3-15*u**4+6*u**5)
        lower = torch.tensor([0., -4., -2.], dtype=torch.float64)
        upper = torch.tensor([10., 4., 2.], dtype=torch.float64)
        logits = torch.atanh(2*(points-lower)/(upper-lower)-1).reshape(-1)
        target = torch.cat([logits, torch.zeros(6, dtype=torch.float64)]).float()
        with torch.no_grad():
            model.trajectory_head.mlp[-1].weight[:15].mul_(.01)
            model.trajectory_head.mlp[-1].bias[:15].copy_(target)
    return model


def identity(mode):
    if mode not in MODES:
        raise ValueError("INITIALIZATION_IDENTITY")
    return dict(mode=mode, component_id="A_random_seeded_k1_v1" if mode=="A_random" else COMPONENT_ID,
                architecture_changed=False, runtime_reference=False, uses_geometry=False,
                nominal_goal_m=[2.4,0.,0.], final_physical_weight_scale=None if mode=="A_random" else .01)


def audit(spec, output):
    """One-shot diagnostics against the rule frozen before any results were observed."""
    import json
    import numpy as np
    from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
    from query_minco.training.physics_only_v1_2.set_objective_v12 import PhysicsV12Objective
    from query_minco.training.physics_only_v1.costs import SyntheticGeometry
    from query_minco.evaluation.validation import norm_max
    from query_minco.evaluation.goal import task_metrics
    from .trainer import evaluate, state_sha, write
    rule=json.loads(open(spec["initialization_rule_path"]).read())
    domain=torch.tensor([[0.,-4.,-2.],[10.,4.,2.]])
    geometry=[SyntheticGeometry(torch.empty(0,2,3),domain,True)]
    objective=PhysicsV12Objective(PhysicsV12Spec(phase="A",aggregation="foundation"),num_candidates=1)
    depth=torch.zeros(1,2,128,160);depth[:,0]=.1;depth[:,1]=1.
    records={}
    for mode in MODES:
        model=build(mode,spec["seeds"][0]).eval();before=state_sha(model);grid=[]
        conditions=[dict(goal=[g,0.,0.],v=v,a=a,in_scope=True)
                    for g in rule["in_scope"]["goal_x"] for v in rule["in_scope"]["v_x"]
                    for a in rule["in_scope"]["a_x"]]
        conditions += [dict(x,in_scope=False) for x in rule["out_of_scope"]]
        with torch.no_grad():
            for case in conditions:
                state=torch.tensor([[case["v"],0,0,case["a"],0,0,*case["goal"],0,0,-1.]])
                raw=model(depth,state)
                out=objective(raw,state,geometry,goal_enabled=torch.ones(1,dtype=torch.bool),task_types=("normal",))
                c=out.trajectory.coeffs[0,0].double().numpy();dt=out.trajectory.durations[0,0].numpy()
                maxima=[max(norm_max(a,float(t),k) for a,t in zip(c,dt)) for k in (1,2,3)]
                goal=task_metrics(out.parameters.tail_pva[0,0,0].numpy(),
                                  out.parameters.tail_pva[0,0,1].numpy(),case["goal"])
                good=bool(out.trajectory.solve_valid.all()) and all(x<=v for x,v in zip(maxima,[3.,8.,100.])) and goal["passed"]
                grid.append(dict(case=case,positions=out.parameters.inner_pts[0,0].tolist()+
                    [out.parameters.tail_pva[0,0,0].tolist()],terminal_velocity=out.parameters.tail_pva[0,0,1].tolist(),
                    terminal_acceleration=out.parameters.tail_pva[0,0,2].tolist(),maxima_v_a_j=maxima,
                    task=goal,numeric_and_dynamic_task_qualified=good))
        if state_sha(model)!=before:
            raise ValueError("INITIALIZATION_AUDIT_UPDATED_MODEL")
        from query_minco.training.physics_only_v1_2_gate1.k1_interface import make_objective
        real=evaluate(model,make_objective(),spec["train"]+spec["dev"])
        records[mode]=dict(component=identity(mode),initial_model_sha256=before,cases=grid,
                          in_scope_qualified=all(x["numeric_and_dynamic_task_qualified"] for x in grid if x["case"]["in_scope"]),
                          real_map_diagnostics=real)
    result=dict(status="QUALIFIED" if records["A_reasonable"]["in_scope_qualified"] else "NOT_QUALIFIED",
                scope=rule["in_scope"],rule=rule,records=records,tuning_rounds=1,
                no_general_state_conditioned_reference_claim=True)
    write(output,result)
    return result
