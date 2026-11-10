"""Detached independent frozen hard checks; no oracle is an online selector."""
import torch
from query_minco.evaluation.goal import task_metrics
from query_minco.evaluation.validation import norm_max, axis_range, certify_piece, evaluate
from .goal_v12 import validate_goal_batch
from query_minco.training.physics_only_v1.costs import check_synthetic_geometry


def hard_diagnostics(result, state, geometry, enabled, task_types):
    validate_goal_batch(state[:, 6:9], enabled, task_types)
    check_synthetic_geometry(geometry, len(state), dtype=state.dtype, device=state.device)
    c = result.trajectory.coeffs.detach().double().cpu().numpy()
    dt = result.trajectory.durations.detach().double().cpu().numpy()
    goal_pass, collision_pass, dynamic_pass, domain_pass, maxima, ids, collision_reasons = [], [], [], [], [], [], []
    for b, g in enumerate(geometry):
        boxes = [{"min": box[0].tolist(), "max": box[1].tolist()} for box in g.boxes]
        domains = g.domain.detach().double().numpy()
        gs, cs, fs, ds, ms, cr = [], [], [], [], [], []
        ids.append("hover_zero_goal_extension_v12" if task_types[b] == "hover" else "frozen_goal_nonzero")
        for q in range(6):
            endpoint = evaluate(c[b, q, -1], dt[b, q, -1], 0)
            velocity = evaluate(c[b, q, -1], dt[b, q, -1], 1)
            if not bool(enabled[b]):
                passed = False  # accompanied by known=False; never a negative training label
            elif task_types[b] == "hover":
                passed = bool((endpoint**2).sum()**.5 <= .15 and (velocity**2).sum()**.5 <= .25)
            else:
                passed = task_metrics(endpoint, velocity, state[b, 6:9].detach().double().numpy())["passed"]
            gs.append(passed)
            max_q = [max(norm_max(piece, T, order) for piece, T in zip(c[b, q], dt[b, q]))
                     for order in (1, 2, 3)]
            ms.append(max_q)
            fs.append(all(x <= bound for x, bound in zip(max_q, (3., 8., 100.))))
            ranges = [axis_range(piece, T) for piece, T in zip(c[b, q], dt[b, q])]
            ds.append(all(bool((r[:, 0] >= domains[0]).all() and (r[:, 1] <= domains[1]).all()) for r in ranges))
            certificates = [certify_piece(piece, T, boxes) for piece, T in zip(c[b, q], dt[b, q])]
            cs.append(all(x["passed"] for x in certificates))
            cr.append([x["reason"] for x in certificates])
        goal_pass.append(gs); collision_pass.append(cs); dynamic_pass.append(fs); domain_pass.append(ds); maxima.append(ms)
        collision_reasons.append(cr)
    flags = {name: torch.tensor(values, dtype=torch.bool) for name, values in
             (("HardGoalPass", goal_pass), ("HardCollisionPass", collision_pass),
              ("HardDynamicsPass", dynamic_pass), ("HardDomainPass", domain_pass))}
    known = enabled.detach().cpu()[:, None].expand(-1, 6)
    return {**flags, "HardGoalKnown": known, "hard_maxima_v_a_j": torch.tensor(maxima, dtype=torch.float64),
            "HardAllPass": known & flags["HardGoalPass"] & flags["HardCollisionPass"] &
                           flags["HardDynamicsPass"] & flags["HardDomainPass"],
            "goal_contract_ids": ids, "collision_certificate_reasons": collision_reasons,
            "qualification": "SYNTHETIC_TEST_ONLY_NOT_FLIGHT_CERTIFICATION"}
