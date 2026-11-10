"""AdamW dead-output-row isolation with persistent step budget and auditable state."""
import hashlib
import json
from pathlib import Path
import torch

COMPONENT_ID = "k1_adamw_dead_rows_restore_moments_v1"
STATE_KEYS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")


def digest(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class DeadRowGuard:
    def __init__(self, model):
        layer = model.trajectory_head.mlp[-1]
        if tuple(layer.weight.shape) != (21, 128) or tuple(layer.bias.shape) != (21,):
            raise ValueError("RAW21_HEAD_REQUIRED")
        self.parameters = (layer.weight, layer.bias)
        self.reference = tuple(p.detach()[15:].clone() for p in self.parameters)

    def sha(self):
        return digest([p[15:] for p in self.parameters])

    def state_report(self, optimizer):
        rows = {}
        for name, p in zip(("weight", "bias"), self.parameters):
            for key in STATE_KEYS:
                value = optimizer.state.get(p, {}).get(key)
                if value is not None:
                    if not isinstance(value, torch.Tensor) or value.shape != p.shape or not torch.isfinite(value).all():
                        raise ValueError("OPTIMIZER_STATE_SHAPE_OR_FINITE")
                    rows[name+"."+key] = dict(nonzero=int(value[15:].count_nonzero()),
                                               norm=float(value[15:].norm()))
        return rows

    def sanitize(self, optimizer):
        before = self.state_report(optimizer)
        with torch.no_grad():
            for p in self.parameters:
                if p.grad is not None:
                    p.grad[15:].zero_()
                for key in STATE_KEYS:
                    value = optimizer.state.get(p, {}).get(key)
                    if value is not None:
                        value[15:].zero_()
        return before

    def assert_frozen(self):
        for p, ref in zip(self.parameters, self.reference):
            if not torch.equal(p.detach()[15:], ref):
                raise ValueError("DEAD_ROW_PARAMETER_DRIFT")

    def step(self, optimizer, ledger, budget):
        self.assert_frozen()
        if type(budget) is not int or budget < 1:
            raise ValueError("INVALID_UPDATE_BUDGET")
        path = Path(ledger)
        counts = json.loads(path.read_text())
        if counts["attempted"] >= budget:
            raise PermissionError("OPTIMIZER_STEP_BUDGET_EXHAUSTED")
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for group in optimizer.param_groups for p in group["params"]):
            raise ValueError("NONFINITE_GRADIENT_STOP")
        before_sha = self.sha()
        dirty_state = self.sanitize(optimizer)
        counts["attempted"] += 1
        path.write_text(json.dumps(counts, indent=2))
        backend_delta = None
        try:
            optimizer.step()
            backend_delta = float(torch.cat([(p.detach()[15:]-ref).flatten()
                                              for p, ref in zip(self.parameters, self.reference)]).norm())
        finally:
            with torch.no_grad():
                for p, ref in zip(self.parameters, self.reference):
                    p[15:].copy_(ref)
            self.sanitize(optimizer)
            self.assert_frozen()
        counts["completed"] += 1
        path.write_text(json.dumps(counts, indent=2))
        after_state = self.state_report(optimizer)
        if any(x["nonzero"] for x in after_state.values()):
            raise ValueError("DEAD_ROW_STATE_NOT_ZERO")
        return dict(dead_sha_before=before_sha, dead_sha_after=self.sha(),
                    state_before_sanitization=dirty_state, state_after=after_state,
                    backend_dead_delta_before_restore=backend_delta,
                    attempted=counts["attempted"], completed=counts["completed"])

    def load_reference(self, values):
        if len(values) != 2 or any(v.shape != r.shape for v, r in zip(values, self.reference)):
            raise ValueError("DEAD_REFERENCE_SHAPE")
        self.reference = tuple(v.detach().clone() for v in values)
        self.assert_frozen()
