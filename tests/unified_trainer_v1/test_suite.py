"""Real fixture tests; failure records and artifacts use exclusive output namespaces."""
import argparse
import copy
import json
from pathlib import Path
import traceback
import numpy as np
import torch
from query_minco.training.io import write, sha, require, digest
from query_minco.training.run_spec import load_spec, identity, authorize, PROFILES
from query_minco.training.qualification import qualify
from query_minco.training.initialization import initialize, head_rows, rows_unchanged
from query_minco.training.recipe import optimizer, lr_at
from query_minco.training.datasets.manifest import FrozenDataset
from query_minco.training.datasets.order import FrozenOrder
from query_minco.training.checkpoints import save_checkpoint, load_checkpoint
from query_minco.inference.identity import state_sha

def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=["unit", "numerical", "restore"], required=True)
    p.add_argument("--trained-run"); a = p.parse_args()
    out = Path(a.output); out.mkdir(); results = []
    def test(name, fn):
        try:
            extra = fn() or {}
            results.append(dict(name=name, passed=True, details=extra))
        except BaseException as e:
            results.append(dict(name=name, passed=False, error=repr(e), traceback=traceback.format_exc()))
        write(out / (name + ".json"), results[-1])
    def rejects(fn, text=None):
        try: fn()
        except (ValueError, FileExistsError, KeyError, OSError) as e:
            if text: require(text in str(e), "WRONG_REJECTION " + repr(e))
            return dict(rejected=repr(e))
        raise AssertionError("NEGATIVE_ACCEPTED")
    s = load_spec(a.config)
    ds, order, gate = qualify(s)

    if a.mode == "unit":
        test("data_shapes_M1_M4", lambda: gate)
        def bad_manifest(name, change):
            m = copy.deepcopy(ds.manifest); change(m)
            target = out / (name + "_manifest.json"); write(target, m)
            b = dict(path=str(target), sha256=sha(target))
            return rejects(lambda: FrozenDataset(b, s["profile"]))
        test("reject_eval_split", lambda: bad_manifest("eval", lambda m: m["rows"][0].update(split="eval")))
        test("reject_M0", lambda: bad_manifest("m0", lambda m: m["rows"][0].update(M=0)))
        test("reject_wrong_cache_SHA", lambda: bad_manifest("sha", lambda m: m["rows"][0]["artifact"].update(sha256="0"*64)))
        test("reject_path_escape", lambda: bad_manifest("escape", lambda m: m.update(roots=[str(out)])))
        def bad_arrays(name, mutate):
            with np.load(ds.rows["synthetic_near"]["artifact"]["path"]) as z: arr = {k: z[k].copy() for k in z.files}
            mutate(arr)
            target = out / (name + ".npz")
            with target.open("xb") as f: np.savez(f, **arr)
            m = copy.deepcopy(ds.manifest); m["roots"].append(str(out))
            m["rows"][0]["artifact"] = dict(path=str(target), sha256=sha(target))
            mp = out / (name + "_manifest.json"); write(mp, m)
            dataset = FrozenDataset(dict(path=str(mp), sha256=sha(mp)), s["profile"])
            return rejects(lambda: dataset.load(["synthetic_near"]))
        test("reject_state_shape", lambda: bad_arrays("shape", lambda x: x.update(state=x["state"][:11])))
        test("reject_NaN", lambda: bad_arrays("nan", lambda x: x["state"].__setitem__(0, np.nan)))
        test("reject_invalid_mask", lambda: bad_arrays("mask", lambda x: x["depth_input"].__setitem__((1,0,0), .5)))
        def wrong_order():
            p = out / "bad_order.npz"
            with p.open("xb") as f: np.savez(f, task_ids=np.array([["eval_goal_fake","synthetic_far"]]),
                                            roles=np.array([["backbone_near","backbone_far"]]))
            return rejects(lambda: FrozenOrder(dict(path=str(p), sha256=sha(p)), ds, s["profile"]))
        test("reject_unknown_order_task", wrong_order)
        def bad_auth():
            p = out / "wrong_auth.json"
            write(p, dict(schema=1, scope="DISPOSABLE_SYNTHETIC_ONLY", run_id=s["run_id"], spec_sha256="0"*64,
                          max_updates=1, authorized_by="fixture", approved=True))
            return rejects(lambda: authorize(s, p), "AUTHORIZATION_SCOPE")
        test("reject_wrong_authorization", bad_auth)
        def negative_match():
            from query_minco.training.matching.hungarian import match
            for c in (torch.empty(6,0), torch.zeros(6,7), torch.full((6,1), float("nan"))):
                rejects(lambda: match(c,"B_set"))
            return dict(rejected=3)
        test("Hungarian_negative_cases", negative_match)
        def schedule():
            require(lr_at(0) == 2e-4 and abs(lr_at(2879)-1e-5)<1e-20, "LR_ENDPOINTS")
            require(all(lr_at(i)>=lr_at(i+1) for i in range(2879)), "LR_MONOTONE")
            require(PROFILES["b2c1_reference_v1"]["checkpoints"] == [0,960,1920,2880], "SAVEPOINTS")
            # Pure arithmetic iteration, no model, optimizer or training updates.
            saves = [0] + [i+1 for i in range(2880) if i+1 in [960,1920,2880]]
            require(saves == [0,960,1920,2880], "BUDGET_SIMULATION")
            return dict(simulated_indices=2880, optimizer_updates=0)
        test("reference_budget_LR_savepoints", schedule)
        def cp_exclusive():
            m = initialize(s); opt = optimizer(m)
            save_checkpoint(out, m, opt, s, 0)
            rejects(lambda: save_checkpoint(out, m, opt, s, 0))
            model2 = initialize(s); opt2 = optimizer(model2)
            blob = load_checkpoint(out/"checkpoint_step0000.pt", model2, opt2, s, restore_rng=True)
            require(state_sha(model2)==state_sha(m) and not opt2.state, "ROUNDTRIP_STEP0")
            return dict(strict_roundtrip=True, exclusive=True)
        test("checkpoint_exclusive_and_step0_restore", cp_exclusive)
        def partial_cp():
            p = out / "partial.pt"
            with p.open("xb") as f: f.write(b"incomplete")
            m = initialize(s); opt = optimizer(m)
            return rejects(lambda: load_checkpoint(p,m,opt,s))
        test("reject_partial_checkpoint", partial_cp)

    if a.mode == "numerical":
        from query_minco.training import objective
        from query_minco.training.teacher.sampling import teacher_evaluate
        from query_minco.training.run_spec import ROOT
        def numerical():
            model=initialize(s);model.train();before=state_sha(model)
            batch=ds.load(["synthetic_near","synthetic_far"])
            loss,records,raw,o,costs=objective.forward(model,batch,"B_set","cpu")
            retained=dict(durations=o.parameters.durations,inner_pts=o.parameters.inner_pts,
                          head_pva=o.parameters.head_pva,tail_pva=o.parameters.tail_pva,coefficients=o.trajectory.coeffs)
            for t in retained.values():
                if t.requires_grad:t.retain_grad()
            loss.backward()
            arrays=dict(loss=loss.detach().numpy(),raw=raw.detach().numpy(),raw_grad=raw.grad.numpy())
            arrays.update({k:t.detach().numpy() for k,t in retained.items()})
            arrays.update({"grad_"+k:t.grad.numpy() for k,t in retained.items() if t.grad is not None})
            for i,c in enumerate(costs):
                arrays["cost_"+str(i)]=c.detach().numpy()
                arrays["matched_q_"+str(i)]=np.asarray(records[i]["queries"])
                arrays["matched_t_"+str(i)]=np.asarray(records[i]["teachers"])
                arrays["teacher_coeff_"+str(i)]=batch[i].teacher_coeffs.numpy()
                arrays["teacher_durations_"+str(i)]=batch[i].teacher_durations.numpy()
            for n,p in model.named_parameters():arrays["parameter_grad_"+n]=p.grad.numpy()
            for i,b in enumerate(batch):
                d=b.teacher_durations[0];t=torch.linspace(0,1,32)[None,:]*d.sum()
                for k in range(4): arrays["teacher_sample_%d_%d"%(i,k)]=teacher_evaluate(b.teacher_coeffs[0],d,t,k).numpy()
            with (out/"actual_arrays.npz").open("xb") as f:np.savez(f,**arrays)
            baseline=ROOT/"research_outputs/migration_v1/runs/training_new001/training_arrays.npz"
            comparisons=[]
            with np.load(baseline,allow_pickle=False) as z:
                require(set(arrays)==set(z.files),"BASELINE_FIELDS")
                for k,v in arrays.items():
                    exact=np.array_equal(v,z[k])
                    comparisons.append(dict(name=k,bitwise_equal=exact,max_abs_error=float(np.max(np.abs(v-z[k])))))
            write(out/"COMPARISONS.json",comparisons)
            require(all(x["bitwise_equal"] for x in comparisons),"NUMERICAL_MISMATCH_STOP")
            require(state_sha(model)==before and not torch.count_nonzero(raw.grad[:,:,19:21]),"MODEL_UNCHANGED")
            for i,m in enumerate(records):
                require(not torch.count_nonzero(raw.grad[i,sorted(set(range(6))-set(m["queries"]))]),"UNMATCHED_GRADIENT")
            return dict(bitwise_comparisons=len(comparisons),optimizer_updates=0,unchanged=True)
        test("historical_array_bitwise_regression",numerical)

    if a.mode == "restore":
        def restore():
            m=initialize(s);opt=optimizer(m)
            blob=load_checkpoint(Path(a.trained_run)/"checkpoint_step0001.pt",m,opt,s,restore_rng=True)
            require(blob["completed_optimizer_updates"]==1 and len(opt.state)==len(list(m.parameters())),"RESTORE_OPTIMIZER")
            require(all(int(v["step"])==1 for v in opt.state.values()),"RESTORE_STEPS")
            ref=torch.load(str(Path(a.trained_run)/"checkpoint_step0001.pt"),map_location="cpu",weights_only=True)
            for x,y in zip(opt.state_dict()["state"].values(),ref["optimizer_state_dict"]["state"].values()):
                for k in x:require(torch.equal(x[k],y[k]),"OPTIMIZER_STATE_EXACT")
            return dict(model_sha=state_sha(m),optimizer_restored=True,rng_restored=True,new_updates=0)
        test("trained_checkpoint_state_restore",restore)
    passed=all(x["passed"] for x in results)
    write(out/"RESULT.json",dict(passed=passed,tests=results,mode=a.mode,actual_optimizer_updates=0))
    print(json.dumps(dict(passed=passed,tests=len(results),mode=a.mode)))
    return 0 if passed else 1

if __name__ == "__main__":raise SystemExit(main())
