"""Read frozen train IDs only; emit new adapted manifests, never change old caches."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from query_minco.training.io import write, read, sha, require
from query_minco.training.qualification import qualify

def bind(p):
    return dict(path=str(p.resolve()),sha256=sha(p))

def main():
    p=argparse.ArgumentParser();p.add_argument("--historical-root",required=True)
    p.add_argument("--preflight-root",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();root=Path(a.historical_root);pf=Path(a.preflight_root);out=Path(a.output);out.mkdir()
    # Explicit manifest schemas; never import an old Python module or glob training samples.
    replay={r["id"]:r for r in read(pf/"REPLAY_MANIFEST.json")}
    orth={r["id"]:r for r in read(pf/"DATASET_MANIFEST.json")["rows"] if r["block"]=="train_orth"}
    evidence=[bind(pf/n) for n in ["DATASET_MANIFEST.json","REPLAY_MANIFEST.json","PROTOCOL_SEAL.json",
                                   "TEACHER_DATA_GATE.json","DATA_CONTRACT_GATE.json"]]
    evidence.append(bind(root/"query_minco/outputs/method_provenance_r0_v1/DATA_BINDINGS.json"))
    records=[];manifests={}
    for arm in ("C0_replay","C1_orthogonal"):
        orderfile=pf/"orders"/(arm+"_seed42.npz")
        with np.load(orderfile,allow_pickle=False) as z: ids=z["task_ids"].copy();roles=z["roles"].copy()
        task_roles={}
        for sid,role in zip(ids.ravel().tolist(),roles.ravel().tolist()):
            task_roles.setdefault(sid,set()).add(role)
        rows=[]
        for sid,allowed in task_roles.items():
            row=orth[sid] if sid in orth else replay[sid]
            if sid in orth:
                tensor=pf/row["npz"]; expected=row["npz_sha256"]
            else:
                tensor=root/row["npz_path"]; expected=row["npz_sha"]
            require(sha(tensor)==expected,"ORIGINAL_CACHE_SHA")
            with np.load(tensor,allow_pickle=False) as z: state=z["state"].copy()
            regime="near" if np.linalg.norm(state[6:9].astype(float))<=3.4 else "far"
            cert=tensor.parent/"teacher_checks.json"
            rows.append(dict(id=sid,artifact=dict(path=str(tensor),sha256=expected),split="train",
                             regime=regime,M=row["M"],roles=sorted(allowed),teacher_status="verified_positive",
                             certificates=[bind(cert)]))
        manifest=out/(arm+"_manifest.json")
        write(manifest,dict(schema=1,qualification="TRAIN",roots=[
            str(pf),str(root/"query_minco/outputs/p2c2c1a_teacher_v1"),str(root/"query_minco/outputs/method_provenance_r0_v1")],rows=rows,evidence=evidence))
        manifests[arm]=bind(manifest)
        for seed in (42,43,44):
            orderfile=pf/"orders"/("%s_seed%d.npz"%(arm,seed))
            spec=dict(profile="b2c1_reference_v1",experiment_arm=arm,seed=seed,
                      dataset=bind(manifest),order=bind(orderfile))
            ds,order,gate=qualify(spec)
            back=Counter(order.ids[np.isin(order.roles,["backbone_near","backbone_far"])].tolist())
            require(len(back)==192 and set(back.values())=={60},"BACKBONE_EXPOSURE")
            records.append(dict(seed=seed,arm=arm,**gate))
    pairing=[]
    for seed in (42,43,44):
        loaded=[]
        for arm in ("C0_replay","C1_orthogonal"):
            sp=dict(profile="b2c1_reference_v1",experiment_arm=arm,seed=seed,dataset=manifests[arm],
                    order=bind(pf/"orders"/("%s_seed%d.npz"%(arm,seed))))
            ds,order,_=qualify(sp)
            cache={s.sample_id:s for s in ds.load(list(ds.rows))}
            loaded.append((order,cache))
        (c0,d0),(c1,d1)=loaded
        require(np.array_equal(c0.roles,c1.roles),"PAIRED_ROLES")
        for a0,a1,role in set(zip(c0.ids.ravel().tolist(),c1.ids.ravel().tolist(),c0.roles.ravel().tolist())):
            x,y=d0[a0],d1[a1]
            require(len(x.teacher_coeffs)==len(y.teacher_coeffs) and
                    np.array_equal(x.depth_input.numpy(),y.depth_input.numpy()) and
                    np.array_equal(np.r_[x.state.numpy()[:6],x.state.numpy()[9:]],
                                   np.r_[y.state.numpy()[:6],y.state.numpy()[9:]]),"PAIRED_INPUTS")
            if role!="supplement_near":require(a0==a1,"NONINTERVENTION_ID_CHANGED")
        pairing.append(dict(seed=seed,passed=True))
    write(out/"RESULT.json",dict(passed=True,orders=records,pairing=pairing,manifests=manifests,
                                model_forward_calls=0,optimizer_updates=0,eval_or_blind_npz_reads=0,
                                original_teacher_regeneration_verified=False))
    print("QUALIFIED",len(records),"orders; forward=0 optimizer=0")
if __name__=="__main__":main()
