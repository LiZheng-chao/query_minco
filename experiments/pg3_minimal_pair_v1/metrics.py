"""Read-only normalized quality and initial matched-safe transitions from original evaluator."""
import csv
import numpy as np
import torch
from query_minco.training.io import read,write,require
from query_minco.training.pg3_next_exact_v1.quality import quality

def supplement(folder,split,step):
    prefix=split+'_step%04d'%step
    with np.load(folder/(prefix+'.npz'),allow_pickle=False) as z:
        q=quality(torch.from_numpy(z['coeffs']).double(),torch.from_numpy(z['durations']).double())
        values={k:v.detach().numpy() for k,v in q.items()}
    with (folder/(prefix+'.csv')).open() as f:rows=list(csv.DictReader(f))
    require(np.isfinite(np.concatenate([v.reshape(-1) for v in values.values()])).all(),'MINPAIR_QUALITY_NONFINITE')
    summary=read(folder/(prefix+'_SUMMARY.json'));result=dict(original=summary,normalized_quality={k:dict(mean=float(v.mean()),p50=float(np.median(v)),p95=float(np.quantile(v,.95))) for k,v in values.items()},quality_semantics='native dynamic_quality_loss; SI integrals and normalized QA/QJ/QT; not tracking errors')
    if step:
        with (folder/(split+'_step0000.csv')).open() as f:initial=list(csv.DictReader(f))
        require([(x['sample_id'],x['query']) for x in initial]==[(x['sample_id'],x['query']) for x in rows],'MINPAIR_EVAL_PAIR')
        result['transitions']=dict(pass_to_fail=sum(a['hard_pass']=='True' and b['hard_pass']!='True' for a,b in zip(initial,rows)),fail_to_pass=sum(a['hard_pass']!='True' and b['hard_pass']=='True' for a,b in zip(initial,rows)),initial_matched_safe_degraded=sum(a['hard_pass']=='True' and a['matched']=='True' and b['hard_pass']!='True' for a,b in zip(initial,rows)))
    write(folder/(prefix+'_QUALITY_TRANSITIONS.json'),result)
