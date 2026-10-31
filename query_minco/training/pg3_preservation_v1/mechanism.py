"""Disposable TRAIN-only update alternative; no loss or optimizer implementation."""
import numpy as np
import torch

def local_direction(pack,adam,selected):
    if adam.ndim!=1 or not bool(torch.isfinite(adam).all()) or float(adam.norm())==0:raise ValueError('NO_FINITE_NONZERO_ADAM')
    rows=sorted(set(selected))
    if not rows or any(type(x)!=int or not 0<=x<6 for x in rows):raise ValueError('NO_DANGEROUS_SELECTED_QUERY_ROWS')
    mask=torch.zeros_like(adam,dtype=torch.bool);offset=0;found=0
    for (name,param),part in zip(pack.named,pack.parts):
        n=int(part.sum())
        if name=='query_decoder.learned_queries':
            if tuple(param.shape)!=(6,128):raise ValueError('QUERY_EMBEDDING_CONTRACT')
            m=torch.zeros_like(param,dtype=torch.bool);m[rows]=True;mask[offset:offset+n]=m[part];found+=1
        offset+=n
    if offset!=len(adam) or found!=1:raise ValueError('PACK_ORDER_OR_QUERY_IDENTITY')
    d=adam*mask
    if float(d.norm())<=1e-15:raise ValueError('NO_PROGRESS_IN_SELECTED_SUBSPACE')
    return d*(adam.norm()/d.norm()),dict(selected_rows=rows,norm_matched=True,global_rows_couple_across_samples=True,self_attention_couples_queries=True,new_loss=False)

def witness(passed,goals,signatures):
    p=np.asarray(passed,bool);g=np.asarray(goals,bool)
    if p.shape!=g.shape or p.ndim!=2 or p.shape[1]!=6 or len(signatures)!=len(p):raise ValueError('LABEL_SHAPE')
    groups=[]
    for b in range(len(p)):
        if len(signatures[b])!=6:raise ValueError('SIGNATURE_SHAPE')
        known={};unknown=[]
        for q in range(6):
            if not (p[b,q] and g[b,q]):continue
            sig=signatures[b][q]
            if sig in (None,'UNKNOWN',''):unknown.append(q)
            else:known.setdefault(sig,[]).append(q)
        groups.append(dict(known=known,unknown=unknown))
    return groups

def retained(groups,passed,goals,signatures):
    p=np.asarray(passed,bool);g=np.asarray(goals,bool)
    if p.shape!=g.shape or p.shape!=(len(groups),6):raise ValueError('LABEL_SHAPE')
    for b,row in enumerate(groups):
        if any(not(p[b,q] and g[b,q]) for q in row['unknown']):return False
        for sig in row['known']:
            if not any(p[b,q] and g[b,q] and signatures[b][q]==sig for q in range(6)):return False
    return True

def superior(candidate,control):
    # No success from zero steps, loss/cosine improvement or reduced norm.
    if candidate is None or control is None:return False
    return bool(candidate['accepted'] and control['accepted'] and candidate['relative_update_norm']>1e-6 and candidate['pre_fail_to_pass']>control['pre_fail_to_pass'] and candidate['pre_safe_pass_to_fail']<=control['pre_safe_pass_to_fail'])
