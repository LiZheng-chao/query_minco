"""Explicit frozen-cache rows; caller supplies roots, ids, and immutable manifest."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
def load_samples(ids,rows,root):
 result=[]
 for sid in ids:
  row=rows[sid];p=Path(root)/row["path"]
  if hashlib.sha256(p.read_bytes()).hexdigest()!=row["sha256"]:raise ValueError("DATA_SHA "+sid)
  with np.load(p,allow_pickle=False) as z:
   if len(z["teacher_coeffs"])==0:raise ValueError("M0_NOT_TRAINABLE")
   result.append(SimpleNamespace(sample_id=sid,**{k:torch.from_numpy(z[k].copy()) for k in ["depth_input","state","teacher_coeffs","teacher_durations"]}))
 return result
