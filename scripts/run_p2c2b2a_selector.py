"""Explicit isolated stages, never auto-enter another development phase."""
import argparse,importlib
from pathlib import Path
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--stage",choices=["prepare","candidates","train","evaluate"],required=True)
 p.add_argument("--experiment-dir",type=Path,required=True);p.add_argument("--dataset-dir",type=Path)
 a=p.parse_args();module=importlib.import_module("query_minco.p2c2b2a."+a.stage)
 if a.stage in ["prepare","candidates"]:
  if a.dataset_dir is None:p.error("--dataset-dir required")
  getattr(module,"prepare" if a.stage=="prepare" else "generate")(a.experiment_dir,a.dataset_dir)
 else:getattr(module,a.stage)(a.experiment_dir)
