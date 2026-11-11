from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from query_minco.audit.forest128_acceptance_v1 import audit
if __name__=='__main__':
 torch.set_num_threads(1)
 r=Path(__file__).resolve().parents[2]
 audit(r,r/'research_outputs/g4b_formal128_and_control_20261111_v1')
