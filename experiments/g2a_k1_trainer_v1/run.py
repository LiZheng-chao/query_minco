"""Thin entry resolving only the authorized mainline package."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from query_minco.training.k1_physics_trainer_v1.cli import main
if __name__=="__main__":
    main()
