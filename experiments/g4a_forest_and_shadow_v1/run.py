from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.training.k1_forest_v1.cli import main
if __name__=='__main__':main()
