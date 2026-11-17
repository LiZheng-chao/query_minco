from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.stop_latency_fix_v1.manual import main
if __name__=='__main__':
 if Path.cwd().resolve()!=ROOT:raise RuntimeError('EXPECTED_REPOSITORY_CWD')
 main(config_root=ROOT/'research_outputs/depth_fault_contract_20261117_v1',binding_name='NEW_SOURCE_BINDING.json')
