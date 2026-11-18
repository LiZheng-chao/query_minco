"""Read-only B0 frozen-artifact check; never grants execution permission."""
from pathlib import Path
import json,hashlib,argparse
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/forest_dev_sparse_seed3_dist6_fixedbirth_v1'
def main():
 p=argparse.ArgumentParser();p.add_argument('--check-only',required=True,action='store_true');p.parse_args()
 b=json.loads((O/'B0_SOURCE_BINDING.json').read_text());bad=[name for name,expected in b['sha256'].items() if not Path(name).is_file() or hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected]
 if bad:raise SystemExit('SOURCE_DRIFT '+repr(bad))
 result=json.loads((O/'B0_OUTSIDE_MINCO_RESULT.json').read_text());assert result['certified_outside_plans']==0
 print('SOURCE_BINDING_MATCH_NO_ROS_NO_ARM')
 print('B0_PHYSICAL=NO_GO: INSUFFICIENT_OBSERVED_SPACE; SAME_VERSION_S_C7_NOT_RUN')
if __name__=='__main__':main()
