"""Fail-closed nonphysical preparation. No ROS or Plant imports."""
from pathlib import Path
import argparse,json,hashlib
R=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true',required=True);p.parse_args()
 c=json.loads(Path(__file__).with_name('config.json').read_text())
 assert len(c['goal'])==3 and c['limits']==[1.,8.,100.]
 print('PREFLIGHT_SOURCE_PASS_PREREQUISITE_PENDING_NO_ROS_NO_ARM')
 print('WAIT_RECOVERY_HANDSHAKE_NOT_INTEGRATED; FOREST10M_RUNTIME_NOT_SEALED; LIVE_ROS_TOPICS_NOT_VERIFIED')
if __name__=='__main__':main()
