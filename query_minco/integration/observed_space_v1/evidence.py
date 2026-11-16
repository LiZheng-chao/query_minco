"""Perception qualification never imports old FREE voxels or revives expired frames."""
import json
from pathlib import Path
from query_minco.evaluation.mirror_holdout_v1 import sha
def verify_e0(path):
 p=Path(path);d=json.loads(p.read_text())
 if d.get('status')!='STATIC_PASS' or d.get('bootstrap_status')!='STATIC_PASS' or d.get('live_native_sensor') is not True or d.get('frames',0)<10:raise PermissionError('E0_SENSOR_AND_BOOTSTRAP_REQUIRED')
 for name,h in d.get('module_sha256',{}).items():
  if sha(name)!=h:raise PermissionError('E0_COMPONENT_SHA')
 expected={str(Path(__file__).resolve().parent/name) for name in ['map.py','space.py']}
 if set(d.get('module_sha256',{}))!=expected:raise PermissionError('E0_COMPONENT_SET')
 for record in d.get('records',[]):
  if sha(record['raw_file'])!=record['raw_sha256']:raise PermissionError('E0_RAW_FRAME_SHA')
 if len(d.get('records',[]))<10:raise PermissionError('E0_RAW_PROVENANCE_REQUIRED')
 owner=json.loads((p.parent/'OWNER_RESULT.json').read_text())
 if owner.get('status')!='STATIC_PASS' or owner.get('bootstrap_status')!='STATIC_PASS' or owner['children']['collector']['actual_exit_code']!=0:raise PermissionError('E0_OWNER_TERMINAL')
 return dict(scope='component/initial-observation qualification only;execution needs fresh live bootstrap again',certificate_sha256=sha(p))
