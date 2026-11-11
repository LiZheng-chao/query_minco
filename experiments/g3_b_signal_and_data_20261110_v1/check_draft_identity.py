from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.training.k1_physics_trainer_v1 import trainer as T
R=Path.cwd();O=R/'research_outputs/g3_b_signal_and_data_20261110_v1'
d=json.loads((O/'RUNSPEC_DRAFT.json').read_text())
binding=json.loads(Path(d['dataset_binding_path']).read_text())
assert T.sha(d['dataset_binding_path'])==d['dataset_binding_sha256']
assert T.sha(binding['index_path'])==binding['index_sha256'] and T.sha(binding['registry_path'])==binding['registry_sha256']
for name,h in binding['source_sha256'].items():assert T.sha(name)==h
for row in binding['samples']:assert T.sha(row['manifest'])==row['manifest_sha256']
assert d['formal_training_authorized'] is False
try:T.verify_spec(O/'RUNSPEC_DRAFT.json')
except ValueError as exc:assert str(exc)=='RUNSPEC_SCHEMA'
else:raise AssertionError('DRAFT_MUST_NOT_RUN_WITH_LEGACY_TRAINER')
T.write(O/'DRAFT_IDENTITY_CHECK.json',dict(status='PASS',dataset_source_join_verified=True,legacy_trainer_rejects_draft=True,draft_sha256=T.sha(O/'RUNSPEC_DRAFT.json'),new_network_updates=0))
print('PASS: dataset/source join exact; preparation draft rejected by legacy trainer schema; no model/optimizer/update')
