"""Regression for actual failed progress receipt; mathematical tests are reused unchanged."""
import json
from query_minco.training.physics_only_v1_2_gate1.e1_r3a_v3_logfix import emit_progress
def test_progress_json_flush_is_on_print(capsys):
    payload=dict(sample_id="fixture",center_outside_queries=6,sampled_unknown=882)
    emit_progress(payload)
    assert json.loads(capsys.readouterr().out)==payload
