"""Versioned K1 interface for fixed-time P0c/L0 qualification; no optimizer or controller."""
from query_minco.models import QueryMincoPlanner
from query_minco.models.config import QueryMincoConfig
from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
from .roi_safe_training import RoiSafeTrainingObjective
from .hard_validator import COMPONENT_ID as VALIDATOR_ID

METHOD_ID = "physics_v12_k1_raw21_A_p0c_L0_interface_v1"
ARCHITECTURE_ID = "query_minco_k1_decoder2_raw21_v1"
DECODER_ID = "raw21_absolute_fixed2s_three_segment_k1_v1"
OBJECTIVE_ID = "physics_v12_phase_a_L0_K_normalized_v1"
DEAD_RAW_SLICE = (15, 21)


def make_planner():
    model = QueryMincoPlanner(QueryMincoConfig(num_queries=1))
    if model.config.architecture_id != ARCHITECTURE_ID:
        raise ValueError("K1_ARCHITECTURE_BINDING")
    return model


def make_objective():
    objective = RoiSafeTrainingObjective(
        PhysicsV12Spec(phase="A", aggregation="foundation"), num_candidates=1)
    objective.method_id = METHOD_ID
    objective.component_id = OBJECTIVE_ID
    return objective


def identity():
    return dict(method_id=METHOD_ID, architecture_id=ARCHITECTURE_ID,
                decoder_id=DECODER_ID, objective_id=OBJECTIVE_ID,
                validator_id=VALIDATOR_ID, raw_dim=21, candidates=1,
                total_time_s=2., segments=3, formal_training_authorized=False)
