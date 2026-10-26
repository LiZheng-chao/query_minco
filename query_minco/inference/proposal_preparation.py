"""Preserve inherited cell expiry when preparing the existing stop bank.
Small lifetime adapter, not a new planner. Used in CPU replay first.
"""
import math
from query_minco.inference.committed_reference import prepare_reference

def prepare_bound_reference(prefix,volume,now_ros,now_mono,activation_lead=.06):
    if not volume.cells:raise ValueError('NO_CONTINUOUS_CELL')
    if not .02<=activation_lead<=.1:raise ValueError('ACTIVATION_LEAD')
    # Reduce the requested lifetime, never extend or rewrite geometry evidence.
    until=min(cell.valid_until for cell in volume.cells)
    valid_for=min(120.,until-now_ros-.001)
    if not math.isfinite(valid_for) or valid_for<1:raise ValueError('INHERITED_CELL_EXPIRED')
    bundle=prepare_reference(prefix,volume,now_ros,now_mono,valid_for=valid_for)
    horizon=max(row.phase+row.polynomial.duration for row in bundle.stops)+activation_lead
    if horizon>valid_for:raise ValueError('FULL_STOP_EXCEEDS_INHERITED_LIFETIME')
    return bundle
