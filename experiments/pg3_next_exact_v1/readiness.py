"""Read-only frozen PG2 pipeline and fail-closed native observation tests; no ROS publication."""
import importlib.util,time,copy
import numpy as np
import torch
from query_minco.training.io import read,write,require
from query_minco.inference.identity import state_sha
from query_minco.inference.planner_adapter import InferenceAdapter
from query_minco.inference.online_selector import select as online_select
from query_minco.inference.observed_space import DepthHistory,DepthFrame
from query_minco.inference.shadow import K
from query_minco.trajectory import TrajectoryPostprocessor
from query_minco.training.pg3_next_exact_v1.quality import quality,select
from query_minco.evaluation.validation import validate_teacher
from query_minco.evaluation.goal import task_metrics
from query_minco.evaluation.validation import evaluate
from experiments.pg3_feasible_mode_v1.common import model_from
from .common import runpath

def run(out,context):
    folder=out/'READINESS';folder.mkdir(exist_ok=False);ds,geo,init,q,train,dev=context;model=model_from(init);blob=torch.load(runpath('A')/'commits/step0128/checkpoint.pt',weights_only=True);model.load_state_dict(blob['model']);model.eval();before=state_sha(model);rows=[]
    for sample,field,meta in train[:8]:
        tick=time.perf_counter()
        with torch.no_grad():
            raw=model(sample.depth_input[None],sample.state[None]);o=TrajectoryPostprocessor()(raw,sample.state[None],num_samples=65);export={k:v[0] for k,v in InferenceAdapter.export(o,raw,o.samples).items()};qq=quality(o.trajectory.coeffs,o.parameters.durations)
        offline=[]
        for qi in range(6):
            c=o.trajectory.coeffs[0,qi].numpy();d=o.parameters.durations[0,qi].numpy();hard=validate_teacher(c,d,o.parameters.head_pva[0,qi].numpy(),o.parameters.tail_pva[0,qi].numpy(),o.parameters.inner_pts[0,qi].numpy(),meta['scene']);goal=task_metrics(evaluate(c[-1],d[-1]),evaluate(c[-1],d[-1],1),sample.state[6:9].numpy())
            offline.append(dict(query_id=qi,solve_valid=bool(o.trajectory.solve_valid[0,qi]),hard_physics_pass=hard['passed'],goal_pass=goal['passed'],**{k:float(v[0,qi]) for k,v in qq.items()}))
        chosen,reason=select(offline);depth=sample.depth_input.numpy().copy();state=sample.state.numpy().copy();fresh=online_select(depth,state,export,0.);stale=online_select(depth,state,export,1.);future=online_select(depth,state,export,-.1);bad=copy.deepcopy(export);bad['solve_valid'][:]=False;invalid=online_select(depth,state,bad,0.)
        history=DepthHistory();history.add(DepthFrame(1.,depth,np.zeros(3),np.eye(3),'B0'),K);unknown=history.diagnostics(np.zeros((1,3)),1.);old=history.diagnostics(np.zeros((1,3)),2.)
        require(all(x['selected_query'] is None for x in (fresh,stale,future,invalid)),'NEXT_ONLINE_GATE_BYPASSED')
        rows.append(dict(sample_id=sample.sample_id,offline_selected=chosen,offline_reason=reason,offline_scope='OFFLINE_GT_SELECTOR_UPPER_BOUND',offline_reference_available=chosen is not None,execution_permission=False,fresh=fresh,stale=stale,future=future,invalid_solve=invalid,native_unknown=unknown,expired_depth=old,diagnostic_latency_s=time.perf_counter()-tick,latency_scope='offline CPU end-to-end diagnostic, not control tracking'))
    require(state_sha(model)==before,'NEXT_READINESS_WEIGHT_MUTATION')
    availability={name:importlib.util.find_spec(name) is not None for name in ('rospy','rclpy','pybullet','mujoco','airsim')}
    write(folder/'RESULT.json',dict(status='BLOCKED_FOR_FOREST_CLOSED_LOOP',offline_pipeline='VERIFIED',all_fail_closed=True,rows=rows,frozen_model_sha256=before,available_modules=availability,observed_volume='BLOCKED: native rays do not certify swept body volume',state_synchronization='BLOCKED: no actual middleware synchronization qualification in this run',control_tracking='NOT_AVAILABLE',control_simulator='NOT_AVAILABLE: no defined and verified controller/dynamics chain',RMS_position_tracking_error='NOT_AVAILABLE',P95_position_error='NOT_AVAILABLE',peak_velocity_error='NOT_AVAILABLE',command_saturation_ratio='NOT_AVAILABLE',replanning_discontinuity='NOT_AVAILABLE',control_latency='NOT_AVAILABLE',ROS_publication='NOT_RUN',forest_closed_loop='NOT_RUN',flight='NOT_RUN',shadow_readiness='BLOCKED_FOR_EXECUTION; fail-closed diagnostic tested'))
