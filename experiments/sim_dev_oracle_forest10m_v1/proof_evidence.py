"""Report observed motion and proof history independently of the verdict."""
import json
import math
from pathlib import Path

import numpy as np


def rows(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def observed(run, result):
    episode = Path(run)/'none'
    events = rows(episode/'supervisor_data/EVENTS.jsonl')
    audit = [e.get('payload',{}) for e in events if e.get('kind')=='t2_planner_audit']
    goal = [e['goal'] for e in audit if e.get('event')=='GOAL_RECEIVED']
    candidates = {e['plan_id']:e for e in audit if e.get('event')=='CANDIDATE_GENERATED'}
    approvals = [e['oracle'] for e in audit if e.get('event')=='ORACLE_PREFIX_COMMITTED']
    commands = {}
    for e in events:
        p=e.get('payload',{})
        if e.get('kind')=='controller_command' and e.get('accepted') and p.get('plan_id'):
            k=p['command_key']
            commands[k['trajectory_id'],k['input_ns']]=p['plan_id']
    consumption=[]
    for e in rows(episode/'controller/CONSUMPTION.jsonl'):
        if e.get('kind')=='COMMAND_CONSUMED':
            pid=commands.get((e['trajectory_id'],e['input_ns']))
            if pid:
                consumption.append(dict(plan_id=pid,ros_time=e['received_ros'],trajectory_id=e['trajectory_id']))
    switches=[]
    for e in consumption:
        if not switches or switches[-1]['plan_id']!=e['plan_id']:
            switches.append(e)
    directions=[]
    if goal:
        for pid,e in candidates.items():
            local=e.get('planner',{}).get('local_goal')
            if local is None:
                continue
            # Direction from the Depth-associated pose when available, otherwise
            # report the actual re-anchored Plan direction and label the origin.
            p=np.asarray(e.get('planner',{}).get('search_origin',e['plan']['origin']))
            u=np.asarray(local)-p
            v=np.asarray(goal[-1])-p
            yaw=math.degrees(math.atan2(u[1],u[0])-math.atan2(v[1],v[0]))
            directions.append(dict(plan_id=pid,origin=p.tolist(),local_goal=local,
                                   global_goal=goal[-1],yaw_delta_deg=(yaw+180)%360-180,
                                   origin_kind='depth_pose' if 'search_origin' in e.get('planner',{}) else 'reanchored_plan_origin'))
    evidence=result.get('execution_evidence') or {}
    return dict(ever_armed=any(e.get('kind')=='plant' and e.get('accepted')
                               and e.get('payload',{}).get('phase')=='ARMED'
                               and e['payload'].get('integration_steps',0)>0 for e in events),
                recorded_candidates=len(candidates),guardian_approvals=len(approvals),
                guardian_plan_ids=sorted({p['plan_id'] for p in approvals}),
                controller_consumed_plan_ids=sorted({e['plan_id'] for e in consumption}),
                controller_consumption_count=len(consumption),
                controller_to_plant_chain_count=len(evidence.get('chains',[])),
                accepted_plan_ids=evidence.get('accepted_plan_ids',[]),
                execution_status=evidence.get('status'),execution_failure=evidence.get('reason'),
                switches=switches,directions=directions,
                proof_windows=[{k:p[k] for k in ('plan_id','issued','expires','source_stamp')} for p in approvals])
