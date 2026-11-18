"""Fail-closed F0 execution evidence. Pure event reduction; no ROS or writes."""
import math
import hashlib
import json

COMPONENT_ID = 'f0_execution_evidence_v1'

def _legacy_assess(events, begin, end, reference_plans=()):
    """Separate admitted plans, SO3 consumption, and unavailable plan lineage.

    Existing native SO3 messages discard PositionCommand identity. Temporal
    overlap must never be promoted to a per-plan Controller consumption proof.
    """
    result = dict(component_id=COMPONENT_ID, status='FAIL', accepted_plan_ids=[],
                  switches=[], consumed_so3_stamps=[], controller_plan_lineage='UNKNOWN')
    if not all(math.isfinite(x) for x in (begin, end)) or begin > end:
        result['reason'] = 'INVALID_PHYSICAL_WINDOW'
        return result
    rows = [e for e in events if isinstance(e.get('monotonic'), (int, float))
            and begin <= e['monotonic'] <= end and e.get('accepted') is True
            and e.get('mode') == 'ARMED']
    if any(b['monotonic'] < a['monotonic'] for a, b in zip(rows, rows[1:])):
        result['reason'] = 'EVENT_ORDER_ROLLBACK'
        return result
    trajectory_ids = set()
    for e in reference_plans:
        record = e.get('plan')
        if e.get('kind') == 'plan' and isinstance(record, dict):
            identity = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            if identity == e.get('plan_id'):
                trajectory_ids.add(identity)
    commands = []
    for e in rows:
        d = e.get('payload', {})
        proof = d.get('authorization') or {}
        identity = d.get('plan_id')
        if (e.get('kind') == 'controller_command' and isinstance(identity, str)
                and identity in trajectory_ids and len(identity) == 64 and all(c in '0123456789abcdef' for c in identity)
                and proof.get('plan_id') == identity
                and proof.get('geometry') == 'FINITE_BIRTH_FULL_TUBE_129_WITH_OCCUPIED_VETO'
                and proof.get('certificate_expiry', -math.inf) >= e.get('ros_now', math.inf)):
            commands.append(e)
    previous = None
    for e in commands:
        identity = e['payload']['plan_id']
        if identity not in result['accepted_plan_ids']:
            result['accepted_plan_ids'].append(identity)
        if previous is not None and identity != previous:
            result['switches'].append(dict(before=previous, after=identity,
                                           monotonic=e['monotonic']))
        previous = identity
    finals = {e['stamp']: e for e in rows if e.get('kind') == 'final_so3'
              and isinstance(e.get('stamp'), (int, float)) and math.isfinite(e['stamp'])}
    for e in rows:
        d = e.get('payload', {})
        stamp = d.get('final_command_stamp')
        final = finals.get(stamp)
        if (e.get('kind') == 'plant' and d.get('phase') == 'ARMED'
                and d.get('state_source') == 'DYNAMICS_INTEGRATED'
                and d.get('integration_steps', 0) > 0 and d.get('command_valid') is True
                and final is not None and final['monotonic'] <= e['monotonic']):
            if stamp not in result['consumed_so3_stamps']:
                result['consumed_so3_stamps'].append(stamp)
    if len(result['accepted_plan_ids']) < 2 or not result['switches']:
        result['reason'] = 'TWO_ADMITTED_PLANS_AND_SWITCH_REQUIRED'
    elif not result['consumed_so3_stamps']:
        result['reason'] = 'PLANT_SO3_CONSUMPTION_REQUIRED'
    else:
        result.update(status='UNKNOWN', reason='NATIVE_CONTROLLER_PLAN_LINEAGE_UNAVAILABLE')
    return result


def assess(events, begin, end, reference_plans=(), controller_rows=None, expected_nonce=None, controller_identity=None):
    """v2: exact identity joins; a temporal overlap never establishes consumption."""
    if controller_rows is None:
        return _legacy_assess(events, begin, end, reference_plans)
    result=dict(component_id='f0_controller_consumption_v1',status='FAIL',
                controller_plan_lineage='UNKNOWN',accepted_plan_ids=[],switches=[],chains=[])
    def reject(reason):
        result['reason']=reason
        return result
    try:
        if not all(math.isfinite(x) for x in [begin,end]) or begin>end:return reject('INVALID_WINDOW')
        if not expected_nonce:return reject('RUN_NONCE_REQUIRED')
        if len(controller_rows)<3:return reject('CONTROLLER_LOG_MISSING')
        opening,closing=controller_rows[0],controller_rows[-1]
        if opening.get('kind')!='AUDIT_OPEN' or opening.get('nonce')!=expected_nonce:return reject('CONTROLLER_RUN_IDENTITY')
        if (closing.get('kind')!='AUDIT_CLOSE' or closing.get('lost')!=0
                or closing.get('write_failed') is not False):return reject('CONTROLLER_LOG_INCOMPLETE_OR_LOST')
        records=controller_rows[1:-1]
        if [e.get('index') for e in records]!=list(range(1,len(records)+1)) or closing.get('records')!=len(records):return reject('CONTROLLER_LOG_SEQUENCE')
        if (not controller_identity or controller_identity.get('verified') is not True
                or controller_identity.get('pid')!=opening.get('pid')
                or controller_identity.get('nonce')!=expected_nonce):return reject('LOADED_CONTROLLER_IDENTITY_REQUIRED')
        plans={}
        for e in reference_plans:
            rec=e.get('plan')
            if (e.get('kind')=='plan' or (e.get('kind')=='t2_planner_audit' and e.get('payload',{}).get('event')=='CERTIFIED_COMMITTED')) and isinstance(rec,dict):
                identity=hashlib.sha256(json.dumps(rec,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                if identity!=e.get('plan_id'):return reject('REFERENCE_PLAN_HASH')
                plans[identity]=rec
        rows=[e for e in events if begin<=e.get('monotonic',-math.inf)<=end and e.get('accepted') is True and e.get('mode')=='ARMED']
        if any(b['monotonic']<a['monotonic'] for a,b in zip(rows,rows[1:])):return reject('SUPERVISOR_EVENT_ORDER')
        # Include pre-ARM commands for global uniqueness; they cannot count as plans.
        commands={}
        for e in events:
            if e.get('kind')!='controller_command' or not e.get('accepted'):continue
            d=e['payload'];key=d.get('command_key')
            if key is None:continue
            if key['nonce']!=expected_nonce:return reject('SUPERVISOR_RUN_IDENTITY')
            tid=key['trajectory_id']
            if tid in commands or not isinstance(tid,int) or not 0<tid<2**32:return reject('DUPLICATE_OR_INVALID_COMMAND_ID')
            commands[tid]=e
        consumed={};last_command=0;last_output=0
        outputs=[]
        for e in records:
            tid=e['trajectory_id']
            if e['kind']=='COMMAND_CONSUMED':
                if tid in consumed or e['command_index']!=last_command+1:return reject('DUPLICATE_CONTROLLER_CONSUMPTION')
                last_command=e['command_index'];source=commands.get(tid)
                if source is None:return reject('UNAUTHORIZED_COMMAND')
                d=source['payload'];key=d['command_key']
                if e['input_ns']!=key['input_ns'] or e['pva']!=[x for a in d['pva'] for x in a] or e['yaw']!=d['yaw'] or e['yaw_dot']!=d['yaw_dot']:return reject('COMMAND_CONTENT_OR_ID_MISMATCH')
                issued=e['input_ns']//1000000000+(e['input_ns']%1000000000)/1e9
                if not -.01<=e['received_ros']-issued<=.2:return reject('EXPIRED_CONTROLLER_COMMAND')
                consumed[tid]=e
            elif e['kind']=='CONTROL_OUTPUT':
                if e['output_index']!=last_output+1:return reject('OUTPUT_SEQUENCE')
                last_output=e['output_index']
                if tid==0:
                    if e['command_index']!=0:return reject('FALSE_STARTUP_HOLD')
                    continue
                c=consumed.get(tid)
                if c is None or c['command_index']!=last_command or e['command_index']!=last_command:return reject('OUTPUT_WRONG_ACTIVE_COMMAND')
                if any(e[k]!=c[k] for k in ['input_ns','pva','yaw','yaw_dot','kx','kv']):return reject('OUTPUT_COMMAND_SNAPSHOT_CONFLICT')
                if not c['monotonic']<=e['monotonic']:return reject('CONTROLLER_CAUSAL_ORDER')
                outputs.append(e)
            else:return reject('UNKNOWN_CONTROLLER_RECORD')
        finals={}
        for e in rows:
            if e.get('kind')=='final_so3':
                ns=e['payload'].get('stamp_ns')
                if ns in finals:return reject('DUPLICATE_FORWARDED_OUTPUT')
                finals[ns]=e
        used=set()
        for e in outputs:
            if e['output_ns'] in used:return reject('DUPLICATE_OUTPUT_TIMESTAMP')
            used.add(e['output_ns']);source=commands[e['trajectory_id']];d=source['payload'];pid=d.get('plan_id');proof=d.get('authorization') or {}
            if source not in rows or pid not in plans:continue  # startup/hold cannot count
            if proof.get('geometry')=='OBSERVED_SPACE_FULL_TUBE_129':
                error=validate_observed_authorization(d,plans[pid])
                if error:return reject(error)
            elif proof.get('plan_id')!=pid or proof.get('geometry')!='FINITE_BIRTH_FULL_TUBE_129_WITH_OCCUPIED_VETO':return reject('PLAN_AUTHORIZATION_MISMATCH')
            output_ros=e['output_ns']//1000000000+(e['output_ns']%1000000000)/1e9
            if proof.get('certificate_expiry',-math.inf)<max(e['received_ros'],output_ros):return reject('EXPIRED_AUTHORIZATION')
            final=finals.get(e['output_ns'])
            if final is None:continue
            fd=final['payload']
            if proof.get('geometry')=='OBSERVED_SPACE_FULL_TUBE_129':
                current=fd.get('observed_authority') or {}
                if any(current.get(k)!=d.get(k) for k in ['plan_id','map_epoch','snapshot_id','certificate_id','acceptance_id']):return reject('OBSERVED_FORWARD_AUTHORITY_CONFLICT')
            if e['force']!=fd['force_world'] or e['orientation']!=fd['q'] or e['kr']!=fd['kR'] or e['kom']!=fd['kOm']:return reject('SO3_OUTPUT_CONTENT_MISMATCH')
            if e['aux']!=fd.get('audit_aux'):return reject('SO3_AUX_MISMATCH')
            if final['monotonic']<consumed[e['trajectory_id']]['monotonic']:return reject('FORWARD_BEFORE_CONSUMPTION')
            plants=[x for x in rows if x.get('kind')=='plant' and x['payload'].get('final_command_stamp')==final['stamp']
                    and x['payload'].get('command_valid') is True and x['payload'].get('phase')=='ARMED'
                    and x['payload'].get('state_source')=='DYNAMICS_INTEGRATED'
                    and x['payload'].get('integration_steps',0)>0 and x['monotonic']>=final['monotonic']]
            if not plants:continue
            c=consumed[e['trajectory_id']]
            result['chains'].append(dict(plan_id=pid,trajectory_id=e['trajectory_id'],command_index=c['command_index'],
                output_index=e['output_index'],output_ns=e['output_ns'],plant_steps=plants[0]['payload']['integration_steps']))
        # Order by actual Controller output order, not Reference or transport receipt order.
        previous=None
        for chain in result['chains']:
            pid=chain['plan_id']
            if pid not in result['accepted_plan_ids']:result['accepted_plan_ids'].append(pid)
            if previous is not None and previous!=pid:result['switches'].append(dict(before=previous,after=pid,output_index=chain['output_index']))
            previous=pid
        if len(result['accepted_plan_ids'])<2 or not result['switches']:return reject('TWO_COMPLETE_PLAN_CHAINS_AND_SWITCH_REQUIRED')
        result.update(status='PASS',controller_plan_lineage='PROVEN',reason='COMPLETE_CAUSAL_CHAINS')
        return result
    except (KeyError,TypeError,ValueError,OverflowError) as error:
        return reject('MALFORMED_EVIDENCE '+str(error))


def validate_observed_authorization(command,record):
    from query_minco.integration.yopo_execution_v4.trajectory import digest,Plan
    try:
        proof=command['authorization'];evidence=proof['evidence'];pid=digest(record)
        if proof['type']!='OBSERVED_SPACE_FULL_TUBE_V1' or proof['plan_id']!=pid or command['plan_id']!=pid:return 'OBSERVED_PLAN_ID'
        if evidence.get('passed') is not True:return 'OBSERVED_GEOMETRY_NOT_PASSED'
        if proof['map_epoch']!=command['map_epoch'] or proof['map_epoch']!=evidence['epoch']:return 'OBSERVED_MAP_EPOCH'
        if proof['snapshot_id']!=command['snapshot_id'] or proof['snapshot_id']!=evidence['snapshot']:return 'OBSERVED_SNAPSHOT'
        if proof['certificate_expiry']!=evidence['valid_until'] or not command['source_stamp']<=command['stamp']<=proof['certificate_expiry']:return 'OBSERVED_EXPIRY'
        expected=digest(dict(plan=pid,snapshot=proof['snapshot_id'],expiry=proof['certificate_expiry']))
        if proof['certificate_id']!=expected or command['certificate_id']!=expected:return 'OBSERVED_CERTIFICATE_ID'
        if proof['acceptance_id']!=command['acceptance_id'] or proof['source_stamp']!=command['source_stamp']:return 'OBSERVED_ACCEPTANCE'
        import numpy as np
        if not np.allclose(Plan(record).at(command['stamp']),command['pva'],atol=2e-5,rtol=0):return 'OBSERVED_COMMAND_CONTENT'
    except (KeyError,TypeError,ValueError):return 'MALFORMED_OBSERVED_CERTIFICATE'
    return None
