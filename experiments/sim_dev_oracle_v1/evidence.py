"""Isolated mode adapter; reuses proven Controller->SO3->Plant causal checks."""
import inspect,json
import numpy as np
from query_minco.integration.f0_continuity_v1 import execution_evidence as strict
from query_minco.integration.yopo_execution_v4.trajectory import Plan,digest

def validate_oracle(command,record):
 try:
  p=Plan(record);a=command['authorization'];proof=a['oracle'];payload=dict(proof);identity=payload.pop('identity')
  if proof['mode']!='SIM_DEV_ORACLE' or proof['type']!='SIM_DEV_ORACLE_PREFIX_BRAKE' or a['geometry']!='SIM_DEV_ORACLE_PREFIX_BRAKE':return 'ORACLE_MODE'
  if 'command_key' in command and proof['nonce']!=command['command_key']['nonce']:return 'ORACLE_NONCE'
  if identity!=digest(payload) or identity!=command['certificate_id']:return 'ORACLE_IDENTITY'
  if proof['strict_depth_qualified'] is not False or proof['plan_id']!=p.id or command['plan_id']!=p.id:return 'ORACLE_PLAN'
  if not proof['issued']<=command['stamp']<=proof['expires'] or not 0<=command['stamp']-proof['source_stamp']<=.2:return 'ORACLE_EXPIRY'
  if proof['prefix']!=.1 or proof['max_delay']!=.35 or proof['revision']!=command['map_epoch'] or proof['map_id']!=command['snapshot_id']:return 'ORACLE_CONTRACT'
  if not np.allclose(p.at(command['stamp']),command['pva'],atol=2e-5,rtol=0):return 'ORACLE_PVA'
 except (KeyError,ValueError,TypeError):return 'ORACLE_MALFORMED'
 return None

def assess(*args,**kwargs):
 code=inspect.getsource(strict.assess)
 if code.count("'OBSERVED_SPACE_FULL_TUBE_129'")!=2:raise ValueError('UPSTREAM_ASSESSOR_DRIFT')
 code=code.replace("'OBSERVED_SPACE_FULL_TUBE_129'","'SIM_DEV_ORACLE_PREFIX_BRAKE'")
 old="if any(current.get(k)!=d.get(k) for k in ['plan_id','map_epoch','snapshot_id','certificate_id','acceptance_id']):return reject('OBSERVED_FORWARD_AUTHORITY_CONFLICT')"
 new="if current.get('plan_id') not in plans or validate_observed_authorization(current,plans[current['plan_id']]) is not None:return reject('ORACLE_FORWARD_AUTHORITY_INVALID')\n                if current.get('snapshot_id')!=d.get('snapshot_id') or current.get('authorization',{}).get('oracle',{}).get('nonce')!=expected_nonce:return reject('ORACLE_FORWARD_SCENE_OR_NONCE')"
 if code.count(old)!=1:raise ValueError('UPSTREAM_FORWARD_EVIDENCE_DRIFT')
 code=code.replace(old,new)
 # Raw C++ outputs may continue during owned shutdown without being forwarded.
 # Consumption must be timely regardless; output expiry applies to executed forwards.
 old="if proof.get('certificate_expiry',-math.inf)<max(e['received_ros'],output_ros):return reject('EXPIRED_AUTHORIZATION')\n            final=finals.get(e['output_ns'])\n            if final is None:continue"
 new="if proof.get('certificate_expiry',-math.inf)<e['received_ros']:return reject('EXPIRED_AUTHORIZATION')\n            final=finals.get(e['output_ns'])\n            if final is None:continue\n            if proof.get('certificate_expiry',-math.inf)<output_ros:return reject('EXPIRED_AUTHORIZATION')"
 if code.count(old)!=1:raise ValueError('UPSTREAM_OUTPUT_EXPIRY_CONTEXT_DRIFT')
 code=code.replace(old,new)
 namespace=dict(strict.__dict__);namespace['validate_observed_authorization']=validate_oracle
 exec(compile(code,__file__,'exec'),namespace);result=namespace['assess'](*args,**kwargs);result.update(mode='SIM_DEV_ORACLE',strict_depth_qualified=False,scope='SIMULATION_DEVELOPMENT_CAUSAL_EVIDENCE_ONLY');return result
