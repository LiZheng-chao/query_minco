"""Pure pre-ARM perception fence. It never issues a permit."""
import math
def require_initial_coverage(heartbeat,now):
 boot=heartbeat.get('bootstrap') or {}
 if heartbeat.get('mode')!='ACK' or (heartbeat.get('plant') or {}).get('integration_steps')!=0:raise PermissionError('INITIAL_HELD_ACK_REQUIRED')
 if boot.get('status')!='STATIC_PASS':raise PermissionError('LIVE_INITIAL_TUBE_UNKNOWN_NO_ARM')
 expires=boot.get('valid_until',float('-inf'));observed=heartbeat.get('ros_now',float('-inf'))
 if not all(math.isfinite(x) for x in [now,expires,observed]) or not 0<=now-observed<=.05 or now>expires:raise PermissionError('INITIAL_COVERAGE_EXPIRED_NO_ARM')
 return True
