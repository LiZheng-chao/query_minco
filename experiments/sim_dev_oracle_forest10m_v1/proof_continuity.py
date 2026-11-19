"""Forest10m-only bridge adaptations. Shared execution contracts stay unchanged."""
from experiments.sim_dev_oracle_v1.prepare_runtime import replace_once as patch


def adapt_bridge(text):
    text=patch(text,"    if now<self.next_plan or self.state_pending:continue\n    if not 0<=now-source<=.2 or abs(actual[2]-source)>.02:continue\n    self.next_plan=now+.2", """    with self.lock:
     if self.state_pending or not 0<=now-source<=.2 or abs(actual[2]-source)>.02:continue
     urgent=getattr(self,'urgent_replan',False)
     if now<self.next_plan and not urgent:continue
     self.urgent_replan=False
     self.next_plan=now+.2""")
    text=patch(text,"    if old is not None and state_pair is not None and old.get('state_pair',{}).get('state_pair_id')==state_pair['state_pair_id']:continue",
               "    if old is not None and (proposal is None or proposal['plan'].id==old['plan'].id) and state_pair is not None and (old.get('state_pair') or {}).get('state_pair_id')==state_pair['state_pair_id']:continue")
    text=patch(text,"    if not result['ok']:raise ValueError(result['reason'])", """    if not result['ok']:
     if result['reason'] in ('TRACKING_PVA','PLAN_HORIZON'):
      with self.lock:
       current=self.committed
       if (current is not None and current['plan'].id==plan.id
           and (self.proposed is None or self.proposed['plan'].id==plan.id)
           and getattr(self,'early_replan_plan',None)!=plan.id):
        self.early_replan_plan=plan.id
        self.urgent_replan=True
        self.record('EARLY_REPLAN_REQUESTED',plan_id=plan.id,reason=result['reason'],
                    proof_expires=current['proof']['expires'])
     raise ValueError(result['reason'])""")
    text=patch(text," def command(self,now):"," def command(self,now,*,dispatch=None):")
    text=patch(text,"   if c is None or actual is None or goal is None or self.audit_overflow or self.evidence_overflow or self.input_fault:return None", """   def reject(reason,**details):
    if dispatch is not None and not getattr(self,'command_failure_reported',False):
     self.command_failure_reported=True
     p=c['proof'] if c is not None else {}
     pair=(c.get('state_pair') or {}) if c is not None else {}
     self.record('COMMAND_REJECTED',reason=reason,dispatch=dispatch,now=now,
                 plan_id=c['plan'].id if c is not None else None,
                 proof_issued=p.get('issued'),proof_expires=p.get('expires'),map_id=p.get('map_id'),
                 proof_depth=p.get('source_stamp'),latest_depth=self.last_stamp,
                 odom_stamp=actual[1] if actual else None,imu_stamp=actual[2] if actual else None,
                 committed_odom_stamp=pair.get('odom_stamp'),committed_imu_stamp=pair.get('imu_stamp'),
                 goal_stamp=goal[1] if goal else None,input_fault=self.input_fault,
                 audit_overflow=self.audit_overflow,evidence_overflow=self.evidence_overflow,**details)
    return None
   if self.audit_overflow or self.evidence_overflow or self.input_fault:return reject('INPUT_INVALID_OR_OVERFLOW')
   if c is None:return reject('NO_COMMITTED_PLAN')
   if actual is None:return reject('ODOM_IMU_STALE')
   if goal is None:return reject('GOAL_STALE')""")
    text=patch(text,"    except OSError:return None\n    if st.st_size!=row['size'] or st.st_mtime_ns!=row['mtime_ns']:return None","    except OSError:return reject('MAP_IDENTITY_INVALID',path=row['path'])\n    if st.st_size!=row['size'] or st.st_mtime_ns!=row['mtime_ns']:return reject('MAP_IDENTITY_INVALID',path=row['path'])")
    text=patch(text,"   if not proof['issued']<=now<=proof['expires'] or not 0<=now-self.last_stamp<=.2 or not 0<=now-actual[1]<=.2 or not 0<=now-actual[2]<=.2 or not 0<=now-goal[1]<=.2:return None", """   if now<proof['issued']:return reject('PROOF_NOT_YET_VALID')
   if not now<=proof['expires']:return reject('PROOF_EXPIRED')
   if not 0<=now-self.last_stamp<=.2:return reject('DEPTH_STALE')
   if not 0<=now-actual[1]<=.2 or not 0<=now-actual[2]<=.2:return reject('ODOM_IMU_STALE')
   if not 0<=now-goal[1]<=.2:return reject('GOAL_STALE')""")
    # Preserve fail-closed behaviour for nonfinite times as well.
    text=patch(text,"   if now<proof['issued']:","   if not proof['issued']<=now:")
    text=patch(text,"for k in ['odom_stamp','imu_stamp']):return None","for k in ['odom_stamp','imu_stamp']):return reject('COMMITTED_PAIR_STALE')")
    text=patch(text,"   if np.linalg.norm(plan.at(now)[0]-actual[0]['p'])>.15:return None",
               "   tracking=float(np.linalg.norm(plan.at(now)[0]-actual[0]['p']))\n   if tracking>.15:return reject('TRACKING_EXCEEDED',tracking_m=tracking)")
    return patch(text,"   command=self.command(now)\n   if command is not None:action(command)",
                 "   command=self.command(now,dispatch=getattr(action,'__name__','control'))\n   if command is not None:action(command)")
