import copy,json,hashlib,unittest
from query_minco.integration.f0_continuity_v1.execution_evidence import assess
NONCE='a'*32

def fixture():
    refs=[];events=[];log=[dict(kind='AUDIT_OPEN',nonce=NONCE,pid=123)];i=0
    for tid in [1,2]:
        rec=dict(label=str(tid));pid=hashlib.sha256(json.dumps(rec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        refs.append(dict(kind='plan',plan=rec,plan_id=pid))
        now=100.+tid;ns=int(now*1e9);mono=float(tid)
        pva=[[tid*.1,0,2],[0,0,0],[0,0,0]]
        proof=dict(plan_id=pid,geometry='FINITE_BIRTH_FULL_TUBE_129_WITH_OCCUPIED_VETO',certificate_expiry=now+.2)
        events.append(dict(kind='controller_command',monotonic=mono,mode='ARMED',accepted=True,
            payload=dict(plan_id=pid,authorization=proof,pva=pva,yaw=0,yaw_dot=0,
                command_key=dict(nonce=NONCE,trajectory_id=tid,input_ns=ns))))
        c=dict(kind='COMMAND_CONSUMED',index=tid*2-1,trajectory_id=tid,input_ns=ns,output_ns=0,
            command_index=tid,output_index=0,received_ros=now+.001,monotonic=mono+.001,
            pva=[x for a in pva for x in a],yaw=0,yaw_dot=0,kx=[5.7]*3,kv=[3.4]*3)
        log.append(c);out=dict(c,kind='CONTROL_OUTPUT',index=tid*2,output_index=tid,output_ns=ns+2000000,monotonic=mono+.002,
            force=[0,0,9.6],orientation=[0,0,0,1],kr=[1]*3,kom=[.1]*3,aux=[0,0,0,0,1,0,0])
        log.append(out)
        events.append(dict(kind='final_so3',monotonic=mono+.003,mode='ARMED',accepted=True,stamp=now+.002,
            payload=dict(stamp_ns=out['output_ns'],force_world=out['force'],q=out['orientation'],kR=out['kr'],kOm=out['kom'],audit_aux=out['aux'])))
        events.append(dict(kind='plant',monotonic=mono+.004,mode='ARMED',accepted=True,
            payload=dict(final_command_stamp=now+.002,phase='ARMED',state_source='DYNAMICS_INTEGRATED',command_valid=True,integration_steps=tid*100)))
    log.append(dict(kind='AUDIT_CLOSE',records=4,lost=0,write_failed=False))
    return events,refs,log,dict(verified=True,pid=123,nonce=NONCE)

def check(f):
    e,r,l,identity=f
    return assess(e,0,4,r,l,NONCE,identity)

class Evidence(unittest.TestCase):
    def test_two_complete_chains_pass(self):
        d=check(fixture());self.assertEqual(d['status'],'PASS',d);self.assertEqual(len(d['switches']),1)
    def test_wrong_plan_id(self):
        f=fixture();f[0][3]['payload']['plan_id']='0'*64;self.assertNotEqual(check(f)['status'],'PASS')
    def test_wrong_command_identity(self):
        f=fixture();f[2][3]['input_ns']+=1;self.assertNotEqual(check(f)['status'],'PASS')
    def test_duplicate_consumption(self):
        f=fixture();f[2][3]['trajectory_id']=1;self.assertNotEqual(check(f)['status'],'PASS')
    def test_expired_command(self):
        f=fixture();f[2][3]['received_ros']+=1;self.assertNotEqual(check(f)['status'],'PASS')
    def test_missing_consumption(self):
        f=fixture();f[2][3]['kind']='MISSING';self.assertNotEqual(check(f)['status'],'PASS')
    def test_false_switch_without_plant(self):
        f=fixture();f[0].pop();self.assertNotEqual(check(f)['status'],'PASS')
    def test_wrong_plant_receipt(self):
        f=fixture();f[0][-1]['payload']['final_command_stamp']+=.001;self.assertNotEqual(check(f)['status'],'PASS')
    def test_startup_hold_cannot_count(self):
        f=fixture();f[0][3]['payload']['plan_id']=None;self.assertNotEqual(check(f)['status'],'PASS')
    def test_lost_log(self):
        f=fixture();f[2][-1]['lost']=1;self.assertNotEqual(check(f)['status'],'PASS')
    def test_unclean_log(self):
        f=fixture();f[2].pop();self.assertNotEqual(check(f)['status'],'PASS')
    def test_wrong_loaded_identity(self):
        f=fixture();f[3]['pid']=124;self.assertNotEqual(check(f)['status'],'PASS')
    def test_output_content_conflict(self):
        f=fixture();f[0][4]['payload']['force_world']=[1,2,3];self.assertNotEqual(check(f)['status'],'PASS')
    def test_output_active_identity_conflict(self):
        f=fixture();f[2][4]['command_index']=1;self.assertNotEqual(check(f)['status'],'PASS')
    def test_old_format_stays_unknown(self):
        self.assertNotEqual(assess([],0,4,[])['status'],'PASS')

    def test_output_after_authorization_expiry(self):
        f=fixture();f[2][4]['output_ns']+=1000000000;self.assertNotEqual(check(f)['status'],'PASS')
    def test_writer_failure(self):
        f=fixture();f[2][-1]['write_failed']=True;self.assertNotEqual(check(f)['status'],'PASS')
    def test_record_gap(self):
        f=fixture();f[2][2]['index']=9;self.assertNotEqual(check(f)['status'],'PASS')
    def test_duplicate_supervisor_id(self):
        f=fixture();f[0][3]['payload']['command_key']['trajectory_id']=1;self.assertNotEqual(check(f)['status'],'PASS')

    def test_real_controller_log_schema_with_explicitly_synthetic_plant(self):
        from pathlib import Path
        root=Path('research_outputs/f0_controller_consumption_20261118_v1/audit/ros_bench_v4')
        logs=[json.loads(x) for x in (root/'CONSUMPTION.jsonl').read_text().splitlines()]
        identity=json.loads((root/'diagnostic_on.LOADED_IDENTITY.json').read_text())
        replay=json.loads((root/'diagnostic_on.REPLAY.json').read_text())
        events=[];refs=[];ids=[]
        for label in ['Plan_A','Plan_B']:
            record=dict(test_contract='REAL_CONTROLLER_SYNTHETIC_SUPERVISOR_PLANT',label=label)
            pid=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            refs.append(dict(kind='plan',plan=record,plan_id=pid));ids.append(pid)
        for c in logs:
            if c['kind']!='COMMAND_CONSUMED':continue
            pid=ids[int(c['trajectory_id']>30)];pva=[c['pva'][i:i+3] for i in [0,3,6]]
            proof=dict(plan_id=pid,geometry='FINITE_BIRTH_FULL_TUBE_129_WITH_OCCUPIED_VETO',certificate_expiry=c['received_ros']+.2)
            events.append(dict(kind='controller_command',monotonic=c['monotonic']-.00001,accepted=True,mode='ARMED',
                payload=dict(plan_id=pid,authorization=proof,pva=pva,yaw=c['yaw'],yaw_dot=c['yaw_dot'],
                    command_key=dict(nonce=NONCE,trajectory_id=c['trajectory_id'],input_ns=c['input_ns']))))
        actual={x['stamp_ns']:x for x in replay['output']}
        for o in logs:
            if o['kind']!='CONTROL_OUTPUT':continue
            raw=actual[o['output_ns']];stamp=o['output_ns']//1000000000+(o['output_ns']%1000000000)/1e9
            events.append(dict(kind='final_so3',monotonic=o['monotonic']+.00001,accepted=True,mode='ARMED',stamp=stamp,
                payload=dict(stamp_ns=o['output_ns'],force_world=raw['force'],q=raw['orientation'],kR=raw['kr'],kOm=raw['kom'],audit_aux=raw['aux'])))
            events.append(dict(kind='plant',monotonic=o['monotonic']+.00002,accepted=True,mode='ARMED',
                payload=dict(final_command_stamp=stamp,phase='ARMED',state_source='DYNAMICS_INTEGRATED',command_valid=True,integration_steps=o['output_index'])))
        events.sort(key=lambda e:e['monotonic'])
        result=assess(events,events[0]['monotonic']-1,events[-1]['monotonic']+1,refs,logs,NONCE,identity)
        Path('research_outputs/f0_controller_consumption_20261118_v1/audit/REAL_CONTROLLER_SYNTHETIC_PLANT_EVIDENCE.json').write_text(json.dumps(dict(scope='REAL_CONTROLLER_SYNTHETIC_SUPERVISOR_AND_PLANT_NOT_PHYSICAL',result=result),indent=2)+'\n')
        self.assertEqual(result['status'],'PASS',result)
        self.assertEqual(len(result['chains']),120)
        self.assertEqual(len(result['switches']),1)
