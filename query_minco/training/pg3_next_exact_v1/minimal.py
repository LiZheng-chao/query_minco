"""New minimal Mode identity: native PG2 Braking remains; no Stop/Geom/Keep evaluation."""
import copy
import torch
from query_minco.training.io import require,digest
from query_minco.inference.identity import state_sha
from query_minco.training.pg2_guided_v2.objective import forward as pg2_forward
from query_minco.training.pg2_guided_v1.geometry import tensor_digest,scene_from_field
from query_minco.training.pg3_feasible_mode_v1.teacher_labels import TeacherBank
from query_minco.training.pg3_feasible_mode_v1.spec import recipe
from query_minco.training.pg3_feasible_mode_v1.losses import mode_query
from query_minco.training.pg3_v2_no_corridor.revision import UniqueSampleFairAssignment,FrozenSafeReferenceV21
from query_minco.training.pg3_v2_no_corridor.objective import KEYS_V21
from query_minco.training.pg3_v2_no_corridor.executor_v21 import Engine,validate_protocol
ID='query_minco_pg3_next_minimal_mode_v1'
LAMBDA_MODE=.010859947391365976
WEIGHTS={k:LAMBDA_MODE if k=='mode' else 0. for k in KEYS_V21}
class MinimalObjective:
    def __init__(self,initial_model):
        self.assignment=UniqueSampleFairAssignment();self.reference=FrozenSafeReferenceV21(initial_model,expected_sha=state_sha(initial_model));self.bank=TeacherBank(dict(recipe(),build_corridor=False));self.weights=dict(WEIGHTS)
    def __call__(self,model,samples,geo,pg2_weights,context=None,step=0):
        z=pg2_forward(model,samples,geo,'query_minco_pg2_guided_v2_safe_v1',context,pg2_weights)
        return self.augment(z,samples,step)
    def augment(self,z,samples,step=0):
        o=z['output'];tr=o.trajectory;p=o.parameters;zero=(z['raw'].sum()+tr.coeffs.sum()+p.durations.sum())*0
        self.assignment.begin_step(step,[s.sample_id for s in samples]);assignments=[];values=[]
        try:
            for b,(s,f) in enumerate(zip(samples,z['fields'])):
                bank=self.bank.get(s,f)
                auth=digest(dict(inputs=tensor_digest(s.depth_input,s.state,s.teacher_coeffs,s.teacher_durations),geometry=scene_from_field(f),bank=bank['identity'],empty=f.row['empty_certified'],trajectory=tensor_digest(tr.coeffs[b],p.durations[b])))
                a=self.assignment.choose_once(s.sample_id,p.inner_pts[b],p.tail_pva[b,:,0],z['base']['matches'][b],z['labels']['passed'][b],bank['teachers'],authenticated_digest=auth)
                assignments.append(a);ml=[mode_query(p.inner_pts[b,q],p.tail_pva[b,q],p.durations[b,q],bank['teachers'][a['route_teacher_id'][q]],self.bank.cfg)[0] for q in a['selected']]
                values.append(torch.stack(ml).mean() if ml else zero)
            mode=torch.stack(values).mean();terms={k:mode if k=='mode' else zero for k in KEYS_V21}
            return dict(method=ID,pg2=z,base_loss=z['loss'],loss=z['loss']+LAMBDA_MODE*mode,raw=z['raw'],output=o,matched=z['matched'],labels=z['labels'],goal_labels=z['goal_labels'],terms=terms,weights=dict(WEIGHTS),stop_mode='legacy',assignments=assignments,geometry=[],branch_terms={k:zero for k in ('SECTION','BNECK_MATCHED','BNECK_FALLBACK')},diagnostics=dict(keep=dict(disabled=True),no_extra_stop_computed=True,convex_corridor_constructed=False,mode_query_denominators=[len(a['selected']) for a in assignments]))
        except BaseException:self.assignment.rollback(step);raise

def validate(spec):
    require(spec['method_id']==ID and spec['arm']=='MINIMAL' and spec['weights']==WEIGHTS and spec['stop_mode']=='legacy','NEXT_MINIMAL_IDENTITY')
    mapped=copy.deepcopy(spec);mapped.update(arm='A',method_id='query_minco_pg2_guided_v2_safe_v1');validate_protocol(mapped)
class MinimalEngine(Engine):
    def __init__(self,model,spec,config):
        validate(spec);mapped=copy.deepcopy(spec);mapped.update(arm='A',method_id='query_minco_pg2_guided_v2_safe_v1');super().__init__(model,mapped,config)
        self.spec=spec;self.objective=MinimalObjective(model);self.exposure=self.objective.assignment;self.reference=self.objective.reference
