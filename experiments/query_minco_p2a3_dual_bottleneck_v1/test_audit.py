from copy import deepcopy
import pytest
from experiments.query_minco_p2a3_dual_bottleneck_v1.audit import choose,SELECTORS
def rows():
    return [dict(step=i,hard={"passed":i%2==0},loss=100.-i,q_alpha=1.+abs(i-30),Task={"TaskSuccess":i%3==0}) for i in range(65)]
@pytest.mark.parametrize("selector",SELECTORS)
def test_selection_does_not_consume_Task(selector):
    r=rows();a=choose(r,selector)["step"]
    for x in r:x["Task"]={"TaskSuccess":not x["Task"]["TaskSuccess"],"oracle_channel":"changed"}
    assert choose(r,selector)["step"]==a
def test_only_Hard_qualified_minima():
    r=rows();r[31]["loss"]=-1000;r[31]["q_alpha"]=-1000
    assert choose(r,"min_L0_Hard")["step"]==64
    assert choose(r,"min_q_Hard")["step"]==30
def test_earliest_tie():
    r=rows();r[10]["q_alpha"]=-1;r[12]["q_alpha"]=-1
    assert choose(r,"min_q_Hard")["step"]==10
def test_no_Hard_returns_none_but_fixed_can_fail():
    r=rows()
    for x in r:x["hard"]["passed"]=False
    assert choose(r,"last_Hard") is None and choose(r,"min_L0_Hard") is None
    assert choose(r,"fixed16")["step"]==16
def test_last_Hard_does_not_select_best_Task():
    r=rows();r[64]["hard"]["passed"]=False
    assert choose(r,"last_Hard")["step"]==62
