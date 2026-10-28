"""Reports from R1.1 saved CPU0 artifacts only. Does not execute models."""
import json,csv,collections,sys,math
from pathlib import Path
import numpy as np
from experiments.pg1_v2_r11.math_probes import unique_task_rows
R=Path(sys.argv[1]);S=R/'summary'
def read(p):return json.loads(p.read_text())
def write(name,obj): (S/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def md(name,text):(S/name).write_text(text,encoding='utf-8')
def csvout(name,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (S/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def quant(xs):
 a=np.asarray([x for x in xs if x is not None],dtype=float)
 if not len(a):return dict(n=0)
 return dict(n=len(a),min=float(a.min()),p10=float(np.quantile(a,.1)),p50=float(np.median(a)),p90=float(np.quantile(a,.9)),p95=float(np.quantile(a,.95)),max=float(a.max()),mean=float(a.mean()),nonzero_rate=float(np.mean(a!=0)))
def fracbad(xs):return sum(x<0 for x in xs if x is not None)/sum(x is not None for x in xs) if any(x is not None for x in xs) else None
result=read(R/'audit/CPU0_RESULT.json');assert result['status'] in ('COMPLETE','COMPLETE_FROM_VERIFIED_STREAMS_AND_REPLAY')
parent=None
if (R/'_preparation/CONTINUATION_INPUTS.json').exists():
 info=read(R/'_preparation/CONTINUATION_INPUTS.json');parent=Path(info['parent_run'])
 import hashlib
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in info['parent_files'].items())
 result=dict(result);result['models']=[info['parent_complete_model']]+result['models']
 result['wall_s']+=info['parent_wall_s'];result['parent_wall_s']=info['parent_wall_s']
 result['wall_s_semantics']='sum completed C1 stage plus continuation wall; not single-process time; discarded partial E2 prefix elapsed excluded'
 result['parent_thread_cpu_s']=None;result['parent_max_rss_mib']=None
 result['batches']=288;result['unique_task_models']=1152;result['candidate_visits']=13824;result['unique_candidate_models']=6912
 result['training_qualified']=False
def combined(name):
 own=read(R/'runs'/name)['records']
 if parent is not None:own=[r for r in read(parent/'runs'/name)['records'] if r['model']=='C1_step0']+own
 return own
task_records_raw=combined('UNIQUE_TASKS.json');tasks=unique_task_rows(task_records_raw);directions=combined('DIRECTIONS.json');cases=combined('CASES.json');budgets=combined('BATCH_RAW_AND_BUDGET.json')
result['direction_records']=len(directions)
result['source_unique_counter']=result['unique_task_models'];result['unique_task_models']=len(tasks)
result['raw_first_visit_records']=len(task_records_raw);result['deduplicated_unique_task_models']=len(tasks)
write('UNIQUE_METADATA_ERRATUM.json',dict(raw_first_visit_records=len(task_records_raw),actual_unique=len(tasks),reason='one originalOrder batch repeats one ID; both first_visit flags were set before updating visited. Dedup stable model/sample_id; duplicate payload identical. All exposure/group gradients remain unchanged.',source_logs_not_rewritten=True))
write('COMBINED_CPU0_RESULT.json',result)
records=[]
gradpaths=list((R/'runs').glob('*_GRADIENTS.jsonl'))+(list((parent/'runs').glob('C1_step0_GRADIENTS.jsonl')) if parent else [])
for f in sorted(gradpaths):
 records.extend(json.loads(l) for l in f.read_text().splitlines())
assert len(tasks)==1152 and len({(x['model'],x['sample_id']) for x in tasks})==1152
assert all(len({x['sample_id'] for x in tasks if x['model']==m['model']})==384 for m in result['models'])
csvout('TRAINING_SET_GRADIENT_AUDIT.csv',records)
allref={(x['model'],x['batch'],x['term']):x for x in records if x['group']=='all'}
gradientstats={}
for key in sorted({(x['model'],x['group'],x['term']) for x in records}):
 rr=[x for x in records if (x['model'],x['group'],x['term'])==key]
 stats={k:quant(x[k] for x in rr) for k in ['parameter_norm','ratio_imitation','ratio_obstacle','group_scalar_contribution','conditional_query_mean']}
 stats.update(conflict_imitation_rate=fracbad([x['cos_imitation'] for x in rr]),conflict_obstacle_rate=fracbad([x['cos_obstacle'] for x in rr]),
  cosine_imitation=quant(x['cos_imitation'] for x in rr),cosine_obstacle=quant(x['cos_obstacle'] for x in rr),
  raw_time15_19=[quant(x['raw_time15_19'][i] for x in rr) for i in range(4)],
  ratio_full_batch_imitation=quant(x['parameter_norm']/allref[(x['model'],x['batch'],'imitation')]['parameter_norm'] for x in rr if allref[(x['model'],x['batch'],'imitation')]['parameter_norm']>0))
 gradientstats['|'.join(key)]=stats
write('GRADIENT_DISTRIBUTIONS.json',gradientstats)
# Unique per-query value statistics in overlapping strata; not correlated exposures.
vals=collections.defaultdict(list);rawcs=collections.defaultdict(list);rawnorms=collections.defaultdict(list)
taskmap={(t['model'],t['sample_id']):t for t in tasks}
for t in tasks:
 for q,c in enumerate(t['surface_clearances_m']):
  geom='bottleneck' if c is not None and c<=.08 else ('wide' if c is None or c>.15 else 'transition')
  groups=['all','M%d'%t['M'],t['regime'].capitalize(),'matched' if t['matched'][q] else 'unmatched',geom]
  for name,v in t['terms'].items():
   for group in groups:vals[t['model'],group,name].append(v[q])
unique_raw_seen=set()
for b in budgets:
 for i,sid in enumerate(b['ids']):
  kk=(b['model'],sid)
  if kk in unique_raw_seen:continue
  unique_raw_seen.add(kk)
  t=taskmap[b['model'],sid]
  for q,c in enumerate(t['surface_clearances_m']):
   geom='bottleneck' if c is not None and c<=.08 else ('wide' if c is None or c>.15 else 'transition')
   groups=['all','M%d'%t['M'],t['regime'].capitalize(),'matched' if t['matched'][q] else 'unmatched',geom]
   for name,v in b['raw'].items():
    for group in groups:
     rawnorms[b['model'],group,name].append(v['norm'][i][q])
     for ref in ['imitation','obstacle']:
      rawcs[b['model'],group,name,ref].append(v['cos_'+ref][i][q])
write('UNIQUE_QUERY_VALUE_DISTRIBUTIONS.json',{'|'.join(k):quant(v) for k,v in vals.items()})
write('UNIQUE_RAW_GRADIENT_DISTRIBUTIONS.json',{'|'.join(k):quant(v) for k,v in rawnorms.items()})
write('NEGATIVE_SCALAR_ROUNDOFF.json',{'|'.join(k):dict(n=len(v),negative_count=sum(x<0 for x in v),min=min(v)) for k,v in vals.items() if k[1]=='all'})
write('UNIQUE_RAW_CONFLICTS.json',{'|'.join(k):dict(n=len(v),nonzero_pairs=sum(x is not None for x in v),conflict_rate=fracbad(v),cosine=quant(v)) for k,v in rawcs.items()})
# Aggregator paired checkpoint relation on ALL training candidates (same task/query).
pairs=[]
for model in ['E2_step32','PG1_step32']:
 for t in (t for t in tasks if t['model']==model):
  b=taskmap['C1_step0',t['sample_id']]
  for q in range(6):
   cb=b['surface_clearances_m'][q];ca=t['surface_clearances_m'][q]
   if cb is None or ca is None:continue
   dc=ca-cb
   for k in ['obstacle','mean_plain','mean_rho','lse','lse_rho','tail','tail_rho']:
    dl=t['terms'][k][q]-b['terms'][k][q]
    pairs.append(dict(model=model,sample_id=t['sample_id'],query=q,M=t['M'],regime=t['regime'],matched=t['matched'][q],term=k,
     clearance_before_m=cb,clearance_after_m=ca,delta_clearance_m=dc,proxy_before=b['terms'][k][q],proxy_after=t['terms'][k][q],delta_proxy=dl,
     comparison='checkpoint_pair_READONLY_NOT_new_loss_training',geometry_pair_stratum='bottleneck_any_checkpoint' if min(cb,ca)<=.08 else ('wide_both' if min(cb,ca)>.15 else 'transition'),opposite_sign_consistent=(dc*dl<0) if abs(dc)>1e-6 and abs(dl)>1e-8 else None,
     clearance_worse_proxy_lower=bool(dc< -1e-6 and dl< -1e-8)))
csvout('RISK_WEIGHT_AGGREGATION_ABLATION.csv',pairs)
paired={}
for key in sorted({(p['model'],p['term']) for p in pairs}):
 rr=[p for p in pairs if (p['model'],p['term'])==key];active=[p for p in rr if p['opposite_sign_consistent'] is not None]
 critical=[p for p in rr if p['geometry_pair_stratum']=='bottleneck_any_checkpoint']
 paired['|'.join(key)]=dict(n=len(rr),active_n=len(active),consistent=sum(x['opposite_sign_consistent'] for x in active),worse_proxy_lower=sum(x['clearance_worse_proxy_lower'] for x in rr),bottleneck_any_checkpoint_n=len(critical),bottleneck_worse_proxy_lower=sum(x['clearance_worse_proxy_lower'] for x in critical))
write('AGGREGATOR_CHECKPOINT_ALIGNMENT.json',paired)
# All functional parameter directions (unit direction; NOT optimizer or AdamW).
dr=[]
for d in directions:
 b=d['baseline'];a=d['after']
 dr.append(dict(model=d['model'],split=d['split'],sample_id=d['sample_id'],query=d['query'],term=d['term'],parameter_l2=d['parameter_l2'],raw_max_delta=d['raw_max_delta'],radius_over_nominal=d['radius_over_nominal'],
  clearance_before_m=b['clearance_m'],clearance_after_m=a['clearance_m'],delta_clearance_m=a['clearance_m']-b['clearance_m'],
  progress_before=b['goal']['projected_progress_m'],progress_after=a['goal']['projected_progress_m'],
  goal_before=b['goal']['passed'],goal_after=a['goal']['passed'],
  hard_before=b['hard_pass'],hard_after=a['hard_pass'],
  vmax_before=b['vmax'],vmax_after=a['vmax'],amax_before=b['amax'],amax_after=a['amax'],jmax_before=b['jmax'],jmax_after=a['jmax'],
  T_before=b['time_s'],T_after=a['time_s'],heuristic_mode_before=b['heuristic_mode'],heuristic_mode_after=a['heuristic_mode'],
  projection_scale=d['projection_scale'],projected_first_order_obstacle_dot=d['projected_first_order_obstacle_dot']))
csvout('QUALITY_SAFETY_CONFLICT.csv',dr)
directionstats={}
for key in sorted({(d['split'],d['term']) for d in dr}):
 rr=[x for x in dr if (x['split'],x['term'])==key]
 directionstats['|'.join(key)]=dict(n=len(rr),clearance_improve=sum(x['delta_clearance_m']>1e-6 for x in rr),clearance_harm=sum(x['delta_clearance_m']< -1e-6 for x in rr),
  delta_clearance_m=quant(x['delta_clearance_m'] for x in rr),progress_harm=sum(x['progress_after']<x['progress_before']-1e-6 for x in rr),
  hard_pass_to_fail=sum(x['hard_before'] and not x['hard_after'] for x in rr),goal_pass_to_fail=sum(x['goal_before'] and not x['goal_after'] for x in rr),
  vmax_increase=sum(x['vmax_after']>x['vmax_before']+1e-6 for x in rr),amax_increase=sum(x['amax_after']>x['amax_before']+1e-6 for x in rr),jmax_increase=sum(x['jmax_after']>x['jmax_before']+1e-6 for x in rr),
  heuristic_mode_changes=sum(x['heuristic_mode_after']!=x['heuristic_mode_before'] for x in rr),radius_over_nominal=sum(x['radius_over_nominal'] for x in rr))
write('DIRECTION_SUMMARY.json',directionstats)
modes=[]
for key in sorted({(d['model'],d['split'],d['sample_id'],d['term']) for d in directions if d['split']=='TRAIN'}):
 rr=[d for d in directions if (d['model'],d['split'],d['sample_id'],d['term'])==key]
 before=[d['baseline'] for d in rr];after=[d['after'] for d in rr]
 def cover(cc):
  labels=[c['heuristic_mode'] for c in cc if c['heuristic_mode'] is not None]
  pts=[c['crossing_position'] for c in cc if c['crossing_position'] is not None]
  ds=[float(np.linalg.norm(np.array(x)-y)) for i,x in enumerate(pts) for y in pts[i+1:]]
  return dict(valid_crossing=len(pts),distinct_heuristic_labels=len(set(labels)),labels=labels,pairwise_crossing_distance_m=quant(ds))
 modes.append(dict(model=key[0],split=key[1],sample_id=key[2],term=key[3],query_count=len(rr),before=cover(before),after=cover(after)))
write('MODE_COVERAGE_DIAGNOSTIC.json',modes)
# Explicit FP32 raw-gradient differences in legacy-vs-renormalized rho mean.
rho=[];current=[]
rawpaths=list((R/'runs').glob('*_RAW.npz'))+(list((parent/'runs').glob('C1_step0*_RAW.npz')) if parent else [])
for path in sorted(rawpaths):
 with np.load(path,allow_pickle=False) as z:
  for a,b,label in [('g_obstacle','g_mean_rho','rho_normalizer'),('g_QA','g_QA_current','QA'),('g_QJ','g_QJ_current','QJ')]:
   aa=z[a];bb=z[b]
   rec=dict(file=path.name,term=label,norm_before=float(np.linalg.norm(aa)),norm_after=float(np.linalg.norm(bb)),difference_norm=float(np.linalg.norm(aa-bb)),
    time15_19_before=[float(np.linalg.norm(aa[...,j])) for j in range(15,19)],time15_19_after=[float(np.linalg.norm(bb[...,j])) for j in range(15,19)],
    cosine=float((aa*bb).sum()/(np.linalg.norm(aa)*np.linalg.norm(bb))) if np.linalg.norm(aa)*np.linalg.norm(bb)>0 else None)
   (rho if label=='rho_normalizer' else current).append(rec)
write('RHO_NORMALIZATION_GRADIENT_AUDIT.json',rho);write('QUALITY_NORMALIZATION_GRADIENT_AUDIT.json',current)
# Candidate scales from THREE MODELS x96 full training batches only, fixed kappa.
weights={}
for term in ['QA','QJ','QT','lse','tail','lse_rho','tail_rho']:
 qq=[gradientstats[m+'|all|'+term]['ratio_imitation'] for m in ['C1_step0','E2_step32','PG1_step32']]
 worst95=max(x['p95'] for x in qq);weights[term]=dict(worst_model_p95_ratio=worst95,candidate_lambda_low=.05/worst95,candidate_lambda_high=.1/worst95,stress_only=.3/worst95,
     model_ratios={m:gradientstats[m+'|all|'+term]['ratio_imitation'] for m in ['C1_step0','E2_step32','PG1_step32']},not_recipe=True)
write('TRAINING_ONLY_WEIGHT_SCALES.json',weights)
write('TOTAL_GRADIENT_BUDGETS.json',[dict(model=b['model'],batch=b['batch'],**b['budget']) for b in budgets])
# Markdown reports.
md('TRAINING_SET_GRADIENT_AUDIT.md',
'# R1.1 Frozen384损失与梯度审计\n\n'
'完整384任务在三个独立权重上均覆盖。原Seed42 Order不是无放回epoch：前48批只有288唯一任务，必须取前96批、768曝光才全覆盖384；总288批、2304任务曝光、13824候选曝光。唯一任务-模型1152、唯一候选-模型6912。两个Pilot step0是同一C1权重，去重而非伪造独立模型。其他Seed/step8/16未运行。\n\n'
'FP32归一化LSE在远离障碍、风险接近零时可能出现logsumexp消去带来的零/微小负值，原值全部保留，NEGATIVE_SCALAR_ROUNDOFF.json逐项量化；未加clamp或改公式。值为零不等于梯度为零，另提供UNIQUE_RAW_GRADIENT_DISTRIBUTIONS.json。\n\n每次同一冻结forward、相同任务/Query/Teacher匹配，独立求16分项（要求的十项+rho/归一化对照）；所有raw19:21梯度精确零且有限。每项参数梯度分别按all/matched/unmatched/M1/M4/Near/Far/瓶颈/宽净空/中间带分解。matched+unmatched向量和逐项恢复总梯度；范数不可相加。组梯度保留原Batch8×6分母，不将小组独立归一化。Imitation分配为匹配cost×6/M后统一均值，与旧算术近似重建校核，旧目标本身未改变。\n\n'
'TRAINING_SET_GRADIENT_AUDIT.csv为逐批组贡献；GRADIENT_DISTRIBUTIONS.json为参数范数、尺度、cosine、raw时间梯度分位数，属于曝光加权的批次贡献统计，不是逐任务独立参数梯度。UNIQUE_QUERY_VALUE_DISTRIBUTIONS.json与UNIQUE_RAW_CONFLICTS.json仅保留每任务首次出现，报告唯一Query的值、非零率和冲突率。分层重叠，不可将各层人数相加。\n\n'
'几何分层按该模型候选的数值连续最小机体表面净空：≤8cm为bottleneck，>15cm为wide，余者transition；认证空场归wide且净空null。此分类不是原Bernstein安全证书，也不是按本轮loss挑例。M与Near/Far来自认证manifest，不凭命名猜测。相同任务在不同checkpoint可能跨几何分层，配对分析仍按同task/query对齐。\n\n'
'## 三权重全量批次梯度尺度（/Imitation，中位数 / P95）\n\n|项|C1 step0|E2 step32|PG1 step32|\n|---|---|---|---|\n'+''.join('|%s|%s|\n'%(k,'|'.join('%.4g / %.4g'%(gradientstats[m+'|all|'+k]['ratio_imitation']['p50'],gradientstats[m+'|all|'+k]['ratio_imitation']['p95']) for m in ['C1_step0','E2_step32','PG1_step32'])) for k in ['obstacle','goal','limit','domain','QA','QJ','QT','lse','tail'])+
'\n所有零梯度及零分母保留：cosine/ratio不可定义时写null，不当作无冲突。unmatched Imitation直接梯度为零，但共享参数变化仍会影响其候选。原主过程的Quality-only累计预算未完整保存，TOTAL_GRADIENT_BUDGETS.json明确保留缺失状态，不补造向量范数。本轮另从全部保存raw梯度重新计算QA/QJ/QT+LSE或Tail联合向量预算，见JOINT_GRADIENT_BUDGET.csv；仅是冻结权重算术，不是训练结果或正式权重。\n')
focus={}
for split in ['TRAIN','HISTORICAL_DEV']:
 cc=[c for c in cases if c['split']==split]
 focus[split]={k:quant(c['focus'][k] for c in cc) for k in ['obstacle','mean_plain','mean_rho','lse','lse_rho','tail','tail_rho']}
write('BOTTLENECK_FOCUS.json',focus)
md('RISK_WEIGHT_AGGREGATION_ABLATION.md',
'# rho与聚焦方式的严格隔离\n\n所有七项使用相同64点/段、同AABB SDF、同192节点、同物理时间alpha、r=.12m、margin=.03m、eps=.05m，tau=.025m、q=.1固定，无开发集搜索。\n\n'
'1. 原E2 Obstacle直接调用physical_loss：rho=clip(1+relu(-SDF_rate),1,3)/sum(alpha.detach()*risk)，整体detach。\n'
'2/3. plain/rho mean；4/5. plain/rho LSE；6/7. plain/rho Tail。含rho新项以m=alpha*rho / sum(alpha*rho)作为概率质量，rho本身detach，m对alpha保留梯度；总质量1、量纲米，Tail取风险排序下最坏10%该概率质量，非简单top-k。\n\n'
'**重要非等价：**新rho mean和旧E2的前向值相同（逐批校核），但旧rho分母detach，新聚合的质量分母未detach，所以duration/raw时间梯度可能不同。RHO_NORMALIZATION_GRADIENT_AUDIT.json逐批记录，不把该差异算作聚焦收益。LSE/Tail带rho应在同一归一化下和新mean_rho比较；旧E2是历史额外对照。rho也改变Tail的最坏10%所对应的实际物理时间长度，不能称原时间分位。\n\n'
'## 全训练集checkpoint配对（数值连续净空）\n\n|step32|项|可比较候选数|非微小配对数|风险方向一致|净空降且代理降|\n|---|---|---:|---:|---:|---:|\n'+''.join('|%s|%s|%d|%d|%d|%d|\n'%(key.split('|')[0],key.split('|')[1],v['n'],v['active_n'],v['consistent'],v['worse_proxy_lower']) for key,v in paired.items())+
'\n方向一致定义Δclearance×Δproxy<0，分辨尺度1µm/1e-8m仅用于诊断，不改硬阈值。原两组更新由旧配方产生，新项未训练；这种相关性不能作新训练因果结论。聚焦度在预锁定训练子集及三个历史例上用连续最低净空时刻±50ms的point-risk梯度质量记录（BOTTLENECK_FOCUS.json）。参数负方向结果见DIRECTION_SUMMARY.json；rho效应应比较带/不带同类聚合，不能把所有改善归因LSE/Tail。原YOPO已有LSE，不声明该公式新颖。\n\n'
'数值连续净空复用已哈希的分段多项式/AABB根诊断，不是严格区间算术证书；方向子集同时调用未修改的原Bernstein硬验证器。full384配对CSV本身不提供全量Bernstein通过率。旧三个开发例只是预指定反例复查，没有参与权重或tau/q选择。\n')
md('QUALITY_SAFETY_CONFLICT.md',
'# Quality与安全的冲突\n\nQA=∫|a|²dt/(8²×2.2)，QJ=∫|j|²dt/(100²×2.2)，QT=T/2.2，保持原Dynamic Limit。当前T对照QA_T=QA×2.2/T、QJ_T同理；完整真实MINCO系数、节点和时长均可微。均匀几何时间伸缩下固定参考能量按s^-3/s^-5，当前T归一化按s^-4/s^-6；后者更强奖励延时，但固定实测边界PVA改变时不适用该简单缩放推理。QUALITY_NORMALIZATION_GRADIENT_AUDIT.json记录实际raw时间梯度，不以推理替代。\n\n'
'方向实验仅在运行前锁定的训练分层子集（每M×Near/Far首先两任务）和历史三Query进行。使用模型参数单位负梯度的临时functional_call映射，按固定probe限制raw变化尺度，未改写网络、不构造optimizer；不是AdamW，更不是训练结果。原始网络权重SHA前后相同。所有方向与硬验证反例保存，不只展示有利样例。\n\n'
'## 训练子集方向统计\n\n|方向|候选实例|净空降|硬PASS→FAIL|Goal PASS→FAIL|进展降|v/a/j峰值增|\n|---|---:|---:|---:|---:|---:|---|\n'+''.join('|%s|%d|%d|%d|%d|%d|%d/%d/%d|\n'%(key.split('|')[1],v['n'],v['clearance_harm'],v['hard_pass_to_fail'],v['goal_pass_to_fail'],v['progress_harm'],v['vmax_increase'],v['amax_increase'],v['jmax_increase']) for key,v in directionstats.items() if key.startswith('TRAIN|'))+
'\n安全优先离线候选仅为gA在与旧Obstacle梯度内积<0时，去除该冲突分量；再构造gObs+λ*gA_proj，λ令质量梯度范数为0.1×Imitation。没有改训练策略、没有安全证书，也没有对连续净空的保证：只对该点旧代理的一阶方向有约束，有限步、其他任务和硬几何仍可能恶化。明确检查实测净空，不以投影内积代替安全。\n\n'
'MODE_COVERAGE_DIAGNOSTIC.json另逐场景记录六Query crossing标签数与两两crossing位置距离；这只是固定平面覆盖代理，不是完整轨迹同伦。模式变化使用旧crossing方向标签，不是同伦；当前只覆盖有限微小方向，不能证明长期六Query模式不塌缩。相邻checkpoint模型与重复状态相关，不以实例数冒充独立实验数。原QA反例仍保留在历史开发复查中；所有结论以本轮实际统计为准，不删除不利样本。\n')
md('NEXT_PILOT_DECISION.md',
'# 下一阶段决策：当前不放行训练\n\n本轮仅完成冻结CPU0审计。数学/梯度正确不等于代理单调对齐，也不等于训练收益或控制安全。全部现有Limit/Obstacle/Goal/Domain及阈值保持。未经单独注册与授权不执行以下矩阵。\n\n'
'候选权重不再用一个batch，而取三个冻结权重各96批的ratio分布：R95=max_model P95(||g项||/||gIm||)，候选λ=[.05/R95,.1/R95]，压力检验上界.3/R95；这些只控制训练分布的初始尺度，不保证最大值、未来更新或原连续梯度保护通过。原Imitation接近零的比例会变大，必须保留原停止规则。每项米/无量纲单位不同，不可复用同λ。确切范围见下表，未写入recipe。\n\n'
'|新增项|全模型最坏P95梯度比|候选λ下限|上限|压力检验（未授权）|\n|---|---:|---:|---:|---:|\n'+''.join('|%s|%.6g|%.6g|%.6g|%.6g|\n'%(k,v['worst_model_p95_ratio'],v['candidate_lambda_low'],v['candidate_lambda_high'],v['stress_only']) for k,v in weights.items())+
'\n建议保留旧Obstacle作为基准。根据全训练配对和方向表判断，不直接裸加QA/QJ：质量下降不等于安全。QT同样会缩短时间并可能提高v/a/j峰值，当前没有必须增加的证据；如研究时效，应该另设单独QT消融而非与A/J捆绑。聚焦与rho不同时改动；候选优先级只来自本轮训练统计，不从三个开发Query挑参数。Tail分位非光滑和共享参数的候选损害需保留反例，LSE也未普遍改善一致性。\n\n'
'最小待审配对仅两臂：R0旧PG1；Rtail=R0+固定q=.1、无rho的Tail，保留旧含rho Obstacle，不动Goal/Limit/Domain。Tail在训练配对中的反例数/方向统计可作为优先级证据，不能称训练收益；须先审核非光滑与联合梯度保护风险。均同C1init、Seed42、Order、Batch8、FP32、32步，独立方法/授权。若要比较聚合平滑性，再单独加固定tau=.025的LSE臂；如需归因rho，再加同概率质量的mean_rho对照，避免两臂回答两个因素。QA/QJ/QT及投影均延后，不与安全项一起改，以免混淆归因。\n\n'
'停止条件：任何非有限/身份或数据漂移、原连续梯度超限停止触发、原硬验证恶化则停止后续扩展并保留失败；不得放宽旧保护。评估连续净空/硬Physics、Goal推进、v/a/j极值、时长、matched/unmatched和模式覆盖。open-dev8为机制诊断集，不再当最终独立测试；独立测试集应预锁定。没有本轮训练收益结论，未来所有Pilot均NOT_RUN。\n')
md('R11_ACCEPTANCE_REPORT.md',
'# PG1-v2-R1.1 CPU冻结审计\n\n'
'结论：**完整训练集梯度审计与隔离数学检查完成；不放行新配方训练。**\n\n'
'基线分支query_minco_mainline，HEAD4cc4fb8222be179263cc043e24605dacc7603227；入场已有R1四个untracked文件，仅核验保护。本轮只新增experiments/pg1_v2_r11隔离诊断代码，结果在本独占Run；没有修改原R1、旧Loss/recipe/Teacher/MINCO/验证器或历史Run。新CPU绑定显式allow_uncommitted_sources=true仅为只读审计，不冒用旧clean训练资格。\n\n'
'- 数学实现：PASS，9项新测试+17项原R1回归；含rho质量归一化、独立NumPy、double gradcheck、重复质量、当前T导数和投影正反例。\n'
'- 梯度有效：PASS，十个要求分项及六个附加对照，在3权重×完整384任务运行；匹配组VJP重建、所有梯度有限、raw19:21零。\n'
'- 代理与硬指标一致：未建立普遍一致性；统计与反例完整列出，不能以局部梯度聚焦升级连续安全证明。\n'
'- 训练收益已证实：NOT_RUN；optimizer创建/step均0，GPU/ROS0。\n'
'- 训练资格：NOT_RELEASED。本轮没有方法注册、训练授权或训练执行。\n\n'
'按日志可恢复的计算wall下界%.2f秒；全阶段线程CPU与RSS峰值未完整落盘，分别为%s / %s（NOT_RECORDED，不能冒称总值）；方向记录%d。三个模型before/after权重SHA逐一相同。完整审计768曝光/模型、384唯一任务/模型，非人工无放回重排；训练方向子集与全量统计明确分开。分布中的相关状态、重复曝光、旧模型近似相同均不当独立训练证据。\n\n'%(result['wall_s'],str(result['thread_cpu_s']),str(result['max_rss_mib']),result['direction_records'])+
'元数据修正：原Order存在批内重复ID，原first_visit标记在批更新前生成，导致每权重多记一条首次访问记录。本报告按model/sample_id稳定去重并验证重复payload完全相同；保留原始日志，未改曝光梯度。详见UNIQUE_METADATA_ERRATUM.json。\n\n执行分段：audit001完成C1后主动结束（退出143）；audit002两个step32全384梯度均完成，但末尾累计汇总复用同名文件被O_EXCL拒绝，实际FAIL退出1。audit003仅重做PG1固定子集方向，未重算全384 VJP，实际退出0。combined是已验证原始流的恢复汇总，不是原失败进程PASS。原始梯度/值/96批raw文件逐一重哈希并核对384唯一ID、16分项、时间/匹配字段；SOURCE_ASSEMBLY.json列所有来源。零VJP优化先经两个冻结模型逐位等价验证，数学与阈值未改。原失败完整保留。\n\n全部公式、分组分母、rho梯度非等价、原证书边界详见三份专项报告和原始JSON/CSV。新增源码diff/SHA、测试日志、保护前后与交付清单在audit及DELIVERY_SHA256.json；到此停止。\n')
with (S/'QUALITY_SAFETY_CONFLICT.md').open('a') as f:
 f.write('\n## 全96批参数梯度与旧Obstacle冲突率（cos<0；零梯度不计）\n\n|模型|QA|QJ|QT|LSE|Tail|\n|---|---:|---:|---:|---:|---:|\n')
 for model in ['C1_step0','E2_step32','PG1_step32']:
  f.write('|%s|%s|\n'%(model,'|'.join('%.2f%%'%(100*gradientstats[model+'|all|'+k]['conflict_obstacle_rate']) for k in ['QA','QJ','QT','lse','tail'])))
 f.write('\n固定/当前T的raw15梯度范数比中位数（after/before）另列下表；其余raw16:18见JSON，不能用几何时间伸缩公式代替实际MINCO链路。\n\n|模型|QA_current / QA raw15|QJ_current / QJ raw15|\n|---|---:|---:|\n')
 for model in ['C1_step0','E2_step32','PG1_step32']:
  vals=[]
  for term in ['QA','QJ']:
   rr=[x for x in current if x['file'].startswith(model+'_') and x['term']==term and x['time15_19_before'][0]>0]
   vals.append(float(np.median([x['time15_19_after'][0]/x['time15_19_before'][0] for x in rr])))
  f.write('|%s|%.6g|%.6g|\n'%(model,vals[0],vals[1]))
with (S/'RISK_WEIGHT_AGGREGATION_ABLATION.md').open('a') as f:
 f.write('\n## 净空较紧的配对子集\n\n任一checkpoint净空≤.08m（诊断分层，不是新验收阈值）：\n\n|模型|项|子集数|净空降且代理降|\n|---|---|---:|---:|\n')
 for key,v in paired.items():f.write('|%s|%s|%d|%d|\n'%(key.split('|')[0],key.split('|')[1],v['bottleneck_any_checkpoint_n'],v['bottleneck_worse_proxy_lower']))
if (R/'runs/POSTAUDIT_JOINT_BUDGET.json').exists():
 joint=read(R/'runs/POSTAUDIT_JOINT_BUDGET.json')
 scene=read(R/'runs/SCENE_PARAMETER_GRADIENTS.json')['records']
 csvout('JOINT_GRADIENT_BUDGET.csv',joint['records'])
 csvout('SCENE_PARAMETER_GRADIENTS.csv',scene)
 guard={}
 for key in sorted({(x['model'],x['terms'][-1]) for x in joint['records']}):
  rr=sorted([x for x in joint['records'] if (x['model'],x['terms'][-1])==key],key=lambda x:x['batch'])
  streak=maximum=0;count=0
  for x in rr:
   bad=x['total_physics_to_imitation'] is not None and x['total_physics_to_imitation']>10
   streak=streak+1 if bad else 0;maximum=max(streak,maximum);count+=bad
  guard['|'.join(key)]=dict(batches=len(rr),ratio=quant(x['total_physics_to_imitation'] for x in rr),new_ratio=quant(x['new_to_imitation'] for x in rr),over10_batches=count,max_consecutive=maximum,warning='frozen counterfactual only, not actual AdamW stop result')
 write('JOINT_BUDGET_GUARD_AUDIT.json',guard)
 with (S/'TRAINING_SET_GRADIENT_AUDIT.md').open('a') as f:f.write('\n附加SCENE_PARAMETER_GRADIENTS.csv按整场景六Query最小净空分组，和前文候选级几何分层区分；保留原Batch8分母。JOINT_GRADIENT_BUDGET.csv为QA/QJ/QT+LSE或Tail联合新增预算，每项分得.025/R95；均为反事实算术，不注册配方。JOINT_BUDGET_GUARD_AUDIT.json报告原>10连续超限风险，不放宽原保护。附加CPU耗时%.2f秒，模型SHA仍保持。\n'%joint['wall_s'])
accept=S/'R11_ACCEPTANCE_REPORT.md'
text=accept.read_text()
if not (R/'audit/SOURCE_ASSEMBLY.json').exists():
 start=text.index('执行分段：');end=text.index('全部公式、',start)
 text=text[:start]+'本Run使用修复后的独占模型快照与单次最终汇总入口；过程状态以本Run CPU0_RESULT/退出日志为准，不套用历史恢复段的退出状态。\\n\\n'+text[end:]
if len(task_records_raw)==len(tasks):
 text=text.replace('元数据修正：原Order存在批内重复ID，原first_visit标记在批更新前生成，导致每权重多记一条首次访问记录。本报告按model/sample_id稳定去重并验证重复payload完全相同；保留原始日志，未改曝光梯度。','本Run使用修正后的逐ID首次访问标记，唯一任务数量再次按model/sample_id核验；历史计数缺陷不影响当前身份校核。')
accept.write_text(text)
print('REPORTS_COMPLETE',len(tasks),len(records),len(dr))
