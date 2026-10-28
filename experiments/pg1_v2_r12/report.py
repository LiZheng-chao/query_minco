"""R12 pretraining reports from actual CPU0 files. No optimization."""
import argparse,json,csv,collections,sys
from pathlib import Path
import numpy as np
from query_minco.training.io import read,sha
from query_minco.training.pg1_v2_r12.spec import R0,RTAIL
p=argparse.ArgumentParser();p.add_argument('--cpu0',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=a.cpu0;o=a.output;o.mkdir(parents=True,exist_ok=False)
result=read(r/'audit/CPU0_RESULT.json');assert result['passed'] and result['optimizer_steps']==0
records=[json.loads(l) for l in (r/'runs/BATCHES.jsonl').read_text().splitlines()]
def save(n,x):(o/n).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def md(n,s):(o/n).write_text(s,encoding='utf-8')
def q(xs):
 xx=np.asarray(xs,dtype=float)
 return dict(n=len(xx),min=float(xx.min()),p50=float(np.median(xx)),p95=float(np.quantile(xx,.95)),max=float(xx.max()),mean=float(xx.mean()))
stats={}
for method in [R0,RTAIL]:
 rr=[x for x in records if x['method']==method];assert len(rr)==96
 stats[method]=dict(ratio=q([x['physical_imitation_gradient_ratio'] for x in rr]),single_alarm_batches=[x['batch'] for x in rr if x['ratio_alarm_streak']],max_alarm_streak=max(x['ratio_alarm_streak'] for x in rr),
  total_gradient_norm=q([x['global_grad_norm_before_clip'] for x in rr]),
  weighted_tail_to_imitation=q([(.0025 if method==RTAIL else 0)*x['parameter_component_norms']['tail']/x['parameter_component_norms']['imitation'] for x in rr]),
  tail_cos_imitation=q([x['component_cosines']['tail']['imitation'] for x in rr]),
  tail_cos_obstacle=q([x['component_cosines']['tail']['obstacle'] for x in rr]),
  tail_coeff_grad_norm=q([x['tail_coeff_grad_norm'] for x in rr]),tail_duration_grad_norm=q([x['tail_duration_grad_norm'] for x in rr]),
  time_gradients={k:[q([x['raw_time15_19'][k][i] for x in rr]) for i in range(4)] for k in rr[0]['raw_time15_19']},
  parameter_groups={k:{g:q([x['parameter_groups'][k][g] for x in rr]) for g in rr[0]['parameter_groups'][k]} for k in rr[0]['parameter_groups']})
save('R12_BUDGET_DISTRIBUTIONS.json',stats)
fields=['method','batch','loss','ratio','streak','total_parameter_norm','tail','weighted_tail_norm','tail_parameter_norm','tail_cos_imitation','tail_cos_obstacle','matched','unmatched']
with (o/'R12_CPU0_BATCHES.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for x in records:
  w.writerow(dict(method=x['method'],batch=x['batch'],loss=x['loss'],ratio=x['physical_imitation_gradient_ratio'],streak=x['ratio_alarm_streak'],total_parameter_norm=x['global_grad_norm_before_clip'],tail=x['losses']['tail'],weighted_tail_norm=(.0025 if x['method']==RTAIL else 0)*x['parameter_component_norms']['tail'],tail_parameter_norm=x['parameter_component_norms']['tail'],tail_cos_imitation=x['component_cosines']['tail']['imitation'],tail_cos_obstacle=x['component_cosines']['tail']['obstacle'],matched=x['matched_queries'],unmatched=x['unmatched_queries']))
before=read(Path('/home/lizheng/query_minco_mainline/research_outputs/pg1_v2_r12/preflight001/audit/PROTECTION_BEFORE.json'))
histpath=Path('/home/lizheng/query_minco_mainline/research_outputs/pg1_v2_r12/preflight001/audit/HISTORICAL_R0_FINAL.json')
hist=read(histpath)
md('R12_IMPLEMENTATION_REPORT.md',"""# R1.2 实现（2026-10-28）
分支query_minco_mainline，HEAD 4cc4fb8222be179263cc043e24605dacc7603227；入场已有R1/R1.1未提交成果，均只读保留。新增namespace为query_minco/training/pg1_v2_r12、tests/test_pg1_v2_r12.py、experiments/pg1_v2_r12。旧registry、E2/PG1-v1、C1、Teacher/Hungarian、MINCO、硬验证器/F3与权重不改。

R0新隔离别名query_minco_pg1_v2_r12_r0_v1；Rtail=query_minco_pg1_v2_r12_tail_v1。R0直接返回旧forward的loss对象，保持旧FP32归约顺序；Rtail仅加.0025×Tail.mean()。旧五项仍各自原样计算，Dynamic权重.1，Goal/Domain1。R1 Tail函数直接复用；192个固定物理时间节点（3×64）、风险eps=.05m，q=.1，不含rho；r=.12m，margin=.03m。Tail单位米；λ是固定待检验数值系数，不声明最优。排序分位切换非光滑，不签发连续安全证书。

训练适配仅复用原run_engine、durable checkpoint/单次更新意图、AdamW/Cosine和原梯度/开发集保护；新增唯一namespace和严格授权。没有执行训练路径。CPU0期间Optimizer基类构造器被禁止，调用次数断言0。旧raw→MINCO重建2e-4、NumPy重建8e-4、FP32、全部参数连通与有限、duration/coeff梯度、raw19:21和head行、裁剪10等门控均保留。两次连续physical/imitation>10仍立即拒绝，分子包括加权Tail；不是只盯单项梯度。

R1.1复现缺口通过真实三模型独占journal单元回归、批内重复ID测试和本轮全96批单次最终汇总复验；没有把旧失败进程改为PASS，旧完整R11修复后CLI仍NOT_RUN。本轮入口实际正常退出，未覆盖同名累计文件。

初版合成夹具等净空直线使时间梯度严格零，错误的“必须非零”测试失败已保留；改为有曲率夹具后通过，不改Loss/阈值。初版cpu0_001先通过，最终保护审查补齐旧门控后独立重跑cpu0_002；当前引用后者，两个版本源码/收据均保留。计划plans001过时，不用于执行；当前plans002亦尚未授权。
""")
md('R12_CPU0_ACCEPTANCE.md',"""# R12 CPU0 验收
**CPU0数学/数值通过；当前Pilot执行BLOCKED，不等于训练收益通过。**

最终证据：%s。C1 step2880 Seed42初始化，真实Frozen384/Sidecar/Teacher，经原G2资格检查，原Seed42 Order前96批、每批8。每方法768曝光，384唯一任务，六Query各自真实解码；两方法总192批记录。前48批只有288唯一任务，未将曝光数冒充唯一样本数。

全部R0/旧PG1 loss和网络参数梯度逐位一致；两方法raw、系数、时长及旧分项/梯度逐位相同。Dynamic speed/acceleration/jerk hinge分项重算逐位不变。新增Tail的coeff/duration/raw/参数梯度均有限，matched/unmatched参数分解之和恢复原向量；raw15:19保留链路，raw19:21以及head对应行严格零。所有网络权重前后SHA=%s。NumPy独立多项式/SDF/贪心分数质量Tail重建通过；旧数值容差未放宽。

12项R12+17项R1测试通过，共29项；含空几何认证/非法几何、FP64系数时长gradcheck、原图关闭Tail、分位切换反例、原两步保护、R11真实独占写入及重复任务统计。CPU0未创建optimizer，更未执行step；GPU/ROS/Formal2880均NOT_RUN。wall=%.3fs、线程CPU=%.3fs、峰值RSS=%.2fMiB（上限4096MiB、MemAvailable至少3072MiB）。

资产与Source SHA见audit和SOURCE.diff。当前code.allow_uncommitted_sources=true仅授权CPU0；不得拿此收据充当clean training资格。没有删除旧失败、没有改变原半径/裕度或硬验证器。
"""%(r,result['model_weights_before'],result['wall_s'],result['thread_cpu_s'],result['peak_rss_mib']))
b=stats
md('R12_LOSS_GRADIENT_BUDGET.md',"""# 固定λ=.0025 梯度预算
分母为原Imitation网络参数梯度L2；分子为原Obstacle+Goal+.1Dynamic+Domain，加Rtail的.0025Tail。R0物理项的求和顺序完全沿用旧代码。FP32原两次连续>10停止规则不变。以下全部冻结模型，不能预测真实AdamW更新后的32批保护。

|方法|ratio中位数|P95|max|>10批（0起算）|最长连续|
|---|---:|---:|---:|---|---:|
%s
单次报警不是被忽略，而是原合同允许一次并记录，第二次连续才中止；本轮没有发生连续两批超限，未调λ取得通过。
Rtail加权Tail/Imitation范数中位数%.6g、P95 %.6g、max %.6g。总梯度范数、Tail↔Imitation/Obstacle cosine、matched/unmatched参数与raw分布、raw15/16/17/18时间梯度见R12_BUDGET_DISTRIBUTIONS.json和逐批CSV。组范数不相加；R0中tail值与梯度仅作诊断，目标权重严格0。
"""%(''.join('|%s|%.6g|%.6g|%.6g|%s|%d|\n'%(name,b[m]['ratio']['p50'],b[m]['ratio']['p95'],b[m]['ratio']['max'],b[m]['single_alarm_batches'],b[m]['max_alarm_streak']) for m,name in [(R0,'R0'),(RTAIL,'Rtail')]),b[RTAIL]['weighted_tail_to_imitation']['p50'],b[RTAIL]['weighted_tail_to_imitation']['p95'],b[RTAIL]['weighted_tail_to_imitation']['max']))
md('R12_PAIRED_PILOT_SPEC.md',"""# 两臂CPU Pilot32（未授权）
R0=旧PG1五项；Rtail=R0+.0025Tail，q=.1且新增项rho关闭。绝不加入QA/QJ/QT、LSE或梯度投影。计划使用当前WSL CPU以保持历史CPU控制可比；本轮未准备/验收CUDA Pilot。未来GPU需独立合同，不将CPU0冒认为CUDA资格。

新计划文件：research_outputs/pg1_v2_r12/plans002/{R0,Rtail}_RUN_SPEC.json。独占ID A42_R0_R12_CPU_Pilot32_run001、A42_Rtail_R12_CPU_Pilot32_run001。各独立优化器、授权、claim和Run目录。共同C1 step2880初始化，Seed42、原FrozenOrder、Batch8、FP32、32额外更新；总谱系计数新阶段2880+32，不补造更早历史。AdamW betas(.9,.999)、eps1e-8、wd0、LR Cosine2e-5→1e-6端点包含、clip10；公开checkpoint0/8/16/32，durable step日志不是额外评价checkpoint。

每Run上限3600s、RSS4096MiB、MemAvailable≥3072MiB、磁盘预算4GiB。保持原两步ratio和原开发集两checkpoint退化停止；原metric_alarm针对Goal/Oracle/模式覆盖，不因Physics下降自动PASS，Physics负面结果始终保留，不自动扩大训练。中断不自动重试/恢复，保留所有commit与模糊更新证据，不复用授权。

历史R0复用审核：32个真实预更新模型的原数据/Order/LR/损失/raw梯度逐位复现=%s。旧全部代码SHA、旧Run Seal、C1权重及数据资产均核验；因此可标为**历史数学对照合格**，但它仍是旧方法、旧源码身份，不能改写为R12新Run。本轮默认生成全新R0/Rtail两臂，确保新执行/评价bundle相同；若选择节省一次R0重跑，应在单独批准时明确选择历史对照，并引用HISTORICAL_R0_FINAL.json及其旧Seal。

现有阻塞：源工作树尚未clean冻结；本轮CPU0 allow_uncommitted=true；两授权模板approved=false；新训练路径实际更新/新checkpoint端到端验证NOT_RUN。不能用旧CPU0/旧Pilot授权覆盖这些缺口。需审核并冻结源码，再以--require-clean在新目录重做当前身份CPU0、生成新plan，用户逐Run批准后才可validate/run。现有plans002是审阅草案，**不可执行训练**。

评价：所有6Query硬Physics、连续数值最小净空/3cm余量、Goal到达/进展、Oracle@6、v/a/j连续极值、T、Teacher匹配/未匹配、原mode启发式覆盖；训练逐步记录Tail/Obstacle/Imitation损失与参数梯度，checkpoint再做同一open-dev8只读梯度导出。模式不冒称同伦；Oracle不当在线Selector；open-dev8不是独立最终测试集。
"""%hist['mathematical_reference_eligible'])
md('R12_QUALITY_METRICS_PLAN.md',"""# Quality只作只读评价
训练目标仍只有原五项+可选固定Tail。QA、QJ、QT与投影均不训练。

step0/8/16/32在同任务/Query上报告：
- Acceleration Energy=Σ段∫||a||²dt，m²/s³；
- Jerk Energy=Σ段∫||j||²dt，m²/s⁵；
- Flight Time=Σ真实物理段时长，s；
- vmax/amax/jmax采用原多项式极值；最小表面净空采用既有分段AABB根分析，余量=净空-.03m；
- 原Bernstein硬Physics为硬判定权威。数值连续净空不是区间证明；不拿64点代理代替硬几何；
- Goal推进与旧模式启发式；同一q索引不一定对应同一Teacher模式。

evaluation.py复用原硬评价，并额外使用R1独立NumPy多项式积分；原reference检查每checkpoint只保留其原RNG消耗，额外replica为deepcopy，不增加随机初始化消耗。外部既有分析脚本路径与SHA写入recipe并在runtime核对。evaluate_checkpoints.py仅冻结模型前向/反向、无优化器；pair.py按固定任务/Query和初始权重SHA配对，输出CSV。历史R0导出已保留，包括29/48→26/48负面基准，不冒称Tail效果。

若Tail改善净空同时恶化能量/时长或模式，必须一起报告。总loss不同量纲/不同目标，不以总loss下降直接比较训练优势。没有Rtail真实更新结果就没有训练收益结论。
""")
save('R12_GATE_STATUS.json',dict(math='PASS',CPU0='PASS_FINAL_SOURCE',ratio_guard='PASS_NO_CONSECUTIVE_ALARM',pilot_execution_ready=False,
 blockers=['UNCOMMITTED_SOURCE_CLEAN_GATE','CLEAN_SOURCE_CPU0_REQUIRED_AFTER_FREEZE','SEPARATE_HUMAN_APPROVAL_REQUIRED'],
 optimizer_created=False,optimizer_steps=0,training='NOT_RUN',training_benefit='NOT_OBSERVED_NOT_TESTED',
 historical_R0_reference=hist['mathematical_reference_eligible']))
print('R12_REPORTS_WRITTEN',result['wall_s'])
