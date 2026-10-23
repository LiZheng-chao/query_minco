# Gate E：人工GPU执行 Pilot32 预注册 v1

任务rsps_gate_e_v1，日期2026-10-23。Gate C/D仍为几何/CPU零更新资格。本轮Codex仅开发和CPU验证；任何optimizer.step及GPU预检均由研究者后续手动执行。正式960-step训练、ROS、Push和合并均未授权。

## 方法与数值身份

profile=rsps_pilot32_v1；Seed42；四个training_method_id=query_minco_rsps_pilot32_e{0,1,2,3}_v1。
共同初始化为实际A42 C1 step2880，weights-only/FreshAdamW；原同一个C1_orthogonal_seed42 order前32批，batch8，绝不shuffle。全部384条train数据及三Seed原order SHA、合格Sidecar、Git/package SHA、初始化文件/参数SHA均绑定到新spec。原C1/E0 reference身份和objective/loop/audit不改。
E0原六项B_set imitation；E1–E3仅增加既定全6Q物理项。fixed64/segment、eps=.05m、radius=.12m、margin=.03m、v_ref=1m/s、w_max=3、lambda=1。E1 beta0；E2 beta1且风险detach后真实时间均值1；E3同E2、matched/unmatched每Query强度1:2，再按每样本总质量归一。
改进采样仍是独立原型，不混入本次Pilot。离散loss不是连续碰撞证明；wrapper provenance仍开放，不宣称Teacher全量再生。

## 独立32步schedule

本Pilot登记局部index0…31：lr=1e-6+0.5*(2e-5-1e-6)*(1+cos(pi*index/31))。首步2e-5，末步1e-6，不复用已结束的2880索引，不改变960-step方案。
使用原recipe.optimizer工厂：AdamW betas(.9,.999)/eps1e-8/WD0，clip10；仅按该新schedule设LR，step0 checkpoint也记录首步LR。固定checkpoint 0/8/16/32，不挑best、不resume，准确上限32。Codex没有执行其更新路径，真正step的端到端性质必须由人工Pilot验收。

## 生命周期与权限

新CLI复用既有isolation.launch，增加唯一显式Pilot32扩展。根只读、隐藏历史/全部研究输出/mnt，仅允许文件FD重新暴露；只有当前独占Run可写，无网络。CUDA设备节点只在人工CUDA入口显式dev-bind，WSL /dev/dxg已核实存在，GPU本身未查询或使用。
原隔离器extension=None路径不变。checkpoint仅新增scheduler_id参数，默认原值不变；原checkpoint完整状态/RNG/cursor/sidecar及独占写策略复用。
人工preflight：每arm一批8任务，两次相同参数/相同RNG forward/backward，要求数组逐位可重复；不创建optimizer，更新0。记录Torch/导入路径/设备/峰值内存/独立重建审计。
人工train：需要CUDA实际预检收据与seal、人工Gate E2审核收据、匹配spec哈希/唯一run_id/max_updates32的approved授权。配置或源码改变即失效。当前只生成approved=false草稿，不提供步骤B训练命令。
Worker预算900s；GPU allocated/reserved峰值不超过min(6144MiB,75%物理显存)，起始可用显存至少512MiB。外部用户命令另有1000s/arm时限，四arm串行、任一错误停止后续。预检失败保留现场，只能经审核后新ID重试。

## 固定开发评价与自动门控

冻结DEVELOPMENT_PROTOCOL.json：原开放development中的8个独立几何任务（4Near/4Far），来自family02，训练family00/01以外。选择在任何训练前固定，Teacher/metadata/NPZ/source SHA完整绑定；dev不进入训练监督。
每checkpoint在同8任务×6Q上按原Goal、原validate_teacher及Bernstein保守数值界评价；未证明的几何一律按保守失败。Goal为48候选比例，Complete Oracle为8任务比例；已知可行模式按原C1A canonical_teacher的1025点crossing主导y/z分类，并且必须完整Physics+Goal通过且在该场景Teacher模式中。Selector无资格标N/A。此mode不是严格拓扑证明。
记录未匹配候选的保守碰撞/裕度失败、域越界，以及fixed64采样无违规但保守核未通过的漏报/未证明计数；不把后者全部宣称为已证实的真实机体碰撞。
NaN/Inf、source/data漂移、raw19:21或head行异常、MINCO失败、审计不可重建、OOM立即停止。E1–E3若采样已显示机体碰撞且对应物理raw梯度全零，在step前停止。E0不错误要求物理梯度非零。
逐batch监测物理/imitation raw梯度L2比，连续2批>10停止。开发指标在8/16/32相对step0检查，任意Goal/Oracle绝对下降>5个百分点或合格模式计数下降>20%计一次警报，连续2个检查点报警停止；这是先验风险控制，不是显著性检验。baseline模式数为0时不计算相对下降，明确披露。
四arm串行fail-fast；若后组早停，不能将不同预算结果作公平胜负结论，只能在已有共同checkpoint比较，或经新授权补充配对，禁止自动继续/恢复。

## 验收边界

CPU E0零点逐位回归、原reference单测/隔离、四arm梯度/重建、授权负例、step0 checkpoint保存回读都不等于32步训练已成功。
GPU/CPU重建允许预先登记的数值容差：raw->MINCO/采样rtol=2e-4,atol=2e-4；raw->imitation成本rtol=2e-4,atol=2e-5；NumPy物理重建rtol=2e-5,atol=2e-7。它们是算术一致性检查，未改变任何Goal/Physics安全阈值；不通过时先保留错误，不自动放宽。
READY_FOR_GPU_PREFLIGHT只表示CPU准备完成；没有研究者真实GPU日志时GPU_PREFLIGHT_STATUS=NOT_RUN，READY_FOR_BOUNDED_PILOT=false，READY_FOR_FORMAL_TRAINING=false。
