# PG1_task_complete_v1：设计与分阶段执行计划（待审核，未实现）

状态：DESIGN_ONLY。此次代码改动只有P0；本文件不授权实现、注册新方法、生成训练权重或运行优化器。PG2不进入本轮。

## 研究问题与方法边界

问题是在相同数据/初始化/预算下，all-query目标、动力学、center-domain梯度能否提高完整轨迹任务通过率，并增加部署计划中真正保留的网络MINCO比例。假设不是既成结果，也不自动构成新颖性结论。

官方DiffPhysDrone代码main_cuda.py在多时间步调用env.run并汇总任务/平滑等loss，env_cuda.py的RunFunction桥接CUDA动力学前后向；这属于动力学展开路径。PG1只对MINCO解析轨迹状态施加任务损失，没有真实Controller或动力学rollout，不能称完整复现DiffPhysDrone。参见：
- https://github.com/HenryHuYu/DiffPhysDrone/blob/master/main_cuda.py
- https://github.com/HenryHuYu/DiffPhysDrone/blob/master/env_cuda.py

本轮仅核对官方代码；Nature正文访问受阻，没有新增“已完整审阅论文全文”的声明。未来PG2跟踪代理仍需独立方案与授权。

## 新方法和计算接口（拟定）

新方法族query_minco_pg1_task_complete_v1，分别锁定goal、goal_dyn、goal_dyn_domain三个消融ID与recipe digest；不改写任何C1/E0/E1/E2/E3注册身份和旧结果。复用现有训练生命周期，批准后只扩展目标dispatch，不再建Trainer。

基础使用E2的障碍项和六Query等权：E2+goal → +dyn → +domain。E3分组权重保持原方法，作为已冻结描述性对照；是否做E3+PG1另审。raw19/20继续无监督，head19:21梯度为零且权重行保持不变。

拟新增纯函数：goal_task_loss、dynamic_limit_loss、center_domain_loss，以及新方法专属解析高阶导数辅助函数。保留segment_samples返回p/v/积分权重的旧接口；第三项不是acceleration。高阶辅助函数使用真实物理t求p/v/a/jerk，时间与系数都保留梯度，不detach duration。

L = L_imit_matched + L_obs_E2_all + λgoal L_goal_all
  + λdyn L_vaj_all + λdomain L_center_bounds_all。

拟定初始权重λgoal=1、λdyn=0.1、λdomain=1，只是审核用候选值。批准后仅在固定训练校准子集用零更新梯度统计检查量纲、冲突与失衡；如需要调整，先更新并冻结新recipe，不能根据开发集/留出集反复调参。

### 目标项

直接复用evaluation/goal.py及goal_config.py的Near/Far分支与阈值：
- Near：goal范数≤3.4，终点误差≤0.15，终端速度≤0.25。
- Far：距离缩减≥1，投影进展≥1，方向余弦≥0.5，终点射线横向偏差≤0.8；不强迫到达远处全局终点。

采用归一化平方hinge代理，各违反量除以对应米/速度阈值。Far余弦分母用明确epsilon防零；角点处只报告选定次梯度，不宣称处处可导。硬评价仍调用原task_metrics，不能以proxy下降替代通过率。

### 动力学与domain项

按每Query总物理时长T归一化积分：Σsegment∫[hinge(||v||/vmax−1)^2 + hinge(||a||/amax−1)^2 + hinge(||j||/jmax−1)^2]dt/T，再六Query平均。

第一版保持原Teacher评价限值3/8/100、时域0.8–2.2s、段长下界0.12，避免一边沿用原Teacher一边静默切换F3执行合同。报告时间分布及触及最大时长的比例，检查仅靠拉长时间降低损失的捷径。

center_domain对经过侧车认证的B0 flight-center上下界做独立、按宽度归一化的越界hinge；它是中心位置边界，不编码成实体occupied slabs。空障碍只令L_obs为零，目标/动力学/domain在适当非饱和样本仍应有梯度。

solve_valid=False或非有限数必须拒绝批次并记录，不能把坏Query置零后只统计剩余者。完整轨迹仍交给原连续数值证书和多项式极值检查；采样proxy不提供连续安全保证。

## 必须拆开的Teacher与执行合同

原Teacher为3m/s、8m/s²、100m/s³；F3执行为1/1.5/10。原状态含1.7m/s初速度，不能在保持该头状态时要求全轨迹速度≤1。静止在2.2s内走2.4m、全程≤1m/s也不可能。因此PG1第一版不承诺原Teacher完整轨迹满足F3。

分别报告：
1. 原始完整MINCO在Teacher/Goal合同下的连续指标；
2. 只读selector/连接/重计时/完整brake之后，在F3原限值下的可执行率；
3. 原MINCO、连接器、制动器各自时长/弧长/进展占比。

若要按F3合同训练，须另审新的时域/局部goal/状态子集及可信Teacher可行性、地图隔离。不能为展示PG1收益放宽现有F3安全阈值。

## CPU验收门（实现获批后）

- free/obstacle/mixed、Near/Far边界、M1/M4、六Query、非法几何/来源、MINCO无效、零向量、单位/帧错误。
- 在非饱和且不处于hinge角点的确定样例上检查raw→MINCO→p/v/a/j→各loss→raw/model非零且有限梯度；空场也必须覆盖goal/dyn/domain梯度。
- 实际管线为FP32。raw中心有限差分以h={1e-2,3e-3,1e-3}检查稳定区域；报告误差随h变化，不能只挑最有利h。标量解析导数辅助函数可另做FP64数学测试，不冒称整个历史MINCO改成FP64。
- raw19:21零梯度；原imitation与E0对照逐位；每项及总和重建；所有输出严格JSON；参数和checkpoint不变。
- 连续多项式极值/几何证书对照采样proxy，记录薄障碍假阴性。第一版固定64点，fixed128/local/geometry refinement是后续独立消融，不混入任务项消融。
- 分开matched/unmatched及M1/M4的每项loss、梯度范数/余弦；记录可行候选数、oracle@6、已知可行模式和候选RMS差异。初版不加多样性loss，不能用撞墙的排斥项凑模式数。

## 预注册实验矩阵与预算（未授权执行）

| 阶段 | 对照/变量 | 预算与准入 |
|---|---|---|
| G1 CPU纯函数/梯度 | E2基础、+goal、+goal+dyn、+goal+dyn+domain | 零更新；上述合成门全部通过 |
| G2 真实CPU零更新 | 人工指定且重新验证的384样本sidecar、三个order、同seed初始C1 | 先每seed/arm一批，完整loss/gradient重建；无optimizer |
| G3 Seed42筛查 | 新鲜配对E2复跑与三个PG1消融 | 4组×32更新=128；batch8，0/8/16/32 checkpoint；逐组单独人类授权 |
| G4 三seed复验（再审核） | 上述4组，各42/43/44 | 共384更新；不是自动2880或正式训练；每seed成对相同初始权重与顺序 |
| G5 独立评价 | 冻结C1/E0–E3描述性对照 + 同预算E2/PG1因果对照 | 相同数据/硬验证器/条件，另审计算预算 |

同seed配对初始模型、样本order、batch8、32更新、cosine lr 2e-5→1e-6、clip10、weight_decay0与旧Pilot合同一致；新增PG1方法自己绑定它们，不改pilot_spec.py旧定义。旧冻结E0–E3与新方法若训练预算/初始化不同，只能描述性比较，不归因于新增loss。seed43/44初始checkpoint真实路径/SHA尚须审批后的资格核查，不填假SHA。

数据仍冻结384训练/Teacher和原开发集合，验证原地图/scene hash隔离。open-dev8只作冒烟筛查；正式研究需事先封存独立留出集，不能把开发8例或32更新当论文结论。旧F1森林资产可只读配对评估，但不当作训练调参集。至少三seed、按共同任务配对bootstrap置信区间，同时报告分母/无合格候选比例，Oracle仅离线上界。

新输出族拟为research_outputs/pg1_task_complete_v1/，每个seed/arm独占Run；训练前锁定dataset/sidecar/order/checkpoint/code/recipe SHA和资源预算。资源上限沿用每Run900s与min(6144MiB,0.75×实际总显存)，不与ROS/渲染并行；这只是审批候选，先由实际预检测量可行性。非有限值、认证失败、19:21梯度、loss重建失败、连续两步梯度比例>10或连续两checkpoint任务退化，沿用原停止规则，不提高门槛。

## 当前可以实际执行的CPU计划

以下只复验已存在的P0/基线测试，产生新的独占CPU输出；不调用尚未实现的PG1接口，也不运行训练：

```bash
set -Eeuo pipefail
cd /home/lizheng/query_minco_mainline
PY=/home/lizheng/anaconda3/envs/yopo/bin/python
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$PWD/research_outputs/rsps_pg1_design_v1"
RUN=$(mktemp -d "$PWD/research_outputs/rsps_pg1_design_v1/cpu_plan_XXXXXX")
mkdir -p "$RUN/tmp" "$RUN/cache"
export TMPDIR="$RUN/tmp" PYTHONPYCACHEPREFIX="$RUN/cache"
git rev-parse HEAD > "$RUN/HEAD.txt"
git status --short > "$RUN/STATUS.txt"
git diff --check > "$RUN/DIFF_CHECK.log" 2>&1
timeout 120s "$PY" -B -s -m unittest discover -s tests -p test_certified_empty_p0.py -v > "$RUN/P0.log" 2>&1
timeout 120s "$PY" -B -s -m unittest discover -s tests -p test_set_physical_risk_v1.py -v > "$RUN/RSPS.log" 2>&1
"$PY" -B -s -m compileall -q query_minco/training tests > "$RUN/COMPILE.log" 2>&1
printf 'CPU_BASELINE_GATES_COMPLETE; PG1_IMPLEMENTATION=NOT_RUN\nResults: %s\n' "$RUN"
```

审批后再实现G1接口、生成真实绑定配置和完整CPU/GPU命令；当前没有PG1 method dispatch，不能提供假装现在可运行的PG1训练命令。执行顺序严格为G1→人工检查→G2→审核/批准G3→再审核G4/G5，任何失败停止下一阶段。CPU门通过也不自动赋予optimizer/GPU许可。

本轮到P0及此设计文档为止，不进入PG1实现、PG2 rollout或F3/ROS。
