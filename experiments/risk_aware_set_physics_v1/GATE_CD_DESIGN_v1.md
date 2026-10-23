# RSPS Gate C/D 追加设计 v1

日期2026-10-23。任务rsps_geometry_trainer_gate_cd_v1；沿用feature/risk-aware-set-physics-v1。上轮四份协议原样保留。本文件记录本轮有限实现，960 updates/arm continuation仍为DESIGN_ONLY。

## 资格与来源

新增GeometrySidecar，schema=rsps_geometry_sidecar_v1，仅对既有TRAIN FrozenDataset提供附加几何；原FrozenDataset没有改动。全部384行绑定NPZ/Teacher证书、scene源文件SHA/JSON pointer、原metadata manifest或TASK_PLAN生成封存、合同metadata、scene摘要、B0 FLU、米/秒、球体半径.12m、表面裕度.03m、物理AABB和独立飞行中心域。真实数据只在新输出gate_cd_001/sidecar。
Loader校验原源摘要、384行一一对应和三Seed完整冻结order，load时再次校验源/样本及全部输入Teacher张量。缺绑定、非train、错hash/shape/frame/domain、篡改状态、未认证空场均拒绝。原free场景含墙、地板、天花板、背板，并非空障碍。
飞行中心域是独立可行范围，不伪装为有限厚度墙或再膨胀一次；新物理loss没有额外domain penalty，仅报告采样越域。

历史wrapper来源仍为still_open，blocks_geometry_qualification=false：192条原metadata与原manifest摘要一致，192条TASK_PLAN场景与生成封存及逐样本geosha一致，无需运行wrapper即可确定标签。launch/common不改、不执行；不宣称Teacher全量再生已证明。
原development 216条元数据与train无精确输入或场景重叠。原计划goal/velocity/amplitude/boundary评估刻意复用训练场景，不能作为新几何泛化；geometry/joint场景不重叠。评估元数据只作泄漏审计，没有进入Sidecar或训练loss。

## 可选采样原型

原set_physical_risk五个函数的AST和默认数值行为保持。仅追加refined_segment_samples：
local在粗采样近障碍区作有限midpoint细分；
geometry使用单段五次多项式与AABB面/中心的交点、各坐标极值，再加入相邻事件中点。
按不等间距真实时间梯形积分，不把追加点简单取算术平均。选点拓扑和归一时间detach；反传通过原解析evaluate_piece及dt，忽略root/选点变化，是明确的局部surrogate。

薄障碍.5001–.5003m案例：fixed32/64/128与local32三轮均可能漏占据；geometry在本例命中，但积分误差不优于所有更密fixed/local方案。离散命中、soft积分精度、反传梯度质量是不同目标。
独立评价复用evaluation.validation.certify_piece的Bernstein hull数值保守界；它以原.12+.03阈值工作，有数值reserve，非严格区间算术或飞行部署证明。没有为迎合loss改变阈值。盒子中心零梯度和非光滑切换按原次梯度记录。

本轮新方法recipe仍采用原预注册fixed64/segment；可选refinement只在薄障碍实验，不混入E1/E2/E3因果对比。未来要把采样策略加入训练，须新recipe/method SHA及独立消融。

## 独立方法绑定与审计

新增objective_rsps，三个显式training_method_id：
query_minco_rsps_e1_cpu_dry_v1、query_minco_rsps_e2_cpu_dry_v1、query_minco_rsps_e3_cpu_dry_v1。
入口cpu_dry_run仅CPU、单批forward/backward、更新数0；严格绑定当前Git/source SHA、checkpoint文件及实际模型参数内容、weights SHA、sidecar、seed/order/batch index及recipe SHA。
复用原objective.forward一次，保持六项imitation与B_set；构造全6Q物理项。分项autograd探针后清零raw/duration保留梯度，再算总loss.backward，避免探针污染最终梯度。
E1 beta0；E2 beta1、vref1、clip至3，风险权重detach后按时间质量归一；eps.05m、radius.12m、margin.03m、lambda1；E3每Query强度a_m=1/a_u=2再按每样本总强度归一，之后batch平均。原recipe与960步计划未事后改写。

audit_rsps从保存raw重建MINCO/时间样本、原六项cost与SciPy Hungarian；独立NumPy实现SDF/法向/接近加权/积分/组分母，重建总loss。审计记录绑定源、真实输入顺序和初始化，校验raw/duration梯度有限性、组梯度和分项相加关系以及19:21为零。
原run_spec/loop/audit/objective完全未改，旧components明确拒绝新ID。此设计提供可审计的CPU扩展接缝，尚未接入既有正式cli train/preflight，也没有替代其隔离/授权/内容审计生命周期。

诊断包含matched/unmatched物理梯度、物理/imitation梯度比及夹角、停滞Query数、采样球碰撞/裕度/越域、.25m坐标RMS贪心不同路径数。该路径数不等于有效独特绕障拓扑；真实Physics/Goal/Oracle和有效模式须由独立冻结评价合同计算。

## 下一轮建议（未授权）

READY_FOR_BOUNDED_PILOT=false：当前仅CPU零更新身份，没有合格GPU执行/隔离训练profile。先单独申请GPU单批零更新preflight与既有受审计生命周期的最小接入；不能把本轮收据视为GPU授权。
若该门通过，再申请Seed42、E0–E3各固定32 updates的配对pilot（总128次更新），同起点/order/LR/recipe，固定存点0/8/16/32。此pilot与960步正式比较分开登记，不以best checkpoint选胜者，不改上轮960预算。
开放开发集预先冻结，blind禁止调参。每存点独立算Goal、Physics、Complete Oracle与通过完整物理/Goal后的独特模式数；按scene聚类，不把6Query当独立样本。
建议预先登记停止警报：非有限值、身份漂移或raw19:21异常立即停；两次检查物理/imitation梯度比>10，或有碰撞但物理梯度持续全零则暂停归因；两次检查Goal/Oracle低于各自step0超过5个百分点，或完整合格模式数相对下降>20%则停止整组配对pilot并报告部分预算，不续调loss掩盖失败。数值门与科研指标警报分开，不作为统计显著性检验。
当前A44部分样本本来就只有1个RMS路径模式，须相对其自身step0判断，不能把起点差异归因于本轮loss。正式GPU训练/ROS仍需新授权。
