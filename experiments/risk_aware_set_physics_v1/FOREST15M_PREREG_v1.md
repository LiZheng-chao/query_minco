# FOREST15M_PREREG_v1：冻结模型离线森林配对评价（待输入/输出合同确认）

日期2026-10-24。F0仅批准离线数学/传感快照资格；F1实际运行NOT_RUN。用户确认本合同前，不签发可执行森林批量命令；任务文件§3明确要求“经用户确认输入/输出合同后，再提供手动可执行脚本与独占Run命令”。代码/CPU验证可先完成，不启动GPU、原生渲染或ROS。无训练、无Loss/网络修改。

## 固定方法与数据

五模型：共同C1 Seed42 step2880 + E0/E1/E2/E3 step32，确切路径/文件SHA/weights SHA见preparation001/configs/FROZEN_MODELS.json。已核验四个Run全部RESULT/CONTENT_AUDIT/RUN_SEAL；不选择best、不改变权重。旧8开发任务只作管线smoke，不能用于森林泛化结论。

新森林21个map seed=31001…31021，每7图tree_dist为6/4/3米。仅称生成参数组，真实局部难度由占据/净空/直线阻挡标签事后说明，不能保证密集组必有狭窄通道。原native Maps::forest与深度CUDA渲染工具复用；源码、tree.ply与只读YOPO原件SHA相等。地图60×60×15m、resolution0.1m、camera128×160、fx=fy100、cx79.5、cy63.5、optical-Z/20+Mask，origin=odom，不使用显示TF的+0.05m。

每图3个预先固定yaw=0,+0.5,-0.5；起点从[-12,12]×[-12,12]×[1.8,2.6]、RNG=seed+20261024，依顺序取第一个在原生有限体素域内且0.6m球无占据的点，最多600次；无起点保留失败不重选地图。此为离线数据构造GT筛起点，不是在线Goal/Selector。所有被拒绝起点与原因持久化，在任何模型预测前冻结。

每起点3帧t=0,.2,.4s，pos=pos0+R*[.8*t,0,0]，姿态固定，v_B0=[.8,0,0]，a_B0=0，gravity_B0=[0,0,-1]。这是共享匀速off-policy快照，不是YOPO真实飞行回放或Query执行。后两帧进入障碍也不丢弃。全部五模型看到同一个task_id/depth/state/pose；网络只收depth/state，绝不接收GT、Oracle、未来帧。在线Selector N/A，不定义GT选轨。

global_goal_W=pos0+R*[15,0,0]，每帧local_goal_B0=R.T*(global-pos)按2.4m直线前瞻限幅。2.4米+v.8/a0/重力z-1均在实际384条train状态中出现，支持证据TRAIN_STATE_SUPPORT.json。此策略不是全局寻路，不读取GT来躲开Goal；local Goal占据、无解/未知、域外与裁剪状态均标签并保留。需要小于2.4m或横向/速度不在本轮支持合同的快照标失败，不强行送15m。三段MINCO原时长0.8–2.2s、每段>=.12s，不把local Goal强制等同decoder终端；重规划间隔.2s只是快照采样，不声称实时可达。

总共21图×3起点×3帧=189森林任务，5670候选，按map配对统计，帧/6Q不当独立地图。seeds3/11/17/23为已见smoke，不能混入新集合。新图SHA与已见F0/F0+地图SHA不得相等，训练/开发AABB与native树体素来源分离；本轮是新的探索性森林开发集，不冒称永久盲测。

## 评价合同和已知缺项

复用已冻结F0连续扫掠体素核与F0+动力学核；机体半径.12m+margin.03m，连续几何budget depth16/nodes50000，数值未决/域外UNKNOWN，不能当安全。结构PVA/C0–C4、时长、速度3/加速度8/jerk100、近Goal误差.15m且末速<=.25，保持现有阈值。辅助1025点/段密采样只能给见证或上界，fixed64 Soft Loss仅诊断，离散无违规不证明安全。

输出每Q Physics/Goal/Complete（PASS/FAIL/UNKNOWN）、全6Q Complete Oracle（仅事后）、体积表面净空保守下界/局部截断距离的密采样上界、域外、NaN与UNKNOWN、native深度/编码/network/MINCO/独立评价P50/P95。GT完全可知也不声称在线已观测自由空间，online_safe_status始终UNKNOWN。

新native森林没有冻结独立Teacher集合，不能给六Q伪造Hungarian“未匹配”身份。因此新森林unmatched collision=N/A_NO_FROZEN_TEACHER；全部Q保守碰撞率仍完整报告，已知开发有Teacher则保留真Hungarian结果。若用户要求本轮必须有森林未匹配率，应先提供/另审独立Teacher生成，不能拿模型Oracle、自身输出或“全部未匹配”冒充。

模式提供已通过完整合同候选的几何连接聚类：仅在两轨迹间逐段Bezier控制凸包的连续插值带被体素核证明清空时合并；未证明不合并并标UNKNOWN关系。它是保守几何连接分组，不是证明不同同伦类；0.25m RMS差异只可另作描述，不充当模式数。独立树实例/严格同伦真值缺失时known_homotopy_count=N/A。空场和已证明blocked负例先用原controlled开发控制集单列，不伪造native森林中的空场/无解地图；自然森林中直线受阻不等于无解，未知一律保留。

主要配对delta为E1-E0及E1-C1的per-map平均Physics/Complete/Oracle；E2/E3同样报告。21图macro平均和固定RNG10000次map bootstrap 95%CI为描述性探索统计，不能当最终发表结论。失败含Goal占据、碰撞/裕度、动力学、时长/边界、域外、数值和未决，保留全部失败样本。

## 执行与输出

源码仅新增本实验入口/有限测试/协议/命令文件，现有package/Loss/网络不动；输出独占research_outputs/rsps_forest15m_v1/{preparation001,smoke001,forest_paired001}。绑定三仓HEAD、冻结模型身份、源码/evaluator/native binary/config/地图及输入SHA、goal协议与实际命令。所有原生导出/渲染、批量候选预测由用户亲自在WSL执行，Codex只做有限CPU测试。按先smoke再完整集顺序，失败非零停止，不覆盖/不自动resume。

F2 Shadow/ROS桥只设计：YOPO唯一控制发布者，Query仅可视化/诊断；本轮不实现或运行ROS桥。F3闭环STOP：没有合格Selector、连续可观测扫掠体积、watchdog、认证制动、同步动态a0及控制跟踪验收。旧v4 A/B/C结果REJECTED，不重分类。

确认项：五冻结模型；21新native森林/189匀速快照；global15m/local2.4m；新森林未匹配率与严格同伦N/A，完整候选物理/Goal/Oracle为主；原生渲染GPU与批量运行由用户执行。确认不授权训练、Shadow执行或闭环。
