# E0–E3 实验预注册 v1

登记日2026-10-23。状态 DESIGN_ONLY / NOT_AUTHORIZED。以下是后续候选配方，不是可执行RunSpec；本轮正式更新为0。修改配方须版本化、在首次训练前冻结并记录原因。

## 假设与因子

H1：全候选净空监督改善6Q物理质量（E1-E0）。
H2：接近加权改善高风险学习信号（E2-E1）。
H3：分组分配改善未匹配候选而保留Oracle与模式（E3-E2）。

E0=原Hungarian六项imitation；E1=E0+lambda*uniform(phi)；E2用相同phi、lambda但时间均值1的detached接近权重；E3用相同risk核和总系数质量，但a_m=1,a_u=2。beta=0及a_m=a_u均为实现等价负对照。
初始候选超参数：eps=.05m,beta=1,v_ref=1m/s,w_max=3,lambda=1(原imitation混合单位下的固定系数)，radius=.12m,margin=.03m，每段64点。仅合成证据，不宣称这些值最优。未归一化risk只作诊断。
E3有效组均值系数：M1时lambda_m=lambda/11、lambda_u=10lambda/11；M4时lambda_m=lambda/2、lambda_u=lambda/2。E2系数为lambda*M/6和lambda*(6-M)/6。每样本先归一化，再batch平均，报告系数与实际梯度，不能仅凭整体增大loss解释收益。

## 起点、预算与配对（待资格与下一轮授权）

选择各Seed历史C1 step2880 weights-only continuation，E0也同样继续训练；禁止拿旧未更新C1充当等预算对照。A42锚：
/home/lizheng/reach_planner/query_minco/outputs/b2c1/train_v3/A42_C1_orthogonal/checkpoint_step2880.pt，
SHA256=2352de6dd1e4a839b78ec12c115b5319b34da3c61cada105dbdeb24a4faec6d1。
其余A43/A44必须在新授权前实际核验SHA，不用猜测路径填入RunSpec。

候选每arm新增960 updates，batch8，fresh AdamW WD0/clip10，cosine LR=2e-5到1e-6，无warmup，固定存点0/320/640/960。主比较固定step960，不能挑best。预算尚未注册到现有Trainer。
全部arm使用同seed初始化、同384行训练manifest、同原C1 order前960批、同网络/6Q/21D、同M1/M4和B_set/六项权重；sidecar不影响order。四角色每批各2，960*8=7680 task曝光/arm。三Seed 42/43/44逐一配对。
先Seed42开放开发集小规模研究，再预先冻结三Seed确认配置。Seed42若用于调参不能同时当独立确认样本；确认报告明确其开发复用，或另行预注册未使用Seed，不能偷偷增加种子。调参次数、所有失败保留；blind禁止调参。B1重新训练属于另一套实验，不与continuation混合。

## 终点与统计

主终点：6Q collision比例、原Physics PASS、Complete Oracle、有效独特绕障模式数、matched/unmatched分层碰撞比例。沿用现有独立几何/动力学/Goal阈值；匹配仅在有Teacher的离线train/dev诊断可定义，不能在无Teacher评估集伪造分组。
次终点：Goal分解、动力学失败、P95推理耗时；独立Selector的selected success留待其单独验证，当前NOT_RUN。
记录每样本/Query的Limit、Lphys、原始/归一risk、组分母/系数、raw前19维物理和imitation梯度范数及夹角、轨迹最小净空、碰撞风险、Query总时长。不可把共享网络梯度解释为raw直接监督。

统计单位为scene/family（共享同几何的Goal与6Q一起聚类），不是Query。逐scene/map按相同起点成对比较，报告绝对比例及百分点差、每Seed结果；固定seed20261023的10000次配对cluster bootstrap、95%区间。只有很少场景/家族时，明确小cluster局限；不把12个网络Run当作大量独立场景。
三个预设对比E1-E0/E2-E1/E3-E2均报告。若做显著性检验，对主终点/对比使用Holm校正；描述性CI与原始失败完整呈现。Complete Oracle和有效模式若下降，先做失败归因，不自动增添diversity项。场景或数值失败不得从分母移除。

## 进入训练前的门

GT_GEOMETRY_BINDING=NOT_READY，所有arm均不能启动。须先合格sidecar、解释历史wrapper差异、完成新method/training identity、独立objective及loss重建audit、CPU/CUDA preflight、资源资格、固定源码commit/权重/data/order SHA、独占run与研究者scoped授权。森林、YOPO同控制器闭环、ROS桥接均另案。
