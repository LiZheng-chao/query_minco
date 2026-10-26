# PG1 第一阶段：CPU 实现与未执行规格

本阶段只实现新损失、独立资格/审计、CPU0 和训练规格准备；不提供训练 executor，不创建优化器，不执行更新。G1 合成通过不等于 G2 或训练准入。用户已明确 G2 保持 NOT_QUALIFIED。

## 新身份和复用边界

代码在 query_minco/training/pg1/，复用原网络、MINCO、Hungarian、segment_samples、E2 obstacle loss 和硬验证器。没有修改旧方法 dispatch、C1/E0–E3 配方、Teacher、F3 或 ROS。

四方法依次为 query_minco_pg1_e2_paired_v1、query_minco_pg1_goal_v1、query_minco_pg1_goal_dyn_v1、query_minco_pg1_task_complete_v1。新增 goal/dyn/domain 权重分别为 (0,0,0)、(1,0,0)、(1,.1,0)、(1,.1,1)。配对 E2 在没有新增目标项时保留旧图与数值。

## 目标与几何语义

认证原始 sidecar/anchor/tensor 后重哈希源 scene/state，调用未修改的 separating_slab_proof。只有可信 proven_infeasible 才屏蔽目标项；没有证明的任务仍有正常 Near/Far 目标监督，不宣称可达。掩码不屏蔽任何已启用的 obstacle/dynamic/domain 项。所有六 Query 受这些物理项监督，只有 imitation 依赖匹配。消融中未启用的项仍计算、记录但权重为零。

Near/Far 阈值及任务含义复用原 goal_config。goal 项为原判据的归一化平方 hinge 代理，不是硬证书。Near/Far 分支按实际存储 FP32 state 提升 FP64 后选择，与原评估一致。统计 goal_mask、reason、masked row/query count；分母仍为 B×6，不因屏蔽改变其他样本尺度。

无证明 M0 明确拒绝。合成证明不可达 M0 支持零 imitation；原真实 Frozen384 仍只接受已验证正 Teacher M1/M4，如实际正 Teacher 与不可达证明矛盾，资格失败，不重写 Teacher。

dynamic 为解析多项式 v/a/jerk 的 3/8/100 归一化超限平方积分代理；center-domain 是中心越界量按轴宽度归一化的平方积分。新 Horner 求值不改原 segment_samples 三元接口。64 点每段梯形权重、原解码总时长 [0.8,2.2] 和每段 >=0.12 保留。F3 的 1/1.5/10 和执行合同没有改变。薄障碍漏检仍存在，独立硬几何验证不可省略。

raw19:21 不提供安全标签，梯度及输出头相应行梯度保持零。naive unmasked goal 只作离线诊断。记录 matched/unmatched、M0/M1/M4 分层、梯度冲突的非零梯度分母、时长分布和 pairwise RMS。

## CPU 审计

独立 NumPy/多项式重建 P/V/A/J、目标/动态/domain、原 obstacle 和总损失。重算 MINCO、Hungarian、源认证及不可达证据。逐项和完整梯度从已绑定 raw 张量重建且逐位核对；分项反传和整项反传的 FP32 累加残差仅作诊断，不能通过提高容差掩盖梯度篡改。

原始 FP32 raw 有限差分使用全部 h={.01,.003,.001}。小动态/时长梯度单独对相应损失做差分，避免大目标标量的抵消；完整总损失差分仍全部保存，包括噪声，不将其声称为全通道高精度证明。解析辅助函数另有 FP64 gradcheck，不改历史 FP32 MINCO。

空场完整验证器内部 +inf 只在 PG1 严格 JSON 导出层对已认证空场的合法净空字段转 null；其他 NaN/Inf 失败。未修改硬验证器。连续验证失败的随机网络候选保留 FAIL，不因损失可导而宣称安全。

## 数据绑定合同

cpu0 仅接收用户指定的 manifest 绝对路径及 SHA256，schema=pg1_data_bindings_v1，包含 dataset、sidecar、orders(seed42/43/44 artifact)、development(open-dev8)、initializations(seed/checkpoint/weights_sha256/lineage)。artifact 都必须是 path/sha256。lineage 必须是独立认证的 pg1_initialization_lineage_v1，绑定 seed、C1_step2880、checkpoint SHA、weights SHA。

复用原 384-row/三 order/sidecar source gate，开发集隔离携带真实训练 ID 与 scene 集；检查所有原正 Teacher 与不可达证据不冲突。旧 checkpoint 若不符合当前已审阅格式，需要单独适配审阅，不能填造元数据。当前所有真实绑定为 null/NOT_QUALIFIED，未选择任何历史目录替代。旧 checkpoint 的只读 SHA 保护检查不构成 G2 初始化资格。

## 未执行训练规格

profile 独立登记 cpu_zero_update、pilot32、screen3seed、formal2880。pilot32 checkpoint 0/8/16/32；formal2880 checkpoint 0/960/1920/2880。同 seed 各 arm 必须相同初始化、order、batch8 和新增预算。

当前仅设计 paired_C1_finetune_v1：已认证 C1 2880 步之后额外 32 或 2880 步，总 lineage 更新分别 2912、5760。两组均用 PG1 独立 cosine 2e-5→1e-6、AdamW betas(.9,.999)、eps1e-8、weight_decay0、clip10。这是待审微调提案，不是旧 C1 2e-4→1e-5 原始重训练；不能混合报告。

授权模板默认 approved=false，绑定完整 spec digest、单 Run、方法、profile、步数和初始化协议。正式规格要求 allow_uncommitted_sources=false、精确 HEAD/source SHA 和干净工作区。没有训练 executor；模板校验不是一次性授权消费系统，也不能触发训练。未来生命周期应复用既有 checkpoint/ledger，在独占 Run 原子写入，另行实现/测试授权锁存、精确步数和异常恢复。

资源 wall cap pilot900s/formal21600s、GPU min(6144MiB,.75×实际显存)、最少余量512MiB、disk8GiB/run 全部是待测待批上限，不是预测耗时或实际 MX450 能力证明。未运行 GPU 预检。

## 仍未具备的正式训练门槛

1. 用户提供并重新认证真实数据、三 order、开发集、三 seed checkpoint/lineage；补 P0 四项真实 skip 与 Gate E 输出驱动回归。
2. 当前新增源文件未提交，工作树不干净；提交/源锁定须另行处理，不能以旧 seal 冒认当前代码。
3. 运行 G2 CPU0 并审核真实梯度尺度，再独立授权 Pilot32。GPU 时间/显存/磁盘需真实测量。
4. 训练生命周期适配/一次性授权消费及完整 checkpoint 内容审计尚未实现，不可执行正式训练。
5. feasible-mode classifier 与 calibrated selector 未绑定，分别报告 null，RMS 不冒充模式数；正式主终点还需完整轨迹与实际连接+制动保留比例、独立留出集、三 seed 配对统计。open-dev8 不足以支持论文结论。
6. PG2/动力学展开不在本阶段；轨迹可微不等于 Controller/physics rollout 可微。
