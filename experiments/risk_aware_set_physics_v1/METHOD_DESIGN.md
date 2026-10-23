# RSPS v1：独立物理监督原型

日期：2026-10-23。TASK_ID=risk_aware_set_physics_v1；METHOD_ID_DRAFT=query_minco_rsps_v1。研究假设，尚无训练或性能证据。当前唯一算法根 /home/lizheng/query_minco_mainline。

## 原方法与调用链

models.QueryMincoPlanner -> [B,6,21] -> TrajectoryPostprocessor -> PhysicalParameterDecoder -> MincoAdapter / _vendor/minco -> sampler_e1 -> training.objective.forward -> imitation.cost_matrix -> Hungarian match(B_set) -> loop -> audit。
objective.forward 返回 loss, records, raw, output, matrices。所有 6 Query 均先求解；六项 imitation 按 M、再 batch 求均值。raw19/20 不构造轨迹。

禁止修改 models、trajectory、_vendor、matching、objective、loop、audit、run_spec、methods/reference.json 和已有 unified_trainer_v1 测试。本轮仅新增 losses/set_physical_risk.py 与可移植测试，没有正式训练绑定。
YOPO 本地 policy/yopo_network.py 的 forward(depth,obs) / inference(depth,obs) 使用原有 9D observation 与 primitive/head；YOPO-MINCO 是独立 Baseline。它不接受 Query 的 12D/6Q/21D 合同。本轮没有验证其 ROS 运行日志。

## 数学与精确定义

盒子外正、内负：q=abs(p-center)-half；SDF_box=norm(max(q,0))+min(max_axis(q),0)。占据集合用各盒子 SDF 的 hard min。机体净空 d=SDF-r_body-margin，phi=eps*softplus(-d/eps)，单位米。常量几何不求梯度。
hard min/axis max 并列时采用 Torch 第一个索引；abs(0) 次梯度为零。盒子中心可能为零梯度驻点，深穿透并不保证总有逃逸方向。非光滑点只验有限性与次梯度，不强求中心差分相等。

segment_samples 复用现有 evaluate_piece，三段分别包含端点，梯形权重 dt_i/(N-1)/sum(dt)。端点共享的重复样本按两侧半权重累加；不同 Query 独立按各自真实时间归一化。默认每段64点，共192点，非全轨迹64点。只增加物理项的积分，不改变原 imitation 的32点/E1语义。

接近率 dot_d=grad_p(SDF)·v。单独 detached probe 计算法向，避免二阶反传。risk=clip(1+beta*relu(-dot_d/v_ref),1,w_max)，再按各 Query 的梯形时间质量归一化至均值1；整个权重 detach。对时间/速度变化的梯度是 stop-gradient surrogate，并非重算权重后的整个目标的全导数。输出未归一化 risk 作为诊断。
返回每 Query 的时间加权 phi；所有 Query 均有物理目标，在线推理不调用此核。

## E0–E3 因子

| Arm | imitation | 物理核 | 接近风险 | 分配 |
|---|---|---|---|---|
| E0_reference | 原六项/B_set | 无 | 无 | 原 matched |
| E1_uniform_set_sdf | 同E0 | 全6Q | beta=0 | 均匀 |
| E2_approach_risk_set | 同E0 | 同E1 | detached、时间均值1 | 均匀 |
| E3_match_aware_risk_set | 同E0 | 同E2 | 同E2 | matched/unmatched 组权重 |

E3 的 reduce_set 使用每 Query 强度 a_m/a_u：
L_phys=mean_b[(a_m*sum_match Lq+a_u*sum_unmatch Lq)/(a_m*M+a_u*(6-M))]。
换算到任务中的组均值形式：lambda_m=lambda*a_m*M/[a_m*M+a_u*(6-M)]，lambda_u=lambda*a_u*(6-M)/同一分母。相等的“每Query强度”还原E2；相等的两个“组均值系数”一般不还原均匀6Q，不能混淆。M1/M4 单独归一化，再平均 batch。
不使用 GT 最优候选重排 Hungarian，不添加 diversity、动力学 penalty 或 score supervision。

## 学术来源与差异

Zhang, Yuang; Hu, Yu; Song, Yunlong; Zou, Danping; Lin, Weiyao. Learning vision-based agile flight via differentiable physics. Nature Machine Intelligence (2025). DOI: [10.1038/s42256-025-01048-0](https://www.nature.com/articles/s42256-025-01048-0)。题名与作者依据[作者代码仓库](https://github.com/HenryHuYu/DiffPhysDrone)的推荐引用；论文全文本轮访问失败，未声称已阅读全文。代码固定提交271936190b5c2a5e760230e718fa6b167718a5bf，SHA见输出 _preparation/EXTERNAL_PINNED_SHA256.json。

| 维度 | DiffPhysDrone | 本原型 |
|---|---|---|
| 目标 | 滚动控制策略，物理展开 | 六条连续三段MINCO候选 |
| 接近量 | 子时间预测点距离差分、无梯度权重 | 有符号AABB法向·解析速度 |
| 碰撞项 | softplus 与避障 barrier | 长度尺度 softplus 净空 |
| 动力学 | env_cuda 自定义 autograd 与 CUDA kernel | 沿用现有 MINCO |
| 监督分配 | 无 Hungarian 集合分组 | M1/M4 matched/unmatched |

来源：[main_cuda.py](https://github.com/HenryHuYu/DiffPhysDrone/blob/271936190b5c2a5e760230e718fa6b167718a5bf/main_cuda.py)、[env_cuda.py](https://github.com/HenryHuYu/DiffPhysDrone/blob/271936190b5c2a5e760230e718fa6b167718a5bf/env_cuda.py)、[dynamics_kernel.cu](https://github.com/HenryHuYu/DiffPhysDrone/blob/271936190b5c2a5e760230e718fa6b167718a5bf/src/dynamics_kernel.cu)。
main 中 distance 的 dim1 对应 env 的 sub_div 子时间轴，不应误当 batch 或 Query 轴。本轮只借鉴“接近越快则碰撞监督越强”和权重隔离原则；没有复制 CUDA 模拟器、完整 loss 或其超参数。
softplus/接近加权已有先例，集合分配仍需公平实验检验；不能据本轮数值测试宣称独立论文创新或优越性。

## 已知限制

认证空场必须显式绑定；缺几何必拒绝。geometry_bound 是原型调用方断言，不是 provenance 验证器。墙/地板/天花板须显式提供；flight_center_domain 与有限厚度墙板不等价，不能随意将中心域缩放成膨胀障碍。
薄障碍在32/64/128每段点均可漏碰，4097点才命中。离散可微损失不构成连续安全证书。保留现有独立 Physics/Goal 判据。
