# Certified-empty P0 修复审查（2026-10-26）

起始HEAD为349569ecb940f74a6f08be93c9c23954222bb958，与任务固定提交一致；起始分支query_minco_mainline、工作树干净。修复分支为fix/certified-empty-p0-20261026，未提交、未Push。

## 缺陷成因与最小边界

可信训练侧车允许形状(0,2,3)且empty_certified=True，源scene还必须有certified_empty=True。physical_loss已有空场可微零分支，但后续unsafe仍把空集合传给故意只支持非空的box_sdf。开发集和独立审计另有stack/min空归约。E0虽然没有物理loss，也会进入unsafe诊断，因此同样受影响。不能据此宣称任何历史Pilot32实际崩溃。

实际又发现validate_teacher最后的dense诊断min([])会早于开发集的stack([])失败。用户已单独明确授权只为这一行加入default=np.inf；未改任何硬判据或阈值，额外文件仅此一行。原始提案和授权范围记录在本轮审查输出。

- pilot_worker.py：先依赖geo.load完成真实来源/锚点/张量认证，再复用validate_semantics及shape绑定检查；认证空场unsafe为六个False，记录certified_empty与sampled_collision。非空公式保留。
- pilot_evaluation.py：空列表必须有明确source认证与free类型；独立evaluate_model遇到空场还会核验原dev-source锚点。缺标记/缺来源/错SHA/矛盾标记拒绝。空场采样碰撞与裕度违反为False、净空null；原validate_teacher和task_metrics仍独立决定physics/goal/oracle。
- audit_rsps.py：实际侧车认证后，空场只在局部计算中使用+inf；输出净空为null及明确布尔数组。domain诊断照算。numpy_physical的数学空集合零值原已存在，没有改变。
- evaluation/validation.py：仅已授权的单行诊断min默认值。box_sdf、certify_piece、domain/动力学/任务阈值不变。

## 诊断语义

| 输出 | 定义 |
|---|---|
| worker.sampled_collision | 每Query采样SDF最小值<0.12m（身体半径）的布尔值 |
| evaluation.minimum_obstacle_clearance | 每Query采样中心点到AABB并集的最小signed SDF，单位m，未减身体半径；认证空场为null |
| evaluation.sampled_collision / sampled_margin_violation | 原采样阈值0.12m / 0.15m，认证空场均False |
| audit.minimum_sampled_surface_clearance | 原采样SDF减radius，认证空场为null |
| audit.sampled_collision / sampled_margin_violation | 该surface clearance<0 / <margin |
| conservative_collision / physics / goal | 沿用原独立数值证书及任务验证，不以空场标记覆盖 |

这些采样诊断没有连续安全证明含义。原验证器内部的空障碍无限净空哨兵仍是+inf；P0的JSON出口不序列化该内部证书dict，而输出null和布尔结果。测试对所有新增公开评估/审计结果使用allow_nan=False。

缺失几何、读取失败、未认证或错误坐标来源不能被猜成free。旧的无明确empty证书的开发集不会被自动补标；应保留旧文件，另建经审核的绑定。本补丁不授予旧收据新代码身份：原source SHA、spec和授权检查仍会拒绝不匹配的当前代码。审计历史Run应使用其封存源码，不重写旧spec/收据来绕过校验。

## CPU与静态结果

解释器/home/lizheng/anaconda3/envs/yopo/bin/python；Python3.8环境，PyTorch2.4.1+cu118、NumPy1.22.3、SciPy1.10.1；CUDA_VISIBLE_DEVICES为空。pytest未安装，改用已有unittest，不安装依赖。

最终44项：40通过、4跳过、0失败。新增P0测试14项均通过，其中包含：
- free/obstacle/mixed × E0/E1/E2/E3，共12个完整batch及pilot_audit组合；实际QueryMincoPlanner四臂混合输入前向/反向，参数不更新；
- E0 raw梯度与原imitation逐项一致，19:21梯度为零；四臂非空数组与HEAD旧实现逐字节一致；
- E1/E2/E3独立cpu_dry_run/audit_record全链路、NumPy对照、认证与内容篡改负例；
- 完整evaluate_model三种场景，缺失认证/来源/错误SHA拒绝，JSON严格序列化；
- 空场的domain/目标/动力学失败仍失败，非空硬验证器完整返回值与旧实现一致；
- 复现旧空路径的真实异常，证明没有用跳过/模拟硬验证器替代完整评估。

真实测试跳过：GeometrySidecar的384样本/三Seed及篡改验证3项，Trainer real integration 1项。RSPS_SIDECAR未由用户显式提供，RSPS_AUDIT_OUTPUT也未指定；未自选旧输出或补环境变量伪称通过。合成RowSidecar只使用原逐行认证路径，不等于384行完整资格。Gate E原测试依赖RSPS_CPU_TEST_ARTIFACTS及历史cpu_configs003，未运行该历史输出驱动套件；本轮四臂合成测试替代的是bug回归，不是实际Pilot32资格。

首轮新增夹具缺order.ids导致1个测试错误，已补齐夹具，原失败日志保留；没有因此改生产门禁。最终全部相关测试通过。

```bash
cd /home/lizheng/query_minco_mainline
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH=$PWD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/lizheng/anaconda3/envs/yopo/bin/python -B -s -m unittest discover -s tests -p test_certified_empty_p0.py -v
```

另已运行compileall -q query_minco/training tests及git diff --check，两者exit0；字节码重定向到本轮输出的pycache，未写历史源码目录。完整逐suite命令/分母见research_outputs/rsps_p0_certified_empty_v1/review001/TEST_RESULTS.json与logs/FINAL_CPU_TESTS.log。

## 保护与停止

历史保护清单见本轮PROTECTION_AFTER.json：原C1/E0–E3配方、Teacher、Decoder/MINCO、objective、Trainer、geometry_sidecar、box_sdf/physical_loss/reduce_set、F3/ROS、五冻结checkpoint及旧Pilot32 seal核验。验证器除了用户授权的单行诊断归约之外保持原样。

PG1仅有docs/PG1_TASK_COMPLETE_V1_DESIGN.md，无函数实现、注册、训练配置或优化更新。本轮未构造训练optimizer、未执行step、未启动CUDA Pilot32/正式训练、ROS/飞控，未覆盖历史输出。P0到此停止。
