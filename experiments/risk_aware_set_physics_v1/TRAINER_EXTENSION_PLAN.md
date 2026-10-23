# Unified Trainer 最小扩展计划（本轮不实施）

现有 query_minco_imitation_e1_trainer_v1 与 query_minco_b2c1_c1_e1_v1 保留原行为。本轮新增loss不被run_spec.components或cli train引用。未复制objective/loop/training树。

## 未来变更接缝

1. 新方法身份分别登记E1/E2/E3，例如query_minco_rsps_e1_v1等，新增training_method_id=query_minco_rsps_trainer_v1及continuation profile；不复用原2880 reference身份。现有训练/推理registry权限不改为自动授权。
2. components显式绑定新objective，复用同model_factory和既有postprocessor/MINCO、imitation/match；新objective只组合原六项与独立物理核。必须保持原forward返回合同；额外日志应显式schema版本化，避免修改reference调用约定。
3. datasets/qualification新增train-only geometry sidecar校验、SHA/ID/order一一映射与frame/单位/empty/occupied/domain语义。Worker可见文件仅限声明的train证据与合格sidecar；GT不进入模型forward、在线inference或selector。
4. loop的reference分支继续强制unmatched raw梯度为0；新增身份单独检查phys对unmatched的直接梯度。无风险/认证空场可为0，不能强制所有样本非零；以有明确非零phi梯度的合成/受控探针作为门。raw19:21及对应头参数保持不变。保留共享参数和raw梯度区分。
5. audit按新身份独立重建Limit+lambda*Lphys、每Query/time权重、组分母、effective lambda、detached risk与损失总量；记录原始raw/轨迹、scene与sidecar哈希。不能仅用旧matched cost重建total，也不能删除旧断言让新方法通过。
6. 保留preflight/authorization/bwrap isolated worker/content audit/checkpoint完整状态链。包/source/Git/data/order/device/budget任一改变使收据失效。本轮E0_NUMERICAL_CONFIG只供CPU数值测试，既没有授权也不能复用为训练入口。
7. continuation从各Seed的C1 checkpoint weights-only开始，fresh optimizer；新预算/schedule/profile单独登记。冻结order不重排，M/角色/near-far合同不变。checkpoint记录method/sidecar/source/config/RNG/cursor，独占输出；不覆盖历史run或自动resume。

## 进入后续Gate的验收

原reference单测与116数组回归保持；新objective同时核验旧imitation项完全等价、物理项分组可重建；负例覆盖错GT哈希、重复/缺ID、非train、路径逃逸、错frame、未经授权与错误device。再做隔离一批无更新CPU/CUDA检查。任何正式optimizer.step或ROS需要研究者后续明确授权。
历史两个wrapper与生成封存不一致是既存阻塞，本轮不修复；精确现有SHA与封存SHA需一并绑定，以scene记录与不可变缓存为事实来源，不能宣称完整Teacher生成重现。
