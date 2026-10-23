# 冻结训练 GT 几何资格审计

GT_GEOMETRY_BINDING=NOT_READY；GEOMETRY_RECOVERABLE=true；GATE_A=FAIL（真实训练资格）。
这不是“历史几何丢失”：384个train任务均找到可追溯scene。尚未建立受资格验证的统一sidecar/loader绑定，且生成封存与现有两份wrapper文本不一致。

## 已核验事实

解释器 /home/lizheng/anaconda3/envs/yopo/bin/python；query_minco.__file__=/home/lizheng/query_minco_mainline/query_minco/__init__.py。
C1 manifest：research_outputs/unified_trainer_v1/tests/reference_qualification002_process/qualification/C1_orthogonal_manifest.json，SHA256=641821a98e9521b2492e93fd3c4645d360b9a72581d2ab66eba0b92a466298d4。
order：/home/lizheng/reach_planner/query_minco/outputs/p2c2c1b2c0_preflight_v1/orders/C1_orthogonal_seed42.npz，SHA256=99c51f72f5be4047ee5566754f6fe3aac5d4783c15f0145c533c70d2da084b7d。

- 逐行检查384个显式train NPZ文件SHA、Teacher证书SHA、state和scene来源；没有加载eval/blind NPZ。
- 原Replay 192条：/home/lizheng/reach_planner/query_minco/outputs/p2c2c1a_teacher_v1/dataset/<sample_id>/metadata.json 含完整scene、fixed B0 FLU、body_radius_m=.12、margin_m=.03。全部192条metadata与原dataset/manifest.json中的SHA一致。
- supplement 192条：/home/lizheng/reach_planner/query_minco/outputs/p2c2c1b2c0_preflight_v1/TASK_PLAN.json 的train_orth项，含scene与state。TASK_PLAN SHA=88fbd8313a3e5bb0d3d17ddf9da7ef95e3bbebc2e0cace2c281b9b6054b52912，与原生成封存一致。只按train ID选择scene；没有从评估数据反推训练几何。
- 原geosha定义为规范JSON {obstacles:scene.obstacles, domain:scene.flight_center_domain_m}，sort_keys=True,separators=(',',':')。全部192条重算一致。最初对整个scene做摘要不匹配属于检查范围错误，已在独立followup纠正，未改旧数据。
- seed42冻结order唯一ID集合与384条manifest一一对应。
- 原 p2c2c1b2c0_preflight/{geometry,generate,plan,config,cache_audit311,__init__}.py 对生成封存一致；launch.py、common.py 不一致。这是入场已存在的文本差异，不能把现有wrapper当作历史同版本生成器。未执行或修复它们。
- 原NPZ无完整AABB；depth_m/valid_mask/hit_id不等于全场GT。

完整逐条证据、所有实际输入SHA在新输出 _preparation/{GEOMETRY_AUDIT.json,GEOMETRY_BINDING_FOLLOWUP.json,INPUT_SHA256.json}。未对历史生成器穷尽再生，也未重新求Teacher。

## 几何语义

B0 FLU：前x、左y、上z；场景已在固定B0坐标，无隐式世界坐标转换。camera optical_to_B0=[[0,0,1],[-1,0,0],[0,-1,0]]。障碍完整列表独立于可见性；遮挡不删除GT障碍，无效深度仍是UNKNOWN。free场景仍含墙、地板、天花板和背板，不能把kind=free当N=0。
body=.12m，历史surface margin=.03m（原metadata与p2c0/config.py）。飞行中心域另有约束：不能将域边界的距离再无条件减一次机体半径，也不能认为有限墙板能限制域外全部空间。本核只覆盖明确占据AABB的碰撞净空，域语义待sidecar绑定。

## 下一轮sidecar方案（未实施）

只读源NPZ、Teacher、manifest、order不变。新独占train-only sidecar每行记录：
sample_id、split=train、regime、M、NPZ路径/SHA、原证书路径/SHA、scene证据路径/JSON pointer/SHA、scene canonical SHA、原geosha及算法版本、生成器source SHA/封存SHA、frame、世界到B0变换（本数据identity）、单位、body radius、margin、occupied AABB语义、flight_center_domain、无效/遮挡规则、sidecar自身摘要。
顶层绑定384行manifest SHA和三Seed对应order SHA，按sample_id join，顺序仍只由原order决定；ID重复/缺失、多对一、非train、坐标/单位/哈希异常立即STOP。不得通过省略有问题样本或换order绕过资格。
本轮仅完成绑定设计，不生成正式sidecar，不用候选核训练历史真实数据。下一轮须先解释/锚定两个wrapper的既存差异并验证上述schema与全部三Seed order，才可改变NOT_READY。
