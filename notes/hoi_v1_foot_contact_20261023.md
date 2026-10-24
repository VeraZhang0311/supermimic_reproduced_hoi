# HOI v1 双脚—物体 contact 扩展

后续授权更新：用户已要求停止warm、保留旧zero，并启动五通道新版从零实验；随后明确足部与作者ULTRA统一。整脚网格物理扩展和新的验收范围另见[ULTRA足部对齐](hoi_v1_ultra_foot_alignment_20261023.md)。下文243项和「尚未部署」描述本次最初单独head扩展完成的历史时点，不是后续mesh版本已验收或GPU作业的实时状态。

用户要求在 `codex/hoi-v1-no-quality-gate` 增加左右脚—物体contact head和真实监督，并完成测试；现有两个实验不动，先不跑新实验。本次接续该分支已推送的 `04a3796`，所有修改仅在本地源码与CPU检查中进行，没有SSH、部署、取消、续训、提交GPU作业或新建W&B run。

## 五通道合同

固定顺序保留原前三项，再追加双脚：

| 索引 | 真实pair |
| --- | --- |
| 0 | left_hand_object |
| 1 | right_hand_object |
| 2 | object_ground |
| 3 | left_foot_object |
| 4 | right_foot_object |

Contact head仍为learned MLP，共享原联合WM backbone，最后一层从3扩为5：生产网络 `2048→128→5`。主动力学输入224D、输出82D（robot70+object12）保持原定义。实际target是computed：每个20ms control interval `(t,t+1]`内，4个5ms原生solver子步的所有有效pair contact法向力非负部分乘子步时长后相加，以总法向冲量除以20ms，严格大于1N为正例。

足部身份来自实际编译模型的 `left_ankle_roll_link` / `right_ankle_roll_link` 有效碰撞几何，不依据名称含left/foot来泛化到小腿或visual几何；所有pair支持颠倒geom顺序。脚—地面、脚—脚、自碰撞、手—地面不会成为脚—物体标签。源数据39个body contact flags可能混有地面等实体，继续仅作为annotation，不替代原生pair监督。左右脚分别累计冲量、validity和正负计数，保留环境隔离及四子步完整性检查。

Replay的contact/contact_valid/impulse、WM BCE与各通道class weighting、autoregressive/teacher-forced holdout、完整时长评估的实际contact事件和W&B白名单同步扩展。BCE仍对有效通道取平均，`lambda_contact=.1`及既有N/P权重截断不变；新增两个通道自然参与该平均。两通道额外replay存储为每transition18bytes，4096环境×1000容量增加73,728,000bytes（70.3125MiB），资源preflight按实际FIELDS重新计算。

Reference expected/valid仍为左右手两项，contact-match reward、hand-approach、missing/extra计数、critic/goal接口与reward上界保持原定义。脚通道参与动力学辅助监督，不自动变成“要求脚接触”的reference目标。现有holdout no_contact/contact/onset/release phase仍显式按双手定义，脚的预测质量通过五通道precision/recall/Brier和逐通道真实事件另报；不能将手部phase当作全身接触phase。

## 兼容性与验收边界

标签schema升为 `hoi_v1_pair_normal_impulse_v2`，feature版本升为 `supermimic_hoi_v1_full_reference_2`，字段顺序写入checkpoint feature schema、训练plan/metrics/completion及评估元数据。新五通道checkpoint可严格保存/恢复；旧三通道HOI checkpoint在修改live model或Adam之前明确拒绝，不自动补两列标签或强行沿用旧Adam。既有motion-only legacy初始化路径继续逐项校验与拷贝原前缀，新增分支独立初始化；原zero-shot入口不改。

用户随后说明无需补旧checkpoint的resume适配，预计从零开始。此次不实现三通道→五通道迁移；既有 `--from-scratch` 入口保留，首次五通道运行仍须等待用户后续启动授权和新版原生验收，不能据此自动开始训练。

Native验收要求新增左右脚—物体正例及脚—地面负对照；实际探针绑定编译后的足部碰撞几何。旧三通道原生证据不可给新版背书。此次未运行新版Linux GPU物理或4096/H32资源验证，CPU回归不代表脚部probe已通过GPU验收，也不代表踢箱子策略已学会或HOI收敛。现有warm1461248和scratch1461253仍使用不可变source13及其原三通道/持续质量门实现；本分支保留固定500 iter warmup后更新日程。

## 本地验证

本地Python为 `D:/Protomotions/.venv/Scripts/python.exe`，CPU检查结果：

- 全量 `python -m unittest discover -s tests -v`：**243/243通过**，44.010s，无skip。日志在ignored `tmp/hoi_v1_foot_contact_full_tests.log`。
- 旧HOI边界 `python -m unittest hoi.test_reference hoi.test_hand_collision -v`：**6/6通过**。日志在ignored `tmp/hoi_v1_foot_contact_legacy_boundary.log`。
- 针对性network/learner29项、train15项、native-contract18项、gates11项、evaluate/diagnostics/track40项均通过，并已包含于全量回归中，不重复计为额外科学证据。
- 新测试覆盖真实CPU MuJoCo编译模型足部实体绑定、非碰撞visual/小腿/相近旧名称排除；双向foot-object及环境隔离；四子步冲量与严格1N阈值；脚—地面排除；真实sphere支撑面2mm浅压几何；左右脚BCE行和共享backbone梯度、valid屏蔽、class weighting计数与replay保存；五通道合同/事件/白名单及旧三通道拒绝；脚logits/实际接触不改变双手reference奖励；固定warmup后的策略更新仍不受质量比较阻断。
- 独立只读review完成，无剩余P1/P2；复核12个实现文件、对应8套测试和文档。前期发现的共享CONTACT_DIM及CPU preflight兼容性问题已修复。`git diff --check`通过。
- `git diff --exit-code -- supermimic hoi/run.py hoi/evaluate.py hoi/hand_collision.py configs cluster`无差异；原zero-shot入口、共享物理/网络/训练、全部生产配置与Slurm启动器未改。所有改动限定在 `hoi/v1`、对应CPU测试与notes。

本轮为本地实现与CPU验收，未commit/push、未SSH、未部署、未创建GPU作业或联网W&B run。新版脚接触probe和4096/H32资源验证仍待后续执行；不宣称新版GPU已通过或可以跳过原生验收直接正式训练。
