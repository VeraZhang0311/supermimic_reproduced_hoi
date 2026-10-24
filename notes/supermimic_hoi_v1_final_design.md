# SuperMimic + HOI v1.0 最终设计方案

分支修订：用户已要求保留并推送 `hoi/v1.0` 基线，在独立的 `codex/hoi-v1-no-quality-gate` 分支恢复固定500 iter WM-only后的交替更新。该分支的WM holdout质量比较仅作诊断，不控制actor/critic更新；native和非有限值等硬错误检查保留。具体语义与CPU验证见[无质量门分支记录](hoi_v1_no_quality_gate_20261023.md)。现有两个source13正式实验保留原行为继续运行，新分支尚未部署或启动实验；下文原始准备阶段的「尚未启动」属于历史时点。

日期：2026-10-23。目标分支：`hoi/v1.0`。本文是设计规格；前两轮为设计与数据交接修订，第一阶段数据层证据见 [数据阶段实现记录](hoi_v1_data_stage1_20261023.md)。随后按用户要求实施完整v1，代码、CPU检查、独立集群数据/原生smoke及首次实验准备状态见 [实现与验收记录](hoi_v1_implementation_20261023.md)。正式训练尚未启动，不能把smoke称为正式实验或收敛。

2026-10-23 22:44 UTC更新：完整实现及首次实验前要求的全量缓存、八资产原生门、4096/H32与64步资源诊断、最终CPU preflight已全部通过。实际冻结身份、证据与未执行的正式启动命令见[首次实验就绪记录](hoi_v1_first_experiment_readiness_20261023.md)；正式训练仍未提交。

原始构思保留在 [supermimic_hoi_v1.md](supermimic_hoi_v1.md)。本文以当前仓库源码和两个参考仓库的实际接口为依据，修正原笔记中与实现不一致的地方。新增维度、网络宽度、reward 权重和阈值均为本方案的起始设计，不冒充原论文配置或已验证最优值。

数据说明已于同日依据 [ULTRA 数据与时间基准交接](ULTRA_DATASET_AND_TIMEBASE_HANDOFF_20261023.zh-CN.md) 更新：补充实际集群 v4 数据、冻结 selection、连续时间采样与证据状态，区分历史 consumer 的源速度处理和本版适配器设计。网络扩展方案沿用原设计。该次文档修订没有重新连接集群、复制数据或运行实验；后续第一阶段的实际只读数据核验及 CPU 检查单独记录，不倒写为当时已取得的证据。

## 1. 结论与范围

**v1 做完整机器人 reference 条件下的单刚体 HOI tracking：用同一个可微世界模型联合预测机器人和物体，用真实模拟器数据训练 WM，再通过 imagined rollout 的 BPTT 学习带物体输入的策略。**

采用“现有架构只增不减”：保留 human goal encoder、32D human latent、99D 机器人 observation 及其 10 帧历史、action decoder 的全部隐藏层、29D action、机器人 dynamics 输出、critic、解析 observation/reward/termination 以及原训练闭环。在这些组件上追加 object goal、object observation/history、shape、物体动力学和实际接触预测。

三个核心问题定案如下：

| 问题 | v1 决策 | 原因 |
| --- | --- | --- |
| Object trajectory 是否进入 goal / latent？ | 语义上属于联合 goal；新增独立 object encoder 得到 `zO=32D`，保留 `zH=32D`，在原 decoder 输入后追加 | 保留原 human 表示及旧权重迁移路径，避免把两种目标重新压入原 32D |
| WM state 是否包含 contact？ | 递归物理 state 只新增物体 pose、linear/angular velocity；另加 pair-specific contact auxiliary head，预测下一步真实接触 | 接触影响必须进入联合动力学；接触标签先用于辅助监督与 reward，不反馈成独立递归 flag |
| 连续 16 步未出现 expected contact 是否终止？ | v1 主实验不启用；记录计数及离线 `>=16` 指标，reward 引导接触 | 接触超时有历史依赖，会额外改变 replay、critic 与可微 survival 逻辑，留作单独消融 |

v1 暂不加入 sparse goal、teacher/DAgger、CVAE/prior/posterior、flow matching、视觉重建、灵巧手控制、多物体或可动关节物体。参考 Ultra / Protomotions 的数据契约与几何经验，不迁入它们的策略训练范式、PD 或 action 定义。

主实验使用集群已有的 ULTRA `OMOMO_retarget_aug_v4`，只学习冻结清单指定的训练分区；开发阶段可在其中的小子集验证闭环，但不能把小子集结果当完整数据集结论。历史 `hoi.run` 的冻结 zero-shot 协议继续保留，新策略明确命名为 **HOI-trained, full-reference**。

## 2. 已核实的基线及“只增不减”含义

本仓库设计基点为从 `hoi/zero_shot` 创建分支时的 Git HEAD；具体 SHA 见文末溯源。以下是代码事实，而不是从论文示意图推测的结构。

| 组件 | 当前实现 | v1 处理 |
| --- | --- | --- |
| Robot physical state | `S_R=(p,R,v,w,q,qd)`；运行时 rotation 为矩阵，replay 为 71D | 保持原字段、顺序、含义；另存 object block |
| Robot observation | 99D；10 帧为 990D | 原样保留，独立追加 object history |
| Human goal | 每未来帧 73D，`K=10`，合计 730D | 原定义和 normalizer 保留 |
| Human encoder | `730 → 2048 → 1024 → 512 → 256 → 32`，ELU，输出 tanh | 保留全部层和 32D latent |
| Action decoder | `1022 → 2048 → 1024 → 512 → 256 → 29` | 仅扩宽输入，隐藏层与输出不减 |
| Robot WM | `133 → 2048 → 2048 → 2048 → 70`，SiLU + LayerNorm | 保留原 133D 输入前缀、隐藏层与 70D 输出块，扩展为联合模型 |
| Critic | `914 → 256 → 128 → 1` | 保留原 914D 特征前缀和隐藏层，追加 HOI 特征 |
| Observation / reward / termination heads | `MDP` 中的解析函数；没有三套独立 head NN | 扩展解析函数；只新增需要学习的 contact head |
| 训练闭环 | collect → replay → 8-step WM fit → 32-step policy BPTT → critic λ-return | 保留；同步扩展所有状态及数据路径 |

当前 robot obs 为：`q-default-offset(29), qd(29), body angular velocity(3), projected gravity(3), previous action(29), reference-relative root rotation(6)`。没有 root translation、root linear velocity 或 contact。`OBS_SCALE` 虽在源码定义，但没有被实际应用；尺度处理依赖 RunningNorm。

当前 human goal 取 `f+1 ... f+10`，每帧为 `dp3 + rotation6 + reference local v3 + w3 + q29 + 0.1qd29`。它相对**当前 reference root 的 heading**构造，是 reference-only；v1 不把它改成相对实际机器人，也不把“完整 reference”解释成无限长未来序列。完整指 reference 模态完整可用，在线窗口仍是未来 10 个 control ticks。

“只增不减”不等于所有参数永久冻结，也不等于旧 checkpoint 可以直接 strict load。新增 HOI 输入与联合动力学需要训练；旧组件可一起微调，但其结构、输出、原有奖励与控制语义保留。历史模型是否保持原 motion-only 能力需要独立回归，不能由网络未删减推导出来。

Action 沿用 29D 无量纲 PD offset：`q_target=q_default+xi_joint_offset+alpha*a`，`alpha=0.25*effort_limit/kp`，不改成 torque、velocity 或绝对关节角，不新增 gripper DOF。保留 collect 端 Gaussian std0.1 探索及 `[-20,20]` clamp；原 imagined actor 没有这一 clamp，v1 记录并监控该已有差异，不顺手增加输出 tanh 或更换控制接口。

## 3. 数据、划分与时间契约

### 3.1 实际源数据与冻结 selection

根据 2026-10-23 交接文档中的 SSH 只读核查记录，源数据位于：

```text
登录入口：tianzizh@grasp-login1.seas.upenn.edu
源数据：/mnt/projects/ll/tianzizh/ProtoMotions/data/omomo/OMOMO_retarget_aug_v4
```

该次核查观察到 **4066 个 `.pt`，总计 3,531,348,002 bytes（约 3.53 GB / 3.29 GiB）**；目录为实际目录，当时不是符号链接，未发现非 `.pt` 条目。冻结训练/验证清单引用的文件均存在。上述是交接提供的既有观测，不是本轮重新检查的集群状态。

实际可复用的冻结训练 selection：

```text
/mnt/projects/ll/tianzizh/ProtoMotions/runs/dagger_object5800_G_recovery_from18600_to20200_20261021_01/inputs/selection.json
SHA256: 8c141eb60650b9df79b92acfeeea785f796c7f18f52bf179467562c7cc315b23
```

同 80 条验证的集群清单：

```text
/mnt/projects/ll/tianzizh/ProtoMotions/runs/dagger_object5800_G_recovery_from18600_to20200_20261021_01/evaluations/requested_19500/inputs/evaluation_cases.json
```

这些 `cases`/entries 绑定 motion 路径、motion SHA256、source_group 和 object/asset。目录中的全部 4066 个文件不是训练集；训练范围仍是 selection 指定的 2651 motions / 381 groups。后续第一阶段已读取实际 selection 和 eval80 并逐项核对；eval80 的实读 SHA 与本地 canonical 相同，见 [实现记录](hoi_v1_data_stage1_20261023.md)。

新项目建立自己的独立 manifest，保存 source selection 路径与 SHA、逐 motion SHA、case/source-group、asset/scale/transform 绑定。先与第 3.2 节的本地四分区 manifest 和 canonical 名单逐项核对；不得仅因条数相同就认定两份清单等价。集群 eval80 清单的 SHA 未在交接中提供，不能直接套用本地 `459a...` hash，实施时需读取核对。保留既有 test40 名单及其所有同源 variants 隔离。

交接中重算的样例 SHA 为：

```text
文件：sub10_largebox_000_075_086_074_100_100_100.pt
大小：671819 bytes
SHA256: 1a0bac419adab2bf6d7f5d10d480e40d0f3c39a74605a7063c7f48b0f99bf97c
```

该核查确认了清单路径存在，并抽查上述文件字节；**没有重新扫描全部 4066 个文件的 SHA**，也没有逐文件重新解码全部 tensor。新项目实际加载所选 motion 时核对其已登记的 SHA/schema。源目录只读使用；新 manifest、缓存、派生 reference 和输出写入独立项目目录，保留源 `.pt` 字节。`[T,630]` 源 reference 与 50 Hz `rollout.pt`、checkpoint 必须按 schema 区分，不凭 `.pt` 后缀混用。

### 3.2 复用现有分组划分，禁止增强泄漏

复用 Protomotions 中已固定的 manifest 及 canonical 名单，按原始 motion filename 的前三段建立 `source_group`，同一原始动作的所有 scale / augmentation variants 必须在同一分区。

| Manifest 分区 | 原始动作组数 | 增强轨迹数 | v1 用途 |
| --- | --- | --- | --- |
| train | 381 | 2651 | 唯一训练来源；真实采样、WM、normalizer、训练 bin sampler 均只接触这里 |
| validation | 80 | 517 | 用 canonical eval80 做模型选择与对照；全部 517 variants 禁入训练 |
| regression | 100 | 655 | 保留为额外诊断集，不并回训练；默认主评估不强制全跑 |
| test | 40 | 243 | 模型及规则锁定后，仅用 canonical test40 作最终测试；全部 243 variants 禁入训练 |

静态解析已确认四分区的 source-group 两两交集为零；eval80 为 validation 中的 80 条 / 80 groups，test40 为 test 中的 40 条 / 40 groups。**eval80 不是全部 517 条 validation variants，test40 也不是全部 243 条 test variants。**

四分区 motion 数相加为 4066，与交接的目录计数相符；数量相符本身不证明文件集合或 hash 相同。实施时仍需核对冻结 selection 和目录对象身份。eval80 已多次用于历史开发/选模，本版继续将它作为 validation，不能改称独立 test。

这些名单原先限定的是 adaptation held-out；旧 teacher pretraining 可能重叠。新实验若从仅 LAFAN1 训练的 SuperMimic checkpoint 初始化，仍要记录该 checkpoint 的训练来源，才能说明新策略的 HOI holdout 边界。不能把复用名单称为“所有历史模型从未见过的独立测试”。测试结果不用于调权重、选 checkpoint 或改变阈值。

### 3.3 `[T,630]` 字段与 reference velocity 契约

| Python slice | 内容 | 适配规则 |
| --- | --- | --- |
| `0:3 / 3:7 / 7:10 / 10:13` | Robot root world position / quat / v / omega | 外部 quat 为 xyzw；按机器人 link-origin 契约核对速度 |
| `13:42 / 42:71` | 29 joints q / qd | 按 joint names 重排，不能假定 asset 顺序相同 |
| `71:74 / 74:78 / 78:81 / 81:84` | Object world position / quat / v / omega | 必须明确 object frame 与速度参考点 |
| `84:201 / 201:357` | 39 body world positions / quats | 用于数据验证和保留的 PHC 指标；不替换原 14-body robot reward |
| `357:474 / 474:591` | 39 body v / omega | 校验参考点与有限差分，不直接当原 SuperMimic FK 真值 |
| `591:630` | 39 body contact flags | 只作为源标签与诊断，不直接当手—物体 pair 真值 |

**历史数据处理事实：**ProtoMotions 保留原 `.pt` 字节，按连续时间插值 pose 及源 velocity 字段，contact 最近邻采样；并没有根据 50 Hz 重算整份发布数据的速度。源 velocity 的生成时钟及 link/COM 参考点仍需核验；仅把控制改成 50 Hz 不会自动纠正这些语义。

**本版适配器的明确设计：**机器人 reference 继续由 root pose + name-mapped joints 进入原 `MotionLib`，按原方法计算 50 Hz 速度及 FK，保留既有 SuperMimic human goal/reward 语义。新增 object reference 用同一时间戳重采样 pose，translation 线性插值、rotation SLERP；其 v/omega 由这份 50 Hz pose 差分生成。两者在新 manifest 分别记录为 `robot_reference_velocity_mode=pose_diff_on_control_grid`、`object_reference_velocity_mode=pose_diff_on_control_grid`。这是已定的 **SuperMimic 新适配策略**，不描述为 ProtoMotions 的既有速度处理，也不改写源数据或历史指标。

适配时同时保留源字段的采样值作审计，按相同 world frame / 参考点比较 source-field 与 pose-derived v/omega 的误差和接触阶段分布。约定 `source_velocity_sampling=linear_interpolation_preserve_source_fields`，表示源字段的保留与查询方式，不表示 actor/reward 最终使用它们。不能对未核验的源速度擅自乘 `60/50` 或 `60/(1000/17)`；差分也不构成真实生成 FPS 的证明。选择其他 velocity mode 必须新建版本、记录 reset/goal/reward 的影响，并重做相应对照。

差分规则沿用原 `MotionLib`：`dt=0.02s`，位置和 joint angle 内部点用中心差分，首尾用单边差分；world angular velocity 内部点为 `log(R[j+1]*R[j-1]^T)/(2dt)`，首尾对应单边旋转差分。新 object 使用同式。保存 `source_v/source_omega` 与 `derived_ref_v/derived_ref_omega` 的不同字段名、差分方法、参考点及派生 cache hash；不足两个 control-grid 状态的 clip 不能构造速度，需在数据 gate 中报告，不能静默伪造或丢弃。

连续时间重采样、reference velocity 派生、source/API 的 COM→link-origin 转换是三项独立操作。只有确认源 velocity 参考点后才能做可解释的逐项差异审计；未知项标为 unknown。复核 ProtoMotions 历史结果时必须沿用它原来的 velocity mode，不用本版新派生速度改写其结果。

### 3.4 Canonical 60 Hz 解释 + 50 Hz 控制

明确区分三种时钟：

| 量 | ProtoMotions 已有证据 | SuperMimic v1 约定 |
| --- | --- | --- |
| Source timebase | consumer 按 60 fps 解释；生成间隔未证实 | 沿用 `canonical_60hz`，标为协议假定 |
| Control dt | 已有实现/运行验证为 0.02s，即 50 Hz | 保留原 `CONTROL_DT=0.02`，不因交接改变 |
| Physics | 0.005s × decimation4 | 保留本仓库 `SimulationCfg(dt=0.02)` + Newton `num_substeps=4`；其 native 执行与有效积分步长另行核验，不能冒用 ProtoMotions 的运行证据 |

对 N 个源状态帧，canonical duration 为 `(N-1)/60` 秒。从 clip 时间零开始，第 j 个控制 tick 查询 `t_j=j*0.02`，机器人与物体共用同一 `t_j`：

```text
source_index = clamp(t_j * 60, 0, N - 1)
i0 = floor(source_index)
i1 = min(i0 + 1, N - 1)
alpha = source_index - i0

continuous fields: linear interpolation
root / object / body quaternions: SLERP
source contact flags: nearest source frame
```

这里连续字段包括需要保留的源 velocity 审计值；actor/reward 使用的 pose-derived reference velocity 按第 3.3 节另行生成。新增几何 expected-contact schedule 在同一 control tick 上生成，其 valid mask 仍遵循第 6.1 节，不与源 39 flags 混为一种标签。

控制时间 `0.00,0.02,0.04,0.06...` 对应源索引 `0.0,1.2,2.4,3.6...`，不是每个 action 简单前进一个源帧。Human/object future 的 10 个 control ticks 对应 `0.02...0.20s`，不是 10 个 60 Hz source frames；其他以秒定义的前瞻偏移也不能通过全局将 `60` 替换为 `50` 来改写。

完整评测的控制步数与状态帧数：

```text
T = floor(((N - 1) / 60) / 0.02) = floor(5 * (N - 1) / 6)
saved state/reference frames = T + 1
action/observation/goal/reward frames = T
```

此固定 60→50 协议可用源整数帧数计算步数，避免 CPU/GPU 浮点舍入在尾端产生一帧差异。最后状态时刻为 `T*0.02`，距 canonical duration 的余量小于一个 control tick，明确保存这一余量；不补一个不等长物理步。N=601 时 duration 为10秒，T=500，保存501个状态。Derived cache 按版本及 hash 独立保存，不批量覆盖源动作为“50 fps 数据”。

**时间证据边界：**本仓库历史 HOI80 和 ProtoMotions 均按 60 Hz 协议解释；Ultra 当前 consumer 标记 `fps_data=60`，而当前 exporter 每控制步捕获一帧，teacher 配置 `17×1ms≈58.823529Hz`。本机 `D:/Ultra` 没有发布动作目录，不等于集群没有实际 v4 数据。当前 exporter 行为不能反向证明发布文件的生成时钟；路径存在、mtime、文件数或 SHA 也不能证明逐帧时间。

| 结论 | 当前证据状态 |
| --- | --- |
| 集群 v4 数据存在、冻结训练/验证路径存在、样例 hash 相符 | 交接中的 2026-10-23 实际只读核查，本轮未重查 |
| ProtoMotions 按 60 Hz reference 解释，并以 50 Hz 控制运行 | 交接的源码及已有运行证据 |
| Robot/object reference 以连续时间同轴采样 | 交接源码及本仓库历史适配约定 |
| 发布数据真实生成间隔为 1/60s | 未由生成来源或原始时间戳证实 |
| 发布数据真实生成间隔为 17ms | 同样未由生成来源或原始时间戳证实 |

结果统一表述为 **“canonical 60 Hz 解释下、50 Hz 控制的实验”**。“保持完整动作时长”指保留这一协议时长及其完整 50 Hz 控制 tick，不声称已确认原始生成的物理时长。若601帧实际按17ms生成，总时长应为10.2秒，和canonical10秒相差2%；换成50Hz控制不会消除这一来源差异。

新 manifest 至少登记以下时间字段；这是新 schema，不声称旧 ProtoMotions 清单已包含它们：

```json
{
  "dataset_name": "OMOMO_retarget_aug_v4",
  "timebase_mode": "canonical_60hz",
  "source_fps_assumed": 60.0,
  "source_timebase_status": "protocol_assumed_not_generation_verified",
  "source_timebase_evidence": "ULTRA_DATASET_AND_TIMEBASE_HANDOFF_20261023.zh-CN.md",
  "reference_sampling": "continuous_time_linear_and_quaternion_slerp",
  "source_contact_sampling": "nearest_source_frame",
  "source_velocity_sampling": "linear_interpolation_preserve_source_fields",
  "robot_reference_velocity_mode": "pose_diff_on_control_grid",
  "object_reference_velocity_mode": "pose_diff_on_control_grid",
  "reference_velocity_reference_point": "link_origin",
  "source_velocity_reference_point_status": "unverified",
  "velocity_difference_method": "central_one_sided_boundaries_so3_world",
  "control_dt_s": 0.02,
  "simulation_dt_s": 0.02,
  "newton_num_substeps": 4
}
```

正式新训练前继续核对生成脚本版本/实际运行配置、发布说明或逐帧时间戳，并在 manifest 记录查证结果与尚存假定。如果取得支持17ms的证据，另建带版本的时间配置，机器人/物体同步处理，并核对 velocity、contact、duration、future offsets、完整帧数及指标；需要的对照在新版本重跑，不能只改训练侧或静默改写历史 HOI80 报告。

### 3.5 统一坐标、速度与几何

- 世界坐标为 z-up。每个环境先减 `env_origin`；robot、object、reference、reward 和渲染都使用同一个环境坐标。数据记录外部 quat 采用 xyzw，网络内部沿用 SuperMimic 的矩阵 / wxyz 工具，并显式转换。
- Robot `v` 沿用 pelvis link origin 的 world linear velocity。Object 同样约定为 object body/link origin 的 world linear velocity，`omega` 为 world angular velocity；不能拼接 link pose 与 COM velocity 后称为同一参考点的 state。
- 若 native API 返回 COM velocity，则按 `v_link = v_com - omega × (R_object * com_local)` 转换。源 Gym tensor 的速度参考点不能仅靠静态 slice 确认，须做非零 COM、纯旋转及 pose 差分 probe。
- 保存 `T_object_from_mesh`、mesh scale、物理 COM、惯量及 inertia frame。Point cloud 采用 **object-local、真实米制尺寸**；不做未记录的单位化或单独居中。若使用中心化点云，必须把中心偏移纳入 transform，不能当 object origin 或 COM。
- 机器人 logical hand marker、wrist link、真实手部 collider 是三个接口。Protomotions 的 logical marker 有 wrist-x=0.12m 的偏移；本仓库橡胶手 mesh 安装约为 x=0.0415m、y=±0.003m。通过名称和固定变换映射，不能把 body index 28/38 或 wrist 原点直接当掌面。

## 4. Actor：保留 human 分支，追加 object 分支

定义当前实际/imagined robot heading 为 `H_t`，物体状态为 `O_t=(pO,RO,vO,wO)`，reference 用上标 `ref`。所有需要实际坐标变换的新特征，在每个 imagined step 使用**当步预测**的 robot/object 重算。

### 4.1 Object current observation：每帧 18D，10 帧历史

新增 actor observation `oO_t`：

1. 当前 object pose：`H_t^T(pO_t-pR_t)` 3D，`rot6d(H_t^T RO_t)` 6D。
2. 当前 object target error：`H_t^T(pO_ref(f)-pO_t)` 3D，`rot6d(RO_t^T RO_ref(f))` 6D。

共 18D，10 帧为 180D。当前 pose 让策略知道物体相对机器人在哪里，当前 error 让它知道物体是否偏离世界 reference，避免只有 reference-only human goal 时隐藏全局漂移。

v1 actor 不额外读取 privileged object velocity、真实 contact force、mass 或 inertia。Object pose 和构造新特征所需的 robot global pose/heading 均由模拟器真值提供，因此本版是 **oracle-pose full-reference tracking**；历史可以表达运动，实机估计器不属于本版验证范围。新 object observation 默认不额外加噪，单独记录这一观测假设；保留原 robot obs 的噪声配置。后续 pose noise / delay 是独立 robustness 实验。

### 4.2 Object future goal：190D → 32D

仍取与 human goal 相同的 `f+1 ... f+10`。每个未来 tick 包含：

- `H_t^T(pO_ref(f+k)-pR_t)`：3D。
- `rot6d(H_t^T RO_ref(f+k))`：6D。
- `H_t^T vO_ref(f+k)` 与 `H_t^T wO_ref(f+k)`：6D。
- 左/右 expected contact：2D；对应 valid masks：2D，未知标签以零值和 mask=0 表示。

每帧 19D，总计 190D。Object encoder 建议 `190 → 256 → 128 → 32`，ELU + tanh。`z_goal=[zH32,zO32]` 在语义上是 64D 联合目标，但实现时保持原 decoder 的输入前缀，不能为了相邻拼接两 latent 打乱旧输入列。

原 human goal 仍是 reference-only；新 object goal 是 state-conditioned reference encoding。reference 原始轨迹固定，依赖当前状态的只是坐标变换。每个 tick 都重算新 goal，包括 imagined rollout，不能缓存起点坐标变换用满 32 步。越过 clip end 的新增 object future pose 复用末帧，与 human future clamp 一致，但相应 contact validity 置零，不假造尾端接触命令。

### 4.3 Shape 与 decoder

点云为每个 asset/scale 离线确定性 FPS 的 `128×3`。先在真实 surface 上生成足量候选点，再做 FPS；保存采样 seed、候选来源、输出点云 hash 及 transform。Ultra 当前使用 `sample_surface_even(256)` 后减 mesh vertex mean，并不是 FPS；本方案的 FPS128 是新设计。

Shape encoder 建议 PointNet：逐点 `3 → 64 → 128 → 128`，max pool，再 `128 → 64`。Actor、WM、critic 各自拥有 shape encoder，禁止训练参数共享；点云原始数组可共用缓存。形状固定不代表 embedding 固定，encoder 更新后不能继续使用上个 update 的已编码 feature。

Decoder 输入固定布局：

```text
[ zH(32), flatten(robot_obs_history)(990),
  flatten(object_obs_history)(180), zO(32), actor_shape(64) ]
                       ↓ 1298D
             2048 → 1024 → 512 → 256 → 29
```

保留旧的前 1022 列；新加 276 列。**不能把每帧 robot/object obs 先拼接再整体 flatten**，那会改变原 robot history 列的位置。旧权重迁移时复制旧 decoder 首层对应块，新列为零，其他 decoder 层完整复制；这样初始化时 action 对新增输入不敏感，之后可训练学会利用它们。

Robot observation、human goal 使用原 normalizer；object observation、object goal 使用新的独立 normalizer，从真实训练采样更新。Shape embedding 不套一个随旧 checkpoint count 冻结的 RunningNorm。评估时全部统计冻结。

## 5. 联合 WM：双向耦合 robot–object dynamics

### 5.1 State、输入与输出

递归物理 state：`X_t=(S_R,t, O_t)`。Robot replay block 保持 71D；object block 为 `p3+quat4+v3+w3=13D`，联合存储 84D。Shape、物理参数是 episode context，不当作预测变量；reference、expected contact 和 contact timeout counter 不输入物理 dynamics。

保留原 WM 输入前缀 `invariant_robot71 + action29 + normalized_robot_xi33 = 133D`，追加：

| 新块 | 维度 | 内容 |
| --- | --- | --- |
| Object invariant feature | 15 | relative p3、heading-relative rotation6、heading-frame `(vO-vR)`3、world omega 转 heading 3 |
| WM shape | 64 | WM 自己的 PointNet embedding |
| Object physical context `xiO` | 12 | mass1、local COM3、COM inertia 的对称独立项6、object sliding friction1、ground sliding friction1 |

主 dynamics 为 **`224 → 2048 → 2048 → 2048 → 82`**。原 LayerNorm / SiLU 和原 70D robot increment 保留，追加 object increment 12D：

```text
robot: dp3, drot3, dv3, dw3, dq29, dqd29  = 70（原定义不改）
object: dp3, drot3, dv3, dw3               = 12
```

Object `dp,dv,dw` 在当前 robot heading 下预测，再变换回 world 积分；`drot=log(RO_t^T RO_t+1)` 为 object-local 的 3D SO(3) 增量，`RO_next=RO*exp(drot)`。内部 rotation 用矩阵，不直接回归未经约束的 quaternion，也不把 6D rotation feature 误当 6D angular increment。Inertia6 表示 object-local 坐标中关于实际 COM 的对称惯量，COM3 相对同一 object pose 原点，不能混入 principal-axis frame 的对角数而遗漏旋转。

Robot height 已在原 invariant 中，配合 relative object height 可确定 object 离地高度。`xiO` 由 native runtime **实际生效值**读回，不能从 shape 猜质量，也不能从 Ultra 的 density/randomization 配方冒充实测物性。Robot–object 的摩擦组合依当前引擎规则以及保留的 robot friction 一起确定。接触 margin、solver、damping、滚动/扭转等其余 native 参数固定并写入 physics manifest；若未来随机化，必须同步扩展 context/schema，不能让隐藏变化留在固定的 12D 之外。

扩展后的 robot 输出也必须能依赖 object state/shape/context，才能学习反作用力。不能只在原 robot WM 旁接一个预测物体的网络，却让 robot 始终忽略物体；也不能把参考物体运动直接写入 `O_next`。

### 5.2 Contact auxiliary head

从联合 WM 的最终 hidden feature 新接 `2048 → 128 → 3` 的 contact logits，预测 `(X_t,a_t,shape,xi) → b_actual(t+1)`：左手—物体、右手—物体、物体—地面。它与 dynamics 共享 WM backbone；ground channel 用于诊断和物理模式检验，v1 不要求 ground expected-contact 目标。这里 ground 严格指场景地面，不能把桌面或其他 support 接触合并命名为 ground；若数据要求额外支撑物，需显式扩展场景与 pair schema。

实际标签按**碰撞实体 pair**收集，对一个 control interval 内的接触 normal impulse 求和，再除以 `dt` 得到平均法向力；每个 pair 默认 `>1N` 记为接触。`b_actual(t+1)` 表示 `(t,t+1]` 这个已执行控制区间的汇总接触，和末端 `X_(t+1)` 一起记录；不是只读取最后一个 physics substep。保存原 impulse/force 数值、单位、聚合方式、有效 mask 和 threshold，同时输出轻触检出率。此 1N 是起始定义，需用已知手—物体、手—地面、物体—地面 probe 校准；标签定义改变必须版本化并在训练前锁定。

Native tensor 是否能提供带 entity IDs 的 GPU pair impulse 是实现前必须验证的能力。若不可用，应实现受支持的 pair filter/contact instrumentation；不能用手腕 net force 替代后宣称配对监督成立。Contact head 不输入 reference contact，也不以 expected contact 作为 BCE 真值。

### 5.3 WM loss、冻结与梯度

保留原 8-step robot normalized state-error loss 及原 70D 归一化定义，独立新增 object 的 position、SO(3)、linear velocity、angular velocity 归一化误差，以及实际 contact BCE：

`L_WM = L_robot_original + lambda_O * L_object + lambda_C * L_contact`。

起始建议 `lambda_O=1, lambda_C=0.1`；object 四组误差先各自按维度取均值，再等权平均，防止单位和维度主导。Object output normalizer 与 component error floors 独立维护，初值参考原同物理量 floors（position 0.003m、rotation 0.005rad、v 0.02m/s、omega 0.05rad/s）。Contact BCE 按真实训练数据估计类别比例，报告逐类 precision/recall/calibration，不能只报由 no-contact 样本主导的 accuracy。

训练使用真实 actions 的多步 autoregressive rollout，同时监督各步 object state 和 contact；mask 保留 terminal transition，屏蔽跨 reset 的后续步。检查 1/8 步 holdout WM error，并用独立真实 action sequences 检查 32 步漂移；只报训练 MSE 不足以开放 actor 更新。

Actor update 时冻结**整个 WM**，包括 WM PointNet、contact head 和 normalizer 更新，但保留其输入 Jacobian。`requires_grad_(False)` 用于 WM 参数，不能用 `torch.no_grad()` 包裹 imagined transition，不能 detach 预测 object state 或 contact probability。Critic 参数同样冻结但保留输入梯度。

## 6. Contact 目标、reward 与 termination

### 6.1 Expected contact 与 actual contact 分开定义

Ultra 当前 exporter 的 39 flags 来源于 per-body **net contact force**；奖励 consumer 的阈值形式与 exporter 还不完全相同。它可能包括地面/自碰撞，不能直接监督“手接触了物体”。

v1 用机器人 reference FK、经过核对的手部 collider 安装变换、同一物体 collider geometry，离线生成**手—物体 reference contact schedule**：最近表面间隙 `<=2cm` 标为 expected=1，`>=4cm` 标为 expected=0，中间区间 mask=0。严重穿透或不可信几何帧也标为未知并进入数据审计。该 schedule 表示几何意图，不等同真实力学接触；源 39 flags 完整保留，另外报告两者差异，不静默改源数据。阈值是本方案的新定义，须在训练分区的固定样本上锁定。

128 点 shape 只用于 shape encoder，不用于精确接触判定或 penetration 检查。Reference 标注使用实际 collider 的表面/距离查询；imagined reward 的距离项使用缓存的可微 collision-surface distance representation（例如经验证的 SDF grid），明确 grid 误差与手掌采样点。不能把 Ultra 名为 `compute_sdf` 的最近点三维向量直接称为 signed distance。

真实与 imagined 的 hand-distance reward 均使用**同一份缓存距离表示及同一公式**；精确 collider surface query 用于离线 schedule 和距离表示的 QA。接触 margin/rest offset 与 distance 的零面不一定重合，两种阈值的对应需在 native probe 中记录，几何靠近不直接等于力学接触。

### 6.2 Reward 保留原项，追加四类 HOI 信号

每个真实 transition 的 reward 都在 `X_(t+1), reference(f+1), a_t, a_(t-1)` 上计算；imagined rollout 使用相同时间索引和相同公式。原 human reward 含 4 个 body tracking 项、2 个 root tracking 项、action-rate 与 joint-limit 惩罚，原样保留，不重复加一套 regularization。

新增项均为 `[0,1]` 的正向 shaping，最后整体乘 `dt=0.02`：

| 项 | 公式/含义 | 起始权重 | 起始尺度 |
| --- | --- | --- | --- |
| Object position | `exp(-||pO-pO_ref||²/sigma_p²)`，共同世界坐标 | 1.0 | 0.10m |
| Object orientation | `exp(-angle(RO^T RO_ref)²/sigma_R²)` | 0.5 | 0.30rad |
| Object linear velocity | 与统一时间基准下 reference v 比较 | 0.25 | 0.50m/s |
| Object angular velocity | 与 reference omega 比较 | 0.25 | 1.0rad/s |
| Hand approach / interaction geometry | 对 expected-positive 的手，奖励 palm samples 到物体表面的连续距离；无正标签时为 0 | 0.25 | 0.05m |
| Contact match | 对有效左右 pair 平均 `b_ref*p_actual+(1-b_ref)*(1-p_actual)`；全未知为 0 | 0.25 | 无 |

Hand approach 的有界定义如下：对每只手选定并保存真实掌面 probe points，经当前 robot FK 和安装变换得到 world 点，再用 `RO^T(x_palm-pO)` 查询 object-local SDF。令各点距离为 `phi_k`，`w_k=softmax(-phi_k²/tau²)`（起始 `tau=0.02m`），`d_hand²=sum(w_k*phi_k²)`，单手 reward 为 `exp(-d_hand²/0.05²)`。仅在 reference 有效且 expected-positive 的手上平均，无激活手则为零，故整个新项处于 `[0,1]`。查询坐标保留对 robot/object state 的梯度；对严重 penetration 另行记录，不能把该接近项当不穿透证明。

真实 reward 中 `p_actual` 使用真实二值 pair label；imagined reward 使用 contact head 的 sigmoid probability，保持梯度。Contact 负例 reward 不以任意多数 no-contact 帧填满训练解释：报告 expected-positive 子集、missing 与 extra contact，并检查 positive phase 是否改善。物体运动与接触概率必须联合在真实 simulator 中验证，禁止仅靠 imagined contact reward 上升判定成功。

接触建立/释放的统计沿用 `(t,t+1]` 区间 actual label 对 `reference(f+1)` 的约定；它与瞬时 endpoint contact 不完全相同，报告 onset/release timing 时注明最多一个控制区间的聚合误差。

原 body position/orientation reward 仍按原方法 re-anchor reference 到实际 root xy/yaw；新增 object tracking 绝不单独 re-anchor。手—物体几何项在实际 robot/object 同一世界计算，使机器人“动作像参考但没有真正碰到物体”不能靠单独对齐获得满分。

保留原 `human_normalized_return` 分母 `0.1/step`。HOI 总奖励最大值为 `(5+2.5)*0.02=0.15/step`，新增 `hoi_normalized_return`，缺失标签时注明可达上限差异。训练 bin sampler 的 `G_max` 使用同一 HOI 上界，原采样混合系数不改；不能把总奖励继续除以旧 0.1 并沿用原指标名。

以上新项权重和尺度在 train-only 校准与 eval80 开发阶段确定，test40 开封前锁定。v1 不复制 Ultra 的乘法 reward `rb*ro*rig*rcg`，因为那会替换原 SuperMimic 的目标结构。

### 6.3 Termination 定案

原 robot hard / soft termination 保留：root height reference error `>0.25m`，以及 `abs(Rzz_ref-Rzz)>0.8`；原 soft sigmoid 温度 `.02/.04` 保留。原 timeout / clip end 继续与 failure 区分。

v1 额外增加宽松的 object positional failure：`||pO-pO_ref|| > max(1.0m, 2*object_bbox_diagonal)`。真实环境用 hard threshold，imagined environment 用同一误差的 sigmoid soft failure（起始温度 0.10m），通过 `alive_joint=alive_robot*alive_object` 合成，避免直接把两个概率相加。非有限状态属于运行异常，记录并使检查失败。

不把 object 接地、rotation error 或 contact mismatch 直接作为主实验硬失败。搬运掉落与有意放置先通过 phase/contact/height/object-tracking 指标区分；接地本身不是失败。

每个有效 expected-positive pair 的连续缺失计数为：

```text
missing_j(t+1) = valid_j * expected_j * (1 - actual_j)
n_j(t+1) = missing_j ? n_j(t)+1 : 0
```

在 50 Hz 下 `n>=16` 对应 0.32s，仅记录为 `contact_timeout16` 诊断，不改变训练 done 或 survival。未知帧重置计数并记录 coverage，避免穿过未知区间声称连续失配。Ultra 源码实际是 `counter>20`、有前段与尾段屏蔽，不能等同本规则。

后续若做 contact-timeout 消融，需单独写入任务 state：replay 保存真实 counter 并在 imagined starts 恢复，critic 追加 counter，说明 hard rule 与 differentiable surrogate 的对应、偏差及历史依赖。不能直接在 head 加一个 logit 后宣称已经实现严格的 16-step 规则。

## 7. Critic、replay 与训练场景

### 7.1 Critic 追加 HOI 条件

原 914D 特征（包括原 730D human goal）全部保留，依次追加：object future goal190、current object invariant15、current object pose/velocity tracking errors15、物理 context12、当前 expected contact 与 valid masks4、critic shape64。总计 **1214D → 256 → 128 → 1**。

新增 error15 为 position3、rotation6、linear velocity3、angular velocity3。Critic 可使用 privileged true velocity 与物理参数；actor 仍只读 pose/history。Critic 不读取 future actual contact 或 future simulated state。v1 不把仅用于日志的 timeout counter 输入价值函数；它不影响 v1 任务的 reward/done。

Critic 自己的 shape encoder 由 value loss 更新；actor update 冻结它及 critic 参数。旧 914D normalizer 独立保留，新增 raw 特征另行归一化，shape feature 不与旧 count 混用。Critic targets 沿用原 imagined λ-return；用于 critic fitting 的轨迹副本 detach，不影响 actor 那条计算图。

**实现时必须调整原 critic 特征缓存边界：**缓存 detached raw state/features、asset/pointcloud ID，critic 的每个 minibatch forward 内重新执行自己的 PointNet，再拼接 shape64。不能在原 `no_grad` 特征构造区预计算并缓存 shape embedding，否则新 encoder 永远收不到 value-loss 梯度。Actor/WM 的 shape embedding 也不得跨参数更新缓存为 detached 常量。

### 7.2 Replay 必须同步扩展

保留现有 `s,s1,a,a_prev,xi,r,f,ep_step,done` 字段，追加：`o,o1`、asset/scale ID、`xiO`、pair impulse/actual labels/validity、human/HOI reward 分解、failure reason、contact missing counter（日志用途）与 source-group/manifest ID 的可追溯映射。避免在每条 transition 重复保存静态点云。

`sample_windows` 和 `sample_starts` 不能只扩展 writer：WM window 要返回 object / context / contact，actor start 及 10 帧 history 要恢复同 episode 的 object pose 和 reference frame。新 episode history 复制该 episode 首帧，禁止跨 reset、跨 clip 或跨 asset 串接。Terminal transition 的 `s1/o1` 在 reset 前记录，不能误存 reset 后状态。

联合 reset 在同一 reference frame 初始化 robot 和 object pose/velocity，使用正确 env origin，执行 native forward/refresh，再核对 FK、碰撞与原始状态误差。v1 保留原 robot reset noise 配置，记录其造成的初始手—物体间隙/穿透，不额外引入未声明的物体 pose/mass 随机化；严重初始穿透应在原生检查中暴露，不能通过禁碰撞、短暂固定物体或 teleport 修补。

### 7.3 多 asset / scale 的场景组织

每个训练环境始终只有一个可交互 rigid object；不同 asset/scale 使用独立的静态 object geometry 分组，由同一进程的训练管理器统一采样、汇入 replay，并共享同一 actor/WM/critic。总环境预算维持原默认 **4096**，是所有分组之和，不是每组 4096。

环境分组由 `(asset,exact_scale,collision_hash,physical_profile)` 确定，reset 只从兼容的 motion bins 抽样，不能在复用 geometry 的环境里换一个不匹配的 object ID。各组配额依据训练 bin 数分配并记录实际采样分布，组内沿用原 return-based bin sampler 的混合系数；这是异构 asset 必要的采样扩展，不能声称与原单 asset 的全局采样逐步等价。

默认组织为 central learner + 固定 asset 的 rollout workers：统一 batch inference 后拆分 action，各 worker step 后汇总 transition。保守实现每个 worker 使用独立仿真进程；只有经 native probe 确认 backend 支持时，才合并成一个 `SimulationContext` 下的分组 scenes。不能在单进程循环实例化现有 `TrackingSim` 产生多个互相覆盖的 context，也不能假定 `replicate_physics=True` 支持任意不同 USD。跨进程通信、总环境预算和吞吐需要原生实测。

原 `hoi/evaluate.py` 的逐 asset 独立评估及 scene monkeypatch 可作为物性读回和评估边界参考，不能直接拿来当异构训练 scene 管理。先验证一个 asset 的物理与梯度闭环，再验证多 asset 的采样/隔离/显存；若 4096 预算放不下，不静默减环境、batch 或 horizon，另行记录并定义资源缩减实验。

## 8. 训练阶段与 checkpoint 迁移

首次正式实验执行配置按用户最新要求固定为：最大 **20000 iterations**，每 **500 iterations** 保存完整 checkpoint，分别在 **5000 / 10000 / 15000 / 20000** 使用刚保存的权重自动提交完整 canonical HOI80 冻结评估。评估使用独立 Slurm GPU 作业与新目录，保存提交身份以防重复；排队和提交不代表完成。其余生产优化配置保留，LR decay schedule 仍为250000 iterations。配置位于 `configs/hoi_v1_first.json`，实际验收状态另见实现记录。

1. **数据与 native gate。** 依据交接的实际 v4 路径核对冻结 selection SHA、逐 motion SHA、四分区及 canonical 名单；记录 source timebase 的证据状态，验证连续时间 60→50 采样、integer frame-count 和源/派生 velocity 模式。再核对 joint/body name mapping、object transforms、COM velocity、physics readback、pair contact、实际碰撞过滤，以及 reference 与当前手部 geometry 的误差。交接中的存在性/样例核查不替代新项目加载时的身份校验。
2. **旧权重初始化。** 保留 human encoder/decoder/robot WM 的可映射权重及统计，新增分支初始化，创建全新的 HOI run、optimizer 和 replay。Physics 必须匹配并显式记录。
3. **真实 replay prefill + WM warmup。** 保留已有 64 vector steps prefill、500 iterations WM-only 起始配置；从训练分区不同 reference phases（no-contact/approach/contact/transport/release）初始化 robot/object，采集真实 dynamics 与实际接触。这要求阶段/接触模式覆盖，不要求尚未训练的策略先成功完成搬运。若真实正接触样本不足，应标明 coverage gate 未过，检查 reference 接触 reset、几何与探索采样；延长 warmup 不能代替覆盖证据。所有动作后状态仍由物理模拟产生，不把 reference transition 当真实动力学样本。
4. **交替联合训练。** 原 collect/fit/BPTT/critic 顺序不变。原起始参数保留：WM window8，WM batches10×batch2048，BPTT horizon32，policy batch4096，UTD0.4，Adam lr2e-4，gamma.99，lambda.95，原 exploration/gradient clipping/critic epochs 设置。
5. **eval80 选择与锁定。** 冻结权重与统计，用 canonical eval80 做真实 simulator 全程评估；选择规则使用真实 object/contact/human 指标，不能只选 imagined return 最大者。
6. **test40。** 锁定数据、reward、threshold、checkpoint、seed 计划后运行，完整记录全部动作与失败。新训练结果不能改写旧 zero-shot 报告。

新 run 从物理配置一致的 SuperMimic checkpoint 初始化优先；旧 hand/self 配置缺失时必须显式指定两项并核对。如果来源配置不同，不能称 resume；作为清楚标注 physics change 的新迁移实验。本版默认 hand ON、body self-collision ON，保留相邻/焊接 link exclusions、真实物体重力/惯量和已有 CoACD recipe。

Checkpoint schema 至少记录：feature version/字段顺序/维度，shape sampling与encoder定义，dataset/source selection 身份及逐 motion manifest hash，data/split/timebase/contact-label hashes、source timebase 的证据状态、实际 control/physics 配置、reference velocity mode/method 与派生 cache hash，asset与physics，三个网络及各自 shape encoder，所有 normalizers，optimizer，iteration 和初始化 checkpoint hash。

迁移细节必须显式实现而不是 `strict=False`：

- Decoder 扩宽首层时旧 1022 列复制，新列零初始化；原后续层完整复制。
- WM 首层旧 133 列复制、其余列零初始化；原 output70 行保留，新增 output12 行独立初始化。独立旧/新输入输出 normalizer 避免旧 count 冻结新增特征；共享 backbone 允许训练时学到双向物体作用。
- Critic 首层旧 914 列仅在完整来源含 critic 时可复制，其余新列零初始化；若只有周期 `ckpt_*.pt`，critic 从头初始化并明确记录。
- 现有 `last.pt` 有 critic/三个 Adam 状态；小 `ckpt_*.pt` 仅含 pi/wm。架构扩展 run 重新建立 optimizer，不强行加载形状不匹配的 Adam tensors，也不延用旧 iteration 假装连续 HOI 训练。
- 初始 action / robot WM 输出的兼容检查，应在冻结旧 normalizer、新增输入权重为零、相同 robot 输入下比较；新增特征归一化不能改变旧块数值。迁移完成不代表 HOI 动力学已经学会。
- 同 schema 续训仍需 fresh replay prefill；当前 checkpoint 没有完整 replay / simulator / RNG 状态，不能称逐位恢复。

训练不增加 teacher producer，不调用旧 DAgger/flow losses。所有阶段使用新输出目录；不覆盖既有日志、checkpoint、rollout 或渲染。

## 9. 验收、实验对照与报告

### 9.1 实现后必须通过的 gate

| Gate | 验证内容 | 不满足时的含义 |
| --- | --- | --- |
| CPU data/schema | selection/逐 motion SHA、`[T,630]` 与 rollout/checkpoint 类型区分、有限值、quat/关节名、分组及 canonical 名单、源/派生 velocity、contact mask、history/reset/window 对齐 | 数据或接口错误，不能解释为学习难度 |
| Time/frame alignment | 独立核对 source60/control50、robot/object 同 `t/i0/i1/alpha`、N−1 时长、整数步数与 T+1/T；601→500 actions/501 states；不把 protocol assumption 当生成事实 | 采样错速或少/多一帧，不以总帧数偶然相同代替时间核对 |
| Legacy compatibility | 相同输入下迁移前后 human latent、initial action、robot WM 输出一致；原 reward/termination 数值一致 | “只增不减”或迁移没有成立 |
| Native physics | hand/self ON 编译配置、真实 object gravity/mass/COM/inertia/friction、无异常穿透/接触overflow；pair probe 排除手碰地误标 | CPU 检查不能替代此 gate |
| Geometry | 单一世界下 robot FK / hand collider / object cloud 对齐，mesh/COM 不混淆，表面距离误差可量化 | 交互 reward/contact label 不可信 |
| WM holdout | 按 no-contact/contact/onset/release 分层报告 1/8-step error、32-step drift；与 zero-increment/constant-velocity baseline 比较 | 总体 MSE 低仍可能没学会接触 |
| Contact | 真实 pair 数据中逐类 PR、positive coverage、概率校准及 native force 定义；失败例可回放 | 不能用 net-force accuracy 替代 |
| BPTT | object reward 对 action 的非零有限梯度；固定观测噪声及输入的小批 rollout 做有限差分方向检查；WM/critic 参数与统计在 actor update 不变 | 梯度断链、错误detach或参数泄漏 |
| Simulator evidence | 短 smoke 之后完整 reference-duration 的真实验证，报告 simulated 与 imagined reward gap | WM 自我奖励不能证明 HOI 成功 |

WM/contact gate 的数值合格线在 train-only native calibration 样本上预先登记，至少要求相关任务分层优于上述简单预测基线、接触正例可检出且 BPTT 方向检查通过；不编造尚未测量的成功率阈值。

后续实现时运行仓库要求的 CPU 单测与 HOI 边界检查；新增训练/物理在已准备的 Linux GPU runtime 检查。此次文档阶段没有运行仿真、smoke 或正式训练。

### 9.2 最小实验矩阵

| 名称 | 定义 | 回答的问题 |
| --- | --- | --- |
| ZS | 冻结旧策略，物体真实物理；明确 checkpoint 和碰撞 profile | 原 motion tracker 在相同条件下的下限 |
| HOI-v1 | 本文完整新增分支、联合 WM、contact辅助、全部新 reward、无16-step硬终止 | 主方法是否在真实 simulator 改善交互 |
| No-contact-objective | 保留网络与 contact 预测诊断，仅将 contact match reward 权重置0，独立训练 | 接触目标是否有额外作用 |
| No-geometry-input | 保留输入接口及网络，将 actor/WM/critic shape feature 一致置0，独立训练 | 显式几何是否帮助跨 asset/scale 学习 |
| Contact-timeout16（后续） | 加完整 task counter、critic与可微规则后独立训练 | 硬 contact termination 的收益及探索代价 |

所有对照使用新 run。增加训练输入的 full-reference HOI 不再称 zero-shot；Ultra/Protomotions teacher 因引擎、控制与训练数据不同，仅作注明条件的外部参考，不作为公平 matched baseline。正式主对照建议使用预先固定的 3 个 seeds，开发 smoke 使用单 seed；预算变化需在运行前记录。

### 9.3 评估与交付数据

- 保留作者 human normalized return / termination，现有 39-body PHC、object-pose pass、human+object pose joint pass，并分别报告；`supermimic_n_failed=0` 不等于 HOI 全成功。
- 新增物体 position/rotation/velocity error 分布、pair actual/expected contact 覆盖、missing/extra contact、连续缺失时长、support/contact transitions，以及阶段可解释时的 drop/release 诊断。
- “交互成功”需在锁定协议中定义为 pose+human+expected-contact 条件的联合结果，并报告标签有效覆盖；object pose pass 或 contact probability 高不能直接称 grasp success。没有经过验证的抓取判据时沿用诊断名称。
- 评估冻结 checkpoint、无 optimizer / normalizer 更新；从 clip 开头到末尾保存真实完整轨迹，failure 后不 reset、不截断。作者 return 可另按首次 failure 停止累计，full-duration HOI 指标仍保留。
- 完整时长按第 3.4 节 canonical `(N-1)/60` 及完整 control ticks 定义。`state/reference=T+1`，`action/observation/goal/reward=T`，同时保存真实时间、源 N/索引/插值方式、协议时长/尾端余量、timebase证据状态、velocity mode、quat、source selection/逐 motion SHA、object/contact schema；不能凭帧数相同认定对齐。
- 完整 raw 明确包含 `body_rot`，允许独立 CPU 从保存状态/reference 重算既有 PHC 与 object 指标，核对模型/normalizer 不变及 physics/API 门禁。ProtoMotions G19500 的验收格式可参考，其成绩属于该模型/观测/物理/训练组合，不作为 SuperMimic 的成功阈值。不同长度 clip padding 只用于 batch，评分严格按各自真实长度 mask。
- 离线视频只回放记录状态，不重跑 policy 或物理；reference robot/object 半透明蓝色，actual 独立配色，同世界、同相机、不分开对齐，保存全帧并核对真实解码帧数。
- W&B 保持现有私有项目，每个对照新 run，仅配置/指标/汇总；不上传权重、原始数据、完整 rollout、源码或凭据。

## 10. 后续实现拆分与当前边界

实现顺序建议按可独立审查的里程碑推进：

1. 新 HOI training data/schema/manifest 层，从只读 v4 源目录与冻结 selection 建立身份绑定，验证连续时间/帧数及派生 velocity；复用并保护 canonical zero-shot 输入与评估，不更改旧协议常量。
2. Native 单 asset scene、object state、pair-contact recorder、联合 reset 与几何/物性 probe。
3. 增量 actor / WM / critic 与 legacy migration，先通过静态尺寸和原输出兼容检查。
4. Replay / observation / reward / soft termination / imagined rollout 全路径接入，完成有限差分检查。
5. 多 asset 分组采样、warmup / WM gate、eval80 / test40 的新训练评估入口。
6. 小规模 train-only smoke 后，再单独准备正式训练配置与资源记录。

预计新建独立 HOI training 入口及专用 data/MDP/network wrapper；共享层可在必要处增加显式扩展接口。具体文件名在实现时确定，避免仅靠修改全局 `OBS_DIM` / `STATE_DIM` 破坏既有 `supermimic.train` 和 `hoi.run`。不复制第二套未经共享配置管理的 physics 默认值。

交接已提供集群数据存在、冻结训练/验证引用存在及样例 SHA 的既有证据；第一阶段另已完成 selection/分区与集群 eval80 身份核对、一条真实 motion 的字节解码和 CPU 时间校验。当前仍需取得的证据包括：全量逐 motion 解码、发布数据实际 timebase、source/derived velocity 物理语义与源/native object velocity 参考点、pair-contact GPU instrumentation 能力、完整 asset/scale 的异构 scene 支持、初始 reference 接触质量及显存预算。CPU 数据验收不等于上述 native 或训练验收；当前实现与实际数据读取范围见 [实现记录](hoi_v1_data_stage1_20261023.md)。

ProtoMotions distill 已按用户要求停止、相关监控暂停；历史20200续训目标已被停训要求覆盖。这里只复用数据/协议和证据格式，不恢复其训练或监控，也不将交接中已有任务终态当成本项目新任务状态。

## 11. 源码与资料溯源

### 本仓库

分支创建基点：`2318dfd604d0a254686760e15c034231a94aa0aa`。首轮新增本文，本轮依据数据交接更新本文；用户原始 v1 笔记与交接文件不改动。

- [DEVELOPMENT.md](../DEVELOPMENT.md)：物理默认、历史边界与实验保护。
- [networks.py](../supermimic/networks.py)、[mdp.py](../supermimic/mdp.py)、[motion.py](../supermimic/motion.py)：原网络/解析 heads/goal 定义。
- [train.py](../supermimic/train.py)、[buffer.py](../supermimic/buffer.py)、[env.py](../supermimic/env.py)：BPTT、replay、normalization、checkpoint 和 native state/control。
- [hoi/prepare.py](../hoi/prepare.py)、[ultra_reference.py](../hoi/ultra_reference.py)、[evaluate.py](../hoi/evaluate.py)、[hand_collision.py](../hoi/hand_collision.py)：历史 canonical 时间契约、真实物体和手部资产审计。

### 新增数据交接证据

- [ULTRA_DATASET_AND_TIMEBASE_HANDOFF_20261023.zh-CN.md](ULTRA_DATASET_AND_TIMEBASE_HANDOFF_20261023.zh-CN.md)：本轮数据更新的直接依据；读取时 SHA256 为 `d4d1e6472b327054ddffb10a555f636b39821da71a28d74247ed464dbfca8fda`。
- 交接引用的 Windows 核查记录：`C:/Users/张天姿/Documents/ChatGPT/ProtoMotions/.local/hoi_dagger_G_continue15200_20261020/data_timebase_20261023/CLUSTER_DATA_OBSERVED.json` 与同目录 `DATASET_COUNT.json`。文档修订阶段通过交接接收其结论，没有独立重读这些记录或重新 SSH；后续第一阶段的只读连接及取样另见实现记录。
- 实际集群源路径、训练 selection SHA、eval80 路径与样例 SHA 见第 3.1 节。数据引用和派生物只写新目录，不覆盖共享源数据及历史结果。

### Protomotions：`hoi/dagger_distillation`

前轮本地审查 HEAD：`79b94ef15ea5dd4bcdb479ecc430d3907a3a851e`。交接明确这是后续验收证据提交，**不是 G19500 的新 worker 源码身份**；该次实际训练/评测的源快照为 `faefebd52e1e5872f008532f48606d86568d86b6`（512 个源文件）。两种身份分别保存，不互相替代。前轮跨仓库 `git status` 受本地权限/工作树路径问题影响，未确认全部已有改动；数据交接文档修订轮未重新审计外仓或连接集群，第一阶段的数据只读核验单独记录。

- `D:/Protomotions/repro/ultra_hoi_20261016/reference/dr/inputs/data_manifest.json`：seed20261013、分组规则和四分区；SHA256 `7d66cd15b27b41988ac3ef761fb82f15204b5ab741a696f95102801dac8b62e2`。
- 同目录 `evaluation_cases.json`：canonical eval80；SHA256 `459a3925faf0f66bee258ce98635c67802ddda00d47d007ecdeb45035f576002`。
- `D:/Protomotions/scripts/hoi/ultra_migration/experiments/grasp_nativehands_adaptive_20261015/test40_triplet_20261017/preregistered/setup/test40_cases.json`：canonical test40；SHA256 `b7e61757f230930fe18672f619149d7201ccdb6ab7a6ac06de0ae1634cf7674d`。
- `D:/Protomotions/scripts/hoi/dagger_train.py:290` 的 net-force contact 使用、`D:/Protomotions/protomotions/envs/hoi/virtual_markers.py:12` 的逻辑 marker 定义，仅作为迁移审计依据，不复用 DAgger/teacher 训练。
- `D:/Protomotions/repro/ultra_hoi_20261016/reference/dr/inputs/baseline/runtime.json:3983`：逻辑 hand marker 的固定偏移；真实 collider 仍以本仓库 `hoi/hand_collision.py` 的可追溯安装变换为准。
- 交接列出的运行协议源码：`D:/Protomotions/protomotions/envs/hoi/ultra_reference.py`、`ultra_motion_bank.py`（同目录），`D:/Protomotions/scripts/hoi/dagger_train.py`、`dagger_evaluation.py`，以及 `D:/Protomotions/protomotions/envs/hoi/failure_segment_sampler.py`：连续时间采样、SHA加载、控制时间断言、完整评测步数及整数源帧时长。浮点尾端规范化不等同确认真实源 FPS。
- `D:/Protomotions/notes/ULTRA_HOI_FROZEN_RUNBOOK.zh-CN.md`：交接引用的50Hz/5ms×4与运行边界。G19500 目录中的 `acceptance/CHECKPOINT_CPU_ACCEPTED.json`、`FINAL_CPU_ACCEPTED.json`、`CLOUD_ACCEPTED.json` 和逐 case `rollout.pt` 仅作历史验收格式溯源，不是本项目验收结果。

### Ultra：`main`

前轮本地审查 HEAD：`ebe4aa276dd3208c33a8604084c0a3aeb37d9061`。当时整体 `git status` 未确认；已只读核对的关键 exporter/consumer/config 与其 Git index 一致。本机该 checkout 没有发布动作目录，前轮未核验数据实物；这不否定新增交接中集群已有 v4 数据的只读观测。本轮未重新读取该外仓。

- `D:/Ultra/ultra/env/tasks/ultra_g1_retarget.py:166`：630D consumer；`:571/:612` 为 contact counter 与 `>20`；`:1198` 附近为 contact/reward。
- `D:/Ultra/ultra/env/tasks/ultra_g1.py:31`：当前 exporter 的状态捕获、39 flags 与启动帧裁切。
- `D:/Ultra/ultra/env/tasks/ultra.py:259`：object cooking；`:278` 为 surface-even 点云与 mesh-mean 中心化。
- `D:/Ultra/ultra/data/cfg/g1_teacher.yaml:15`、`ultra/utils/config.py:39`、`docs/deploy.md:16`：17ms 与 60Hz 的区别。
- `D:/Ultra/ultra/env/tasks/humanoid.py:924`：名为 SDF 的最近点位移，不是 signed-distance field。

### 论文资料

- [SuperMimic](https://arxiv.org/abs/2610.28480)：联合训练 WM 与 policy、通过短 imagined rollout 的 BPTT 学习；本文网络尺寸以本仓库实现为准。
- [ULTRA v3](https://arxiv.org/html/2603.03279v3)：dense-reference teacher 到 multimodal student 的框架；其训练范式不属于本版迁移范围。论文中的 contact mismatch 20 frames 概述与本地具体 `>20` 规则分开记录。
- [ContactMimic v1](https://arxiv.org/html/2607.08742v1)：明确区分 reference 几何接触标签与执行中的力学接触。本文的 2/4cm mask 与 contact-head 方案是新设计，并非该文原配置。
