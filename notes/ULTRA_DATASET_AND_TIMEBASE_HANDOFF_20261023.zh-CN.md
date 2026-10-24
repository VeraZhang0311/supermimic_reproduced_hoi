# ULTRA 数据与时间基准交接：ProtoMotions → SuperMimic

记录日期：2026-10-23（America/New_York）。供 `D:/supermimic_reproduced_hoi` 的 Codex 复用已有数据与核对时间协议。

**最重要的事实：ProtoMotions 使用的是“按 60 Hz 协议解释源动作，再以 50 Hz 控制机器人”。50 Hz 控制已经在我们的实现和运行中验证；发布数据真实生成时的采样间隔是否为 1/60 秒，仍未由生成来源或原始时间戳证实。集群上确实保存着实际数据。**

本文件是已有工作的事实交接，不是启动训练、重跑评测、改时间基准或改目标项目物理配置的授权。本次写入时目标仓实际分支为 `hoi/v1.0`；保留了该分支及已有文件，没有切换分支。

## 1. 我们做了什么

- 在 `D:/ProtoMotions` 的 `hoi/dagger_distillation` 开发线上，完成了 ULTRA 数据/teacher 接入 Isaac Lab、50 Hz 连续时间 reference 采样，以及后续 teacher 适应/PPO 和 DAgger 学生实验。
- 保留原始动作 `.pt` 字节，通过路径、SHA-256 和固定 selection 清单绑定数据；没有把共享源数据批量覆盖成另一套 50 fps 文件。
- 源动作使用既有的 60 Hz 解释协议；仿真控制时间以秒推进，按时刻插值得到机器人与物体 reference。
- 训练与验证按 `source_group` 隔离，训练采用 2651 条动作 / 381 个 source groups；固定同 80 条验证使用 80 个 source groups。80 条已多次用于开发/选模，不能称为独立 test。
- 完整评测保存实际状态、`body_rot`、reference、时间和逐条 raw，独立 CPU 重算指标并核对 SHA、物理门禁、模型不变、云端及作业终态。
- 最近一次 G19500 同 80 条评测已验收：PHC 72/80、object pose 24/80、联合 24/80；全部 80 条包含 23798 个保存状态帧。对象 pose 通过不等于已验证抓取成功。
- 已按用户要求停止该 distill，相关监控已暂停。历史 20200 续训目标被用户停训要求覆盖；本交接不恢复该训练或提交其余历史任务。

这些结果属于 ProtoMotions 的具体模型、观测、物理和训练组合。数据可复用，不代表直接复制这些设置后就与 SuperMimic 实验等价，也不应自动给 SuperMimic policy 加入物体观测。

## 2. 集群数据的实际位置

登录入口：

```text
tianzizh@grasp-login1.seas.upenn.edu
```

实际源数据目录：

```text
/mnt/projects/ll/tianzizh/ProtoMotions/data/omomo/OMOMO_retarget_aug_v4
```

2026-10-23 通过已校验主机的 SSH 实际只读检查得到：

| 项目 | 实际观察 |
|---|---|
| 直接位于目录中的 `.pt` 文件 | 4066 个 |
| `.pt` 文件总字节数 | 3,531,348,002 bytes，约 3.53 GB / 3.29 GiB |
| 目录类型 | 实际目录；检查时不是符号链接 |
| 非 `.pt` 条目 | 该目录本次未发现 |
| 冻结训练/验证清单引用的 motion 文件 | 全部存在 |
| SHA 抽查 | 下述样例与冻结清单相符 |

样例文件：

```text
/mnt/projects/ll/tianzizh/ProtoMotions/data/omomo/OMOMO_retarget_aug_v4/sub10_largebox_000_075_086_074_100_100_100.pt
```

该文件为 671819 bytes，SHA-256：

```text
1a0bac419adab2bf6d7f5d10d480e40d0f3c39a74605a7063c7f48b0f99bf97c
```

本次核查确认了所有清单路径存在，并重算了该样例 SHA；不是重新对全部 4066 个文件做了一次 SHA 扫描。文件存在、文件修改时间或 SHA 一致，都不能单独证明逐帧物理采样间隔。

这里的动作文件是现有 consumer 按 `[T, 630]` Tensor 解码的 ULTRA 打包 reference，含机器人、物体、body pose/velocity 与 contact 等字段。它们与 50 Hz 仿真产生的 `rollout.pt`、学生 checkpoint 是不同类型的数据。具体字段布局、关节/物体映射、四元数顺序须继续按解码源码核对，不能凭文件后缀混用。

共享目录作为源数据使用。新项目的清单、派生重采样文件和实验输出写到独立目录，保留原文件与 SHA。

## 3. 可复用的冻结训练/验证清单

训练清单（2651 motions / 381 source groups）：

```text
/mnt/projects/ll/tianzizh/ProtoMotions/runs/dagger_object5800_G_recovery_from18600_to20200_20261021_01/inputs/selection.json
```

本次实际读取的训练清单 SHA-256：

```text
8c141eb60650b9df79b92acfeeea785f796c7f18f52bf179467562c7cc315b23
```

同 80 条验证清单（80 motions / 80 source groups）：

```text
/mnt/projects/ll/tianzizh/ProtoMotions/runs/dagger_object5800_G_recovery_from18600_to20200_20261021_01/evaluations/requested_19500/inputs/evaluation_cases.json
```

清单的 `cases` 中包含 `motion`、`motion_sha256`、`source_group`、物体/资产等绑定。复用历史对照时应复用明确的清单与分组隔离，不能将目录中的 4066 个文件直接等同于训练集，也不能重新随机切文件造成同一 source group 跨训练/验证。

路径是检查时的绝对路径，不是随意拼出来的目录示例。新项目可以建立自己的 manifest，但应记录所用源 selection 的 SHA 和每个 motion 的 SHA。

## 4. 50 Hz 是如何实现的

我们区分源动作时间协议、物理步长和控制步长：

| 量 | ProtoMotions 这套实现的值 | 含义 |
|---|---|---|
| 源动作解释帧率 | 60 fps | consumer 的既有协议；生成采样率仍待来源确认 |
| 物理步长 | 0.005 s | 200 Hz 物理积分 |
| 控制 decimation | 4 | 一个 action 保持 4 个物理步 |
| 控制步长 | 0.005 × 4 = 0.02 s | 50 Hz 策略/控制更新 |
| 源动作解释时长 | `(N - 1) / 60` s | N 个源状态帧间有 N−1 个间隔 |
| 完整评测控制步数 | `floor(((N - 1) / 60) / 0.02)` | 保存状态包含初态，即步数 + 1 帧 |

对任意 reference 时间 `t`：

```text
source_index = clamp(t * 60, 0, N - 1)
i0 = floor(source_index)
i1 = min(i0 + 1, N - 1)
alpha = source_index - i0
```

- 位置等连续字段在相邻源帧间线性插值。
- 根、物体和 body 的四元数使用 SLERP。
- contact 字段按最近源帧取样，沿用既有语义。
- 机器人与物体共用同一个 `t`、源帧索引和插值权重，保持同一时间轴。
- 每个控制步 reference 时间前进 0.02 秒；不是简单地每个控制步取下一源帧。
- 沿用 consumer 的速度字段语义，并非已经根据 50 Hz 重新计算了整份发布数据的速度。

从 reference 时间零开始时：

```text
控制时间(s)：0.00  0.02  0.04  0.06 ...
源帧索引：  0.0   1.2   2.4   3.6  ...
```

例如 601 个源状态帧，按 60 Hz 协议解释为 10 秒；50 Hz 控制下为 500 个控制步，保存 501 个状态帧。源 reference 中以秒定义的前瞻偏移也必须保留其原语义，不能把所有出现的 `60` 一律改为 `50`。

这说明“如何在 50 Hz 下使用既有 reference”，不证明源数据本来就是 50 fps，也不证明原始生成时钟已经确认。SuperMimic 若选择其他控制频率，应显式记录它自己的值；本节不是要求修改该项目的默认控制频率或物理配置。

## 5. 60 Hz 协议与 17 ms 差异的证据边界

本机当前 `D:/Ultra` 的 G1 consumer 标记 `fps_data = 60`；当前 exporter 在控制步捕获状态。另一项目已指出 `17 × 1 ms = 17 ms`、约 58.823529 Hz 的差异。**当前源码的导出行为不能反向证明既有 `OMOMO_retarget_aug_v4` 发布文件必定由同一代码、配置和时间设置生成。**

本交接的结论分开记录：

| 结论 | 状态 |
|---|---|
| ProtoMotions consumer 按 60 Hz 解释源动作 | 源码已核实 |
| ProtoMotions 控制是 50 Hz，物理为 5 ms × 4 | 源码/已有运行证据已核实 |
| reference 按连续时间插值，机器人与物体同轴 | 源码已核实 |
| 集群实际有 v4 数据、清单文件存在、样例 SHA 相符 | 本次只读检查已核实 |
| 发布文件真实生成间隔为 1/60 秒 | 本次未由生成来源或时间戳证实 |
| 发布文件真实生成间隔为 17 ms | 本次也未由生成来源或时间戳证实 |

因此，已有实验应准确描述为“canonical 60 Hz 解释下、50 Hz 控制的实验”。此前“保持动作时长”的表述，指保持 `(N−1)/60` 这一协议时长，不是已经证实原始生成的物理时长。

若 601 帧实际对应 17 ms 间隔，则真实总时长是 10.2 秒，而不是 canonical 的 10 秒，差异为 2%。换成 50 Hz 控制本身不会消除这一来源问题。

正式新训练前，应继续查生成脚本的实际版本/运行配置、发布说明或原始逐帧时间戳，并把“已证实”与“仅协议假定”写入新 manifest。若取得支持 17 ms 的新证据，另建带版本的时间配置，机器人与物体同步处理，并核对速度、接触、时长和评测语义；必要对照在新版本中重新运行。不能只改训练侧、原地覆盖源数据，或静默改写已有 HOI80 历史报告。

下面仅是建议的新 manifest 字段示意，**不是说旧 ProtoMotions 清单已经包含这些字段**：

```json
{
  "timebase_mode": "canonical_60hz",
  "source_fps_assumed": 60.0,
  "source_timebase_status": "protocol_assumed_not_generation_verified",
  "reference_sampling": "continuous_time_linear_and_quaternion_slerp",
  "physics_dt_s": 0.005,
  "control_decimation": 4,
  "control_dt_s": 0.02
}
```

示例里的物理/控制值描述 ProtoMotions；目标项目应填写自己实际选定并验证的值。

## 6. 可核对的源码与本地证据

### ProtoMotions 源码

- `D:/ProtoMotions/protomotions/envs/hoi/ultra_reference.py`：60 Hz 时长、连续时间索引、线性/SLERP/最近 contact 取样。
- `D:/ProtoMotions/protomotions/envs/hoi/ultra_motion_bank.py`：按文件 SHA 读取 `[T,630]` 动作、时长与多动作取样。
- `D:/ProtoMotions/scripts/hoi/dagger_train.py`：`cfg.sim.fps == 200`、`decimation == 4`，控制时间推进断言。
- `D:/ProtoMotions/scripts/hoi/dagger_evaluation.py`：`full_steps_for_frames` 与独立 reference/metric 重算。
- `D:/ProtoMotions/protomotions/envs/hoi/failure_segment_sampler.py`：基于整数源帧数的 `(N−1)/60` duration 元数据。此前 CPU/CUDA 微小舍入差异的规范化修复不等于证明真实源 FPS，也没有更改源 reference 时间协议。
- `D:/ProtoMotions/notes/ULTRA_HOI_FROZEN_RUNBOOK.zh-CN.md`：50 Hz、5 ms × 4、固定划分与运行边界。
- `D:/Ultra/ultra/env/tasks/ultra_g1.py`：当前 consumer 与当前导出路径；仅用于核对当前代码行为。

G19500 实际训练/评测源快照为 `faefebd52e1e5872f008532f48606d86568d86b6`（512 个源文件）；`79b94ef15ea5dd4bcdb479ecc430d3907a3a851e` 是后续验收证据提交，不是新的 worker 源码身份。

### 本次集群数据核查记录（Windows）

```text
C:/Users/张天姿/Documents/ChatGPT/ProtoMotions/.local/hoi_dagger_G_continue15200_20261020/data_timebase_20261023/CLUSTER_DATA_OBSERVED.json
C:/Users/张天姿/Documents/ChatGPT/ProtoMotions/.local/hoi_dagger_G_continue15200_20261020/data_timebase_20261023/DATASET_COUNT.json
```

核查仅使用目录/JSON/文件字节读取及样例 SHA；未在 login 节点加载 Torch、调用 GPU、提交 Slurm 作业或启动仿真。

### 最近 G19500 完整验收示例

评测目录：

```text
/mnt/projects/ll/tianzizh/ProtoMotions/runs/dagger_object5800_G_recovery_from18600_to20200_20261021_01/evaluations/requested_19500
```

其中 `acceptance/CHECKPOINT_CPU_ACCEPTED.json`、`acceptance/FINAL_CPU_ACCEPTED.json`、`acceptance/CLOUD_ACCEPTED.json` 以及 `results/cases/<case_id>/rollout.pt` 可用于理解已有 raw/时间/指标保存约定。完整 raw 含 `body_rot`，CPU 重算和物理/API 门禁通过，评测所有 Slurm step 为 COMPLETED 0:0，GPU worker 已退出。

[该次 W&B 评测结果](https://wandb.ai/tianzizh-university-of-pennsylvania/hoi-dagger-distillation/runs/hoi_object5800_G_recovery18600_to20200_20261021_01_eval_requested19500)

## 7. 交给接收项目 Codex 的操作边界

1. 先读本项目自己的 `AGENTS.md`、`DEVELOPMENT.md` 及 v1 设计；数据交接不覆盖本项目的模型、观测或碰撞开关约定。
2. 以实际集群路径读取数据；先核对 selection、source group 划分与 SHA，再建立新项目的 manifest。不要将共享源目录改写为派生产物。
3. 将 `source_fps`、控制频率、物理步长和时间基准证据状态分别记录。`canonical_60hz` 是协议选择，不能写成“源数据真实 60 Hz 已证实”。
4. 比较实验时保持完整 motion 时长与机器人/物体同一时间轴；不能因失败重置或截短来改变评测。沿用各项目真实的状态/action 帧对齐约定并单独核对，不只比一个总帧数。
5. 若新项目按自身授权做了物体观测、碰撞、时间基准或训练变化，作为新配置/新实验记录，不冒充 ProtoMotions 或本项目的历史基线。
6. 源数据、权重和完整 raw 不加入 Git，不上传到 W&B。本文不会复制数据、启动任务、修改历史结果，或恢复已停止的 ProtoMotions distill。
