# Frozen privileged GMT + residual PPO HOI teacher

本分支在 `supermimic_reproduced_hoi` 的 `boneseed/privileged` 实现上增加
`hoi_residual/`。目标是使用本体与物体 oracle 的 HOI teacher；没有部署观测约束、
student 蒸馏或 world model 学习。ResMimic 对照仓库位于 `D:/ResMimic`，
固定源版本 `fa20d98dbbf9b5b903db20435d90a4964bcf09d9`。

## 继承边界

- 使用指定 26000 GMT checkpoint 的 `pi` 及其全部 RMS。加载器不实例化 WM、
  GMT critic 或旧 optimizer；使用 `Policy(99, 730, hist=10, privileged_dim=3371)`。
- GMT schema、38-link/29-joint 顺序、URDF、物理与 analytic implementation SHA
  必须严格匹配当前运行代码。只有 reference dataset 与 manifest SHA 可以变化。
  checkpoint 自身 SHA256 写入新结果，`assert_unchanged()` 逐值检查全部权重和 RMS。
- 原 GMT 的 24 个共享源码文件保持字节不变。Isaac Lab + Newton + MuJoCo Warp、
  robot asset、显式电机、PD、action scale、控制延迟、碰撞与材质继续由共享实现管理。
  50 Hz action / 200 Hz 电机子步不变。
- HOI scene 添加真实动态物体。原构造器缺少 scene 参数，`HOISim` 用
  `FunctionType` 在私有 globals 副本中绑定扩展 scene，执行原构造器字节码；
  不修改原模块 globals、原函数或共享文件。每步只调用一次共享 `step_target`，
  不叠加第二套 PD；共享 runtime 的 finite/overflow 硬错误保留。
  新场景显式把 contact/constraint 容量从 GMT 的 `nconmax=128, njmax=500`
  增为 `512, 2048`，用于额外物体 hull；这是 buffer 容量变更，不改变 solver cone、
  integrator、时步、摩擦、碰撞几何或机器人参数，且写入 run config。

## 网络和动作

残差 actor 与 critic 均输入 `3442D oracle + 29D frozen base action`。
Oracle 由原样 3371D robot privileged block、15D 当前物体状态、两个未来时刻
的共 30D 物体误差、12D 物体/地面物理上下文、8D asset one-hot、6D shape bounds 组成。
未来偏移为 `[1, 13]`，对象特征采用机器人 heading 坐标系；所有字段在
`features.py` 有版本化 schema，不能静默换序或加减字段。

两套独立 MLP 的隐藏层均为 `[512, 512, 256, 128]`、SiLU、最后隐藏层 LayerNorm。
残差最后线性层全零初始化，因此 deterministic residual 初始为零；可训练 Gaussian
初始标准差为 `0.1`。这保留 ResMimic 的 additive residual 设计，但用明确的 oracle
schema 取代上游 student history/future 拼接网络。上游发布版的 object 7D 与 future
切片不一致，以及 `final_layer_gain` 未传递的问题不带入本实现。

执行动作是 `clamp(base + sampled_residual, -20, 20)`，随后仍由原 action scale
映射到关节目标。PPO storage 保存**未裁剪的 sampled residual**及其 log probability；
不能用裁剪后的 combined action 反推 residual 并计算 likelihood。
GMT 的原 AMP 设置由训练入口恢复；残差 PPO 使用 FP32。

HOI 启动器在导入 Isaac Lab / Newton 前调用
`runtime.configure_warp_forward_only()`：通过官方 `wp.config.enable_backward=False`
及 `wp.set_module_options` 关闭未使用的 Warp 反向代码生成，显式覆盖 Newton 的
MuJoCo kernel module；PyTorch 的 PPO autograd 仍正常启用。这里不改 kernel 源码、
forward 算法、求解器参数或共享 runtime，只改变编译入口并保存配置 receipt。
Warp 在模块注册时复制全局默认，因此此调用必须放在 simulator imports 之前。

此设置有具体兼容依据：R525 失败缓存中 contact-conversion forward 参数为
3016 bytes，自动生成的 backward 为 6008 bytes，后者是该模块唯一超过 4096 bytes
的入口；超过此限制需要 R530，旧驱动报 `CUDA_ERROR_NOT_SUPPORTED`。
参见 [NVIDIA kernel parameter compatibility](https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/)
与 [Warp 1.17 configuration](https://nvidia.github.io/warp/v1.17/user_guide/configuration.html)。
这说明可移除的具体驱动要求，不保证任意旧 GPU/runtime 均受支持；真实 GPU smoke
通过后才能判定当前组合可运行。

## PPO 与多资产

直接采集真实物理 rollout，再执行 PPO；没有 WM rollout、replay 或 DAgger/KL teacher loss。
默认每轮 24 步、5 epochs、4 minibatches、Adam `2e-4`、`gamma=.99`、`lambda=.95`、
clip `.2`、value coefficient `1`、entropy `.005`、gradient norm 上限 `1`。
学习率明确采用 fixed schedule，区别于上游 adaptive-KL 学习率。

Residual normalizer 只在下一轮 rollout 开始前更新一次；采集和该轮全部 PPO epochs
使用相同统计量。Protected eval 不更新任何 normalizer。
真正 failure 的 bootstrap 为零；time limit/motion end 从 **reset 前最后观测** 的 value
bootstrap；两种边界都截断 GAE trace，不能接入下一个 episode 的 value/advantage。

checkpoint 另绑定 `contract.py` 中 12 个残差学习实现文件的 SHA，包括训练时序、
奖励、观测、资产处理与随机化。评估拒绝实现、manifest 或资产 catalog 身份不符的权重。

默认完整训练由 8 个（或 8 的整数倍）distributed ranks 共同训练**同一套策略**：每个 rank 固定一种
exact asset geometry，并从该资产的 train clips 采样，保持单 worker 中所有 MuJoCo Warp
worlds 拓扑一致。训练入口广播初始参数与 buffers，随后跨 ranks 汇总 normalizer moments、
advantages 与 gradients；梯度平均后再 clip。各 rank 的 rollout 长度、环境数、epochs
和 minibatch 数必须一致。当前入口把 `LOCAL_RANK` 映射为该节点的 CUDA device，
因此每节点的 rank 数须与分配 GPU 数匹配；资源由 Slurm/torchrun 显式配置。
单资产运行必须显式 `--asset`，或作为 `--smoke`；它不代表四类八资产训练已完成。

## 奖励适配

`rewards.py` 采用 ResMimic 的 robot tracking 与物体跟踪核心，机器人 FK 使用当前
GMT 的 14 个 tracked moving links、29 个关节等权。正奖励为：

| 项 | 权重与函数 |
|---|---|
| 关节位置 | `2 exp(-.15 sum(dq²))` |
| 关节速度 | `.2 exp(-.01 sum(dqd²))` |
| 根高度 / 姿态 | 各 `exp(-5 error²)`，姿态误差为旋转角 |
| 根线速度 / 角速度 | 各 `exp(-sum(error²))` |
| heading-local body / global body | 分别 `1` / `10` 乘 `exp(-10 sum(body_error²))` |
| 物体几何 | `2 exp(-10 mean_i ||R p_i+t - (R_ref p_i+t_ref)||₂)` |

物体项使用 1024 个固定、按面积采样的碰撞组件表面点，seed 2024，长度单位米。
它是**对应点平均位移**，不是 Chamfer 或平均平方距离。采样的是 bound convex components，
重叠组件可带有内部面，不声称精确重建原 visual surface。

保留 action rate（`-.01 ||a-a_prev||₂`）、joint limits（权重 `-5`）、joint velocity
（`-1e-4 sum(qd²)`）/ acceleration（`-5e-8 sum(((qd-qd_prev)/dt)²)`）正则项；
具体版本化公式以 `rewards.py` 为准。没有移植上游 feet slip/stumble/air-time/net-force heuristics 或 torque
辅助项，也不把源数据的 39-body contact flags 当作真实 pair-contact impulse。
全部奖励乘当前 `0.02 s` 控制时基。机器人 termination 沿用 GMT；训练额外在物体对应点
平均误差大于 `0.3 m` 时结束 episode。Protected eval 应保留完整参考时长并累计失败，
不得通过失败重置或截短后报告成功。

## 域随机

机器人 DR 完整继承 privileged GMT，不引入 ResMimic 的另一套 motor 或碰撞参数。
物体每次 reset 独立随机：质量乘 `[.7, 1.3]`、COM 每轴偏移 `[-.01, .01] m`、
sliding friction `[.4, .8]`；COM inertia tensor 随质量同比缩放。Oracle 包含实际抽样参数。
Nominal eval 关闭 DR，物体使用 bound nominal profiles；地面摩擦沿用 GMT 的
训练 `.3` / nominal eval `1.0`，oracle 写入实际采用的地面值。

这一点刻意不同于 ResMimic 发布的 suitcase 配置：上游 `randomize_object=False`，
其 mass/COM/friction 随机化实际被关闭；本任务要求物体和机器人训练 DR 开启。
必须在真实 simulator smoke 中核对属性同步及物体动态自由刚体，不以 CPU/mock 证明
native 求解器物理已生效。

## ULTRA train / Protected eval

数据范围为 `largebox / smallbox / plasticbox / suitcase`，每类 `080 / 100` 两种 exact
asset，共 8 种。沿用已有 canonical source-group split，绑定 split bytes、membership
与各 motion SHA；同一 source group 的增强版本不能跨 split。

| 角色 | motions | source groups |
|---|---:|---:|
| train | 2651 | 381 |
| Protected eval: validation | 517 | 80 |
| Protected eval: regression | 655 | 100 |
| Protected eval: test | 243 | 40 |
| Protected eval 合计 | 1415 | 220 |

后三部分对本次训练均禁止 optimizer、normalizer、课程或采样权重更新。validation 与
regression 历史上曾用于开发，不能称为 untouched test；test 本次 held out，也不额外
断言历史上从未访问。HOI80 是开发评估，不是独立泛化测试。

原 630-column tensors 只读，按已绑定 joint names 重排到当前 hw3 29-joint 顺序。
机器人和物体共用精确 60→50 Hz 网格；60 Hz 是 canonical 协议约定，原数据生成时钟
未独立证实。姿态 SLERP，速度在单条 clip 的 50 Hz 网格内重算，不跨 clip 差分。

## 运行与验收

先准备新的 manifest/output，使用已有冻结 asset bundle 和 GMT checkpoint；输入均只读。
smoke 示例（只在 Slurm 分配的 GPU 环境内运行）：

```bash
python -m hoi_residual.train \
  --manifest /absolute/new/manifest.json \
  --asset-bundle /absolute/frozen/assets.json \
  --gmt-checkpoint /absolute/frozen/ckpt_026000.pt \
  --output /absolute/new/hoi_teacher_smoke \
  --device cuda:0 --num-envs 2 --rollout-steps 24 --iterations 2 --smoke
```

完整任务入口相同，去掉 `--smoke`，按启动器分配 8 ranks；不要在 GRASP login node
执行 GPU workload，不改运行中的 GMT job、冻结 source 或共享 Python runtime。
例如，已经获得单节点 8 GPU 分配时可用 `torchrun --standalone --nproc_per_node=8
-m hoi_residual.train ...`。`--asset INDEX` 仅用于明确指定的单资产实验；其索引采用
canonical alphabetically sorted asset names，旧 bundle 的 `source_asset_id` 只作来源记录。
默认 `--num-envs 512` 是每个 rank 的环境数；默认 `--iterations 100000`、
`--save-every 100`、`--gmt-iteration 26000`。CPU 功能检查：

```bash
python -m unittest discover -s tests -p 'test_hoi_residual_*.py' -v
```

PPO 专项 CPU 测试包括真实 GMT policy/RMS 加载与冻结、0 WM 构造、零残差初始化、
未裁剪 action likelihood、timeout/failure GAE、真实 actor/critic optimizer 更新、
normalizer 冻结，以及 2-rank Gloo 的参数/梯度/统计同步。多进程测试需要本机 loopback。
2026-11-06 本地已有 `.venv-hoi-tests` 的 PPO 专项 7 项全部通过（15.312 s），包括真实
两进程 Gloo；Windows 沙盒禁止 loopback 的第一次运行失败，经允许本机通信后通过。
数据/观测/奖励 15 项与完整时长评估协议 2 项也通过，Warp 前向编译配置 3 项通过，
新增 CPU 测试共 27 项。
CPU 测试、native physical smoke、PPO smoke 与正式训练收敛是不同证据；只跑通 smoke
不能声称得到高性能 HOI teacher。实际 GPU 作业、checkpoint SHA、输出路径和通过项
以对应 run receipts 为准。

2026-11-06 的真实 GPU smoke 已通过：Slurm `1508407`，`ll-l40-0`，
NVIDIA L40 / driver `535.216.01`；美东 `13:40:44–13:52:37`，退出码 `0:0`。
8 种资产分别使用 8 个物理环境、32-step rollout、2 轮 PPO，共 4096 transitions、
16 轮 PPO / 320 个 optimizer steps。每个资产均验证 actor/critic 更新、GMT 权重与
RMS 不变、物体动态刚体和 DR 原生参数读回、局部 reset 隔离、timeout bootstrap、
机器人属性一致、无原生 contact/constraint overflow。

另用 asset 0 的两轮 checkpoint 对一条 train clip 完整执行 220 steps（4.4 秒），
验证失败后继续跑到参考末帧、无中途 reset、无 optimizer/RMS/课程更新。
该轨迹发生物体跟踪失败（首个失败在第 138 步，物体表面平均误差约 0.175 m）；
因此 `passed` 表示评估实现和物理流程通过，不表示 HOI 任务成功或 teacher 收敛。
Protected 数据未用于本轮 smoke。八个独立资产 smoke 也不是八资产联合长训；
原生多 GPU / NCCL 尚未验证，已通过的是 CPU 两进程 Gloo 同步测试。

验收回执与完整报告：
`provenance/smoke_1508407_receipt.json`、`provenance/smoke_1508407_full.json`、
`provenance/smoke_1508407_train_eval.json`。
这些是本地 ignored 验收结果，不随源码提交；集群保留原始报告。
集群结果根目录是 `/mnt/projects/ll/tianzizh/hoi_residual_ppo_20261106_01/smoke_1508407`，
执行的冻结代码是同级 `source_smoke05`；其 12 个 learning module SHA 与本地一致。
正式联合长训尚未启动；默认调度资源使用 `liu-compute / liu-account / ll-med / L40`。

`.gitattributes` 对 `supermimic/*.py` 与 `hoi_residual/*.py` 关闭 Git 换行转换，
保留已验证文件的原始字节，以便新 checkout 继续匹配 checkpoint 的源码 SHA。

当前固定 Warp 1.17 / Newton / MuJoCo-Warp runtime 的 ULTRA 凸碰撞前向 kernel
参数区为 4296 bytes，超过 R525 的 4096-byte 上限；本实现需要 R530 或更新驱动。
2080 Ti / R525 的真实首步已失败并留存日志，不将初始化成功算作物理验收。
Warp 只关闭 PPO 用不到的 simulator 反向代码生成；PyTorch policy autograd、
碰撞前向代码、求解器参数和 GMT 的 BF16 推理保持原样。
驱动要求参考 [NVIDIA kernel parameter limits](https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/)。

Protected eval 入口示例：

```bash
python -m hoi_residual.evaluate --checkpoint /path/residual.pt \
  --gmt-checkpoint /path/ckpt_026000.pt --manifest /path/manifest.json \
  --asset-bundle /path/assets.json --partition test --output /path/new_eval
```

省略 `--asset` 时要求 checkpoint 覆盖全部八资产，再分八个进程顺序运行。
每条动作执行完整时长，failure 不触发 reset；报告 return、robot failure、object surface
failure、物体姿态及机器人误差，不把姿态误差阈值称作抓取成功。
实现烟测只允许 `--partition train --smoke-eval`；Protected 集不用于调通代码。
原始 rollout 未导出，报告明确保存的是逐动作数值汇总。

设计参考：[ResMimic repository](https://github.com/amazon-far/ResMimic)。
本新增实现没有复制其 simulator/PD 源码。若后续直接搬运上游文件，应保留根 Apache-2.0
`LICENSE` / `NOTICE`，以及 `rsl_rl`、`legged_gym` 各自 BSD-3-Clause 文件头与许可证。
