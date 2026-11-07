# Frozen privileged GMT + residual PPO HOI teacher

本分支在 `supermimic_reproduced_hoi` 的 `boneseed/privileged` 实现上增加
`hoi_residual/`。目标是使用本体与物体 oracle 的 HOI teacher；可共享物体输入
对齐 v1.1 actor，为后续蒸馏准备接口。本分支尚未实现 student 蒸馏，也不做 world model 学习。
ResMimic 对照仓库位于 `D:/ResMimic`，
固定源版本 `fa20d98dbbf9b5b903db20435d90a4964bcf09d9`。

## 继承边界

- 当前使用用户指定 Boneseed 续训第4000轮 checkpoint 的 `pi` 及其全部 RMS：
  `/mnt/projects/ll/tianzizh/runs/gmt_bones_filtered_from26k_20k_20261106_02/ckpt_004000.pt`，
  SHA256 `0b6df18de18306204972530d8d1b6ffff489e9c06066dc1dacabcc0526e3ab31`。
  它的训练来源是原26k GMT；此前1508407验证用的是原26k权重。
  加载器不实例化 WM、
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

当前观测 schema 是 `hoi_oracle_hw3_torso_fps128_v2`，checkpoint schema 为
`hoi_residual_ppo_teacher_v2`。旧 residual checkpoint 因观测与网络不一致会被拒绝。
残差 actor 与 critic 均使用 `3647D oracle + 29D frozen base action`，归一化后再拼接
各自的 64D shape embedding，MLP 输入共 3740D。

| Raw oracle 子块 | 维度 | 定义 |
|---|---:|---|
| Robot privileged | 3371 | 原 GMT 输入逐字段保留，仍含其原本的 1/13 帧参考 |
| Object history | 90 | 10×9；每个历史时刻物体相对该时刻实际 torso 的位置3、heading下旋转6 |
| Object future reference | 63 | 7×9；未来参考位置3/旋转6，统一锚定当前实际 torso |
| Explicit tracking errors | 105 | 同一7时刻各15D：位置3、旋转6、线速度差3、角速度差3 |
| Object oracle velocity | 6 | 当前物体世界线/角速度转到当前 torso heading；保留teacher特权 |
| Object physical context | 12 | 实际质量、COM、惯量6分量、物体与地面摩擦 |

物体参考偏移严格采用 v1.1 的 `[1,2,5,7,12,18,25]`，即
`[.02,.04,.10,.14,.24,.36,.50] s`。位置锚点为当前 `torso_link` link origin，
旋转基底为其 yaw heading（不是完整 torso roll/pitch 旋转）；旋转6D使用X/Z列。
torso FK来自当前冻结hw3机器人，包含waist joints；没有导入旧机器人配置。
历史启动时重复首帧，reset只重复该world的新初始观测；历史各tick保留自身采集坐标系，
不重投影到当前torso。未来参考越界重复当前clip末帧，不跨clip、不增加validity mask。

FPS128复用已绑定v1.1 geometry cache的精确float32点，保持物体link局部米制坐标和实际尺寸；
验证cache SHA、collision mesh、scale、sampling contract与point hash后，导出独立shape bank。
该采样协议为4096个暴露碰撞并集表面候选点→确定性FPS128，剔除内部重叠表面。
训练只读新bank，不依赖旧HOI代码。actor/critic各自拥有可训练PointNet：
逐点`3→64→128→128`（ELU），max pooling，ELU+`128→64`。
shape点与embedding不进入RunningNorm；每个PPO minibatch从原始点重新计算embedding，
梯度同时更新两个encoder。asset one-hot和包围盒已移出网络输入；asset ID仅用于物理与数据索引。
每个训练rank只负责一种资产，因此rollout只保存一份原始`[128,3]`点云；
每个minibatch中两个encoder各算一次，再广播64D embedding，并将所有样本的梯度汇总回encoder。
这与逐样本重复编码在数学上等价；FP32归约顺序可有微小数值差异，不缓存或detach学习到的embedding。
混合资产batch仍支持逐样本点云。训练在stack完成后释放逐步副本，下一次RMS更新后释放旧输入，
以免PyTorch的无用rollout引用挤占Warp物理所需显存。
用于reward的1024点、奖励公式和物理配置保持原样。

两套独立 MLP 的隐藏层均为 `[512, 512, 256, 128]`、SiLU、最后隐藏层 LayerNorm。
残差最后线性层全零初始化，因此 deterministic residual 初始为零；可训练 Gaussian
初始标准差为 `0.1`。这保留 ResMimic 的 additive residual 设计，但用明确的 oracle
schema 与 v1.1 物体 history/future/shape 接口。上游发布版的 object 7D 与 future
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
物体9D历史统计在10个槽间共享，只以当前tick累计一次；63D目标按future slot统计。
统计更新遵守PPO的rollout边界，不在采集中改变动作分布的预处理。
真正 failure 的 bootstrap 为零；time limit/motion end 从 **reset 前最后观测** 的 value
bootstrap；两种边界都截断 GAE trace，不能接入下一个 episode 的 value/advantage。

checkpoint 另绑定 `contract.py` 中 14 个残差学习实现文件的 SHA，包括训练时序、
奖励、观测、资产处理与随机化；也绑定shape bank SHA与观测协议。
评估拒绝实现、manifest、资产catalog或FPS128 bank身份不符的权重。

完整训练由 8 个（或 8 的整数倍）distributed ranks 共同训练**同一套策略**：每个 rank 固定一种
exact asset geometry，并从该资产的 train clips 采样，保持单 worker 中所有 MuJoCo Warp
worlds 拓扑一致。训练入口广播初始参数与 buffers，随后跨 ranks 汇总 normalizer moments、
advantages 与 gradients；梯度平均后再 clip。各 rank 的 rollout 长度、环境数、epochs
和 minibatch 数必须一致。当前默认launcher采用一张L40上的8个独立物理进程，
每rank512环境，共4096环境。显式`--shared-gpu`要求调度器只暴露一张卡；所有rank使用cuda:0，
Gloo通过独立CPU缓冲同步broadcast、RMS与梯度，完成后同步拷回CUDA，规避NCCL重复设备限制。
仍按8个等量资产rank平均梯度，不改变PPO超参数、物理或参考数据量。
短验收结束时比较所有rank完整policy/RMS的SHA，必须完全一致。
多GPU模式省略`--shared-gpu`，按`LOCAL_RANK`使用不同CUDA设备和NCCL；
launcher可用`HOI_SHARED_GPU=0`配合显式8张GPU分配选择该模式。
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

本次观测v2已通过本地42项专项CPU测试（8.828 s），包括torso/七参考时刻精确对齐、
历史reset隔离、共享历史RMS、FPS128身份校验、两套shape encoder真实PPO更新，
固定点云编码的输出/梯度/Adam近似等价，以及两进程Gloo参数/梯度/Adam同步。
集群CPU作业 `1509569` 同样通过42项测试（6.662 s）。当前原生验收绑定冻结
`source_smoke08`，归档SHA256为
`d693476d2d4cb0112dec5bd4b8eaa73d1af5b4036249b6319f7edab36262b9b3`；
93个源码文件、14个learning模块、24个GMT共享模块及75个机器人文件均已核对。

观测v2的L40完整验收已通过：Slurm `1509570`，`ll-l40-2`，
NVIDIA L40 / driver `535.216.01`；美东2026-11-06 `19:14:14–19:36:51`，
22分37秒，退出码 `0:0`。验收包括：

- 八种资产分别使用8个环境、32-step rollout、2轮PPO。
- 单张L40上的8 ranks各用8个环境、24-step rollout、2轮联合PPO。
- 同一单卡上8 ranks各用512个环境，完整覆盖2651条train动作，
  24-step rollout、3轮PPO，共294912 transitions；第三轮覆盖完整RMS更新的容量峰值。
- 两阶段联合PPO均验证八个rank最终policy/RMS SHA完全一致，两套shape encoder及
  actor/critic真实更新，冻结GMT权重/RMS不变，DR启用，history/reset隔离通过。

GPU显存5秒采样峰值为 `44243 / 46068 MiB`（约43.2 / 45.0 GiB）；
这是实际全数据容量运行的观测值，不能用理论激活节省替代。
train-only评估完整执行一条220-step（4.4秒）轨迹，零optimizer/RMS更新、
无中途reset，未访问Protected动作张量。该条轨迹未触发failure，但这只是实现验收，
不能据此声称teacher收敛或具有总体HOI任务成功率。

当前完整报告与回执为本地ignored文件：
`provenance/smoke_1509570_full.json`、`provenance/smoke_1509570_train_eval.json`、
`provenance/smoke_1509570_receipt.json`、`provenance/smoke_1509570_formal_gate.json`。
集群原始输出保留于 `/mnt/projects/ll/tianzizh/hoi_residual_ppo_20261106_01/smoke_1509570`。
下方1508407是旧观测v1的历史记录，不替代本次v2验收。

先准备新的 manifest/output，使用已有冻结 asset bundle 和 GMT checkpoint；输入均只读。
FPS bank一次性准备命令（输出路径须尚不存在）：

```bash
python -m hoi_residual.shape --source-bundle /path/legacy_bound_bundle.json \
  --catalog /path/current/assets.json --output /path/new/shape_bank_v2.json
```

smoke 示例（只在 Slurm 分配的 GPU 环境内运行）：

```bash
python -m hoi_residual.train \
  --manifest /absolute/new/manifest.json \
  --asset-bundle /absolute/frozen/assets.json \
  --shape-bank /absolute/frozen/shape_bank_v2.json \
  --gmt-checkpoint /absolute/frozen/ckpt_004000.pt --gmt-iteration 4000 \
  --output /absolute/new/hoi_teacher_smoke \
  --device cuda:0 --num-envs 2 --rollout-steps 24 --iterations 2 --smoke
```

完整任务入口相同，去掉 `--smoke`，按启动器分配 8 ranks；不要在 GRASP login node
执行 GPU workload，不改运行中的 GMT job、冻结 source 或共享 Python runtime。
例如，已经获得单L40分配时可用 `torchrun --standalone --nproc_per_node=8
-m hoi_residual.train --shared-gpu ...`。`--asset INDEX` 仅用于明确指定的单资产实验；其索引采用
canonical alphabetically sorted asset names，旧 bundle 的 `source_asset_id` 只作来源记录。
默认 `--num-envs 512` 是每个 rank 的环境数；默认 `--iterations 100000`、
`--save-every 100`、`--gmt-iteration 4000`。CPU 功能检查：

```bash
python -m unittest discover -s tests -p 'test_hoi_residual_*.py' -v
```

PPO 专项 CPU 测试包括真实 GMT policy/RMS 加载与冻结、0 WM 构造、零残差初始化、
未裁剪 action likelihood、timeout/failure GAE、真实 actor/critic optimizer 更新、
normalizer 冻结，以及 2-rank Gloo 的参数/梯度/统计同步。多进程测试需要本机 loopback。
旧观测v1在2026-11-06使用本地 `.venv-hoi-tests` 时，PPO专项7项通过（15.312 s），包括真实
两进程 Gloo；Windows 沙盒禁止 loopback 的第一次运行失败，经允许本机通信后通过。
数据/观测/奖励 15 项与完整时长评估协议 2 项也通过，Warp 前向编译配置 3 项通过，
当时新增CPU测试共27项；当前v2对应上文42项测试。
CPU 测试、native physical smoke、PPO smoke 与正式训练收敛是不同证据；只跑通 smoke
不能声称得到高性能 HOI teacher。实际 GPU 作业、checkpoint SHA、输出路径和通过项
以对应 run receipts 为准。

原始观测v1的真实 GPU smoke 已通过（不作为本次v2观测改动的验收）：Slurm `1508407`，`ll-l40-0`，
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
执行的冻结代码是同级 `source_smoke05`；其12个learning模块SHA属于旧commit
`5001210307bdd9d72397f1e38aee7d8d6d466d83`，不与当前14模块实现混用。

正式联合训练作业 `1509634` 已在美东2026-11-06 `19:36:56` 于 `ll-l40-2` 启动。
Slurm的 `afterok:1509570` 及独立guard均已通过；guard重新验证完整验收、源码与输入SHA，
随后从新run开始，不恢复smoke或容量测试的optimizer状态。
正式使用单张L40、8 ranks、每rank512个环境，目标100000轮，保存间隔100轮，
冻结GMT为用户指定的Boneseed `ckpt_004000.pt`，输入shape bank及train/Protected划分保持不变。
运行目录为 `/mnt/projects/ll/tianzizh/hoi_residual_ppo_20261106_01/train_1509634`，
日志为同根目录 `formal-source08-1509634.log`；进度以各rank的 `metrics.jsonl` 为准。
调度资源为 `liu-compute / liu-account / ll-med`，1 L40、24 CPU、96 GiB主机内存、72小时时限。

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
  --gmt-checkpoint /path/ckpt_004000.pt --gmt-iteration 4000 --manifest /path/manifest.json \
  --asset-bundle /path/assets.json --shape-bank /path/shape_bank_v2.json \
  --partition test --output /path/new_eval
```

省略 `--asset` 时要求 checkpoint 覆盖全部八资产，再分八个进程顺序运行。
每条动作执行完整时长，failure 不触发 reset；报告 return、robot failure、object surface
failure、物体姿态及机器人误差，不把姿态误差阈值称作抓取成功。
实现烟测只允许 `--partition train --smoke-eval`；Protected 集不用于调通代码。
原始 rollout 未导出，报告明确保存的是逐动作数值汇总。

### v1.1 奖励对照

`configs/hoi_residual_v11_reward.json` 选择 `hoi/v1.1-observation-matching@66d9184` 的完整
14 项训练奖励。原来的训练命令默认使用 ResMimic 奖励。对照实验沿用 GMT、残差观测、
PPO 参数、域随机、采样算法和 30 cm 物体表面误差早停。采样器的回报归一化上限按
奖励定义从每步 0.384 调为 0.15。按 clip 的 SHA、资产名称和帧数匹配 v1.1 的训练接触标签；
实际接触来自 4 个 5 ms 物理子步的左右手–物体法向冲量。

以下两个 Slurm 命令使用独立实验根目录，`ROOT/input` 指向与基线相同的输入，
`SOURCE` 是这次对照的冻结源码。先运行 smoke，再提交正式训练。

```bash
CFG="$SOURCE/configs/hoi_residual_v11_reward.json"
BUNDLE=/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_15/results/contact5_ultrafoot_validation_20261023_01/bundle_train/bundle.json
sbatch "$SOURCE/cluster/hoi_residual_smoke.sbatch.sh" "$ROOT" "$SOURCE" all "$CFG" "$BUNDLE"
sbatch "$SOURCE/cluster/hoi_residual_train.sbatch.sh" "$ROOT" "$SOURCE" 100000 "$CFG" "$BUNDLE"
```

每个 rank 的 `metrics.jsonl` 保留 14 项已乘权重和控制步长的逐项均值。
`reward_terms_by_asset` 对应该 rank 的资产，`reward_terms` 是全局均值，各项之和等于
`reward_mean`。正式对比使用相同的 HOI80 通过率、物体误差和固定评测奖励。

2026-11-07，v1.1 对照训练 `1510679` 的第 1300 轮 checkpoint 完成了固定 HOI80
评测（L40 作业 `1511413`，8 种资产、80 条轨迹、23,718 个控制步）。人体通过 79/80，
物体通过 3/80，联合通过 3/80。三条联合通过的轨迹均来自原尺寸 largebox。
物体表面误差按轨迹等权平均为 40.56 cm。[W&B 评测记录](https://wandb.ai/tianzizh-university-of-pennsylvania/hoi-residual-ppo-teacher/runs/hoi-1510679)

### 基线 checkpoint 恢复与碰撞保护

基线作业 1509634 在第 688 轮采样时触发原生 `EPA_HORIZON` 标志。
恢复命令可用 `HOI_RESUME_CHECKPOINT` 和 `HOI_RESUME_SHA256` 指定最后一个完整的
PPO checkpoint。恢复模式默认将每对凸体的 EPA 边容量从 24 提高到 64，并在首次物理步
检查实际分配的维度。其余接触与约束缓冲区沿用 512/2048。再次溢出时记录求解器标志和
受影响环境，丢弃未完成的 rollout。

恢复作业每 10 轮原子更新 `recovery_latest.pt`，每 100 轮保留编号归档。
actor、critic、shape encoder、归一化统计、Adam 和 PPO 更新计数从 checkpoint 载入。
旧 checkpoint 未保存仿真状态、下一次归一化所需的 rollout、采样课程和各 rank 的随机流，
新进程会重新初始化这些状态。两个作业的同轮数记录分别保存在各自的目录和 W&B run。

2026-11-07，原版 ResMimic 奖励的恢复作业 `1510748` 主动停训，8 个 rank 最后一次完整记录均为
第 1779 轮。最后的编号归档为 `checkpoint_001700.pt`。第 1770 轮的滚动恢复文件保存在
`/mnt/projects/ll/tianzizh/hoi_residual_resmimic_resume600_20261107_01/train_1510748/recovery_latest.pt`，
SHA-256 为 `58ef115668b3be3757a56330b61c3a4f6cf418b29426a243acd7bca1f7a10e26`。[W&B 训练记录](https://wandb.ai/tianzizh-university-of-pennsylvania/hoi-residual-ppo-teacher/runs/hoi-1510748)

设计参考：[ResMimic repository](https://github.com/amazon-far/ResMimic)。
本新增实现没有复制其 simulator/PD 源码。若后续直接搬运上游文件，应保留根 Apache-2.0
`LICENSE` / `NOTICE`，以及 `rsl_rl`、`legged_gym` 各自 BSD-3-Clause 文件头与许可证。
