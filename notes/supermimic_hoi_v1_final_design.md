# SuperMimic + HOI v1 最终设计

现行开发基线：`codex/hoi-v1-no-quality-gate`，2026-10-25整理。本文保留v1增量架构与数据契约，并统一为当前已实现、已验收的新版：完整手/脚碰撞、身体自碰撞、五通道learned contact、固定500 iteration WM-only后交替更新。WM质量比较只作诊断；native、数据身份和有限性硬检查保留。

当前实验为全部from-scratch，最后完整8500pt已完成canonical HOI80及同固定20条视频验收；训练因24h作业时限结束，没有完成20k，不宣称收敛。冻结权重、源码、bundle、物理、数值与媒体证据见[8500pt最终验收](hoi_v1_final8500_eval_handoff_20261025.md)。设计与本次记录不自动授权续训或新实验。

数据依据为[ULTRA数据与时间基准交接](ULTRA_DATASET_AND_TIMEBASE_HANDOFF_20261023.zh-CN.md)。发布数据真实生成时钟尚未证实；本版使用canonical 60Hz解释与50Hz控制。网络尺寸、物体分支、奖励权重和阈值属于本项目设计，不冒充论文原配置或已验证最优值。外部仓库与操作入口见[文档索引](README.md)。

## v1.1 observation matching

2026-10-25新增本地实验分支 `hoi/v1.1-observation-matching`。以下是独立新实验合同，
不改写8500pt的v1结果。最初授权范围为实现、CPU回归和可导性检查；后续用户明确授权必要GPU验证后启动新20k训练，
通过独立源码/输出及同一GPU allocation内的验证→训练串行流程执行。随后用户授权整理代码并commit/push到同名分支；数据、权重和运行结果不提交。
显式配置 `configs/hoi_v1_1_observation_matching.json` 选择 `observation_profile=v1_1_observation_matching`、
`domain_randomization_profile=v1_1_mass_object`。省略配置仍使用原v1 profile。

### Actor观测与保留的差异

| 输入 | v1.1合同 |
| --- | --- |
| 本体历史 | 10×93，最旧到当前；pelvis局部gyro3、q-default-PD零位偏置29、qd29、上次raw action29、投影重力3 |
| 人体未来 | 7×64：`rotation_XZ(R_torso_actual(t)^T R_torso_ref(t+k))`6、现有派生参考qd×0.1的29维、参考q29 |
| 物体历史 | 10×9，当前torso原点/heading中的位置3和物体旋转XZ6；删除当前物体reference误差后9D |
| 物体未来 | 7×9，同一当前实际/imagined torso-heading中的reference位置3、旋转XZ6；无速度/期望接触/接触有效性输入 |
| 未来偏移 | 人体/物体共用control ticks `[1,2,5,7,12,18,25]`，即0.02/0.04/0.10/0.14/0.24/0.36/0.50秒 |
| 历史/末尾 | 启动重复首帧；未来越界重复末帧，不读下一clip；不添加mask，固定时间由MLP输入槽位表示 |
| 旋转编码 | 拼接旋转后X轴3维和Z轴3维；只用于新actor，原WM/critic旋转编码保持 |
| 几何 | 原128×3物体局部米制FPS不变，原actor PointNet/shape64不变 |

这是V5可见字段/坐标语义匹配加用户指定例外，不是V5数值接口/网络/训练方法的逐位复现。
V5本体IMU仍读pelvis完整局部坐标系，人体未来读完整torso相对旋转，只有物体位姿使用torso heading。
现有50Hz参考重采样、姿态派生joint qd、qd输入缩放0.1与FK顺序保持；不改原数据或reward/reference velocity。
关节零位沿用原PD偏置/传感器校准约定，不把V5独立encoder offset符号直接移植。
传感噪声保持原幅度，仅重排93维并删除已移除的6D旋转噪声；干净评估不加观测/探索噪声。

本体/人体目标使用新建统计的RMS和[-5,5]裁剪，稳定项为1e-5；不复制或冻结一个未训练的V5 RMS。
按用户后续明确要求，物体历史和未来参考也做RunningNorm：独立9D与63D RMS，均不裁剪。
历史10帧共用9维统计，真实采集每次只用当前帧更新一次；未来63维保留各时间槽位/特征的独立统计。
两套RMS使用`(x-mean)/sqrt(var+0.01^2)`，分母下限0.01，max_count=2e8，均从真实采集新建统计；
imagination/eval只读取统计并保留输入Jacobian。原始replay/评估输入仍按米制/XZ旋转保存，网络内部标准化。
MLP融合保留：human448→原hidden→32、object63→原object_hidden→32，
decoder输入为`32+930+90+32+64=1148`，hidden2048/1024/512/256与29维动作不变。
没有Transformer、flow matching或额外物体速度/真实接触/质量actor输入。

`hoi/v1/observations.py`是actor专用适配层；collect、imagination、冻结evaluate使用同一合同，
每个imagined tick用当步预测torso/物体重新计算，保持状态/动作Jacobian。
原MDP、MotionLib.goal、190D object goal继续供旧critic/reward读取，不被actor配置全局替换。
按用户后续授权，WM输入224→225、critic输入1214→1215，只在原输入末尾追加1D torso质量上下文；
WM输出82、全部hidden、五通道contact监督、奖励/终止/PD/手脚物理与训练配比保持。
新feature/checkpoint记录完整actor和DR合同，旧8500不能直接resume或legacy迁移为新actor；需要from-scratch或相同新合同resume。

### DR配置

| 参数 | 范围和抽样时机 |
| --- | --- |
| torso质量 | nominal±3kg，每个环境初始化抽一次；nominal必须>3kg |
| 物体质量 | 每episode uniform `[max(nominal-1.5kg,0.1×nominal), nominal+1.5kg]`；不先生成负质量再clamp |
| 物体COM | 相对nominal各轴±1cm，每episode；保持object link-local定义 |
| 物体摩擦 | uniform[0.4,0.8]，每episode |
| 机器人摩擦/torso COM/关节零位/reset扰动 | 原范围与抽样时机保持 |
| 惯量 | 只随对应质量同比缩放原COM惯量，保持几何及回转半径；无独立惯量随机化 |
| 地面摩擦/PD/重力/延迟/推搡/物体初始位姿 | 原固定profile；不增加这些随机化 |

物体DR从nominal重抽，subset reset不改变其余环境；写入后读回质量/COM/惯量/摩擦，实际12D xiO随transition保存。
Robot mass只改torso，其他link质量/惯量不变。原33D robot xi保留原语义，torso质量通过独立上下文提供给WM/critic。
正常HOI80用eval_mode，新增DR与原随机化均关闭，读取nominal物性。

### Torso质量训练上下文

2026-10-25按用户后续要求补齐，schema为`supermimic_torso_mass_context_1`。
native在DR之前缓存`torso_mass_nominal`，每次snapshot和pre-action transition读取生效的`torso_link`质量为`torso_mass`；
两个字段均以kg、shape[B,1]保存，终止后的reset不能覆盖上一条transition的质量。
网络读取`(torso_mass−torso_mass_nominal)/3kg`这一维，固定缩放、不RMS、不裁剪；名义评估为0。

| 链路 | 合同 |
| --- | --- |
| Worker/Replay | 原始实际/名义质量随环境合并，保存到ring buffer、WM窗口、imagination start/history；新DR缺失/非正/非有限质量立即报错 |
| WM | 原224维后追加1维，训练、autoregressive rollout、teacher-forced holdout、执行动作评估诊断都条件化质量；原raw normalizer不扩维 |
| Imagination | 起点质量全程固定，预测状态/动作及其Jacobian继续递归；不从reference推断质量 |
| Critic | 原1214维后追加相同1维，bootstrap与全部H+1状态/value minibatch使用对应起点质量；原914/236 normalizer保持 |
| Actor | 不消费质量，部署输入保持93/448/9/63；现行追加独立物体RMS，Policy.forward、原观测/FK字段和MLP结构不变 |
| 评估 | nominal质量上下文为0；保存T条实际/名义质量及编码值，校验snapshot/transition/after一致；0optimizer/0normalizer更新 |
| Checkpoint | 现行feature版本`supermimic_hoi_v1_1_observation_matching_3`，保存质量/RMS合同及两套物体RMS全部统计；恢复前拒绝raw-pose旧v1.1、缺质量合同、错误DR或旧输入维度 |

旧v1默认仍是224D WM/1214D critic，旧8500按原合同读取。先前未加质量的v1.1+质量DR checkpoint不能直接resume为新版；
checkpoint仍不保存simulator/replay/RNG，续训须fresh真实prefill，新采样的质量随新transition记录。
定期评估入口已从硬编码旧feature schema改为版本化校验，兼容新checkpoint合同；本次没有提交评估作业。

新DR的native setter路径尚未在Linux GPU验证。旧native evidence和CPU mock均不认证新DR；
训练入口另检查native evidence的`domain_randomization`合同及实际property-readback、subset-reset隔离、nominal-eval验证。
这些原生检查未完成时，新配置可CPU预检，但不能启动正式训练。

观测改造阶段CPU证据在ignored `tmp/hoi_v1_1_observation_matching_cpu_20261025_02/`（01保留前轮结果）：
全量276项检查完成、275通过、1项因可选pxr不可用跳过；原HOI边界6项通过。
新增测试用明确的合成waist chain核对命名torso FK/完整FK/解析旋转，验证七槽位、XZ编码、历史与末尾重复、
参考qd缩放、旧critic输入不变、checkpoint合同隔离、actor BPTT与固定噪声有限差分，以及DR采样/写入mock/读回/隔离。
另验证冻结评估完整T/T+1、新维数落盘且无权重/normalizer更新；新配置preflight不启动workers并明确标记DR原生门未通过。
额外CPU检查包括batch2/H32的合成小网络rollout反传，以及默认完整规模actor（8883741参数）的batch2前向/反向，梯度有限。
该阶段的13项AST对照确认当时WM、critic、shape encoder、原kin.forward、原native.step/_advance、奖励终止和旧object_goal未改；
本次追加质量后WM/critic/native.step已有上述明确变更，不沿用这些旧AST断言作为本次证据。

质量链路最终CPU证据在ignored `tmp/hoi_v1_1_mass_context_cpu_20261025_01/validation.json`：
全量286项检查、285通过、1项可选pxr跳过；原HOI边界6项通过。
新增质量缺失/数值/读回、pre-reset记录、replay回绕、WM fit/1步和32步holdout、固定imagination质量、
critic bootstrap/value minibatch、checkpoint文件读回/拒绝旧合同、定期评估读取和冻结nominal评估诊断检查。
batch2/H32反传及固定噪声方向有限差分通过（autograd=0.05280915，FD=0.05272627）；
float64直接动作/位置Jacobian有限差分通过，默认完整8883741参数actor前向/反向有限。
Policy/ShapeEncoder AST及observations.py字节SHA与本次质量改造前一致；WM/critic在actor反传中无参数梯度。
这些是CPU软件和梯度合同证据，不是原生物理、GPU资源、训练稳定性或HOI成功证据。

### v1.1验证与训练提交

2026-10-25用户授权必要验证后从零训练20k，并要求超过24小时的作业时限。
已读取真实queue、QoS/association和partition：`liu-compute` MaxTime=14天，`ll-med`无额外MaxWall，账户允许该QoS。
新训练申请96小时，20k结束即退出；原固定500轮WM-only、4096环境/H32/完整batch及优化配置保持。
用户随后明确授权加物体独立RMS并替换排队作业；CPU/可导性检查通过后，仅取消仍PENDING的1467430/1467434，
已记录CANCELLED回执，旧raw-pose源码和数据完整保留。下表为当前RMS版本作业。
用户又要求验证和训练使用同一GPU任务以避免二次排队；已在两项作业仍PENDING时取消1467536/1467537，
替换为单GPU作业1467570。RMS冻结源码及训练参数不变，只新增独立operations启动包装。

| 项目 | 实际提交合同 |
| --- | --- |
| 冻结源码 | `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_1_object_rms_20261025_01`，不覆盖任何历史快照 |
| Source content SHA | `ade8dafe66473b0ac15ae4c68a2ced8656185ce5d90fba9f6e0faf5ca13edac2`；183个文件逐项核对 |
| GPU验证 | 合并Slurm `1467570`的第一阶段，L40×1/24CPU/96GiB，单阶段7200秒超时保护；当前PENDING，尚无GPU验证通过结论 |
| 验证内容 | 八资产×训练DR/名义eval；真实reset隔离、Newton与生效MuJoCo-Warp质量/COM/惯量/摩擦读回；4096环境/replay1000/64prefill、2048×8 WM、4096×32 actor和全部critic minibatch反传；0真实optimizer step |
| 正式训练 | 同一Slurm `1467570`第二阶段，无dependency/二次sbatch；总时限96小时覆盖验证和训练，20k from-scratch；当前PENDING，没有开始训练 |
| 训练输出 | 新源码的`results/hoi_v1_1_object_rms_scratch_20k`；不读取8500初始化权重 |
| 运行环境 | 原已验收read-only kitless-cu128和IsaacLab；OMP/MKL=3，两个allocator变量均为`expandable_segments:True` |
| 数据/碰撞 | 原source15验收的完整train2651/HOI80 bundle；原八资产碰撞证据只证明原配置，新DR须额外通过新作业检查 |
| W&B | 训练实际启动时创建独立私有run，project=`SuperMimic_HOI_v1_20261023`，不复用8500 run；仅curated配置/指标/汇总 |
| 定期评估 | 每5000轮完整HOI80，独立GPU作业；已测8500完整80耗时42m18s，因此本轮独立评估启动器预留2小时，线程/allocator同上 |

GPU验证脚本和训练入口均检查冻结stage/hash，训练还要核验新DR证据与production resource报告passed、source/config一致和参数未更新。
合并启动器使用`set -e`和7200秒validation timeout：验证失败/超时立即退出，不调用训练；不把原native evidence改成声称已验证新DR。
额外检查native/capacity证据属于当前SLURM_JOB_ID，前后GPU/节点/allocation一致，probe为0真实optimizer step；检查失败也不进入训练。
验证和训练分属前后两个Python进程，验证进程及其probe模型退出后再创建from-scratch训练进程，保持同一GPU allocation，不复用probe权重或RMS。
WANDB_SYNC只在训练阶段为1，验证为0；训练仍按原500轮WM-only和固定更新日程运行。
新原生校验是显式validation worker开关，普通训练collector不执行probe/reset；只为核对惯量比例保存DR前机器人质量/惯量副本，不改变抽样或物理参数。

合并操作启动器单独冻结于`/mnt/projects/ll/tianzizh/supermimic_hoi_v1_1_object_rms_combined_20261025_01`，
保持183文件源码快照不变；五个辅助文件和operations manifest均逐项校验，SBATCH脚本bash-n通过。
原独立启动器`/mnt/projects/ll/tianzizh/supermimic_hoi_v1_1_object_rms_operations_20261025_01`及其取消作业回执保留。
旧raw-pose排队版回执/287项CPU证据保留在ignored `tmp/hoi_v1_1_launch_20261025_01/`。
当前RMS版回执、stage/operations manifest和证据见ignored `tmp/hoi_v1_1_object_rms_20261025_01/validation.json`：
293项CPU回归、292通过、1项可选pxr跳过，原HOI边界6项通过；验证独立RMS、历史计数、无裁剪与0.01分母下限、
真实采集更新、imagination/eval冻结、保存/恢复统计和拒绝旧raw schema。
batch2/H32使用非空真实RMS统计反传与固定噪声有限差分通过（autograd=0.05541336，FD=0.05531311）；
float64动作/状态直接Jacobian和完整8883741参数actor前后向有限，网络参数量与输入维数不变，仅增加146个统计标量buffer。
Scope审计确认WM/critic、ShapeEncoder、Policy.forward、Learner.action/imagine、原观测采样/坐标变换及物理/奖励/参考代码未变。
真实完整bundle的CPU预检确认新9D/63D无裁剪合同、source/config一致、原native门通过；新DR门保持未通过，等GPU作业验证。
合并流程另外在真实Linux runtime执行10项CPU检查：验证失败/证据缺失或篡改/不同job或GPU/非零probe更新均阻断训练；
实际POSIX shell的成功顺序、phase环境与失败早退通过。记录在ignored `tmp/hoi_v1_1_object_rms_combined_20261025_01/`。
提交整理将这些可复用Python辅助代码归入`cluster/hoi_v1_1/`，并提供tracked combined/两小时evaluate启动器。
独立stage/operations manifest仍为运行时输入；日志、证据、checkpoint、结果与私有回执保持ignored。
整理后的入口供下一次独立staging使用；1467570的原冻结源码及operations不改动、不重提，旧CPU/native证据不冒充整理后新快照的GPU验证。
提交前重新运行完整CPU回归：304项检查、303通过、1项可选pxr跳过；原HOI边界6项通过，5份v1.1 shell脚本bash-n通过。
新增11项流程测试包含manifest/证据硬错误、零optimizer更新及真实Git Bash中的成功顺序与失败早退；仅使用stub命令，不启动GPU验证或训练。
本轮CPU日志及暂存审计留在ignored `tmp/hoi_v1_1_commit_20261025_01/`。
以上明确区分CPU已完成和合并GPU作业排队；PENDING不代表RUNNING、通过、稳定或收敛。

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

设计基点为原motion-only/zero-shot实现，v1设计创建基点为 `2318dfd604d0a254686760e15c034231a94aa0aa`。下表列出原网络与当前HOI增量；原始fork分支及既有zero-shot入口保留，当前开发分支为 `codex/hoi-v1-no-quality-gate`。

| 组件 | 原motion-only结构 | 当前HOI处理 |
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

这些 `cases`/entries 绑定 motion 路径、motion SHA256、source_group 和 object/asset。目录中的全部 4066 个文件不是训练集；训练范围仍是 selection 指定的 2651 motions / 381 groups。实际适配已读取selection和eval80并逐项核对，eval80实读SHA与canonical `459a3925faf0f66bee258ce98635c67802ddda00d47d007ecdeb45035f576002` 相同。完整训练2651条逐SHA/schema/时间审计已通过：908982个源状态生成756524个50Hz状态、753873个完整控制间隔；当前新版验收入口见[8500pt最终报告](hoi_v1_final8500_eval_handoff_20261025.md)。

新项目建立自己的独立 manifest，保存 source selection 路径与 SHA、逐 motion SHA、case/source-group、asset/scale/transform 绑定。先与第 3.2 节的本地四分区 manifest 和 canonical 名单逐项核对；不得仅因条数相同就认定两份清单等价。集群eval80实读身份核对已完成；后续加载仍要按登记SHA验证，不能仅凭名单路径或数量沿用身份。保留既有 test40 名单及其所有同源 variants 隔离。

交接中重算的样例 SHA 为：

```text
文件：sub10_largebox_000_075_086_074_100_100_100.pt
大小：671819 bytes
SHA256: 1a0bac419adab2bf6d7f5d10d480e40d0f3c39a74605a7063c7f48b0f99bf97c
```

该核查确认了清单路径存在，并抽查上述文件字节；**没有重新扫描全部4066个文件的SHA**。本项目随后逐文件解码并审计所选2651条训练动作，没有把全部4066文件当成训练集。新项目实际加载所选 motion 时核对其已登记的 SHA/schema。源目录只读使用；新 manifest、缓存、派生 reference 和输出写入独立项目目录，保留源 `.pt` 字节。`[T,630]` 源 reference 与 50 Hz `rollout.pt`、checkpoint 必须按 schema 区分，不凭 `.pt` 后缀混用。

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

首条训练动作的独立source39/FK只读诊断覆盖266个源状态和221个采样状态：原ULTRA link FK相对source39的最大位置差0.736微米；采样后独立body pose插值与q/root插值再FK存在最大0.633mm的非线性差异，未重新对齐或重算reference来消除。该单clip证据支持link-origin pose解释，不能证明全部源velocity参考点或真实生成时钟。

连续时间重采样、reference velocity 派生、source/API 的 COM→link-origin 转换是三项独立操作。只有确认源 velocity 参考点后才能做可解释的逐项差异审计；未知项标为 unknown。复核 ProtoMotions 历史结果时必须沿用它原来的 velocity mode，不用本版新派生速度改写其结果。

### 3.4 Canonical 60 Hz 解释 + 50 Hz 控制

明确区分三种时钟：

| 量 | ProtoMotions 已有证据 | SuperMimic v1 约定 |
| --- | --- | --- |
| Source timebase | consumer 按 60 fps 解释；生成间隔未证实 | 沿用 `canonical_60hz`，标为协议假定 |
| Control dt | 已有实现/运行验证为 0.02s，即 50 Hz | 保留原 `CONTROL_DT=0.02`，不因交接改变 |
| Physics | 0.005s × decimation4 | 保留本仓库 `SimulationCfg(dt=0.02)` + Newton `num_substeps=4`；本版native recorder逐个5ms子步积分pair impulse，四子步及20ms控制区间已独立验证 |

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
| 集群v4数据、冻结训练/验证路径及字节身份 | 数据交接与本项目后续完整2651训练数据审计、canonical80加载验收；原4066目录数为交接时观察 |
| ProtoMotions 按 60 Hz reference 解释，并以 50 Hz 控制运行 | 交接的源码及已有运行证据 |
| Robot/object reference以连续时间同轴采样 | 本项目适配器、完整缓存CPU审计与8500pt完整80验收 |
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

### 3.6 当前完整手脚碰撞profile

当前HOI训练强制hand collision ON、body self-collision ON，保留引擎相邻/焊接link过滤。双手使用原橡胶手STL、已验证安装变换和不可变hand overlay；双脚profile为 `ultra_g1_stl_coacd5v16_v1`，foot manifest schema为 `hoi_v1_ultra_g1_foot_mesh_overlay_v1`。

ULTRA HOI训练实际从 `g1_teacher.yaml` / `humanoid_g1.py` 加载 `g1/g1_29dof.urdf`，两脚collision为完整 `left_ankle_roll_link.STL` / `right_ankle_roll_link.STL`，mount xyz/rpy均0、scale1。当前overlay绑定同SHA原STL和原变换，只关闭每脚旧4个球碰撞，追加CoACD最多5hull/每hull最多16vertex；已验收cook实际为每脚5hull。原质量、COM、惯量、visual、PD和其他碰撞均保留。

作者PhysX使用VHACD5/16/resolution60000；本版Newton/MuJoCo使用CoACD，`equivalent_to_physx_vhacd=false`，不把60000映射为CoACD参数，不声称凸块、接触或物理逐位相同。训练与评估要求bundle显式绑定foot manifest、cooked USD、原hand receipt及实际native compiled audit；只改默认资产路径不能代替此身份。此profile不自动改写原zero-shot或motion-only入口的裸机器人。

HOI runtime使用独立容量 `nconmax=512/njmax=2048`，原motion-only/zero-shot的128/500保留。碰撞/约束overflow立即失败且不写入replay，不能把截断的pair数据当合法负例。原生CUDA graph关闭，使用已准备runtime支持的配置，不修改共享驱动或安装。

新原生门覆盖八资产、实际编译脚凸块及原物性、左右脚—物体正例、脚—地面负对照、四子步冲量与环境隔离；旧三通道/球脚的报告不能给新版背书。当前source15与8500pt验收身份见[最终报告](hoi_v1_final8500_eval_handoff_20261025.md)。

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

每帧 19D，总计 190D。Object encoder为 `190 → 256 → 128 → 32`，ELU + tanh。`z_goal=[zH32,zO32]` 在语义上是 64D 联合目标，但实现时保持原 decoder 的输入前缀，不能为了相邻拼接两 latent 打乱旧输入列。

原 human goal 仍是 reference-only；新 object goal 是 state-conditioned reference encoding。reference 原始轨迹固定，依赖当前状态的只是坐标变换。每个 tick 都重算新 goal，包括 imagined rollout，不能缓存起点坐标变换用满 32 步。越过 clip end 的新增 object future pose 复用末帧，与 human future clamp 一致，但相应 contact validity 置零，不假造尾端接触命令。

### 4.3 Shape 与 decoder

点云为每个 asset/scale 离线确定性 FPS 的 `128×3`。先在真实 surface 上生成足量候选点，再做 FPS；保存采样 seed、候选来源、输出点云 hash 及 transform。Ultra 当前使用 `sample_surface_even(256)` 后减 mesh vertex mean，并不是 FPS；本方案的 FPS128 是新设计。

Shape encoder采用PointNet：逐点 `3 → 64 → 128 → 128`，max pool，再 `128 → 64`。Actor、WM、critic 各自拥有 shape encoder，禁止训练参数共享；点云原始数组可共用缓存。形状固定不代表 embedding 固定，encoder 更新后不能继续使用上个 update 的已编码 feature。

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

从联合 WM 的最终 hidden feature 新接 `2048 → 128 → 5` 的 contact logits，预测 `(X_t,a_t,shape,xi) → b_actual(t+1)`，固定顺序为左手—物体、右手—物体、物体—地面、左脚—物体、右脚—物体。它与 dynamics 共享 WM backbone。脚通道使踢箱子等真实交互进入辅助监督；新增脚与原ground channel用于动力学辅助监督和诊断，当前不增加reference expected-contact目标或脚接触奖励。这里 ground 严格指场景地面，不能把桌面或其他 support 接触合并命名为 ground；若数据要求额外支撑物，需显式扩展场景与 pair schema。

实际标签按**碰撞实体 pair**收集，对一个 control interval 内的接触 normal impulse 求和，再除以 `dt` 得到平均法向力；每个pair的平均法向力严格 `>1N` 记为接触。`b_actual(t+1)` 表示 `(t,t+1]` 这个已执行控制区间的汇总接触，和末端 `X_(t+1)` 一起记录；不是只读取最后一个 physics substep。保存原 impulse/force 数值、单位、聚合方式、有效 mask 和 threshold，同时输出轻触检出率。该1N为当前已锁定标签定义；五个pair正例以及手/脚—地面负对照已通过新版原生校准。标签定义改变必须版本化并重新验收。

当前recorder由Newton/MuJoCo-Warp solver的实体pair解码接触法向力，逐子步积分；由编译模型的有效碰撞几何绑定双手、物体、地面及左右 `ankle_roll_link`，按环境隔离并支持geom顺序颠倒。脚—地面、手—地面与自碰撞均不进入脚/手—物体监督。源39个net-contact flags只作annotation。Contact head不读取reference contact，expected contact不作为BCE真值。固定标签schema为 `hoi_v1_pair_normal_impulse_v2`，feature版本为 `supermimic_hoi_v1_full_reference_2`，与checkpoint、replay、诊断、完整评估同步保存。

### 5.3 WM loss、冻结与梯度

保留原 8-step robot normalized state-error loss 及原 70D 归一化定义，独立新增 object 的 position、SO(3)、linear velocity、angular velocity 归一化误差，以及实际 contact BCE：

`L_WM = L_robot_original + lambda_O * L_object + lambda_C * L_contact`。

当前 `lambda_O=1, lambda_C=0.1`；object 四组误差先各自按维度取均值，再等权平均，防止单位和维度主导。Object output normalizer 与 component error floors 独立维护，初值参考原同物理量 floors（position 0.003m、rotation 0.005rad、v 0.02m/s、omega 0.05rad/s）。Contact BCE逐通道按fit真实接触数据的N/P比例加权，权重clip到 `[1e-3,100]`，无正例时权重1；每步只对有效的五通道取平均，时间mask求和后除以batch数，与原WM窗口归约一致。报告逐类precision/recall/Brier及正负覆盖，不能只报由no-contact主导的accuracy。

训练使用真实 actions 的多步 autoregressive rollout，同时监督各步 object state 和 contact；mask 保留 terminal transition，屏蔽跨 reset 的后续步。检查1/8步holdout WM error及真实action序列的32步漂移，分层报告双手no-contact/contact/onset/release及五通道预测质量。它们只作诊断：固定500迭代WM-only后正常更新actor/critic，质量未达baseline或合法holdout覆盖不足都不重新关闭策略更新；非有限loss/梯度/holdout诊断仍立即失败。

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

原 `hoi/evaluate.py` 的逐 asset 独立评估及 scene monkeypatch 可作为物性读回和评估边界参考，不能直接拿来当异构训练 scene 管理。先验证一个 asset 的物理与梯度闭环，再验证多 asset 的采样/隔离/显存；当前新版显式足部profile及全量bundle已通过4096/H32资源验证；后续若真实OOM，保留phase/worker证据，先定位实现或allocator问题，不静默改环境、batch或horizon。

## 8. 训练日程、初始化与checkpoint

生产配置位于 `configs/hoi_v1_first.json`：4096环境、replay1000、WM window8、每轮10×batch2048 WM fit、actor batch4096/H32、UTD0.4、Adam lr2e-4、gamma.99、lambda.95、critic5epochs×4minibatches、原探索/gradient clipping与bf16设置保留。上限20000 iterations、每500保存、5000/10000/15000/20000自动完整canonical80冻结评估；原LR decay250000不随20k上限缩短。评估为独立GPU任务/新输出，记录claim与回执防止重复；提交或排队不等于完成。

已验收正式运行对parent和8个CPU-IPC workers使用 `PYTORCH_ALLOC_CONF=expandable_segments:True` 与兼容别名 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，OMP/MKL各3线程。这是实际runtime内存/线程配置，应在新生产启动前显式保留并核对runtime支持；不向共享环境写配置。现有critic target在 `no_grad` 下按原critic minibatch上限分块，完整保留135168个bootstrap rows及相同λ-return，PointNet仍在每个训练minibatch中重算。该数学等价内存修复不改变batch、H或优化配比；资源探针不保证长期无allocator碎片。

日程固定为：64 vector steps真实fresh replay prefill；全局前500个iteration只训练WM；从第501个已完成iteration开始，每轮正常进行actor/critic更新。`policy_update_schedule.mode=fixed_after_wm_warmup`，`wm_quality_gate_enabled=false`。每50轮仍对真实holdout计算1/8/32步质量、zero-increment/constant-velocity对照、覆盖和接触概率；`wm_gate_*.json`与checkpoint的历史字段名保留，但 `controls_policy_updates=false`。W&B `gate/policy_passed` 表示固定warmup日程，`diagnostics/wm_quality_passed` 表示独立质量比较。没有质量回落后反复暂停actor的行为。

训练reset从各资产兼容的全部训练bins采样，并在bin内随机取帧，包含动作中段及reference接触阶段；沿用50%均匀/40%回报soft分布/10%其EMA。没有专门接触阶段quota，也没有contact-aware replay window oversampling。Robot和object在同一reference时刻联合重置，保留原robot扰动，不能保证reference接触帧立即有实际冲量。真实接触仍来自后续物理，不把reference transition伪装为实测动力学。

每20个50帧训练bin保留1个WM holdout，fit与normalizer不使用holdout行；这仍属于train split，与eval80/test40隔离。窗口包含terminal transition，之后reset、reference jump、跨asset部分不得串接。稀少合法holdout窗口保留32轮快速拒绝采样，预算不足后枚举相同元数据合法起点均匀有放回补齐；真正无合法起点只记录coverage不足诊断，不伪造窗口，也不控制actor日程。非有限诊断或其他异常仍为硬失败。

当前新版首选显式 `--from-scratch`：三个网络与optimizer从零，不使用旧motion或三通道HOI权重。保留可选motion-only legacy迁移接口，必须逐项验证，而非 `strict=False`：

- Human encoder及normalizer完整保留；decoder旧1022列复制、新276列零初始化，后续层完整复制。
- WM首层旧133列、原70行输出及旧normalizer复制；新输入列零初始化、object输出12行独立初始化。联合backbone后续可微调。
- Critic只在来源含critic时迁移旧914列；若仅有周期pi/wm小checkpoint，则critic从零。新增normalizer独立维护。
- 来源physics缺失时显式声明hand/self；不同physics属于新迁移实验，不能标作同实验resume。架构扩展重新建立Adam，不加载形状不匹配的旧moments或伪沿用iteration。

新checkpoint严格保存feature schema/version、五通道顺序与标签定义、三网络及各自PointNet、全部normalizers/三个Adam、iteration/update counts、variant、固定日程、source/dataset/bundle/timebase/物理和初始化身份。旧三通道HOI checkpoint明确拒绝，未实现3→5自动迁移，不静默补标签或Adam行。相同schema可在新输出目录resume，但必须fresh replay prefill；不含完整replay/模拟器/RNG，不能称逐位连续恢复。

所有训练或评估使用独立新输出，不覆盖现有权重、rollout、日志和渲染。原 `supermimic.train` / `hoi.run` 的motion-only/zero-shot入口继续保留，不向它们自动增加物体policy观测。

## 9. 验收、评估与对照

### 9.1 硬性验收与学习质量诊断

| 类别 | 验证内容 | 约束 |
| --- | --- | --- |
| CPU data/schema | selection/逐motion SHA、630D/schema、joint/body/quat、分组、source与derived速度、cache/contact/history/window对齐 | 身份或接口错误立即失败 |
| Time/frame alignment | source60/control50共时轴，整数N−1时长及T+1/T，601帧→500actions/501states | 不以相同总帧数替代采样核验，不冒称真实生成时钟已确认 |
| Legacy compatibility | 仅针对显式迁移：human latent、初始action/robot WM及原reward/termination数值一致 | 迁移不兼容不得假称“只增不减”成立 |
| Native physics / geometry | hand/self ON、显式ULTRA足部manifest、八资产真实物性、compiled hull/FK/palm/距离一致性、五pair正例与手脚碰地负对照、无overflow | 硬门；旧球脚/三通道证据不能替代新版 |
| BPTT | object reward对action有限非零梯度、有限差分方向、actor更新时WM/critic参数和统计不变 | 梯度断链或泄漏立即失败 |
| Resource | 全量bundle、实际4096配额、replay1000、64prefill、WM2048×8/actor4096×32/critic生产尺寸 | CPU或小fixture不代替全量native资源验证 |
| WM / contact质量 | 双手phase分层1/8/32步误差、简单baseline、五通道precision/recall/Brier及正负覆盖 | 只作诊断，不控制固定warmup后的actor/critic；非有限仍硬失败 |
| Simulator evidence | 完整动作时长真实验证、human/object/contact与imagined/real gap分别报告 | imagined回报不能证明真实HOI成功 |

本分支不设置持续WM质量门。Native/数据/有限性和梯度等实现硬约束，与WM是否优于简单预测baseline属于不同判断。当前已完成的八资产、全量资源、正式训练及完整80/视频证据见[8500pt验收](hoi_v1_final8500_eval_handoff_20261025.md)，不得以表中规范反向虚构未做的实验。

### 9.2 独立对照入口

| 名称 | 定义 | 用途 |
| --- | --- | --- |
| ZS | 原冻结motion策略、真实物体、显式checkpoint/碰撞profile | 保留原zero-shot边界 |
| HOI-v1 | 当前五通道/完整手脚碰撞/固定更新日程，全部新增目标，无16步硬终止 | 当前主开发基线 |
| No-contact-objective | 仅contact-match权重0，保留head/BCE/诊断 | 独立配置 `hoi_v1_no_contact_objective.json` |
| No-geometry-input | 三套PointNet与接口保留，shape feature一致置0 | 独立配置 `hoi_v1_no_geometry_input.json` |
| Contact-timeout16（后续设计） | 需完整counter state/critic及可微规则，当前未启用 | 另行设计与授权，不能直接改终止 |

这些入口不是启动授权。每个新对照用独立run；full-reference HOI不称zero-shot。Ultra/ProtoMotions因引擎/观测/训练范式不同，只作注明条件的外部参考。多个变量同时改变的旧三通道/球脚和新版比较不能被当作单变量消融。eval80用于开发/选模；test40及额外seeds须在数据、模型、奖励、阈值和checkpoint计划锁定并获授权后独立执行。

### 9.3 完整评估与交付

评估冻结checkpoint及全部统计，optimizer/normalizer updates均0；每条从开头到完整canonical时长，failure后继续真实物理，无reset或截断。状态/reference为T+1，action/observation/goal/reward为T，保存body_rot、实际时间/source N/indices/尾余、速度模式、quat、数据/物理/标签身份。不同长度padding仅服务batch，评分按各条真实长度mask。

当前 `hoi.v1.evaluate` 按八资产组依次执行，组内多环境batch inference与物理并行，canonical80的batch为15/5/16/4/14/9/11/6；既非逐环境串行，也非80路同时。原39body PHC独立于14body human reward。分别报告human normalized return/作者终止、HOI normalized return/HOI终止、人体与手部误差、物体位置/角度误差、object-pose及PHC+object联合通过、五通道实际接触与双手expected/missing/extra/timeout诊断。

PHC要求所有评分帧39体平均世界位置误差≤0.5m；object pose使用位置0.1m或角度30deg超限持续≥0.3s的工程规则。误差逐clip均值后宏平均，不分别对齐轨迹。作者终止、PHC与object-pose是不同判据；`supermimic_n_failed=0`或object-pose通过不代表抓取成功。没有锁定并验证的抓取判据时沿用诊断名称。

离线视频仅回放已保存状态，0policy/physics/FK重构。固定四类各080/084/089/094/099，不按成绩选；720p/50fps全部T+1帧，reference机器人/物体半透明蓝色RGBA[.08,.43,1,.35]，actual原色/橙物体，共世界固定相机，不单独对齐或隐藏漂移，使用绑定原USD视觉。远端和本地20单条+合集均完整解码及SHA核对，并实际查看contact sheet与四类首中末帧后才称媒体验收完成。

W&B项目保持私有，各对照独立run，仅配置/指标/评估汇总；不上传权重、源码、原数据、完整rollout或视频。源码、资产、数据与报告使用哈希及新输出目录绑定，已完成阶段不因同步失败重跑推理/物理。

## 10. 当前实现与资料索引

当前完整实现位于 `hoi/v1/`：`networks.py` / `learner.py` 定义网络与优化，`contacts.py` / `native.py` / `foot_collision.py` 定义五pair实体监督与显式碰撞profile，`buffer.py` / `motion.py` 管理真实replay和reference，`mdp.py`定义解析观测/奖励/终止，`train.py` / `evaluate.py` / `render.py`分别提供训练、完整冻结评估和保存状态渲染。数据schema与60→50适配位于 `hoi/data/`，原共享组件位于 `supermimic/`。

使用入口、资产recipe与preflight方式见[HOI v1 README](../hoi/v1/README.md)、[准备说明](../hoi/v1/PREPARATION.md)、[数据适配README](../hoi/data/README.md)。先读[DEVELOPMENT.md](../DEVELOPMENT.md)及[AGENTS.md](../AGENTS.md)，CPU回归和Linux native检查各有证据边界，不把提交或RUNNING当成稳定训练。

目前仍未证实发布数据真实生成间隔及源velocity全部物理参考点；当前reference使用已记录的pose-derived control-grid模式。Native证据、完整缓存及8500pt评估已通过，不再把早期“全量数据/接触instrumentation/显存待验证”作为当前待办。新引擎、资产、schema或物理变更必须另行验收。

外部资料索引见[notes/README](README.md)及[数据时基交接](ULTRA_DATASET_AND_TIMEBASE_HANDOFF_20261023.zh-CN.md)：

- `D:/Protomotions` 的 `hoi/dagger_distillation`：分组manifest SHA `7d66cd15b27b41988ac3ef761fb82f15204b5ab741a696f95102801dac8b62e2`；canonical eval80 SHA `459a3925faf0f66bee258ce98635c67802ddda00d47d007ecdeb45035f576002`；test40 SHA `b7e61757f230930fe18672f619149d7201ccdb6ab7a6ac06de0ae1634cf7674d`。历史G19500 worker源为 `faefebd52e1e5872f008532f48606d86568d86b6`，后续证据提交 `79b94ef15ea5dd4bcdb479ecc430d3907a3a851e` 不替代worker身份；只复用数据/评分经验，不恢复其训练。
- `D:/Ultra` 原审查HEAD `ebe4aa276dd3208c33a8604084c0a3aeb37d9061`：实际HOI训练URDF/整脚STL、630D consumer、exporter与参考接触规则的核对依据；当前exporter不是发布数据生成时钟证据。
- 本仓库源代码为网络尺寸与实际语义依据；[SuperMimic](https://arxiv.org/abs/2610.28480)、[ULTRA v3](https://arxiv.org/html/2603.03279v3)、[ContactMimic v1](https://arxiv.org/html/2607.08742v1)为概念参考。本文新增encoder、2/4cm几何schedule、五通道head及新reward是本项目定义，不宣称论文原配置。
