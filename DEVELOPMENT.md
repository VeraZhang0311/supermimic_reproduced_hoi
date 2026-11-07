# G1 privileged GMT：LAFAN1 与 Boneseed

## HOI residual 扩展入口

当前独立开发分支为 `codex/hoi-residual-ppo-teacher`。新增 `hoi_residual/`，
以冻结 Boneseed 续训4000轮 GMT + 特权残差 PPO 训练 ULTRA HOI teacher，详见
[HOI_RESIDUAL.md](HOI_RESIDUAL.md)。下文保留父分支与原实验记录。
`supermimic/` 科学实现逐字节继承当前 privileged 冻结快照；不启动原 GMT 实验。

当前分支 `boneseed/privileged`，直接从 `main@a1508d4` 创建。
本次 checkout 位于 `D:/supermimic_reproduced_hoi/tmp/boneseed_privileged`；
原目录的 v1.1 分支及其未提交修改保持原样。

## Boneseed 筛选版：从26k再训练20k（2026-11-06）

用户指定从LAFAN1 **26k** checkpoint另开训练，`iterations=20000`表示新run的20k次更新；
`wm_only_iters=0`，新iteration从0开始。64步真实采集用于新replay预填，不是500轮WM-only阶段；
第1次更新即包含WM及actor/critic。新run/output独立，不在原输出上续写。

- 新实验根目录：`/mnt/projects/ll/tianzizh/gmt_bones26k_20261106_02`
- 新输出：`/mnt/projects/ll/tianzizh/runs/gmt_bones_filtered_from26k_20k_20261106_02`
- 父完整checkpoint：`input/parent_026000_full.pt`，SHA256 `878d9dba477ce1c3025f64bebbbe9dc336593c0b0eeb09b8b5dbe38a773fb9eb`
- 对应小checkpoint：`input/ckpt_026000.pt`；CPU逐值校验wm/pi与完整父checkpoint完全相同，metadata均26000。
- 新W&B run ID：`d1c131df`，同一私有project `supermimic_gmt`，独立于原LAFAN1 run。
- 首选资源：`liu-compute / liu-account / ll-med / gpu:l40:1`，24CPU/96G；维持4096环境、replay1000、H32、bf16及原batch/更新配比。

当前正式重试作业`1507980`，实际节点`ll-l40-1`、单张L40；独立CPU日志/评估作业`1507981`。
Linux作业`1507977`的65项协议测试全部通过，真实batch解析结果与26k父参数逐项匹配，
full/eval512 manifest及父checkpoint SHA校验均通过。全量reference实际GPU常驻13.541GiB，
4096环境与replay1000初始化后allocated19.714GiB；真实首轮WM+actor+critic完整更新已通过。
当前验收快照到150轮，WM=`0.0828654`、pi=`-9.52007`、v=`0.00437795`，全部CSV指标finite；
W&B服务器独立readback确认iteration150、stage=`actor_critic`、iterations20000、wm_only_iters0、parent26000和eval512。
首轮Torch peak allocated31.918GiB、reserved33.449GiB；Slurm GPU实际历史峰41450MiB（约40.48GiB），
单L40可完成此全规格学习路径。100轮时去掉首轮编译后的速度约0.58 it/s；不据此声称20k完成或已收敛。
运行后`source.sha256`检查返回0；本地回执`.cache/gmt_bones_finetune_runtime.json`、`.cache/gmt_bones_wandb_readback.txt`。
此前本地136项回归中5项接触测试曾受Windows长路径限制；用已有`G1_CONTACT_TEST_CACHE`短路径配置
重跑接触6项，5通过、1CUDA项CPU跳过；其余129项通过，另1CUDA项CPU跳过，未修改测试物理参数。

`supermimic.gmt_transfer`显式加载全部WM、policy、critic、RMS及三套Adam；仅允许数据集和motion manifest改变，
机器人/物理/38-link/schema/analytic实现SHA必须逐值匹配。旧的strict `--resume`行为保留。
LR衰减沿用父26k，局部iteration 0..19999使用LR index26000..45999；checkpoint保留`transfer`来源与SHA。
不重置RMS；若父RMS计数已达其max_count，仍按原实现保留冻结统计，不声称它会自动重估新数据。
父checkpoint不含完整replay/simulator/RNG，因此新数据重新采集，不声称逐位连续恢复。

首次启动`1507942`在strict参数比较处退出，尚未创建sim或做optimizer更新；其`_01`快照及失败run保留。
根因是KitLauncher会把启动前`livestream=-1`及原解析中缺失（或为None）的camera/viewer保存成`0/False/False`。
新transfer入口显式解析这三项关闭状态，科学参数和contract比较没有放宽；从零训练/strict resume不受影响。
重试使用独立`_02` source/run，input和reference只读复用`_01`的已核验快照和数据，未重新转换或筛选。
Linux校验另解析实际batch命令及train的argparse/规范化代码，与真实26k父参数比较，通过后才允许拿GPU。

数据准备作业`1507906`将旧`SuperMimic_HOI/data/bones_filtered_generic_2ca8074_packed`的253D缓存
无损投影为`reference/full`的71D：原列`58:71`（world p/quat wxyz/v/w）拼`0:58`（q/qd），
依据真实producer和index核对，不从旧14-body特征重建，不重新筛选/重采样/裁剪。
保留全部131593 clips、48399361帧、50Hz，约268.1543小时；完整源49GB SHA及新states.bin SHA进入manifest。
训练库使用`reference/full`；新数据含约12.8GiB状态及约0.75GiB索引，L40全规格显存仍须由真实首轮更新验收。

用户选择固定512条完整Boneseed动作每2000轮eval：按原clip名字排序，以`random.Random(0)`固定抽样，
清单和seed存于`reference/eval512/selection.json`，所有训练clip仍保留，故它是训练集评估子集，**不是held-out**。
`tools/gmt_eval_worker.py`仅加载小eval库到GPU，校验其full-parent manifest及每条完整长度/时基，
允许manifest改变而严格核对物理/网络/schema；沿用冻结evaluate的reward/termination和failure计分，0训练/RMS更新。
独立CPU writer `tools/gmt_auto_eval.py --create-run`创建唯一W&B run，训练等待ready握手；
训练结束立即释放L40，CPU监听器排空已经保存checkpoint的独立A10评估后收尾。`eval/*`横轴为`eval/iteration`。
新脚本拒绝重复producer及覆盖已有输出，Slurm不自动requeue；意外中断后须明确检查checkpoint和日志再恢复。

## 此前LAFAN1训练（2026-11-05，已按用户要求停止）

用户已授权从零开始GMT：单GPU，`iterations=30000`，优先Liu组、`ll-med`；后续排队A10或L40均可。
此前作业`1506052`，`liu-compute / liu-account / ll-med`，节点`ll-l40-0`，单张L40（46068MiB）。
用户2026-11-06明确要求停止，已确认Slurm CANCELLED；最后训练日志27950轮，新训练仍固定使用26k快照。
该分区当前是3台×8张L40；A10属于共享batch分区。训练保持单卡，独立评估只短时申请A10。
保留4096环境、replay1000、H32、bf16、500轮WM-only及原优化配比。训练已越过500轮warmup，
W&B服务器确认`stage=actor_critic`、`iterations=30000`；不是仅依据Slurm RUNNING推断已开始优化。

- 冻结源码：`/mnt/projects/ll/tianzizh/gmt_l40_20261105_02/source`
- 新输出：`/mnt/projects/ll/tianzizh/runs/gmt_lafan1_hw3_v3_30k_20261105_02`
- Slurm stdout：`/mnt/projects/ll/tianzizh/gmt_l40_20261105_02/slurm-1506052.log`
- 私有W&B项目：`tianzizh-university-of-pennsylvania/supermimic_gmt`
- 当前run：<https://wandb.ai/tianzizh-university-of-pennsylvania/supermimic_gmt/runs/c0107041>

`tools/wandb_gmt_monitor.py`在独立SDK进程中读取已存在CSV，仅上传16个数字指标和白名单配置摘要；
SDK依赖与训练进程隔离，复用现有认证，不将key写入脚本/日志，不上传源码、模型、数据或rollout。
启动器等待online握手才启动trainer，结束时传递exitcode并final drain。SDK在独立telemetry_venv中，
共享kitless-cu128环境未改。监控协议8项CPU测试通过，运行时兼容测试12项通过；完整83项中82通过、1个CUDA图测试CPU跳过。

第一次正式启动`1506032`在driver535.216.01的conditional CUDA graph捕获处失败，尚无优化更新，原输出保留。
当前独立快照只对不支持conditional graphs的CUDA设备设`mjw_model.opt.graph_conditional=False`，
并做实际读回检查；不改求解迭代上限、接触/电机/网络/batch。受支持设备保持原选项。
当前作业已通过该阶段并产生WM优化日志。此前学习实现的SHA逐文件未变，唯一运行时兼容差异在
`.cache/gmt_job_1506052.json`和`.cache/gmt_training_source.sha256`中记录，项目回执`.cache/gmt_wandb_project.json`。
正式4096环境H32 actor显存峰值与训练收敛仍须按后续真实日志判断；未commit/push。

## 此前LAFAN1训练的自动评估

每2000轮评估一个原子保存的`ckpt_*.pt`，直到30000轮；会补评已有的第2000轮。
原训练停止后CPU监听器已自动收尾，2000..26000均已评估；未产生28000/30000 checkpoint，属于用户停止后的未完成日程。
`tools/gmt_eval_worker.py`直接复用当前冻结源码中的`evaluate()`，校验checkpoint/schema/物理/数据身份，
按main协议评估同一40条完整LAFAN1 clip：nominal摩擦、无DR/观测噪声/动作噪声、确定性策略，
沿用当前训练termination和tracking reward。触发failure后的计分方式与main一致；0 optimizer/normalizer更新。
成功结果在checkpoint SHA及策略状态不变校验后原子发布；失败独立保存error receipt。

`tools/gmt_auto_eval.py`是CPU监听器，本次作业`1506307`（batch，2CPU/4G，无GPU请求）。
控制目录`/mnt/projects/ll/tianzizh/gmt_auto_eval_20261105_01`，部署脚本为`auto_eval.sbatch.sh`和
`eval_worker.sbatch.sh`；`state/ledger.json`记录提交/重试/结果，`state/attempts/`保留每次回执。
监听器一次只提交一个短时A10评估，评估结束释放GPU，等待下一checkpoint期间不占GPU。
独占锁、提交前持久claim与Slurm核对防止重复作业；失败最多3次，训练退出后排空已保存checkpoint并结束。

旧CSV日志侧进程及其SDK core已精确识别并停止，新监听器作为唯一W&B写入者恢复同一run `c0107041`。
训练日志横轴保持`iteration`，`eval/*`使用独立`eval/iteration`，晚到的2000轮评估不会被后续训练步丢弃。
结果同时写入原run的`eval.csv`；权重、源码、原数据和rollout不上传。训练进程与冻结训练源码未改，未重启/续训。
调度器11项CPU测试已在Linux通过（含真实fcntl锁）；worker另有11项CPU协议/AST测试。

真实首评`1506312`已COMPLETED/ExitCode=0:0（含初始化24m35s）并被监听器验证接收：iteration=2000，40/40条覆盖，
normalized return=`0.6387278438`，frame-weighted return=`0.6668889523`，18/40条触发failure。
这是评估链路成功，不是全部动作成功。checkpoint SHA和策略/normalizer状态不变，0训练/optimizer更新。
原run的`eval.csv`已写入；W&B服务器独立readback确认`eval/iteration=2000`、同样分数和失败数，
`eval/logged_iterations=[2000]`、`eval/error_count=0`；训练指标仍更新到4400轮。
第4000轮checkpoint被自动发现后已顺序启动评估`1506432`，无需手动提交；等待结果时训练持续运行。
本地真实回执为`.cache/gmt_auto_eval_runtime.json`和`.cache/gmt_eval_wandb_readback.txt`；
原训练启动时`source.sha256`全量检查返回0，冻结训练实现未变。

当前评估有性能开销：main每步reward计算一次当前机器人FK，reference body在MotionLib加载时预计算；
本分支privileged在此之外每步再计算当前状态及+1/+13目标的三次38-link FK，尚未单独compile/缓存。
因此“三次FK”指新增privileged部分，不能当成main的原有开销，也未据此断言所有耗时都来自碰撞物理。

第一阶段训练不读取物体的 privileged GMT，后续 residual learning 单独实现。
本分支不迁入 HOI reward、object/shape/contact head 或 HOI 训练入口。
第一阶段此前使用 **LAFAN1 G1 fullset，全部40条完整clip**；用户现要求按上节从26k迁移Boneseed筛选全集。
分支名保持不变。通用CLI默认仍为LAFAN1；Boneseed新训练启动器显式传dataset及准备好的全量筛选库。

## 当前数据：LAFAN1 fullset

训练默认 `--dataset lafan1`、`data/lafan1_g1/*.csv`。启动要求恰好40条唯一clip；
缺少数据、子集、重复身份或格式错误会明确失败，不静默用少量clip训练。
沿用main的G1 retarget数据和时基：30Hz无表头CSV，米、xyzw根四元数、29关节弧度，
完整clip重采样到50Hz，不裁剪、不筛动作。动作来源沿用 `tools/download_data.sh` 中的LAFAN下载；
机器人资产改用下述冯桥已校验tar，不再依赖原main的机器人USD。
每个CSV的内容SHA、名称、源帧数、源fps与控制dt共同绑定checkpoint数据身份；
评估读取训练args中的dataset和motion_dir，覆盖同一fullset，0优化/normalizer更新。

## 观测

参考 `codex/hoi-three-gpu-collector@cc99253` 的 `hoi/v1/privileged.py`。
保留原 `main` 的 99D 本体观测、10帧历史、730D dense motion goal 和 32D motion latent；
向原 decoder 追加独立 RMS 的 **3371D** 无噪声 robot privileged block，采用冯桥 hw3 的38-link定义及控制边界执行器状态：

| 每个 lookahead（+1、+13 个 50Hz 控制帧） | 维度 |
| --- | ---: |
| 当前38-link状态：root height、去root相对位置、heading坐标姿态/速度 | 568 |
| reference body target及position/orientation/velocity errors | 912 |
| previous action、current q/qd、previous q/qd | 145 |
| 两档合计 | 3250 |
| 共享 raw robot context：friction、29 joint offsets、3 torso COM offsets | 33 |
| 控制边界执行器：上个绝对关节目标29、共享delay1、电机强度29、关节摩擦倍率29 | 88 |

原 decoder 隐藏层不变，输入从1022增至4393。`RobotKinematics38` 使用冯桥
hw3 原始 URDF 的 link顺序、安装变换和 nominal COM marker；根输入仍为link-origin速度。
原39-link定义去掉`d435_link`、`left_rubber_hand`、`right_rubber_hand`，
新增`LL_FOOT`、`LR_FOOT`；软手碰撞块挂在腕部，不另增手指自由度。
`mid360_link`的安装旋转也采用 hw3 定义。reference FK与模拟资产使用同一来源，
`robot_privileged_gmt_hw3_actuator_v3`明确拒绝旧39-link及3283D/v2 checkpoint。
rotation6D仍为参考分支的x/z列。
删除 object pose/velocity/error、object context/shape、interaction graph、object/ground
expected region contacts及validity、two-cuboid scene context，未填任何object占位零值。
reference source分支添加的object goal encoder也不引入此motion-only模型。

本体前缀与追加块均不加观测噪声（与teacher profile一致），探索action noise保留。
采集、replay、WM imagination和评估共用特征函数；episode首帧previous q/qd等于当前值。
RMS仅真实采集更新，想象和评估不更新。WM/critic增加88D执行器输入，隐藏层宽度/深度、奖励权重、终止阈值、50Hz控制、
4096环境、H32、batch、优化器和500轮WM-only日程保持main设置。
机器人PD和接触物理改为共享hw3配置；每个50Hz控制区间执行四个5ms物理步。
replay明确保存previous q/qd。checkpoint绑定新schema、hw3物理配置和资产身份、URDF SHA、body/joint identity与
数据manifest SHA；旧LAFAN/HOI checkpoint不能直接resume。
resume缺少replay/simulator/RNG，必须fresh prefill，不声称逐位连续恢复。

## 可选后续数据：BONES-SEED

来源：[BONES-SEED G1](https://huggingface.co/datasets/bones-studio/seed)，
使用 `g1.tar.gz` 解出的 `g1/csv/{session}/*.csv`。数据访问条件由用户在数据源完成，
这里不下载全库、不处理凭据。原始数据和生成reference不入Git。

已按[NVIDIA官方转换器](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic/data_process/convert_soma_csv_to_motion_lib.py)
核对原生列名、厘米、Euler xyz度、关节度，转米/弧度/wxyz，120Hz→50Hz。
按列名重排29关节，SLERP根姿态，差分在单clip内计算，不跨clip。
拒绝缺帧、非有限值、错误列及短于一个控制transition的输入。

准备命令（CPU环境需要numpy、torch、mujoco；指定全新输出）：

```bash
python -m supermimic.boneseed --source /absolute/bones-seed/g1/csv \
  --output /absolute/new/boneseed_g1
# 可选：--selection /absolute/frozen_selection.json
```

selection是相对source的CSV路径JSON列表；未传时使用指定目录下的全部CSV。
manifest逐条记录source路径、SHA、50Hz帧数，绑定完整float32 `states.bin` SHA；
禁止覆盖已有输出，未完成输出没有manifest。输入不裁剪、不重定向朝向、不增加增强。
原论文Appendix E还有name+feasibility筛选，完整最终关键词/clip清单和floating-foot
stationary gate细节未取得。本入口**不自动仿造筛选、不宣称精确129,514条**；
manifest明确 `exact_paper_selection=false`，后续确认的selection应作为新独立数据产物。

reference仅驻留71D float32 generalized states，根矩阵与body FK按索引计算，
不预计算全库body矩阵、不降为fp16。47.3M帧时状态约12.5GiB，clip/bin索引另计；
不是实际全库容量验收。分批评估默认256环境，逐条覆盖完整选定库，不改变clip时长。
`--record`只支持可同时放入eval_envs的库；大库先无record完整评估。

修复main的末帧空bin：当帧数为50n+1时，不让reset采样到没有后继的clip最后一帧；
该修复同时用于LAFAN与Boneseed；其余50%/40%/10% return-based reset配比保持不变。

## 共享 G1 hw3 物理

2026-11-05按用户授权移植冯桥 `gear_sonic@4aed2bb`，统一GMT和后续residual HOI的机器人层。
来源tar MD5：`4389c0463f2037ef8f04520eda1f949b`。先完成物理/学习链路验证，后续正式训练见顶部记录。
原tar、STL和生成MJCF保存在ignored `data/g1_hw3/`；不将原资产/数据/权重放入Git。
源参数冻结对照与文件SHA见 `tests/fixtures/g1_hw3_source.json`。

- `g1_physics.py`：唯一电机/摩擦/材质参数来源。髋俯仰7520-22，139 Nm cap、20 rad/s配置；
  PD增益/armature/action scale同步。每5ms按延迟→显式PD→effort裁剪→驱动/制动包络裁剪；
  near-stall braking、速度退额和每环境/关节reset强度U(.9,1.1)保留。
  MuJoCo没有与PhysX完全相同的硬速度约束，包络而非事后裁剪速度决定可输出扭矩。
- 每环境共享2–4个物理步的命令延迟，partial reset清除本环境FIFO，不污染其他环境。
  关节摩擦在startup按关节U(.8,2.0)，腿U(.8,2.5)缩放；reset不重抽摩擦。
  nominal评估保留尺度1、共享延迟3步、无参数DR的当前工程约定。
- `g1_assets.py`：从verified tar只取source URDF及引用mesh，保持38-link、29运动关节、源质量/COM/惯量。
  47个显式碰撞体，身体mesh用MuJoCo凸包；14个原cylinder按FQ spawner转换为capsule。
  不使用旧HOI的CoACD手/脚overlay，不宣称同一STL经过不同cooker后与PhysX碰撞逐点一致。
- 关节摩擦适配：source slow/fast水平分别作为静/动阈值，fast>slow时取min；
  以native `dof_frictionloss`约束实现，速度0.01 rad/s附近做平滑静/动过渡，viscous=0。
  该过渡速度是版本化工程选择，不是原实测Stribeck拟合；不再额外减一遍tanh摩擦力矩。
- 软手保持腕部的掌/四指整体/拇指hull分块，掌和腕壳刚性，四个指部hull共用软材质。
  MuJoCo用nominal body inverse-weight归一化的direct solref及constant solimp=.9；
  每接触点180 N/m、80 Ns/m，实际左右四指hull/平面1–10N增量标定约0.54 N/mm。
  同材质左拇指约0.54、右拇指约0.36 N/mm（接触流形不同），不单改右拇指来凑门限。
- `elliptic`＋`impratio=10`由CPU对照选择：法向斜率不变，2N侧载/5N压入下滑移约0.000833m/s，
  原pyramidal1约0.01666m/s。仍有软接触creep，不声称与PhysX完全静止粘着相同。
  0.25–4kg静压和卸载数值验证通过；单独软指承担4kg/3cm跌落时可有大穿透，已超出源0–10N标定，
  仅记录数值收敛，不作为大载荷保真通过或训练就绪证据。
- 表面摩擦使用MuJoCo单一sliding coefficient，body训练U(.3,1.2)、hand=1、地面=.3；
  nominal评估body/ground=1。涉及手与身体/物体的接触用模型级`callback.contactfilter`在
  narrowphase后、约束建立前执行source `multiply`：物体mu=.2/.8时接触滑动摩擦分别为.2/.8，
  保留手的priority及软法向solref/solimp；不改其他接触或共享runtime文件。
  GMT地面native mu=.3是body max混合的工程约定，手—地面保持source nominal有效mu=1。
  回调绑定已核对的MuJoCo Warp 3.15源码顺序，CUDA图内执行；不支持的版本或被替换的回调明确失败。
  其余接触仍用MuJoCo native max，和source PhysX材质组合规则有差异；
  source独立static表面摩擦和restitution DR未作一一映射，保留在版本化差异清单。
  source额外torso/wrist质量随机化与push是任务recipe，本次不添加；现有GMT torso COM/joint offset保持原约定。

`g1_runtime.G1PhysicsRuntime`是共同运行接口：future HOI将GMT target与residual相加后调用
`step_target(combined_target, after_substep=contact_accumulator)`；两种任务不能再各写一套PD或摩擦。
callback在每个5ms步提供真实接触积分入口。当前分支还没有PPO/residual训练实现，本次不增加该训练器。
外部物体/地面的碰撞priority须为0，保留手priority=1的软法向材质；初始化检查实际active碰撞体，不满足则报错。
运行时核对实际compiled hull支持面/局部变换、质量/完整惯量、关节限位/armature、solver选项和材质读回；
不满足时明确失败，不静默退回原资产。只更新G1拥有的native数组行，保留未来物体材质及随机化。
checkpoint合同绑定schema、全部资产SHA、profile与共享实现SHA，旧物理/39-link权重明确拒绝resume。

WM继续预测原71D物理状态，输入由133增至221；critic输入由914增至1002，二者及teacher均见同一88D执行器状态。
`actuator_state.py`精确表示50Hz控制边界：每周期四次写相同target、delay≤4时，最近四槽必相同，
第五槽下步读取前会被覆盖，因此上一absolute target29+delay1+strength29+friction29即可确定后继。
真实导出检查FIFO边界/参数范围；未来若改200Hz逐子步target、delay范围或decimation需升schema，不能沿用该压缩。
replay保存reset前的当前执行器状态；env另外导出reset前下一状态供一致性验证，避免autoreset串episode。
WM多步loss从首帧执行器状态open-loop解析推进，不逐帧读取未来FIFO；H32想象同样解析推进并保留action梯度。
不学习FIFO、不把delay当静态零值，也不把想象中的上一target固定。静态59D参数在episode内原样传播。
只存当前88D，不重复存可重算的下一状态；默认replay增加1.343GiB，总计约5.848GiB，不是整段训练显存峰值。
checkpoint同时绑定新状态schema、head配置/实现及物理实现；旧权重须从零重新训练。

head审查修正两处实际坐标错误：privileged姿态误差改为heading系的world relative rotation，
reward/critic的body参考重定向改为严格`heading(actual) @ heading(reference).T`，而非从含roll/pitch的relative rotation再提yaw。
非零关节/根姿态差及大倾角的共同yaw不变性已有真实FK回归。
采集已有的action±20界限统一应用想象及评估。保留99D本体观测、730D目标、14个tracked moving bodies、
所有reward权重/std、指令action-rate惩罚、真实绝对q限位、height .25/tilt .8 hard/soft终止阈值，
以及20ms reward和gamma=.99时基。tilt仍是参考Rzz差的原baseline定义，未改成另一个终止判据。

## 资产准备与验证（不训练）

CPU需要numpy、torch、mujoco、trimesh。使用全新输出，原tar和已生成资产不覆盖：

```bash
python -m supermimic.g1_assets --archive /absolute/GR00T-WholeBodyControl-4aed2bb.tar.gz
python -m supermimic.g1_validation --output .cache/hw3_cpu_validation.json
python -m unittest discover -s tests -v
```

真正Isaac Lab/Newton GPU验收在已准备的固定Linux runtime执行。以下命令本身不提交作业，
不读取motion/policy/optimizer；本次使用独立短时Slurm物理验证作业包装执行：

```bash
python -m supermimic.g1_usd
python -m supermimic.g1_native_smoke --num_envs 2 --steps 25 --output .cache/hw3_native_smoke.json
```

该命令检查转换后物理variant、contact字段读回、floor/外部材质保留、每周期四次200Hz电机更新、
partial reset隔离、真实接触、finite/overflow。缺依赖或失败写不通过报告并非零退出。
MJCF→USD使用 `g1_usd.ensure_usd()` 的独立子进程，不在活动Isaac Lab进程内调用
`MjcfFileCfg`转换。worker先import `isaaclab`设置USD库路径，再注册standalone importer
的`schemas`插件，然后才导入`MJCFImporter`；禁用MuJoCo→PhysX多物理转换，
将生成USD的`Physics` variant选为`mujoco`、保存并重开核验。
`robot.py`用`UsdFileCfg(variants={"Physics": "mujoco"})`加载返回的真实USD路径；
缓存位于`data/g1_hw3/usd/<input-and-helper-SHA>/`，receipt绑定源资产身份、XML、
转换helper/选项及所有输出文件SHA，缺失、失败或篡改均明确报错，不回退旧资产。
无缓存时运行入口会自动完成这一步；`python -m supermimic.g1_usd`可显式提前准备。
只使用standalone转换库与schema注册，不启动Isaac Sim/Kit/PhysX应用。
仅对33个源机器人碰撞mesh写`newton:maxHullVertices=-1`，禁用Newton默认64顶点简化；
该值同时是MuJoCo合法的不限顶点设置。Newton的0会被原样传给MuJoCo，不能用于这条转换链。
A10上已完成独立USD转换及完整native smoke，结果见下述验证回执。

## 自碰撞与后续运行

默认hand collision与body self-collision均ON，保留engine相邻/weld/exclude过滤。
`G1_CFG`的native驱动kp/kd/friction全零，避免显式控制之外再叠加伺服或摩擦。
每个物理步保留finite/overflow硬错误；当前CPU结果不代表GPU吞吐或训练收敛。

以下为通用训练/评估接口；已提交的集群正式配置见顶部记录及Slurm启动器：

```bash
export SUPERMIMIC_ROOT=/absolute/isolated/boneseed_privileged
cd "$SUPERMIMIC_ROOT"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
python -m supermimic.train --dataset lafan1 --motion_dir /absolute/lafan1_g1 \
  --log_dir /absolute/new/lafan1_privileged_run
python -m supermimic.evaluate --run_dir /absolute/new/lafan1_privileged_run \
  --ckpt /absolute/new/lafan1_privileged_run/last.pt
```

`run_train.sh`也默认使用LAFAN1 fullset入口，保留两项allocator和3线程配置，
传入SUPERMIMIC_ROOT应为container内此分支独立挂载路径。
新训练只写全新log_dir；验证阶段未训练，之后按用户授权从零启动上述独立W&B run，没有续训既有实验。
后续Boneseed须显式传 `--dataset boneseed --motion_dir /absolute/new/boneseed_g1`，
使用独立log_dir，不沿用LAFAN checkpoint resume。

## 验证

当前WM/teacher状态版本为v3：本地74项，73通过、1个仅CUDA图测试在CPU跳过；
日志`.cache/gmt_actuator_cpu_tests.log`。新增测试覆盖真实motor全相位/2–4tick延迟的88D充分性、
partial reset、解析advance梯度、实际训练函数H32反传及未来replay context故意置NaN仍不被teacher-force读取。
真实hw3 FK测试验证heading/姿态特征共同yaw不变性、exact reference满分、hard/soft终止边界和offset单位。
新的原生验收入口（不创建optimizer、不更新参数）：

```bash
python -m supermimic.gmt_preflight --device cuda:0 --num_envs 2 \
  --motion_dir /absolute/lafan1_g1 --output /absolute/new/gmt_preflight.json --compile
```

它使用实际train函数定义运行WM8步loss反传、H32 actor through frozen WM和critic bootstrap，
并检查weights逐值及SHA不变、旧checkpoint拒绝、新schema临时checkpoint roundtrip。
初始WM输出层为0，其动力学Jacobian在初始化时为0是正常的；新执行器输入可微和命令记忆推进梯度另行检查。
不以此声称WM已学会新电机、完成收敛或验证正式4096环境吞吐。

当前原生结果：GRASP A10作业`1505964`，COMPLETED（11m03s）；源快照
`/mnt/projects/ll/tianzizh/gmt_actuator_preflight_20261105_02/source`。
完整40条LAFAN1加载身份与既有fullset一致；N=2采集40个控制步，10次timeout自动reset。
88D解析next context和真实controller快照最大差0；160次物理子步扭矩送达最大差0 Nm。
实际train helpers启用torch.compile：WM8步loss/梯度、H32 actor through frozen WM及critic bootstrap均finite且梯度非零；
模型参数torch.equal及SHA前后相同，0 optimizer构造/更新，新checkpoint roundtrip及旧contract拒绝通过。
本次小规模峰值allocated约2.12GiB、reserved约2.17GiB，不是正式4096配置的峰值。
回执`.cache/gmt_preflight_1505964.json`与`.log`，源码逐文件回执`.cache/gmt_preflight_source_1505964.sha256`。
共享环境只读复用、GPU作业已结束；没有开始正式训练。

GRASP正式启动器是`cluster/gmt_privileged.sbatch.sh`，沿用liu-compute/liu-account/ll-med/L40、24CPU/96G请求；
默认30000 iterations、4096环境、replay1000、H32、bf16、500轮WM-only及原优化配比，不添加质量门。
`SUPERMIMIC_ROOT`须指向该v3快照，`GMT_RUN_DIR`必须是不存在的绝对路径。
在login节点通过Slurm提交，运行中的源码不要修改。以下为模板，当前正式源码/run见顶部：

```bash
ROOT=/mnt/projects/ll/tianzizh/gmt_l40_20261105_02
RUN=/mnt/projects/ll/tianzizh/runs/gmt_hw3_v3_$(date -u +%Y%m%dT%H%M%SZ)
sbatch --export=ALL,SUPERMIMIC_ROOT="$ROOT/source",GMT_RUN_DIR="$RUN" \
  "$ROOT/source/cluster/gmt_privileged.sbatch.sh"
```

启动器保留两个allocator变量、OMP/MKL=3，复用已验证kitless-cu128环境，不更新环境或覆盖旧run。
W&B SDK独立venv默认在源码同级`telemetry_venv`，可用`GMT_WANDB_PYTHON/GMT_WANDB_PATH`指定，不能加入trainer的PYTHONPATH。
源码SHA、GPU和Slurm信息写入新run；启动器默认project为`supermimic_gmt`。此处先前N2验收不代表正式训练吞吐。

以下是此前物理移植阶段的独立回执，仍支持相同物理参数/软手标定；
其旧v2源码hash不能冒充当前v3学习链路回执。

```bash
python -m unittest discover -s tests -v
bash -n run_train.sh
```

本地使用匹配cluster的MuJoCo/MuJoCo-Warp 3.15、Warp1.17 CPU wheel验收：56项中55项通过，
仅CUDA graph项在CPU明确skip；相同接触测试在A10另跑6项全部通过，包括CUDA capture/replay。
独立USD转换的8项CPU测试是mock/cache协议测试；真正转换及完整Newton物理步另在A10验收。
本地完整日志：`.cache/tests_final.log`；软手完整压入/卸载/跌落与全机器人CPU报告：
`.cache/hw3_cpu_validation_20261105_final.json`。

2026-11-05 GRASP Slurm作业`1505917`，NVIDIA A10、driver580.178.04，复用只读
`kitless-cu128`环境；独立源快照`/mnt/projects/ll/tianzizh/g1_hw3_port_20261105_08/source`。
两个环境26个50Hz控制区间，共104个5ms子步：每个子步核查真实Warp时步和
`qfrc_applied+qfrc_actuator`与显式命令一致，最大扭矩误差0 Nm，非零命令5633项。
观测到最多60个跨环境接触，partial reset隔离、材质变更后G1字段恢复与外部材质保留均通过，
无finite/overflow错误；0 optimizer updates、无policy/训练。
47个collider实际安装位置/凸包支持函数（容差30um）、armature/限位、质量/COM/完整惯量均核对。
仅`LL_FOOT/LR_FOOT`的源零惯量marker采用Newton具名数值修正，约`1.0001e-6 kg m²`各主轴；
不将它们称为与源惯量逐值相同，普通刚体采用更严格完整tensor容差并分别记录差值。
回执：`.cache/hw3_native_1505917.json`、`.mjb`、`.model.json`及`.log`；
`.cache/contactmix_cuda_1505917.log`记录手—物体摩擦与图回放6项通过。
`.cache/validated_source_sha256.txt`绑定实际cluster源码，`.cache/slurm_1505917.txt`保存分配信息。
验证分配已释放，Slurm状态COMPLETED；未修改共享环境、未训练、未提交Git commit或push。

CPU测试覆盖native CSV单位/列序/时基及坏输入、38-link FK与独立MuJoCo数值对照、
clip边界、goal/reward、特征yaw/平移不变性和有限差分梯度、H32反传、
previous-joint replay、分批评估和legacy checkpoint拒绝、自碰撞parent过滤。
原39-link实现曾通过11项CPU测试，并在旧LAFAN fullset上验证40条、264,705源帧、
441,140控制帧，数据身份SHA为
`ee88d32c6376df53b89d828b875ed324a0bad5633d3e516f9f0eec80dc903abb`。
旧3361D观测与参考分支的逐元素相等回执属于迁移前版本，不能作为当前38-link验收。
当前测试新增严格38-link/schema身份检查，已准备真实hw3资产时逐一比较全部38个
link的世界位置、姿态及nominal COM速度与独立MuJoCo FK。
Windows只做CPU验证；测试包含独立USD转换的缓存/receipt/失败协议mock，不能替代真实转换。
真实Isaac Lab/Newton self-contact、overflow接口、GPU编译及上述短物理rollout已在Linux GPU runtime验收；
这不代表4096环境吞吐/显存、运动跟踪、GMT收敛或HOI任务成功，尚未训练。
