# HOI v1 资源验收独立审查（2026-10-23）

后续真实优化暴露了本次静态无step诊断未覆盖的多轮allocator碎片OOM。source13保留科学参数，以no_grad target等价分块及运行时expandable allocator修复，219项CPU回归与50轮完整GPU优化已通过。正式恢复warm1461248/scratch1461253及真实显存/holdout/policy结果见[部署与续训记录](hoi_v1_holdout_resume12_20261023.md)；本页余下结论仅属于下述source09历史范围。

本审查只读核对 `hoi/v1.0` 当前实现与 source09 的实际资源报告，未修改 learner、模型、超参数、物理配置或已有结果，未 SSH、提交作业或执行 optimizer step。后续额外获授权仅写本说明。

**结论：未发现使 source09 在其明确声明的测试范围内产生资源假阳性的 P0/P1 代码问题。8 clip fixture 的通过不能代表 full train2651 就绪。只有全量缓存及合同审核、同完整 bundle 的八资产原生门、实际全量 bins/quota 的 4096/H32 资源检查，以及生产 64 步 prefill 诊断全部通过后，才可通知准备好首次实验。仍不能保证长期不会发生物理 overflow，也不能据此宣称稳定训练或收敛；本轮不启动正式训练。**

审查依据为 `hoi/v1/resource_probe.py`、`train.py`、`learner.py`、`workers.py`、`motion.py`、`prepare.py`、`buffer.py`、`networks.py` 与 `checkpoint.py`，及本地只读回拷报告 `data/hoi_v1_native_receipts_20261023_01/resource4096_source09_20261023_01/resource4096_20261023_01/report.json`。下列行号对应本次审查时的工作树。

## 实际覆盖与正式入口的一致性

| 内容 | 代码位置 | 审查结果 |
|---|---|---|
| 环境、replay、actor batch、H | `resource_probe.py:30-38,238-241`；`config.py:14-31` | 保留 4096 / 1000 / 4096 / 32，拒绝缩小这些尺寸。 |
| 全部真实固定几何 worker | `workers.py:39-62,130-142`；`resource_probe.py:265-273` | 使用正式分配函数与同一 WorkerGroup；按兼容训练 bins 分配，八组环境数之和保持 4096。 |
| WM 图与 loss | `resource_probe.py:286-299`；`learner.py:114-149` | 同一 `wm_rollout` 和 loss，2048×8，额外筛选每一步全部 active 的真实窗口，避免 reset mask 缩小容量图。测量一个完整 batch，未假称 10 batch 的迭代吞吐。 |
| actor H32 前后向 | `resource_probe.py:304-326`；`learner.py:157-201,243-260` | 同一 `imagine`，真实 replay 起点与同 episode history，包含所有资产。WM/critic 仅冻结参数，保留对状态和动作的 Jacobian。 |
| critic target 与 minibatch | `resource_probe.py:327-365`；`learner.py:203-241` | 同样保留 actor trajectory 和 actor gradients；先计算 33×4096=135168 行完整 target-value 前向，再对 32768 行执行带梯度 minibatch。PointNet 仍在前向内。未声称已跑 5 epochs×4 minibatches。 |
| 无 optimizer step | `resource_probe.py:283-299,310-365,367-371` | 手动执行前向、loss、backward 与 clipping，未调用 `optimizer.step()`；没有解锁训练 WM gate，也没有进入正式 train loop。资源测试会更新其临时模型的运行归一化统计，不应将其描述为冻结策略评估。 |

source09 九个阶段完成、梯度有限，replay 实际分配 4,739,072,000 bytes；报告中的 `optimizer_steps=0` 与代码一致。它是容量诊断，不是一次正式优化迭代。

## 全量 references 与 fixture 代表性

`motion.py:35-63` 保存机器人与物体 reference、14-body FK、contact schedule 和局部速度；`prepare.py:552-555` 额外保存 global frame 映射。`workers.py:130-132` 的每个子进程也载入自身资产的完整分 bank，因此完整数据不仅增加 central bank，还增加所有 worker 分 bank 的总驻留。

按 train 不保留 `source_reference`、float32 reference/FK、int64 索引估算，长期驻留约为每状态 356 个 float32 加 4 个 int64，即约 **1456 bytes/state/bank**，另有 clip/bin 元数据和几何缓存。756524 状态的 central bank 与各 worker 分 bank 合计约 2.20 GB；相对 2088 状态 fixture 新增约 2.197 GB。此为静态量级估算，不是 CUDA allocator 的实测增量。

`allocate_workers()` 用当前 bundle 的 `bin_asset` 计数，8 clip fixture 的比例不能替代 full2651 的真实 quota；相同总环境数也不表示每组模拟器内存相同。完整 bundle 的实际 worker 分配、references、接触/约束容量必须一起实测。

source09 的同步全进程最大 GPU used 为 46,663,335,936 / 47,590,277,120 bytes，global free 约 884 MiB。但相应 critic 阶段 parent **reserved 27,030,192,128 bytes，而 allocated 约 6,063,409,664 bytes**，其中约 20.97 GB 属于 allocator 缓存。不能用“新增 references 大于 884 MiB”直接判定 OOM；新增分配可能复用或回收缓存，也可能因实际配额、进程隔离或碎片受限。真实 fullbundle 复测才给出结论。

## 正常迭代与 probe 的小幅内存差异

以下是必须保留的边界，未发现需要据此改变模型、batch、H 或正式 learner 的问题。

| 差异 | 代码与静态量级 | 对结论的影响 |
|---|---|---|
| 上一轮 actor gradients | probe 在 `resource_probe.py:307-308` 前置清空；正式 `learner.py:250-252` 在 `imagine` 后、backward 前清空 pi gradients。生产第二次 actor forward 可能多驻留 **39,203,956 bytes**。 | probe 略欠保守，量级约 37.4 MiB；不是漏掉一整套 BPTT 图。可用等字节诊断 reserve 覆盖，无需修改正式训练时序。 |
| 后续 WM batch 的 gradients | `learner.py:136-146` 在新前向/loss 后才清空 WM gradients；probe 的首个 WM batch 没有上一 batch gradients。静态多 **37,319,252 bytes**。 | 10 个 batch 串行执行，不会按 10 倍保留图；只可能多一份旧 gradient。 |
| 后续 critic minibatch gradients | `learner.py:229-234` 的清空顺序相同，最多多约 **1,509,636 bytes**。 | 20 个 minibatch 串行执行，旧 backward 保存的激活已释放；不是 20 倍图驻留。 |
| Adam states 与 scratch | `learner.py:45-47` 是三个非 AMSGrad Adam；probe `resource_probe.py:259-264` 为每个参数预留两份 moments，共 **156,065,688 bytes**，与三模型参数合计 **78,032,844 bytes** 一致。 | 三个 optimizer 不等于三份 moments。未执行 foreach 临时分配/内核。默认普通 foreach 的主要额外平方根张量量级约当前模型参数一份，最大约 39.2 MB；真实 runtime 实现仍以实测为准。step 在 backward 后发生，不同时保留整套前向激活。 |
| checkpoint 深拷贝 | `checkpoint.py:87-88` 深拷贝 CUDA optimizer states 与 model states；`train.py:126-139` 在迭代结束保存。三 optimizer 都建立 states 后，额外约 **234.1 MB** 加少量 buffers/scalars。 | probe 未直接覆盖保存阶段。它发生在 backward 后、policy_update 返回后，容量明显低于 actor 激活峰；没有依据为此提前改变 checkpoint 语义。 |

这些遗漏使“逐项完全相同的正式迭代峰值”表述不成立，但 source09 已明确未执行 optimizer step、未测完整迭代吞吐。尤其 actor stale gradients 约 39.2 MB，应在极小实际余量时作为保留项，不能隐藏。也不能简单将这些量加到全局 used 的 884 MiB 余量上推导 OOM，因为缓存占用与各阶段存活张量不同。

## WM validation 与采样是否还需扩大资源测试

`learner.py:263` 对 `wm_validation()` 整体使用 `@torch.no_grad()`；其 `B2048×32` 自回归仅保留状态/物体/logit，teacher-forced 一步检查在 `295-307` 行逐步串行执行，每次 PointNet 仍为 B2048。完整 replay 窗口约 75.8 MB，rollout 状态量级约 24.6 MB，另有诊断 mask/误差与单步临时张量。静态结构没有出现大于 WM2048×8 带梯度图或 actor4096/H32 的新图。

因此本审查**不将新增 `wm_validation` GPU phase 列为容量验收的必要阻塞项**。它的真实 holdout 覆盖与 gate 数值是首次实验的学习质量检查，不能借资源通过预判 gate 将开启。probe 只有 12 步 replay 时，本来就不足以直接取完整 32 步窗口。

replay 已分配完整 1000×4096 容量，history 固定 10 帧；`buffer.py:66-96` 与 `98-145` 的采样均有界，不会随迭代次数积累图。probe 的 full-active WM 筛选比普通训练对 reset mask 更保守。actor 起点有放回采样，因此并不建立独立 episode 数量、全时间段接触覆盖或完整数据行为覆盖。

生产 prefill 为 64 步，source09 资源 probe 当前为 12 步。12 步已经填满固定 history，但可能遗漏模拟器延迟 workspace 和较长真实 collection 的缓存行为。根代理已决定在 fullbundle 绑定八资产检查及 12 步资源作业 `1459914` 成功后，追加 **resource-only `--collect-steps 64`** 诊断，复用相同 fullbundle 的成功 native evidence；无需再做八资产门、无需 optimizer、无需降参数或改正式 learner。这是足够小的补验范围，不再无限扩大无关测试。

## 就绪表述的边界

完成上述 fullcache、fullbundle gate、实际 quota 与 64 prefill 检查后，可报告“已准备好第一次实验的输入、原生物理及 4096/H32 资源条件”；应同时保留 source/checkpoint/bundle/物理配置身份、GPU型号、实际峰值与未执行 optimizer step 的限制。

不能将该结论写成 20k 训练稳定、完整迭代吞吐已验证、WM gate 一定开启、HOI 交互已成功或模型已收敛。物理 overflow 仍需正式运行中即时检查并拒绝污染 replay。用户本轮要求完成验证后通知，**未授权自动启动正式训练**。

## 后续启动线程预算审查

完整bundle原生门通过后的补审发现一个P2：source09的`cluster/hoi_v1_train.sbatch.sh`把OMP/MKL强制设置为整个Slurm allocation的24。八个spawn worker与parent共享同一24核allocation，并非各自获得24核；因此可能产生8×24线程池超订阅。resource probe则显式让parent与worker使用3线程。CPU线程数不直接放大CUDA张量、H32图或物理capacity，不能据此推翻全量显存实测，但3线程诊断的采集吞吐不能外推给原启动器。

root按本次补审采用最小修复：正式launcher尊重显式`OMP_NUM_THREADS`/`MKL_NUM_THREADS`，未设置时OMP默认3，MKL继承OMP。首次实验启动命令显式设置两项为3。该项不改learner、模型、数据、batch、H、PD、质量或物理参数。09冻结源码与正在运行的原生/资源作业保持不变；修订启动器用新source10快照，核对全部Python代码与09逐字节相同后复用09的GPU证据，无需重复成功GPU阶段。shell语法和默认/显式环境预算另行验证，结果与快照身份由root记录。
