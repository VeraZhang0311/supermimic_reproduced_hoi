# HOI v1 holdout 采样修复部署与 checkpoint 续训

用户明确授权：部署修复，先做 GPU 验证，停止当前两版，从各自最新完整 checkpoint 续训，确认 holdout 与 policy 更新；其他参数不变。本记录取代旧启动文档中「修复尚未部署、两个旧任务仍运行」的时点状态。GPU 验证通过前不提交正式续训。

当前状态（2026-10-24 02:24 UTC）：修复已部署，恢复和实际优化执行检查完成。正式warm **1461248** 从1500到1900，恢复后已完整执行50次policy更新；scratch **1461253** 从500到600，已执行100次。两版holdout报告有限、无OOM/非有限梯度/容量错误；warm最新1901原质量门因release/onset误差重新关闭，继续WM，scratch601门仍通过。没有放宽阈值或改变科学参数。source13+expandable连续50轮GPU **1461203 COMPLETED0:0**、219项CPU回归及独立审查均通过。旧失败输出全部保留，详情见文末。

## 修复范围与独立 review

根因与 CPU 复现见 [holdout 核查](hoi_v1_reset_contact_and_holdout_audit_20261023.md)。只修复 `ReplayBuffer.sample_windows()` 在稀少合法 holdout 起点上耗尽拒绝抽样预算而误报 coverage 的问题。保留 32 轮快速路径，不足时按完全相同的 split/terminal prefix/reset/asset/reference-jump 条件枚举元数据起点并均匀有放回补齐。

source11 → source12 除测试和说明外唯一执行代码差异为 `hoi/v1/buffer.py`；模型、奖励、gate 及阈值、训练超参数和物理配置未改。修复改变随机数消耗，不声称逐位一致。完整 CPU 回归 216/216，通过额外 200 个随机 metadata fixture、1034 次合法 pool 逐 ID 对照。独立 review 对 sampler、GPU probe 与 resume launcher 未发现 P1/P2。

冻结 source12：

- `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_12`
- archive SHA256 `f071dec3aea3ed53ef85c103dbff419c00c8391505c09181205aaaa12ac01821`
- executable content SHA256 `06c5bfc5e91ac23c009c573becd3baabc67f9370d98cc505a84dc69f31eed6b4`
- 177 文件逐一哈希验证；buffer SHA256 `8b2f70b9935292f32c81873513e31b32251bbacaeffbd06cc2c7aeae99d999cb`
- 独立辅助 launcher 与 probe 位于快照 `data/`，分别哈希绑定；没有覆盖 source11、旧结果或共享 runtime。

## 旧任务停止与 checkpoint

| 版本 | 旧 Slurm job | 最新完整 checkpoint | SHA256 |
|---|---:|---|---|
| warm-start |1460279|`source11/results/hoi_v1_first_20k_20261023_01/checkpoint_001500.pt`|`4975c797cde9d18623d090a2518257e09e114b3fb9d91c64f14a443754a7a087`|
| 从零初始化对照 |1460725|`source11/results/hoi_v1_scratch_20k_20261023_01/checkpoint_000500.pt`|`c680a8fa7f6c3519f7cd3c1d88e757b9447e7f109ae5bdc82bdddb9f7c045d9b`|

`source11` 表示相邻的 `supermimic_hoi_v1_development_20261023_11` 根目录，不是新的文件复制。两任务均按用户授权精确 scancel；从零版原无 checkpoint，先保留至首次 500 保存，完整载入及模型有限性检查后再停止。保留其余全部旧输出，不删除或改写。停止后再次验证 schema、iteration、完整 config、data identity、物理两开关、三个优化器键、全部模型 tensor finite，并重复核对 checkpoint hash。

stop/audit 回执在 source11：`data/holdout_fix_warm_stop_and_checkpoint_20261023_01.json` 和 `data/holdout_fix_scratch_stop_and_checkpoint_20261023_01.json`。warm checkpoint 的 WM/policy 更新数为 15000/0，scratch 为 5000/0；不能把旧任务描述为 actor 已正常训练。

## 不变配置与恢复语义

两版都使用原 `configs/hoi_v1_first.json` 完整字典，与各自 checkpoint 逐项相等：4096 env、replay1000、H32、WM2048×8/10 batches、actor4096、critic5×4、64 prefill、500 WM-only、LR decay250000、hand ON/self ON。目标为全局 **20000** iter，每500保存，5000/10000/15000/20000独立 GPU 完整 canonical80；不重新计额外20k，不重置学习率时钟。

恢复模型、三个 Adam、global iteration 与已保存累计统计；原 checkpoint 不含 replay、模拟器和 RNG，按既有流程 fresh64 prefill，因此不是逐位连续。新输出 run 的 WM/policy 更新计数从0开始，并明确其作用域，不把它们当作全历史总数。新 W&B run 保持原私有项目与数值/配置/评估摘要上传边界，记录 checkpoint 来源；不上传权重、源码、原始数据或完整 rollout。

## GPU 验证与续训状态

GPU probe job **1460928**，comment `hoi-v1-probe-source12-20261023-01`，已 **COMPLETED 0:0 / 7m11s**。输出 `source12/results/holdout_gpu_validation_20261023_01`。使用 warm1500 完整模型/optimizer恢复、8 个真实 native workers、4096 env、64 真实 prefill、B2048/k32 holdout、原 `wm_validation()` 的 1/8/32 步报告；禁止三个 optimizer 的 step、W&B 关闭。

GPU 实测与独立 review：

- 新采集 262144 条真实 transition；B2048/k32 抽样 fallback 实际调用1次，55766 active intervals 全属 holdout，terminal prefix mask 保留。
- 原 `wm_validation()` 另外抽样，1/8/32 有效评分样本分别 55194 / 1857 / 1514，诊断 finite，三个 optimizer step 均为0。
- sampler 与诊断执行验收通过。原质量 gate 唯一 blocker 为 **1-step release phase**：887 样本，learned normalized error `0.47411558`，zero increment `2.10667920`，constant velocity `0.33293369`；尚未优于两条 baseline。其余整体/phase baseline、接触覆盖及三个 pair 的 Brier、precision、recall 满足原门槛。未降低门槛。
- 左/右真实接触 positive 5168/6693、negative 50026/48501；32步 contact/onset/release 252/13/20。计数含重叠及有放回样本，不能当独立 motion 或 episode 数。32步归一化 learned error216.81虽优于zero518.34/CV802.54，不能称绝对准确或 HOI 成功。
- parent peak allocated `6474964992` bytes、reserved `7407140864`；报告末 all-process GPU used `29600645120 / 47590277120` bytes（不是同步全程 GPU 峰值），probe loop217.315s。无 OOM、非有限数据或碰撞容量错误。原有 Warp cast/COM 性能 warning 在旧 source11 日志同样存在，真实 transition 与诊断有限性已另行通过。

随后按授权提交两个正式续训，均先验证 probe `COMPLETED0:0`、checkpoint SHA、source177个文件/辅助脚本哈希、旧 producer 结束、完整 config/native preflight 与独占提交 claim：

| 版本 | 新 job | 起点 | source12 下全新输出 |
|---|---:|---:|---|
| warm-start |1460984|1500|`results/hoi_v1_warm_resume_20k_20261023_01`|
| 从零初始化对照 |1460989|500|`results/hoi_v1_scratch_resume_20k_20261023_01`|

提交回执 `source12/data/warm_submission.json` / `scratch_submission.json`；首次提交时均正常 PENDING(Priority)。最初 warm helper 在提交前查询已过期旧 jobID 时 `squeue -j` 返回 Invalid job id，未调用 sbatch、未写提交 claim；只把观测改为成功读取整个用户队列后精确匹配 ID，再唯一提交。该辅助查询修正不涉及冻结源码或科学参数。

两版恢复后的真实 holdout、优化器、显存与 W&B 状态待核实后追加。GPU probe 只验证 warm1500，不替代 scratch500 的实际恢复后诊断。若原质量 gate 仍关闭，保持关闭并报告真实 blocker，不放宽阈值来制造 policy 更新。

## source12 恢复后观测与真实 OOM

2026-10-24 01:19 UTC（纽约10月23日晚）已核实：两版均 strict resume，8 native workers/4096环境、原配额及64 fresh replay prefill通过。两个新 run 的 API 读回项目 `PRIVATE`、host为空，数值指标已上线：

- warm1460984：[W&B](https://app.forge.coreweave.com/tianzizh-university-of-pennsylvania/SuperMimic_HOI_v1_20261023/runs/d943cc55c0384eb7ba3ab8fd1b9004d5)，首1501为10 WM step、0 policy；holdout finite，1/8/32样本55131/1859/1524，唯一质量 blocker 为 release1（learned0.589319/CV0.437545）。后续1551仍同一 blocker（0.541563/CV0.410350），无新执行错误，WM继续。
- scratch1460989：[W&B](https://app.forge.coreweave.com/tianzizh-university-of-pennsylvania/SuperMimic_HOI_v1_20261023/runs/a0866a6133a145d1b8fb3da320419efc)，首501 **原gate passed**，真实10 WM、1 actor、20 critic mini优化已完整返回；pi_loss=-0.969903、v_loss=0.072829、pi_gradnorm=0.139816均有限。holdout 1/8/32样本41379/1722/546，各phase实际覆盖。随后下一轮 **critic target-value 全量前向真实 CUDA OOM**，job `FAILED1:0`，不能称稳定训练。

完整调用栈：`train_loop→policy_update→critic_update:217→Critic.forward→ShapeEncoder.forward→run_mlp net(x).float()`。全量135168×128点×128feature的float32临时需要 **8.25GiB**，失败时GPU总44.32GiB/global free7.52GiB，parent allocated10.92GiB、reserved但未allocated5.20GiB。第一次全轮成功不证明随后分配可持续；现有cache/碎片状况与较大连续分配共同构成压力，不能把全局free简单当可用峰值余量。失败输出、worker资源与完整trace均保留，scratch没有新的完整checkpoint，恢复来源仍是原500。

本地只读凭证及JSON回拷在ignored `data/hoi_v1_resume12_20261023_01/`，不含模型权重或完整rollout。W&B把SDK finish显示finished时，`run/completed=false`及本地`tracking/live_finish.json status=failed`、Slurm FAILED才是本次实际结论，不能以UI finished认定20k完成。

## source13 最小等价修复与下一阶段

真实OOM后选择有界计算修复：仅将 **no_grad critic target forward** 按现有最大训练mini行数 `ceil(H*B/critic_minibatches)=32768` 分为 `[32768,32768,32768,32768,4096]`，保留全部135168行及顺序，拼回相同lambda递推输入。full normalization仍一次更新135168行，actor BPTT、PointNet结构/精度、5×4优化mini、optimizer、随机序列及全部科学参数不改；没有缩环境/batch/H，也没有改gate。

逐样本MLP/固定RunningNorm/每对象点amax没有跨batch依赖，因此数学目标相同，CUDA GEMM分块不保证逐位相同。新增CPU回归用真实critic/PointNet连续两轮对照旧整批实现，校验values、实际反传MSE targets、loss、模型参数、Adam状态近似等价；归一化状态、原始mini顺序与RNG一致，另测生产135168全行调度和20个32768行训练mini。独立review确认无P1/P2。

新GPU诊断使用原生产train_loop与config，scratch500恢复、64真实prefill、原holdout gate，不修改迭代/科学参数；需连续3次完整actor/critic更新、30 WM/3 pi/60 critic step、三个模型parameter-only hash变化，且target单块≤32768/总行数不变才通过。诊断W&B关闭、独立输出，不称正式20k。失败阶段/显存与已完成step也写独立凭证。此时尚未以真实GPU完成验证，不提前宣告OOM修复成功。

CPU 最终回归 **219/219 passed / 26.795s**，日志 `tmp/hoi_v1_source13_cpu_tests_v02.log`。首次全量218项有一项测试容差失败，证实不是learner语义错误：Torch2.7 CPU整批/分块values最大差1.19e-7、实际MSE反推targets差5.96e-8，40个Adam mini累积参数最大差9.55e-6（Torch2.14原容差通过）；随后新增float64 critic严格两轮对照，values/targets/参数/Adam最大差1.11e-16，阈值1e-10。保留FP32 values/targets/loss/Adam检查及raw输入/normalizer/RNG完全相同检查，FP32参数依据实测设置2e-5绝对界。learner代码没有为测试改动，独立review无剩余P1/P2。两版本CPU targeted均3/3通过。

source13 已冻结并部署：`/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_13`，179文件逐一核实；archive SHA256 `e51804fbffd4eeeaefac2ebf3015cf2284c1b170bb350b28e03cd26c76941a31`，executable content SHA256 `12aa6471e4246b71af940c5ed6f60d681741036217fe14ac5c874efc2c438897`。相对12唯一执行差异为learner，buffer修复SHA不变。scratch500实际CPU preflight通过，plan SHA256 `c12d2354215bd64c68fa13187f8792539d252f848a99af4148bf5ee35c19c8f2`，完整config/native gate均不变。

独立辅助probe SHA256 `65be7237934712e6efaa3fc92193051a9d2f517bcd3d41888afb92a5cfa55d6a`，launcher SHA256 `77a63ae63a9292c5a766f41c397c50b21257b8948ca938583aa3c3aea9e96481`。GPU多轮验证唯一提交 **1461093**，comment `hoi-v1-probe-source13-20261023-01`，输出 `source13/results/stability_gpu_validation_20261023_01`，首次PENDING(Priority)。只有真实报告通过且sacct COMPLETED0:0，才允许source13正式续训。正常source12 warm仍在继续WM，不提前取消。

## source13 连续 GPU 验收与正式恢复

**1461093 已 COMPLETED0:0 / 7m46s，独立review最终验收通过。** 原生产train_loop、4096/H32、真实64 prefill、原gate passed；501/502/503连续三轮全部完成，真实WM/pi/critic step数 **30 / 3 / 60**。三模型parameter-only SHA均改变，排除了仅normalizer buffer变化的假阳性；各轮pi/v loss和梯度均有限。full normalization为 `[135168]×3`，no_grad target为 `[32768,32768,32768,32768,4096]×3`，完整行数/顺序与20个训练mini不变。无OOM、非有限数或碰撞容量错误。

parent CUDA peak allocated `23192899584` bytes、reserved `24863834112`；末次全进程used `45604012032 / 47590277120` bytes，非同步全程峰值。观测器的`MagicMock.call_args_list`额外保留三次normalizer的六块raw输入，每次 `135168×1150×4=621772800` bytes，解释了allocated约0.62GB/轮的增长。生产没有该mock，不能称生产内存泄漏；这次成功含额外观测保留压力，内存值不能直接当正式训练的精确峰值。独立review核实该来源，无需仅为去掉保守观测压力重复已成功的三轮GPU优化。

source12 warm在1801实际质量gate通过后，**相同旧full-target分配真实OOM**，1460984 `FAILED1:0 / 21m26s`。trace仍是critic target forward/8.25GiB/globalfree7.82GiB；最后完整metric为1800 WM3000/policy0。调用栈位于critic_update，actor step可能已执行但整次policy/critic更新尚未返回，不能按日志0断言其内存中从未执行过actor optimizer，也不能算一次完整成功更新。没有新的checkpoint，所以这段WM与未保存优化不能恢复；最新完整warm仍1500。此前保留其运行是为了等待下一完整checkpoint，失败后按真实结果重选，而非假称进展已保存。

重新检查旧producer均已退出，完整载入并复核所有候选checkpoint、schema/config/data/physics/模型finite和重复SHA，最终仍选原source11 warm1500 / scratch500。回执 `source13/data/warm_checkpoint_for_resume13.json`、`scratch_checkpoint_for_resume13.json`，本轮无额外scancel。GPU通过与sacct终态核实后唯一提交：

| 版本 | source13正式job | 恢复iteration | 独立全新输出 |
|---|---:|---:|---|
| warm-start |1461144|1500|`source13/results/hoi_v1_warm_resume_20k_20261023_01`|
| 从零初始化对照 |1461178|500|`source13/results/hoi_v1_scratch_resume_20k_20261023_01`|

首次均PENDING(Priority)，正常排队。独占提交claim/receipt在 `source13/data/{warm,scratch}_submission{.request,}.json`，没有重复producer或覆盖旧输出。所有参数仍与原checkpoint完整config字典相同；新W&B run沿用已确认PRIVATE项目。正式恢复后状态、holdout、实际优化与显存待观测追加。

## source13 正式 scratch 的后续 backward OOM 与 allocator 验证

2026-10-24 01:50 UTC 核实，scratch1461178 `FAILED1:0 / 3m20s`。首501完成WM10/pi1/critic20、原质量门passed；下一轮进入critic训练mini的 `loss.backward()` 时申请4.00GiB失败。完整trace为 `train_loop→policy_update→critic_update:238→loss.backward→autograd`，不同于已修复的full-target 8.25GiB分配。失败时GPU44.32GiB、globalfree2.71GiB、parent allocated12.26GiB及reserved-unallocated8.67GiB。旧输出/日志保留，未产生新的完整checkpoint；最新scratch仍原500。W&B finished不表示20k完成。

warm1461144仍在同一不可变source13运行，最新完整metric1600为WM1000/policy0，1601真实holdout 1/8/32有效样本51570/1807/1295、finite，release1 learned0.37959尚未优于CV0.31136，其余原门槛保留。观测GPU32004/46068MiB，没有OOM。先保留健康producer及其WM进度，验证通过后从当时最新完整checkpoint接续；不提前取消来丢弃尚未保存的进度。

三轮诊断观察器的额外tensor保留改变allocator布局，因此即使其已分配内存更大，也不能据此证明无观察器的正式训练具有更低碎片峰值。此前三轮成功仅对该次实测有效；正式第二轮失败是扩大验证的真实依据，不是生产tensor泄漏的证据。

本轮保持source13全部179个快照文件和科学配置不变，仅在新launcher给parent及8个CPU-IPC worker设置相同的 `PYTORCH_ALLOC_CONF=expandable_segments:True` 和兼容别名 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。不升级或写入共享runtime。集群实际Torch2.11.0+cu128/native支持该实验性选项；[官方CUDA内存文档](https://docs.pytorch.org/docs/2.11/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf)与[2.11 allocator源码](https://github.com/pytorch/pytorch/blob/v2.11.0/c10/cuda/CUDACachingAllocator.cpp)供溯源。现有跨进程传输为CPU张量，原生CUDA graph关闭，无CUDA tensor IPC/P2P/MemPool用法；设备是否实际启用VMM必须以GPU `memory_snapshot()` 的 `is_expandable` 段计数>0确认。

新独立probe v02使用普通函数观察器（`patch.object(new=...)`），只保留数值/参数hash，不保留输入tensor。恢复原scratch500、64真实prefill、原生产train_loop和gate，要求连续50次完整actor/critic更新、真实pi50/critic1000、WM每轮10次、normalizer每次完整135168行、target每次完整5块/总135168行、三模型parameter-only hash均变化。最多103原生产迭代，不降低质量门；每5轮写数值进展，诊断时限2h，W&B关闭，既不保存新实验权重也不运行正式20k。只有报告通过且Slurm COMPLETED0:0，才允许用新输出和新W&B run分别续训；gate关闭则如实报告，不强制policy更新。

新GPU诊断已唯一提交 **1461203**，comment `hoi-v1-probe-source13-expandable-20261023-01`，输出 `source13/results/stability_expandable_gpu_validation_20261023_01`。截至01:52 UTC已在 `ll-l40-0.grasp.maas` RUNNING（同一失败节点，实际分配的L40 UUID与失败任务不同），此时仍初始化，不能称验证通过。v02 probe SHA256 `86e97c6d34d06710f05fd5ef6e98c56d9b7a68e17ca723f43c49d6072d2dbb55`；新launcher SHA256 `15f9cb0d4f796367236a21d6dad03aad2cc60b88e28712db2f85c001fbd32632`。stage与submit/stop/status均经独立review无P1/P2，语法检查通过；新preflight plan与原source13 hash一致。辅助脚本及`resume13_expandable_staging.json`独占写入data/，没有重写旧probe/launcher或快照文件。

01:59 UTC进一步核实：旧warm1461144也已 `FAILED1:0 / 19m50s`，1751原gate首次passed后在critic mini `loss.backward()`申请4GiB真实OOM（globalfree2.62GiB、parentallocated12.26GiB、reserved-unallocated8.76GiB）。最后完整metric1750为WM2500/policy0；完整policy_update尚未返回，不按其计数推断actor内存中从未step。没有新checkpoint，最新warm仍1500；与scratch相同，本轮无需取消正常producer，两旧job均已实际失败。此证据再次说明source13 target分块与allocator碎片问题是不同层面。新expandable诊断45轮已完成且无OOM/retry，但尚等50轮与Slurm终态。

## expandable 的50轮最终验收与两版正式恢复

**1461203 COMPLETED0:0 / 8m13s**，报告loop420.825s，独立数值/hash/源身份review验收通过。严格501–550连续50个完整policy_update返回，实际WM500/pi50/critic1000；norm135168×50且两个RunningNorm的count均增加6758400；250个target块逐组32768×4+4096，完整行数和顺序不变。三模型parameter-only SHA均改变，全部数值递归finite，1/8/32有效holdout41495/1726/565，原gate passed。观察器无tensor保留。

实际native allocator初末均2个expandable segments，零OOM、零allocation retry；完成各轮allocated恒定6161698304bytes。parent peak allocated21737799168bytes（20.245GiB），reserved25505562624（23.754GiB）。末全进程47405268992/47590277120bytes（44.150/44.322GiB），driver free176.44MiB，同时parent reserved-unallocated约18.015GiB可回收；末used不是全进程同步峰值，不能只据free推断OOM。该50轮是本次连续分配验收，不保证20k长期稳定或HOI收敛。

GPU前置通过后再次独立载入所有候选checkpoint，校验schema/config/data/physics/全部模型finite/三optimizer与重复SHA，确认两旧producer已FAILED且无更晚完整checkpoint。新回执在source13 `data/{warm,scratch}_checkpoint_for_expandable.json`，本轮没有scancel活跃任务。正式唯一提交：

| 版本 | 新job | 恢复全局iteration | source13独立输出 |
|---|---:|---:|---|
| warm-start |1461248|1500|`results/hoi_v1_warm_resume_expandable_20k_20261023_01`|
| 从零初始化对照 |1461253|500|`results/hoi_v1_scratch_resume_expandable_20k_20261023_01`|

提交claim/receipt在 `data/{warm,scratch}_expandable_submission{.request,}.json`；两次均先检查179源码hash、辅助脚本hash、原producer终态及完整config/native preflight。20k总目标、每500保存、每5000完整canonical80、LR decay250000、4096/H32/生产batch及其余科学参数逐项不变；只有运行时allocator选项新增。两版新W&B run与真实运行验收结果待下文更新。旧失败调用中的policy_update未完整返回；每50步日志不能确定失败前其他调用的实际完成次数，不能把旧last metric的0当精确总数。

02:13 UTC正式恢复观测：scratch1461253已经550，恢复后WM500/policy50，pi_loss=-1.1488843、v_loss0.03622528、pi_gradnorm7.3430972（原clip前梯度范数）均有限；551新holdout 1/8/32有效35513/1523/426，原gate再次passed。GPU采样45188/46068MiB，无OOM/非有限数/碰撞容量错误。独立只读资源review同时确认parent和8 workers存活且20秒CPU计数全部增长，P0/2490MHz，501→550日志空窗是计算和原50iter日志节奏，不是卡死。warm1461248此时1750，WM2500/policy0，1751 holdout finite但release1 learned0.61695218未优于CV0.60705239；不改门槛，继续等待warm实际policy更新。

新W&B API读回两个run均running、项目PRIVATE、host为空，数值配置/指标同步，weight/source/raw/rollout/artifact上传均关闭：

- warm1461248：[run 91858e55](https://app.forge.coreweave.com/tianzizh-university-of-pennsylvania/SuperMimic_HOI_v1_20261023/runs/91858e5503124a2aabfe51cfd39cd2f2)。
- scratch1461253：[run c21c0d4a](https://app.forge.coreweave.com/tianzizh-university-of-pennsylvania/SuperMimic_HOI_v1_20261023/runs/c21c0d4acff54c06ac64c98f54187977)。

上述50为本次fresh replay之后的完整policy_update计数，不是全部历史总数；每次包含一个actor step和原5×4 critic mini step。API观测不init新run，不访问或上传模型文件。两正式plan的完整config、source identity、native gate、checkpoint iteration与提交的两个allocatorflags已另外本地逐项比对，全部相同/通过。

## 正式恢复后运行验收结论（02:24 UTC / 纽约10月23日22:24）

本轮部署、checkpoint恢复与实际完整优化执行已完成检查，两个正式producer继续运行。独立agent直接读回1900与1901文件后确认，warm已经完成1851–1900的50个完整policy_update，不能再描述为「只有WM、从未完成actor更新」。当前两版如下：

| 版本 | job / 状态 | 最新完整日志iter | 恢复后WM / policy计数 | 最新holdout |
|---|---|---:|---:|---|
| warm |1461248 / RUNNING|1900|4000 / 50|1901 finite，1/8/32有效50737/1776/1261；原gate因release1与onset8重新关闭|
| 从零对照 |1461253 / RUNNING|600|1000 / 100|601 finite，1/8/32有效37403/1585/490；原gate passed|

warm1900 pi_loss=-6.06387997、v_loss=0.34661608、pi_gradnorm=3.47986317、imagined reward/step=0.08245610、survival=0.80230021均有限；随后1901 release learned0.32648838未优于CV0.29288736，8-step onset也未优于两基线，因此后续policy按原规则暂停，WM继续。warm通过门→实际更新50次→新数据诊断后重新关闭是本次真实状态，不把执行验收等同于质量门一直通过或HOI成功。scratch600 pi_loss=-1.37779665、v_loss=0.03785836、pi_gradnorm=2.85671616、imagined reward/step=0.04231199、survival=0.41929382均有限，601门通过。

两卡当前采样约45189/45188MiB used（各total46068MiB），日志无OOM/非有限梯度/碰撞容量错误；显存高包含可回收allocator cache，前述50轮报告给出了actual VMM、peak allocated、reserved及零OOM/retry的独立证据。formal worker均为8组真实全量quota/4096，64 fresh prefill，无batch/H/env降配。独立资源进程检查与数字指标相互印证，不能只凭RUNNING判训练正常。

W&B API 02:24读回两个新run均PRIVATE/running，`last/train/iteration`与`last/train/policy_updates`分别1900/50、600/100，全部上传数值有限；host为空，未创建观测run。原log_every=50保留，因此UI最近一次1900的gate=true不是1901最新gate的实时状态；最新质量门以完整`wm_gate_001901.json`为准，下次记录再同步更新，不改日志频率制造更密数据点。

全部训练config逐项与原checkpoint相同，source13内容SHA仍为12aa6471e4246b71af940c5ed6f60d681741036217fe14ac5c874efc2c438897。20k全局目标、每500保存、每5000独立GPU完整canonical80和LR decay250000保持；没有test40/消融、额外正式producer、commit/push、共享runtime或历史结果改写。原Slurm24h上限与no-requeue也保留；20k为总目标，不保证单次24h内跑完，本轮不额外安排自动重启。checkpoint缺少replay/RNG/模拟器状态，fresh replay恢复仍非逐位连续。

元数据回执保存在本地ignored `data/hoi_v1_expandable_20261023_01/`（最新完整JSON回拷022409、W&B读回及022423独立状态），远端对应source13 data/中的exclusive staging/submission/checkpoint审计回执。只传元数据，不含权重或完整rollout。后续观测应读取此处两个真实producer及最新日志，不重复运行已完成GPU诊断，也不据旧失败run的SDK finished认定20k完成。
