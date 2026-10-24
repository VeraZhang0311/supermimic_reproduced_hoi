# HOI v1 第一次正式实验

用户在全量验收通过后明确授权启动：20,000 iterations，每500保存，每5,000自动独立GPU评估完整canonical80，并注册新的W&B项目。此前就绪文档的“尚未授权/尚未提交”为当时的历史状态。

后续用户另要求全部从零初始化对照：独立job1460725已启动并通过首次真实WM优化/私有W&B检查，见[从零对照记录](hoi_v1_from_scratch_20261023.md)。本页保留 source11 的首次启动历史。

最新迁移：用户已授权停止两版、先 GPU 验证修复再从最新完整 checkpoint 续训。source12 GPU 采样/诊断已通过；warm1460279停后采用1500 checkpoint，正式续训job1460984已提交，其他参数不变。实际恢复后 gate、policy 和 W&B 检查见 [source12 部署与续训](hoi_v1_holdout_resume12_20261023.md)。

当前恢复：source12的8.25GiB target OOM已以数学等价分块修复；source13旧warm1461144随后仍因critic backward碎片OOM退出。source13+expandable allocator经219项CPU回归及连续50次完整GPU优化1461203验收，现正式warm为 **1461248**，从最新完整1500恢复。实际holdout/policy/W&B验收见同一迁移记录，旧run不代表20k完成。

后续发现需处理的P1：iter1550的WM更新15500但policy更新0，holdout拒绝采样预算会在合法数据存在时误报coverage不足。本地修复及216项CPU回归通过，尚未部署或改动本运行；详见[reset/contact与holdout核查](hoi_v1_reset_contact_and_holdout_audit_20261023.md)。本页原首次启动健康结论仅覆盖当时的WM更新，不代表actor已训练。

## 冻结实验

- 分支：`hoi/v1.0`；独立执行快照source11，保留source09/10与全部历史目录。
- 配置：`configs/hoi_v1_first.json`。4096环境、replay1000、H32、WM2048×8/10 batches、actor4096、critic5×4、64步prefill、500迭代WM-only、LR decay250000；physics hand ON/self ON。
- 初始化：原hand OFF/self ON 24000 checkpoint SHA256 `e2b6e716c058e5448ab26caa22febc874dd50e2459435f1530f515dcd9fe8d9f`；显式允许本次物理变化，三个optimizer与iteration从零开始，不续训原实验。
- train2651 bundle content SHA256 `318482c78f699d33aa5ae7743f2918f19d96fa3d78e2869add50f4a890d022c8`；eval80 `ea182588bca74cc17f9027e9af3184d524467c9042db5e61e2eed3bcacb6f57c`。
- 原生证据SHA256 `daece1e92ce6eeffdad51f0a0b9fc282c271c7226e45fa9ed30934b3fbc4da8d`；全量4096/H32和64prefill诊断已通过，见[就绪记录](hoi_v1_first_experiment_readiness_20261023.md)。
- 5000/10000/15000/20000保存的checkpoint触发冻结完整80条评估，独立GPU、无optimizer/normalization更新，不用于训练门或超参数选择。
- Slurm：1×L40、24CPU、96GB、24小时上限，OMP/MKL各3；20k为目标上限，尚无全程耗时保证。不自动重启、续训或重复提交。

## 私有实时记录

新项目[SuperMimic_HOI_v1_20261023](https://app.forge.coreweave.com/tianzizh-university-of-pennsylvania/SuperMimic_HOI_v1_20261023/workspace)已创建并API读回`PRIVATE`。原项目不修改。使用既有Forge端点和现有认证；凭据不写入源码、receipt、Slurm export或SDK异常输出。

原只读仿真runtime缺少完整SDK，使用独立tracking overlay `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_tracking_runtime_20261023_01/site-packages`：官方wandb0.23.1/protobuf6.33.5/sentry-sdk2.42.1；安装来源及SHA保存于该独立目录。共享runtime不变。

source10→11只增加实时记录、动态项目/周期eval传递与SDK bootstrap；不改变模型、learner、物理或科学参数。每次实验fresh UUID、resume never；首步和每50步同步白名单数值，checkpoint只同步摘要；禁止源码、权重、原数据、完整rollout和artifact上传，关闭console/git/code/system metadata/stats/requirements/error reporting，固定无路径program标识并清空host。SDK在八个worker spawn后加载，秘钥仅留在learner进程。

本地完整CPU回归212/212、原HOI边界6/6通过；相关targeted41项与独立review通过。实际提交身份、SDK有效设置、preflight和启动健康观测在启动后追加；提交前不把RUNNING称为训练正常。

## 实际启动记录

- **正式job：1460279**，job name `sm-hoi-v1-first20k`，unique comment `hoi-v1-first-20k-source11-20261023-01`。提交前实际用户队列为空、目标输出不存在；exclusive request/receipt防重复。2026-10-23 23:16 UTC提交，随后已分配`ll-l40-2.grasp.maas`。23:24 UTC已确认真实WM优化和在线指标，首次启动检查通过。
- 源码：`/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_11`；archive SHA256 `c9ff63e63310442c4815023bdb89449c6fb03cf77e42d96c04e1e7b694a3b8c0`，174文件逐一验证。source10→11除测试/说明外仅训练launcher、train记录、track、periodic_eval变化，另加bootstrap；learner/网络/物理/科学配置逐字节相同。
- 输出：`..._11/results/hoi_v1_first_20k_20261023_01`；日志：`..._11/hoi_v1_first_20k_20261023_01.1460279.slurm.log`。提交回执`..._11/data/first_formal_submission.json`保存精确命令、显式非敏感export及前置队列。没有导出API key、JWT或`ALL`。
- 实际Linux SDK0.23.1在`update_from_system_environment()`后依然host为空、program/program_relpath为`hoi_v1_metrics`，其RunRecord host/git为空，隐私开关全部生效；检查未调用init、无联网。preflight SHA256 `79b8301c70859e982ecda39a8ecc9e37a2f3d7293d13685e5893970186fcf4fa`，严格全量native gate、2651数据和20k/500/5000配置通过，无worker/optimizer启动。
- 本地ignored启动凭证目录：`data/hoi_v1_first20k_startup_20261023_01/`；保存project registration、preflight verification、submission和后续只读观测。全程状态和W&B run读回结果待确认后追加。

## 首次启动健康确认（2026-10-23 23:24 UTC / 19:24 EDT）

**结论：正式训练已运行，64步预填充、首轮10次WM优化和实时W&B指标均已通过真实观测确认。**

- Slurm job和batch均`RUNNING`，八个native worker报告齐全，4096实际环境配额不变。`prefill.json`记录64 vector steps、262144真实transition、fresh replay。
- `metrics.jsonl`第1 iter记录sim_steps270336、wm_updates10、policy_updates0；10个WM total loss均有限，范围11.6554–13.5848，均值12.515217。每个`wm_update()`经过finite gradient检查并真实执行optimizer.step()，本次不是0step容量诊断。
- 当前为计划中的前500 iter WM-only阶段。首个WM gate因holdout replay窗口覆盖不足保持关闭；不提前绕过gate，500之后也须通过原train-only门才能进行policy更新。未达到500 checkpoint或5000 eval；当前无法宣称这些未来阶段已实际执行。
- 指定allocation的L40 UUID `GPU-7d5c52cf-73b7-8d86-9710-c0c12170c2e1`；23:24:40 UTC采样显存31939MiB（约31.2GiB），总46068MiB（约45.0GiB），全局free13446MiB、utilization78%。这是当前采样，非全程峰值。没有OOM、非有限数或contact/constraint overflow报错；仅有既有Newton COM property性能提示。
- 新私有项目中的[正式training run](https://app.forge.coreweave.com/tianzizh-university-of-pennsylvania/SuperMimic_HOI_v1_20261023/runs/132d6c3e68fb470c9c608a8203312ae6) ID `132d6c3e68fb470c9c608a8203312ae6`。独立只读API已读回project access `PRIVATE`、run `running`，首步iteration1/wm_updates10、各loss/real reward/contact counts均已进入在线summary；host为空。观察脚本没有创建第二个run。
- 可选SDK文件列表接口返回TypeError，因此不声称远端文件列表已审计。实际配置/白名单边界/真实SDK有效设置与源码review均已验证；没有调用artifact、源码、checkpoint、数据或rollout上传。
- `sacct`里的`.0 FAILED6:0`属于最初只读nvidia-smi观测step：请求gpu:0导致没有device可见。随后观测step改用本allocation的gpu:l40:1均成功；训练batch自身仍RUNNING，无失败或重提。保留原观测凭证。

这里只确认启动和初期WM更新健康，不代表20k稳定完成、actor阶段已验证、学习收敛或HOI成功。正式job继续运行；没有重启、取消、修改冻结source11或启动第二个producer。原验收heartbeat保持PAUSED，不会回放其历史禁止正式训练的旧指令。
