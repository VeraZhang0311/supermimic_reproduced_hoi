# HOI v1 全部从零初始化对照

用户在确认首次实验使用24k motion checkpoint warm start后，要求另跑一版所有网络从零学习；随后要求核对全部节点的GPU可用情况。此为新的独立初始化对照，不覆盖首次实验。

最新迁移：用户已授权部署 holdout 采样修复并续训。原job1460725保存完整500 checkpoint后已停止；source12 GPU采样/诊断通过，scratch续训job1460989已提交，仍保留从零初始化来源与全部原参数。实际恢复后 gate/policy/W&B 检查见 [source12 部署与续训](hoi_v1_holdout_resume12_20261023.md)。本页其余段落保留首次启动历史。

当前恢复：source12的target OOM已以数学等价分块修复；source13旧scratch1461178随后仍因critic backward碎片OOM退出。source13+expandable allocator经219项CPU回归及连续50次完整GPU优化1461203验收，现正式scratch为 **1461253**，从最新完整500恢复。旧run的SDK finished不等于20k完成；实际holdout/policy/W&B验收见同一迁移记录。

后续只读核查：iter300累计3000 WM更新、policy更新0（仍在500 warmup内）；与warm-start相同的holdout拒绝采样预算缺陷已本地复现并准备修复，尚未部署或改变本任务。参见[reset/contact与holdout核查](hoi_v1_reset_contact_and_holdout_audit_20261023.md)。

## 集群核对与提交

2026-10-24 00:28 UTC（当地10月23日晚）先核liu-compute的3台L40，再扩展至全部Slurm节点、各分区、实际CfgTRES/AllocTRES及本人association。账号可用liu-account的ll-med和normal；48GB及以上的L40/L40S/A40/A6000/Blackwell当时全部已分配。少量2080Ti/A10未分配卡不能容纳已验证的4096/H32约45GB配置，另有3090节点DRAIN/INVALID_REG。不能把CPU空闲或MIXED当GPU空闲。

**已唯一提交job1460725，sm-hoi-v1-scratch20k，comment hoi-v1-scratch-20k-source11-20261023-01。** 初始PENDING Priority；00:32 UTC已获分配并RUNNING于ll-l40-2.grasp.maas，独立L40 UUID `GPU-80a38e56-6386-a7e1-5a1d-4335ef5616ab`。原warm-start1460279仍RUNNING，使用另一UUID `GPU-7d5c52cf-73b7-8d86-9710-c0c12170c2e1`。

- 执行源码保持冻结source11不变：`/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_11`；archive SHA256 `c9ff63e63310442c4815023bdb89449c6fb03cf77e42d96c04e1e7b694a3b8c0`，全部174源文件逐一核对。
- 源码content SHA256 `9c17eccb6f92ce6cdc4b08db5c16ba28a5541ca92c74767bcfb0568e37fe2bd8`，与warm-start完全相同。
- 输出：`..._11/results/hoi_v1_scratch_20k_20261023_01`；日志：`..._11/hoi_v1_scratch_20k_20261023_01.1460725.slurm.log`。
- 不修改原checkpoint专用cluster launcher。独立one-off脚本放在data目录：`..._11/data/hoi_v1_scratch20k_20261023_01.sbatch.sh`，SHA256 `eac3a6f647f9dfef847bf4edc0baf803a8222ffc0c2988e5b4d8b6231bf97f1e`；单独记录其identity，不混入或改变冻结package source identity。
- Exclusive提交claim/receipt：`..._11/data/scratch20k_submission.request.json`和`..._11/data/scratch20k_submission.json`。没有重复提交、取消、重启或修改warm-start任务。

## 对照合同与验证

仅初始化方式改变：显式`--from-scratch`，没有`--init-checkpoint`、`--resume`、legacy flags或physics-change flag。Actor、WM、Critic及各自PointNet/新旧normalizers全部fresh constructor，三个Adam、replay、history、iteration和contact counters全新。WM原构造器的机器人零增量输出初始化保留；“从零训练”不意味着所有参数都设为零。

以下与warm-start逐项相同：全train2651/eval80、原生bound evidence、源码、robot MJCF/资产/runtime、hand ON/self ON、seed0、4096环境、replay1000、H32、WM2048×8/10 batches、actor4096、critic5×4、64prefill、前500 iter WM-only及原train-only gate、LR decay250000、max20000、checkpoint500、5000/10000/15000/20000独立GPU冻结完整80条评估。仍为1L40/24CPU/96GB/24小时，OMP/MKL=3；不改变训练参数来适配小卡。

CPU preflight路径`..._11/results/scratch20k_preflight_20261023_01`，plan SHA256 `bb539915e82df28fbaaa8b49f86a221c89c16152f9dff1a2d30941ec509e73d4`。config/data/source/physics/seed/schedule/tracking/evalplan逐项等于warm-start，严格native gate passed，CPU没有worker/optimizer/GPU步骤。另用生产三模型constructor检查：pi4个、wm4个、critic2个RunningNorm均count0/mean0/var1；三Adam state entries均0。训练/周期评估22项CPU检查、local及remote bash-n通过。

新W&B run在同一已确认PRIVATE项目`SuperMimic_HOI_v1_20261023`创建，fresh UUID/resume never，WANDB_SYNC=1，同Forge endpoint和独立SDK overlay；尚待启动回执确认run ID与首条指标。只允许既有白名单配置/数值/评估汇总，不上传源码、权重、原数据或rollout。

后续实际run已确认：[HOI-v1 scratch training](https://app.forge.coreweave.com/tianzizh-university-of-pennsylvania/SuperMimic_HOI_v1_20261023/runs/2d8ab91554f54d10b7d3c3b4ca5dd06e)，ID `2d8ab91554f54d10b7d3c3b4ca5dd06e`，不是warm-start的132d6c3e run。

## 实际启动观测与审查补充

00:33 UTC：`initialization.json`实际为`explicit_from_scratch`；plan的init/resume/legacy均null、from_scratch=true。八个native worker均ready、4096配额770/592/444/128/449/495/770/448，实际replay4739072000bytes。当前无OOM、overflow或非有限数报错。prefill与真实optimizer更新尚待完成，不能仅凭RUNNING称训练健康。

独立review对scratch语义、相同科学配置与周期eval路径无P1/P2。最终补充指出one-off脚本v01有latent P2：三个环境变量guard用`&&`串联，bash set-e不保证前两项失败时退出。反馈到达时1460725已提交；不覆盖已冻结launcher，不为未发生条件取消正常任务。已核实际sbatch client和export没有INIT_CHECKPOINT/API key/identity token，且实时只读检查实际learner及八native worker的初始environ均无这些变量；因此本任务未触发该缺陷，也未继承checkpoint或向native进程暴露凭据。W&B core在native spawn之后由SDK创建并持认证是预期行为，不能概括成所有子进程都无凭据。安装的Slurm本机官方man注明显式export列表可能get-user-env，因此实际进程检查比只查命令更强。防御修正仅生成新本地future v02脚本（三个独立test），未部署、未替换本任务，也没有改变训练。

本地ignored观测和回执保存到`data/hoi_v1_scratch20k_startup_20261023_01/`。后续追加真实prefill、WM更新、显存与W&B读回后，才给首次健康结论；不声称20k收敛或HOI成功。warm-start含额外24k motion预训练预算，二者不是相同总计算量比较。

## 首次启动健康确认（2026-10-24 00:35 UTC）

**scratch1460725的预填充、真实首轮WM优化和实时W&B均已通过启动检查。**

- 64 vector steps、262144真实transition的fresh prefill完成；第1 iter累计270336 sim_steps，10次实际WM optimizer.step、policy更新0；前500 iter WM-only及原gate保持。
- 首轮10 batches数值均有限：平均total20.564456、robot6.722418、object13.758052、contact0.839858。真人体normalized reward0.392472、总HOI0.384816；仅作初始化观测，不与已多训一小时的warm-start当前曲线冒称同进度比较。
- 独立W&B API读回run running、project PRIVATE，上述iter1、wm_updates10和所有分项已同步。可选文件列表接口仍TypeError，不将其当训练失败或声称远端文件列表已审核。
- 同allocation GPU采样：UUID80a38e56、32421MiB used/46068MiB total（约31.7/45.0GiB），free12964MiB、utilization99%。无OOM、非有限数、contact/constraint overflow报错，两个训练job/batch均仍RUNNING。
- 实际learner+8 native workers+resource tracker共10个Python进程的初始environ均无API key/identity token/INIT_CHECKPOINT；单独wandb-core在worker spawn后有认证，符合SDK通信需要。
- future-only v02 guard已通过3项合成禁止变量拒绝测试、1项clean environment通过测试及bash-n，未部署或改动v01。

首次500 checkpoint和5000周期评估尚未到达；当前只确认初期WM更新健康，不声称actor更新、20k稳定完成、收敛或HOI成功。任务继续运行，未安排重复producer或改动旧runtime。
