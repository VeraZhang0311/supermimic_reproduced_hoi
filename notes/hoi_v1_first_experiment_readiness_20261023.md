# HOI v1 首次实验就绪记录

后续状态：用户已明确授权正式实验，source11的job1460279已启动并通过首步WM优化及私有W&B在线指标检查。实际运行身份与观测见[第一次正式实验记录](hoi_v1_first_experiment_20261023.md)。下文保留验收完成时的历史状态和未执行的source10命令。

**结论：用户要求的全量缓存、八资产原生门、完整数据4096/H32显存与64步预填充诊断全部通过。首次实验已准备好，正式训练尚未提交或启动。**

最终观测与凭证汇总时间：2026-10-23 22:44 UTC（18:44 EDT）。分支`hoi/v1.0`。本记录覆盖准备条件，不把容量诊断称为稳定20k训练、收敛或HOI成功。

验收完成后的本聊天后续检查`hoi-v1`已按用户要求停止（automation状态PAUSED）。本文中的未来启动命令已提取后用`bash -n`检查通过，未执行；没有正式训练job id。

## 已完成的验收

| 项目 | 实际证据与结果 |
| --- | --- |
| 源数据及60→50Hz | 2651/2651 SHA/schema/整数时钟审计通过；冻结canonical60Hz协议，不冒充真实导出时钟已证明。 |
| 全量训练缓存 | Job1459845 `COMPLETED0:0`/1:21:48；2651 clips、381 groups、756524 states、753873 controls，非subset。 |
| 全量独立CPU审计 | Job1459949 `COMPLETED0:0`/1:16；所有source/cache/质量/共同几何合同通过；8组local→global映射逐张量一致，环境配额770/592/444/128/449/495/770/448，总和4096。 |
| 完整bundle八资产原生门 | Job1459914内全部8组真实8环境/64步检查、实体pair正负对照、四子步冲量标签、物理readback、独立GJK/FK及几何一致性通过；严格聚合gate passed、无blocker。root及preflight复核原报告字节/SHA。 |
| 完整数据4096/H32容量 | Job1459914 `COMPLETED0:0`/25:25；完整replay1000、实际全reference和配额、WM2048×8 fully active backward、actor4096×32 backward、critic135168 target前向与32768 mini backward均通过。 |
| 64步生产预填充补验 | Job1459951 `COMPLETED0:0`/6:12；262144条真实transition、全部9个资源phase通过；复用同bundle成功native证据，新增native校准0、optimizer step0。 |
| CPU与旧入口 | source09全套197/197、无skip，原HOI边界另6项通过；source10全部112份Python代码与09逐字节相同。旧zero-shot入口保留。 |
| 最小真实管线 | 64步prefill、2次WM优化、checkpoint保存/重载、220步/221状态完整冻结评估通过；policy更新0，不称HOI成功。离线重放221/221远端与本地独立解码、root图像QA通过。 |
| 最终CPU preflight | source10严格native gate和20k/500/5000生产配置通过，170份源码及8份原生输入报告字节校验通过，`training_started=false`，无worker/prefill/checkpoint。 |

独立资源review发现的CPU预算P2已最小修复：launcher尊重显式OMP/MKL、未设时默认3；shell语法及默认/显式线程预算检查通过。首次实验显式使用两项3线程，与GPU诊断一致。仅launcher和说明改变，所有Python模型、learner、物理、数据与优化数值未变，09及其原生结果没有覆盖或重跑。

## 冻结身份与显存

- 最终执行源码：`/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_10`，archive SHA256 `ebbcd611b4c30a500b3730e683f2821d5fa4f1d272c257ed6dbca225cea3fdca`。
- GPU验证源码source09 SHA256 `b03e3df856a4848b7d96f852a1cc5782937eb7de8e7999efdac35f7db31c5184`；全部112份Python与source10在远端直接逐字节核对一致。
- train bundle content SHA256 `318482c78f699d33aa5ae7743f2918f19d96fa3d78e2869add50f4a890d022c8`，文件 SHA256 `815ed700b8cda2b382a37087ae43619615a38867325de4c51216c99f0848c157`。
- canonical eval80 bundle content SHA256 `ea182588bca74cc17f9027e9af3184d524467c9042db5e61e2eed3bcacb6f57c`。缓存及共同合同已审，正式的5000迭代评估尚未执行。
- fulltrain native evidence SHA256 `daece1e92ce6eeffdad51f0a0b9fc282c271c7226e45fa9ed30934b3fbc4da8d`。
- 64步资源报告 SHA256 `23c1a1bab8e56187400430bf410ad80cf015bc88c80ba89803f4c4addf3bd7d7`；12步资源报告 SHA256 `f2e1b66c65d44e91c7b5d3ddb721832c8e76e6f6ed5f9bae78fc97b2f0e387b5`。
- 最终preflight plan SHA256 `185ded469e03dfd2c2e5bae06a454627db31406f352417b4b9df970a18d45225`，远端`..._10/results/first_experiment_preflight_20261023_01/`。
- 初始化权重真实SHA256 `e2b6e716c058e5448ab26caa22febc874dd50e2459435f1530f515dcd9fe8d9f`；来源为历史self ON/hand OFF 24000，目标hand ON/self ON。零扩展是新实验初始化，三个optimizer与iteration从头建立，不是resume旧实验。

64步诊断在L40上最大同步全进程GPU used读数为45,086,277,632 / 47,590,277,120bytes（约45.09/47.59GB）；parent allocator精确峰值allocated21,916,311,040bytes、reserved23,376,953,344bytes。全PID的1秒采样峰42944MiB，物理GPU UUID已绑定，无其他job进程。12步独立全量诊断也通过。全局同步读数与1秒采样不等于连续全进程峰值，不能把allocator cache全部当活跃张量。

资源诊断始终0 optimizer step；预留全部参数两份Adam moments，未实际执行foreach scratch/内核。仅测一个完整WM batch、一个critic minibatch，不声称完整10WM/20critic迭代吞吐或20k完成时长。旧gradients、checkpoint深拷贝与WM validation静态内存边界见[独立资源审查](hoi_v1_resource_review_20261023.md)。正式运行仍须对overflow和非有限数立即报错；policy仍先经过500迭代WM-only及原train-only启动门，不保证门一定开启。

本地ignored凭证与可机器读取的最终交叉核验：`D:/supermimic_reproduced_hoi/data/hoi_v1_final_readiness_receipts_20261023_01/readiness_verification.json`，其中`ready=true`，`formal_training_submitted=false`、`formal_training_started=false`。全部资源data identity/config、实际worker specs、有限梯度、源码及原生门SHA已经交叉核对。原始观测在`data/hoi_v1_readiness_monitor_20261023_01/`，没有上传原数据、权重或rollout。

## 首次正式实验命令（已准备，未执行）

配置为20k上限、每500保存、5000/10000/15000/20000各自自动提交完整canonical80冻结评估，原250000 LR decay及其他生产数值不变。独立评估不更新optimizer/normalizer，排队不等于评估完成。默认W&B离线，旧结果、runtime及历史工作区不修改。

仅在用户后续明确启动时，于集群执行以下命令；本轮没有执行`sbatch`。实际入口自带新输出目录保护，不能复用已有RUN_DIR：

```bash
set -euo pipefail
export SUPERMIMIC_ROOT=/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_10
export SUPERMIMIC_RUNTIME=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128
export ISAACLAB_PATH=/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi/IsaacLab
export BUNDLE=/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_06/data/bundle_train2651_graph_20261023_02/bundle.json
export EVAL_BUNDLE=/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_06/data/bundle_eval80_graph_20261023_02/bundle.json
export NATIVE_EVIDENCE=/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_09/results/fulltrain2651_bound4096_20261023_01/evidence/native_evidence.json
export INIT_CHECKPOINT=/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_01/data/initialization/ckpt_self_on_hand_off_024000_e2b6e716c058e5448ab26caa22febc874dd50e2459435f1530f515dcd9fe8d9f.pt
export CONFIG="$SUPERMIMIC_ROOT/configs/hoi_v1_first.json"
export RUN_DIR="$SUPERMIMIC_ROOT/results/hoi_v1_first_20k_20261023_01"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 WANDB_SYNC=0
test ! -e "$RUN_DIR"
/opt/slurm/bin/squeue -u tianzizh
# 再核对实际队列、日志和checkpoint，确保没有同一正式producer后才提交。
/opt/slurm/bin/sbatch --export=ALL \
  --output="$SUPERMIMIC_ROOT/hoi_v1_first_20k_20261023_01.%j.slurm.log" \
  "$SUPERMIMIC_ROOT/cluster/hoi_v1_train.sbatch.sh"
```

launcher当前资源为1×L40、24CPU、96GB、24小时walltime。20k是迭代目标，不由容量检查推导为24小时内可完成。checkpoint每500保存；本轮没有授权正式训练、自动续训、test40或消融。
