# 当前开发文档

开发仓库：`D:/supermimic_reproduced_hoi`，分支 `codex/hoi-v1-no-quality-gate`。后续开发以无持续质量门、完整手脚碰撞、五通道contact head的新版为基线；固定500 iteration WM-only之后正常更新actor/critic，质量比较保留为诊断。原始fork分支只作溯源。

| 文档 | 用途 |
| --- | --- |
| [v1最终设计](supermimic_hoi_v1_final_design.md) | 现行网络、数据时基、接触、奖励、物理与训练评估契约 |
| [ULTRA数据与时间基准交接](ULTRA_DATASET_AND_TIMEBASE_HANDOFF_20261023.zh-CN.md) | 原始数据位置、冻结划分和canonical 60Hz解释的证据边界 |
| [8500pt最终验收](hoi_v1_final8500_eval_handoff_20261025.md) | 当前新版冻结身份、完整HOI80、固定20条视频与本地证据入口 |

先读仓库[DEVELOPMENT.md](../DEVELOPMENT.md)及[AGENTS.md](../AGENTS.md)。操作说明见[HOI v1入口](../hoi/v1/README.md)、[数据适配](../hoi/data/README.md)和[资产/缓存准备](../hoi/v1/PREPARATION.md)。历史阶段handoff和旧版实验记录不作为当前执行指令；Git历史可追溯已经提交的文档。

## 外部HOI参考索引

| 本地仓库 | 固定用途与入口 |
| --- | --- |
| `D:/Ultra` | 原作者G1 HOI路径：`ultra/data/cfg/g1_teacher.yaml` → `ultra/env/tasks/humanoid_g1.py` → `ultra/data/assets/g1/g1_29dof.urdf`；整脚STL、原VHACD recipe。`ultra/env/tasks/ultra_g1_retarget.py` 为630D consumer，`ultra/env/tasks/ultra_g1.py` 为当前exporter；当前exporter不能反证发布数据真实时钟。 |
| `D:/Protomotions` 的 `hoi/dagger_distillation` | 固定selection/source-group划分、canonical80/test40、连续时间采样与39体PHC参考；`protomotions/envs/hoi/ultra_reference.py`、`ultra_motion_bank.py`，`scripts/hoi/dagger_evaluation.py`，`notes/ULTRA_HOI_FROZEN_RUNBOOK.zh-CN.md`。不自动恢复该仓实验或迁入teacher/DAgger训练。 |

外仓具体历史commit、清单SHA与数据路径见最终设计及数据交接。当前足部采用同源STL+CoACD5hull/16vertex，不能称为与作者PhysX VHACD逐位相同。Learned contact预测五项，reference接触目标与奖励仍为双手；原zero-shot入口保留。
