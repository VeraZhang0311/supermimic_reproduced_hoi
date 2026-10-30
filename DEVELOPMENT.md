# SuperMimic HOI v1 开发入口

本工作区为用户指定的 **`hoi/privileged_teacher`**，从整理后的 v1 基线
`fd626e752686ef61b3e7e1b604f12412a9d5cd2b` 创建，不包含 v1.1/v1.2 感知改动。
显式配置 `configs/hoi_v1_privileged_teacher.json` 开启无观测噪声的全身特权 actor；
字段、ULTRA 对照及 contact/reward 扩展提案见 [teacher 说明](hoi/v1/PRIVILEGED_TEACHER.md)。
本次实现 actor 链路，五通道接触与现有奖励保持基线；未启动训练。
离线数据管线独立维护在 `D:/hoi_data_pipeline`；来源转换、50Hz expected-contact
标注和标注 Slurm 作业使用该仓入口。本仓接入新数据仍需准备匹配的 bundle、真实接触
记录及 native evidence。数据标注成功不等于机器人已可执行。

本工作区沿用 **`codex/hoi-v1-no-quality-gate`** 的科学基线，以下说明仍适用；
开发、检查与 Git 命令显式指定当前 teacher worktree，不切换其他分支的工作区。
`main` 保留初始fork的复现代码（`a1508d42c9d1699a52e8233ab16d08c488441ab9`），用于对照。
原 `hoi/zero_shot`、`hoi/v1.0` 已全部合入当前分支；整理后删除两个本地及GitHub分支，提交历史仍在当前分支中。

## 当前新版

- 无持续WM质量门。前500轮WM-only，之后按固定日程更新actor/critic；WM holdout质量只作诊断。数据/资产身份、原生物理和非有限值硬错误检查保留。
- 原机器人网络结构保留，追加物体观测/目标/几何及联合动力学。learned contact head输出依次为左手—物体、右手—物体、物体—地面、左脚—物体、右脚—物体。
- hand/self均ON。脚使用与ULTRA同源完整STL、原安装变换和已验证CoACD hull，关闭旧脚球，原质量/COM/惯量不变；不等同PhysX VHACD cooked geometry。
- 五通道真实标签为控制区间pair法向冲量；脚—地面不能混为脚—物体。reference expected-contact及contact-match reward仍只针对双手。
- 冻结train2651、canonical HOI80、60→50Hz时间契约不变。保留独立zero-shot入口和motion-only代码。

设计、数据及外部项目索引见 [notes目录](notes/README.md)。当前验收基点是 **8500pt**，完整80和固定20条视频均通过，见 [8500验收报告](notes/hoi_v1_final8500_eval_handoff_20261025.md)。
本地数值/媒体在 `data/hoi_v1_last_new80_20261025_01/`；checkpoint仅在集群，真实路径与SHA见报告。

## 代码入口

| 工作 | 入口 |
| --- | --- |
| HOI训练 / 评估 / 离线渲染 | `python -m hoi.v1.train` / `hoi.v1.evaluate` / `hoi.v1.render` |
| 模型与固定训练日程 | `hoi/v1/networks.py`、`learner.py`、`train.py`、`config.py` |
| 接触/手脚物理/证据 | `native.py`、`contacts.py`、`foot_collision.py`、`gates.py` |
| 数据与时基 | `hoi/data/`、`hoi/v1/prepare.py` |
| 历史zero-shot | `python -m hoi.run`、`tools/render_hoi_rollouts.py` |

从零训练须显式使用 `--from-scratch`，传完整bundle、匹配五通道/整脚native evidence和新输出。
`cluster/hoi_v1_train.sbatch.sh`当前仍要求INIT_CHECKPOINT，属于warm-start启动器，不能直接用作from-scratch入口。
已验收source15的实际独立启动器保留在集群原位置；下次实验在新checkout准备明确的新命令/输出。整理不启动或续训实验。
保留已验证的allocator配置 `PYTORCH_ALLOC_CONF=expandable_segments:True` 及兼容的 `PYTORCH_CUDA_ALLOC_CONF`，OMP/MKL线程预算均为3；不要在整理启动方式时遗漏这些实际runtime合同。

新版实验显式设置 `WANDB_ENTITY=tianzizh-university-of-pennsylvania`、`WANDB_PROJECT=SuperMimic_HOI_v1_20261023`，保持私有。
未设置环境变量时代码旧默认project仍为 `SuperMimic_Upstream_Reproduction`。只同步配置/指标/汇总，不传权重、源码、原数据、完整rollout或视频。

## 数据与历史边界

`data/`保留当前8500结果、冻结selection/splits、ULTRA足部原输入、source15归档及必要安装输入，见 `data/README.md`。
2026-10-25按用户授权清理旧版与8500以前的checkpoint/HOI80；比较表中的旧数字不意味着旧权重/rollout仍存在。
8500仍引用集群source01/03/05/06/15及原ULTRA/zero-shot输入：不要按目录日期删除这些依赖、缓存或runtime。

`D:/SuperMimic`及selfcollision/handcollision历史工作区与其集群环境不在本仓库同步/清理范围。
新Slurm实验显式SUPERMIMIC_ROOT和全新RUN_DIR/OUTPUT，不覆盖冻结源码/结果；commit/push不自动授权部署、重启或续训。

## 检查

```powershell
python -m unittest discover -s tests -v
python -m unittest hoi.test_reference hoi.test_hand_collision -v
```

Windows做CPU检查；训练、native物理及显存验证在已准备的Linux GPU runtime。
渲染只回放完整保存状态，0推理/0物理/0FK重构；reference半透明蓝，actual独立配色，共世界固定相机不对齐漂移，并验证实际解码帧数。
