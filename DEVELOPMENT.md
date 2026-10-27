# SuperMimic HOI v1 开发入口

当前实验在独立工作区 **`D:/supermimic_reproduced_hoi/tmp/hoi_v1_2_perception_fusion`**、
分支 **`hoi/v1.2-perception-fusion`** 开发。原 `D:/supermimic_reproduced_hoi` 仍保留
`hoi/v1.1-observation-matching` 及其未提交改动；不要向原目录回写本次实现。
v1.2从该完整工作状态复制，29个未提交文件的原始内容与SHA保存在原目录ignored
`tmp/hoi_v1_2_workspace_setup_20261025_01/baseline.json` 及同目录 `baseline_files/`。
Git分支基点为 `fd626e7`；用户已追加授权：完整GPU验证通过后commit/push v1.2，
再以独立输出和新私有W&B run从零训练20k。不启动、取消或修改v1.1作业。
无持续质量门的v1科学基线分支为 `codex/hoi-v1-no-quality-gate`。
`main` 保留初始fork的复现代码（`a1508d42c9d1699a52e8233ab16d08c488441ab9`），用于对照。
原 `hoi/zero_shot`、`hoi/v1.0` 已全部合入当前分支；整理后删除两个本地及GitHub分支，提交历史仍在当前分支中。

## 当前新版

v1.2使用显式配置 `configs/hoi_v1_2_perception_fusion.json`：沿用v1.1观测和DR合同，
仅选择 `actor_architecture=five_token_fusion_v1`。本体历史930、人体参考448、物体历史90、
物体参考63和shape64各投影成一个256D token，经2层/4头/FFN1024的joint self-attention，
读取本体与物体历史两个token的512D，追加至原1148D直连；decoder为1660→2048→1024→512→256→29。
原human/object encoder、PointNet、MLP隐藏层、imagined-return objective及物理/训练日程完整保留。
不选择新配置时仍为原MLP actor；新版checkpoint显式区分架构，不能直接resume v1.1为v1.2。
集群流程先运行 `cluster/hoi_v1_2_validate.sbatch.sh`：8资产DR/名义评估和真实4096环境，
验证actor batch4096/H32及全部Adam状态驻留时的反传；成功并commit/push后才允许新20k训练。
训练/评估使用独立v1.2启动器，保留双allocator和OMP/MKL=3；完整GPU结果以独立回执为准。
本地CPU回归305项（304通过、可选pxr跳过1项）、历史边界6项通过；完整actor batch2/H32合成反传有限。
证据在本工作区ignored `tmp/hoi_v1_2_cpu_20261025_01/`，不构成GPU显存或HOI效果验收。
完整合同见[设计中的v1.2章节](notes/supermimic_hoi_v1_final_design.md#v12-perception-fusion)。
v1.2首个冻结目录为 `supermimic_hoi_v1_2_perception_fusion_20261025_01`。验证作业1467631已实际获得L40，
于2026-10-26 02:14:32–02:17:14（America/New_York）在native DR初始化失败，FAILED/1:0；未进入batch4096/H32资源探针。
最小修复改用Newton selection setter写摩擦并从模型重新读回，支持strided/gathered属性，不改变DR/模型/物理或训练规模。
旧快照和失败日志保留，后继验证使用独立新快照；自动跟进仍取消，最终GPU验收见下文1470647。
摩擦修复后继1468983于2026-10-26 16:18:18–16:27:34（America/New_York）实际运行；
8资产训练DR/名义eval、真实4096环境及WM2048×8反传通过，actor4096/H32构图时L40显存耗尽，FAILED/1:0。
原冻结源码/失败报告保留。仅v1.2 imagination的actor与WM前向改用non-reentrant activation checkpoint，
在反传时重算中间激活，保留完整H32计算图、输入Jacobian、原网络/目标/规模/优化参数；原MLP执行路径不变。
CPU H32对照的动作/目标值、WM-action Jacobian及全部actor梯度一致；该实现已由下述1470647完成GPU实测。
激活重算修复后的完整CPU回归319项（318通过、1项可选pxr跳过）及6项HOI边界通过，证据见
本工作区ignored `tmp/hoi_v1_2_activation_checkpoint_20261026_01/`。
独立后继快照 `supermimic_hoi_v1_2_perception_fusion_actckpt_20261026_01` 的190个文件和启动器校验通过，
验证1470647已在 `ll-l40-1.grasp.maas` 的L40完成，真实sacct为COMPLETED/0:0，运行10分51秒。
8资产DR/名义eval、真实4096环境及完整batch4096/H32反传通过，WM1/actor2/critic40共43次真实Adam更新，
第二次actor的五路融合/原encoder/PointNet梯度有限非零，BF16、WM-action Jacobian与冻结RMS检查通过，0 OOM。
父进程allocated/reserved峰值16.37/18.82GiB；整卡含8个模拟器的采样峰值约39.25GiB，采样峰值是下界。
完整验收SHA及随后commit/push、独立fresh20k的真实提交状态见上述ignored目录state.json和回执；不恢复定时任务。
摩擦接口修复后的完整CPU回归318项（317通过、1项可选pxr跳过）及6项HOI边界通过，证据见本工作区ignored
`tmp/hoi_v1_2_dr_friction_fix_20261026_01/`；不能替代新的native/GPU验证。
stage/source/queue/提交回执与后续Git/训练回执保留在本工作区ignored `tmp/hoi_v1_2_cluster_20261025_01/`。

本地实验分支 `hoi/v1.1-observation-matching` 另提供显式配置
`configs/hoi_v1_1_observation_matching.json`：actor-only V5字段/坐标约定加新版质量/物体DR。
torso质量另以1D训练上下文接入WM/critic（225D/1215D），actor不读取质量；采集/replay/评估/checkpoint使用版本化合同。
不传该配置仍是下述v1基线，旧8500权重不作为v1.1直接resume。
2026-10-25按用户追加要求，物体历史/未来分别使用独立9D/63D RMS，不裁剪；只在真实采集更新统计。
按用户后续要求已取消独立排队1467536/1467537，合并为单GPU作业1467570：先验证，成功后同一allocation从零训练20k，失败退出；总时限96小时，验证单阶段最多2小时。
当前合并作业仍在排队，尚无原生通过或训练运行结论；模型CPU证据见ignored `tmp/hoi_v1_1_object_rms_20261025_01/`，合并流程/作业回执见`tmp/hoi_v1_1_object_rms_combined_20261025_01/`及下述v1.1设计章节。
完整合同和V5差异见[设计中的v1.1章节](notes/supermimic_hoi_v1_final_design.md#v11-observation-matching)。

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
