# Repository instructions

2026-11-03最新授权：1493722与1494432均已PREEMPTED；在liu-compute/ll-med排3×L40，
从train_01/checkpoint_000500.pt恢复完整模型/优化器/iteration，fresh replay后501轮直接policy更新。
checkpoint每100轮、日志每10轮commit=True；其余科学配置不变。新source/output，保留旧实验。
ll-med仍可被组内ll-high抢占，不承诺绝不抢占。以下运行状态为历史记录。

当前工作区为用户授权的新分支 `codex/hoi-three-gpu-collector`，从8538851派生。
原正式训练1493722继续运行，不修改/停止其冻结source_train_01或输出train_01。
本分支直接启动独立3×L40正式训练对照：learner cuda:0，两个28资产collector各占cuda:1/2；
完整数据/4096环境/训练参数保持，仅日志每10轮、commit=True。新输出与W&B run独立。
以下历史说明中“当前分支”指原两卡实验；以本条新授权为准。

## 开发入口

- 先读 `DEVELOPMENT.md`，再按需读 `notes/README.md`；默认目录 `D:/supermimic_reproduced_hoi`。
- 本工作区开发分支 **`codex/hoi-compact-weighted-teacher`**，由用户授权从 `hoi/multi-asset-collectors` 的 `85d8c1e` 创建。精简数据、独立坐下评估、16类加权采集及2×L40容量预检已通过。用户2026-11-02最新授权正式训练：完整数据/4096环境、2×L40，W&B项目superteacher，run名teacher_v1-<美东日期时间>；原始完整数据与旧实验保留。提交与启动事实必须从回执核对。
- 继承分支的多资产共进程实现。旧作业1488620已在容量预检失败，冻结source_02保留。当前用户授权变更数据范围和采样比例，其他训练/物理参数保持；2048/4096仅作资源/配额对照。共享科学基线仍是 `codex/hoi-v1-no-quality-gate`；不自动切换其他 worktree，不重建已删除分支。
- 特权 actor 使用显式配置 `configs/hoi_v1_privileged_teacher.json`；实现与验证边界见 `hoi/v1/PRIVILEGED_TEACHER.md`。独立离线标注代码位于 `D:/hoi_data_pipeline`，其几何意图标签不等于本仓真实 pair 冲量标签。
- `main`仅保留初始fork复现代码供对照。旧分支历史已包含在当前分支中；不重建中间分支，除非用户要求。
- 修改、Git和检查命令显式指定本仓库workdir，先确认分支及已有改动。
- `D:/SuperMimic`是早期自写实现。selfcollision/handcollision历史工作区及其集群环境保留，不自动同步、切分支、清理或覆盖。

## 当前新版科学基线

- 无持续WM质量门：保留500轮WM-only warmup，之后固定日程更新actor/critic。holdout/WM质量只诊断，不能重新成为actor开关。
- 用户2026-10-31授权本teacher分支扩展为16区域contact；精确顺序/映射见 `contacts.py`。原机器人网络前缀保留，reference目标、actor/critic和replay同步扩展。
- 标签schema `hoi_v1_region_pair_normal_impulse_v3`，feature `supermimic_hoi_v1_full_reference_4`，teacher profile v3；额外26D reset-held桌面上下文进入WM/actor/critic与replay/imagination。旧3/5通道HOI权重和旧双手bundle拒绝直接加载。
- hand collision ON、body self-collision ON；共享CLI配置不另造默认值。保留引擎相邻/焊接link过滤。
- 脚用同源ULTRA完整STL/原变换及已验证CoACD≤5 hull/≤16 vertices，关闭旧脚球；质量/COM/惯量不变，不宣称与PhysX VHACD cooker完全相同。
- 16通道真实标签为区域pair区间法向冲量，身体—地面/自碰撞不能混为身体—物体。WM普通masked BCE；reference组内any-positive/all-negative/else-unknown；统一正概率均值减负概率均值，unknown不参与。
- native身份、overflow、非有限值硬错误检查保留。无质量门不等于取消这些检查。
- 不为通过检查擅改模型、数据、观测、奖励、终止、PD、资产物理、环境数、batch、H、训练配比或优化参数；修复须记录差异和验证证据。

## 代码入口

- 新HOI训练/评估/渲染：`hoi/v1/train.py`、`evaluate.py`、`render.py`；模型 `networks.py`，训练日程 `learner.py`/`train.py`/`config.py`。
- 数据/manifest/60→50Hz：`hoi/data/`与 `hoi/v1/prepare.py`；接触/手脚物理 `native.py`/`contacts.py`/`foot_collision.py`，准入证据 `gates.py`。
- `supermimic/`保留motion-only组件和共享物理；`hoi.run`保留独立zero-shot，不因清理删除这些代码。
- 从零训练显式 `python -m hoi.v1.train --from-scratch`，使用匹配native evidence和新输出；`cluster/hoi_v1_train.sbatch.sh`仍强制INIT_CHECKPOINT，不是from-scratch默认启动器。
- 当前8500 checkpoint/source/bundle/启动器和验收见 `notes/hoi_v1_final8500_eval_handoff_20261025.md`，不猜测路径或将已删权重当作存在。
- 生产启动保留已验证 `PYTORCH_ALLOC_CONF=expandable_segments:True` 及兼容的 `PYTORCH_CUDA_ALLOC_CONF`，OMP/MKL线程预算3；不要在整理启动器时漏掉allocator合同。

## 实验与数据保护

- 新实验使用独立checkout和新RUN_DIR/OUTPUT，显式SUPERMIMIC_ROOT，不向冻结/运行中的实验覆盖源码。
- 提交前读真实queue/log/checkpoint与claim/receipt，避免重复producer；正常排队不取消，SSH失败仅说明观察不可用。
- 不升级驱动、改系统、写共享只读环境/原资产，兼容修复优先官方开关。
- commit/push、合并或清理分支不自动授权部署、重启、取消、续训。
- checkpoint缺完整replay/simulator/RNG；resume后的fresh prefill不是逐位连续恢复。
- data输入与结果必须区分：原motion、冻结selection/splits、原手脚资产、recipe和8500仍引用的cache/runtime不能按日期删除。
- 用户2026-10-25已授权删除旧版及新版8500以前的checkpoint/HOI80；当前保留8500结果及必要依赖。未来删除其他结果仍需明确授权和依赖审查，新输出不覆盖8500。
- 删除前生成精确路径清单、核对无运行producer、排除symlink/junction与保留依赖；删除后核对保留SHA/链接。本次清单见 ignored `data/cleanup_20261025/`。

## 评估与W&B

- Curated数据用完整protected_eval（约10%按task/source group分层，已有保护不可进入train）；缺task保留unspecified。历史canonical HOI80继续保护。冻结checkpoint，0optimizer/normalizer更新、完整物理时长、不因失败重置截断。
- Compact teacher 只由冻结父分区确定性派生：ULTRA curated 全保留；HOI 排除4类与ULTRA重叠物体；Omni保留carry桌面交互与全部soccer；GRAIL训练原ID 2/122/169/172/200、独立eval原ID35/48/87/124/199。不做轨迹增强，排除的eval仍由父分区保护。新密集asset ID必须附原ID映射。
- 权重绑定在训练bundle：ULTRA每类6%、HOI九类各4%、Omni桌面15%、足球15%、GRAIL坐下10%。足球五task及五训练椅子内部均衡。配额核对实际transition起始帧，不能以reset抽签或合成样本替代；评估不加权、不抽样缩短。
- T+1状态/reference，T action/obs/goal/control；保留时基/四元数。HOI-trained full-reference policy已有物体输入；历史zero-shot不得加物体policy观测来冒充旧协议。
- 分开报告normalized return/作者终止、39body PHC、object pose及联合判据。PHC0.5m；object位置0.1m/角度30°持续0.3s为工程姿态判据，不等于抓取成功。
- 新对照用独立私有W&B run，仅用户指定续训同实验时恢复原run。
- 新版显式 `WANDB_PROJECT=SuperMimic_HOI_v1_20261023`，entity `tianzizh-university-of-pennsylvania`；代码旧默认仍是 `SuperMimic_Upstream_Reproduction`，勿把环境覆盖称为已改代码默认。
- 只传配置/指标/汇总，不传权重、源码、原数据、完整rollout或视频，不输出/提交凭据。

## 文档与检查

- notes只保留最终v1设计、必要数据/外部项目索引、当前8500验收；入口 `notes/README.md`。轮询/提交回执留ignored data，不再堆阶段Markdown。
- 更新设计/验收时原地维护；删旧结果后移除失效链接，保留旧比较数字时标明完整旧输出已清理。
- CPU：`python -m unittest discover -s tests -v`；历史HOI边界：`python -m unittest hoi.test_reference hoi.test_hand_collision -v`。
- 改Slurm后bash-n，改渲染后看真实图像/视频；CPU/native/smoke/正式训练分开陈述，RUNNING不等于稳定或收敛。
- 离线视频只回放真实保存状态，0策略推理/0物理/0FK重构；reference机器人/物体半透明蓝，actual独立配色，共世界固定相机，不对齐漂移，验证完整解码帧数。
- 提交前检查diff和暂存范围，原资产/数据/权重/结果/凭据不入Git。未获用户commit/push指令，不自动提交本次整理。
