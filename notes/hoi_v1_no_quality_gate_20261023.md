# HOI v1 固定更新日程分支

本记录描述 `04a3796` 的单独去质量门变更。后续按用户要求在同分支增加五通道双脚接触监督，模型/replay等新增差异与验证另见[双脚contact扩展](hoi_v1_foot_contact_20261023.md)；下文「模型不变」指去门提交本身。

后续执行状态：用户已进一步授权停止warm、保留旧zero、启动五通道且ULTRA网格脚的新从零实验。source15八资产严格门与全量资源检查通过，正式job1461591已注册新W&B并开始采集，见[启动记录](hoi_v1_contact5_scratch_experiment_20261023.md)。下文「不启动新实验」仅描述最初去门提交时点。

用户要求先commit/push当前实现，再新增分支去掉WM质量门；先让现有两个实验跑完，不启动新实验。

基线 `hoi/v1.0` 已提交并推送，commit `b127d8a`，保留持续质量门、既有实现与历史验收记录。新分支 `codex/hoi-v1-no-quality-gate` 从该提交创建。基线发布前219项CPU回归、原HOI边界6项和六份HOI Slurm脚本的 `bash -n` 通过；原始数据、资产、权重、结果、临时运行回执和凭据未加入Git。五处额外文件尾空行作无语义清理，运行中的冻结source13保持不可变。

## 行为

- 保留原64 vector-step fresh replay prefill和全局500 iter WM-only warmup；按原零基iteration条件，从第501个已完成iteration开始每轮更新actor/critic。
- 删除holdout质量比较对策略日程的控制。即使误差未优于zero/CV基线、contact分层质量下降，或holdout暂缺合法窗口，也不会反复暂停actor/critic。
- Holdout仍使用原reserved train bins，保留1/8/32步、phase、接触概率及baseline报告，不读eval80/test40校准阈值。原 `gate_every` 字段及数值保留，现仅表示诊断周期。
- Native证据、数据/物理身份、有限性和梯度检查继续执行。非有限holdout诊断是硬错误，不以关闭质量门的方式静默继续训练；其他非coverage诊断异常仍原样抛出。
- 模型、观测、奖励、终止、物理、采样、4096/H32、batch/UTD/Adam/LR/clip/critic更新数均不改变。目标20k、每500保存、每5000完整canonical80评估不改变。

## 记录含义

`plan.json`、checkpoint extra和completion写入 `policy_update_schedule`，其 `wm_quality_gate_enabled=false`。训练metrics中的 `policy_gate` 是固定warmup日程的允许状态；`wm_quality` 是独立诊断结果。

历史诊断文件名 `wm_gate_*.json` 和checkpoint的 `wm_gate` 字段保留以兼容读取，但其 `controls_policy_updates=false` 明确表示只作诊断。W&B新增 `diagnostics/wm_quality_passed`，与实际日程的 `gate/policy_passed` 分开；白名单配置新增布尔字段 `training/wm_quality_gate_enabled`，不扩大任何文件、源码或原始数据上传范围。读取旧记录时缺少新字段保持缺省，不把旧实验误标为无质量门。

## 实验边界与验证

现有warm1461248和scratch1461253继续使用不可变source13及其原质量门；本地/GitHub分支操作不会部署到它们。没有取消、重启、续训或提交新GPU任务，没有更换它们的源码、参数、输出或W&B run。新分支只做CPU回归与代码检查，不表示已经验证无门版本的GPU运行或科学表现。两版完成后，再讨论后续实验。

新增回归覆盖固定warmup边界、从未通过质量比较也能更新、通过后质量回落不暂停、恢复后三optimizer及fresh replay、holdout稀少时仅记录诊断、非有限诊断硬失败，以及W&B实际日程与质量状态的独立记录。

针对性train/track检查31项通过；新增兼容性断言后，全量CPU回归224/224通过（29.762s，无skip），日志在ignored `tmp/hoi_v1_no_quality_gate_full_tests.log`。`git diff --check`通过。相对基线配置JSON、Slurm启动器、共享supermimic层、数据适配器、模型、learner、replay、reward、checkpoint载入器均无差异。独立只读review通过，未发现必须修复的P1/P2，复核全局warmup时钟、恢复语义、质量回落/coverage状态、硬错误和指标含义；本分支不进行GPU验证或新实验。
