# 五通道contact从零实验

用户最新授权：停止当前warm-start，保留旧zero-start继续，用扩展后的contact head版本启动新的全部从零实验。此前「先不跑新实验」的限制已被本次明确启动要求更新。

## 原任务保护

2026-10-24 03:33:45 UTC，先读取实际队列、运行配置、日志和checkpoint。精准取消warm **1461248**（comment `hoi-v1-warm-source13-expandable-20261023-01`），Slurm返回`CANCELLED`。最后完整checkpoint为source13 `results/hoi_v1_warm_resume_expandable_20k_20261023_01/checkpoint_003500.pt`，234,379,241bytes，SHA256 `8aafed3fd332a7bde39d701bd45727d2ba6e4d68fc955f4ed4bd194059ee40fb`，全部原结果保留。用户将WM曲线判为不值得继续；该科学判断不等同于本次观察到非有限数或执行错误。

旧zero-start **1461253**（comment `hoi-v1-scratch-source13-expandable-20261023-01`）保持RUNNING，停warm前后均已核对。最新完整日志1200 iter，恢复后WM7000/policy400，原质量门1201通过、日志无执行错误。该任务继续使用不可变source13、三通道和原持续质量门，不同步本分支源码，不取消/重启/续训它。

## 新实验合同

部署独立source14快照 `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_14`，来自本地 `codex/hoi-v1-no-quality-gate`。五通道顺序为双手—物体、物体—地面、左右脚—物体，label v2/features2，详见[实现与CPU验收](hoi_v1_foot_contact_20261023.md)。

正式命令固定 `--from-scratch`，不传init/resume/checkpoint；pi、WM、critic、各PointNet、所有normalizers与三Adam、replay、计数全新。保持64 prefill、前500 iter WM-only，随后固定更新actor/critic，holdout质量比较只作诊断。所有生产配置保持：全train2651/eval80、4096env/replay1000/H32、WM2048×8×10、actor4096、critic5×4、LR decay250000、handON/selfON、seed0、20k目标、每500保存、5000/10000/15000/20000独立GPU冻结完整80条评估。

先取得同fullbundle的五通道八资产原生正负对照及真实4096/H32资源证据。资源probe从fresh constructors开始，64真实prefill、生产WM/actor/critic反传、Adam两moment reserve，但不执行optimizer.step，其模型不用于正式run。正式训练在另一个全新输出从0开始。

新W&B run使用既有私有 `tianzizh-university-of-pennsylvania/SuperMimic_HOI_v1_20261023` 项目、fresh UUID/resume never；只上传白名单配置、数值和评估汇总，不上传源码、权重、原始数据或完整rollout。沿用已验证的双`expandable_segments:True` allocator环境选项和只读runtime，不修改系统/驱动/共享环境。

本记录先写明确授权、停止回执和待验收合同；实际snapshot SHA、GPU job、结果及正式启动状态将在取得真实证据后追加，不把提交或RUNNING当成验证通过、20k完成或HOI成功。

## 用户追加ULTRA网格脚对齐与source15

用户追加要求统一原ULTRA完整脚STL碰撞，故正式新实验包含五通道contact、去持续质量门、同源STL足部overlay三项已授权改动。前述source14验证因缺少新目录的默认FK MJCF路径而在创建仿真前失败；结果保留，不复用为成功证据。实现、物理差异边界和source15身份见[ULTRA足部对齐记录](hoi_v1_ultra_foot_alignment_20261023.md)。source15验证job1461502已开始，只有八资产门与fresh完整资源检查成功后才提交新的正式from-scratch producer。

03:54 UTC复核旧zero-start1461253仍RUNNING，最新1300 iter、本次续训policy更新500，actor/critic数值有限，无执行错误。W&B当前run `c21c0d4acff54c06ac64c98f54187977`已恢复显示running；不能根据短暂UI crashed状态认定计算作业崩溃。保持该旧任务原样继续，不重启或迁移新碰撞配置。

## 正式新producer已提交

04:16 UTC，source15验证1461502 `COMPLETED 0:0`：全量八资产严格门与fresh4096/H32/64prefill生产反传资源检查均通过，optimizer step0。完整验收数字及报告SHA见[GPU验收记录](hoi_v1_ultra_foot_alignment_20261023.md)。独立提交脚本review通过、CPU真实路径preflight通过后，唯一提交正式job **1461591**，comment `hoi-v1-contact5-ultrafoot-scratch-source15-20261023-01`，新输出 `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_15/results/hoi_v1_contact5_ultrafoot_scratch_20k_20261023_01`。04:17:55 UTC已RUNNING，尚在初始化；不能把这时的RUNNING称为已完成训练更新。

preflight记录为source15 `results/contact5_ultrafoot_formal_preflight_20261023_01/plan.json`，实际启动器 `data/hoi_v1_contact5_scratch.sbatch.sh`，完整非secret export/命令及排重claim/receipt在 `data/contact5_ultrafoot_formal_submission{.request,}.json`。训练命令固定：

```bash
python -m hoi.v1.train --config "$CONFIG" --bundle "$BUNDLE" \
  --native-evidence "$NATIVE_EVIDENCE" --output "$RUN_DIR" --eval-bundle "$EVAL_BUNDLE" \
  --evaluation-runtime "$SUPERMIMIC_RUNTIME" --evaluation-isaaclab "$ISAACLAB_PATH" \
  --from-scratch --hand-collision --self-collision
```

其中BUNDLE/EVAL_BUNDLE/NATIVE_EVIDENCE都明确指向source15 `results/contact5_ultrafoot_validation_20261023_01/`中的新bundle/header/evidence，不沿用staging模板里的旧球体足身份。此为已提交命令的溯源记录，不应重复提交同一producer。旧zero-start1461253原样保留；04:16 UTC最新1500 iter、本次续训WM10000/policy650，actor/critic有限。

## 正式启动检查通过（04:22 UTC）

job1461591实际RUNNING，`initialization.json.mode=explicit_from_scratch`，8组native worker已就绪，真实64prefill完成262144条transition。首条正式metrics为iteration1/sim_steps270336/WM更新10/策略更新0；十次WM优化loss均有限，均值total20.61566、robot6.66530、object13.85228、contact0.98077。fit接触正例五通道为26914/26066/165239/4357/6785，双脚计数已进入新head的实际训练数据。

`wm_gate_000001.json`的holdout `finite=true`。fresh模型尚未优于简单基线，`wm_quality.passed=false`，但 `controls_policy_updates=false`；当前actor为0仅因固定前500 iter WM-only，预期首个actor优化在501，当前尚未实际验证正式501步Adam或后续长期稳定性。首次500 checkpoint、5k独立80评估尚未到达，不把配置/提交当成执行完成。

新私有W&B run为 [HOI-v1-train-61f87216](https://app.forge.coreweave.com/tianzizh-university-of-pennsylvania/SuperMimic_HOI_v1_20261023/runs/61f872168c46457f9fc74bfe9d63ea9b)，resume never。实际API回读state=running、PRIVATE、contact/channel_count5、schema v2、wm_quality_gate_enabled=false，source/dataset/native机器人SHA与source15新bundle完全一致，已同步WM更新10和双脚正负计数。API可选files列表未取得，不据此声称远端没有文件；实际上传边界继续由已审查白名单tracker限制。

正式优化后现有allocation内只读nvidia-smi：L40 used30631MiB/46068MiB、利用率39%，主训练与8个worker进程都在；日志没有异常栈、OOM或数值硬失败。初始化出现既有runtime的deprecation/float cast提示，旧source13也有同类提示；未改runtime，实际native/prefill/WM/holdout数值检查通过。此为启动健康证据，不是20k完成、收敛或HOI成功结论。
