# SuperMimic HOI v1

设计依据：`notes/supermimic_hoi_v1_final_design.md`。新入口位于 `hoi.v1`；现有 `hoi.run` 仍是冻结、无物体 policy observation 的 zero-shot 入口。

## 数据与真实几何

先用 `hoi.data.manifest` 固定 split、selection、motion SHA 和 canonical60→50 Hz 契约，再准备训练 bundle。资产 recipe 必须声明实际 collision mesh、精确 scale、原生资产 SHA、质量/COM/惯量/摩擦与原手 STL/mount。不能把视觉 mesh 或 net contact force 当成实际碰撞监督。

```bash
python -m hoi.v1.prepare --manifest /absolute/manifest.json \
  --assets /absolute/assets.json --output /new/bundle --partition train
python -m hoi.v1.native --bundle /new/bundle/bundle.json --asset-id 0 \
  --num-envs 8 --steps 64 --output /new/native --hand-collision --self-collision
```

`--limit` 仅生成显式 subset，正式配置拒绝 subset。Native gate 需覆盖全部训练静态 geometry groups，验证四个5ms子步的 pair normal impulse、手/物体/地面身份、正负碰撞控制、实际物理与 reward geometry 一致性、真实掌面 probe 选择。未完成的项目保持 blocked。

完整recipe格式、固定`native_robot_asset`手部overlay和显式CPU并行预处理见 [PREPARATION.md](PREPARATION.md)。同recipe独立CoACD重算可能改变hull，因此worker复用哈希绑定的已验证overlay，不为每个进程重新生成手碰撞体。

## 训练与启动检查

```bash
python -m hoi.v1.train --config configs/hoi_v1_first.json \
  --bundle /absolute/train/bundle.json --native-evidence /absolute/native_reports.json \
  --eval-bundle /absolute/eval80/bundle.json \
  --output /new/preflight --preflight
python -m hoi.v1.train --config configs/hoi_v1_first.json \
  --bundle /absolute/train/bundle.json --native-evidence /absolute/native_reports.json \
  --eval-bundle /absolute/eval80/bundle.json \
  --init-checkpoint /absolute/motion_checkpoint.pt \
  --no-legacy-hand-collision --legacy-self-collision --allow-initialization-physics-change \
  --hand-collision --self-collision --output /new/run
```

旧 checkpoint 缺失 physics 时必须显式声明来源的两项开关。新增列采用零扩展，保留旧网络和 normalizer；新物体分支有独立统计。迁移不继承旧 optimizer/iteration。相同 schema `--resume` 恢复三个 optimizer 和 iteration，但使用全新输出目录及真实 replay prefill，不是逐位恢复。

训练保留原默认4096环境、32步BPTT、8步WM、10×2048 WM batch、4096 policy batch及优化设置。`configs/hoi_v1_pipeline_smoke.json` 显式减少环境/batch/迭代，仅做管线验证，网络生产尺寸不变。多 geometry worker 的环境总和保持4096，单一中央 learner；完整资源可行性需另有 GPU 实测。

按用户最新要求，`hoi_v1_first.json`与v1默认把上限设为20000迭代，每500迭代保存完整checkpoint。250000迭代的原LR decay schedule及其余优化配置保持原值。

每5000迭代自动提交一次独立GPU作业，对刚保存的固定checkpoint跑完整canonical HOI80：5000、10000、15000、20000。必须预先准备同manifest/recipe/固定手部资产的完整`eval80` bundle；校验失败会在训练前拒绝。评估与训练有独立进程/显存/输出，提交记录防止重复producer；异步排队不等于评估完成。全部80条跑完整时长、无optimizer/normalizer更新，不用这些评测结果修改WM启动阈值。

actor的固定shape只在一次imagination开始时编码，整个32步复用同一计算图，保留encoder梯度；不会跨optimizer更新缓存。critic仍在每个value minibatch重新计算shape encoder。

设计中的两个独立消融入口也已实现：`configs/hoi_v1_no_contact_objective.json`只把contact-match reward权重置0，保留contact head、BCE和诊断；`configs/hoi_v1_no_geometry_input.json`保留三个PointNet及64D接口，把actor/WM/critic的shape feature一致置0。两份配置继承全部生产数值默认，包括20k/500/5000；`variant`写入plan/checkpoint/评估并严格校验恢复，必须使用新run。主实验固定`variant=hoi_v1`。共同HOI归一化仍按0.15/step，另报实际objective上界（no-contact为0.145）的归一化结果。本轮未启动消融训练。

每个 global reference 的50帧 bin 中，编号模20为0的 bin 保留为 WM 真实 transition holdout，WM训练和统计不使用这些行。它来自 train split，独立于 validation/eval80/test40。每次更新采用实际执行 action；接触标签来自 `(t,t+1]` 四个物理子步。实际正负计数用于 BCE class weighting。

本分支 `codex/hoi-v1-no-quality-gate` 保留500迭代WM-only warmup，之后每轮正常更新actor/critic。真实holdout的1/8/32步误差、zero-increment/constant-velocity对照、contact/onset/release coverage及pair概率仍按原 `gate_every=50` 周期计算，但只作诊断，不再阻断策略更新。历史文件名 `wm_gate_*.json` 保留，并明确 `controls_policy_updates=false`；plan/checkpoint/completion记录固定更新日程。Native/数据身份检查以及非有限数据、loss/梯度和holdout诊断的硬错误仍保留。

W&B的 `gate/policy_passed` 表示固定warmup后是否允许策略更新，`diagnostics/wm_quality_passed` 单独表示WM质量比较是否达标，配置记录 `training/wm_quality_gate_enabled=false`。`hoi/v1.0` 保留旧持续质量门。当前两个正式实验仍用原冻结source13，本分支未部署或启动新实验；见[分支变更记录](../../notes/hoi_v1_no_quality_gate_20261023.md)。

## 冻结完整时长评估

```bash
python -m hoi.v1.evaluate --training-bundle /absolute/train/bundle.json \
  --bundle /absolute/eval80/bundle.json --checkpoint /absolute/checkpoint_002000.pt \
  --output /new/eval80 --hand-collision --self-collision
```

评估保存 state/reference `T+1`、action/observation/goal `T`，失败后继续真实物理至动作结束；所有权重和 normalization 冻结。PHC使用哈希绑定的39body原始虚拟URDF，14body追踪特征不冒充39body指标。分别报告 human normalized return/termination、HOI termination、PHC、object pose 与 pair contact，联合成功不称为 grasp success。所有输出 exclusive create，不上传原始数据/权重/rollout 到 W&B。

CPU检查：`python -m unittest discover -s tests -v` 和 `python -m unittest hoi.test_reference hoi.test_hand_collision -v`。原生 Linux GPU smoke 与正式训练分别记录；CPU通过不能代替原生验证。

## 保存状态的离线渲染

```bash
python -m hoi.v1.render --evaluation /absolute/evaluation.json \
  --bundle /absolute/eval80/bundle.json --robot-urdf /absolute/g1_visual.urdf \
  --use-collision-geometry --output /new/render
```

只回放已保存的真实世界状态，不重新推理policy、FK或物理步进。actual保留原色/橙物体，reference半透明蓝色；共享固定世界相机。完整保留并独立解码校验T+1帧。缺少物体美术资产时需显式`--object-visual`或`--use-collision-geometry`，后者在报告中注明collision display，不悄悄替代视觉资产。旧`tools/render_hoi_rollouts.py`仍保留自己的输入协议。

## 私有W&B配置/指标同步

`python -m hoi.v1.track --run-dir /absolute/run --output /new/tracking --dry-run`只写可审阅白名单payload，不连接网络。显式`--online`才向既有私有项目创建全新run，`resume=never`；仅配置、有限数值指标和汇总，禁用源码/git/console/system capture，不上传权重、原始数据、rollout或Artifacts。Slurm环境显式设置`WANDB_SYNC=1`时，训练完成和每次独立评估完成后各同步新run；默认不联网。原`hoi.track`仍用于旧zero-shot报告。
