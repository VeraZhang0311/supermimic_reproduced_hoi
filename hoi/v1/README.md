# SuperMimic HOI v1

当前开发基线为 `codex/hoi-v1-no-quality-gate`：无持续WM质量门、完整手脚碰撞、五通道contact head。使用同SHA的完整ULTRA脚STL/零安装变换，在不可变手overlay上关闭旧脚球并追加CoACD5hull/16vertex凸包，保留原质量/COM/惯量。原VHACD60000只作作者配置溯源，不冒称相同cooker。Train/eval必须显式绑定foot manifest、USD和新bundle身份，不能只依赖默认机器人路径。

设计依据为[v1最终设计](../../notes/supermimic_hoi_v1_final_design.md)。新版source15的原生/资源与正式训练身份，以及最后8500pt完整HOI80和固定20条视频证据，见[最终验收报告](../../notes/hoi_v1_final8500_eval_handoff_20261025.md)。训练已因24h作业时限结束，没有完成20k；操作示例不自动授权续训。原 `hoi.run` 仍为冻结、无物体policy observation的zero-shot入口。

## 数据与真实几何

先用 `hoi.data.manifest` 固定 split、selection、motion SHA 和 canonical60→50 Hz 契约，再准备训练 bundle。资产 recipe 必须声明实际 collision mesh、精确 scale、原生资产 SHA、质量/COM/惯量/摩擦与原手 STL/mount。不能把视觉 mesh 或 net contact force 当成实际碰撞监督。

```bash
python -m hoi.v1.prepare --manifest /absolute/manifest.json \
  --assets /absolute/assets.json --output /new/bundle --partition train
python -m hoi.v1.native --bundle /new/bundle/bundle.json --asset-id 0 \
  --num-envs 8 --steps 64 --output /new/native --hand-collision --self-collision
```

`--limit` 仅生成显式 subset，正式配置拒绝 subset。Native硬门须覆盖全部训练geometry groups，验证显式完整脚profile/compiled hulls与原物性、四个5ms子步pair normal impulse、五类实体pair正例、手脚碰地负对照、实际物理与reward geometry一致性及掌面probe。旧三通道或球脚证据不能替代新版验收。

本分支新增左右脚—物体真实监督：contact head 为 `2048→128→5`，顺序固定为左手—物体、右手—物体、物体—地面、左脚—物体、右脚—物体。足部绑定实际 G1 `left_ankle_roll_link` / `right_ankle_roll_link` 的有效碰撞几何，按同一20ms区间内四个5ms子步累加法向冲量，再以平均法向力严格大于1N标记正例；脚—地面不计入脚—物体，源39个body flags不作为pair监督。Replay、BCE权重、holdout、评估和W&B同步扩为五通道；reference expected-contact与contact-match reward仍只对应双手。

标签schema为 `hoi_v1_pair_normal_impulse_v2`，feature版本为 `supermimic_hoi_v1_full_reference_2`。旧三通道HOI checkpoint明确拒绝resume，未实现3→5自动迁移；motion-only legacy初始化路径仍保留。完整脚profile为 `ultra_g1_stl_coacd5v16_v1`，不会自动改写其他motion-only/zero-shot入口的裸机器人。

完整recipe格式、固定`native_robot_asset`手部overlay和显式CPU并行预处理见 [PREPARATION.md](PREPARATION.md)。同recipe独立CoACD重算可能改变hull，因此worker复用哈希绑定的已验证overlay，不为每个进程重新生成手碰撞体。

## 训练与启动检查

已验收生产运行使用下列allocator/线程设置，parent与8个worker继承相同值。新运行先核对已准备runtime支持；不改共享环境，原无step资源probe不保证长期无碎片OOM。

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
python -m hoi.v1.train --config configs/hoi_v1_first.json \
  --bundle /absolute/train/bundle.json --native-evidence /absolute/native_reports.json \
  --eval-bundle /absolute/eval80/bundle.json \
  --output /new/preflight --preflight
python -m hoi.v1.train --config configs/hoi_v1_first.json \
  --bundle /absolute/train/bundle.json --native-evidence /absolute/native_reports.json \
  --eval-bundle /absolute/eval80/bundle.json \
  --from-scratch --hand-collision --self-collision --output /new/run
```

当前新版从零实验使用显式 `--from-scratch`。可选motion-only迁移才使用 `--init-checkpoint`；来源缺失physics时必须显式声明legacy手/self两项，physics变更需显式标注。新增输入列零扩展、保留旧网络/normalizer、新分支统计独立，迁移不继承旧optimizer/iteration。相同五通道schema的 `--resume` 恢复三个optimizer及iteration，但必须新输出与fresh真实replay prefill，不是逐位恢复。

训练保留原默认4096环境、32步BPTT、8步WM、10×2048 WM batch、4096 policy batch及优化设置。`configs/hoi_v1_pipeline_smoke.json` 显式减少环境/batch/迭代，仅做管线验证，网络生产尺寸不变。多geometry worker的环境总和保持4096，单一中央learner；当前完整脚新版全量bundle已通过独立GPU资源验证，证据范围见最终报告。

按用户最新要求，`hoi_v1_first.json`与v1默认把上限设为20000迭代，每500迭代保存完整checkpoint。250000迭代的原LR decay schedule及其余优化配置保持原值。

每5000迭代自动提交一次独立GPU作业，对刚保存的固定checkpoint跑完整canonical HOI80：5000、10000、15000、20000。必须预先准备同manifest/recipe/固定手部资产的完整`eval80` bundle；校验失败会在训练前拒绝。评估与训练有独立进程/显存/输出，提交记录防止重复producer；异步排队不等于评估完成。全部80条跑完整时长、无optimizer/normalizer更新，不把验证集用于训练holdout或改变固定WM warmup日程。

actor的固定shape只在一次imagination开始时编码，整个32步复用同一计算图，保留encoder梯度；不会跨optimizer更新缓存。critic仍在每个value minibatch重新计算shape encoder。

设计中的两个独立消融入口也已实现：`configs/hoi_v1_no_contact_objective.json`只把contact-match reward权重置0，保留contact head、BCE和诊断；`configs/hoi_v1_no_geometry_input.json`保留三个PointNet及64D接口，把actor/WM/critic的shape feature一致置0。两份配置继承全部生产数值默认，包括20k/500/5000；`variant`写入plan/checkpoint/评估并严格校验恢复，必须使用新run。主实验固定`variant=hoi_v1`。共同HOI归一化仍按0.15/step，另报实际objective上界（no-contact为0.145）的归一化结果。本轮未启动消融训练。

每个 global reference 的50帧 bin 中，编号模20为0的 bin 保留为 WM 真实 transition holdout，WM训练和统计不使用这些行。它来自 train split，独立于 validation/eval80/test40。每次更新采用实际执行 action；接触标签来自 `(t,t+1]` 四个物理子步。实际正负计数用于 BCE class weighting。

本分支 `codex/hoi-v1-no-quality-gate` 保留500迭代WM-only warmup，之后每轮正常更新actor/critic。真实holdout的1/8/32步误差、zero-increment/constant-velocity对照、contact/onset/release coverage及pair概率仍按原 `gate_every=50` 周期计算，但只作诊断，不再阻断策略更新。历史文件名 `wm_gate_*.json` 保留，并明确 `controls_policy_updates=false`；plan/checkpoint/completion记录固定更新日程。Native/数据身份检查以及非有限数据、loss/梯度和holdout诊断的硬错误仍保留。

W&B的 `gate/policy_passed` 表示固定warmup日程，`diagnostics/wm_quality_passed` 表示独立质量比较；配置记录 `training/wm_quality_gate_enabled=false`。质量未通过、回落或合法holdout暂缺都不重新关闭actor/critic；非有限诊断仍立即失败。旧分支/冻结source的质量门行为不因整理当前文档而改变。

## 冻结完整时长评估

```bash
python -m hoi.v1.evaluate --training-bundle /absolute/train/bundle.json \
  --bundle /absolute/eval80/bundle.json --checkpoint /absolute/checkpoint_008500.pt \
  --output /new/eval80 --hand-collision --self-collision
```

评估保存 state/reference `T+1`、action/observation/goal `T`，失败后继续真实物理至动作结束；所有权重和 normalization 冻结。PHC使用哈希绑定的39body原始虚拟URDF，14body追踪特征不冒充39body指标。分别报告 human normalized return/termination、HOI termination、PHC、object pose 与 pair contact，联合成功不称为 grasp success。所有输出 exclusive create，不上传原始数据/权重/rollout 到 W&B。

CPU检查：`python -m unittest discover -s tests -v` 和 `python -m unittest hoi.test_reference hoi.test_hand_collision -v`。原生 Linux GPU smoke 与正式训练分别记录；CPU通过不能代替原生验证。

## 保存状态的离线渲染

```bash
python -m hoi.v1.render --evaluation /absolute/evaluation.json \
  --bundle /absolute/eval80/bundle.json --robot-urdf /absolute/g1_visual.urdf \
  --output /new/render
```

只回放已保存的真实世界状态，不重新推理policy、FK或物理步进。actual保留原色/橙物体，reference半透明蓝色；共享固定世界相机。完整保留并独立解码校验T+1帧。此例要求bundle可解析绑定的原USD美术资产；单资产可用显式 `--object-visual`。只有另行要求碰撞展示时才使用 `--use-collision-geometry`，并在报告注明collision display，不能悄悄替代原视觉资产。固定20条需要按canonical manifest映射选择四类各080/084/089/094/099；该底层命令未传 `--case` 时默认全部case，不自动筛选20条或生成合集。旧`tools/render_hoi_rollouts.py`仍保留自己的输入协议。

## 私有W&B配置/指标同步

`python -m hoi.v1.track --run-dir /absolute/run --output /new/tracking --dry-run`只写可审阅白名单payload，不连接网络。显式`--online`才向既有私有项目创建全新run，`resume=never`；仅配置、有限数值指标和汇总，禁用源码/git/console/system capture，不上传权重、原始数据、rollout或Artifacts。Slurm环境显式设置`WANDB_SYNC=1`时，训练完成和每次独立评估完成后各同步新run；默认不联网。原`hoi.track`仍用于旧zero-shot报告。
