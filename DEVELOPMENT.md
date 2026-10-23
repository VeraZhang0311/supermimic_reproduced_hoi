# 统一开发入口

以后在 `D:\supermimic_reproduced_hoi` 的 `hoi/zero_shot` 分支开发。
`supermimic_reproduced_hoi_selfcollision` 和 `supermimic_reproduced_hoi_handcollision`
保留为历史实验工作区；本次合并不向这些目录或集群旧目录同步源码，不移动结果。

## 新实验的物理配置

训练和 HOI 评估共用 `supermimic/physics.py`：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--hand-collision` / `--no-hand-collision` | 开启 | 原 HOI 橡胶手几何，独立 CoACD hull overlay |
| `--self-collision` / `--no-self-collision` | 开启 | 机器人身体自碰撞，保留引擎相邻/焊接 link exclusions |

这两个开关独立。新的默认 **hands ON + self ON 是新组合**，不是之前任一组结果的重命名。
关节、PD、质量/惯量、奖励、观测和训练超参数不因开关而改变。
橡胶手几何仍使用已验证的 STL 和安装坐标；CoACD 仍不是原 PhysX cooker。
没有物体的 motion-tracking 训练也使用相同手部资产准备流程，无需 HOI 数据集。

下面的训练/仿真命令在已准备好的 Linux + IsaacLab runtime 中运行：

```bash
# 新默认：手碰撞 ON，身体自碰撞 ON；选择全新的日志目录
python -m supermimic.train --iterations 14000 --log_dir logs/new_both_on

# 原始基线的物理开关（历史结果仍以原源码快照为准）
python -m supermimic.train --no-hand-collision --no-self-collision --log_dir logs/new_stock_profile

# 只有身体自碰撞
python -m supermimic.train --no-hand-collision --self-collision --log_dir logs/new_self_only

# 只补橡胶手碰撞体，保持原 HOI80 补手对照的物理开关
python -m hoi.run --prepared results/hoi80_inputs --checkpoint /path/to/ckpt_014000.pt \
  --output results/new_hands_only --hand-collision --no-self-collision

# HOI80 默认组合，两项都开；run id 默认每次新生成
python -m hoi.run --prepared results/hoi80_inputs --checkpoint /path/to/checkpoint.pt \
  --output results/new_both_on
```

`hoi.run` 始终使用冻结权重，保存80条完整 rollout，不加入物体 policy 观测。
可用 `--baseline-report /path/to/old/evaluation.json` 生成同 checkpoint、同数据的逐动作对照。
结果记录两项 `physics` 配置、checkpoint 训练配置（旧 checkpoint 标记 unknown）、源码和资产哈希。
`hoi.track` 使用报告里的新 run id，拒绝恢复既有 W&B run，不上传权重、数据或源码。

## Checkpoint 续训

自碰撞分支的 fresh replay prefill 和 CSV header 修复已整合到训练入口。
模型和三个 Adam 状态照常恢复；旧 checkpoint 没有 replay、模拟器或 RNG 状态，续训并非逐位连续。
新 checkpoint 保存物理开关；续训默认继承并拒绝改变它们。旧 checkpoint 必须显式声明两项：

```bash
python -m supermimic.train --resume /path/to/old_self_ON_last.pt \
  --no-hand-collision --self-collision --iterations 24000 --ckpt_every 1000 \
  --log_dir logs/new_continuation
```

所有训练和 HOI 输出都需要新目录，不能覆盖已有实验。评估 watcher 继承该 run 的配置；
评估旧 run 时也需要显式声明缺失的碰撞设置。

## 离线渲染

```bash
python tools/render_hoi_rollouts.py --input /path/to/completed_HOI80 \
  --assets /path/to/data/g1_description --output results/new_render20
```

四类各选080、084、089、094、099，不按成绩筛选。全部 T+1 状态以720p/50fps回放；
蓝色半透明 reference 机器人和物体与实际轨迹共用世界坐标、固定相机，不消除真实漂移。
不做新策略推理或物理步进。保存单条、合集、索引、缩略图和完整解码帧数/哈希校验。
Windows 默认 GLFW，Linux 默认 EGL，可用 `MUJOCO_GL` 覆盖。
预览使用 `--preview-only` 和独立输出目录，正式渲染另选新目录。

## 历史与集群

| 历史实验 | hand | self | 保留来源 |
|---|---|---|---|
| 原配置基线 | OFF | OFF | `codex/grasp-upstream-reproduction`，原目录旧结果 |
| 身体自碰撞训练/续训 | OFF | ON | 原 selfcollision 目录，原分支 `322f007` |
| 补手 HOI80 / 已交付视频 | ON | OFF | 原 handcollision 目录；科学实验源码树 `d0620b1`，渲染提交 `8bfd2a1` |

旧专用 Slurm / W&B 脚本完整保存在 `cluster/legacy/` 供溯源；
复核历史实验应进入原 checkout/commit，不在统一分支用旧脚本覆盖旧目录。
新的 Slurm 入口 `cluster/train.sbatch.sh` 要求指定独立 `SUPERMIMIC_ROOT` 和全新 `RUN_DIR`；
历史集群根目录被拒绝。合并与 GitHub push 不会自动部署或重启任何集群作业。

CPU 检查：`python -m unittest discover -s tests -v`。
