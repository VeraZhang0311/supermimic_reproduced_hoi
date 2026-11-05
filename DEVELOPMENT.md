# LAFAN1 fullset privileged GMT

当前分支 `boneseed/privileged`，直接从 `main@a1508d4` 创建。
本次 checkout 位于 `D:/supermimic_reproduced_hoi/tmp/boneseed_privileged`；
原目录的 v1.1 分支及其未提交修改保持原样。

第一阶段训练不读取物体的 privileged GMT，后续 residual learning 单独实现。
本分支不迁入 HOI reward、object/shape/contact head 或 HOI 训练入口。
按用户最新要求，第一阶段先使用 **LAFAN1 G1 fullset，全部40条完整clip**。
分支名保持不变。Boneseed适配保留为显式选项，不是当前训练默认。

## 当前数据：LAFAN1 fullset

训练默认 `--dataset lafan1`、`data/lafan1_g1/*.csv`。启动要求恰好40条唯一clip；
缺少数据、子集、重复身份或格式错误会明确失败，不静默用少量clip训练。
沿用main的G1 retarget数据和时基：30Hz无表头CSV，米、xyzw根四元数、29关节弧度，
完整clip重采样到50Hz，不裁剪、不筛动作。来源与机器人资产准备沿用 `tools/download_data.sh`。
每个CSV的内容SHA、名称、源帧数、源fps与控制dt共同绑定checkpoint数据身份；
评估读取训练args中的dataset和motion_dir，覆盖同一fullset，0优化/normalizer更新。

## 观测

参考 `codex/hoi-three-gpu-collector@cc99253` 的 `hoi/v1/privileged.py`。
保留原 `main` 的 99D 本体观测、10帧历史、730D dense motion goal 和 32D motion latent；
向原 decoder 追加独立 RMS 的 **3361D** 无噪声 robot privileged block：

| 每个 lookahead（+1、+13 个 50Hz 控制帧） | 维度 |
| --- | ---: |
| 当前39-link状态：root height、去root相对位置、heading坐标姿态/速度 | 583 |
| reference body target及position/orientation/velocity errors | 936 |
| previous action、current q/qd、previous q/qd | 145 |
| 两档合计 | 3328 |
| 共享 raw robot context：friction、29 joint offsets、3 torso COM offsets | 33 |

原 decoder 隐藏层不变，输入从1022增至4383。39-link FK沿用原URDF安装变换，
速度为nominal COM marker；根输入仍为link-origin速度。rotation6D为参考分支的x/z列。
删除 object pose/velocity/error、object context/shape、interaction graph、object/ground
expected region contacts及validity、two-cuboid scene context，未填任何object占位零值。
reference source分支添加的object goal encoder也不引入此motion-only模型。

本体前缀与追加块均不加观测噪声（与teacher profile一致），探索action noise保留。
采集、replay、WM imagination和评估共用特征函数；episode首帧previous q/qd等于当前值。
RMS仅真实采集更新，想象和评估不更新。WM/critic结构、奖励、终止、PD、50Hz控制、
4 substeps、4096环境、H32、batch、优化器和500轮WM-only日程保持main设置。
replay明确保存previous q/qd。checkpoint绑定新schema、URDF SHA、body/joint identity与
数据manifest SHA；旧LAFAN/HOI checkpoint不能直接resume。
resume缺少replay/simulator/RNG，必须fresh prefill，不声称逐位连续恢复。

## 可选后续数据：BONES-SEED

来源：[BONES-SEED G1](https://huggingface.co/datasets/bones-studio/seed)，
使用 `g1.tar.gz` 解出的 `g1/csv/{session}/*.csv`。数据访问条件由用户在数据源完成，
这里不下载全库、不处理凭据。原始数据和生成reference不入Git。

已按[NVIDIA官方转换器](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic/data_process/convert_soma_csv_to_motion_lib.py)
核对原生列名、厘米、Euler xyz度、关节度，转米/弧度/wxyz，120Hz→50Hz。
按列名重排29关节，SLERP根姿态，差分在单clip内计算，不跨clip。
拒绝缺帧、非有限值、错误列及短于一个控制transition的输入。

准备命令（CPU环境需要numpy、torch、mujoco；指定全新输出）：

```bash
python -m supermimic.boneseed --source /absolute/bones-seed/g1/csv \
  --output /absolute/new/boneseed_g1
# 可选：--selection /absolute/frozen_selection.json
```

selection是相对source的CSV路径JSON列表；未传时使用指定目录下的全部CSV。
manifest逐条记录source路径、SHA、50Hz帧数，绑定完整float32 `states.bin` SHA；
禁止覆盖已有输出，未完成输出没有manifest。输入不裁剪、不重定向朝向、不增加增强。
原论文Appendix E还有name+feasibility筛选，完整最终关键词/clip清单和floating-foot
stationary gate细节未取得。本入口**不自动仿造筛选、不宣称精确129,514条**；
manifest明确 `exact_paper_selection=false`，后续确认的selection应作为新独立数据产物。

reference仅驻留71D float32 generalized states，根矩阵与body FK按索引计算，
不预计算全库body矩阵、不降为fp16。47.3M帧时状态约12.5GiB，clip/bin索引另计；
不是实际全库容量验收。分批评估默认256环境，逐条覆盖完整选定库，不改变clip时长。
`--record`只支持可同时放入eval_envs的库；大库先无record完整评估。

修复main的末帧空bin：当帧数为50n+1时，不让reset采样到没有后继的clip最后一帧；
该修复同时用于LAFAN与Boneseed；其余50%/40%/10% return-based reset配比保持不变。

## 自碰撞与运行

`G1_CFG` 默认 `enabled_self_collisions=True`。启动读取实际compiled MuJoCo模型，
要求存在非相邻机器人碰撞对；保留engine parent/weld/exclude过滤，不改mesh、
质量、COM、惯量或PD，不额外增加collision reward。每步保留finite/overflow硬错误。
本分支以main资产为基线，未迁入HOI专用hand/foot overlay。

在独立Linux GPU checkout准备main的原始机器人资产与已验证Isaac Lab/Newton runtime后：

```bash
export SUPERMIMIC_ROOT=/absolute/isolated/boneseed_privileged
cd "$SUPERMIMIC_ROOT"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
python -m supermimic.train --dataset lafan1 --motion_dir /absolute/lafan1_g1 \
  --log_dir /absolute/new/lafan1_privileged_run
python -m supermimic.evaluate --run_dir /absolute/new/lafan1_privileged_run \
  --ckpt /absolute/new/lafan1_privileged_run/last.pt
```

`run_train.sh`也默认使用LAFAN1 fullset入口，保留两项allocator和3线程配置，
传入SUPERMIMIC_ROOT应为container内此分支独立挂载路径。
新训练只写全新log_dir。这里未启动集群作业、未创建W&B run，未部署/续训现有实验。
后续Boneseed须显式传 `--dataset boneseed --motion_dir /absolute/new/boneseed_g1`，
使用独立log_dir，不沿用LAFAN checkpoint resume。

## 验证

```bash
python -m unittest discover -s tests -v
bash -n run_train.sh
```

CPU测试覆盖native CSV单位/列序/时基及坏输入、39-link FK与独立MuJoCo数值对照、
clip边界、goal/reward、特征yaw/平移不变性和有限差分梯度、H32反传、
previous-joint replay、分批评估和legacy checkpoint拒绝、自碰撞parent过滤。
11项CPU测试全部通过，另完成Python编译、diff检查和启动器bash-n。
只读验证本地已有LAFAN fullset：40条、264,705源帧、441,140控制帧；
数据身份SHA为`ee88d32c6376df53b89d828b875ed324a0bad5633d3e516f9f0eec80dc903abb`。
对每条clip抽一帧，将参考分支原始3962D输出精确切掉object相关字段，
与当前3361D输出逐元素比较，最大绝对误差为0。真实数据校验回执留ignored
`.cache/real_lafan_report.json`，未复制或修改早期工作区的数据/资产/源码。
Windows只做CPU验证；真实Isaac Lab/Newton self-contact、overflow接口、GPU编译/显存、
真实物理rollout及正式训练仍需在Linux GPU runtime验收，不把CPU通过称为native通过。
