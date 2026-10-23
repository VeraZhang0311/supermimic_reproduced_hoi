# Repository instructions

## 开发入口

- 本文件适用于整个仓库。先读 `DEVELOPMENT.md`，了解入口和历史实验边界。
- 除非用户另行指定，统一在 `D:\supermimic_reproduced_hoi` 的 `hoi/zero_shot` 分支开发。
- 执行修改、Git 和检查命令时显式指定此仓库为工作目录，并先确认当前分支及已有改动。
- 聊天的默认工作区可能仍是 `D:\SuperMimic`；它是早期自写实现，不是当前开发入口。
- `D:\supermimic_reproduced_hoi_selfcollision` 和 `D:\supermimic_reproduced_hoi_handcollision`
  是保留的历史实验工作区。不要自动向它们同步新代码、切分支、清理文件或覆盖结果。

## 代码定位

- `supermimic/physics.py`：共享碰撞配置、CLI 开关和资产校验。
- `supermimic/robot.py`、`env.py`：G1/PD 配置与物理场景创建。
- `supermimic/mdp.py`：观测、解析奖励、终止条件和 critic 特征。
- `supermimic/motion.py`：reference 与 goal；`networks.py`：世界模型、actor、critic。
- `supermimic/train.py`、`evaluate.py`：训练、checkpoint 续训和完整 motion 评估。
- `hoi/run.py`、`evaluate.py`：冻结策略的 HOI80 评估；`hand_collision.py`：橡胶手碰撞资产。
- `tools/render_hoi_rollouts.py`：已记录 HOI rollout 的离线渲染。
- `cluster/train.sbatch.sh`、`hoi/evaluate.sbatch.sh`：新集群任务入口。
- `cluster/legacy/`：历史启动器源码快照，用于溯源，不作为统一分支的新任务入口。

## 物理配置与科学边界

- 新实验默认 **hand collision ON、body self-collision ON**，两项独立。
- 使用 `--[no-]hand-collision` 和 `--[no-]self-collision`；不要绕过共享配置硬编码另一套默认值。
- 历史基线为 hand OFF / self OFF；历史自碰撞训练为 hand OFF / self ON；
  历史补手 HOI80 为 hand ON / self OFF。新默认组合不能冒充这些历史结果。
- 自碰撞仍保留引擎对相邻或焊接 link 的过滤；ON 不代表强制所有 link 两两碰撞。
- 手部 STL、安装坐标和已验证 hull recipe 应保持可追溯。CoACD 不能称为与原 PhysX cooker 完全相同。
- 除非属于用户要求的实验改动，不要为通过检查而改变模型、数据、观测、奖励、终止、PD、
  质量/惯量、环境数、batch、训练配比或优化超参数。必要修复需记录差异及验证证据。
- 新 checkpoint 保存物理开关；续训继承并校验配置。旧 checkpoint 缺少物理信息时必须显式指定两项。
- 原 checkpoint 不含完整 replay、模拟器和 RNG 状态；fresh replay prefill 后的续训不是逐位连续恢复。

## 实验与集群保护

- 日志、checkpoint、rollout、渲染均使用新输出目录，不复用或覆盖已有科学结果。
- 保留 `D:\SuperMimic`、两个历史实验工作区及其集群对应目录中的原代码、环境、数据和结果。
- 不从开发分支覆盖已提交或运行中的实验源码。新 Slurm 任务使用独立开发 checkout，
  显式设置 `SUPERMIMIC_ROOT` 和全新的 `RUN_DIR` / `OUTPUT`。
- 提交训练前读取实际队列、日志和 checkpoint，避免重复 producer。正常资源排队不是 bug。
- 不升级集群驱动、修改系统或写入共享只读环境/资产。兼容修复优先使用官方支持的运行开关。
- 作业状态以最新 `squeue` / `sacct` 和日志为准；SSH 失败表示观测不可用，不能据此认定排队或训练失败。
- 合并、commit、push 不等于部署，也不自动授权重启、取消或续训已有实验。

## HOI 评估、指标与 W&B

- 默认使用 canonical HOI80；冻结 checkpoint，禁止优化器及归一化更新，物体保留真实物理。
- 未经用户要求，不给 policy 增加物体观测。保存完整动作时长，不因失败重置或截断记录。
- 状态和 reference 为 T+1 帧，action/observation/goal 为 T 帧；保留时间与四元数约定。
- 分别报告作者 normalized return / termination、PHC、object pose、联合判据；
  `supermimic_n_failed=0` 不代表全部 HOI 交互成功，object-pose 判据也不等同抓取成功。
- 新对照实验使用独立 W&B run；只有用户指定续训同一实验时才恢复其原 run。
- 当前项目为 `tianzizh-university-of-pennsylvania/SuperMimic_Upstream_Reproduction`，保持私有。
- 仅上传配置、指标和评估汇总；不上传权重、原始数据、完整 rollout 或源码，不输出或提交凭据。

## 渲染与检查

- 离线视频直接回放保存的真实状态，不重新推理策略或执行物理步进。
- reference 机器人与物体采用半透明蓝色，actual 保留独立配色；共享世界坐标和相机，
  不单独对齐轨迹来隐藏漂移。保留完整帧并验证实际解码帧数。
- 常规 CPU 回归：`python -m unittest discover -s tests -v`。
- HOI 边界检查：`python -m unittest hoi.test_reference hoi.test_hand_collision -v`。
- 修改 Slurm 启动器后用 `bash -n` 检查；修改渲染后检查真实 rollout 的图像/视频。
- 训练和 IsaacLab 原生仿真在已准备的 Linux GPU runtime 验证；本地 Windows 可做 CPU 与离线渲染检查。
- 明确区分 CPU 检查、原生物理检查、smoke 和正式训练；不能把提交作业或进程 RUNNING 称为稳定训练或收敛。
- 提交前检查 diff 和暂存范围，只加入源码、说明及适当测试；原始资产、数据、结果和凭据不进入 Git。
