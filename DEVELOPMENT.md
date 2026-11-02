# SuperMimic HOI v1 开发入口

本工作区当前分支 **`codex/hoi-compact-weighted-teacher`**，从资源优化分支
`hoi/multi-asset-collectors` 的 `85d8c1e` 派生。用户授权精简数据、独立坐下评估和
16 类 weighted sampling，并要求测试后 commit/push，**先不启动训练**。
精简数据入口 `python -m hoi.data.compact_teacher`，采样合同 `hoi/v1/task_sampling.py`，
实际 transition 审计 `hoi/v1/sampling_audit.py`，零训练验收脚本
`cluster/hoi_compact_teacher_verify.sbatch.sh`。完整方案见 teacher 说明的 compact 小节。
精简数据已在集群 `hoi_compact_teacher_20261101_01/dataset_04` 生成：6537 train、1085 eval，
其中GRAIL独立评估50条/5椅。2026-11-02作业1490096已COMPLETED/exit 0：4096环境、56资产、
128步实采524288条transition通过配额审计，原生证据继承和最终preflight均通过。
正式训练未启动。完整learner容量预检1490289（2×A10）已FAILED：4096和2048均完成实采及
WM backward，但actor imagination前向CUDA OOM；完整结果见teacher说明。
用户随后授权先不拆multi-teacher，改用2×L40原配置重测。作业1493079已COMPLETED/exit 0，
01:04:10：4096环境完整实采、WM backward、actor H32 backward及critic检查全部通过。
learner/collector显存采样峰32.0488/10.8242GiB，进程树PSS峰18.4300GiB，整作业batch
MaxRSS 34.6634GiB。source_06/dataset_04及训练参数不变；未触发2048回退。
入口 `cluster/hoi_compact_teacher_capacity.sbatch.sh`，报告在capacity_l40_01。
容量测试0 optimizer step；支持采用2×L40完整数据配置，尚未验证optimizer临时开销与长期运行。
用户随后明确授权正式训练：superteacher项目、teacher_v1-<美东日期时间>名称。
新入口cluster/hoi_compact_teacher_train.sbatch.sh绑定通过的56资产packed报告与原参数；
仅track.py增加显式HOI_WANDB_RUN_NAME，25项tracking测试及两个启动器语法检查通过。
VPN恢复后已实时核对superteacher项目为PRIVATE，正式训练作业1493722已RUNNING（dj-l40-0，
2×L40、16CPU/64GiB、7天上限、不自动requeue）。run名teacher_v1-20261102-154206-EST。
冻结源码source_train_01、输出train_01、提交回执train_01_submission位于同一compact实验根目录。
当前仅确认作业启动，首次optimizer更新/W&B在线run仍须从训练日志核对，不能将RUNNING称为收敛。

以下记录继承的资源优化与teacher历史：原资源优化分支从teacher提交`0d1a11e`派生。
原teacher排队作业1488620及冻结源码保持不动；新分支只做多资产共进程和4096/2048环境
资源对照，不启动另一个训练。teacher此前从整理后的 v1 基线
`fd626e752686ef61b3e7e1b604f12412a9d5cd2b` 创建，不包含 v1.1/v1.2 感知改动。
显式配置 `configs/hoi_v1_privileged_teacher.json` 开启无观测噪声的全身特权 actor；
字段、ULTRA 对照及 contact/reward 扩展提案见 [teacher 说明](hoi/v1/PRIVILEGED_TEACHER.md)。
2026-10-31 已扩展16区域真实接触、普通masked BCE、reference三态组内替代与
正负分别归一化奖励，并接通actor/critic输入及curated数据接口。用户确认按task/source
group分层约10% protected_eval；缺失task保留unspecified。旧5通道checkpoint/bundle
拒绝直接加载。新物体采用显式质量/摩擦约定及来源绑定，Omni桌面接入真实kinematic场景；
完整资产逐项原生验收已完成。2026-11-01用户授权训练，superteacher作业1488620已提交；
提交时等待资源，全量并发容量检查通过后自动从零训练。进度与边界见teacher说明。
离线数据管线独立维护在 `D:/hoi_data_pipeline`；来源转换、50Hz expected-contact
标注和标注 Slurm 作业使用该仓入口。本仓接入新数据仍需准备匹配的 bundle、真实接触
记录及 native evidence。数据标注成功不等于机器人已可执行。

本工作区沿用 **`codex/hoi-v1-no-quality-gate`** 的科学基线，以下说明仍适用；
开发、检查与 Git 命令显式指定当前 teacher worktree，不切换其他分支的工作区。
`main` 保留初始fork的复现代码（`a1508d42c9d1699a52e8233ab16d08c488441ab9`），用于对照。
原 `hoi/zero_shot`、`hoi/v1.0` 已全部合入当前分支；整理后删除两个本地及GitHub分支，提交历史仍在当前分支中。

## 共享历史 v1 基线（当前 teacher 差异以上文及 teacher 说明为准）

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
