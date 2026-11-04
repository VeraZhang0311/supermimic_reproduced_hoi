# SuperMimic HOI v1 开发入口

## protect_eval_sm 开发子集（2026-11-04）

用户要求从当前1085条protected_eval中选10%并先保存，不启动评估、不改自动评估日程。
入口 `python -m hoi.data.protected_eval_subset --training-bundle ... --parent-bundle ... --output ...`。
固定seed20261104，按dataset/task最大余数法选109条（10.046%），层内优先资产覆盖、
source group多样性；SHA256排序保证同seed可复现，不依赖checkpoint成绩。
ULTRA27、HOI-Retarget29、Omni48（桌面16、足球32）、GRAIL5（独立eval椅每把1条）。
共39资产、109不同source group，完整参考时长15.779分钟。
集群保存于 `/mnt/projects/ll/tianzizh/hoi_compact_teacher_20261101_01/protect_eval_sm/`；
本地镜像 `data/protect_eval_sm/`。selection.json是选择名单，clips.jsonl保留父bundle原始行，
README.zh-CN.md解释范围；不是重新构造的bundle。未来消费者加载绑定完整bundle后选择原索引。
名单内容SHA `24e75b76cc59ef92687754af87e19caa6be5c5bff653f5f7b37d8cc038ccdd71`。
3项CPU测试（含错误分区/泄漏/重复/篡改/覆盖拒绝）、实际名单重现、父文件哈希及缓存路径检查通过。
所有未抽中条目仍保持protected；全量数据、训练冻结源与正在运行的作业均未修改。

## 独立 canonical HOI80 小测（2026-11-04）

用户授权用自动续训 campaign 的 checkpoint 2000 额外跑固定 HOI80，当前训练继续。
入口 `cluster/hoi_teacher_hoi80.sbatch.sh`，单张 Liu L40；评估使用独立源码/新输出。
`evaluate --canonical-hoi80-bundle` 只选择当前完整 protected_eval motion bank 中的固定80条，
不改冻结bundle、不用旧五通道contact cache。旧registry固定content SHA作为名单依据，
同时核对原motion SHA、原/目标帧数、保护标记、四类各20条及source group不进入train。
仍按当前teacher的物理/观测/完整时长评估，报告明确为80条子集；不触发GRAIL全量汇总，
不改变5000轮周期完整protected_eval。该入口不用于部署新训练源，原训练冻结源保持不变。
28项选择/评估/GRAIL测试与Bash语法检查通过；实际GPU结果另以evaluation.json为准。
集群CPU预检已通过：2000 checkpoint完整有限、当前模型严格加载成功、80条共23718控制步。
权重SHA d58fe2d2fb3bd6625c4611d7b1d33c104bd79cefa4a6e44665522c10bbdb1426；
路径为compact实验根下train_liu_auto_01/attempts/job_1498746_r0010/run/checkpoint_002000.pt。
独立评估源码source_eval_hoi80_2000_01，source SHA
1caf7ab8f015bdb7c00be89286a8828baf277a9c1a41c754e872028e4f33a60d；
作业1500362（1×L40/4CPU/64GiB/4小时），输出eval_hoi80_2000_01，
回执eval_hoi80_2000_01_submission/receipt.json。提交后PENDING(Resources)，原1498746 RUNNING。
此为提交/CPU预检事实，尚无本次GPU评分结果；排队状态必须实时查询。

## 抢占自动续训与梯度检查（2026-11-03晚）

1496475在17:23 EST被抢占，最后日志1320轮，完整checkpoint1300含800次本段policy更新。
1300 checkpoint SHA 2cafed04a190881682997bc06b196ef37248ebcc86a97e32ba7c3708b68cada2。
用户授权三卡Liu从1300恢复及自动requeue。集群PreemptMode=REQUEUE，先前no-requeue已移除。
入口cluster/hoi_compact_teacher_requeue.sbatch.sh及compact_teacher_requeue.py；同一Slurm job
自动重新排队，按restart count创建独立attempt/run、W&B run后缀；flock防并发producer。
仅扫描该campaign的checkpoint与绑定初始1300；校验结构、数据/物理/config身份、有限值后选最新。
checkpoint先fsync临时文件，再hardlink原子发布且不覆盖旧文件；中途partial不会成为恢复候选。
每次恢复需要fresh replay，不是逐位续跑；普通代码错误不会主动重提。5000倍数恢复点会补查原
checkpoint目录中的eval提交回执，避免保存后抢占漏评估或重复提交。训练参数不变。
第1270轮pi_gradnorm=3909.38、1280=256.75、1300=1.56、1320=1.70均为裁剪前指标；
原实现先norm<=1，再每元素<=0.1，检查非有限后才Adam.step。补充裁剪后norm/absmax日志，
不改阈值或动作定义。巨大原始梯度的具体样本原因无法从每10轮汇总倒推，不能宣称没有不稳定性。
账户仅ll-med/normal，ll-high需要管理员开通。以下为历史记录。

正式自动续训已提交1498746：liu-compute/ll-med、3×L40、16CPU/64GiB、7天、--requeue。
冻结source_train_liu_auto_01、campaign train_liu_auto_01、回执train_liu_auto_01_submission；
source SHA 19d5497e434b609ad59cb214a4e5e65cd10dee4c8393b4126e20a5a9924a29e1。
W&B superteacher，run前缀teacher_v1-liu3gpu-auto1300-20261103-195859-EST，attempt后缀-r0000等。
本地80项综合测试通过，随后新增单梯度分量case与优化器结构检查后的25项相关测试通过；
集群实际文件系统8项恢复/原子checkpoint测试通过，真实1300权重选择验证通过。
无GPU Slurm requeue测试1498745：同一job restart0/1均执行、最终COMPLETED/exit0。
此测试验证主动requeue路径；尚未在正式GPU作业上再次发生抢占并完成恢复，不伪称已实测该事件。

## Liu 三卡恢复（2026-11-03）

两卡1493722与三卡1494432均因PREEMPTED退出；两卡最后日志700轮/7000WM/200policy更新。
已读回500轮checkpoint，199个张量均有限。用户授权在liu-compute、liu-account、ll-med上
排3×L40续训，使用同schema --resume恢复模型、优化器与iteration；不重新执行WM-only warmup，
但缺失replay必须fresh prefill，501轮即执行policy更新。log_every=10、ckpt_every=100；
完整数据/4096环境/原优化参数/5000轮protected eval保持。ll-med仍可被ll-high抢占。
新冻结source_train_liu_01、输出train_liu_01与提交回执train_liu_01_submission保留完整出处。
已提交1496475：liu-compute/ll-med、3×L40、16CPU/64GiB、7天，run名
teacher_v1-liu3gpu-resume500-20261103-080340-EST（superteacher PRIVATE）。
source SHA 7434de3896d316edfe3cbbd27ee13c251f7c1a0578848a5f555f5379f7e344fd；
resume SHA 91955006d11280758657d3de4d3fc670941767c41a1b5cc8c43a8c487f6b8f07。
39项训练/恢复/追踪测试与bash/内嵌Python语法检查通过；包括501轮policy更新和600轮保存。
下列旧作业状态仅为历史。

## 三卡采集对照（2026-11-02）

本工作区分支codex/hoi-three-gpu-collector，用户授权直接启动独立3×L40训练。
原1493722保持运行；新部署为1张learner、2张collector，两个进程仍各28资产。
完整数据、4096环境、batch/H32、物理、500轮WM预热与优化参数不变，仅log_every=10、
W&B commit=True。入口cluster/hoi_compact_teacher_three_gpu.sbatch.sh。
50项tracking/bootstrap/worker设备分配/packed进程测试与启动器语法检查通过。
复用两卡容量通过作为学习器/数据基线，明确三卡部署尚未实测，不将旧报告冒充三卡验收。
比较时使用日志elapsed_seconds差分，排除初始化/prefill，并区分WM-only与actor/critic阶段；
日志频率差异属于本次对照的已知额外改动。以下是原两卡实验历史。

新正式训练作业1494432已提交：batch/liu-account、3×L40、16CPU/64GiB、7天、no-requeue。
2026-11-02读回为PENDING(Resources)，预计11月3日07:00:59 EST，估计可能变化。
原1493722同时读回仍RUNNING，未停止、修改或重启。W&B仍用PRIVATE superteacher，独立run
teacher_v1-3gpu-20261102-192402-EST，启动后创建在线记录。
冻结source_train_3gpu_01和输出train_3gpu_01均在hoi_compact_teacher_20261101_01根目录；
source identity d4326560bc47b88dd422e9f65e2bc8b2093bd81731d1a4ed422db1327b085305。
回执train_3gpu_01_submission/receipt.json。未获得三卡吞吐数据，不宣称已有加速。

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
