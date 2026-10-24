# HOI v1 原生验收交接（持续更新，2026-10-23）

后续正式实验已经用户明确授权启动及修复续训；当前source13+expandable的warm1461248/scratch1461253，最新真实验收见[部署与续训记录](hoi_v1_holdout_resume12_20261023.md)。以下为初次原生就绪阶段的历史证据，「未启动」不表示当前仍未授权；不得重跑已通过的八资产/全量缓存检查或覆盖其输出。

工作分支 `hoi/v1.0`。**最终22:44 UTC验收全部通过，首次实验已就绪，正式20k训练未提交或启动。** 最新结论、真实路径命令及限制见[首次实验就绪记录](hoi_v1_first_experiment_readiness_20261023.md)。1459951已COMPLETED0:0/6:12，完整数据64步的9phase全部passed、0optimizer steps；source10部署与CPU preflight也已通过。以下历史观测保留，不再据旧待完成记录重提任何作业。

最新root观测补充（22:37 UTC）：全量缓存1459845与CPU审计1459949均COMPLETED0:0；完整bundle content SHA为`318482c78f699d33aa5ae7743f2918f19d96fa3d78e2869add50f4a890d022c8`。1459914已COMPLETED0:0/25:25，八组同hash原生gate与全量4096/H32九阶段资源报告均passed，0optimizer steps；同步最大GPU used45,086,081,024 / 47,590,277,120bytes。root本地独立重验gate passed，evidence SHA `daece1e92ce6eeffdad51f0a0b9fc282c271c7226e45fa9ed30934b3fbc4da8d`。1459951的fulltrain64步诊断已正常RUNNING，尚待最终报告。下文旧RUNNING/待缓存记录保留历史，以上新实际观测覆盖旧状态。

正式launcher另有CPU预算P2已审：source09强制OMP/MKL24，probe为3；8spawn会超订阅24CPU allocation。仅launcher修为尊重显式线程数、未设默认3，shell及默认/显式预算验证通过；新source10 archive SHA `ebbcd611b4c30a500b3730e683f2821d5fa4f1d272c257ed6dbca225cea3fdca`。全部112个Python文件与09逐字节相同，原生/显存无需重复，09不覆盖。root部署会话仍未返回（本地exec session42381，setup辅助SSH PID54064，22:38 UTC观测）；不能直接重跑setup或宣布部署成功。新root为`/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_10`，需确认后运行ignored `tmp/hoi_v1_final_preflight_20261023.py`，该文件仅CPU `--preflight`，不会启动训练。以后状态以最新真实观测为准。

22:39 UTC后续确认：source10部署会话42381已成功退出0、远端archive SHA一致；CPU preflight会话54600也已退出0并验证170份源文件、112份09/10 Python一致性、8份原报告SHA及内嵌原字节。preflight实际目录为`..._10/results/first_experiment_preflight_20261023_01`，plan SHA `185ded469e03dfd2c2e5bae06a454627db31406f352417b4b9df970a18d45225`；严格native门passed、配置无smoke deviation、未创建worker/prefill/checkpoint。无需重跑setup或preflight。本地`data/hoi_v1_final_readiness_receipts_20261023_01/`已有preflight与完整resource12报告，尚待1459951的64tick最终报告后才能全部ready。

## 冻结源码与运行环境

- 最新候选源码：`/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_09`，快照 SHA256 `b03e3df856a4848b7d96f852a1cc5782937eb7de8e7999efdac35f7db31c5184`，167 文件，CPU197项通过。09 相比08仅 resource probe 显式 `cuda.set_device` / `cuda.init` 后再重置显存峰值，核心 physics/models/learner 和超参数不变。
- 07源码 SHA256 `e97110819e38b0d0eb881159a92ea772b534ffd5b0aec83bdee16a96f1b70167`；相比06仅checkpoint/eval中性stage、this_run计数、prepare短TMP wrapper等元数据调整。08源码 SHA256 `9723db9e7431e2734a655426a2957d20c29188596235babd58b1b52381c7ef22`，部署已确认，增加renderer显式mesh-root资料路径修复；195/196项CPU及旧边界6项分别通过。
- 当前原生作业冻结源码：`/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_06`，快照 SHA256 `60f6426942c983fba0dfbaf8337a02fa7858b9bd49381ac99eae1ba689ed129c`。不要覆盖任何已冻结目录中的执行源码。
- 只读 runtime：`/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128`。
- 只读 IsaacLab：`/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi/IsaacLab`。
- 新开发缓存可复用：`/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_01/.cache/hoi-v1-native/1459665`。不得改历史 runtime、共享资产、旧 checkout 或旧结果。
- SSH：`ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o UpdateHostKeys=no tianzizh@grasp-login1.seas.upenn.edu`。Slurm 可执行文件在 `/opt/slurm/bin/`。作业状态须最新 `squeue` / `sacct` 与日志联合确认。
- 曾遇port22连接超时，之后实际SSH恢复；只代表当时远程观测不可用，不能判定任何作业失败，不重提或取消。使用 `ConnectTimeout=10` 与 keepalive，连接之间有界backoff，避免并发/频繁SSH。
- Linux CPU tensor IPC 必须使用短 TMPDIR：`source "$SUPERMIMIC_ROOT/cluster/hoi_v1_runtime.sh"; hoi_v1_short_tmpdir`。Warp/Inductor cache 可以长路径；AF_UNIX listener 所在 TMPDIR 不可以。

## 当前 producer 与不可变输出

下表状态是最近读到的状态，并非之后时刻的保证；恢复工作先重新读取队列和产物，禁止直接重复提交。

| Job | 作用 | 最近状态 | 输出 |
|---|---|---|---|
| 1459842 | 06 单真实 train clip 全链路诊断 | FAILED1:0，6:26；native gate、64 prefill、2 次 WM update、cp1/cp2、完整220步冻结eval成功；policy update 0；仅renderer的历史URDF缺相对meshes/pelvis.STL，由网络agent修路径并复用真实rollout，禁止重跑成功WM或eval | `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_06/results/pipeline_train1_20261023_01` |
| 1459848 | 八资产 512/2048 新容量 + 浅压 pair probes + 64 真实 reference fixture ticks | COMPLETED0:0，14:39；八组全部通过，`NATIVE_LIGHT_VALIDATION_FAILURES=0`；仍是fixture-only，不能替代bound gate | `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_06/results/native_light8_20261023_01/asset_{legacy_id}` |
| 1459844 | 八资产各首条真实 train clip 的 resource fixture 缓存 | COMPLETED0:0，8clips；可开始bound native gate和4096实测 | `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_06/data/bundle_resource8_graph_20261023_02/bundle.json` |
| 1459845 | 全 train2651 几何/接触缓存 | 最近RUNNING22:39，545/2651 reference+contact caches；final bundle尚无，正常缓存生产 | `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_06/data/bundle_train2651_graph_20261023_02/bundle.json`（待最终原子写出） |
| 1459846 | canonical eval80 几何/接触缓存 | COMPLETED0:0，80clips/23798states/23718controls，subset=false，独立loader合同审计通过 | `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_06/data/bundle_eval80_graph_20261023_02/bundle.json` |
| 1459882 | 真 resource8 bundle 的逐组bound gate → 资源初始化 | FAILED1:0，7:44；八组bound gate已全部通过并保存；随后resource probe在任何模型/replay/worker分配前因CUDA allocator未显式初始化而报Invalid device argument。不是OOM，不能声称资源通过；09已修，八组native不重跑 | `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_06/results/resource8_bound4096_20261023_01` |
| 1459900 | 08离线重放06已保存真实221状态帧 | COMPLETED0:0，远端/本地实际解码均221帧，root已view首末图；仅资料mesh-root修复，无新policy或物理执行 | 网络agent维护08新render输出；不覆盖06失败目录 |
| 1459905 | 09独立4096/H32资源验收 | sacct COMPLETED0:0，00:00:59，MaxRSS18,574,064K；report passed=true，9个phase完成，0optimizerstep。先require上述八组同resource8证据，无重复native；1L40/24CPU/96GB | `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_09/results/resource4096_20261023_01` |
| 1459914 | 全train2651精确bundle八组bound gate + 实际fullquota4096/H32资源验收 | 已排队 `afterok:1459845`；45min/1L40/24CPU/96GB，source09，0optimizerstep；不取消正常cache45，不重复producer | `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_09/results/fulltrain2651_bound4096_20261023_01` |
| 1459949 | fulltrain最终bundle独立CPU合同/quality/实际quota/worker local→global映射审计 | 几何agent已排队 `afterok:1459845`，2CPU/64GB/30min；不重复已完成resource8/eval80合同检查 | `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_09/results/full_train_bundle_audit_20261023_01/full_train_bundle_audit.json` |
| 1459951 | fulltrain resource-only正式64tick prefill容量补验 | 已排队 `afterok:1459914`，30min/1L40/24CPU/96GB；严格复用1459914 fullbundle八组evidence，0optimizerstep，不重复native calibration | `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_09/results/fulltrain2651_resource64_20261023_01` |

GPU 日志：

- `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_06/pipeline_train1_20261023_01.slurm.log`
- `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_06/native_light8_20261023_01.slurm.log`
- `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_06/resource8_bound4096_20261023_01.slurm.log`
- `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_09/resource4096_20261023_01.slurm.log`
- `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_09/fulltrain2651_bound4096_20261023_01.slurm.log`
- `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_09/fulltrain2651_resource64_20261023_01.slurm.log`

resource8 bundle内容SHA256为 `d51b4f590a005610641bf92c431c5970ae7358a79ec41e4a9b727619326b4b48`。已通过的严格八组证据是 `..._06/results/resource8_bound4096_20261023_01/evidence/native_evidence.json`，同目录 `gate.json` 为passed=true且无blocker；`native_asset_0..7/native.json`均包含正确global ID及bundle hash。1459905只复用该不可变证据，不重做八组检查。

09资源实测：12×4096=49152真实transition，collection9.090s，约5407 env-step/s；WM每步2048有效行×8窗口、B4096/H32 actor和135168行critic target+32768 mini均完成，梯度有限。Replay实际4,739,072,000bytes，Adam双moment reserve156,065,688bytes。parent allocator峰allocated20,788,129,280bytes、reserved27,030,192,128bytes；同步全process GPU used最大46,663,335,936/47,590,277,120bytes，表面free约884MiB。**不能据该free推断fulltrain将OOM**：critic phase实际parent allocated约6.06GB，其余约20.97GB是可回收parent allocator cache；fullreferences约2GiB可额外使用/回收cache。1s own-PID sampled峰39704MiB只是下界；该GPU无其他job进程。**这不是全train容量通过**：fullcache/实际quota完成后仍须全数据实测，不能由8clip结果外推成功或失败，也不提前改chunk。没有optimizer.step及其foreach临时张量，也没有完整10WM/20critic iteration吞吐。

1459914已实际提交的命令（执行前已实读queue仅1459845正常RUNNING，并检查同name无重复、`bash -n`通过）：

```bash
/opt/slurm/bin/sbatch --parsable --dependency=afterok:1459845 \
  --output=/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_09/fulltrain2651_bound4096_20261023_01.slurm.log \
  /mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_09/fulltrain2651_bound4096_20261023_01.sbatch.sh
```

该helper完整源码另存本地ignored receipts同名文件。它严格要求fulltrain `scope.is_subset=false`、2651clips、globalasset0..7；按asset逐一执行source09 `python -m hoi.v1.native --bundle "$BUNDLE" --asset-id "$id" --num-envs 8 --steps 64 --output "$OUTPUT/native_asset_$id" --hand-collision --self-collision`，再 `python -m hoi.v1.gates --bundle "$BUNDLE" --native-report ...`聚合真实fullbundle hash证据。门通过后执行：

```bash
python -m hoi.v1.resource_probe --bundle "$BUNDLE" --output "$OUTPUT/resource_probe" \
  --init-checkpoint "$LEGACY_IMMUTABLE_24000" --expected-groups 8 --collect-steps 12 \
  --worker-cpu-threads 3 --no-legacy-hand-collision --legacy-self-collision \
  --allow-initialization-physics-change --hand-collision --self-collision
```

`BUNDLE`为上表fulltrain2651最终路径；脚本使用actual fullbank bins分配ΣN4096，全部756524帧驻留与子worker自身asset bank均真实加载，不带subset许可、不缩超参数。任何native失败/physics overflow/OOM立即保留新output并退出。若1459845非成功，依赖不能运行，须先检查生产日志，不能直接解除依赖绕过未完成bundle。

1459914采集长度已冻结为12tick，未覆盖正式64tick replay prefill全部可能延迟workspace分配。独立review未发现P1：WM validation为no_grad逐步B2048×32，静态峰低于已测大图，不必新增该phase；旧actor约39MB/WM约37MB grads与未执行foreach scratch是明确限制，Adam两moment预算正确。root已批准另补64tick，已真实提交1459951，不修改任何源码或超参数：

```bash
/opt/slurm/bin/sbatch --parsable --dependency=afterok:1459914 \
  --output=/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_09/fulltrain2651_resource64_20261023_01.slurm.log \
  /mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_09/fulltrain2651_resource64_20261023_01.sbatch.sh
```

执行前最新queue已核对，仅45RUNNING、14/49正常Dependency，无同名producer。helper本地/远端逐字节SHA256均为 `91c7e6c7adc7478101764713c5ae1cadbba457f201833e0338b665543e55e96a`，`bash -n`通过，本地ignored receipts同名保留。它先严格require `..._09/results/fulltrain2651_bound4096_20261023_01/evidence/native_evidence.json`对应同fullbundlehash的全部8asset，通过后只执行同source09 `resource_probe --collect-steps 64`（其余命令和4096/H32/WM/critic/物理配置与12tick相同），**不重跑8native，不执行optimizer.step**。若1459914失败，51保持依赖不运行，先保留14失败定位；不解除依赖或取消正常producer。正式训练仍禁止。

1459842 读取的真实 v2 标签 bundle 是 `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_04/data/bundle_train1_largebox100_graph_20261023_01/bundle.json`。该诊断使用完整生产网络，但显式 smoke 配置为 8 env、WM batch8、2 iterations；不能外推为 4096 的容量或收敛证据。首次 WM-only gate 保持关闭符合 500 次 warmup 设计。

## 已完成的证据与必要修复

- 全量原始 train2651 的 SHA/schema/60→50Hz 适配审计已 2651/2651 通过，Job1459687，约 4:13；这不是全量几何缓存完成。canonical manifest：`/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_01/data/manifest_train2651_20261023_01/manifest.json`，文件 SHA256 `3db66a8db3b1681271773b359dd451383db7fc6a4c08c6779574edd1c823d6da`。
- 原生实际碰撞边界由 `mjModel.mesh_graph` 导出，不能以原 visual `mesh_face` 代替。largebox100 实测 43 object hull、每手 5 hull；四个独立 `mujoco.mj_geomDistance` 分离姿态与缓存 exact surface 的误差最高约 1.02nm。native30 link FK 与解析 FK 最大位置误差约 0.734µm、旋转矩阵误差约 1.4e-6。
- 手掌 16 个 probe 是真实中央掌面 ray patch，固定源 STL、mount、源码 SHA、source/compiled face IDs、PNG QA 与 evidence hash；没有用随机 whole-hand 点冒充掌面。
- 手 overlay 必须精确绑定 `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_01/results/native_probe_20261023_02/robot_assets/g1_rubber_hand_collision.usda` 及相邻 `hand-asset.json`。CoACD 独立 recook 并非逐位稳定，也不声称 PhysX cooker 相同。
- 共享 zero-shot 原容量 128 contacts / 500 constraints 保留。05 真实 plasticbox080 reference-reset zero-action rollout 出现 NEFC500 overflow；06 在独立 deepcopy HOI cfg 中将容量增至 `nconmax512/njmax2048`，不改 dt=.02、四子步、PD、solver iterations、质量/惯量/摩擦或碰撞开关。overflow 仍立即失败、不得入 replay。配置与真实 readback 写进 physics report。CPU 检验原 cfg 与其他参数不变。
- pair 标签来自实体 pair 的四个 5ms 子步 normal impulse 总和 / 20ms，严格 >1N；手地面负对照不能进入手物体 channel。仅支持已验证 `use_cuda_graph=False` 调度；不是对 runtime 的兼容 monkeypatch。
- 深埋 object center 的旧 calibration 已替换为真实掌面与物体支撑面 2mm 浅压，记录 normal/support vertex/初始 pose/速度/每 interval impulse；物体速度匹配掌面速度。仍保存所有失败和 traces。
- 真正旧 checkpoint 24000 的 immutable 副本为 `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_01/data/initialization/ckpt_self_on_hand_off_024000_e2b6e716c058e5448ab26caa22febc874dd50e2459435f1530f515dcd9fe8d9f.pt`。旧来源 hand OFF/self ON，没有完整旧物理元数据；必须显式声明并授权独立 physics-change 初始化，不能称 resume。
- 实际 checkpoint55 个旧 tensor 前缀逐位一致，新增输入列严格为 0；actor/critic 数值相同，float64 WM 最大约 1.51e-14；float32 扩展 GEMM 改变求和顺序，WM 最大约 9.15e-6，最初 synthetic 阈值下的失败已保留。没有修改模型或静默放宽阈值。严格 v1 save/reload 已通过。

## 八资产与 ID 约定

不可混淆 legacy fixture ID 与正式 merged bundle ID：

| object key | legacy ID | merged bundle ID |
|---|---:|---:|
| largebox_100_100_100 | 2 | 0 |
| largebox_080_080_080 | 3 | 1 |
| smallbox_100_100_100 | 1 | 2 |
| smallbox_080_080_080 | 0 | 3 |
| plasticbox_100_100_100 | 5 | 4 |
| suitcase_100_100_100 | 7 | 5 |
| suitcase_080_080_080 | 6 | 6 |
| plasticbox_080_080_080 | 4 | 7 |

完整实际 graph recipe 已合并于 `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_05/data/recipe_train8_graph_20261023_01/assets.json`。master 是 03 asset2 的 graph hands + 固定手 overlay；object0/1/2/3/5 来自03 exports，4/6/7 来自05完整 exports。几何导出完整不等于对应原生 gate 通过；06 真 bundle 各组仍需重新验收。

## 后续必须完成，才能通知“准备好首次实验”

1. 1459842的fullT220/221状态frozen eval已成功；08renderer路径修复后1459900已完整221帧解码并图像QA通过，直接复用真实saved rollout，没有重复成功WM或eval。汇总raw39FK、32-step diagnostics和保存/reload报告。
2. 1459848已确认八组全部通过。回拷 native.json与实际 `sacct` /日志；fixture-only 的 report 没有 bundle hash，不能作为正式训练 native evidence。
3. resource8的正式global asset IDs0..7已全部通过bundle geometry/FK/GJK/物理readback/浅压pair calibration/64tick及严格同hash聚合门。后续全train bundle仍须相同身份合同检查；不同bundle hash的证据不可直接冒充。
4. 1459905的上述全尺寸资源fixture检查及sacct终态已pass；保存hash索引。每sample的asset/ref/f/history一致、八组均出现，真实history允许同episode prefix padding；不能把8clip distribution外推fulltrain。默认env/batch/horizon没有缩小。
5. resource probe 预留全部参数两份 Adam moments，保留 actor trajectory/grad 与生产 critic locals。报告要明确没有执行 optimizer.step 的 foreach scratch/kernels，也不是10WM/20critic完整迭代吞吐。1s nvidia-smi 样本峰是下界；parent allocator peak精确。按本 job root+children PIDs及实际物理GPU UUID区分自身、同GPU其他进程，过滤其他GPU。
6. 若 OOM 或 physics overflow，保存真实失败与配置，不靠减小默认参数通过；必要时选择集群现有更大 GPU 或有证据的数学等价实现修复，再新 freeze/新 output。禁止改驱动/runtime。
7. resource fixture 各组配额来自8clip自己的 bins，与全2651 quota可能不同。全量 train bundle 完成后，还须确认全量实际 bins/quota、ΣN4096、正式八组 bound native gate与全量资源配置。resource fixture不能替代全数据缓存或正式 distribution。
8. 完整 train2651和eval80缓存、来源/版本/hash/时基/接触mask QA全部通过后，更新最终ready说明和首次20k实验命令。**用户未授权启动正式训练。**

## 本地 ignored receipts

`D:/supermimic_reproduced_hoi/data/hoi_v1_native_receipts_20261023_01/` 已由 Git ignore；`receipt_index_20261023_01.json`记录45个本地元数据文件 SHA。包括：

新增 `receipt_index_20261023_02.json`对当前129个receipt/derived检查文件逐一记录字节数与SHA，不覆盖原45文件索引。

- `checkpoint_migration_02.json`、最初float32失败日志与后续float64/逐 tensor验证日志。
- `graph8_asset_2/` 真实 graph/GJK/FK/pair报告及左右掌面PNG。
- `graph8_03_partial_failure.slurm.log`：旧128/500深压校准失败，保留不覆。
- `light8_05/asset_{0..7}/`：五组完整native报告、三组真实overflow失败、所有 compiled recipe 与已有pair traces。
- `pipeline04_native_gate.json`与`pipeline04_ipc_failure.slurm.log`：真实bound gate已过，随后长TMPDIR导致IPC失败，未执行WM更新。
- `native06_completed_20261023_01/`：06全八组fixture native.json、真resource8逐组native.json、严格aggregate gate/evidence、1459882原始失败resource report和完整日志。压缩receipt SHA256 `8acf52e07ed4a8115bec539dcbca244483f3a9c425e00a49225dba0c31e9fb01`；不含权重/原数据。
- `resource4096_source09_20261023_01/`：成功09资源report、九个phase、request、八worker native报告/derived碰撞recipe与完整Slurm日志；不含checkpoint或原始motion。

后续06/07报告、raw rollout、解码帧数与视频QA另加新文件，不能覆盖现存失败证据；权重和原数据不进入Git，不能上传W&B。
