# HOI v1 数据准备验收交接（2026-10-23）

后续正式实验已经用户明确授权启动及修复续训；当前source13+expandable的warm1461248/scratch1461253，最新真实验收见[部署与续训记录](hoi_v1_holdout_resume12_20261023.md)。以下为初次数据就绪阶段的历史证据，「未启动/不得启动」为当时约束，不表示当前仍未授权；数据保持不可变，不重建已通过缓存。

**最终状态22:44 UTC：全量缓存1459845、独立CPU审计1459949、完整bundle八资产原生/4096H32作业1459914、64步资源作业1459951全部COMPLETED0:0且各报告passed；source10 CPU preflight通过。首次实验已就绪，正式训练未提交。** 完整身份与命令见[首次实验就绪记录](hoi_v1_first_experiment_readiness_20261023.md)。下文保留历史，不能按旧待完成状态重提作业。

最新只读观测：2026-10-23 20:53:52 UTC。用户明确要求：完整缓存、八资产原生门与4096环境/H32显存验收全部通过后才通知准备就绪；**不得启动正式20k训练**。运行中或正常资源排队不等于完成，也不应重复提交producer。

最新完成状态补充：1459844 resource8 已 `COMPLETED 0:0`，14:25，sacct MaxRSS 3083120KiB；1459846 eval80 已 `COMPLETED 0:0`，13:00，MaxRSS 7912088KiB。两个最终 bundle 已存在。1459845 fulltrain 仍 `RUNNING`（最近sacct 11:01），保持不变。只读共同合同/canonical80/逐clip质量验收 Job1459877 的脚本已成功返回 `passed:true`，7.68秒，使用07 loader；这只是CPU缓存合同通过，原生及4096/H32验收仍由nativeagent完成。

验收报告：`/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_07/results/bundle_contract_audit_resource_eval80_20261023_01/bundle_contract_audit.json`。本地副本 `D:/supermimic_reproduced_hoi/data/hoi_v1_bundle_contract_audit_resource_eval80_20261023_01/`，包含复现脚本。资源fixture为8 clips、2088状态/2080控制；eval80为原canonical80、23798状态/23718控制。全部source/cache SHA、八组geometry与hand字节SHA、manifest/recipe、固定overlay、物理profile、14FK/39URDF共同合同一致，每clip quality及分组汇总重算一致，grid64/FPS128/QA≤1cm保持原值。报告已交nativeagent。

真实参考质量限制（不作为主观pass阈值）：eval80总LR valid为22500/21057，positive11702/9959，unknown1218/2661；plastic100的交点unknown为860/1516（4027个端点），plastic80右手交点unknown722（1618端点），保守mask比例显著。资源fixture的plastic100交点unknown203/233（330端点），plastic80左手positive为0，整个8clip fixture的adjacent-valid onset/release均0。不能把这个资源fixture说成所有phase/双手参考coverage全面；原生actual-pair probe与这些reference intent统计各自报告，不用eval80修改WM门槛。

## 冻结输入、源码与三个producer

数据生成固定使用源码目录 `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_06`，archive SHA `60f6426942c983fba0dfbaf8337a02fa7858b9bd49381ac99eae1ba689ed129c`。07仅metadata/prepare短TMP的小修，physics/geometry相同，**不用重新生成06数据**；后续原生验收按root指定07入口运行。

- canonical manifest：`/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_01/data/manifest_train2651_20261023_01/manifest.json`，字节SHA `3db66a8db3b1681271773b359dd451383db7fc6a4c08c6779574edd1c823d6da`。
- 完整八资产recipe：`/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_05/data/recipe_train8_graph_20261023_01/assets.json`，SHA `462a4571f6d34edf321879ae85550ce1d93b3821e76379b32b674f4cd76994c1`。
- 固定native robot/橡胶手USD：`/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_01/results/native_probe_20261023_02/robot_assets/g1_rubber_hand_collision.usda`，SHA `a86962d71ca54f33dbca4bb02442e2c05f0578d9e4fc76d7243d8cad5c67feea`。不得每worker重新CoACD。
- master手部graph facets、掌面证据、reference URDF取03的 `results/native_graph8_20261023_01/asset_2/compiled_geometry/assets.json`；03 legacy0/1/3/5贡献object，05 legacy4/6/7补齐object。merger严格核对train八个key和实际USD字节。
- 只读Python runtime：`/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128`；不改环境或共享资产。
- CPU外层wrapper：06的 `data/hoi_v1_prepare_short_tmp_20261023.sh`，显式source `cluster/hoi_v1_runtime.sh`并调用 `hoi_v1_short_tmpdir`，再执行冻结的 `cluster/hoi_v1_prepare.sbatch.sh`。

| Job | 范围/资源 | 完成后bundle绝对路径 |
| --- | --- | --- |
| 1459844 | 每asset首条真实train motion，显式resource fixture；24CPU/64GiB，8个clip进程 | `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_06/data/bundle_resource8_graph_20261023_02/bundle.json` |
| 1459845 | 完整train2651；32CPU/64GiB，32进程，4h时限 | `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_06/data/bundle_train2651_graph_20261023_02/bundle.json` |
| 1459846 | canonical eval80；24CPU/64GiB，24进程，4h时限 | `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_06/data/bundle_eval80_graph_20261023_02/bundle.json` |

对应日志在06 `results/prepare_resource8_graph_20261023_02_1459844.out`、`results/prepare_train2651_graph_20261023_02_1459845.out`、`results/prepare_eval80_graph_20261023_02_1459846.out`。

本次观测的实际队列/阶段如下，后续以更新的squeue/sacct与日志为准：

```text
1459845 sm-hoi-v1-trdata RUNNING 7:49 32 ll-l40-2.grasp.maas
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0
2026-10-23 20:51:58,038 2058887 geometry_complete asset=4 elapsed_seconds=116.044085
2026-10-23 20:51:58,038 2058887 geometry_start asset=5 name=suitcase_100_100_100 components=51
2026-10-23 20:53:19,768 2058887 geometry_complete asset=5 elapsed_seconds=81.729473
2026-10-23 20:53:19,768 2058887 geometry_start asset=6 name=suitcase_080_080_080 components=40
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0
```

1459839/40/41是同次刚创建后为统一短TMP wrapper而取消的初始CPU producer；其 `*_20261023_01` partial目录完整保留，不覆盖、不要误当ready数据。没有观测到实际长TMP失败，先核实际TMPDIR本可避免这次重提。替换经过与新job/路径/资源已记于06 `data/cpu_preparation_submission_receipt_20261023_01.json`，本地副本 `D:/supermimic_reproduced_hoi/data/hoi_v1_cpu_preparation_submission_20261023_01/submission_receipt.json`。不要再因假设风险替换正常运行任务。

## 已完成的真实验证与边界

全train2651源字节/schema/60→50 audit已COMPLETED0:0：2651/2651、381 source groups、756524个状态、753873个控制间隔；详见 [完整数据审计](hoi_v1_full_train_data_audit_20261023.md)。源60Hz仍是canonical assumption，不能冒充已查证发布端timebase。

首train clip全部266原始状态与221采样状态的source39/原ULTRA URDF FK只读审计已完成，无对齐、无参考修复；详见 [39-body审计](hoi_v1_source39_fk_audit_20261023.md)。训练human reward继续原14-body FK，PHC保持独立原39-body reference。

精确查询优化没有改变网格64、FPS128、候选数、接触阈值或训练参数。真实221状态的06私有预验证路径与04冻结contact cache中六个输出全部bitwise一致：gap、sampled vertex penetration、surface intersection、expected、expected-valid、geometry-valid。cProfile接触阶段80.081→18.907秒。证据为04 `results/profile_contacts06_20261023_01/`，本地ignored `data/hoi_v1_contact_profile06_20261023_01/`。公共mesh API仍完整校验，静态拓扑/edge只在受信刚体FK路径复用。

已经跨节点比较resource fixture与eval80的首asset与手部cache，字节SHA完全一致：

- `geometry_0000.pt`：`97749846c9b836e22fd094f28b9c0a79fd41ccb086c2325c4bd1300d86f4e162`。
- `hand_geometry.pt`：`8d8677987225b0bb63add7454e8505a9cc3fe3066b564122043628becc8b6fbf`。

八份compiled export齐全只证明可进行CPU几何准备。legacy4/7先前原生probe曾overflow，不能将“8份导出”称为八组完整native pass。所有bundle仍写 `native_physics:not_run`，后续必须用真bundle完成原生门。

## 完成后按顺序验收

1. 先读job的最终sacct、日志返回码、bundle.json。partial的geometry/hand文件存在不代表bundle已完成；bundle.json是最后创建的。正常PENDING(Resources)保持等待，不重复producer。
2. 资源fixture必须是原canonical manifest/hash、八个真实binding、每组一条、`scope.is_subset=true`且purpose为resource fixture。用loader重新验证manifest、recipe、source、native USD、手部receipt、URDF及全部cache字节。该fixture只用于smoke/资源门，不能formal训练。
3. 立即把完成的fixture路径和bundle content SHA交给nativeagent。global asset顺序是large100、large80、small100、small80、plastic100、suitcase100、suitcase80、plastic80；不要套用legacy0..7索引。需要在实际bundle上跑八组mesh_graph/GJK/FK、手/自碰撞、质量COM惯量摩擦重力readback、四substep pair正例与hand-ground负控、完整真实HOI reward有限性，再做用户要求的4096环境/H32显存门。运行或提交不等于通过。
4. 完整train必须覆盖manifest原2651 motion/381 group、无重复/无跨partition、`scope.is_subset=false`、T+1状态/T控制、canonical采样无stretch；重新核每个source/cache SHA。eval80必须为原canonical80身份及完整各motion时长，无optimizer或normalizer更新。
5. 对fixture、train、eval按object key核共同manifest/recipe/native overlay/物理profile/精确scale/mesh-to-object/39 URDF/14 FK合同；比较八个geometry cache及hand cache SHA。bundle content SHA会因分区、clips、输出路径和计时不同而不同，不能要求它们相等。geometry输入与字节不同则停止下游，先解释原因，不静默更换或降低分辨率。
6. 审阅每clip及bundle `reference_contact_quality`：T端点的LR valid/positive/negative/unknown、灰区、surface intersection、严重vertex depth、相邻有效onset/release。原因计数可重叠，不跨clip计phase，没有主观pass阈值。不能把全部unknown说成coverage充分，也不能把source39 flags当actual pair truth。首条T220实际valid216/192、positive116/91、negative100/101、gray4/28、intersection/severe0、相邻有效onset/release0，仅是首条统计。
7. fulltrain bundle的原生门必须绑定它自己的content SHA；resource fixture报告不能直接复用为fulltrain证据。用 `python -m hoi.v1.gates --bundle ... --native-report ...`重复八组，输出全新的evidence目录，要求gate通过并保留各报告字节SHA。WM policy门仍是显式未经验校准的train-only heldout1/8/32 relative-baseline proposal；不能拿eval80调门槛。
8. root汇总完整缓存、八资产真实原生门、4096/H32显存证据和CPU/zero-shot回归后，才通知用户可以跑第一次实验。配置 `configs/hoi_v1_first.json`为上限20000、checkpoint500、独立完整HOI80 eval5000；原250k LR schedule及其他默认不变。**本轮不提交正式训练，不启动20k，不取消/续训旧实验。**

后续真实进度补充：本agent有界单次SSH读到1459845 `RUNNING 22:39`，545/2651（20.6%）条reference/contact cache完成，尚无final bundle。nativeagent报告8组resource-bundle原生门1459882已全部通过，证据在06 `results/resource8_bound4096_20261023_01/evidence/native_evidence.json`与`gate.json`；其后4096/H32资源检查在allocator初始化前失败，属于接口问题而非OOM，source09修复后单独Job1459905再验，不重复八组原生门。资源fixture通过不能代替完整train bundle的content identity、实际frame/bin quotas及完整reference驻留内存验收；fulltrain完成后新的CPU脚本会提供每asset实际clip/frame/control/bin数量供native/root绑定最终资源证据。

已预排完整train新增产物的只读验收：**Job1459949，`afterok:1459845`，2CPU/64GiB/30min**，输出 `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_09/results/full_train_bundle_audit_20261023_01/full_train_bundle_audit.json`。队列已检查无重复producer；1459845最近观测`RUNNING25:41`。脚本在09 `data/audit_hoi_v1_full_train_bundle_20261023.py`，只load新fulltrain，复用并绑定已通过的07 resource/eval audit SHA，不重复二者完整审核。除2651/381/756524states/753873controls、source/cache hashes与逐clip质量重算外，会验证八个worker-filter bank的local→global frame/bin及S/O bitwise映射，给出真正`allocate_workers(...,4096)`配额和central/各worker deduplicated tensor-storage bytes（排除Python/allocator/native/replay/model/gradient，不能冒充显存峰值）。nativeagent已收到同一fullbundle与audit精确路径；已有fullbank4096原生/资源作业1459914按依赖等待，保持不变。

root后续已报告GPU资源Job1459951按`afterok:1459914`预排；GPU脚本/输出与最新状态以nativeagent交接为准，不再创建重复producer。根聊天已安排15分钟后续自动验收，数据agent可在没有新真实错误时停止频繁SSH；本轮仍未启动正式训练。

### 审计脚本与helper字节SHA

| 文件（数据脚本在各自新data目录，不是冻结源码修改） | SHA256 |
| --- | --- |
| 09 `data/audit_hoi_v1_full_train_bundle_20261023.py`，Job1459949 | `dbe0a9f4d2ec53e43f9a38fd3107fb825c34e3c926abc5c4ec7d24d94c041863` |
| 07 resource/eval审计的已保存 `audit_script.py` | `6b3ddd880aab038db8a2305a40438f337724ff70b900c2cb8cb186a705216ae1` |
| 06 `data/hoi_v1_prepare_short_tmp_20261023.sh` | `327e0f52eb5f1334328999764f4c88f794fb08fb3d2ae299f94b31c44830f582` |
| `cluster/hoi_v1_runtime.sh` helper | `bd855812d9597204f924c91856cde9757e7726675ed995d677ca7c8d88430d49` |

另已用纯stdlib审计同一冻结manifest：train2651/381 source groups与eval80/80 groups、test40/40 groups在motion身份及完整source-group两个层面交集均为空，结果`passed:true`，未使用eval/test更新归一化、优化器或门槛。共有的八类object资产是设计允许的共同合同，不冒称asset-heldout泛化。receipt在09 `results/manifest_train_eval_isolation_20261023_01/isolation.json`，本地ignored `data/hoi_v1_manifest_isolation_audit_20261023_01/isolation.json`。

## 不可改的边界

保留旧zero-shot入口及历史目录、代码、数据、环境和结果。不修改冻结06/07源码、不覆盖任何输出、不升级runtime、不重烹固定手部USD。遇到真实错误先保留现有证据并定位；代码修复使用新快照和全新输出。正常排队或SSH观测不可用都不说明计算失败。

连接状态补充：root在上传08archive后曾观测SSH port22 timeout，随后一次有界SSH已恢复并确认08部署SHA；root恢复观测时CPU1459845 `RUNNING 17:12`、GPU1459882 `RUNNING 5:23`。继续采用ConnectTimeout10与keepalive、合理间隔读取，不并发反复SSH，不把临时观测不可用当作作业失败。resource8/eval80及07共同合同审计已完成，无需重复；fulltrain新增产物完成后才进行其全量loader/质量/共同合同验收。1459845/1459882保持原样，08部署不授权覆盖既有数据或启动正式训练。
