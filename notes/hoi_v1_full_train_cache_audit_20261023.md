# HOI v1 全训练集缓存验收（2026-10-23）

**全量缓存及独立 CPU 合同审计已通过；后续22:44 UTC完整bundle八资产原生门、4096/H32和64步资源补验亦全部通过，正式训练未启动。** 最终状态见[首次实验就绪记录](hoi_v1_first_experiment_readiness_20261023.md)，下文22:15观测保留为过程证据。

实际观测时间为 2026-10-23 22:15 UTC。缓存 Job1459845 为 `COMPLETED 0:0`，耗时1:21:48；依赖 CPU 审计 Job1459949 为 `COMPLETED 0:0`，耗时1:16，报告 `passed=true`。当时1459914完整 bundle 原生/资源检查正在运行，1459951正常等待上游成功，不重复提交或解除依赖。

## 不可变输入与报告

- 缓存源码 source06 archive SHA256：`60f6426942c983fba0dfbaf8337a02fa7858b9bd49381ac99eae1ba689ed129c`。
- 最终完整 train bundle：`/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_06/data/bundle_train2651_graph_20261023_02/bundle.json`。
- bundle 文件 SHA256：`815ed700b8cda2b382a37087ae43619615a38867325de4c51216c99f0848c157`；content SHA256：`318482c78f699d33aa5ae7743f2918f19d96fa3d78e2869add50f4a890d022c8`。文件与 content hash 的定义不同，不互换。
- 审计源码位于 source09 `data/audit_hoi_v1_full_train_bundle_20261023.py`，SHA256 `dbe0a9f4d2ec53e43f9a38fd3107fb825c34e3c926abc5c4ec7d24d94c041863`。root已阅读脚本，未重新执行已通过的缓存生产或 loader 审计。
- 完整审计报告：`/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_09/results/full_train_bundle_audit_20261023_01/full_train_bundle_audit.json`。
- 本地 ignored 逐字节副本：`D:/supermimic_reproduced_hoi/data/hoi_v1_full_train_bundle_audit_20261023_01/full_train_bundle_audit.json`，SHA256 `75eb67e8c5b8e6377d89e60d040a2fc1e4df23ab59ffc47520d192b7dfd3d495`。
- 已通过的 resource8/eval80 共同合同报告 SHA256：`405a7a887398bbe0c28b505feeab48e791fe56fe0d27ec247ba9bdf56d57649a`。本次复用该身份作为对照，不重复它们的完整审核。

## 已确认的合同与真实配额

全部2651条、381个 source groups、756524个50Hz状态、753873个控制间隔符合冻结 manifest；`scope.is_subset=false`，没有额外动作或丢失动作。`load_bundle`重新核对绑定 source/cache/资产字节和时基，逐 clip 接触质量重新计算与 bundle 汇总完全一致。

完整 train 与已审 resource8/eval80 的 manifest、recipe、固定橡胶手 overlay、真实物体质量/COM/惯量/摩擦、exact scale、mesh transform、39body URDF、14body FK、八份 geometry cache 及 hand cache 合同一致。网格仍64³、shape点仍128；最大网格误差约5.20mm，小于原1cm QA上限。没有降低分辨率、接触阈值或训练尺寸。

| global asset | clips | states | controls | bins | 实际4096环境配额 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 largebox100 | 490 | 142905 | 142415 | 3056 | 770 |
| 1 largebox80 | 381 | 110249 | 109868 | 2347 | 592 |
| 2 smallbox100 | 297 | 81106 | 80809 | 1761 | 444 |
| 3 smallbox80 | 100 | 23303 | 23203 | 506 | 128 |
| 4 plasticbox100 | 292 | 81764 | 81472 | 1780 | 449 |
| 5 suitcase100 | 323 | 91420 | 91097 | 1963 | 495 |
| 6 suitcase80 | 469 | 143874 | 143405 | 3057 | 770 |
| 7 plasticbox80 | 299 | 81903 | 81604 | 1778 | 448 |
| 合计 | 2651 | 756524 | 753873 | 16248 | 4096 |

八个 worker 的 local→global frame、bin 映射及 robot/object reference 状态已逐张量 bitwise 校验。这是 loader/分区合同证据，不是 GPU 原生检查。

CPU 去重 tensor storage：central bank为1,101,952,520bytes，全部 worker 分 bank 合计同为1,101,952,520bytes，central shape bank为9,788,088bytes。该数字排除 Python、allocator、物理场景、replay、网络及梯度，不能称为实际显存峰值。

## Reference intent 质量及边界

| 全 train 的 T 个端点统计 | left | right |
| --- | ---: | ---: |
| valid | 702786 | 665709 |
| positive | 412715 | 378968 |
| negative | 290071 | 286741 |
| unknown | 51087 | 88164 |
| gray unknown | 10684 | 20384 |
| surface intersection | 40403 | 67780 |
| severe vertex penetration | 9 | 31 |
| 相邻有效 onset | 262 | 301 |
| 相邻有效 release | 131 | 99 |

unknown原因可重叠；onset/release只连接同clip相邻有效端点，不跨clip拼接。严重 vertex penetration 已按原规则 mask，不为了通过审计删除动作或降低阈值。两个 plastic 组存在显著交点 unknown：plastic100左右20602/35085，plastic80左右18502/31669；仍保留逐组报告。

这些是几何 reference intent，不是实际实体 pair normal impulse。CPU 审计没有主观“接触覆盖充分”门槛，也不预判训练 WM policy gate一定开启。源60Hz仍为冻结canonical协议假设，不倒写为发布数据实际生成时钟已证明。

## 剩余就绪条件

1459914须对本完整bundle自己的content SHA通过全部八组真实原生门，并以本表实际配额运行4096/H32资源检查；1459951随后在同bundle下执行64步真实采集及完整尺寸图，均0 optimizer step。只复用同完整bundle已成功的native证据，不将resource8报告改名冒充。

两项均通过后再给出首次实验ready结论及真实路径命令。正式配置保持20k、checkpoint500、完整canonical80评估5000，原LR decay250000及其余生产参数不变；本轮不提交正式训练。
