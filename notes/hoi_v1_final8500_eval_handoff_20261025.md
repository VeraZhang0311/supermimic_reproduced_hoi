# HOI v1 新版8500pt最终验收

2026-10-25整理说明：本文7000pt数值仅作历史比较。旧版及8500以前的checkpoint/HOI80完整输出已按用户授权清理，不能再按旧路径加载；8500 checkpoint、完整80、20条媒体与必要输入/原生证据保持。精确清单与保留SHA在本地`data/cleanup_20261025/`；共享数据、资产、runtime及冻结source15未改动。

验收日期：2026-10-25（America/New_York）。开发基线为 `codex/hoi-v1-no-quality-gate`：无持续WM质量门、双手碰撞和身体自碰撞开启、ULTRA同源整脚STL凸分解碰撞、五通道learned contact head。完整canonical HOI80和固定20条离线视频均已完成验收；本次监控已暂停。

## 冻结模型、数据与物理

| 项目 | 固定身份 |
| --- | --- |
| 原源码快照 | `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_15` |
| Source content SHA256 | `1bf7c0cfad3e9b64d2bf460112db3d85f5733887b59e58a8336613f57439e7fd` |
| 训练目录 | 上述快照的 `results/hoi_v1_contact5_ultrafoot_scratch_20k_20261023_01` |
| 最后完整checkpoint | 训练目录的 `checkpoint_008500.pt`，234382121 bytes |
| Checkpoint SHA256 | `ea3447d7425337265bc967fb57a06966334a5f9d21e54c476308cb62c71a7dd0` |
| 初始化及更新数 | 全部from-scratch；8500 iterations、85000次WM更新、8000次policy更新 |
| Feature / contact schema | `supermimic_hoi_v1_full_reference_2` / `hoi_v1_pair_normal_impulse_v2` |
| Contact通道顺序 | 左手—物体、右手—物体、物体—地面、左脚—物体、右脚—物体 |
| 手/身体碰撞 | hand ON / self ON，保留相邻及焊接link过滤 |
| 足部profile | `ultra_g1_stl_coacd5v16_v1`：原ULTRA脚STL与安装变换，每脚5个实际凸块、每块最多16顶点，替换原四球，保留质量/COM/惯量 |
| 训练数据 | 2651 motions / 381 source groups，756524个50Hz状态、753873个control intervals |
| Train bundle | 原源码快照的 `results/contact5_ultrafoot_validation_20261023_01/bundle_train/bundle.json` |
| Train bundle content digest | `e88b4e69a23d4f95555949f16233cbb79e509367e5819327ff5d15fe6625c983` |
| Eval80 bundle | 同一验证目录的 `bundle_eval80/bundle.json` |
| Eval80 bundle content digest | `adc9875cbe823546efc151b1ad9d07589a71808636675687ee549c0f790e0715` |

Contact head为共享WM backbone后的 `2048→128→5` learned logits；真实标签由每个20ms区间内4个5ms子步的实体pair法向冲量计算，平均法向力严格大于1N为正。脚—地面不计入脚—物体。Reference expected-contact/valid、contact-match reward和hand-approach仍只定义双手，没有新增脚接触奖励。该CoACD结果与作者PhysX VHACD并非相同cooker或逐位相同碰撞。

实际正式launcher对parent与8个worker显式使用 `PYTORCH_ALLOC_CONF=expandable_segments:True` 及兼容别名 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，OMP/MKL各3线程；这些运行参数属于已验收启动合同。HOI原生碰撞/约束容量为512/2048，overflow立即失败，不污染replay。

训练保留64 vector-step fresh replay prefill及500 iteration WM-only；随后固定每轮更新actor/critic，holdout质量仅作诊断，native/数据身份和非有限值检查仍为硬错误。正式目标20k、每500保存、每5000自动完整80；训练作业1461591因24h时限结束，最后完整保存8500pt，并未完成20k。checkpoint未保存完整replay、模拟器和RNG，续训不等于逐位恢复。

新版运行前独立验证1461502为 `COMPLETED 0:0`。八资产严格native证据SHA256为 `8b40f0958b504b58b598d22a7bbed5b532c4e58ff0b851f2f8b9dbc7b44ed9ab`；源快照的 `results/contact5_ultrafoot_validation_20261023_01/evidence/native_evidence.json` 绑定逐资产报告。4096环境/H32/replay1000/64prefill完整生产反传检查的报告位于源快照的 `results/contact5_ultrafoot_resource64_20261023_01/resource_validation.json`，SHA256 `2e26b8615526f0ddef6e05e06d1ce24d9b197e65a42bba8016d613b066dde1cd`。资源探针optimizer steps为0，不能冒充训练收敛证据；8500存盘中的8000次policy更新是后续正式训练事实。

## 完整HOI80结果

本次操作根：`/mnt/projects/ll/tianzizh/supermimic_hoi_last_new80_20261025_01/new8500`。评估1466531为 `COMPLETED 0:0`，Slurm耗时42m18s，报告内部耗时2475.755608s。`results/evaluation.json` SHA256为 `8cfe6c6da1029133d1d1830a982967b9da80f2e41146cae0e0664ffca0f0f119`。

核验通过：80个唯一canonical case、23718个完整control intervals、每条T+1状态/T控制、原source_frames对应完整50Hz时长、no_failure_reset、physical_objects、optimizer updates 0、normalizer updates 0。checkpoint/source/bundle身份及共同39体评分实现均匹配，aggregate由逐case重算一致。

| 完整80指标 | 同版7000pt历史数值 | 最后8500pt |
| --- | ---: | ---: |
| PHC通过 | 76/80（95.00%） | 77/80（96.25%） |
| 人体39体世界位置误差 cm | 10.6855 | 10.2610 |
| 双手位置误差 cm | 11.2602 | 11.0194 |
| 物体位置误差 cm | 11.1637 | 8.2924 |
| 物体角度误差 deg | 26.5197 | 22.0318 |
| 关节RMSE rad | 0.202301 | 0.200123 |
| Object-pose通过 | 27/80（33.75%） | 33/80（41.25%） |
| PHC+object-pose联合通过 | 27/80（33.75%） | 33/80（41.25%） |
| 作者人体终止数 | 1/80 | 2/80 |
| HOI终止数 | 2/80 | 3/80 |
| Human normalized return | 0.819323 | 0.825239 |
| HOI normalized return | 0.758767 | 0.777314 |

7000pt比较仅保留已验收数值；不依赖旧checkpoint、HOI80或视频文件。误差先对每条动作求平均，再对80条等权宏平均，无独立轨迹对齐。PHC要求所有评分帧的39体平均世界位置误差不超过0.5m；物体位置误差超过0.1m或角度超过30deg持续至少0.3s即object-pose失败。这是工程姿态判据，不等于抓取成功。作者终止数与PHC为不同判据，分别报告。

| 8500pt物体类别（各20条） | PHC通过 | Object-pose/联合通过 | 物体位置 cm | 物体角度 deg |
| --- | ---: | ---: | ---: | ---: |
| largebox | 20 | 11 | 5.3501 | 23.6259 |
| plasticbox | 20 | 9 | 8.1714 | 19.0965 |
| smallbox | 17 | 2 | 15.3933 | 30.8157 |
| suitcase | 20 | 11 | 4.2547 | 14.5891 |

[本地8500数值summary](../data/hoi_v1_last_new80_20261025_01/new8500_curated_summary.json) SHA256：`4dc35439662c57da9e143998c627762807ebc1df5016b83adfaab9778d99e0a7`。该文件保留逐case数值和共同评分实现SHA，可在清理旧数据后独立查阅。

私有W&B：[8500冻结评估](https://app.forge.coreweave.com/tianzizh-university-of-pennsylvania/SuperMimic_HOI_v1_20261023/runs/f5b01592674447e1b641ae67e998c1b0)；[新版from-scratch训练](https://app.forge.coreweave.com/tianzizh-university-of-pennsylvania/SuperMimic_HOI_v1_20261023/runs/61f872168c46457f9fc74bfe9d63ea9b)。每个对照独立run，仅配置、指标及汇总，不上传权重、源码、原始数据、完整rollout或视频。

## 固定20条视频与验收证据

[20条合集](../data/hoi_v1_last_new80_20261025_01/new8500/render20/all20.mp4) · [逐条视频索引](../data/hoi_v1_last_new80_20261025_01/new8500/render20/index.html) · [20条中帧总览](../data/hoi_v1_last_new80_20261025_01/new8500/render20/contact_sheet.jpg)

四类分别选择080/084/089/094/099，经canonical manifest映射，不按成绩选。唯一render1466607为 `COMPLETED 0:0`、28m55s；远端输出为本次操作根的 `render20/`。使用原source15 `hoi.v1.render` 回放保存状态：1280×720、50fps、全部T+1帧、0 policy inference、0 physics step、0 FK reconstruction，原USD视觉，无碰撞几何替代。Actual为原色机器人/橙物体；reference机器人和物体RGBA为 `[0.08,0.43,1,0.35]`，共享原世界坐标及固定相机。

20单条和合集均在远端及本地完整解码，通过逐文件SHA核对。合集5260帧、105.20秒，SHA256为 `63c766f3dac2669d9f4844403f3f97ae91f6c9f34b3784670be00f72ce659c9d`。实际查看20中帧contact sheet与四类080首/中/末12帧，原USD解析、完整时长和配色通过；检查帧无类别缺失或裁切，真实人体/物体漂移保留，初始重合表面叠纹未通过移轨迹隐藏。

本地证据均位于 `data/hoi_v1_last_new80_20261025_01/new8500/`：

- [完整媒体与解码回执](../data/hoi_v1_last_new80_20261025_01/new8500/local_media_verification.json)：21个视频完整解码及逐条SHA。
- [实际图像QA回执](../data/hoi_v1_last_new80_20261025_01/new8500/root_visual_QA.json)：20中帧与12首中末帧实看记录。
- [四类首中末图](../data/hoi_v1_last_new80_20261025_01/new8500/visual_QA_four_classes.jpg)。
- [远端渲染完成记录](../data/hoi_v1_last_new80_20261025_01/new8500/render20/rendering.json)：原模型/源码/评估身份、20条映射、完整帧数与媒体SHA。

## Batch实现与运行边界

评估组内是batch inference与并行物理，8个资产组依次执行。对应largebox100/largebox80/smallbox100/smallbox80/plasticbox100/suitcase100/suitcase80/plasticbox80的batch分别为15/5/16/4/14/9/11/6；既非逐环境串行，也非80条同时跑。组内推进至最长clip，各条只保存和评分自己的完整时长。

每组独立worker初始化；逐tick有CPU/GPU与pipe同步，物理结束后还有逐clip评分、落盘及执行action的WM诊断。原日志无phase profile，不能断言某一项占主要耗时或虚构比例。完整评估协议未因吞吐疑问改变。

原始数据的真实生成间隔仍未证实，结果表述为canonical 60Hz解释、50Hz控制；见[数据时基交接](ULTRA_DATASET_AND_TIMEBASE_HANDOFF_20261023.zh-CN.md)。本报告证明完整评估与媒体交付，不证明20k完成、策略收敛或抓取成功。后续开发依据[最终设计](supermimic_hoi_v1_final_design.md)，实验使用独立新输出，不覆盖本次冻结证据。
