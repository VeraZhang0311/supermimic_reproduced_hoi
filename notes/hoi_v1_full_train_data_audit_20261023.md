# HOI v1 全训练集 CPU 数据审计

日期：2026-10-23。分支：`hoi/v1.0`。本次只检查源数据字节、schema、关节重排及 canonical 60→50 Hz 时间采样；未训练、未执行物理或评估模型。

独立开发目录：`/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_01`。原有 selfcollision Python runtime 仅只读执行，没有修改环境、历史源码、数据或结果。作业入口为 `cluster/hoi_v1_data.sbatch.sh`，CPU2、12G、无GPU，使用一个 Torch/BLAS 线程，不保存适配 tensor 缓存。

Slurm **1459687**：`sacct` 实读 `COMPLETED`、`ExitCode=0:0`、耗时 **00:04:13**；分配 `billing=2,cpu=2,mem=12G,node=1`。这证明 CPU 数据检查完成，不代表 native 或训练门禁通过。

| 检查范围 | 实读结果 |
| --- | --- |
| Frozen train | 2651/2651 motions，381 source groups，非子集，错误0 |
| 源文件总字节 | 2,294,586,216 bytes |
| 源状态帧总数 | 908,982；单 clip 范围176–816 |
| 完整控制区间总数 | 753,873；单 clip 范围145–679 |
| 控制网格状态总数 | 756,524，逐 clip 为 `T+1` |
| 源协议时长 | 累计15,105.516667秒，按各 clip 的`(N−1)/60` |
| 完整50Hz控制时长 | 累计15,077.46秒；尾余合计28.056667秒，没有拉伸末尾不足一个tick的区间 |

每个 motion 校验登记 SHA 后，解码同一份 bytes 为有限浮点 `[N,630]`；检查 root/object/39-body quaternion、二值源 contact，并按 metadata joint names 重排机器人关节。机器人与物体共用时间、索引及插值权重；pose-derived v/omega 与保留的源速度分开记录。完整逐文件 metadata 和 source/derived 数值差异保存在报告。

冻结四分区、训练 selection 与 cluster eval80 同本地 canonical 身份逐项核对；本次只加载 train tensor，未把 validation/regression/test motions 并入训练审计。源 joint order 来自冻结外部 fixture metadata，其 `native_verification=not_performed` 状态没有改写。

远程结果：

- Manifest：`data/manifest_train2651_20261023_01/manifest.json`，SHA256 `3db66a8db3b1681271773b359dd451383db7fc6a4c08c6779574edd1c823d6da`。
- 报告：`results/data_train2651_20261023_01/validation.json`，SHA256 `1547097d454c11da5ddd2187517222f4f6efe8bebf8e93593fa9e9ddd123284d`。
- 执行身份：同目录 `execution_receipt.json`，记录 Python3.12.4、Torch2.11.0+cu128、源码/启动器 SHA 和只读 runtime 路径。receipt 在提交后读取记录，没有声称恢复 Slurm/模拟器/RNG。
- 日志：`.slurm/hoi-v1-data-1459687.out` / `.err`。成功日志为 `success=true, validated_motions=2651, errors=[]`。

已把上述 manifest、完整逐文件报告和 receipt 复制到本地 Git 忽略目录 `data/hoi_v1_full_train_data_audit_20261023_01/`，下载报告 SHA 与远程一致；源 `.pt` 没有复制进仓库或提交。源码 receipt 与本地对应数据适配/rotations/启动器字节一致。

科学边界保持不变：发布源的真实生成间隔没有因这次检查得到确认，仍是 `canonical_60hz_protocol_assumed_not_generation_verified`；源 velocity 的物理参考点未确认，差分差异只作数值审计。碰撞/物性、掌面、pair contact、全量几何标签、WM和真实 simulator 交互仍使用各自独立门禁。本次没有生成全2651条几何标签，也没有启动训练。
