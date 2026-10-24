# 首条 train motion 的 source39 / ULTRA FK 自洽审计

2026-10-23，独立 CPU 只读诊断完成，返回码 0。范围为冻结 train manifest 中第一条 motion：全部 266 个 60Hz 源状态与 canonical 采样后的全部 221 个 50Hz 状态。没有更新归一化、优化器、参考数据或物理状态，也没有做平移、旋转、轨迹对齐。

按 manifest 中 `source_joint_names` 到原 ULTRA metadata `dof_names` 的显式名称映射，将 source root pose、q 输入已绑定的原 ULTRA 39-body URDF。重建 link pose 与 source630 中保存的 39-body pose 在同一世界坐标比较；额外比较 URDF COM 候选位置，以诊断参考点差异。该诊断只覆盖首条 train motion，不能替代全数据或原生物理验证。

| 比较 | 最大位置误差 | 位置 RMS | 最大旋转误差 | 旋转 RMS |
| --- | ---: | ---: | ---: | ---: |
| 266 个源状态：ULTRA link FK vs source39 | 0.000736 mm | 0.000207 mm | 0.000106° | 0.0000371° |
| 221 个采样状态：ULTRA link FK vs sampled source39 | 0.633 mm | 0.0385 mm | 0.0613° | 0.00397° |

源与采样状态的 body0 位置均与 root position 字段精确相等。原始 60Hz 状态中，COM 候选位置相对 source39 的 RMS 为 98.3mm，最大为 449.9mm；结合 link FK 的亚微米误差，这条 motion 支持既有 DOF 映射和 source39 link-position 解释。它没有支持将 source39 替换成 COM。

50Hz 结果保留 source body pose 独立插值与 q/根姿态插值后再 FK 的非线性差异，没有通过重新 FK 或对齐去消除误差。PHC 仍用保存的 sampled source39 作为 reference，训练原有 human reward 仍用原 14-body FK；两者定义分别记录。

原始报告包含每个 body 的位置/旋转 max、RMS、均值，世界/root-local/body-local 偏移与时间标准差、URDF COM offset、完整 29-DOF 名称映射及输入/脚本 SHA。未为该诊断虚构经验校准的通过阈值，报告状态为 `diagnostic_only_no_empirically_calibrated_acceptance_threshold`。

- 冻结执行源码：`/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_03`，源码 archive SHA `96f4cb22663fedb1aea20afa56c685edba76adfcd5e45280c1985fa7adea03b8`。
- 真实报告与脚本：该目录 `results/source_fk39_audit_20261023_01/`。
- 本地审计副本：[source_fk39_audit.json](D:/supermimic_reproduced_hoi/data/hoi_v1_source_fk39_audit_20261023_01/source_fk39_audit.json)、[audit_script.py](D:/supermimic_reproduced_hoi/data/hoi_v1_source_fk39_audit_20261023_01/audit_script.py)。这些数据保持 ignored。
- 原 manifest 及全量数据审计见 [2651 条 train 数据审计](hoi_v1_full_train_data_audit_20261023.md)。

脚本复现命令（将输出换成全新目录）：

```bash
cd /mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_03
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/mnt/projects/ll/tianzizh/supermimic_reproduced_hoi_selfcollision/envs/kitless-cu128/bin/python \
  data/audit_hoi_v1_reference_kinematics_20261023.py \
  --manifest /mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_01/data/manifest_train2651_20261023_01/manifest.json \
  --recipe /mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_01/results/native_probe_20261023_02/compiled_geometry/assets.json \
  --output results/source_fk39_audit_NEW
```
