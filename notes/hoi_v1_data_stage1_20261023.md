# HOI v1 第一阶段：数据层实现与 CPU 验证

日期：2026-10-23。分支：`hoi/v1.0`。本轮只实现数据层，没有启动训练、policy 推理、物理仿真或 Slurm 作业。

## 已实现

- 新增独立 `hoi/data/`：纯标准库 manifest 审计、纯 Torch reference adapter 和 CPU CLI，使用方式见 [README](../hoi/data/README.md)。
- 冻结 split / selection / canonical panels 的 SHA、4066 条 motion 身份和 source-group 隔离；canonical reader 还校验固定 membership digest，不能只靠重算 self hash 冒充冻结清单。
- 保留原 source motion 路径和 SHA，显式 prefix remap 仅改变读取位置。新输出 fresh/exclusive，拒绝写入 source roots；原始动作和既有结果不覆盖。
- 精确整数 `T=5*(N-1)//6`、T+1 状态、float64 control clock；机器人/物体/39 bodies 同一采样网格，translation 连续插值、quaternion SLERP、源 contact 最近邻。
- 按来源声明的 joint names 重排 q/qd；源 velocity 与 control-grid pose 差分 velocity 分开保存，保留 source reference-point unknown 状态。
- `weights_only=True` 解码同一份经过 SHA 校验的字节，拒绝 checkpoint/rollout 容器、坏 schema、非有限数、零/非单位四元数及非二值 contact。

原 `hoi.run` / `hoi.prepare` / `hoi.evaluate` / `UltraReference` 和 `supermimic` 的网络、训练、奖励、PD、物理配置未修改。只在 `hoi/README.md` 增加新入口链接；`.gitignore` 为 `hoi/data/` Python 源码目录加例外，根 `data/` 的源数据、manifest 和派生物仍被忽略。

## 实际源身份核验

基于已有 known_hosts，以只读 SSH 读取 `stat/sha256sum/cat`，没有在 login 节点运行 Torch 或 GPU。下载到全新、被 Git 忽略的目录：

```text
D:/supermimic_reproduced_hoi/data/hoi_v1_source_validation_20261023_3d696390/
```

该目录保留 selection、eval80、一个源 motion 和 `READ_ONLY_SOURCE_VALIDATION.json` 收据。

| 输入 | 本轮核验 |
| --- | --- |
| 真实训练 selection | SHA `8c141eb60650b9df79b92acfeeea785f796c7f18f52bf179467562c7cc315b23`；2651 motions / 381 groups，与本地 train 五个身份字段逐项一致 |
| 实际集群 eval80 | SHA `459a3925faf0f66bee258ce98635c67802ddda00d47d007ecdeb45035f576002`；整份 JSON 与本地 canonical 清单相同 |
| 源样例 | `sub10_largebox_000_075_086_074_100_100_100.pt`，671819 bytes；SHA `1a0bac419adab2bf6d7f5d10d480e40d0f3c39a74605a7063c7f48b0f99bf97c` |

四分区 JSON 审计为 train2651/381、validation517/80、regression655/100、test243/40；总计4066条，source-group互斥。eval80/test40 始终是 held-out partitions 中的 canonical 子集。

## 真实 motion 的端到端结果

```text
manifest: D:/supermimic_reproduced_hoi/data/hoi_v1_stage1_20261023_4f41b499/manifest/manifest.json
report:   D:/supermimic_reproduced_hoi/data/hoi_v1_stage1_20261023_4f41b499/validation/validation.json
bindings: D:/supermimic_reproduced_hoi/data/hoi_v1_stage1_20261023_4f41b499/source_bindings.json
```

按 `--partition train --limit 1 --cache` 执行新 CLI，结果通过。**这是1/2651的 motion 字节解码与采样 subset，不是全部4066文件的 tensor 或 native 验收。**

| 项 | 结果 |
| --- | --- |
| 源 tensor | float32 `[266,630]`，有限；SHA 与冻结清单相符 |
| Canonical duration | `(266-1)/60 = 4.416666666666667s` |
| 完整 control intervals | 220；仅声明时间区间，没有生成或执行 action |
| State/reference frames | 221；末 control time=4.4s，尾余=1/60s |
| Control time | float64；与 `source_index=time*60` 一致至浮点精度 |
| 派生 cache | 安全 `weights_only=True` 回读通过，shape `[221,630]`，source bytes 未改变 |
| Manifest content SHA | `927934870c892d988778e11fdfe7264e4ba29ce16399ba6ac32c7ffa6ebbbcc0` |
| Cache SHA | `e5f0ea65f351250ea1d815d30f03efdca52b31bcd78ea34afd50bddd999e8809` |

Joint order 读取封存外部 fixture 的 `metadata.dof_names`，其文件 SHA 为 `9ce8e0c00698b17602642a0e44d542e3f4cd6e7a3ccbe7aaa6a5dc72e867a625`；路径和用途写入 bindings。它是外部 source-order 声明证据，不是该 motion tensor 自带名称，也不证明当前 native asset 顺序或所有文件的生成顺序。

## 检查与剩余边界

使用已有 `D:/SuperMimic/.venv/Scripts/python.exe` CPU 环境，不修改该环境。执行：

```text
python -m unittest discover -s tests -v
python -m unittest hoi.test_reference hoi.test_hand_collision -v
```

新测试覆盖数据 schema/clock/joint mapping、分区泄漏、SHA与membership、输出保护、CLI真实序列化与缓存回读、损坏输入失败报告、601→500区间/501状态，以及与原 world-SO(3) 差分定义一致的非交换旋转。常规 CPU 回归共60项：59通过、1跳过；旧 HOI 边界检查6项全部通过。跳过项为 MuJoCo 可选 CPU 检查，该解释器未安装 MuJoCo；未运行 native 物理检查。

此阶段 `adapter_ready` 只表示提供了可执行的 joint mapping；`training_ready=false`。发布文件真实生成间隔、源/native velocity 参考点、几何/scale/COM 变换、pair contact 与 full-dataset 解码仍需后续验证。时间基准仍准确表述为 canonical60Hz解释、50Hz控制，未因读取真实文件而宣称生成时钟已证实。
