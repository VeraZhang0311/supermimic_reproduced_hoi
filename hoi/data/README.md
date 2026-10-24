# HOI v1：CPU 数据适配与时间校验

本目录实现 v1 的第一阶段：ULTRA 原始 reference 解码、冻结清单审计、连续时间 60→50 Hz 采样和独立派生缓存。不会导入 IsaacLab、加载 policy 或启动训练。原 `hoi.run`、`hoi.prepare`、`hoi.evaluate` 和旧 `UltraReference` 保持各自的 zero-shot 协议。

## 两个入口

```bash
python -m hoi.data.prepare manifest --help
python -m hoi.data.prepare validate --help
```

`manifest` 只用 Python 标准库读取 JSON，不需要 Torch。`validate` 需要已安装 Torch 的 CPU Python。Windows 本轮使用现有的 `D:/SuperMimic/.venv/Scripts/python.exe`，只执行测试，不安装或修改该环境。

### 1. 建立独立 manifest

```bash
python -m hoi.data.prepare manifest \
  --profile canonical-v4 \
  --splits /path/to/frozen/data_manifest.json \
  --eval-cases /path/to/frozen/evaluation_cases.json \
  --test-cases /path/to/frozen/test40_cases.json \
  --train-selection /path/to/exact/selection.json \
  --run-eval-cases /path/to/cluster/evaluation_cases.json \
  --bindings /path/to/source_bindings.json \
  --path-map /mnt/projects/ll/tianzizh/ProtoMotions/data/omomo/OMOMO_retarget_aug_v4=/path/to/local/source \
  --output /path/to/new_manifest_directory
```

各路径替换为已保存的真实文件；`--output` 必须是全新目录。源文件、源 JSON 不修改，原 motion 路径和 SHA 保留在 manifest 内；`--path-map` 仅显式改变实际读取位置，不按 basename 搜索替代文件，不允许 `..` 逃逸。

- `canonical-v4` 固定四分区 / canonical eval80 / test40 的已知 JSON SHA、分组数量，以及全体 motion/panel identity digest。提供训练 selection 时同时固定其已知 SHA，并逐项核对它与 train 全集相同。
- `--run-eval-cases` 可核对另一个实际运行目录的清单：case ID、motion SHA、source group、object/asset 和帧数都必须与 canonical eval80 一致。不会根据条数相同就假定清单等价。
- `custom` profile 允许小型 fixture 或明确的新清单，但不会标记成 canonical v4。所有四分区都必须存在且非空，增强 variants 不得跨 source group 分区。
- 训练 selection、source joint bindings 可以暂缺，用于先审计已有 JSON；缺失状态写明。缺少 joint bindings 时 `validate` 会拒绝猜测关节顺序。`training_ready` 在本阶段始终为 false。
- Source tensor 尚未读取时，manifest 明确 `motion_bytes=not_read`；manifest 成功不等于全部 4066 文件已经验证。

### 2. 绑定 source joint order

源 `.pt` 是没有关节名称的 `[N,630]` tensor，source 顺序需要外部可信 metadata；不能从当前机器人 runtime 顺序反推。`--bindings` 是 JSON，格式为：

```json
{
  "source_joint_names": ["这里填写真实 metadata.dof_names 中的完整29项"],
  "joint_order_evidence": {
    "kind": "external_fixture_metadata",
    "fixture_path": "/path/to/frozen/fixture.pt",
    "fixture_sha256": "真实文件SHA256",
    "field": "metadata.dof_names",
    "scope": "external_fixture_not_embedded_in_each_motion",
    "native_verification": "not_performed"
  },
  "objects": {}
}
```

上面是结构说明，单项占位列表不能直接使用。实际 source 名称必须恰好覆盖现有29个G1关节；target order 以 AST 读取 `supermimic/robot.py` 的既有定义，避免导入仿真环境。适配器按名称重排 q 和源 qd，同时保留 `source_sampled` 的原始列顺序。

`objects` 可保存经过来源绑定的声明，但本阶段不从 asset 名字推断物理缩放或 COM/mesh 变换。比如 `_080_080_080` 可能已经烘焙在几何里，不能再把它与 USD spawn scale 重复相乘。本阶段不声明 asset transforms 或 native physics 已验证。

### 3. 校验 motion；可选保存缓存

```bash
python -m hoi.data.prepare validate \
  --manifest /path/to/new_manifest_directory/manifest.json \
  --partition train \
  --limit 1 \
  --cache \
  --output /path/to/new_validation_directory
```

`--partition` 支持四分区或 `eval80` / `test40` panels；默认 train。`--limit` 是清单顺序的显式 prefix subset，报告 available/selected/validated 数量；不加时验证所选分区全部 motion，不跳过缺失/坏文件。CLI 成功返回0，数据失败且已写报告返回1，配置错误返回2。

每个文件先按准确字节核对 SHA，再对同一份 bytes 使用 `torch.load(weights_only=True,map_location='cpu')` 解码。仅接受有限、dense floating `[N,630]` tensor；checkpoint/rollout 容器拒绝。零/非单位 quaternion 和非二值源 contact 拒绝。没有两个完整 control-grid 状态的短 clip 拒绝，不填造速度。

输出包含：

- `validation.json`：source/resolved 路径、字节 SHA、schema、N/T/T+1、时钟、尾余、joint order evidence、源/派生速度数值差异及错误。
- `--cache` 时另存 `reference_<motion_sha>.pt`，每个记录自身 SHA 和源 manifest/content SHA。源数据不覆盖；输出不能放入 source root。即使验证失败，已有审计目录也保留，不复用或清理科学结果。

速度差异只是数值诊断：源 velocity 的 link/COM 参考点仍未证实，不将差值小、有限或 clock 对齐描述为物理语义已证实。

## 时间与缓存 schema

固定 `canonical_60hz`，控制50Hz；这是源时间解释协议，不证明发布文件由1/60秒生成，也不自动采用当前 exporter 的17ms。

```text
T = 5 * (N - 1) // 6
state/reference times = arange(T+1, float64) / 50
source index = 6 * tick / 5
last fractional control interval is not stretched
```

源索引的 i0/i1/alpha/nearest 由同一整数 clock 生成，机器人与物体共用。位置及源 velocity 线性插值；根、物体、39 body quaternions 用 SLERP（外部 xyzw）；源 contact 最近帧取样。控制网格不落在半源帧 tie 上。601个源状态产生500个完整 action intervals、501个状态，10秒协议时长。

缓存主要结构：

| 块 | 内容 |
| --- | --- |
| `time,source_index,i0,i1,alpha,nearest` | 同一 T+1 采样网格 |
| `source_sampled` | `[T+1,630]`，源 joint/body order、源 velocity/contact 保留 |
| `robot` | root pose（xyzw/矩阵）、name-mapped joints、`source_v/omega/joint_qd`、`derived_ref_v/omega/joint_qd` |
| `object` | pose（xyzw/矩阵）、`source_v/omega`、`derived_ref_v/omega` |
| `body` | 源39-body pose、velocity 和 contact flags；不等同机器人 FK 或 hand-object pair truth |
| `metadata,provenance` | 版本、证据状态、reference point、joint mapping、整数帧数与源身份 |

所有状态字段为 T+1；本阶段不生成 action/observation/goal，也不声称它们已经实现，metadata 仅声明完整控制区间数 T。Robot/object 派生速度采用原 SuperMimic 形式的中心/边界单边差分、world SO(3) angular differences，和源字段分开命名。不改源速度，不额外乘60/50，不做未验证的COM转换。

## CPU 检查

```bash
python -m unittest tests.test_hoi_data_reference tests.test_hoi_data_manifest tests.test_hoi_data_prepare -v
python -m unittest discover -s tests -v
python -m unittest hoi.test_reference hoi.test_hand_collision -v
```

这些是 CPU/data 检查；FK、pair contact、几何安装、native mass/COM/inertia/摩擦和训练仍属于后续里程碑。
