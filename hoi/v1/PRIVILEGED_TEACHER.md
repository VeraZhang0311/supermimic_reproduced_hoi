# Privileged tracking teacher

## 实现范围

用户确认从整理后的 v1 `fd626e752686ef61b3e7e1b604f12412a9d5cd2b` 创建
`hoi/privileged_teacher`，不包含 v1.1/v1.2 感知路径。本次实现 actor 特权观测及
collection / imagination / frozen evaluation / checkpoint 完整接线。contact 扩展和
按任务阶段的接触奖励仍为下文设计，尚未改变现有五通道、奖励、终止或物理资产。

配置：`configs/hoi_v1_privileged_teacher.json`，profile=`privileged_teacher_v1`。
使用现有 `python -m hoi.v1.train --from-scratch --config ...` 入口，还必须提供
已验证的 bundle、native evidence 和新的 output。这里没有为新数据集生成可训练
bundle，也没有提交训练作业。当前 default 配置仍为原 v1，所有数值训练日程不变。

## Actor 输入与 ULTRA 对照

对照本机 `D:/Ultra` 的 `ultra/env/tasks/ultra_g1_retarget.py`，commit
`ebe4aa276dd3208c33a8604084c0a3aeb37d9061`，配置 `ultra/data/cfg/g1_teacher.yaml`。
原 teacher 每个 horizon 为 2026D，两个 horizon 共 4052D。不是旧 SMPL 的
`humanoid_g1.py` 实现，也不是 MuJoCo sim2sim 中会屏蔽字段的 builder。

在原 v1 actor 的 1298D decoder 输入后追加 **3906D** oracle block，decoder 总输入
5204D。原机器人/物体目标编码器、10 帧历史和 shape encoder 均保留。新 block
直接送入原 action decoder，不再先压成 32D goal latent。网络隐藏层、PD action
含义、探索 action noise、4096 env、batch、H32、WM8 及固定500轮 warmup 不变。

| 新增内容 | 构造 |
| --- | --- |
| 39 link 当前全身状态 | root height、heading-local position、x/z 6D rotation、COM linear velocity、angular velocity |
| 39 link reference 与误差 | future body position/rotation、actual-reference pose/velocity errors |
| 关节与动作 | raw q/qd、上一个控制 tick 的 q/qd、previous action |
| 物体运动 | 当前 pose/velocity、future pose/velocity errors |
| 全身 interaction graph | 39 link 到物体 surface point 的最近向量与 reference 差，沿用 ULTRA 距离衰减 |
| 已有 reference 接触意图 | 两个 horizon 的左右手 expected + validity；clip 外有效性为0 |
| 真实物理上下文 | robot33：friction、29 joint offsets、torso COM offset；object12：mass、COM、inertia、object/ground friction |

两个 lookahead 在本项目50Hz网格为 +1/+13 步，即20/260ms，对应 ULTRA canonical
60Hz 的 +1/+16 步取最近网格；原有10步 dense goals同时保留。两端均使用相同的
hash-bound ULTRA39 URDF FK，reference 用 q/root 和 pose-derived velocities重建，
不读原630D中可能与新机器人不一致的缓存速度。root velocity 为 link-origin，
39-body速度转成名义 URDF COM 标记点速度，不宣称是native求解器合并body后的
COM传感器读数；真实torso COM随机化参数另以xi输入。实际、reference 和 imagined
FK使用相同合同。

ULTRA tan/normal rotation 是 x列再z列拼接，不能直接用旧 SuperMimic 的 x/y 6D函数。
body/object rotation error 的乘法顺序及 heading change-of-basis按照原 teacher实现。
reference IG用各自未来reference heading，actual IG用当前actual/imagined heading。

显式差异：

- 当前3906D不是ULTRA4052D checkpoint兼容格式；它是版本化的本项目观测。
- 使用已有128点object surface cloud，非原ULTRA点采样的逐位复制。
- 关闭actor观测噪声，包括ULTRA原代码中的heading扰动；不改变动力学随机化或探索噪声。
- 原ULTRA torque槽被主动乘零，此处不复制占位。原ULTRA contact向量虽然39D，只有
  `[6,7,13,14,28,38]` 六个body槽实际保留；它来自净接触力阈值，不区分碰到物体/地面。
  **本次没有把这些实测contact槽接入actor**：现有WM不能一致提供其语义，几何距离也
  不能冒充它们。后续full-body pair recorder + WM扩展后再接相同语义的反馈。
- 新物理上下文和显式object pose是额外oracle输入；没有伪造未知contact标签。

独立RMS只由真实collection更新，imagination和冻结evaluation不更新。新feature
schema为 `supermimic_hoi_privileged_teacher_1`；checkpoint同时记录profile和精确
schema，baseline↔teacher直接resume均拒绝。非teacher运行保持旧feature schema。

## 当前 contact head 的作用

`contacts.py` 对每20ms控制区间的4个5ms子步累加真实实体pair法向冲量，再以区间
平均法向力>1N标注。Replay中的`contact/contact_valid`监督WM的masked BCE，默认
contact loss权重0.1。不是用reference接触标记监督“执行后实际发生了什么”。

WM和critic参数在actor更新时冻结，但其对动作/状态的Jacobian保留：

```text
actor action -> joint WM -> next robot/object state + contact logits
                        -> sigmoid(contact logits) -> imagined contact-match reward
                        -> differentiable actor objective
```

当前5通道中，双手进入contact-match reward（权重0.25后乘control_dt），物体—地面
及双脚—物体仅为辅助WM监督。状态动力学另有自己的robot/object增量输出；contact
head不是碰撞求解器，不能凭一组接触概率证明质量被正确支撑。

## 独立离线标注与后续 WM 扩展

离线 expected-contact 已在独立仓库 `D:/hoi_data_pipeline` 实现，schema 为
`g1_expected_surface_contact_v1`。它绑定 v1.2 验收的 G1 碰撞几何，保留
39 个 canonical link 的命名空间，另有 object-ground，共40列；当前几何有
32个有效身体通道，其余保持 unknown 和无效掩码。控制关节的29 DOF不等于
身体接触通道数。来源时基与50Hz标签同时保存，完整长片段不按长度自动切分。

该管线的距离、相交标记和接触意图是 reference 几何证据，尚未接入本分支的
replay、WM、actor 或奖励。本仓实际 contact head 仍为原5通道。
早期讨论的11个语义通道仅是候选，不作为已实现的数据接口。

下一步 native recorder 需要记录真实碰撞link与partner身份，再明确固定link与
合并刚体的映射及语义聚合。臀部映射pelvis相关碰撞体，大腿映射hip相关实体，
前臂映射elbow/wrist实体，不能照抄人体joint index。WM扩展需要新的label/feature
schema、真实正例、地面负对照和native evidence；旧5通道验收文件不能证明扩展
通道已通过验证。原schema不静默更名或就地扩维。

机器人—地面、机器人—目标物体、物体—地面/桌面分别保留partner身份；桌面不是
地面。必要时额外记录法向/竖直冲量及相对速度，供支撑效果判断或后续强度回归。

### 两种标签

| 来源 | 已核实的reference数据 | 限制 |
| --- | --- | --- |
| ULTRA | 原630D有39个body flags；本项目另由手部/物体mesh生成双手intent+validity | body flags没有可靠pair身份，不能代替新sim真实标签 |
| HOI-Retarget OMOMO | 四个掌/踝-object flags、object-local接触点、双脚-ground flags、object接触序列 | 只读检查schema和前三条；无力/冲量或骨盆/躯干标签；非接触点含NaN |
| OmniContact | 全2231条均有`contact_info[T,4,1]`，左脚/右脚/左手/右手-object | 几何/运动启发式与修补的reference intent，非真实力；无臀/背通道 |
| GRAIL sitting | 全1748 meta无逐帧contact；抽查5对PKL有左右手接触点字典 | 无骨盆/大腿/躯干；`smpl_joints`样例全零，不能用于几何推断 |

GRAIL上游VLM会按视频区间推断Pelvis/Hip/Torso等接触语义，但G1 motion library
导出没有保留完整全身监督，手部点也不是retarget后重新仿真测得的接触。

因此：WM实际contact标签由我们的真实sim生成，不依赖dataset是否有contact标签；
reference intent缺失部位先标unknown，用G1 collision surface与object surface距离、
相对运动、支撑方向、动作阶段生成带置信度伪标签。缺失不是负例。

### 超出双手的 reward

需要扩展，但按任务阶段的功能接触计算，允许合理接触变化，而不是所有body按帧
做独立exact match。例如sit要求pelvis或大腿形成有效支撑；lean允许torso或前臂
支撑；kick只在击球事件时间窗要求脚—球命中。手搬物保留其必要支撑/搬运阶段。

对允许的接触集合G，可用平滑union `1-prod(1-p_i)` 作为接触概率，再按阶段/有效
角色归一化；它仅是设计候选，需要验证WM概率校准与冗余通道影响。unknown目标
不参与奖励，额外的合理接触不自动罚；明确禁用的碰撞走单独物理安全判据。

“碰到”必须与物体抬升/定向位移、稳定支撑和无意外掉落一起判断。坐姿不能仅因
1N轻触就算承重。kick的命中是短时事件，不能套用持续接触覆盖率。接触目标要与
允许的事件时序/阶段容差一起定义，避免为了复刻时刻而牺牲合理交互。

新增通道的BCE仍监督所有真实接触；reward只用任务相关且有效的目标，按角色
归一化，防止坐椅由于通道多而自动放大总奖励。先做pose-only/加contact的分任务
对照和人工判读，再冻结阈值；旧pose/PHC指标继续单独报告。该扩展不在本次实现
中悄悄开启。

## 验证边界

CPU覆盖：独立四元数FK对照、COM速度、rot6D约定、state梯度、H32 actor梯度、
WM/critic参数冻结、真实collection-only RMS、clip边界、冻结完整evaluation及
checkpoint跨profile拒绝。全套CPU回归270项：269通过，1项因可选pxr未安装skip；此前不含
MuJoCo的环境出现5项missing-dependency错误，已在现有完整CPU环境重跑通过。
另外6项历史reference/hand-collision边界检查通过；真实ULTRA39 URDF上的独立
四元数/矩阵FK对照通过（位置最大绝对误差2.8e-7m，COM标记速度1.2e-6m/s）。

尚未执行原生GPU/显存/速度验证，也未训练；不能据此宣称tracking已提升。
新datasets的scene/asset/adapter和contact schema仍需各自native验证。
