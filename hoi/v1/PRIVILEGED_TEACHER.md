# Privileged tracking teacher

## 实现范围

用户确认从整理后的 v1 `fd626e752686ef61b3e7e1b604f12412a9d5cd2b` 创建
`hoi/privileged_teacher`，不包含 v1.1/v1.2 感知路径。本次实现 actor 特权观测及
collection / imagination / frozen evaluation / checkpoint 完整接线。2026-10-31 按用户授权扩展为 16 区域接触、统一 reference 驱动奖惩、普通 masked BCE。
真实接触采集、reference loader、actor/critic、replay、评估和版本校验同步更新。
2026-11-01 按用户授权将倾斜终止改为 reference 相对重力夹角，并给物体位置容限加 2m 上限（下文详述）。
PD、动力学随机化、物理资产和固定训练日程保持原值。未启动训练。

配置：`configs/hoi_v1_privileged_teacher.json`，profile=`privileged_teacher_v3`。
使用现有 `python -m hoi.v1.train --from-scratch --config ...` 入口，还必须提供
已验证的 bundle、native evidence 和新的 output。完整新数据集bundle已完成，
冻结 source10 的全资产 GPU 验收和视觉复核已完成；新终止规则的 CPU 回归另行记录，
不把 source10 的物理结果当作修改后源码的 GPU 验收。未提交训练作业，所有数值训练日程不变。

## Actor 输入与 ULTRA 对照

对照本机 `D:/Ultra` 的 `ultra/env/tasks/ultra_g1_retarget.py`，commit
`ebe4aa276dd3208c33a8604084c0a3aeb37d9061`，配置 `ultra/data/cfg/g1_teacher.yaml`。
原 teacher 每个 horizon 为 2026D，两个 horizon 共 4052D。不是旧 SMPL 的
`humanoid_g1.py` 实现，也不是 MuJoCo sim2sim 中会屏蔽字段的 builder。

在原 v1 actor 的 1298D decoder 输入后追加 **3962D** oracle block，再追加
独立 **26D** 桌面场景输入，decoder 总输入 **5286D**。原机器人/物体目标编码器、10 帧历史和 shape encoder 均保留。新 block
直接送入原 action decoder，不再先压成 32D goal latent。网络隐藏层、PD action
含义、探索 action noise、4096 env、batch、H32、WM8 及固定500轮 warmup 不变。

| 新增内容 | 构造 |
| --- | --- |
| 39 link 当前全身状态 | root height、heading-local position、x/z 6D rotation、COM linear velocity、angular velocity |
| 39 link reference 与误差 | future body position/rotation、actual-reference pose/velocity errors |
| 关节与动作 | raw q/qd、上一个控制 tick 的 q/qd、previous action |
| 物体运动 | 当前 pose/velocity、future pose/velocity errors |
| 全身 interaction graph | 39 link 到物体 surface point 的最近向量与 reference 差，沿用 ULTRA 距离衰减 |
| 已有 reference 接触意图 | 两个 horizon 的16区域 expected + validity；clip 外有效性为0 |
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

- 当前3962D不是ULTRA4052D checkpoint兼容格式；它是版本化的本项目观测。
- 使用已有128点object surface cloud，非原ULTRA点采样的逐位复制。
- 关闭actor观测噪声，包括ULTRA原代码中的heading扰动；不改变动力学随机化或探索噪声。
- 原ULTRA torque槽被主动乘零，此处不复制占位。原ULTRA contact向量虽然39D，只有
  `[6,7,13,14,28,38]` 六个body槽实际保留；它来自净接触力阈值，不区分碰到物体/地面。
  **本次没有把这些实测contact槽接入actor**：现有WM不能一致提供其语义，几何距离也
  不能冒充它们。本次输入扩展的是全区域 reference 接触目标；实测接触反馈未额外加入 actor。
- 新物理上下文和显式object pose是额外oracle输入；没有伪造未知contact标签。

独立RMS只由真实collection更新，imagination和冻结evaluation不更新。新feature
schema为 `supermimic_hoi_privileged_teacher_3`；checkpoint同时记录profile和精确
schema，baseline↔teacher直接resume均拒绝。非teacher运行使用独立的
`supermimic_hoi_v1_full_reference_4` schema，接触目标同样为16区域。

## 桌面场景观测

Actor、WM、critic共享固定两个槽（table1、table2）的26D输入，每槽13D：presence1、
当前机器人heading下桌面中心相对位置3、同heading下桌面旋转6D、完整尺寸3。
旋转沿用 `mat_to_6d` 的前两列行序展开；尺寸固定为原作者cuboid的0.3×0.25×0.01m。
支撑物材质与运动模式由hash绑定的固定recipe限定，故不再添加自由材质/速度维度。

真实collector在采样reset frame放置桌面，整个episode保持世界位姿。replay额外保存
`support_pose_xyzw[...,2,7]`（与robot相同的环境原点）和`support_present[...,2]`。
想象rollout只冻结这份起始世界位姿，随预测机器人位置和heading重算相对场景，
不随reference frame移动桌面；到机器人状态的梯度保留，桌面常量不参与优化。
没有桌面的历史ULTRA数据仍可由native bank明确构造两个全零输入槽，无需伪造桌面。

三个网络各有独立scene RMS；统计仅由实际collection/replay/value-fitting输入更新，
冻结evaluation和actor imagination不更新。原始缺席槽严格全零，RMS计入这些零观测；
归一化后重新应用presence mask并保留未经归一化的0/1 presence，避免缺席槽因非零均值
产生虚假场景输入。缺少新字段的旧replay明确拒绝，不从reference猜出真实桌位。
新checkpoint精确记录此输入schema及26D RMS，旧无scene权重不能直接resume。

## 16 区域接触合同

固定顺序由 `contacts.CHANNELS` 和 `BODY_GROUP_LINKS` 定义：

| 区域 | 数量 | canonical collision links |
| --- | --- | --- |
| 左/右手末端 | 2 | rubber_hand、wrist_yaw_link、wrist_pitch_link |
| 左/右前臂 | 2 | elbow_link、wrist_roll_link |
| 左/右上臂 | 2 | shoulder_pitch/roll/yaw_link |
| 躯干 | 1 | torso_link、logo_link |
| 骨盆 | 1 | pelvis_contour_link |
| 左/右大腿 | 2 | hip_pitch/roll/yaw_link |
| 左/右小腿 | 2 | knee_link |
| 左/右脚 | 2 | ankle_pitch_link、ankle_roll_link |
| 头 | 1 | head_link |
| 物体—地面 | 1 | object 与 ground 的独立 pair |

前三项各先左后右，再躯干、骨盆、大腿、小腿、脚、头、物体—地面。
区域schema `g1_contact_regions_15_v1`；实际标签schema
`hoi_v1_region_pair_normal_impulse_v3`。原32个有效身体link均有归属。
编译器合并固定link时根据已验证的body/geom身份保留原碰撞link语义，不能用
合并后的torso净接触力替代head/hand/torso的独立pair。

真实标签：对每20ms控制区间的4个5ms子步、组内所有碰撞点累加法向冲量，
区间平均法向力严格大于1N为正。多link接触不重复产生多份reward；阈值施于
区域总冲量，不能把每link先阈值再OR称作同一个实际标签合同。
身体—地面保留15列独立负对照，不进入身体—物体通道；自碰撞也不进入。
overflow、实体/环境身份、非有限值和原生资产检查保留。

WM输出16个独立logits；用真实replay的contact/contact_valid和terminal mask
做普通 `binary_cross_entropy_with_logits`，无动态pos_weight。正负计数仅诊断。
Actor imagination中用sigmoid连续输出，冻结WM/critic参数但保留动作/状态Jacobian。
Sigmoid仍需在真实holdout上检查校准，不能仅凭BCE或低错误率证明物理效果。

## Reference 与统一奖励

离线管线 `g1_expected_surface_contact_v1` 是几何意图：≤2cm正、≥4cm负，
中间、穿透、无几何unknown；不是实际力接触。原39身体link+object-ground
标签保留不改，消费时聚合成16区域：任一正→正，所有active成员负→负，其余unknown。
无几何link不补成负例。手腕正而同组橡胶手负时，区域为正，橡胶手实际接触可满足。

先空间聚合，再按原始与50Hz采样的时间支持区间聚合：任何观测正事件保留；
负要求两套网格支持内全部明确为负。保留短暂击球事件，不插值二值标签，
覆盖比例不解释成概率。`expected_contact/expected_valid` 为 `[T+1,16]`，
row i描述 `(i-1,i]`，row0无前驱且全部无效。

每步contact reward为：

```text
mean(p over known-positive regions) - mean(p over known-negative regions)
```

每个空集合项为0；unknown无奖惩/梯度。正负集合各自归一化，共用现有
`w_contact=.25`与control_dt；所有task使用同一公式。全负reference的contact
最大奖励为0，而不是1。身体/物体pose/velocity tracking继续保留；手部approach
仍只在左右手正目标上激活。接触匹配不是承重、抓取或击球成功的充分条件。

当前object tracking四项仍统一使用位置/朝向/线速度/角速度指数奖励；误差尺度分别
为0.10m、0.30rad、0.50m/s、1.0rad/s，权重1.0、0.5、0.25、0.25，再乘control_dt。
本次只读审查未据任务扩展改写这些项，也未新增足球朝向对称性规则。现有
hand_approach仍测量内掌面采样点的SDF距离：contact项已支持手部组内替代，
approach辅助项仍有掌面偏好；不能把接触项等价宣称为整个交互奖励等价。
上述目标取舍尚未改动，不属于本次已经完成的奖励修改。


Dense object goal每tick为47D，10tick为470D；critic原914D保留，新增raw object
block544D加shape64D及scene26D，总1548D。WM输入为原224D加scene26D，共250D；
baseline actor为1324D。Teacher两个lookahead各有16目标+16validity，
特权块3962D，总decoder输入5286D。原机器人网络前缀保留。
全reference feature版本4、teacher feature版本3、checkpoint版本3；旧无场景输入或3/5通道
checkpoint及旧双手contact bundle不可直接加载。WM warmup仍为500轮，之后按
固定日程更新actor；holdout只诊断，不新增持续质量门。

## Curated 数据与划分

入口 `hoi.data.curated` 校验原selection READY/manifest/清单SHA，消费明确included
的11,120条：ULTRA2731、HOI-Retarget4421、GRAIL1748、OmniContact2220。
不能遍历旧derived根重新纳入被排除的11条。原始labels、assets、split均不改写。

用户确认protected_eval目标约10%，按dataset/task分层、source group原子分配。
已知protected eval必须保留，共享OMOMO来源的ULTRA/HOI-Retarget跨数据集同组；
Omni按官方capture_id。GRAIL保留随机变体的来源关联，并在报告中说明已验证
分组粒度及跨资产原始动作身份的限制。类别缺失时保留 `unspecified`（用户已确认），
不根据物体名称猜carry。任务比例、实际eval比例、强制保留和分组限制写入报告。

2026-10-31实际产物为 `data/curated_contact16_20261031/split_90_10/split.json`：
train **9575**、protected_eval **1545（13.8939%）**。目录名表示10%目标，
不表示实际恰好90/10。Omni原validation/test强制保留655条（该源29.50%），
其余GRAIL175/1748、HOI-Retarget442/4421、ULTRA273/2731均约10%。
11个dataset/task分层均在两组有覆盖；已知来源组交集为0，被排除的11条未纳入。
完整task比例见同目录 `report.json` / `task_distribution.csv`，独立重算和原字节SHA
见 `independent_verification.json`。目录包含冻结selection和metadata快照，可整体迁移。

Split schema为 `hoi_curated_task_group_split_v1`，独立train/protected_eval JSONL
与split.json绑定SHA。Protected eval不能进入replay、normalizer或learner。
`load_curated_clip`直接读取已标注50Hz root/joint/object轨迹，保持米制、WXYZ和
canonical29 joint顺序；速度由pose求导。新来源不伪造ULTRA630D source字段。

`python -m hoi.v1.prepare --curated --manifest <split.json> --assets <recipes.json>
--partition train --output <fresh-directory>` 通过新schema构建训练/评估bundle。
原生recipe必须按 `object_geometry_sha256` 显式匹配标注几何，绑定native USD、
碰撞mesh、mass/COM/inertia、手脚与FK；不猜测物理参数、不自动重cook。
recipe还必须显式绑定 `robot_geometry_sha256`。全量审计的2,448个标签几何身份
对应303组完全相同的实际表面（GRAIL189、HOI-Retarget13、Omni93、ULTRA8）。
去重还必须满足完整物理参数、native USD与碰撞几何一致；不能仅按物体类别合并。
每个标签身份保留独立alias绑定。共享recipe与轻量alias避免重复保存大网格。
完整train/protected_eval bundle采用同一资产ID表，分区的motion保持隔离。

## 新物体与桌面支撑

`hoi.data.native_assets`提供audit/build/finalize阶段。先从原尺度碰撞几何生成输入，
由真实GPU native importer导出实际convex hull，再绑定最终完整物理profile；
`--curated-cooking-fixture`只导出几何，0物理步、无pair通过声明，不能用于训练。
最终bundle必须重新完成标准GPU pair、geometry、physics、support验收。
source-visible surface、native cooked surface与训练geometry cache分别绑定SHA，
不声称与作者PhysX/VHACD完全相同。

真实导入发现的退化collider单独处理，不能静默删除标签或跳过相关clip。
`hoi.data.native_repairs`将原始source/build、失败导入、候选几何、批准记录及实际
重新导入绑定为独立证据链；原输入不覆盖。任何增加碰撞体积的修复先等待用户确认
具体候选，批准不能由等待时间推断。完全位于已有碰撞实体内部的零体积冗余片，
只有在实际collision hull上证明严格包含且保留组件字节一致时才允许清理。
修复后仍须通过真实GPU导入与完整物理gate；CPU几何检查不能代替这些步骤。
显式来源COM/惯量不改写，原deferred模式才从实际闭合组件计算；质量/摩擦、标签
与分区不因修复改变。几何近似诊断报告实际差异，不宣称与原作者cooker完全相同。
用户已明确批准垃圾桶两片零厚度小面沿原面两侧各增加5µm（总厚度10µm），以及
GRAIL王座椅微小退化片并入邻近碰撞块。批准记录绑定两个具体候选SHA；该约定
不自动授权其他物体改变几何。两项source08真实GPU重新导入均通过组件保真检查。

用户2026-10-31确认的质量约定（kg）：clothesstand 2、floorlamp 2.5、largebox 2、
largetable 6、monitor 3、plasticbox 1、smallbox 0.5、smalltable 4、suitcase 3、
trashcan 1、tripod 1.5、whitechair 2.5、woodchair 4。保留其他来源已提供质量。
缺失COM/惯量按米制碰撞几何、均匀密度计算；非闭合源网格先cook，再用实际闭合
convex components的体积和二阶矩计算，重叠部分按独立collider累计，明确记录为工程
近似。未经该步骤的临时import参数不得进入最终训练profile。
组件积分采用包围盒中心平移与最大边长归一化，体积及中心惯量分别按尺度三次方、
五次方还原，再按原体积加权及平行轴公式合并。退化检查作用于无量纲体积；
固定米³下限会把有效微小四面体误判为零体积，远离原点的直接积分也会发生惯量
消减。两项实际微小组件已用80位Decimal独立行列式/单纯形矩公式复核；输入几何、
质量和重叠累计约定保持不变，显式来源COM/惯量仍直接保留。

摩擦保留来源值：HOI-Retarget物体0.9；GRAIL物体/地面1/1；Omni地面1。
Omni物体采用用户确认的作者动态区间中点：push/slide 0.20、足球0.55、carry箱体0.45，
原区间作为来源证据保留，不复制作者随机化日程。确实缺失的摩擦才使用物体0.6、
地面0.3工程fallback。不同引擎的friction combine规则不宣称等价。

Omni table1/table2独立作为native kinematic cuboid，尺寸0.3×0.25×0.01m，
来源质量10000kg、阻尼1000、摩擦1、restitution 0；原始contact/rest offset只作为
来源证据，native实际margin/gap另行读回。按作者语义在reset选定reference帧放置，
整episode保持世界坐标，不逐tick追踪reference桌面轨迹。table1未使用时停在地下，
table2保留自身来源pose。90Hz源姿态到50Hz绑定、原始payload和cache均校验SHA。
身体—桌面、物体—桌面的真实pair冲量单独诊断，不混入16个身体—物体/物体—地面通道。
每个桌面必须验证真实kinematic flag、尺寸/材质、reset后位置与跨步静止，以及隔离接触。
评估保存实际桌面和reset-held reference桌面状态；离线渲染只回放已保存状态。

GPU操作使用独立目录
`/mnt/projects/ll/tianzizh/supermimic_hoi_teacher_native_20261031_01`，源码按snapshot SHA
冻结，原资产与已有实验不覆盖。原生GPU验收仍在进行：CPU或cooking成功不等于通过，
所有标准gate完成前不声明training-ready。未启动训练。

本次固定runtime为Warp1.17 / Newton1.8dev / MuJoCo-Warp3.15。完整物理需要CUDA
driver≥12.1：CCD forward的实际参数块4296B超过旧驱动4096B上限，不能通过关掉
backward解决。旧driver12.0只允许0-step geometry cooking，并对Newton MuJoCo模块
使用官方enable_backward=False去掉未用的6008B backward entry。独立GPU probe03
验证56/56 forward函数一致、真实forward执行正确、0 physics；该证明不等于完整
碰撞runtime兼容。共享runtime、driver、solver参数均未改写，PyTorch WM梯度独立保留。

source04b的ULTRA largebox验证已通过（job1485516）：8个环境、16个object正例与
15个body-ground负例逐环境合格、64-step smoke完成，最终gate无blocker。范围仅为
绑定的单资产native_validation bundle，不能复用于不同的完整bundle身份。
三来源pilot02（job1485508）已完成Omni/GRAIL/HOI的实际GPU cook，0 physics steps，
其代表资产现已完成最终惯量绑定和source06的标准native gate（job1486015）：
每个8环境、31类pair probes、64步smoke，汇总无blocker；仍仅代表3条原train clip，
不替代完整分区验收。Omni两桌的真实body/object-support冲量独立成立，16学习通道
严格零泄漏，reset-held桌位误差0。source05的body-support探针因读取旧solver几何而
放置错误；source06改用同一真实compiled model的已校验legal hand pose，保持其关节
PD目标并读回初始状态，不改场景、阈值、求解器或额外增加物理步，旧失败证据保留。
零步reset后的solver几何缓存不能
作为同步场景图；可信预览使用--capture-smoke-visual在已有smoke步结束、autoreset前
抓取，随后0FK/0physics离线渲染；不增加物理步，cooking禁用此选项。

source06三来源还通过首个真实CUDA transition的WM250、teacher3962、16 logits与
26D scene输入检查；实际出现的unknown槽数分别为3/17/13，均保持masked zero。
随机未训练网络仅做输入/前向校验，0optimizer、0RMS更新、0额外物理，RNG恢复，输出
不写回仿真。`native_batch`对完整train与protected_eval按partition/asset独立运行，
严格绑定各自bundle hash；只复用编译cache，成功receipt重新验gate与输入/源码SHA，
失败用新attempt。完整全量验收尚未执行完毕，不能用上述样例宣布training-ready。

三脚架去除严格包含的零体积冗余片后，source07的独立单clip验证也通过全部
8环境/31 pair/64步以及真实CUDA输入和保存状态渲染（job1486084）；质量1.5kg、
原摩擦保留，缺失COM/惯量按批准的实际闭合组件规则计算，不宣称来源实测惯量。
其余292个新物体cooking组的双向采样表面统计及所有输入/输出SHA复核已完成；
统计包含重叠组件内部表面，属于近似差异观察，不是完整物理通过证据。
已批准的垃圾桶与GRAIL王座椅局部修复连同原导出，现已完成303组严格导入收集；
source09已完成303组最终物理配方，以及所有303组距离缓存的严格核对；两分区
bundle也已完成并通过独立数据审计，逐资产GPU物理验收仍待完成。七个资产的64³ SDF未满足既有固定探针
10mm精度界，已明确将各自distance_recipe改为96³，实测最大探针误差为
6.89–9.03mm；其余296组缓存保持原产物身份。碰撞实体、质量/材质、原采样种子、
4096候选点和128点FPS均不变，不把加密网格称作新的碰撞几何修复，也不把
采样QA解释为全空间误差保证。新配方保留303全局资产ID及2448标注几何身份，
旧recipe/producer失败记录与逐项继承证据保留；当前对应
`full_native_assets_02/recipes_distance_refined_01.json` 和
`full_geometry_cache_02/geometry_manifest.json`。
source08只修正repair CLI直接脚本入口的导入路径，并保留旧作业与真实导出身份。

完整batch要求真实smoke快照、二进制模型、GPU数组、渲染记录及PNG逐项绑定SHA，
成功回执重验这些文件和渲染源码。额外`--audit-scene-resets`在原物理、输入检查和
快照全部完成后，对每个片段首/中/末及桌面presence切换边界做实际reset读回；
期望值独立取自绑定的reference记录，检查桌位、零速度和presence。此终端辅助
审计禁止物理积分并校验步数不变，随后丢弃状态；不再运行网络或继续采集。
它补充全片段的索引/parking检查，范围不等于全部时间帧的物理tracking成功。


评估支持完整protected_eval分区，使用真实保存状态和相同bound URDF的reference FK；
没有原始ULTRA630字段时直接评分G1 reference，原source flag诊断标记不可用。
16区域missing/extra均参与匹配判据；unknown不算错。回放读取保存的reference
transforms，不在渲染时执行FK、policy或physics。所有评估保持0optimizer/normalizer更新。

## 验证边界

本次运行CPU数值、采集合同、bundle/reference、split防泄漏、梯度与冻结评估回归。
source10完整CPU回归442项：440通过，2项因可选pxr与Windows symlink权限跳过；
历史HOI边界6项通过。Linux实际venv symlink入口已独立验证。真实三资产的并行几何
缓存与原串行产物逐张量一致（points、SDF、边界、collision meshes）；原采样/距离
算法及seed不变。共享geometry manifest绑定recipe/算法/缓存SHA，仅复用静态几何，
两个分区的reference/labels仍隔离。完整原生数据集验收仍在进行。
四个真实来源reference转换smoke通过；计数、日志及划分身份见
`tmp/regions16_validation/native_assets_cpu_suite_10.log`、
`data/curated_contact16_20261031/native_acceptance/operation_status.json` 与派生数据回执。
未启动训练；未把MuJoCo CPU合同测试当作MuJoCo-Warp/IsaacLab GPU原生校准通过。
新16通道native evidence要求所有区域正接触与身体—地面负对照，旧5通道evidence拒绝复用。


source10仅为完整header验证增加调用内共享字段编码/身份结果复用：完整alias的
canonical字节哈希、文件/物理/repair/split等所有校验保持不变，不设跨调用可信缓存。
同一实际train包的顺序cProfile A/B返回对象及canonical SHA完全相同，耗时
194.192s→47.157s；这是同进程顺序测量，不是独立冷启动的严格倍率。独立审查的
4000组随机alias字节等价检查通过。source09、冻结数据及原失败证据均保留。
两个完整包实际为train9575条/3658248状态帧、protected_eval1545条/818352状态帧；
303全局资产中分别使用273和106个，按分区/资产独立验收共379组。独立数据审计
逐条验证了contact三态掩码、冻结membership、已知source-group不交叉及约定物理值。
完整GPU结果仍待收集；上述CPU结果不替代原生物理通过证据。

source10的完整原生验收计划已生成并通过CPU规划检查（作业1486683，COMPLETED 0:0），
full_native_batch_02/plan.json覆盖273个train与106个protected_eval资产任务，共379项。
首批8项真实GPU验收（数组1486694）全部COMPLETED 0:0，原生成功回执齐全。
独立复核64个保存环境、2446个片段和7338个reset采样点：桌面位姿/速度误差均为0，
额外物理步数为0，scene26复算最大误差2.384e-7；8张真实env0图已查看。
这些结果只覆盖首批8个分区/资产任务，不能据此宣布全379项通过或完成tracking评估。

CPU作业1486786严格重验首批8项成功回执后生成剩余371项新调度；
GPU数组1486788现已全部COMPLETED 0:0；共93个槽位、每槽最多4项、最多4卡并发，每槽4CPU/32GiB、
2小时上限。每项仍保持8环境、64控制步、31个pair探针、CUDA输入、保存状态、
渲染及terminal scene reset检查。成功任务不重复执行；最终仍须全量收集和实际图像QA。
单片段对照确认一处carry双桌重叠来自原始发布坐标；soccer地下桌的presence不能当作
屏幕可见双桌覆盖。来源特殊外形核对和图像范围说明保存在独立QA回执中。

最新计划、提交、清理和QA回执保存在ignored data/curated_contact16_20261031/native_acceptance/，
恢复入口为其中的operation_status.json与shutdown_recovery_20261031.json。未启动训练。

2026-11-01全量机器验收结果：379/379个分区/资产任务成功、0失败；
CPU汇总1486825与保存状态渲染1486831均COMPLETED 0:0。两个分区的全部11120个片段
共33360个terminal reset采样点已纳入审计。选出46个实际视图，其中4个非env0视图
完成零物理/零FK/零策略的补充渲染，原env0图保留原SHA。
最终逐图复核尚未完成，不能把机器汇总成功等同于完整视觉覆盖通过。
动态截图未覆盖两个分区的soccer/case2_kick_left和case3_kick_right；
桌面mask在动态快照和terminal审计中均只出现00/11，未观察到01/10。
QA工具的ULTRA类别预期8是实例/类别计数混用：实际8个资产ID全部存在，
对应largebox、plasticbox、smallbox、suitcase四类，并非漏验四个资产。
完整机器回执在ignored native_acceptance/full_native_final_machine_01中。未启动训练。

2026-11-01视觉复核补充：46张原始PNG已逐张查看并核对文件SHA；用户也报告除三处
询问外其余43张未发现问题。asset257为GRAIL鱼形木椅，asset193为GRAIL球舱椅，
均属sitting。asset67椅脚周围的不规则薄片已确认存在于来源visible-surface网格，
并非新增桌面或阴影；其语义未标注，作为来源几何质量问题保留，未擅自清除或无条件放行。
单视角遮挡仍限制对坐姿/接触的判断；逐图观察和来源网格证据见ignored
native_acceptance/final_visual_review_01/。机器验收成功不等同于来源资产完全无缺陷。

左右踢球补图数组1488395的train/protected_eval两项均COMPLETED 0:0。
独立tools/hoi_native_task_visual.py调用冻结source10，按脚部有效正标注选取参考时刻，
每分区保持8环境、64零动作控制步及原pair/support校准，保存首次真实积分后的状态。
四张新增图覆盖两个分区的case2_kick_left/case3_kick_right，图像SHA、任务/片段/帧
身份及零物理/零FK离线渲染均已核对。右踢图中双臂展开、球位于两脚附近，单帧不证明
完成右向踢球；四张图已供用户复核，尚未收到用户反馈。原379项结果、数据划分和source10
未变；selector三项针对性测试通过。补图不运行训练，不作为训练策略成功率证据。

用户随后选择查看数据参考动作。已从与四张补图相同、SHA核对过的OmniContact原始
90Hz片段制作4秒/30fps参考视频，每3帧取一帧、正常速度、固定世界相机；原始数据不改。
四段截图时刻附近视频和两段右踢9.5–13.5秒动作阶段共6段/720解码帧全部通过检查。
右踢前段确有双臂展开停留，不能以该姿态单帧证明踢球覆盖；后续动作片段单独保留。
参考准备阶段使用绑定URDF做FK，离线渲染仅应用保存的世界姿态（无关节FK、无物理积分、
无策略调用）；蓝色机器人为原始URDF外观，橙色为作者绑定足球。视频不是native rollout。
本地交付目录为HOI/privileged_teacher_kick_reference_videos，索引/脚本在ignored
final_visual_review_01/reference_video；26个弃用预览文件已精确清理，6段视频SHA保持不变。


2026-11-01用户已确认截图和参考动作视频目测通过，并决定数据清洗在此收尾。
asset67保留来源薄片；上述历史“待用户复核”状态在此关闭。该接受范围为数据视觉检查，
不等于训练策略tracking成功。用户确认回执保存在ignored final_visual_review_01/。

## 扩展任务的终止条件（2026-11-01）

用户授权后，HOI使用独立ReferenceGravityMDP，保留共享supermimic motion-only实现。
令R为pelvis到世界的旋转，g=R[2,:]为体坐标中的世界竖直方向（两边同时取负即重力）。
倾斜误差改为atan2(||g_actual × g_reference||, g_actual · g_reference)，世界yaw不影响它。
倾斜硬终止阈值robot_tilt_limit_rad=acos(0.2)，约78.463°：保留旧规则在reference直立时
|1-cos(theta)|>0.8的边界，但能识别reference与actual反方向倾斜。跟随reference的坐下、
倚靠不因绝对姿态低/斜而失败。高度仍为相对reference的pelvis高度误差>0.25m。

物体位置失败阈值为clamp(2 × 资产包围盒对角线, 1m, 2m)，距离仍是实际物体位置相对
reference物体位置的欧氏误差。CPU作业1488521（COMPLETED 0:0）逐一核对完整计划和
303个实际geometry cache的SHA后读取尺度：对角线中位数0.971675m，最大1.952116m；
旧位置容限最大3.904233m。2m上限保留163个资产原阈值、收紧140个；若用1.5m上限则
会影响250个。选择2m是基于现有尺度分布的宽松工程界限，尚无训练结果证明最优。
完整逐资产审计保存在ignored native_acceptance/termination_review_01/。

真实采集reset和failure_reason位共享termination_components；冻结评估使用相同human
判据。imagined rollout的soft termination也使用同一角度/位置阈值，sigmoid温度分别为
倾斜0.04弧度、高度0.02m、物体0.10m；倾斜温度的单位已由旧标量差变为弧度。
角度用cross/dot的atan2避免acos在完全对齐处的无穷梯度；半精度输入先升为float32。
片段结束/500控制步timeout、接触仅作诊断的终止约定不变，无任务类别专属分支。

新增8项回归覆盖坐下/倚靠与yaw不变性、反向倾斜、硬/软阈值一致性、大小资产的下限/
中间值/上限、对齐及反向旋转的有限梯度、批维/低精度、非法参数，以及实际collector step
的reset和reason bits。collector测试用张量替身隔离物理，是CPU控制流证据；本次没有重跑
GPU原生物理，也没有修改冻结source10、原资产、数据或划分。未启动训练。

本次完整CPU回归453项：451通过、2项因可选pxr/Windows symlink权限跳过；历史HOI边界6项全部通过。
定向20项也全部通过；日志与验证范围保存在ignored termination_review_01。提交前队列为空。


## superteacher 启动与资源部署（2026-11-01）

用户已授权开始完整privileged teacher训练，W&B新项目为superteacher，entity沿用
tianzizh-university-of-pennsylvania；API已创建并读回PRIVATE权限。使用新run UUID，
不续接旧实验；仅上传既有白名单配置和指标。白名单现包括显式actor_observation_profile。

启动检查发现train的273个静态资产原来各占一个常驻仿真进程、全部落在同一GPU。
379项逐资产原生验收没有验证273进程并发。独立三worker容量作业1488549成功结束，
source_01保持真实资产和正式每资产环境配额：asset266/2env约380MiB、asset31/4env约
384MiB、asset209/375env约2688MiB GPU内存，均完成12个真实控制步、0优化更新。
这是三个样本，不构成完整训练容量验收；不能外推为已通过全部273组。

为保留完整数据和4096总环境，新增显式--collector-devices cuda:N列表，与中央learner
的--device分开。每个完整WorkerSpec按原顺序轮转到collector卡；资产、环境配额、seed、
IPC拼接次序及同步控制步不变。child在CUDA/Warp初始化前隔离到所分配的Slurm可见卡，
记录实际visibility token，内部使用cuda:0。可选--worker-startup-concurrency只节流初始化，
不会轮换/删除已ready的资产。初始化后释放不再使用的完整bundle/导出临时对象并做一次GC，
不修改数据验证、几何、物理或任何训练数值。

resource_probe现接正式teacher config和--from-scratch，配置同样的39-body oracle；
旧init-checkpoint入口保留。critic target使用与训练一致的分块前向，全行及顺序保留。
资源检查从真实Replay.sample_starts候选中分层保留每资产至少1行，再填满4096，以免小配额
资产在一次随机抽样中缺席造成假失败；每行history/horizon仍验证，最多32轮，缺数据就失败。
这只用于资源覆盖检查，正式Learner/Replay的采样不变；报告明确production_sampling_changed=false。

专用cluster/hoi_teacher_train.sbatch.sh要求全新PROBE_DIR/RUN_DIR和冻结source SHA，
先运行完整273组、正式batch/H32/replay的资源与反传检查（0优化更新），身份/配置/资产覆盖
全部一致且passed后才调用正式train --from-scratch。GPU数、队列、CPU/RAM和时限由本次
独立提交显式指定，不能以减少数据或batch绕过容量检查。该资源检查不是WM学习质量门。
评估使用独立hoi_teacher_evaluate.sbatch.sh，完整1545条protected_eval，保留5000轮调度，
单GPU逐资产、OMP/MKL=3与既有allocator合同。两个launcher也纳入source_identity。
500轮WM-only、20000总轮次及其余科学配置不变；实际提交/运行状态保存在ignored启动回执，
资源预检运行不能表述为正式训练已经开始。

启动部署实现的完整CPU回归472项：470通过、2项环境限制跳过；历史HOI边界6项通过。

两卡真实采集验收1488603已COMPLETED 0:0：asset266/2env和asset209/375env各运行12步，
通过真实PID/GPU UUID确认分别位于两张卡；0优化更新。新初始化清理后，两worker的匿名
内存分别约2.80和3.06GiB；这仍不是273组并发容量结论。

正式启动流水线1488620于2026-11-01提交，冻结代码2754081/source_02，source content SHA
为0b31d8eafa45b3a03514ff79e73ab3bb0e92faab6761474c12620178f95866a8。
独立目录/mnt/projects/ll/tianzizh/superteacher_20261101_01，PROBE_DIR=full_capacity_01、
RUN_DIR=train_01；申请已验证兼容的al-l40s-0上5张L40S（learner 1、collector 4）、
96 CPU、980GiB RAM、7天。提交时PENDING(Resources)，尚无优化更新；容量检查通过后
自动开始训练，失败则退出，不缩减数据/环境/batch/H。W&B项目地址为
https://app.forge.coreweave.com/tianzizh-university-of-pennsylvania/superteacher。
提交回执保存在ignored data/superteacher_20261101_01/training_submission.json；实际运行状态
以Slurm和该独立目录的日志为准，本文提交状态不代表后续运行结果。


## 多资产共进程资源实验（hoi/multi-asset-collectors）

原superteacher作业1488620保持排队及冻结source_02不变。新分支显式
--assets-per-process K，把相邻完整资产WorkerSpec打包到同一spawn进程；默认K=1
保留原入口。每资产仍常驻独立scene/model/solver、局部motion bank、bin_G/bin_w_avg，
quota/全局行序/seed规则不变，不轮换或删掉资产。每个asset独立保存CPU及本CUDA设备
RNG，构造、step/autoreset、显式reset期间切换，退出恢复调用者状态。

当前Newton MuJoCo-Warp多world要求同构拓扑，各资产hull数不同，不能直接使用一个
异构batched solver。本实验共享Python/CUDA运行时，同时保留每资产独立solver。
IsaacLab没有公开的多个活动SimulationContext接口；本仓simulation_slot.py集中管理
串行active-context lease和官方use_stage scope，11个已审计runtime文件SHA不匹配就拒绝。
严格kitless/headless、单线程、同device/physics，无render/camera/builder hooks/reset回调；
每slot独立USD stage，所有native操作在对应lease内，最后整个pool shutdown才clear。
这属于针对固定runtime的适配，不能泛称官方支持任意多个SimulationContext。

packed_probe先用4资产小面板进行原方式A/A重跑和A/B共进程逐字段精确对照，包含
真实双桌、复杂垃圾桶、足球及最小quota资产，固定open-loop action、真实autoreset及
显式reset，0optimizer。小面板通过不能替代全273资产容量结论。后续完整resource_probe
保留actor batch4096/H32/WM batch/window/replay深度1000，只允许显式
--probe-num-envs 2048作为用户授权的独立部署对照；其report scope和差异记录与默认
4096 readiness区分，不能被原启动器误接为4096通过证据。原数据/划分/物理/奖励/PD
与训练日程均未改，是否节省内存/显存与吞吐以真实GPU报告为准。

同进程内还显式复用完整只读ShapeAssetBank；每资产仍独立motion/FK/bin状态。
每次load仍验证bundle、native/source/reference及静态geometry/hands文件SHA；仅省掉
重复的静态反序列化和registry分配，并检查内存中原geometry/physical tensor是否被意外写入。
collector-only probe用已绑定clip长度计算精确outgoing bin数，再调用原allocate_workers，
避免父进程额外构建一份无用的全量CPU motion/FK；每child仍加载完整所分配数据并核对实际bins。

CPU最终完整回归536项：534通过、2项环境限制跳过；历史边界6项通过。GPU新版数值对照作业1488639，独立实验根目录
/mnt/projects/ll/tianzizh/hoi_packed_collectors_20261101_01，source_02/parity_02。
旧诊断1488637因父进程重复全量CPU加载被新版替代并主动停止，回执保留；不是训练取消。
原训练1488620保留。当前尚未得到packed数值对齐及完整273组容量结论，不据此减少原作业资源。


GPU集成smoke1488653已COMPLETED 0:0：asset266（无桌）和asset0（双桌）各2env，
同一个进程完成12步、显式reset和整池shutdown；context/model/state/solver对象身份独立，
原物理与确定性配置未改。采集GPU采样峰948MiB、含父进程PSS峰6.44GiB，仅是两资产样本。

生产模式A/A对照1488639未达到逐位一致，尚未运行A/B：初态/动作/xi完全相同，但旧架构
两次运行的asset209首步物体位置相差0.379mm、角速度0.356rad/s；64步内有接触标签差异，
asset266末段发生一次高度终止分歧。native资产/几何/质量/摩擦读回证据相同，但没有完整
solver隐状态证据来归因。当前Newton模式是not_guaranteed；接触/约束分配中存在atomic
操作，不能因此宣称本次差异已被完全解释，也不以删除字段/放宽阈值判通过。

新增显式--audit-isolation小面板诊断：构建、step或reset一个asset前后，对其他未激活asset
的真实Newton/MJW数组（含contact/efc）、episode/bin状态及保存的RNG做原始字节SHA；
要求完全不变，保留NaN payload，不用对象id代替内容证据。仅smoke/parity小面板开启，
正式train/resource默认关闭，物理参数、确定性模式和接触/终止阈值均不改变。

全273资产4096/2048容量作业1488771已提交到已通过native物理验收的dj-a40-0（实际A10），
3GPU/30CPU/96GiB，source_03；仅执行资源采集与完整batch反传，0optimizer，不能凭其
容量通过就替换原teacher训练。A/A及隔离验收边界与容量测量分开记录。

非干扰诊断实现后的完整CPU回归545项：543通过、2项环境限制跳过。真实缓冲区隔离GPU验收待完成；CPU替身测试不冒充物理证据。
