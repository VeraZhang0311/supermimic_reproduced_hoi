# 与ULTRA训练足部碰撞几何对齐

用户在授权五通道contact新版从零训练后，追加要求核对ULTRA足部并统一。该要求允许新实验改变脚部碰撞几何；旧zero-start1461253继续使用原source13与球体足，绝不原地同步或重启。warm1461248已按授权停止，最后完整3500 checkpoint保留。

## 核对依据

`D:/Ultra/ultra/data/cfg/g1_teacher.yaml:27` 的 `robotType` 为 `g1/g1_29dof.urdf`，`ultra/env/tasks/humanoid_g1.py:54–68` 通过实际 `gym.load_asset` 导入该URDF。左右 `ankle_roll_link` 的 collision 均为完整脚STL，origin xyz/rpy全0、scale缺省为1；不是四个碰撞球。实际训练开启VHACD，max hulls5、每hull最多16顶点、resolution60000。URDF的visual标签不是本结论的依据；另一个sim2sim MJCF也有active mesh-foot，但主结论以HOI训练加载路径为准。

ULTRA原始三份资产：`ultra/data/assets/g1/g1_29dof.urdf`、`meshes/left_ankle_roll_link.STL`、`meshes/right_ankle_roll_link.STL`。运行时按SHA绑定副本，不往Git加入原始STL/完整资产。源路径、字节数、SHA另写foot manifest。

ProtoMotions的原ULTRA URDF脚部同样是STL，但迁移后的Lab运行时资产可能用不同的capsule近似，不能把它冒充作者ULTRA的实际训练足部。

## 新版本范围

在既有不可变手overlay之上创建独立foot overlay：只关闭左右脚旧有效碰撞体，添加从原ULTRA脚STL烹制的explicit convex hulls。原手部STL、已验收手hulls、物体/源数据/参考时基/FK、visual、质量/COM/惯量、joint/PD、其余碰撞与self过滤规则均保留。旧资产、旧runtime和旧结果不写入。

当前Newton/MuJoCo不是作者PhysX。新转换复用同源STL与安装变换、采用已安装CoACD的5hull/16vertex约束；原VHACD5/16/60000与实际CoACD recipe分别记录，明确 `equivalent_to_physx_vhacd=false`，不把CoACD采样参数映射为PhysX的60000 voxel resolution，不声称凸分解或接触逐位相同。

原hand-asset.json原字节复制，只描述原手overlay；foot-asset.json独立绑定原hand overlay/receipt、新USD、原URDF/STL、实际cook结果、限定属性变化与native profile。共享loader增加显式foot-extension校验，保留旧手校验路径。train/eval80创建新bundle header，只替换显式native机器人碰撞身份与扩展记录，原source/reference/object/hand caches保留同路径同SHA；新完整bundle digest与native gate不能复用旧版本。

原生探针改为从实际编译foot mesh_graph凸包取支撑面及最低点；重新验证左右脚—物体正例、脚—地面负对照、实体/环境隔离、四子步阈值、全环境mesh身份和原物性。之后才做fresh4096/H32资源检查与正式from-scratch20k/500保存/5000完整80评估。

## 已保留的失败

独立source14八资产原生job1461431在asset0加载默认FK MJCF时退出：新目录缺少 `data/g1_description/g1_29dof.xml` 绑定。此时尚未创建仿真，未生成真实transition或执行optimizer；不是脚contact数值失败。source14 archive `0540654bf17453a6f2c76265180eb2b372682df5493bf15846fbf4ff3899c4c1` 与原失败输出保持不可变。下一快照补全指向既有不可变G1资产的只读路径绑定，并纳入本次网格足扩展后再验，不改旧任务。另在login节点读取USD遇系统libstdc++版本不足，仅影响该只读检查；不改共享库/驱动，USD authoring在已准备的compute runtime执行。

## source15 验证进展（2026-10-24 UTC）

本地最终完整CPU回归261项：260通过、1项因Windows缺少USD库跳过；旧HOI边界6/6通过，独立实现review未发现待修P1/P2。source15在新GPU节点实际执行USD authoring，已成功生成overlay及独立foot receipt，补上本地无法运行的USD创建路径；这一步不等同于原生接触门通过。

冻结目录 `/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_15`，185文件逐项SHA校验；archive SHA256 `b1f58fb4da1acee3c09bfa764b21dd669aa7fac34503399ec437e0f25eae4fe5`，可执行源码内容SHA256 `1bf7c0cfad3e9b64d2bf460112db3d85f5733887b59e58a8336613f57439e7fd`。新目录绑定既有只读G1 MJCF/USD；原source13/14及其结果不变。

独立验证job **1461502**，comment `hoi-v1-ultrafoot-validation-source15-20261023-01`，2026-10-24 03:54 UTC已RUNNING。输出为source15下 `results/contact5_ultrafoot_validation_20261023_01`。机器人新USD SHA256 `22070392b2adcbd926a784db3e6a51459d80abf6719927904443406aa1e36919`；新train bundle digest `e88b4e69a23d4f95555949f16233cbb79e509367e5819327ff5d15fe6625c983`，新eval80 bundle digest `adc9875cbe823546efc151b1ad9d07589a71808636675687ee549c0f790e0715`。旧所有reference/geometry caches保持原路径及SHA。

流水线按顺序完成八资产原生门后，才运行fresh4096/H32/64prefill资源检查；仅反传、不执行optimizer.step。原生门、完整资源结果、正式提交与启动状态以之后追加的真实回执为准。目前未提交新正式训练。

04:13:41 UTC，八份 `native_asset_0..7/native.json` 均已生成：每资产8env/64steps、`native_pair_calibration_complete=true`、`compiled_foot_audit.passed=true`，所有双手/物体/双脚正例与手脚碰地负对照通过。各模型环境左右脚均为5个真实编译mesh hull，几何与cook的最大顶点误差分别约5.16e-9m/4.37e-9m，原质量/COM/link-frame惯量检查通过。随机零动作64步没有要求每个通道都自然出现正例；受控正例单独确保5个通道可记录。这不是学得策略的HOI成功率。

## GPU验收通过与正式提交

04:16 UTC，job1461502最终 `COMPLETED 0:0`，八资产聚合严格门 `passed=true/blockers=[]`，evidence SHA256 `8b40f0958b504b58b598d22a7bbed5b532c4e58ff0b851f2f8b9dbc7b44ed9ab`。完整资源报告为source15 `results/contact5_ultrafoot_resource64_20261023_01/resource_validation.json`，SHA256 `2e26b8615526f0ddef6e05e06d1ce24d9b197e65a42bba8016d613b066dde1cd`，`passed=true`。全train2651/381groups/756524states/753873controls，8worker真实4096配额、replay1000、H32，全部生产参数保留。

fresh64真实prefill共262144条transition，五通道均有262144个有效interval；正例数依次为27431/27053/168515/4486/6879。WM2048×8全有效反传时每通道16384个监督位置，左右脚输出行梯度范数约1.062e-4/5.155e-5；actor4096×32反传梯度有限，critic135168 target及20个32768 minibatch反传均有限。holdout有限；fresh模型质量基线未通过是预期诊断结果，不控制固定warmup后的actor更新。probe没有执行optimizer.step，最终三Adam均仍空、参数SHA未变，iteration0；这些fresh probe模型未保存或用于正式实验。

显存：已预留156067752bytes的两组完整Adam moments；父进程allocated峰21773291008bytes、reserved峰22127050752bytes；本作业进程nvidia-smi采样峰41966MiB，最终全局used44060704768bytes，OOM0/alloc retries0，无其他job进程占用此GPU。采样峰仅为采样下界；资源probe不执行Adam kernel，不替代正式连续优化稳定性证据。

04:17 UTC，真实路径CPU preflight与唯一提交守卫通过，正式job **1461591** 已提交并开始初始化。新输出source15 `results/hoi_v1_contact5_ultrafoot_scratch_20k_20261023_01`；独占comment `hoi-v1-contact5-ultrafoot-scratch-source15-20261023-01`，提交回执 `data/contact5_ultrafoot_formal_submission.json` 绑定源快照、helper、preflight、resource/native报告SHA及实际命令。固定 `--from-scratch`，无init/resume；20k/500保存/5000完整80评估不变。正式首次日志、W&B和显存结果后续补记。

04:22 UTC正式启动检查通过：8worker、64真实prefill及首轮10次WM优化完成，holdout有限，W&B新独立run `61f872168c46457f9fc74bfe9d63ea9b` API为PRIVATE/running，五通道/新数据/网格脚身份全部回读一致，当前GPU used30631MiB。前500 iter为WM-only，正式actor首次501尚未到达；详细数值与界限见[正式启动记录](hoi_v1_contact5_scratch_experiment_20261023.md)。
