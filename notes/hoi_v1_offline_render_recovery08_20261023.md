# HOI v1 source08：既有 source06 rollout 离线渲染验收

2026-10-23。此次只修复并验证离线渲染，没有重新运行 WM、policy inference、评估仿真或正式训练。source06/07 冻结源码及原 checkpoint、evaluation、失败 render 目录均保留。

## 故障与最小修复

原 pipeline 作业 `1459842` 已完成 native gate、2 次 WM 更新、checkpoint 和完整 reference-duration 评估。评估命令退出码为 0；失败发生在后续 renderer。所用 `metadata/ultra_g1.urdf` 是搬迁后的虚拟39-body URDF，其相对路径 `meshes/pelvis.STL` 在 metadata 目录不存在。

source08 新增显式 `--robot-mesh-root`。该目录替代 URDF 所在目录解析相对 mesh 名称；原 URDF、visual origin、scale、保存的世界状态及两个橡胶手 STL SHA 校验均保留。默认路径缺失仍报错；不按 basename 搜索，不静默换为 collision artwork。receipt 保存实际解析后的 root、每个 visual mesh 的路径和 SHA。

本地7项 renderer CPU tests 通过，包含默认缺失报错、显式 root 成功、错误手 SHA 仍拒绝。另用既有286帧历史录制 fixture 验证搬迁 URDF 的真实预览；该 fixture 仅用于渲染回归，不是本次 v1 策略成绩。

## 本次实际作业与输入

恢复渲染作业 `1459900` 使用新独立输出，提交前读取 live queue，并以 `submit.claim` 防止重复 producer。最新 `sacct` 和 `scontrol` 确认 `COMPLETED`、`ExitCode=0:0`、运行6秒；申请 L40×1、4 CPU、8 GiB、10分钟，batch MaxRSS 为 `922552K`。L40 只提供 EGL 离线图像上下文。

- 执行源码：`/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_08`。部署包 SHA：`9723db9e7431e2734a655426a2957d20c29188596235babd58b1b52381c7ef22`。
- renderer 源文件 SHA：`43abae14ce17d8a6787b3f31f438e8f215e64cb0217843da7d23e4e76f1176e1`，与本地审阅文件一致。
- 既有评估：`source06/results/pipeline_train1_20261023_01/evaluation/evaluation.json`，实际完整路径见本地 request；SHA 为 `4ed4dd07061c92c58eaa3a79651b1da28777413c073666e7116618efc95e06bd`。
- 既有 checkpoint_000002.pt SHA：`974c3c56d6ce693c9b99f41ba3fca9996c935b612a0374777d8edbf16e4ff28b`；既有 rollout SHA：`00925256985d9b4bc7505dbc00ee216be14d4c7c3ae397350b99838af91bc2d1`。
- 保留原 `metadata/ultra_g1.urdf`，SHA：`1791b76b498fd60f4b29dfbf6cc82d2144df6ea739ef6d3cf32442168a3097a1`。
- 显式 mesh root 参数：`source06/data/g1_description`。receipt 记录 symlink 解析后的只读实际目录；两个手部原始 STL 校验通过。
- 物体采用显式请求的43-hull collision display，共3402个三角形；这是已记录的碰撞体显示，不宣称与原美术资产相同。
- 远端新输出：`/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_08/results/render06_saved_20261023_01`。

## 完整帧与视觉检查

源录制为220个 control intervals、221个 state/reference frames。远端完整编码后实际解码221帧；下载后本地 ffmpeg 再次独立解码仍为221帧。视频 SHA 与远端 receipt 一致：`6c4dc77811072ffd93f6b6adc1361ebc25702d8e03fcf00cd70fba7869cf22c6`。

50 fps，最后真实状态时间为4.40秒，视频显示时长为4.42秒；二者相差最后一帧的显示间隔。没有截断最后状态。receipt 明确 `physics_steps=0`、`policy_inferences=0`、`forward_kinematics_reconstructions=0`。

已用 view_image 检查第0、110、220帧：actual 保留独立机器人配色和橙色物体，reference 为半透明蓝色；共享固定相机和原世界坐标，没有分别对齐。末端 reference 物体与 actual 物体的漂移清楚可见。

## 科学边界

原 smoke 的 completion 记录 `wm_updates=2`、`policy_updates=0`、policy gate 未通过。旧 source06 报告中的 `HOI-trained` protocol 文本不能作为 actor 已进行 HOI 更新的证据；source07 已采用中性描述及明确 this-run 计数。此次渲染没有改写原报告。

这是一条 train-partition motion 的管线诊断，不是 canonical80、test40 或20k正式训练。原评估 human normalized return 为0.854098、HOI normalized return 为0.748037，PHC pass 1/1，但 object-pose pass 0/1；不能称为 HOI 或抓取成功。全量缓存、8资产证据及4096/H32资源验收分别记录，不能由此视频替代。

## 本地复核文件

- [请求与资源](../data/hoi_v1_render08_20261023_01/request.json)、[独立解码核验](../data/hoi_v1_render08_20261023_01/verification.json)、[Slurm log](../data/hoi_v1_render08_20261023_01/slurm-1459900.log)。
- [render 总 receipt](../data/hoi_v1_render08_20261023_01/render/rendering.json)、[case receipt](../data/hoi_v1_render08_20261023_01/render/case_00000/rendering.json)、[完整视频](../data/hoi_v1_render08_20261023_01/render/case_00000/rollout.mp4)。
- [初帧](../data/hoi_v1_render08_20261023_01/render/case_00000/frame_000000.png)、[中帧](../data/hoi_v1_render08_20261023_01/render/case_00000/frame_000110.png)、[末帧](../data/hoi_v1_render08_20261023_01/render/case_00000/frame_000220.png)。

视频、图像、几何与结果保存在 ignored data 目录，不加入 Git；未向 W&B 上传。
