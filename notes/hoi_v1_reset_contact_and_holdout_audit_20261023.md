# HOI v1 随机接触阶段初始化与holdout采样阻塞核查

后续状态：holdout修复已部署；真实OOM另以no_grad target等价分块及运行时expandable allocator处理，连续50轮完整GPU优化已通过。两版旧任务退出后，现warm1461248/scratch1461253分别从最新完整1500/500恢复，原科学参数和质量门保留。实际作业与恢复后验证见 [部署与续训记录](hoi_v1_holdout_resume12_20261023.md)。以下「尚未部署」为发现问题时的历史状态。

用户担心每次从动作开头重置，policy碰不到物体，导致WM接触片段稀缺。本轮核对真实代码与两份运行日志，未修改冻结source11或任何运行任务。

## 训练reset实际语义

`HOITrackingSim`的`AssetTrackingSim.sample_frames()`在当前固定asset/scale/profile的全部兼容训练bins中抽样，不限定clip起点：50%均匀、40%低回报soft分布、10%其EMA，再在选中的50帧（1秒）bin内随机抽一帧。动作中段和reference已有手物接触的阶段都可成为episode起点；不是专门按contact phase增采样，没有固定的接触起点配额。

`HOITrackingSim.reset()`先调用原robot reset，恢复该时刻根部/关节位姿与速度，再把物体pose、link-origin线速度/角速度恢复到同一reference时刻，并校验物体readback。训练保留原机器人reset扰动，例如根部xy各±5cm、关节±0.1rad；物体仍取reference。因此抽到reference接触帧，不保证初态实际力学接触或长期抓持。没有粘手约束，也不把reference接触标签伪造为实际接触标签。

后续真实动作在物理引擎执行，四substep内积分pair正常冲量，平均力>1N才为真实contact positive，再写入joint replay。WM窗口从真实replay采样；terminal transition保留，reset之后的跨episode部分mask掉。模型fit正例按negative/positive ratio加权，但没有contact-aware replay window抽样。完整canonical80 eval仍执行完整源时长，不采用随机起点来提高成绩。

## 本轮只读真实观测

| 实验 | 最近日志iter | WM更新 | policy更新 | 累计有效fit左手正例 | 右手正例 |
|---|---:|---:|---:|---:|---:|
| warm-start1460279 |1550|15500|0|1114877|974164|
| scratch1460725 |300|3000|0|356557|321552|

这些是`Learner.observe()`中新采集、valid且属于fit split的控制区间计数，不是WM重复抽样次数；多次reset同一reference phase仍可重复计数，所以不能证明独立episode多样性、持续抓稳或contact/onset/release的1/8/32步覆盖。

最近warm `wm_gate_001551.json`和scratch `wm_gate_000301.json`均报`insufficient holdout replay coverage for WM windows`，没有horizon reports。warm已经超过500 WM-only warmup仍无policy更新，需要定位，不能称已正常进行actor训练。

## 已复现P1：有限拒绝预算误报coverage

原`ReplayBuffer.sample_windows(B=2048,k=32,split=holdout)`最多32轮，每轮约2×remaining个uniform(age,env) proposal。holdout为reference每20个50帧bin保留1个；连续轨迹中完整32步窗口的合法起点率约(50-32+1)/1000=1.9%。期望32轮后remaining约2048×(1-2×0.019)^32≈593，因而有合法数据仍频繁报coverage不足。

root用原实现、连续T1000/N2 replay（19/969起点年龄合法、有放回抽样）在seed0/1/2均复现。独立agent另用T160/N1000、129000候选/2451合法起点（合法数本身大于2048），三个seed仍分别仅取得1433/1484/1424，全部错误报不足。这是实现的抽样预算问题，不是WM预测通过/失败，也不能据此推断实际接触稀缺。

## 本地候选修复及验证

本地`hoi/v1/buffer.py`保留原32轮快速拒绝路径；不足时只枚举f/ep_step/done/asset_id元数据的精确合法起点，再uniform with replacement补齐remaining。不存在任何合法起点时才报coverage。接受条件与原`_windows_at`逐项一致，保留terminal prefix、reset/reference jump/asset隔离、fit/holdout划分、B/k及所有训练参数；不修改接触比例、不放宽gate、不缩batch或H。

原拒绝目标是uniform proposals条件于合法起点。已有accepted样本与fallback都在同一合法(age,env)集合均匀有放回抽取，合并仍同分布；随机数消耗不同，不声称逐位一致。没有缓存会随新replay变旧的候选集合。

验证：新增4个回归覆盖生产B2048/k32稀有split、ring/reset/asset/reference jump、terminal prefix及真正空split有界失败；完整CPU回归216/216通过（26.466s）。独立review另外200个随机metadata buffer、1034次pool逐ID对照全部一致，未发现剩余P1/P2。第一次新test运行只有fixture int64赋值到int32的测试数据错误，已修正；不是生产实现失败。

默认1000×4096枚举只需O(TN)元数据而非全部state窗口，额外临时显存估计约0.15–0.2GB；此为静态估计，尚未GPU实测。修复尚未freeze/部署/提交GPU任务；当前两个实验仍是原source11。没有取消、重启、续训、改旧runtime或commit/push。

若后续部署，须新源码快照、新输出、实际GPU采样/WM验证与checkpoint来源记录；不能覆盖运行源代码。恢复checkpoint仍需fresh replay，不是逐位连续。即使修复采样成功，原baseline改善与contact phase gate仍可能因真实学习质量不足而关闭，不能承诺policy马上开始更新。
