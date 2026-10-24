# HOI v1 实现与首次实验准备记录

分支：`hoi/v1.0`。依据 `supermimic_hoi_v1_final_design.md` 和 ULTRA 数据/timebase 交接实施。本记录区分代码、CPU 检查、原生物理检查和正式实验；下文准备阶段的「尚未启动」均为历史状态。

**最新正式恢复：用户已明确授权启动两版并部署修复后从最新checkpoint续训。source13的holdout sampler修复、数学等价critic target分块和运行时expandable allocator已通过219项CPU回归及50轮完整GPU优化。当前warm1461248/scratch1461253从1500/500恢复，20k/保存500/完整80评估5000及其余科学参数不变；实际holdout、policy、显存与W&B验收见[部署与续训记录](hoi_v1_holdout_resume12_20261023.md)。不要按旧准备阶段文字重复提交或停止已获授权的正式任务。**

**最终验收（22:44 UTC）：全量缓存、完整bundle八资产原生门、实际全数据4096/H32及64步预填充资源诊断、source10 CPU preflight全部通过。首次实验已就绪，正式训练尚未提交。** 最终源/数据/证据身份、完整命令与容量限制见[首次实验就绪记录](hoi_v1_first_experiment_readiness_20261023.md)。下文保留过程历史，以该最终记录覆盖旧“待完成”状态。

## 代码范围

- `hoi/data/`：canonical manifest、split/source-group/selection SHA、严格630D schema、整数时钟60→50Hz、source/derived velocity 分离。
- `hoi/v1/prepare.py`、`geometry.py`、`probes.py`、`motion.py`：不可覆盖的哈希绑定 bundle；真实 compiled collider、原手 STL 与安装坐标；固定中央掌面 ray probes；FPS128、signed grid、精确 reference surface gap、contact unknown mask；原14body human reference/goal。
- `state.py`、`networks.py`：joint robot/object state；原 actor/WM/critic 前缀保留，追加18D object observation、190D object goal、独立 PointNet 和 normalization；contact head；实际生产尺寸不因 smoke 缩小。
- `mdp.py`：复用原 human reward/termination/obs/PD 语义，追加 object pose/velocity、palm approach、expected/actual contact reward 和宽松 object positional failure。总奖励上界0.15/step，human仍0.1/step。
- `contacts.py`、`native.py`、`workers.py`：真实动态物体，质量/COM/惯量/摩擦 readback；link/COM velocity；每5ms子步的实体配对 normal impulse；四子步累计为20ms标签；固定asset/scale/physical group worker，中央单一 learner，显式seed和CPU IPC。
- `buffer.py`、`learner.py`、`train.py`、`gates.py`：真实联合 transition、连续episode history、reset/window mask；真实执行 action 的8步 WM；32步 actor BPTT；冻结WM/critic参数但保留输入Jacobian；critic每次mini-batch重新计算PointNet；1/8/32步和contact分层诊断；native和policy启动门槛。
- `checkpoint.py`：严格feature/architecture/data/contact/physics身份校验，三个optimizer保存恢复；旧 checkpoint 零扩展初始化，来源physics不同时显式记录新迁移实验；相同schema续训仍需fresh real replay。
- `evaluate.py`：冻结权重与统计，完整时长、不因failure重置，T+1真实状态/reference与T控制字段；独立39body ULTRA FK/PHC、human/object/contact/joint指标。
- `configs/hoi_v1_first.json`、`configs/hoi_v1_pipeline_smoke.json`、`cluster/hoi_v1_train.sbatch.sh`：正式默认与显式小规模验证配置分开，独立checkout和新输出目录。

共享层只增加 scene factory、CPU-safe robot metadata/MJCF选择，以及只用于类型标注的延迟import。原 `hoi.run`、`hoi.prepare`、zero-shot policy observation、原网络/PD/奖励公式保持原入口。历史工作区、已运行实验和共享runtime未被覆盖。

## 训练数据与统计边界

WM holdout 采用训练数据的global reference 50帧bin编号模20为0的区间。未mask的WM fit window不能进入这些区间；holdout行也不更新WM normalizer或BCE class counts。policy的真实观测统计仍来自train分区。该holdout独立于validation/eval80/test40，属于新增启动诊断，不能称原motion-only实验完全等价的采样过程。

policy在至少500个WM-only训练迭代后（默认每迭代10次WM batch更新），还需通过明确记录的train-only startup proposal：1/8/32步真实动作预测优于zero/CV对照，接触/无接触/onset/release有覆盖，三对实际接触的正负样本和概率诊断满足门槛。尚未经验标定，不能由通过门槛推导HOI成功。门未开时只继续WM，不减小horizon或借用eval/test调阈值。

每条transition保存failure reason bitmask：height=1、tilt=2、object-distance=4、clip-end=8、timeout=16；后三者与failure/timeout分别统计。`r`为HOI总奖励，`r_human`为原human奖励，`r_hoi`为新增部分。global frame与不可变bundle提供clip/source-group/manifest映射，不重复存静态点云。实际episode-reset frame bin计数另存`sampling.json`。

## CPU 验证

已在原有CPU环境执行相关单测与CLI检查；本地没有安装或升级依赖。`D:/Protomotions/.venv/Scripts/python.exe` 提供真实MuJoCo CPU能力，可执行编译碰撞过滤及FK检查；`D:/SuperMimic/.venv/Scripts/python.exe` 没有MuJoCo，其可选检查会明确skip。

本轮阶段性全套结果为128项通过；原 `hoi.test_reference`、`hoi.test_hand_collision` 另6项通过。新增渲染及真实原生bundle检查完成后更新最终数量。两份新增Slurm脚本通过`bash -n`；`git diff --check`通过。CPU合成worker测试仅验证流程和梯度，不作为原生物理证据。

后续含渲染/mask/证据链的全套已达138项通过。三个新增Slurm脚本均通过`bash -n`。渲染用历史真实状态作显式fixture，完整286帧独立解码286/286，root实际检查预览；不是新策略的成绩。

第04快照含surface-intersection保守mask修复后，全套 **166/166通过，无skip**（`D:/Protomotions/.venv/Scripts/python.exe`），原HOI边界另6项通过；五份新增Slurm脚本全部通过`bash -n`。日志保留于本地忽略目录`tmp/hoi_v1_cpu_snapshot04_tests.log`。第04不可变源码archive SHA256：`485df315aa87d9dce9bb99551025875da4e9b442ecc7fd41078d5ad50581c678`，部署到全新`supermimic_hoi_v1_development_20261023_04`目录；第02/03检查中的源码未被覆盖。

第05快照进一步补齐设计要求的报告与两个消融入口、浅压pair校准及完整资源probe，全套 **188/188通过，无skip**。日志`tmp/hoi_v1_cpu_snapshot05_tests.log`。SHA256：`624a4e824df09e17b312a56c2480c250645c03c196ea3e6b58dbbc980bdab295`，164个源文件，独立第05目录。source39 hand flags按绑定body names查找，只作差异诊断；actual/expected contact与support的onset/release只连接相邻有效interval；同真实初态、真实执行动作的1/8/32步预测reward gap只作冻结评估，不进入训练或启动gate。

两个消融明确记录`variant`并在checkpoint恢复时严格核对：No-contact-objective只将contact-match权重置0，head/BCE保留，objective/sampler上界0.145；No-geometry-input保留三个PointNet及维度，将shape64一致置0。默认主实验保持0.15与全部原数值设置；本轮不启动消融。

Reference mask新版本显式记录surface intersection：细长凸块的边/面相交可能没有任何顶点处于另一个凸块内部，不能仅凭vertex penetration=0当作可信positive。检测到的边/面交叉及无法区分的边界touch标unknown；不改变2/4cm和严重穿透2cm的设计阈值。完全包含、没有surface交叉的小于2cm浅穿透仍按原阈值处理。

全train数据审计`1459687`已`COMPLETED 0:0`，4m13s，2CPU/12GB，无GPU/训练。2651/2651条均通过SHA/schema/clock检查，读取2,294,586,216bytes、908,982源状态，得到753,873完整控制区间、756,524个50Hz状态，`errors=[]`。报告SHA256 `1547097d454c11da5ddd2187517222f4f6efe8bebf8e93593fa9e9ddd123284d`。详见[全训练集审计记录](hoi_v1_full_train_data_audit_20261023.md)。这确认canonical协议采样，不证明发布数据真实生成步长为1/60s，也不代替源velocity link/COM物理语义证据。

## 已完成的真实原生验证

独立checkout：`/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_01`。base source archive SHA256：`d4f19b05f725e6d9b546cacce5df92034961c35a380a688bb1ec794d868815c4`；随后单文件修订各自记录SHA。运行只读复用准备好的runtime，G1资产采用只读路径引用，Warp/cache/output均在新目录。

Slurm `1459665`：1×L40、24CPU、96GB，`COMPLETED 0:0`，6m15s。输出为新checkout下 `results/native_probe_20261023_01`。属于legacy fixture单asset instrumentation验证，没有HOI reward/optimizer/replay，也没有启动正式训练。

- hand ON/self ON编译审计、真实质量/COM/惯量/摩擦、link-origin velocity reset通过。
- `(t,t+1]` 四个5ms子步pair normal impulse记录通过；64个zero-action control steps完成，无capacity overflow。
- 故意overlap的左右手—物体positive probe均产生配对力；object-ground positive probe通过。overlap力值不是自然动作成绩。
- 手—地面负对照确有地面接触力，两个手—物体channel均严格为0，排除用body net force误标。
- 该首轮尚未绑定正式训练bundle，不能直接打开training native gate；掌面定义与bundle/compiled geometry一致性由后续轮次验证。

第二轮`1459688`：largebox100、8环境，`COMPLETED 0:0`，1m46s，输出`results/native_probe_20261023_02`。pair校准和64ticks通过；左右中央掌面16点与原STL最大距离分别8.62mm/6.61mm，保留逐点source/collider triangle IDs、点/源码/图片/碰撞几何哈希。root实际审阅左右手三投影图，点位在中央掌面。

真实验证发现同recipe/seed重新生成CoACD手hulls仍会变化。因此canonical bundle新增`native_robot_asset`，绑定已验收overlay USD及hand-asset manifest SHA；worker只读复用同一资产，再比对实际compiled手/物体collider与cache几何。该修复用于保持reward/scene几何一致性，不更改关节/PD/质量/原始STL，也不声称与PhysX cooker相同。

第03快照largebox100的graph-authoritative导出已完成：43个object凸块及每侧5个手凸块，surface来自MuJoCo实际`mesh_graph`凸包；独立CPU `mj_geomDistance`四个分离姿态与cached triangle surface最大差1.02e-9m，原生机器人30body FK最大位置差0.734µm，四项pair校准及64control ticks均通过。legacy fixture报告尚未绑定新bundle，不能代替正式训练native gate。真实source39的独立审计另见[原始39body/FK审计](hoi_v1_source39_fk_audit_20261023.md)：266个source帧最大位置差0.736µm，支持link-origin pose；不据此确认source velocity语义或生成时钟。

实际24000轮legacy checkpoint的迁移诊断通过：55个旧张量对应前缀逐元素相等，新增输入列为0；actor/critic前向无差异，WM的float64最大绝对差1.51e-14。原float32严格断言未通过，WM最大差9.15e-6，完整保留失败证据与扩维矩阵求和舍入诊断，没有更改模型来通过检查。三个optimizer的新schema checkpoint保存/重载已通过；该文件仅为诊断身份，不作为真实bundle训练checkpoint。

第03多资产深度重叠校准`1459765`在plasticbox080触发BROADPHASE capacity overflow后停止，不能称8资产全通过。前5组已完成readback/graph/GJK/FK/pair/64ticks；剩余组继续使用新输出。第05把手—物体校准改为真实中央掌面与物体union支撑面的2mm轻压，保存逐probe pose/支撑面/normal/深度/interval trace与失败文件；仍使用原contact128/constraint500容量，不吞掉overflow。真实reference collection若发生overflow，必须另行记录并验证HOI容量资源配置。

第04单条真实v2 reference bundle已经完成（221states，2m58s，约1GiB CPU）；`1459824`正在检查同bundle native gate、生产尺寸网络64步prefill/2次WM更新、checkpoint重载、完整220步真实评估与离线渲染。该作业属于smoke，未完成前不作通过结论。

后续真实结果：第04单条bundle native gate已经通过，但worker首次IPC因长`TMPDIR`触发Linux `AF_UNIX path too long`，未执行WM更新；失败输出保留。第06新增短job临时目录helper，限制base长度并用`mktemp`新建，cache与科学结果目录不变，不更换IPC协议。

第05轻压校准证明plastic080左右手—物体及object-ground正例、hand-ground负对照均成立；但真实reference reset后的64tick检查仍触发`NEFC`约束容量500溢出，确认不能只修probe。第06为**HOI独立sim config**明确增加contact容量128→512、constraint容量500→2048；深拷贝共享cfg，原motion-only/zero-shot仍保留128/500，solver迭代/积分/PD/质量/摩擦/重力不变。该资源配置变化记录于native report，overflow仍立即拒绝进入replay；需重新实测8组与4096环境显存，不能用CPU测试代替。

第06 CPU几何优化只缓存已验证的静态拓扑/edges，并保留每帧有限/形状检查及原float64刚体变换和全部精确triangle/edge/intersection算式。真实221states的6个contact输出与第04cache逐位相等，计算80.081s→18.907s（4.24×）；独立旋转/平移/嵌套/分离测试也逐位相等。reference质量按T个后端点、逐clip统计，不跨clip制造transition；首clip左右valid216/192、positive116/91、negative100/101、gray unknown4/28，无surface intersection/severe depth。plastic实际有134/121凸块，首largebox43凸块速率不能直接当全量耗时。

第06完整CPU回归 **194/194通过，无skip**，原HOI边界另6项通过。冻结源码archive SHA256：`60f6426942c983fba0dfbaf8337a02fa7858b9bd49381ac99eae1ba689ed129c`，165个源文件，日志`tmp/hoi_v1_cpu_snapshot06_tests.log`。第06原生重验中，先前失败plastic080已通过四项轻压probe和64真实tick；单条bundle native gate也通过并进入WM管线，完整8组/全部管线尚未宣告通过。

后续报告修正把评估名称改为中性的HOI v1 full-reference，checkpoint保存`wm_updates_this_run`和`policy_updates_this_run`，定义为当前新输出run的更新次数，续训不把旧run计数误作本次累计。评估只据已保存真实计数报告阶段；旧checkpoint缺字段标unknown，不能由iteration或当前gate推断actor已经训练。该修正不改变训练目标、梯度或超参数。

## 首次实验配置

正式配置保留4096环境和原优化设置。按用户最新要求，上限20000迭代、每500迭代保存，5000/10000/15000/20000自动提交完整80条冻结评估；LR decay仍采用原250000迭代schedule。replay每transition1157bytes，默认固定buffer约4.414GiB，尚不含模型、BPTT激活、worker模拟器或cache；全资源可行性需实测，不能从8环境smoke外推。

周期评估采用独立Slurm GPU作业与新输出目录，绑定刚保存的不可变checkpoint和同数据/几何契约的完整eval80 bundle；训练进程不复用评测状态或更新统计。提交receipt防重复producer，排队/提交成功不等于评测完成。W&B使用专用白名单入口，只有显式online或`WANDB_SYNC=1`才联网，且每次新run、不上传源码/数据/权重/rollout。

优先初始化来源是已稳定读取的历史self ON / hand OFF、24000轮完整checkpoint。新目录冻结副本：

`data/initialization/ckpt_self_on_hand_off_024000_e2b6e716c058e5448ab26caa22febc874dd50e2459435f1530f515dcd9fe8d9f.pt`

SHA256为文件名中的64位digest；218548065bytes。来源没有完整physics元数据，启动显式声明`--no-legacy-hand-collision --legacy-self-collision`，目标hand ON/self ON并传`--allow-initialization-physics-change`。这是新的迁移实验，iteration与optimizer从头建立，不称旧实验resume，不恢复旧W&B run。

完整命令和门槛见`hoi/v1/README.md`。在真实数据bundle、原生掌面/collider/pair gate、完整pipeline和资源检查完成前，准备状态保留未验收。正式训练未提交。

## 后续验收进度（以实际任务报告为准）

- 完整canonical eval80缓存及8条资源fixture已完成，07只读合同审计通过：eval80为23798状态/23718控制间隔，resource8为2088状态/2080间隔；八组geometry与hands cache字节SHA及manifest/recipe/overlay/physical/FK完全相同。塑料组存在较多surface-intersection unknown，保留原mask并逐组报告，不冒称完整contact phase覆盖。
- `1459848`八资产fixture物理校准与64ticks已全部通过。`1459882`进一步对真实resource8 bundle的全部八组通过严格native gate并完成聚合，无blocker；该证据只绑定资源fixture，完整train bundle完成后仍需自己的bound gate。
- `1459882`资源部分在CUDA allocator尚未初始化时调用峰值计数器而失败，尚未分配4096环境，不能视为OOM或显存验收完成。第09显式device选择/CUDA初始化后再reset计数器，保留原生产参数；仅资源部分在新目录重跑，不重复成功的八资产检查。
- 完整单条smoke的64步prefill、2次WM更新、checkpoint保存重载及220步/221状态冻结真实评估已经通过，policy更新为0，object pose失败，不能称HOI成功。离线渲染最初因metadata URDF缺相对mesh失败；第08新增显式`--robot-mesh-root`，保留原virtual39 URDF/visual transform/手STL SHA。`1459900`补验已COMPLETED0:0，221保存帧、远端221解码帧、本地独立221解码帧；渲染无policy推理/物理步进/FK重构。root审阅首末帧，actual原色/橙物体、reference透明蓝，共世界固定相机，末端漂移保留。视频SHA256 `6c4dc77811072ffd93f6b6adc1361ebc25702d8e03fcf00cd70fba7869cf22c6`，仅为管线证据。
- 第09完整CPU回归 **197/197通过，无skip**；原HOI边界6项此前已通过。09源码SHA256 `b03e3df856a4848b7d96f852a1cc5782937eb7de8e7999efdac35f7db31c5184`，部署于独立第09目录；08源码SHA256 `9723db9e7431e2734a655426a2957d20c29188596235babd58b1b52381c7ef22`，只用于新渲染补验。所有冻结06/07源均未覆盖。
- 全train2651缓存`1459845`继续正常运行。待其完成、全量source/cache/质量审核、完整bundle八资产native门与真实fulltrain配额下4096/H32资源通过后，才给出正式就绪结论。后续检查在本聊天每15分钟继续，只有完成/失败/需要处理时通知；**不会自动启动正式20k训练**。
- 第09资源单独重验已提交`1459905`，新输出`/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_09/results/resource4096_20261023_01`，同名Slurm日志在09根目录；先require既有resource8 bound native evidence，再实测4096，不重跑成功的八资产检查。该条仅记录提交与producer身份，不是显存通过结论。08实际视频补验详见[离线渲染恢复记录](hoi_v1_offline_render_recovery08_20261023.md)。
- `1459905`后续真实报告`passed=true`，九个资源阶段全部完成，所有optimizer step为0：4096环境×12步得到49152条真实transition；WM实际2048×8全部active且梯度有限；actor B4096/H32前后向与critic135168行target前向、32768行mini backward均通过。parent allocator peak allocated20.79GB、reserved27.03GB；同步全进程GPU已用46,663,335,936 / 47,590,277,120 bytes，仅余约884MiB。**这仍只是8条resource fixture的通过，不是全量训练集就绪**；全量reference及八个worker分bank可能超过现余量，必须用完成后的2651条实际配额复测。Adam两份moments已预留，未执行optimizer foreach临时分配；不能由此推导正式迭代吞吐或稳定训练。真实报告本地副本：`data/hoi_v1_native_receipts_20261023_01/resource4096_source09_20261023_01/resource4096_20261023_01/report.json`。
- 独立资源review补充：critic阶段parent实际allocated约6.06GB、reserved约27.03GB，差额包含可回收allocator cache；884MiB是同步CUDA全局空闲，不能直接据此断言增加全量reference将OOM。完整数据的真实复测结果才决定是否需要修复。全量验收后续已排依赖：`1459949`只读CPU合同/逐clip质量/全量配额与各bank storage审计、`1459914`完整bundle八资产严格native门及4096/H32资源实测，两者均`afterok:1459845`；生产源码与输出不覆盖，不重复缓存producer。`1459914`冻结的资源协议为12步收集，64步生产prefill补验将在独立review及该次真实结果后决定，仅新增resource-only输出，不重跑已成功native。
- 64步生产prefill补验已落实为`1459951`，`afterok:1459914`，source09无需代码改动。新输出`/mnt/projects/ll/tianzizh/supermimic_hoi_v1_development_20261023_09/results/fulltrain2651_resource64_20261023_01`；helper SHA256 `91c7e6c7adc7478101764713c5ae1cadbba457f201833e0338b665543e55e96a`，本地/远端一致且`bash -n`通过。先严格require完整bundle同hash的八资产native evidence，再执行4096环境、replay1000、64真实采集步及原WM/actor/critic完整尺寸图，0optimizer step。若上游失败不会启动依赖作业；不取消运行中的缓存，不重跑成功原生检查。全部真实结果通过前仍未ready。
- 后续只读状态helper：本地ignored `tmp/hoi_readiness_status_20261023.py`，有界SSH、实际`squeue/sacct`、partial数量、final report及日志tail；每次原始观测保留于ignored `data/hoi_v1_readiness_monitor_20261023_01/`新时间戳文件。2026-10-23 21:16 UTC实际观测：`1459845`仍RUNNING，sacct elapsed31:46，动态文件计数reference916/contact917，final bundle尚无；三个下游作业正常PENDING(Dependency)。文件计数为顺序读到的并发生产快照，不能据差1判断缺失或宣布clip完成。无新失败，无重提或训练启动；继续等待已有依赖链。
- 22:15 UTC真实观测：全量缓存`1459845`已`COMPLETED0:0`/1:21:48，2651 references与2651 contacts及final bundle齐全；独立CPU审计`1459949`已`COMPLETED0:0`/1:16，报告passed。root已回拷并审阅脚本/报告、实际4096配额、逐组质量及bank storage。完整bundle content SHA256为`318482c78f699d33aa5ae7743f2918f19d96fa3d78e2869add50f4a890d022c8`。详细证据见[全训练集缓存验收](hoi_v1_full_train_cache_audit_20261023.md)。`1459914`正在运行，`1459951`正常依赖等待；仍未完成全部GPU验收，不宣称ready，不启动正式训练。
- 22:37 UTC真实观测：`1459914`已`COMPLETED0:0`/25:25，完整bundle八资产门passed且0 blockers；root回拷evidence后独立再验passed，八asset0..7齐全，evidence SHA256 `daece1e92ce6eeffdad51f0a0b9fc282c271c7226e45fa9ed30934b3fbc4da8d`。本地ignored副本`data/hoi_v1_full_train_native_evidence_20261023_01/`。全数据4096/H32资源报告也passed，全部九个phase完成、optimizer steps0；同步最大used45,086,081,024 / 47,590,277,120bytes，实际全reference与配额均真实驻留。`1459951`的64步补验已正常运行，尚待报告，不能提前宣布全部ready。
- 补审确认正式launcher强制每进程24线程与probe的3线程有CPU超订阅差异（P2，不改变GPU图/物理容量）。已仅修`cluster/hoi_v1_train.sbatch.sh`：尊重显式OMP/MKL，默认3；全部HOI shell语法及默认3/3、显式5/2保留验证通过。候选source10 archive SHA256 `ebbcd611b4c30a500b3730e683f2821d5fa4f1d272c257ed6dbca225cea3fdca`，170文件；112个Python文件与source09全部逐字节一致，除说明外仅训练launcher变化，GPU证据可复用。09及运行中作业未修改。source10传输后的新目录部署命令仍在等待文件系统返回；未确认部署完成前不当ready。ignored `tmp/hoi_v1_final_preflight_20261023.py`仅准备了source10的CPU `--preflight`，需部署确认后执行；该脚本会核所有源文件、09→10 Python一致性及原生报告原字节/输入SHA，无worker/optimizer/正式训练。
- 后续部署命令已成功返回0且远端archive SHA与source10一致；22:39 UTC其CPU `--preflight`已通过。全部170个source10文件哈希核对、与source09的112个Python文件逐字节核对、8个原native报告SHA及内嵌报告内容核对均通过；native gate passed，20k/500/5000、4096/H32、LR decay250000及physics ON/ON无偏离，`training_started=false`且没有worker/prefill/checkpoint。计划路径`..._10/results/first_experiment_preflight_20261023_01/plan.json`，SHA256 `185ded469e03dfd2c2e5bae06a454627db31406f352417b4b9df970a18d45225`。本地ignored `data/hoi_v1_final_readiness_receipts_20261023_01/preflight/`保存plan与verification；同根`resource12/report.json`已回拷完整2651资源报告，确认初始化checkpoint真实SHA为`e2b6e716c058e5448ab26caa22febc874dd50e2459435f1530f515dcd9fe8d9f`、真实worker配额与CPU审计一致、九phase有限梯度，实际GPU无其他job进程。只剩1459951的64步补验待终态，不重复已通过阶段，不启动正式训练。
