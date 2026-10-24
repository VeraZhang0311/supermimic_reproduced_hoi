supermimic + hoi

我目前v1的设想是：

先从最简单的开始，policy 给完整 human motion reference, 物体shape (fps 128x3）， 物体current pose, future pose trajectory，本体。

然后world model的状态预测部分加入 object world position, world rotation, v, omega。

observation head 加入 object current pose。

reward head 加入object相关reward，比如该有的接触，object的tracking

termination head 不知道需不需要把 object 在，lets say 0.32 (16步）内没有发生expected contact，作为一个termination 条件

需要你帮我理清楚：

1. 原本supermimic 是把motion tracking的未来10帧humanoid轨迹 g_t 给到 goal encoder 产出 32D z_t。 现在加入 object 轨迹，这个也融合进g_t吗 / z_t吗 还是作为一个单独的 latent？
2. 世界模型要预测的state里面，要不要包含object的contact？
3. termination head 关于 object 刚我说的termination 条件

ChatGPT Pro 回复：**我觉得这个 v1 的范围很合适：先保留完整 human reference，把问题限定为“通过可微 WM 学会 human–object tracking”，暂时不加入稀疏 goal、prior/posterior 或多模态动作生成。**

对于你的三个问题，我建议：

**Human 和 object reference 共同组成 \(g_t\)，但分支编码后再拼接；contact 先作为 WM 的辅助预测输出，不强制加入递归 state；“连续 16 步没有 expected contact”先作为监控指标，暂时不作为第一版的硬终止条件。**

下面逐个展开。其中最值得注意的是：**物体的 shape 和物理参数也需要进入 WM；而涉及“连续多少步”的 termination，需要额外的计时状态。**

---

## 1. Object trajectory 放进 \(g_t\) 还是单独的 latent？

### 先区分三个角色：目标、当前观测、物体属性

你提供给 policy 的输入，可以整理成：

| 信息                          | 角色                       | 我建议放在哪里              |
| ----------------------------- | -------------------------- | --------------------------- |
| Future human reference        | 希望机器人怎么运动         | Goal encoder                |
| Future object pose trajectory | 希望物体怎么运动           | Goal encoder                |
| Object current pose           | 物体现在在哪里             | Observation 分支            |
| Proprioception/history        | 机器人现在是什么状态       | Observation 分支            |
| Object shape                  | 正在操纵什么几何形状的物体 | Shape encoder，作为 context |

所以在语义上：

\[\boxed{ g_t= \left( g_t^{\mathrm{human}}, g_t^{\mathrm{object}} \right) } \]

**加入 object trajectory 后，\(g_t\) 就从“人体运动目标”扩展成“人和物体的联合运动目标”。** 它不必是一个未经处理的大向量，也不要求只能通过一个 encoder。

原始 SuperMimic 的确使用未来 \(K=10\) 帧 reference，经 motion encoder 得到 32D、tanh-bounded 的 latent，再与 observation history 一起送入 action decoder。[arXiv](https://arxiv.org/pdf/2610.28480)

### 我的 v1 建议：两个 encoder，拼成一个组合 goal latent

具体可以是：

\[ \begin{aligned} z_t^H &= E_H(g_t^{\mathrm{human}}),\\ z_t^O &= E_O(g_t^{\mathrm{object}}),\\ z_t^{\mathrm{goal}} &= [z_t^H,z_t^O]. \end{aligned} \]

然后：

\[ \boxed{ a*t= D*\theta \left( h_t^{\mathrm{proprio}}, e_t^{\mathrm{object\ current}}, e^{\mathrm{shape}}, z_t^{\mathrm{goal}} \right) } \]

其中：

\[ e^{\mathrm{shape}}=E\_{\mathrm{shape}}(P), \qquad P\in\mathbb R^{128\times3}. \]

一个直接的起始配置是：

\[ z_t^H\in\mathbb R^{32}, \qquad z_t^O\in\mathbb R^{32}, \qquad z_t^{\mathrm{goal}}\in\mathbb R^{64}. \]

**这里 object 32D、组合 64D 是我建议的起始超参数，不是论文设定，也不代表最优。**

这样，“分别编码”和“融合成同一个 latent”并不矛盾：

> **先分别提取 human/object 的目标特征，再由 action decoder 的前几层学习它们之间的协调关系。**

第一版甚至不需要额外 fusion Transformer。拼接后接 MLP 就可以。也不需要强制 \(z^H\) 和 \(z^O\) 相等——它们描述的是互补信息，而不是同一个动作的两种等价编码。

### 为什么不把所有东西都塞进原来的 32D？

不是不能这样做，而是我不建议第一版同时改变太多东西。

保持 human encoder 的结构，增加 object-goal 分支，可以更清楚地检查：object trajectory 是否被利用、latent 容量是否成为瓶颈、移除哪个输入会导致失败。

**当前 object pose 也不必经过 goal bottleneck。** 让 decoder 直接获得“物体现在在哪”和“希望物体去哪”，会更容易分析闭环控制有没有真正起作用。

### 两个实现细节很重要

**时间必须对齐。** 第一版我会让 human/object reference 使用相同的未来采样时间戳。完整人体 reference 也应当已经 retarget 到机器人，并与 object trajectory 几何一致；否则可能出现“人体目标要求手在 A，但物体目标要求必须在 B 才能抓住”的冲突。

**坐标变换必须跟着 imagined state 更新。** WM 可以预测 world-frame object state，但 actor 输入可以采用机器人 heading frame 下的相对物体 pose。例如：

\[ p*{O,t}^{\mathrm{rel}} = R*{\mathrm{heading},t}^{\top} (p*{O,t}^{w}-p*{\mathrm{root},t}^{w}). \]

在 imagined rollout 中，这个变换要使用**当前预测的 robot/object state**，不能一直使用 replay 中原轨迹的坐标变换。原始 reference 是固定的外部目标，但它相对于预测机器人的位置会不断变化。

---

## 2. WM 的 state 要不要包含 object contact？

**我的建议是：物体的 pose、velocity 必须加入；contact 建议预测，但第一版不必把它作为下一步 dynamics 的递归输入。**

这里需要区分：

> **“WM 要输出 contact”不等于“contact 必须是 autoregressive state 的一部分”。**

### 2.1 先明确递归预测的物理状态

你提出的扩展是合理的：

\[ \boxed{ x*t= \left[ s_t^{R}, p*{O,t}^{w}, R*{O,t}^{w}, v*{O,t}^{w}, \omega\_{O,t}^{w} \right] } \]

这里我用 \(x_t\) 表示物理状态，后面还会单独讨论 termination 的任务计时状态。

Dynamics 应该同时更新 robot 和 object：

\[ \boxed{ \hat x*{t+1} = F*\omega \left( \hat x*t,a_t,e*{\mathrm{shape}}^{\mathrm{WM}},\xi \right) } \]

**不能只给 WM 增加 object 输出，却忘记给它 object geometry 和物理参数输入。**

对于不同物体，建议让 \(\xi\) 包含物体质量、惯量、质心及相关摩擦参数；shape 则提供尺寸和几何信息。即使形状完全相同，轻箱子和重箱子的动力学也不同。

FPS 点云我会使用 **object-local coordinates，并保留真实尺寸**。如果做尺寸归一化，需要把 scale 另外传进去。WM 和 actor 可以分别使用 shape encoder；第一版不建议共享一个会在 actor update 中变化的 shape encoder，以免“冻结 WM”时其实还改变了它的输入表示。

### 2.2 为什么 contact 不一定是独立 state？

在常见刚体建模中，接触几何和接触力由当前构型、速度、作用力、几何及物理参数共同决定；并不是必须把一个独立的二值 contact flag 加入状态，才能描述接触动力学。MuJoCo 的公式也将约束力写成 \(f(q,v,\tau)\)。不过，具有内部动态的 actuator、延迟或控制器历史，可能需要额外状态。[MuJoCo Documentation](https://mujoco.readthedocs.io/en/stable/computation/index.html)

因此，**不把 contact flag 递归输入，不代表模型忽略接触**。接触的作用仍必须体现在下一步 robot/object 的速度和位姿变化里。

但对于你的任务，仅靠 pose prediction loss，未必容易检查模型是否学对了接触。因此，我建议加一个辅助 contact head。

### 2.3 建议增加一个 pair-specific contact head

对于搬箱子的 v1，可以先关注：

\[ b_t= [ b_t^{L\text{-hand,obj}}, b_t^{R\text{-hand,obj}}, b_t^{obj,ground} ]. \]

这里用 \(b\) 表示 contact，避免与 SuperMimic 的 survival weight \(c_t\) 混淆。

让 contact head 输出概率：

\[ \boxed{ \hat b*{t+1} = C*\omega \left( \hat x*t,a_t,e*{\mathrm{shape}}^{\mathrm{WM}},\xi \right) } \]

它可以和 dynamics 共享 backbone，但有独立输出层。然后用 simulator 中实际记录的接触标签监督：

\[ \mathcal L*{\mathrm{contact}} = \sum_j \operatorname{BCE} \left( \hat b*{t+1}^{j}, b\_{t+1}^{\mathrm{sim},j} \right). \]

**第一版：预测 \(\hat b*{t+1}\)，用来计算 contact reward、辅助训练和诊断；不必再把它拼回 \(\hat x*{t+1}\) 作为下一步 dynamics 输入。**

后续如果发现只靠物理状态预测难以区分接触模式，再测试“contact feature 反馈进 dynamics”是否有效。这应当是一个消融，而不是一开始就认定必须如此。

### 2.4 一定要区分 actual contact 与 expected contact

这是整个设计里很容易混淆的一点：

\[ \underbrace{b*t^{\mathrm{sim}}}*{\text{模拟器实际发生了什么}} \qquad\neq\qquad \underbrace{b*t^{\mathrm{ref}}}*{\text{任务希望发生什么}}. \]

**Contact head 学前者；reward 比较前者与后者。**

不要用 expected contact 去监督 contact head，否则模型可能只是学会输出“应该有接触”，而不是预测“实际上有没有接触”。我也不会把 reference contact 作为这个物理预测 head 的必要输入。

接触标签的定义还要区分“几何上靠近”和“发生了力学接触”。例如 ContactMimic 从 reference 的表面距离提取期望接触，而执行过程的接触奖励使用力阈值判定；两者不是同一个量。[arXiv](https://arxiv.org/html/2607.08742v1)

对你的实现，我建议从 simulator 记录**指定 body–object pair** 的接触，而不是只看手腕的总接触力，否则手碰到地面也可能被误算成抓住物体。

### Observation 和 reward 可以这样安排

**Object observation 不需要再独立预测一遍。** 已经有 \(\hat p_O,\hat R_O\) 后，直接解析计算 actor 所需的相对 object pose，再加上部署设定中的观测噪声即可。SuperMimic 本身也允许 observation/reward 等 head 使用解析计算，不要求都是独立 NN。[arXiv](https://arxiv.org/pdf/2610.28480)

Reward 则可以分工：

\[ \boxed{ \hat r= r*{\mathrm{human}} +w_O r*{\mathrm{object}} +w*A r*{\mathrm{approach}} +w*C r*{\mathrm{contact}} -r\_{\mathrm{regularization}} } \]

我的设计意图是：**靠近物体用连续的距离／palm-to-surface reward；建立和保持接触用 contact reward；真正搬动了没有，用 object pose/velocity tracking 检验。**

不能只优化 contact probability——“模型认为手碰到了箱子”不等于“箱子被稳定地搬起来”。因此接触预测和物体运动预测必须一起在真实 simulator 中验证。

---

## 3. “0.32 秒没有 expected contact”要不要作为 termination？

**这个规则有合理性，但我不建议作为 v1 一开始就启用的硬终止条件。**

ULTRA 的确使用了持续 20 帧 contact mismatch 的 early termination，说明这类设计有先例；但这不能直接证明相同规则适合你的 learned-model BPTT 设置。[arxiv.org](https://arxiv.org/html/2603.03279)

### 3.1 首先，它应该是“该接触的时候，持续没接触”

不能定义成：

> Episode 开始后 16 步内没碰到物体，就终止。

机器人可能仍在接近箱子，也可能正在有意放下、松手。

更合适的是：

> **某个 contact pair 被 reference 标记为应该接触，但连续 \(N\) 个 control steps 没有发生接触，才判定失败。**

例如，定义：

\[ m_t^j = b_t^{\mathrm{ref},j} \left(1-b_t^{\mathrm{sim},j}\right) \]

表示第 \(j\) 个接触对“应该接触，但没有接触”。

然后记录连续失配步数：

\[ \boxed{ n*t^j= \begin{cases} n*{t-1}^j+1,&m_t^j=1,\\ 0,&m_t^j=0. \end{cases} } \]

终止条件为：

\[ d_t^{\mathrm{contact}} = \mathbb 1\left[\max_j n_t^j\geq N\right]. \]

在你采用的 50 Hz 控制频率下，\(N=16\) 对应 \(0.32\) 秒。**这个容忍时间与 imagined rollout horizon \(H\) 是两个独立参数。**

这里也不是在当前时刻预测“未来 16 步都不会接触”并立即结束，而是在实际执行中累计已经发生的失配。

### 3.2 最关键的问题：这个 termination 具有历史依赖

考虑两个 rollout：

| 当前状态                  | 已连续缺失接触的时间 |
| ------------------------- | -------------------- |
| 手距离箱子 2 cm，尚未接触 | 1 步                 |
| 手距离箱子 2 cm，尚未接触 | 15 步                |

它们的当前物理状态可以非常相近，但下一步是否应该终止完全不同。

因此：

\[ \hat d*t=D*\omega(\hat x_t,g_t) \]

**仅凭当前物理状态和 reference，不一定能判断是否满足 16 步规则。**

你需要让终止模型知道计时状态，例如：

\[ \boxed{ \hat d*t= D*\omega(\hat x_t,g_t,\hat n_t) } \]

Critic 也应该知道这类影响未来存活时间的任务状态。

同时，**从 replay buffer 启动 imagined rollout 时，要恢复记录的 \(n_t\)，不能每次都重置成 0。** 否则模型会把“马上要超时失败”的状态误认为还有完整的 0.32 秒机会。

这个 counter 是任务逻辑，不一定要让物理 dynamics MLP 学习。可以单独更新；但 imagined rollout 中若要保留关于接触和 termination 的梯度，还需要设计可微近似，不能直接硬阈值化接触后声称整条路径仍然可微。

**所以，这不是在 termination head 上多加一个输出就结束了，而是增加了一段带记忆的任务状态更新。**

### 3.3 为什么我建议 v1 暂缓它？

我的顾虑主要不是“这个条件不合理”，而是它会同时引入三个新问题。

**探索被截断。** 随机初始化的 policy 可能稍微晚一点才找到接触。如果过早结束，就少了观察“差一点抓住 → 调整 → 成功”的机会。

**接触时序不完全准确。** Retarget 后的手掌几何、参考标注和实际 PD 执行可能存在小偏差。严格追踪接触发生时间，可能把可恢复的延迟当成失败。

**WM 多了一条容易出错的路径。** Contact prediction、计时更新、termination prediction 任意一处有偏差，都可能使 imagined return 与 simulator return 不一致。

这不是说以后不能加，而是第一版需要先回答：**模型是否能预测接触附近的 object motion，并提供有用的 policy gradient？**

### 我建议的 v1 termination

第一版保留人体失稳／严重偏离参考的 termination，再增加**较宽松、明确的 object failure 条件**，例如物体偏离目标过远或离开允许的工作区域。阈值需要根据任务尺度选择。

物体落地也必须结合任务阶段：**搬运阶段掉落可以是失败，放置阶段接触地面则是目标行为。**

与此同时，把“连续缺失 expected contact 的时长”记录成指标，并在 reward 中惩罚失配。等 policy 开始稳定建立接触后，再加入 \(N=16\) 或其他容忍时间的 termination 消融。

---

## 我会采用的 v1 配置

| 模块                   | 建议                                                                                                        |
| ---------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Goal**               | Full human reference＋object future trajectory；分支编码后拼接                                              |
| **Actor observation**  | Proprioception history＋relative object pose/history＋shape feature；不要求真实接触标签作为部署输入         |
| **Dynamics**           | 联合预测 robot/object state；输入 object geometry 和物理参数                                                |
| **Contact prediction** | 新增辅助 contact head，预测实际的 pair-wise contact；暂不递归输入                                           |
| **Observation/reward** | 从预测状态解析构造 object observation、tracking reward；contact reward使用预测接触与 reference contact 比较 |
| **Termination**        | 人体失败＋宽松的 object failure；暂缓严格的 16-step contact timeout，先记录指标                             |

**最核心的划分是：\(g_t\) 描述“希望发生什么”，物理 state 描述“现在是什么”，contact head 描述“实际发生了什么接触”，termination 描述“什么情况值得结束这次尝试”。**

把这四者分清，你这个 v1 就可以在不引入 teacher、CVAE 或复杂 latent 对齐的前提下，先验证最重要的事情：**一个联合 robot–object WM，能不能真正帮助 policy 学会搬运，而不只是让 imagined reward 变高。**

训练数据集：ULTRA retargeted and augmented G1 data `OMOMO_retarget_aug/`，`[T,630`]

然后留出 train/eval/test 划分，可以就采用我们 hoi 的划分(eval80, test 40)
