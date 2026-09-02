---
layout: post
title: "OPD 中 reverse KL 的 in Reward 与 in Loss"
date: 2026-09-03
description: On-Policy Distillation 里的 reverse KL 恰好落在六宫格中"永远对"的 k1-in-Reward 格子；本文接着分析换成 k3、discount 设为 0、以及教师只暴露 top-k logprobs 这几个工程现实分别会踩什么坑。
categories: reinforcement-learning
lang: zh
---

这篇算是[上一篇 KL 笔记](/reinforcement-learning/2026/09/02/kl-divergence-correct-form.html)的姐妹篇。上一篇讲的是 RLHF / GRPO 里"KL 正则化"这个场景：给任务 reward 加一个"别跑太远"的惯罚项。这一篇要讲的是 Thinking Machines 那篇 [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)（下面简称 OPD）——一个表面上很不一样，但骨子里用的是同一套数学的场景：这次 reverse KL 不是"加在任务 reward 上的惯罚项"，它**就是**整个训练信号本身。

正因为两篇用的是同一套底层数学，上一篇里六宫格、k1/k2/k3 的所有结论可以几乎不打折扣地搬过来用——但 OPD 的具体设计（尤其是 discount=0、以及教师只给 top-k logprobs 这个工程现实）又引出几个上一篇没碰到的新问题。这篇就写这几个新问题。

---

## 一、OPD 是什么，它的 reverse KL 长什么样

先把 OPD 的设定说清楚，免得后面的分析没有落脚点。

Thinking Machines 把后训练方法按"采样方式"和"奖励密度"分成三类：

| | 采样 | 奖励密度 |
|---|---|---|
| SFT | off-policy（数据来自别处） | dense（每个 token 都有监督信号） |
| RL | on-policy（数据来自当前策略） | sparse（一条轨迹一个标量奖励） |
| **OPD** | **on-policy** | **dense** |

OPD 想要的是"RL 的 on-policy 采样"和"distillation 的逐 token 密集信号"两头的好处：从学生 $\pi_\theta$ 自己采样出一条轨迹，然后让一个更强的教师 $\pi_{\text{teacher}}$ 给轨迹里**每一个 token** 打分——类似棋类引擎给每一步棋标注"这步是漏着/失误/精彩"，而不是等一整局下完才给一个胜负。

具体的训练信号是**逐 token 的反向 KL**：

$$k_1(y_t) = \log\pi_\theta(y_t|s_t) - \log\pi_{\text{teacher}}(y_t|s_t)$$

（这就是上一篇定义的 $k_1$，只是把"ref"换成了"teacher"。）他们把这个量的负值直接设成每个 token 的 advantage：

```python
reverse_kl = sampled_logprobs - teacher_logprobs   # 就是 k1(y_t)
trajectories["advantages"] = -reverse_kl
# 套用已有 RL 脚本的 importance-sampling loss 完成训练
training_client.forward_backward(trajectories, loss_fn="importance_sampling")
```

选反向 KL（而不是正向 KL，或者别的散度）不是随手选的，博客给了两个理由，值得记一下：

- **mode-seeking，"unhackable"**：反向 KL 只要求学生在自己生成的地方跟老师对齐，不要求覆盖老师所有的模式。这意味着低 KL 总对应老师视角下的高质量行为——不像奖励模型，模型没法通过"钻 KL 估计器的空子"来刷分。
- **per-token、无需等完整 rollout**：算这一项只需要教师做**一次前向传播**（给定学生已经采出的序列，一次 forward 就能拿到每个位置的 logprob），不需要教师自己生成。而且他们特意把 discount factor 设成 **0**——这一点很关键，第三节会展开。

博客里还有一句容易被忽略但对本文很重要的话：他们**没有**在实验里用"logit（top-k）distillation"，只说这是"可以进一步提升算力效率的方向"，并且声称"对序列采样求期望，和直接对老师完整分布做 logit distillation，理论上是同一个目标的无偏估计"。这句话对不对、在什么条件下对，是第四节要拆开讲的。

---

## 二、OPD 的 reverse KL 落在六宫格的哪一格

上一篇笔记最核心的结论是一张六宫格：k1/k2/k3 三个估计器，分别放进"KL in Reward"（IR，梯度只保留得分函数项）和"KL in Loss"（IL，梯度只保留路径导数项）两种位置，梯度是否精确等于真梯度，结论完全不对称。现在把 OPD 的实现对进这张格子里。

### 2.1 OPD 用的就是 k1-in-Reward——而且是六宫格里唯一"永远对"的格子

看 OPD 的代码：`reverse_kl` 是在 `torch.no_grad()` 语境下算出来的一个数值（教师 client 只做推理，不参与反传；学生的 `sampled_logprobs` 也是采样时刻的、被当作快照使用的），然后被塞进 `advantages`，走标准的 RL importance-sampling loss（即用 $\rho_t=\pi_\theta(y_t)/\pi_{\theta_{\text{old}}}(y_t)$ 这个比值乘 advantage，只对 $\rho_t$ 里的 $\theta$ 反传）。

这精确对应上一篇 §4.1 的 **k1-in-Reward**：advantage 里的 KL 值被 stop-grad，只有得分函数项（通过 $\rho_t$）在反传。而 §4.1 证明过一个很强的结论——k1 的路径导数期望恒为 0，所以 **k1-in-Reward 的梯度和"完整、正确"的反向 KL 梯度精确相等，不是近似**：

$$\nabla_\theta \mathbb{E}_{y\sim\pi_\theta}[k_1(y)]\Big|_{\text{完整}} = \mathbb{E}_{y\sim\pi_\theta}\big[k_1(y)\cdot\nabla_\theta\log\pi_\theta(y)\big]\Big|_{\text{k1-in-Reward}}$$

也就是说，OPD 的 reward 单独存在的话（把它想象成"$r(x,y)=0$、只有 KL 惩罚"的特例代入上一篇的框架），**在梯度层面精确等于对 $D_{\text{KL}}(\pi_\theta\|\pi_{\text{teacher}})$ 做梯度下降**，没有任何近似、没有任何因为"用 RL 脚手架实现"而引入的偏差。OPD 的作者在博客里说"我们的实现相对于已有的、带 KL 正则化的 RL 实现只是一行代码的改动"——现在可以更精确地说：**这一行代码改动，恰好把训练目标从"任务 reward 减去 k1 惩罚"换成了"纯粹的 k1 惩罚"，而两种写法在梯度精确性上是同一个量级的——都精确无偏，因为改动前后都停在了六宫格里同一格。**

这不只是走运。选反向 KL、走 RL 的 advantage/重要性采样通道，这两个设计决策合在一起，正好落在了上一篇六宫格里"永远对"的那一格。

### 2.2 如果图省事把它写成 loss，会精确复现"梯度恒为 0"的陷阱

现在设想一个很容易冒出来的"简化"想法：既然目标就是让学生逼近老师，何必绕一圈套 RL 的 advantage/importance-sampling 脚手架？直接把逐 token 的 $k_1$ 当成一个 loss，对着它反传不就完了？

```python
# 一个看起来更直接、但是错的写法
y = sample_from(student, no_grad=True)          # 采样本身不可微，天然 stop-grad
loss = (student.logprob(y) - teacher.logprob(y)).mean()
loss.backward()
```

这正是上一篇 §4.2 的 **k1-in-Loss**：采样分布的 $\theta$ 被 stop-grad（因为采样这一步本来就不可微），但 log-prob 项里的 $\theta$ 保留梯度。而 §4.2 证明的结论是致命的：

$$\nabla_\theta\,\mathbb{E}_{y\sim\pi_\theta}\big[\log\pi_\theta(y)-\log\pi_{\text{teacher}}(y)\big]\Big|_{\text{k1-in-Loss}} = \mathbb{E}_{y\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(y)\big] = 0$$

**这个 loss 的梯度期望恒为 0。** 训练看起来在跑（loss 数值会动，因为它记录的是当前采样批次的偶然波动），但从统计意义上讲，**这个 loss 不会把学生往老师那边推一寸**。这是本文想强调的第一条、也最容易在"重新实现一遍 OPD、觉得 RL 脚手架太重"时踩的坑：**OPD 的正确性不是来自"用了反向 KL"，而是来自"反向 KL 被放对了位置"。把同一个量从 reward 挪到 loss，看起来只是搬了个地方，梯度性质却从"精确无偏"变成"精确的零"。**

（如果你想知道"为什么不干脆用交叉熵蒸馏 loss（即标准的、off-policy 的 logit distillation，教师给分布、学生对着算 cross entropy）"——那是另一件事，那个 loss 确实有效，但它对应的是**正向 KL**、**off-policy 采样**，训练动态和 OPD 完全不同，不在本文讨论范围内。本文只讨论"想要 on-policy 采样 + reverse KL"这个组合下，reward 和 loss 两种放法的区别。）

### 2.3 一个更隐蔽的冲动：为了降方差换成 k3-in-Reward——这是 OPD 里最不该做的事

$k_1$ 的缺点在上一篇讲过：单样本方差极大，尤其是分布越接近、信噪比越差。OPD 训练到后期，学生已经很接近老师了，这时候 $k_1$ 的方差问题会很扎眼——直觉上很容易想到"换成 $k_3$，无偏、非负、方差小三个都要"。

**这恰恰是 OPD 场景下最不该做的替换。** 上一篇 §4.4 已经证明并用数值验证过：

$$\nabla_\theta\big[\text{k3-in-Reward}\big] = \underbrace{\nabla_\theta D_{\text{KL}}(\pi_\theta\|\pi_{\text{teacher}})}_{\text{反向 KL 梯度，我们想要的}} - \underbrace{\nabla_\theta D_{\text{KL}}(\pi_{\text{teacher}}\|\pi_\theta)}_{\text{正向 KL 梯度，混进来的}}$$

两项在 $\pi_\theta\approx\pi_{\text{teacher}}$ 时到二阶为止相等，**一阶抵消**。上一篇的数值表（换成这里的记号）：

| 学生 vs 老师的差距 | 真梯度模长 | k3-in-Reward 梯度模长 | 剩余比例 |
|---|---|---|---|
| 大 | 1.76e-1 | 3.04e-2 | 17% |
| 小 | 1.52e-2 | 7.54e-4 | 5.0% |
| 很小 | 3.03e-2 | 2.25e-4 | **0.74%** |

问题在于：**OPD 的整个训练过程就是在把"学生 vs 老师的差距"从大推向很小**——这正是 k3-in-Reward 梯度塌陷最严重的方向。用 k3 当 OPD 的 reward，train 到后期（也就是最需要精细打磨的阶段），你会发现梯度信号几乎完全消失，而且不会有任何报错——loss/KL 监控数值看起来一切正常（k3 的值本身还是低方差、非负的），只是模型不再往老师那边靠近了。这比"方差大但至少方向对"的 $k_1$ 危险得多。

**结论：OPD 场景下 $k_1$-in-Reward 的高方差是必须忍受的代价，不能用 $k_3$ 抄近路。** 如果真的要降方差，方向应该是标准的 RL 方差缩减手段（group baseline、per-prompt 归一化、多样本平均），而不是换估计器本身。

---

## 三、Discount = 0：故意不要 Part 2

上一篇 §5 有一个重要发现：把 KL 惩罚折进 reward、走标准 policy gradient 的 reward-to-go，会**自动**把"当前 token 对后续所有 token KL 的级联影响"（Part 2）计入当前 token 的梯度——这被认为是 KL-in-Reward 的一个结构性优势，因为大多数 KL-in-Loss 的实现都漏掉了这一项。

OPD 明确反着来：博客里说"我们没发现 discount factor > 0 能提升效果，所以选择了 0"。Discount = 0 意味着 $t$ 位置的 return 就是 $r_t$ 本身，**不会**累加任何未来 token 的信号——**Part 2 被主动关掉了**。

这不是疏忽，是设计选择，而且是有道理的：

- **KL 正则化想要 Part 2**，因为它关心的是"整条轨迹"离参考模型有多远，当前 token 选择会把后续生成引到什么样的分布区域去，这个级联效应正是你想控制的东西。
- **OPD 不想要 Part 2**，因为它的比喻是"棋步评分"：每一步的好坏应该独立打分，不应该因为后面几步走飞了就往前面这步身上倒扣分（也不应该后面走好了就给前面这步加分）。这是标准的 credit assignment 诉求——你想知道的是"这一步本身选得好不好"，而不是"这一步选完之后整条轨迹变成什么样"。开启 Part 2 会让每个 token 的训练信号变成"自己的 KL + 一堆跟自己没有直接因果关系、只是恰好排在后面的 token 的 KL"，这会显著拉低信噪比，也会让 credit assignment 变得模糊。

一句话总结这个对比：**同一个"reward-to-go 自动带不带 Part 2"的性质，在 KL 正则化里是优点，在 OPD 里如果不设 $\gamma=0$ 就会变成缺点。** 这也是个提醒：六宫格的结论是"哪个格子梯度精确"，但"你到底想不想要 Part 2 那个级联项"是另一个独立的设计选择，取决于你的训练信号本身想表达什么语义，不能照抄。

---

## 四、Top-k 实现要注意的地方

这一节是本文的第二个主题，也是纯工程但很容易被忽略的一块。

### 4.1 k1（以及 k3）天然只需要"一个点"，不需要整条分布

先说一个好消息。仔细看 $k_1(y_t) = \log\pi_\theta(y_t) - \log\pi_{\text{teacher}}(y_t)$——它只涉及**采样到的那一个 token** 在两个分布下的概率，不涉及对整条词表求和。这跟"logit distillation"（也叫 dense KD，在每个位置对**整条**教师分布做加权，比如 $\sum_v \pi_{\text{teacher}}(v)\log\frac{\pi_{\text{teacher}}(v)}{\pi_\theta(v)}$）是本质不同的计算模式：

- **k1/k3 式（OPD 的做法）**：每个位置只需要教师给出"采样到的那个 token"的一个 logprob 数值。
- **logit distillation 式**：每个位置需要教师给出整条词表的分布（或者至少覆盖大部分概率质量的一个子集）。

这个区别决定了两者对"教师只暴露 top-k logprobs"（很多推理服务只返回 top-k，出于带宽或接口限制）这件事的敏感程度完全不同。**k1/k3 式的方法天然对 top-k 更宽容**：只要采样到的 token 落在教师返回的 top-k 列表里，你就拿到了精确的 $\pi_{\text{teacher}}(y_t)$，跟教师给不给你剩下几万个 token 的概率完全无关。

（这也是 Thinking Machines 那句"序列采样是老师完整分布的无偏估计，理论上是同一个目标"的准确含义：把 $k_1(y_t)$ 对 $y\sim\pi_\theta$ 求期望，恰好就等于 logit distillation 在该位置算出的那个精确的逐位置反向 KL 值——见上一篇 §1 对 k1 无偏性的证明，这里只是换了个"ref"的名字。序列采样版本方差更大，但期望上是同一个目标，这个等价关系没有问题。）

### 4.2 但"宽容"是有代价的：学生犯大错的时候恰好最容易缺数据

问题出在 top-k 之外。学生训练早期、或者某次采样运气不好，完全可能采到一个**教师认为极不可能**的 token——而"教师认为极不可能"恰恰意味着这个 token 大概率不在教师返回的 top-k 列表里。**这是最需要精确惯罚的时刻，却也是信息最缺失的时刻。**

具体会有多缺失？做个数值实验：词表 2000，教师分布稍微peaky（前几个 token 概率明显更高），只暴露 top-20：

```
教师 top-20 概率质量占比: 0.4270（尾部质量 0.5730——真实场景通常比这更集中，这里故意选了个分布更平的例子来放大问题）
```

**在 top-k 内部**，如果用"除以 top-k 概率总和"做归一化（一个很自然的、让概率重新加起来等于 1 的实现方式），会给**每一个** in-top-k token 的 log-prob 加上完全相同的常数偏移：

```
第 1 名 token：真实 log p=-1.9111   renorm 后=-1.0601   偏移 = +0.851
第 5 名 token：真实 log p=-4.3041   renorm 后=-3.4531   偏移 = +0.851
第20名 token：真实 log p=-5.7293   renorm 后=-4.8783   偏移 = +0.851
```

偏移量恒等于 $-\log(\text{top-k 概率质量})$，跟具体是哪个 token 无关——这一点其实还好，因为它是个**常数**：如果学生几乎所有采样都落在 top-k 内部，这个常数偏移只是把每个 token 的 $k_1$ 整体加了个偏置，相当于给这条轨迹加了个常数 baseline，advantage 里的相对大小关系没变，对训练方向影响有限（虽然会让 KL 监控数值系统性偏低，不能直接拿去做绝对量级的判断）。

**真正麻烦的是 top-k 之外。** 常见的偷懒实现是：查不到就退化成"clip 到第 k 名的概率"或者"把尾部质量平摊到剩余词表"。数值上看这两种退化会怎样低估惩罚：

```
一个"学生犯大错"的例子：采到全词表里概率倒数第一的 token
  真实 log p = -12.7843
  若 clip 成第20名的概率(log p=-5.7293)：k1 被低估了 7.06 nat
```

**低估 7 个 nat 意味着什么？** 意味着这个本该被重罚的 token，它对应的负 $k_1$（也就是 advantage）会比真实值小了 $e^7\approx1096$ 倍量级的惩罚力度（在 log 空间差 7，概率空间就是三个数量级）。也就是说：**学生犯的错越离谱，你手上的估计器就越可能失效，而且失效的方向永远是"往小了估"**——这是一个系统性的、方向确定的偏差,不是随机噪声,标准的方差缩减手段（多采样平均）救不了它,因为每次采样只要落在 top-k 之外都会遇到同样方向的低估。

### 4.3 三种 fallback 策略，选哪个

汇总一下工程上常见的三种处理方式，以及各自的问题：

| 策略 | 做法 | 问题 |
|---|---|---|
| 当成概率 0 | $\log\pi_{\text{teacher}}(y_t)=-\infty$ | $k_1\to+\infty$，advantage 炸穿，训练直接 NaN。**不能用。** |
| Clip 到第 k 名 | $\log\pi_{\text{teacher}}(y_t)\approx\log\pi_{\text{teacher}}(\text{top-}k\text{ 里最小的})$ | 数值稳定，但系统性低估惩罚，且犯错越大低估越狠（上面的例子）。 |
| 尾部质量平摊 | $\log\pi_{\text{teacher}}(y_t)\approx\log\big(\text{tail\_mass}/(V-k)\big)$ | 比 clip 更接近"假设尾部均匀"这个（通常错误的）假设，实践中跟 clip 量级差不多，一样是低估，只是低估程度依赖 $V$ 的估计是否准确。 |
| **额外单独查询该 token 的精确 logprob** | 发现采样 token 不在 top-k 里时，单独再问教师要这一个 token 的精确 logprob（大多数推理服务的 logprobs 接口支持"给定 token 查其 logprob"，不需要重新做完整的 top-k 排序） | 需要多一次（通常很便宜的）教师查询，但是唯一不引入系统性偏差的方案。**只要接口支持，应该优先选这个。** |

如果接口完全不支持"查指定 token 的 logprob"（只能拿 top-k），退而求其次要在 clip 和尾部平摊里选一个时，倾向于选**尾部平摊**并且把 $V$（有效词表大小，可以用分词器词表减去明显不会被采样到的特殊 token）估得保守一点——这样至少低估的量级是可控、可解释的，而不是像 clip 那样，"低估多少"完全取决于第 k 名概率恰好多大，跟真实的犯错程度没有关系。

### 4.4 如果做的是 logit distillation（dense loss），偏差机制是另一种

上面讨论的都是"OPD 式、逐点采样"的场景。如果你选择了 Thinking Machines 提到的另一条路——直接做 logit distillation，在每个位置对教师的**整条**分布求和，那 top-k 截断造成的偏差是另一种性质：**不是"某个点的数据缺失"，而是"整条分布被系统性削尖"。**

具体说：如果只用 top-k 内的 token 计算 $\sum_{v\in\text{top-}k}\pi_{\text{teacher}}(v)\log\frac{\pi_{\text{teacher}}(v)}{\pi_\theta(v)}$（把尾部直接丢弃，不做任何补偿），相当于在做一个"条件在 top-k 上"的近似 KL，丢弃的尾部概率质量越大，这个近似跟真实反向 KL 的差距越大——而且这个偏差是**在所有位置上都存在的系统性偏差**，不像 4.2 里的"只在学生犯大错时才现形"。这种偏差通常还需要配合 §2 的"in reward / in loss"选择一起看：如果这个截断后的量被拿去做 reward（k1-in-Reward 式的 IS），路径导数为零的那个"免疫"性质（§2.1 的核心论证）并不依赖于"用的是精确 KL 还是截断近似"——只要这个量本身不含 $\theta$ 的路径导数（即：教师分布和截断集合的选取都跟 $\theta$ 无关，只由采样到的 $y$ 决定），stop-grad 免疫这条结论依然成立；但截断带来的**数值上的系统性偏差**（把 reward 的绝对量级估歪）不会因为"放对了位置"而消失，这是两个独立的问题，不要混为一谈。

### 4.5 把 in-Reward / in-Loss 搬到 top-k 场景下重新推一遍

前面几节讨论的是"单点采样时教师给不给你那个具体 token 的 logprob"这个数据缺失问题。这里换一个角度：如果不满足于单点采样（想降低方差），而是把更新**摊到整个候选集合 $K$ 上**（$K$ 可以是任意一种 top-k，先不纠结是谁的 top-k），会发现 §2 那套"in Reward 精确、in Loss 会出问题"的故事在这里重演了一遍，而且能推出一个更精确的定量关系。

先把两种写法在这个"摊到整个 $K$"的设定下重新定义。记 $K$ 上学生分配的总概率质量为

$$Z_\theta(K) = \sum_{v\in K}\pi_\theta(v)$$

**in-Loss（截断求和，直接反传）**：把反向 KL 截断到 $K$ 上，直接对这个截断和求梯度：

$$\hat D_K^{\text{Loss}} = \sum_{v\in K}\pi_\theta(v)\,k_1(v)$$

**in-Reward/Adv（每个候选 token 单独算 ratio，advantage 在 $K$ 内部重新归一化、并 stop-grad）**：先把 $K$ 内的学生概率重新归一化成一个"局部分布" $\pi_\theta^K(v)=\pi_\theta(v)/Z_\theta(K)$，用它加权 $k_1$ 作为（stop-grad 的）advantage，再走标准的重要性采样 PG loss。

按 §2.1 同样的套路（advantage 停梯度，只有 ratio 那条路径反传；on-policy 时 ratio 的梯度化简为 $\nabla_\theta\log\pi_\theta(v)$），这个写法的梯度是纯得分函数项：

$$\nabla_\theta\mathcal{L}_K^{\text{Adv}} = \sum_{v\in K}\pi_\theta^K(v)\,k_1(v)\,\nabla_\theta\log\pi_\theta(v)$$

再把 $\hat D_K^{\text{Loss}}$ 用乘积法则展开（$\nabla_\theta\pi_\theta(v)=\pi_\theta(v)\nabla_\theta\log\pi_\theta(v)$，教师项不含 $\theta$）：

$$\nabla_\theta \hat D_K^{\text{Loss}} = \underbrace{\sum_{v\in K}\pi_\theta(v)\,k_1(v)\,\nabla_\theta\log\pi_\theta(v)}_{\text{得分函数项}} + \underbrace{\nabla_\theta Z_\theta(K)}_{\text{路径导数项}}$$

对比这两个式子，把第一个代进去，得到两者的精确关系（我做过数值验证，用有限差分核对，残差在 $10^{-10}$ 量级）：

$$\boxed{\nabla_\theta \hat D_K^{\text{Loss}} = Z_\theta(K)\cdot\nabla_\theta\mathcal{L}_K^{\text{Adv}} + \nabla_\theta Z_\theta(K)}$$

这里有个容易被忽略的细节：两者的差距**不只是**多出一个 $\nabla_\theta Z_\theta(K)$ 项，共享的那部分得分函数项之间还差了一个 $Z_\theta(K)$ 的**尺度因子**——因为 in-Reward 用的是 $K$ 内部重新归一化过的概率 $\pi_\theta^K(v)$，in-Loss 用的是原始的 $\pi_\theta(v)$，两者天然差 $Z_\theta(K)$ 倍。

**全词表**（$K=\mathcal V$）时，$Z_\theta(\mathcal V)\equiv1$ 恒成立（softmax 归一化的定义），于是 $\nabla_\theta Z_\theta=0$ 且尺度因子也等于 1——两个差异同时消失，两种写法梯度精确相等。这其实是 §2.1"k1 路径导数期望为 0"这条结论在"对 $K$ 精确求和"而不是"对 $y$ 采样求期望"这个语境下的另一种呈现，殊途同归。

**Top-k**（$K$ 是严格子集）时，$Z_\theta(K)<1$ 且随 $\theta$ 变化，两个差异项都不为 0——两种写法不再等价。

**这个差异往哪个方向推？** 把 $\hat D_K^{\text{Loss}}$ 当成要最小化的目标（让学生逼近老师），梯度下降沿 $-\nabla_\theta\hat D_K^{\text{Loss}}$ 走，其中 $+\nabla_\theta Z_\theta(K)$ 这一项在优化里的效果是**压低 $Z_\theta(K)$**——也就是把学生的概率质量推出 $K$ 之外。这不是猜测，是可以直接算出来看到的一条"作弊捷径"：$\hat D_K^{\text{Loss}}$ 是按 $\pi_\theta(v)$ 加权、只在 $K$ 上求和的，$K$ 外的 token 完全不参与这个求和，所以只要把概率往 $K$ 外面搬，$K$ 内每个 $\pi_\theta(v)$ 都会跟着变小，求和值跟着下降——**不需要学生真的更像老师，loss 数值就能降**。我做了个数值实验验证这条捷径确实存在：人为把 $K$ 内 token 的 logit 统一调低一点（概率质量往 $K$ 外挪，$Z_\theta(K)$ 从 0.863 降到 0.793），$\hat D_K^{\text{Loss}}$ 从 0.231 直接掉到 0.145——纯粹靠"藏概率"，跟"逼近老师"这个目标毫无关系。

而 in-Reward 的写法因为用的是**归一化**过的 $\pi_\theta^K(v)$，$K$ 内部的"相对占比"和 $Z_\theta(K)$ 这个"总闸门"被解耦了——advantage 只关心 $K$ 内部谁该多谁该少，不关心 $K$ 整体应该开多大，所以这条捷径天然被堵死。这跟 §2 六宫格的结论完全对得上：**in-Loss 天然容易在意想不到的地方开条缝让优化钻空子，in-Reward（因为 advantage 被 stop-grad）反而更稳。**

这也把这一节和之前讨论过的"反向 KL 的稠密截断该按谁的 top-k 做"接了起来：如果 $K$ 选的是教师的 top-k，而学生自己的高概率区域跟教师重叠不多（训练早期的典型情况），$Z_\theta(K)$ 本来就会偏小——这时候如果又叠加 in-Loss 写法，"把概率推向 $K$ 外"这条捷径会格外好走，因为学生本来就有大量概率在 $K$ 外面。**选错 $K$（该用学生的 top-k 却用了教师的）和选错放置位置（该用 in-Reward 却用了 in-Loss），这两个坑叠在一起是最危险的组合。**

最后把 OPD 实际的单点采样方式也安放进这个框架：它相当于 $|K|=1$、$K=\{y_t\}$（只含采样到的那一个 token）、不做归一化、只走 Adv 通道的极限情形——此时 $Z_\theta(K)=\pi_\theta(y_t)$，"in-Loss 与 in-Reward 不等价"这条结论依然成立，但因为 OPD 本来就走 Adv 通道（第二节已经证明这是精确无偏的格子），这个坑天然被绕开了。

---

## 五、落地清单

把这篇的结论收成几条可执行的检查项：

1. **确认你的 reverse KL 走的是 reward 通道，不是 loss 通道。** 如果你想脱离现成的 RL 脚手架自己实现 OPD，一定要保证采样、以及塞进 advantage 的 KL 值都是 `no_grad` 的，只让重要性采样比值 $\rho_t$ 里的 $\theta$ 反传。直接对 log-ratio 反传当 loss，梯度期望恒为 0（§2.2）。
2. **别在 OPD 里为了降方差把 k1 换成 k3。** OPD 的目标就是把学生逼近老师，这正是 k3-in-Reward 梯度塌陷最严重的区间（§2.3）。方差问题用标准 RL 的方差缩减手段解决，不要换估计器。
3. **Discount factor 按你要不要 Part 2 级联效应来定，不要照抄别的场景的默认值。** KL 正则化想要级联（reward-to-go 自动带上），OPD 不想要（$\gamma=0$，独立评估每一步）——想清楚你的训练信号该有的语义再定这个超参（§3）。
4. **教师只暴露 top-k logprobs 时，优先为落在 top-k 之外的采样 token 单独查询精确 logprob。** 如果接口不支持，用尾部质量平摊而不是 clip 到第 k 名——后者的低估程度和真实犯错程度成反比，越离谱的错误越被轻判（§4.2、§4.3）。
5. **top-k 内部的 renormalize 偏移是个常数，通常不影响训练方向，但会让你的 KL 监控数值系统性偏低——不要拿它去做跨 run、跨 $k$ 值的绝对量级对比。**
6. **逐点采样式 reverse KL 和 logit distillation 是两种不同的偏差机制**：前者是"个别点缺失数据"，后者是"整条分布被系统性削尖"——排查问题时先分清楚你用的是哪一种，别用错误的模型去猜错误的成因（§4.4）。

---

## 附录：数值验证代码

第四节的 top-k 数值实验：

```python
import numpy as np
np.random.seed(0)

V, K = 2000, 20
logits = np.random.randn(V) * 1.3
logits[:5] += 4.0
p_teacher = np.exp(logits - logits.max())
p_teacher /= p_teacher.sum()

order = np.argsort(-p_teacher)
topk_idx = order[:K]
topk_mass = p_teacher[topk_idx].sum()
p_topk_renorm = p_teacher[topk_idx] / topk_mass   # renormalize 到 top-k 内部

# top-k 内部：renorm 偏移是常数 -log(topk_mass)
for rank in [0, 4, 9, 19]:
    tok = topk_idx[rank]
    true_lp, renorm_lp = np.log(p_teacher[tok]), np.log(p_topk_renorm[rank])
    print(rank+1, true_lp, renorm_lp, renorm_lp - true_lp)   # 恒等于 -log(topk_mass)

# top-k 外部：clip 到第 k 名，低估程度随真实概率越低而越严重
worst = order[-1]
print(np.log(p_teacher[worst]) - np.log(p_teacher[topk_idx[-1]]))   # 越负，低估越狠
```

运行结果见第四节正文。核心数字：top-k 内部的偏移恒为常数（$-\log(\text{top-k 质量占比})$，这里是 $+0.851$ nat）；top-k 外部用 clip 兜底时，学生犯的错越大，低估越狠——例子里差了 7 个 nat，对应概率空间三个数量级。

第 4.5 节"$Z_\theta(K)$ 尺度因子"恒等式的验证：

```python
import numpy as np
np.random.seed(7)
V = 30
def softmax(z):
    z = z - z.max(); e = np.exp(z); return e/e.sum()

z_teacher = np.random.randn(V)*1.5
z_theta = z_teacher + 0.6*np.random.randn(V)
pi_teacher = softmax(z_teacher)
K = np.argsort(-softmax(z_theta))[:8]   # 任选一种 top-k，结论跟 K 是谁的 top-k 无关

def k1_of(z, K):
    p = softmax(z)
    return np.log(p[K]) - np.log(pi_teacher[K])

def L_loss(z, K):                       # \hat D_K^Loss：截断求和，不归一化
    p = softmax(z)
    return (p[K] * k1_of(z, K)).sum()

def Z_of(z, K):
    return softmax(z)[K].sum()

def grad_Ladv_analytic(z, K):           # 纯得分函数项：sum piK(v) * k1(v) * grad log pi(v)
    p = softmax(z)
    Z = p[K].sum()
    piK = p[K] / Z
    k1v = k1_of(z, K)
    G = np.eye(V) - p[None, :]          # G[v] = grad_z log pi(v)
    return sum(piK[i]*k1v[i]*G[v] for i, v in enumerate(K))

h = 1e-6
grad_Lloss_numeric = np.array([
    (L_loss(z_theta + h*e, K) - L_loss(z_theta - h*e, K)) / (2*h)
    for e in np.eye(V)])
grad_Z_numeric = np.array([
    (Z_of(z_theta + h*e, K) - Z_of(z_theta - h*e, K)) / (2*h)
    for e in np.eye(V)])

Z = Z_of(z_theta, K)
grad_Ladv = grad_Ladv_analytic(z_theta, K)

print("||grad_Lloss - (Z*grad_Ladv + grad_Z)|| =",
      np.linalg.norm(grad_Lloss_numeric - (Z*grad_Ladv + grad_Z_numeric)))   # ~1e-10，恒等式成立
print("||grad_Lloss - (grad_Ladv + grad_Z)||   =",
      np.linalg.norm(grad_Lloss_numeric - (grad_Ladv + grad_Z_numeric)))    # ~0.036，没有 Z 尺度因子的版本不成立

# loss-hack 方向验证：把 K 内 logits 统一调低，概率质量往 K 外挪，L_loss 是否会无关地下降
z_pushed = z_theta.copy(); z_pushed[K] -= 0.5
print("Z(K) 从", Z, "变成", Z_of(z_pushed, K))
print("L_loss  从", L_loss(z_theta, K), "变成", L_loss(z_pushed, K))
```

数值结果：带 $Z_\theta(K)$ 尺度因子的关系式残差 $\sim3\times10^{-10}$（精确恒等式）；不带尺度因子的简化版本残差 $\approx0.036$（不成立，说明这个尺度因子是必要的，不能省略）。loss-hack 方向验证：$Z_\theta(K)$ 从 0.863 降到 0.793 时，$\hat D_K^{\text{Loss}}$ 从 0.231 降到 0.145——纯粹靠往 $K$ 外藏概率，loss 数值就会下降。

## 参考

1. [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) —— Thinking Machines Lab，OPD 的原始博客，本文第一、三节的事实性描述均来自这里。
2. [KL散度的正确形式是什么](/reinforcement-learning/2026/09/02/kl-divergence-correct-form.html) —— 本文的姊妹篇，六宫格、k1/k2/k3 的无偏性/方差分析、in-Reward/in-Loss 的梯度证明全部来自这篇，本文第二节全程复用其结论。
3. [OPD 实现细节拆解：KL 放在 Loss，还是放在 Advantage？](https://zhuanlan.zhihu.com/p/2068017517788927592) —— 第 4.5 节 top-k 场景下 in-Loss/in-Adv 梯度对比的问题framing来自这篇；本文用自己的符号独立重新推导了梯度关系式，并发现共享项之间还有一个 $Z_\theta(K)$ 尺度因子（原文的简化表述没有包含这一项，数值验证见附录），同时补充了 loss-hack 机制的数值验证。
