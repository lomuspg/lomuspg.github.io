# 强化学习的三条主线： 价值函数 V、Q 和策略 π 的训练

强化学习的核心问题是：**智能体如何通过与环境交互，学习一个能够最大化长期累计奖励的决策策略？**

解决这一问题，可以从三个不同的对象入手：

```math
\boxed{
\begin{aligned}
V^\pi(s)&=\mathbb E_\pi[G_t\mid S_t=s]\\
Q^\pi(s,a)&=\mathbb E_\pi[G_t\mid S_t=s,A_t=a]\\
\pi(a\mid s)&=P(A_t=a\mid S_t=s)
\end{aligned}}
```

其中：

- $`V(s)`$：状态价值，衡量一个状态有多好。
- $`Q(s,a)`$：动作价值，衡量在某状态下执行某动作有多好。
- $`\pi(a\mid s)`$：策略，决定在某状态下采取什么动作。

三者对应三条训练路径：

1. **训练 $`V`$**：如何准确评价状态？从 MC 到 TD。
2. **训练 $`Q`$**：如何根据动作价值选择最优动作？从 SARSA、Q-learning 到 DQN。
3. **训练 $`\pi`$**：能否直接优化决策策略？从 REINFORCE 到 Actor-Critic、PPO、DDPG。

现代强化学习通常通过 $`V`$ 或 $`Q`$ 评价策略，再利用评价结果改进 $`\pi`$。

---

## 一、V 的训练：如何判断一个状态的价值？

### 1. 从 Bellman 方程开始

考虑一个马尔可夫决策过程（MDP），智能体在状态 $`s_t`$ 下选择动作 $`a_t`$，环境给出奖励 $`r_t`$ 并转移到 $`s_{t+1}`$。

定义折扣回报：

```math
G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k}
```

对于给定策略 $`\pi`$，状态价值满足 Bellman 期望方程：

```math
V^\pi(s)=
\mathbb E_\pi[
r_t+\gamma V^\pi(s_{t+1})
\mid s_t=s]
```

这个方程提供了一个重要思路：

**不必直接预测遥远未来的全部奖励，可以把当前状态的价值拆解为即时奖励与下一状态的价值。**

问题是：如何实际计算 $`V`$？

### 2. Dynamic Programming：有环境模型时直接求解

如果知道环境转移概率 $`P(s'|s,a)`$ 和奖励模型，可以迭代：

```math
V_{k+1}(s)=
\sum_a\pi(a|s)\sum_{s'}P(s'|s,a)
[r(s,a,s')+\gamma V_k(s')]
```

这就是策略评估中的动态规划（DP）。

**局限：需要已知环境模型。** 在真实环境中，状态转移规律往往未知，甚至无法显式表示。

因此需要直接从交互数据中学习。

### 3. Monte Carlo：用真实回报监督 V

MC 不需要环境模型，而是在一条轨迹结束后计算实际回报：

```math
G_t=r_t+\gamma r_{t+1}+\cdots
```

并更新：

```math
V(s_t)\leftarrow V(s_t)+
\alpha[G_t-V(s_t)]
```

如果使用神经网络，可以最小化：

```math
L_V(\phi)=
\mathbb E[(V_\phi(s_t)-G_t)^2]
```

**问题：必须等到轨迹结束才能获得完整回报，且回报方差可能很大。**

对于回合很长的任务，这种训练方式效率较低。

### 4. Temporal Difference：不等待最终结果

TD 利用 Bellman 方程，将真实回报替换为一步预测目标：

```math
y_t=r_t+\gamma V(s_{t+1})
```

定义 TD Error：

```math
\delta_t=
r_t+\gamma V(s_{t+1})-V(s_t)
```

TD(0) 的更新为：

```math
\boxed{
V(s_t)\leftarrow V(s_t)+\alpha\delta_t
}
```

这种方法称为 **Bootstrapping（自举）**：用当前的价值估计更新自身。

与 MC 相比，TD 可以在每一步交互后更新，不需要等待完整回合。

| 方法 | 更新目标 | 是否需要环境模型 | 主要特点 |
|---|---|---|---|
| DP | Bellman 期望 | 需要 | 利用已知模型进行规划 |
| MC | 完整真实回报 $`G_t`$ | 不需要 | 无自举偏差，但方差较大 |
| TD(0) | $`r+\gamma V(s')`$ | 不需要 | 能在线更新，使用自举估计 |

TD 的目标依赖当前的价值估计，因此会引入自举误差，也使函数逼近下的稳定性问题更加重要。

**到这里，我们已经能够评价状态，但仍然不能直接回答一个问题：在当前状态下应该选择哪个动作？**

因为 $`V(s)`$ 只评价状态，不直接比较各动作。在没有环境模型的情况下，需要进一步学习动作价值函数 $`Q(s,a)`$。

---

## 二、Q 的训练：从动作价值到最优决策

### 1. 从 V 到 Q：为什么需要动作价值？

对于给定策略：

```math
Q^\pi(s,a)=
\mathbb E_\pi[
r_t+\gamma V^\pi(s_{t+1})
\mid s_t=s,a_t=a]
```

二者满足：

```math
V^\pi(s)=\sum_a\pi(a|s)Q^\pi(s,a)
```

如果已经得到最优动作价值函数 $`Q^*`$，就能直接构造最优策略：

```math
\boxed{\pi^*(s)\in\arg\max_a Q^*(s,a)}
```

因此，训练 Q 不只是为了评价动作，还可以间接得到策略。

### 2. SARSA：学习当前策略的 Q

SARSA 使用真实执行的下一动作构造 TD 目标：

```math
\boxed{
Q(s_t,a_t)\leftarrow Q(s_t,a_t)
+\alpha[
r_t+\gamma Q(s_{t+1},a_{t+1})
-Q(s_t,a_t)]
}
```

其名字来自更新所用的五元组：

```math
(S_t,A_t,R_t,S_{t+1},A_{t+1})
```

SARSA 是 **On-policy** 方法：采样动作与更新目标都依赖当前执行的策略。

例如使用 $`\varepsilon`$-greedy 探索时，SARSA 会把探索过程中可能采取的次优动作也考虑进价值估计。

**局限：如果当前策略仍有大量探索行为，SARSA 学习的是该探索策略对应的价值，而非直接逼近最优动作价值。**

### 3. Q-learning：用最优动作构造目标

Q-learning 不再使用实际执行的下一动作，而是直接选择下一状态中 Q 值最大的动作：

```math
\boxed{
Q(s_t,a_t)\leftarrow Q(s_t,a_t)
+\alpha[
r_t+\gamma\max_{a'}Q(s_{t+1},a')
-Q(s_t,a_t)]
}
```

这就是 Bellman 最优方程对应的更新。

| 比较 | SARSA | Q-learning |
|---|---|---|
| TD Target | $`r+\gamma Q(s',a')`$ | $`r+\gamma\max_{a'}Q(s',a')`$ |
| 策略类型 | On-policy | Off-policy |
| 学习目标 | 当前执行策略的 $`Q^\pi`$ | 最优动作价值 $`Q^*`$ |
| 是否考虑下一步探索动作 | 是 | 目标中直接取最大值 |
| 典型特点 | 行为相对保守 | 目标更偏向贪心最优 |

这里需要区分 **Behavior Policy** 和 **Target Policy**：

- Behavior Policy：实际与环境交互、产生数据的策略。
- Target Policy：正在评价或改进的策略。

Q-learning 可以用 $`\varepsilon`$-greedy 收集数据，却利用贪心动作构造目标，因此属于 Off-policy 学习。

### 4. DQN：状态空间太大，无法存储 Q 表

Tabular Q-learning 需要维护每个 $`(s,a)`$ 的 Q 值。

对于 Atari 游戏等高维状态空间，无法使用显式 Q 表，于是引入神经网络：

```math
Q_\theta(s,a)\approx Q^*(s,a)
```

DQN 用神经网络拟合动作价值函数，最小化 TD 误差：

```math
L(\theta)=
\mathbb E[
(Q_\theta(s,a)-y)^2]
```

其中：

```math
y=r+\gamma\max_{a'}Q_{\theta^-}(s',a')
```

这里省略了终止状态的折扣项处理。

但直接使用神经网络进行 Q-learning 存在两个关键问题。

**问题一：连续采集的数据高度相关。**

解决方法：Experience Replay。

将交互数据存入 Replay Buffer，随机采样 Mini-batch，减少样本之间的时间相关性，并提高数据利用率。

**问题二：网络更新时，TD Target 也随之变化。**

解决方法：Target Network。

使用独立的参数 $`\theta^-`$ 构造目标，并周期性或缓慢更新，使训练目标在一段时间内相对稳定。

这两个机制是 DQN 稳定训练的关键。

### 5. Double DQN：为什么 DQN 容易高估 Q？

DQN 在计算 TD Target 时，使用最大化操作：

```math
\max_{a'}Q_{\theta^-}(s',a')
```

当 Q 估计存在噪声时，最大化操作更容易选中被高估的动作，产生过估计偏差。

Double DQN 将**动作选择**和**动作评价**分离：

```math
a^*=\arg\max_{a'}Q_\theta(s',a')
```

```math
\boxed{
y=r+\gamma Q_{\theta^-}(s',a^*)
}
```

在线网络负责选择动作，目标网络负责评价动作，从而缓解过估计问题。

需要注意：Target Network 和 Double DQN 并不相同。

- Target Network：主要解决移动目标导致的不稳定。
- Double DQN：主要缓解最大化操作造成的过估计。

### 6. Q-learning 路线的根本限制

DQN 适用于有限离散动作空间，因为可以直接枚举动作并计算：

```math
\arg\max_a Q(s,a)
```

但如果动作是连续向量，例如机器人关节控制，就无法简单枚举所有动作。

因此产生了一个新的问题：

**能否不再通过遍历 Q 值选择动作，而是直接训练一个能够输出动作的策略？**

这引出第三条主线。

---

## 三、π 的训练：从策略梯度到 Actor-Critic

### 1. Policy Gradient：直接优化策略

与 Q-learning 先学习价值再选择动作不同，Policy-based 方法直接参数化策略：

```math
\pi_\theta(a|s)
```

其优化目标为：

```math
\boxed{
J(\theta)=
\mathbb E_{\tau\sim\pi_\theta}[R(\tau)]
}
```

策略梯度定理允许我们利用采样估计梯度。对于 REINFORCE，常用的梯度估计为：

```math
\nabla_\theta J
\propto
\mathbb E[
\nabla_\theta\log\pi_\theta(a_t|s_t)G_t
]
```

因此可以通过最小化：

```math
L_\pi(\theta)=
-\mathbb E[
\log\pi_\theta(a_t|s_t)G_t
]
```

来实现梯度上升。

直观地说：

- 如果某动作带来了较高回报，就增加该动作的概率。
- 如果某动作带来了较低甚至负的回报，就降低其相对概率。

但这种更新存在问题：**真实回报波动很大，导致梯度估计具有较高方差。**

### 2. Baseline：如何降低策略梯度方差？

考虑两个动作：

- 动作 A 得到奖励 10。
- 动作 B 得到奖励 12。

如果使用原始回报，两个动作都可能受到正向强化。但如果该状态下通常能得到 11，A 实际上低于预期，而 B 高于预期。

因此引入 Baseline：

```math
\nabla_\theta J\propto
\mathbb E[
\nabla_\theta\log\pi_\theta(a_t|s_t)
(G_t-b(s_t))
]
```

只要 Baseline 不依赖当前动作，就不会改变策略梯度估计的期望。

最常见的选择是：

```math
b(s_t)=V^\pi(s_t)
```

于是得到 Advantage：

```math
\boxed{
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
}
```

它衡量的是：**某动作相对于当前状态下平均表现的好坏，而不是绝对回报大小。**

这也解释了为什么前面训练的 $`V`$ 可以服务于策略梯度。

### 3. Actor-Critic：为什么要同时训练 π 和 V？

REINFORCE 即使使用 Baseline，也可能需要完整轨迹才能计算回报。

既然已经有 TD 方法，可以使用价值网络构造一步 Advantage 估计：

```math
\hat A_t=
r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
```

这个量恰好就是 TD Error：

```math
\hat A_t=\delta_t
```

由此得到 Actor-Critic 框架：

**Actor：训练策略 $`\pi_\theta`$**

```math
L_{\text{Actor}}=
-\mathbb E[
\log\pi_\theta(a_t|s_t)
\operatorname{sg}(\hat A_t)
]
```

其中 $`\operatorname{sg}`$ 表示将优势视为固定训练信号。

**Critic：训练价值 $`V_\phi`$**

```math
L_{\text{Critic}}=
\mathbb E[
(V_\phi(s_t)-
[r_t+\gamma V_\phi(s_{t+1})])^2
]
```

Critic 负责评价，Actor 根据评价结果改进策略。

A2C（Advantage Actor-Critic）是这种思想的一个具体实现，通常结合多环境同步采样和优势估计，而不是引入一种全新的策略优化原理。

### 4. PPO：为什么普通策略梯度仍然不稳定？

Actor-Critic 能够降低方差，但并不能自动防止策略更新过大。

如果一次梯度更新使动作概率发生剧烈变化，旧策略采集的数据可能无法有效指导新策略。

PPO 通过限制新旧策略的概率比率，缓解这一问题。

定义：

```math
r_t(\theta)=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{\rm old}}(a_t|s_t)}
```

PPO-Clip 的目标为：

```math
L^{\rm CLIP}(\theta)=
\mathbb E\left[
\min\left(
r_t(\theta)\hat A_t,\thinspace
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)
\hat A_t
\right)
\right]
```

其核心是限制过大的策略改进收益，从而抑制激进更新。

PPO 并没有取消价值学习，常见实现仍使用 Critic 估计 $`V`$，并据此计算 Advantage。

### 5. DPG 与 DDPG：如何直接输出连续动作？

对于连续动作空间，可以使用确定性策略：

```math
a=\mu_\theta(s)
```

与随机策略输出动作概率不同，确定性策略直接输出动作值。

DPG（Deterministic Policy Gradient）通过 Q 函数计算策略梯度：

```math
\boxed{
\nabla_\theta J
=
\mathbb E[
\nabla_\theta\mu_\theta(s)
\nabla_a Q^\mu(s,a)
\big|_{a=\mu_\theta(s)}
]
}
```

含义是：

1. Critic 计算当前动作的 Q 值。
2. 计算 Q 对动作的梯度。
3. 将梯度通过 Actor 反向传播，更新策略参数。

因此 DPG 不必枚举连续动作空间，也无需直接计算随机策略的对数概率。

DDPG 则将 DPG 与 DQN 的稳定化机制结合：

- 使用 Actor 网络输出连续动作。
- 使用 Critic 网络估计 $`Q(s,a)`$。
- 使用 Replay Buffer 复用数据。
- 使用 Target Networks 稳定 TD Target。
- 在行为动作上加入噪声实现探索。

其 Critic 目标为：

```math
y=r+\gamma
Q_{\phi^-}(s',\mu_{\theta^-}(s'))
```

DDPG 能够进行连续动作控制，但仍可能面临 Q 值高估、训练不稳定及探索不足的问题。后续 TD3 使用双 Critic 等机制进一步提高稳定性。

---

## 四、三条主线如何汇合？

前面虽然按照 $`V`$、$`Q`$、$`\pi`$ 分别展开，但实际算法往往联合训练这些对象。

| 算法 | 主要训练对象 | 价值信息的作用 | 策略如何得到 |
|---|---|---|---|
| MC / TD(0) | $`V^\pi`$ | 评价当前策略下的状态 | 本身不直接改进策略 |
| SARSA | $`Q^\pi`$ | 评价当前行为策略的动作 | 可基于 Q 进行 ε-greedy 改进 |
| Q-learning / DQN | $`Q^*`$ | 逼近最优动作价值 | $`\arg\max_a Q(s,a)`$ |
| REINFORCE | $`\pi`$ | 使用采样回报 | 直接更新策略 |
| Actor-Critic / A2C | $`\pi+V`$ | 估计 Advantage | 通过 Critic 指导 Actor |
| PPO | $`\pi+V`$（常见实现） | 优势估计和价值拟合 | 限制更新幅度的策略优化 |
| DDPG | $`\mu+Q`$ | 利用 Q 对动作的梯度 | Actor 直接输出连续动作 |

### 一个典型的融合案例：AlphaGo Zero

AlphaGo Zero 将策略、价值与搜索结合起来。

神经网络同时输出：

```math
(p,v)=f_\theta(s)
```

其中：

- $`p`$：当前状态下各动作的先验概率。
- $`v`$：当前状态的胜负价值预测。

MCTS 利用策略先验和价值预测引导搜索，得到改进后的搜索策略 $`\pi_{\rm MCTS}`$。

自我对弈生成训练样本：

```math
(s_t,\pi_{\rm MCTS},z_t)
```

其中 $`z_t`$ 是最终对局结果。

网络优化目标：

```math
\boxed{
L=(z-v)^2-\pi_{\rm MCTS}^{\mathsf T}\log p
+c\Vert\theta\Vert_2^2
}
```

这形成一个循环：

**神经网络提供策略与价值估计 → MCTS 改进决策 → 自我对弈产生数据 → 更新神经网络。**

这里并不需要单独训练一个 Q 网络；MCTS 在搜索树中维护动作价值估计，用于比较候选动作。

这也说明三条主线并非必须对应三个独立的神经网络，而是强化学习中三个不同的学习对象。

---

## 五、总结：如何理解强化学习算法的演进？

三个问题构成了强化学习的训练逻辑：

**第一，如何评价？**

从需要环境模型的 DP，到直接使用真实回报的 MC，再到利用自举实现在线学习的 TD，解决的是价值估计的效率问题。

**第二，如何选择动作？**

从 SARSA 学习当前策略的 Q，到 Q-learning 直接逼近最优 Q，再到 DQN 使用神经网络处理高维状态、Double DQN 缓解高估，解决的是动作选择与价值逼近的问题。

**第三，如何直接优化策略？**

从 REINFORCE 直接优化期望回报，到 Baseline 和 Actor-Critic 降低梯度方差，再到 PPO 限制策略更新、DDPG 支持连续动作，解决的是策略优化的效率、稳定性和适用范围问题。

最终，这些算法可以统一理解为一个不断迭代的过程：

```math
\boxed{
\text{与环境交互}
\longrightarrow
\text{估计价值 }(V,Q)
\longrightarrow
\text{改进策略 }\pi
\longrightarrow
\text{再次交互}
}
```

这也是强化学习中广义策略迭代（Generalized Policy Iteration）的核心思想：**策略评估与策略改进相互促进，逐步获得更好的决策能力。**
