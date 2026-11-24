# Three Building Blocks of Reinforcement Learning

An agent learns by acting in an environment and observing what happens next. The goal is to choose actions that lead to higher expected long-term return. One way to organize the algorithms is to ask what each one learns:

$$
\begin{aligned}
V^\pi(s) &= \mathbb E_\pi[G_t \mid S_t=s],\\
Q^\pi(s,a) &= \mathbb E_\pi[G_t \mid S_t=s,A_t=a],\\
\pi(a\mid s) &= \Pr(A_t=a\mid S_t=s).
\end{aligned}
$$

- The **state value** $V^\pi(s)$ estimates the return from a state when the agent follows policy $\pi$.
- The **action value** $Q^\pi(s,a)$ estimates the return from taking a particular action first and then following $\pi$.
- The **policy** $\pi$ specifies how the agent chooses actions.

These three quantities answer different questions. They often appear together: a value estimate can guide an update to the policy that generated the data.

## Learning state values

### The Bellman equation

In a Markov decision process, the agent takes action $A_t$ in state $S_t$, receives reward $R_{t+1}$, and moves to $S_{t+1}$. With discount factor $\gamma$, the return splits into an immediate reward and a discounted remainder:

$$
G_t=\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}
   =R_{t+1}+\gamma G_{t+1}.
$$

Conditioning on the current state gives the Bellman expectation equation:

$$
V^\pi(s)=\mathbb E_\pi[R_{t+1}+\gamma V^\pi(S_{t+1})\mid S_t=s].
$$

The same recursion leads to different update targets depending on what information is available.

### Dynamic programming

If the transition probabilities $P(s'\mid s,a)$ and expected rewards $r(s,a,s')$ are known, policy evaluation can average over them explicitly:

$$
V_{k+1}(s)=\sum_a \pi(a\mid s)\sum_{s'}P(s'\mid s,a)
\bigl[r(s,a,s')+\gamma V_k(s')\bigr].
$$

Repeated backups propagate information about later rewards to earlier states. This approach assumes access to an environment model.

### Monte Carlo

Monte Carlo evaluation learns from sampled episodes. After an episode ends, its rewards give a complete sample $G_t$ for each visited timestep:

$$
V(S_t)\leftarrow V(S_t)+\alpha[G_t-V(S_t)].
$$

With a neural network, the same idea gives the squared-error loss $\mathbb E[(V_\phi(S_t)-G_t)^2]$. The target uses observed rewards throughout the remainder of the episode. Waiting until the end delays updates, and the return can vary considerably across trajectories.

### Temporal-difference learning

TD(0) uses a one-step transition. It adds the observed reward to a discounted estimate of the next state's value:

$$
\begin{aligned}
y_t &= R_{t+1}+\gamma V(S_{t+1}),\\
\delta_t &= y_t-V(S_t),\\
V(S_t)&\leftarrow V(S_t)+\alpha\delta_t.
\end{aligned}
$$

Using a value estimate in its own training target is called **bootstrapping**. It lets the agent update after each transition, at the cost of making the target depend on the current estimates. The value of a terminal state is taken to be zero.

| Method | Target | Environment model needed? |
| --- | --- | --- |
| Dynamic programming | An expectation over possible transitions | Yes |
| Monte Carlo | The sampled return $G_t$ | No |
| TD(0) | $R_{t+1}+\gamma V(S_{t+1})$ | No |

## Learning action values

### From state values to action choices

For discrete actions, state and action values are related by averaging over what the policy would do:

$$
\begin{aligned}
Q^\pi(s,a)&=\mathbb E[R_{t+1}+\gamma V^\pi(S_{t+1})\mid S_t=s,A_t=a],\\
V^\pi(s)&=\sum_a\pi(a\mid s)Q^\pi(s,a).
\end{aligned}
$$

A state value alone does not score each available action. Given the optimal action value $Q^*$, a greedy optimal policy can choose any action in $\arg\max_a Q^*(s,a)$.

### SARSA

SARSA builds its target from the next action actually selected. Its name comes from the five quantities in the update: state, action, reward, next state, and next action.

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+
\alpha\bigl[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)\bigr].
$$

This is an **on-policy** update. If the agent behaves $\varepsilon$-greedily, the estimated value reflects the exploratory actions it may take later.

### Q-learning

Q-learning uses the largest estimated action value at the next state in its target:

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+
\alpha\bigl[R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a')-Q(S_t,A_t)\bigr].
$$

The **behavior policy** determines how interactions are collected. The **target policy** is the policy the update evaluates or improves. Q-learning can collect experience with an exploratory behavior policy while using a greedy target. This makes it **off-policy**.

| | SARSA | Q-learning |
| --- | --- | --- |
| Next-state term | $Q(S_{t+1},A_{t+1})$ | $\max_{a'}Q(S_{t+1},a')$ |
| What the target follows | The current behavior policy | Greedy action selection |
| Update type | On-policy | Off-policy |

### DQN and Double DQN

A table becomes impractical when states are high-dimensional, such as raw game frames. A deep Q-network (DQN) uses $Q_\theta$ to approximate action values and fits it to a target $y$:

$$
\begin{aligned}
L(\theta)&=\mathbb E[(Q_\theta(s,a)-y)^2],\\
y&=r+\gamma\max_{a'}Q_{\theta^-}(s',a').
\end{aligned}
$$

The continuation term is zero for terminal transitions. **Experience replay** samples past transitions in batches, reducing the effect of training on successive, highly correlated observations. The **target network** with parameters $\theta^-$ changes more slowly than the online network, so the regression target moves less from one update to the next.

Taking the maximum of noisy estimates can favor actions whose values are overestimated. Double DQN separates the choice of action from its evaluation:

$$
a^*=\arg\max_{a'}Q_\theta(s',a'),
\qquad y=r+\gamma Q_{\theta^-}(s',a^*).
$$

The online network selects the action and the target network evaluates it. Slowing the target's updates helps with instability, while the separation in Double DQN addresses maximization bias.

DQN can enumerate the actions when the action space is small and discrete. For continuous actions, such as a vector of joint torques, it is useful to learn a policy that produces an action directly.

## Learning policies

### Policy gradients and REINFORCE

A stochastic policy $\pi_\theta(a\mid s)$ is parameterized by $\theta$. For the discounted objective $J(\theta)=\mathbb E_{\pi_\theta}[G_0]$, REINFORCE estimates the gradient from sampled trajectories:

$$
\nabla_\theta J(\theta)=
\mathbb E_{\pi_\theta}\left[
\sum_t\gamma^t\nabla_\theta\log\pi_\theta(A_t\mid S_t)\,G_t
\right].
$$

Each timestep contributes a score gradient weighted by the return that followed the action. In practice, the sampled returns can be treated as fixed weights when differentiating a loss built from the negative log probabilities. Even for the same action in the same state, sampled returns can vary widely, so the gradient estimate can have high variance.

### Baselines and advantage

Suppose two sampled actions yield returns of 10 and 12 from a state whose expected return is 11. Subtracting 11 gives a signal of $-1$ for the first sample and $+1$ for the second. The update can now distinguish a below-expectation result from an above-expectation one.

In the policy gradient, the weight $G_t$ can be replaced by $G_t-b(S_t)$. A baseline that does not depend on the sampled action leaves the expected gradient unchanged because the conditional expectation of $\nabla_\theta\log\pi_\theta(A_t\mid S_t)$ is zero. Taking $b(s)=V^\pi(s)$ gives the advantage function:

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
$$

### Actor-critic

An actor-critic method learns a policy and a value function together. A one-step advantage estimate is the temporal-difference error of the critic:

$$
\hat A_t=R_{t+1}+\gamma V_\phi(S_{t+1})-V_\phi(S_t).
$$

With an exact value function, the conditional expectation of this quantity given the state and action is the advantage. With a learned critic, it is a bootstrapped approximation. The actor uses that signal to update the policy, and the critic fits the value function to a target:

$$
\begin{aligned}
L_{\mathrm{actor}}&=-\mathbb E[\log\pi_\theta(A_t\mid S_t)\,
   \operatorname{sg}(\hat A_t)],\\
L_{\mathrm{critic}}&=\mathbb E[(V_\phi(S_t)-
   \operatorname{sg}(R_{t+1}+\gamma V_\phi(S_{t+1})))^2].
\end{aligned}
$$

Here $\operatorname{sg}$ means that the expression is held fixed when differentiating. Advantage actor-critic (A2C) is a practical version that commonly collects data synchronously from multiple environments.

### Proximal policy optimization

With data sampled from an older policy, large changes in the action probabilities can make an update unreliable. PPO-Clip compares the current and old probabilities of the sampled action:

$$
\rho_t(\theta)=\frac{\pi_\theta(A_t\mid S_t)}{\pi_{\theta_{\mathrm{old}}}(A_t\mid S_t)}.
$$

The objective to maximize is:

$$
L^{\mathrm{CLIP}}(\theta)=
\mathbb E\!\left[\min\!\left(
\rho_t(\theta)\hat A_t,\,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)\right].
$$

The clipped objective removes the incentive to move the ratio further in the beneficial direction once it crosses the relevant clipping boundary. Common implementations still train a value critic to help estimate advantages.

### Deterministic policies and continuous actions

A deterministic actor outputs an action $a=\mu_\theta(s)$. Deterministic policy gradient (DPG) updates the actor by differentiating the critic's score with respect to that action and applying the chain rule:

$$
\nabla_\theta J \;\propto\;
\mathbb E_s\!\left[
\left.\nabla_a Q^\mu(s,a)\right|_{a=\mu_\theta(s)}^\mathsf{T}
\nabla_\theta\mu_\theta(s)
\right].
$$

The state distribution in the expectation depends on the objective and sampling scheme. Deep deterministic policy gradient (DDPG) pairs a neural actor with a neural critic, uses experience replay and target networks, and injects noise into behavior actions for exploration. The critic uses a target actor to select the next action:

$$
y=r+\gamma Q_{\phi^-}(s',\mu_{\theta^-}(s')).
$$

As in DQN, terminal transitions have no continuation term. Q-value overestimation and unstable updates can still occur. TD3 addresses these problems with changes that include using two critics and delaying actor updates.

## How the three approaches fit together

| Method | What is learned | How it informs the policy |
| --- | --- | --- |
| Monte Carlo and TD(0) | $V^\pi$ | Evaluate the current policy |
| SARSA | $Q^\pi$ | Use action values for decisions such as $\varepsilon$-greedy selection |
| Q-learning and DQN | An estimate of $Q^*$ | Choose actions greedily from the estimates |
| REINFORCE | $\pi_\theta$ | Update action probabilities using sampled returns |
| Actor-critic and PPO | A policy and usually $V$ | Estimate advantages to guide policy updates |
| DDPG | A deterministic actor and $Q$ | Use the critic's action gradient to train the actor |

### AlphaGo Zero: policy, value, and search

AlphaGo Zero uses one network to produce move priors $p$ and a value estimate $v$ for a position:

$$
(p,v)=f_\theta(s).
$$

Monte Carlo tree search uses those predictions to produce a search policy $\pi_{\mathrm{MCTS}}$. Self-play generates tuples $(s_t,\pi_{\mathrm{MCTS}},z_t)$, where $z_t$ is the final game outcome from the perspective of the player at position $s_t$. The network learns to fit the outcome and the search policy:

$$
L=(z-v)^2-\pi_{\mathrm{MCTS}}^\mathsf{T}\log p+c\lVert\theta\rVert_2^2.
$$

The learned priors and values improve the next search, and that search supplies a new training target for the network. The tree keeps action value estimates for comparing moves during search.

## Summary

How much is a state worth? Dynamic programming answers using a model, Monte Carlo uses complete sampled returns, and TD learns from shorter transitions plus a value estimate. Which action is worth taking? SARSA evaluates the actions the current policy will actually choose, while Q-learning and its deep variants use greedy targets. How can we improve the decision rule itself? Policy gradient methods adjust the policy directly, often with a critic supplying a more useful learning signal than a raw return.

These ideas meet in **generalized policy iteration**. Policy evaluation provides information about how the agent is doing, policy improvement changes how it acts, and the improved policy changes what it experiences next.
