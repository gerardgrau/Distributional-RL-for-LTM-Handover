# Distributional Reinforcement Learning applied to LTM Handover Decisions

**Abstract:**

**Index Terms:** Distributional Reinforcement Learning, LTM Handover

## Introduction

### The LTM HO



### Modelling the problem as a MDP

We can model this problem as a Markov Decision Process (Justificació!)

**Finite state:** An 88-dimensional vector: UE speed (1) and serving-cell tenure (1), the one-hot serving sector (21), per-sector RSRP (21), moving-average MCS (21) and moving-average SINR (21), and the UE position $(x,y)$ (2).

**Actions:** At each 100ms time step, our algorithm chooses which Sector to connect to. If the sector is the same as before, no HO is performed. If it's different, a HO is performed

PARLAR DEL STATE OF THE ART A LTM HO (CMAB, LMMSE...)

## Distributional Reinforcement Learning

Distributional Reinforcement Learning is a variant of reinforcement learning, where instead of predicting the Q-value for any state-action pair $Q(x, a)$, we try to predict the actual distribution of this Q-value, by modelling it as a random variable $Z(x, a)$.

**Distributional Bellman Equation:** $$Z(x,a) \overset{D}{=} R(x,a) + \gamma Z(X', A')$$
where $\overset{D}{=}$ denotes equality in distribution, $R(x,a)$ is the reward, $\gamma$ is the discount factor, and $(X', A')$ are the next state and action random variables.

Even if we later only use this variable to choose the best action according to it's expected value $\mathbb{E}[Z(x, a)]$, this method allows our network have a richer learning signal, which in turn makes it that the final model is better.


In order to predict any arbitrary distribution, diverse algorithms exist. However, most of them have a common denominator: they approximate its cumulative probability distribution as a "stairwise function"

PARLAR DEL STATE OF THE ART DE DISTRL



## Risk-Aware Distributional RL

As mentioned, in the standard Distributional RL, we choose the best action greedily over the values of $\mathbb{E}[Z(x, a)]$, which can be approximated as the average value of all the predicted quantiles: $\sum_{i=1}^{N} \frac{1}{N} \cdot \theta_i(x, a)$

However, one may of a different strategy to aggregate the $N$ values of $\theta_i$ in order to achieve a different "best action", following a different criterion.

Although it has been proven that it doesn't exists any such strategy that can perfectly order all distributions [REVISE] [MISSING CITATION]

One can come up with strategies to minimize risk. For example, a very common strategy is (instead choosing the action with highest expected value) choose the action with highest Conditional Value at Risk (CVaR) at level $q%$. This is computed as the average of the worst $q%$.

So if we have a Uniform distribution from $0$ to $1$, the CVar at level $20%$, is the average of the worst 20% of the results, that is the average of the values from $0$ to $0.2$, which is $0.1$.


In problems (sinonim????) such as the LTM Handover decisions, Risk aware decision-making is favorable (sinonim???) because being able to avoid the worst case scenarious (such as HO faliures or radio link faliures) is much better (sinomin???) than just maximing over the expected value



## Quantile positioning

QR-DQN parametrises $Z(x,a)$ by its inverse CDF at $N$ quantile fractions $\tau_1, \dots, \tau_N \in [0,1]$, learning $\theta_i(x,a) \approx F^{-1}_{Z(x,a)}(\tau_i)$. The aggregated Q-value used for action selection,
$$\mathbb{E}[Z(x,a)] \;=\; \int_0^1 F^{-1}_{Z(x,a)}(\tau)\, d\tau,$$
is replaced by a finite-sample quadrature $\sum_{i=1}^{N} w_i \cdot \theta_i(x,a)$ with weights $w_i \ge 0$, $\sum w_i = 1$. The choice of $(\tau_i, w_i)$ defines the quantile-positioning scheme.

Throughout the literature, QR-DQN papers default to the midpoint rule. In this section, we explore three alternatives and ask whether either the integration accuracy of the aggregator, or the placement of the learnable quantiles, affects the policy quality on the LTM HO task.

### The Midpoint Rule

Vanilla QR-DQN places $\tau_i = (i - \tfrac{1}{2})/N$ with uniform weights $w_i = 1/N$. This is the midpoint Riemann sum for $\int_0^1 F^{-1}(\tau)\,d\tau$ and has integration error $O(1/N^2)$ for smooth integrands. All HP-search results, including the champion in §[hp-search], use this scheme.

### Gauss-Legendre Quadrature

Gauss-Legendre quadrature places the $N$ nodes at the roots of the $N$-th Legendre polynomial shifted to $[0,1]$, with weights derived from the same polynomial basis. The resulting rule integrates polynomials up to degree $2N-1$ exactly — a quartic improvement over midpoint for smooth $F^{-1}$.

In practice, the network still outputs $N$ predicted quantiles, but the $\tau$ used in the QR loss matches the Gauss-Legendre node positions (denser near the support boundary, sparser around $\tau=0.5$), and the aggregator uses the corresponding non-uniform weights. The output dimension is unchanged, so the parameter count is identical to midpoint.

### Trapezoidal Rule with Fixed Endpoints

The trapezoidal rule uses uniform spacing including the endpoints $\tau_0 = 0$ and $\tau_{N-1} = 1$, with weights $w_0 = w_{N-1} = 1/(2(N-1))$ and $w_i = 1/(N-1)$ for interior $i$. The caveat is that the rule requires evaluating $F^{-1}$ at the endpoints, where the inverse CDF is unbounded for distributions with infinite support.

We side-step this by fixing the endpoints to constants chosen a priori from physical considerations: $F^{-1}(0) = 0$ (returns are non-negative under the Ainna multiplicative reward) and $F^{-1}(1) = q_{\max}$ (a conservative upper bound on the discounted return, set empirically from the midpoint baseline's learned distribution). The network then only predicts the $N - 2$ interior quantiles, with the QR loss applied to those alone.

### Truncated CVaR — Dropping Unused Quantiles

When the policy is CVaR $_\alpha$, the action selection rule only integrates over $\tau \le \alpha$. The upper $1 - \alpha$ of the predicted quantile vector is generated by the network and then discarded — wasted capacity. We test the alternative: train the network on only $k = \lceil \alpha N \rceil$ quantiles placed uniformly in $[0, \alpha]$, removing the upper lobe from both the prediction and the QR loss. Action selection in this regime is simply the average of the $k$ predictions, which is exactly CVaR $_\alpha$ under the midpoint rule.

The hypothesis is that concentrating network capacity on the part of the distribution that actually drives the policy improves the quality of the CVaR estimate.

# Results

### Evaluation protocol and metrics

All agents are trained and evaluated on the same 1,000 pre-computed UE trajectories (300 s each, 10 ms simulation resolution, one HO decision every 100 ms). We deliberately use the full trajectory set for both training and final evaluation rather than a held-out split: the channel-gain dataset is fixed and the policy never observes future returns, so the relevant question is how well each agent exploits the available propagation environment, not generalisation to unseen geometry. Final numbers are obtained by freezing the learned weights, setting $\epsilon = 0$ (greedy), and replaying all 1,000 UEs. Every configuration is run for **5 random seeds $\times$ 2,000 episodes**; we report the cross-seed mean $\pm$ standard deviation.

We track the reward (the multiplicative Ainna HO reward the agents optimise) together with the eight physical KPIs used by the prior work [1]: handover rate (HO/min), handover-failure rate (HOF/min), ping-pong rate (PP/min), radio-link-failure rate (RLF/min), spectral efficiency / capacity (bps/Hz), link reliability (%), cell preparations (/min) and resource reservation (%).

### Models and hyperparameters

We compare three learned agents, all sharing an identical two-layer MLP trunk ($256\times256$) and the same observation, action and reward definitions, so that any performance difference is attributable to the *learning target* rather than to capacity or input:

1. **DQN** — vanilla scalar value learning; action selection by $\arg\max_a Q(x,a)$.
2. **QR-DQN (risk-neutral, RN)** — distributional, but the policy is still the mean $\mathbb{E}[Z(x,a)]$ (standard QR-DQN). Predicts 10 quantiles over the full support $[0,1]$.
3. **QR-DQN (risk-aware, RA)** — distributional with a $\mathrm{CVaR}_{0.5}$ policy realised by the truncated scheme of §[truncated-cvar]: the network predicts only the $k=\lceil\alpha N\rceil=10$ lower quantiles ($\alpha=0.5$, $N=20$), so action selection is the mean of the worse half of the return distribution.

The two QR-DQN variants are matched to the **same effective number of predicted quantiles** ($k=10$), hence the same head size and parameter count; they differ only in *where* those quantiles sit (full $[0,1]$ vs. lower half $[0,0.5]$) and in the corresponding selection rule (mean vs. CVaR). This isolates risk-aware decision-making from model capacity. Hyperparameters are listed in Table III.

**TABLE III: Agent hyperparameters (V3 backbone).**
| Hyperparameter | DQN | QR-DQN (RN) | QR-DQN (RA) |
| :--- | :---: | :---: | :---: |
| Hidden layers | $256\times256$ | $256\times256$ | $256\times256$ |
| Learning rate | $10^{-4}$ | $10^{-4}$ | $10^{-4}$ |
| Discount $\gamma$ | 0.9 | 0.9 | 0.9 |
| Replay buffer | $10^5$ | $10^5$ | $10^5$ |
| Batch size | 256 | 256 | 256 |
| Train frequency | 1 | 1 | 1 |
| Target update | soft, $\tau=0.01$ | soft, $\tau=0.01$ | soft, $\tau=0.01$ |
| $\epsilon$ schedule | $1.0\!\to\!0.05$ ($\times0.995$) | idem | idem |
| Huber $\kappa$ | — | 0.25 | 0.25 |
| Nominal quantiles $N$ | — | 10 | 20 |
| Predicted quantiles $k$ | — | 10 | 10 |
| Risk policy | — | mean $\mathbb{E}[Z]$ | $\mathrm{CVaR}_{0.5}$ |
| Quantile support | — | $[0,1]$ | $[0,0.5]$ |

All three agents share a single backbone (soft target update, $\tau=0.01$); only the value head differs. We separately verified that the hard-vs-soft target-update choice is within cross-seed noise for both DQN and QR-DQN, so fixing one shared scheme costs nothing and keeps the comparison strictly apples-to-apples.

### Choosing the risk geometry: a de-confounded sweep

The risk-aware agent couples two knobs: the number of in-use quantiles $k$ and the CVaR level $\alpha$. Because $k=\lceil\alpha N\rceil$ in the truncated scheme, a naïve sweep that fixes the *nominal* $N$ and varies $\alpha$ silently changes $k$ too — conflating "how risk-averse" with "how many quantiles". We therefore sweep the two axes separately, on the final backbone (1,000 episodes $\times$ 3 seeds per point).

**Quantile count.** Fixing $\alpha=0.5$ and varying the head so $k\in\{10,13,15,25,50\}$ leaves reward essentially flat — $1083.8$, $1084.6$, $1083.4$, $1082.4$, $1083.1$ (all within cross-seed noise) — so we adopt the small, cheap $k=10$ head.

**Risk level.** Holding $k=10$ and varying $\alpha$ (adjusting $N$ so $\lceil\alpha N\rceil=10$ at every point, keeping the head identical) traces a concave curve with an interior optimum (Table IV).

**TABLE IV: Reward vs. CVaR level $\alpha$ at fixed $k=10$ (1000 ep $\times$ 3 seeds).**
| $\alpha$ | 0.30 | 0.40 | 0.50 | 0.70 | 1.00 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Reward | $1081.2\pm4.7$ | $1081.5\pm1.7$ | $\mathbf{1083.8\pm1.2}$ | $1081.2\pm1.1$ | $1074.6\pm1.9$ |

Reward peaks at $\alpha\approx0.5$ atop a broad interior plateau ($\alpha\in[0.3,0.7]$ all within $\sim1081$–$1084$) and falls off sharply toward the risk-neutral extreme: $\alpha=1.0$ (which, since $\mathrm{CVaR}_{1.0}=\mathbb{E}[Z]$, *is* the risk-neutral policy) is $9.2$ reward points — about $7\sigma$ — below $\alpha=0.5$, and is simultaneously worst on every safety KPI (HOF $1.83$ vs. $1.28$, RLF $0.148$ vs. $0.103$). This reverses what a confounded $N$-fixed sweep suggests (reward apparently rising monotonically with $\alpha$): once the quantile count is held fixed, the apparent benefit of large $\alpha$ vanishes and a genuine risk-aware optimum near $\alpha\approx0.5$ emerges. We adopt $\alpha^\star=0.5$ — CVaR of the worse half — the most stable point of the plateau and the most interpretable risk level.

### Three-way comparison

Table V reports the learned agents at the canonical $2000\times5$ budget, with the prior-work LTM reference of [1] for context. The three headline agents (DQN, RN, RA) form the matched-capacity comparison; the fourth QR-DQN column is the single-quantile ($k{=}1$) risk-averse ablation (analysed in §[bridge]). Arrows mark the desirable direction; bold marks the best learned agent per row.

**TABLE V: Final performance (5 seeds $\times$ 2000 ep, mean $\pm$ std).**
| KPI | DQN | QR-DQN (RN) | QR-DQN (RA) | QR-DQN (RA, $k{=}1$) | LTM [1] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Reward $\uparrow$ | $1067.4\pm2.1$ | $1081.5\pm1.7$ | $\mathbf{1086.3\pm1.2}$ | $1080.6\pm1.2$ | — |
| HOF/min $\downarrow$ | $2.233$ | $1.432$ | $\mathbf{1.176}$ | $1.469$ | $1.10$ |
| RLF/min $\downarrow$ | $0.143$ | $0.107$ | $\mathbf{0.093}$ | $0.104$ | $0.068$ |
| Capacity (bps/Hz) $\uparrow$ | $3.909$ | $3.951$ | $\mathbf{3.974}$ | $3.974$ | $3.75$ |
| Reliability (%) $\uparrow$ | $95.82$ | $96.10$ | $\mathbf{96.25}$ | $96.19$ | $95.0$ |
| HO/min $\downarrow$ | $\mathbf{13.86}$ | $14.74$ | $15.47$ | $16.12$ | $11.0$ |
| PP/min $\downarrow$ | $\mathbf{1.827}$ | $2.005$ | $1.946$ | $2.041$ | $3.45$ |
| Prep/min $\downarrow$ | $\mathbf{1010.8}$ | $1038.8$ | $1057.6$ | $1072.9$ | $780$ |
| Res. reservation (%) $\downarrow$ | $\mathbf{8.022}$ | $8.244$ | $8.394$ | $8.515$ | $5.7$ |

Two observations structure the result.

**Distributional learning helps, even with a risk-neutral policy.** Moving from DQN to risk-neutral QR-DQN — same mean-based action rule, same architecture, only a distributional learning target — raises reward by $+14.1$ ($11.7\sigma$) and cuts HOF by $36\%$ and RLF by $25\%$, while lifting capacity and reliability. The richer per-action learning signal alone is worth a large, highly significant margin. This gain genuinely needs *multiple* quantiles, however: a risk-neutral QR-DQN restricted to a single quantile is statistically indistinguishable from DQN, so the improvement is a property of the distributional representation rather than of the architecture (the ablation in §[bridge] traces reward from this DQN-equivalent point up to the full $k{=}10$ plateau).

**Risk-awareness adds a second, separable gain.** Switching the *policy* from the mean to $\mathrm{CVaR}_{0.5}$ — at identical capacity ($k=10$) — adds a further $+4.8$ reward ($5.1\sigma$) and cuts HOF by another $18\%$ (to $1.176$) and RLF by another $13\%$ (to $0.093$). The two effects stack: the risk-aware agent improves reward, HOF, RLF, capacity and reliability monotonically over both baselines.

**The trade-off is interpretable.** The same ordering that improves the objective and safety KPIs *increases* the activity/overhead KPIs: the risk-aware agent performs more handovers ($15.5$ vs. DQN's $13.9$/min), more cell preparations and more resource reservation. Distributional risk-aware decision-making is *more willing to act*, but its actions are far more reliable — HOF nearly halved relative to DQN ($1.176$ vs. $2.233$, $-47\%$), RLF cut by $35\%$ — and yield higher throughput. The agent buys reliability and capacity with signalling activity.

### Where the gain comes from: a quantile-bridge ablation

To localise the two gains we sweep the head from a single quantile up to the full $k=10$ (1000 ep $\times$ 3 seeds for the ablation; the single risk-averse point is also re-run at the canonical $2000\times5$ budget).

**The distributional gain is a multi-quantile effect.** A risk-neutral QR-DQN with a *single* quantile at $\tau=0.5$ reduces, by construction, to DQN: the pinball loss at $\tau=0.5$ is a symmetric $\tfrac{1}{2}$-Huber loss on the median, collapsing the agent to scalar value learning. Our ablation confirms this empirically — the two are statistically indistinguishable ($1063.8$ vs. $1060.9$, $0.8\sigma$). Reward then climbs smoothly with the quantile count ($N{:}1\to10$ gives $1063.8\to1075.4$), saturating by $N\approx5$. The distributional advantage thus requires $N>1$ and is essentially complete by $\sim10$ quantiles.

**A single risk-averse quantile is *not* enough.** At the canonical $2000\times5$ budget, a single lower-tail quantile ($k=1$, $\tau=0.25$, $\alpha=0.5$) converges to *risk-neutral* performance — reward $1080.6\pm1.2$, HOF $1.469$ — statistically tied with the $k=10$ risk-neutral agent ($1081.5$, $1.432$) and $5.7$ reward ($7.8\sigma$) / $0.29$ HOF ($6.2\sigma$) below the full $k=10$ CVaR agent. Tellingly, it incurs the *highest* handover, preparation and reservation overhead of all agents while delivering only risk-neutral outcomes: a single tail quantile is a high-variance risk estimate that over-reacts without the payoff, whereas CVaR's averaging over the lower $k=10$ quantiles is a stable estimate that yields fewer handovers, lower overhead and the best outcome.

**Both axes are required.** Risk-aversion is necessary but not sufficient on its own — it needs a stable, multi-quantile (CVaR) estimate. Neither a single risk-averse quantile nor a risk-neutral 10-quantile model reaches the champion; only their combination does.

### Comparison with the prior LTM baseline

Operating on the *same* bare-LTM substrate as [1] — i.e. without that work's orthogonal RIS, LMMSE-prediction or CMAB enhancements — the risk-aware agent improves on the published LTM numbers where it matters for the user: $+6.0\%$ capacity ($3.974$ vs. $3.75$), $+1.25$ points of reliability ($96.25\%$ vs. $95.0\%$), and a $44\%$ lower ping-pong rate ($1.95$ vs. $3.45$/min), while essentially matching its HOF ($1.176$ vs. $1.10$). It does so at a higher control-plane cost — more handovers, preparations and resource reservation — and a somewhat higher RLF ($0.093$ vs. $0.068$). Because RIS, LMMSE and CMAB act on signal quality and target pre-selection rather than on the value estimate, they are complementary to the distributional-RL decision policy proposed here and could be stacked on top of it; we leave that combination to future work.

### Summary

On the LTM HO task, distributional RL delivers two cleanly separable improvements over scalar DQN: a large gain from the distributional learning target itself (DQN $\to$ RN), and a second, smaller-but-significant gain from a CVaR-based risk-aware policy (RN $\to$ RA) whose optimal level ($\alpha\approx0.5$) we establish with a de-confounded sweep. The resulting risk-aware QR-DQN attains the best reward and the best safety and throughput profile of any learned agent, halving the handover-failure rate of DQN and exceeding the prior LTM baseline on capacity, reliability and ping-pong rate. A quantile-bridge ablation confirms both ingredients are necessary: a single risk-neutral quantile collapses to DQN, and even a single *risk-averse* quantile only reaches the risk-neutral 10-quantile level — the combined gain requires the full $\sim\!10$-quantile CVaR representation.

## Results on the Atari 2600 benchmark

The Atari 2600 is well known as a standard benchmark for finite-action RL agents. We want to see if our improvements are only domain-specific, or if they provide a meaningful improvement over the base algorithm in 




## Further 
- IQN, FQF networks

## References: 

[1] Ainna Yue Moreno-Locubiche, et. al, Contextual Bandits and Reconfigurable Intelligent Surfaces for Predictive LTM Handover Decisions, 202?

[2] Distributional Reinforcement Learning