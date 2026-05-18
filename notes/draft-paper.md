# Distributional Reinforcement Learning applied to LTM Handover Decisions

**Abstract:** 

**Index Terms:** Distributional Reinforcement Learning, LTM Handover

## Introduction

### The LTM HO



### Modelling the problem as a MDP

We can model this problem as a Markov Decision Process (Justificació!)

**Finite state:** Uses the 82 variables ...

**Actions:** At each 100ms time step, our algorithm chooses which Sector to connect to. If the sector is the same as before, no HO is performed. If it's different, a HO is performed

## Distributional Reinforcement Learning

Distributional Reinforcement Learning is a variant of reinforcement learning, where instead of predicting the Q-value for any state-action pair $Q(x, a)$, we try to predict the actual distribution of this Q-value, by modelling it as a random variable $Z(x, a)$.

**Distributional Bellman Equation:** $$Z(x,a) \overset{D}{=} R(x,a) + \gamma Z(X', A')$$
where $\overset{D}{=}$ denotes equality in distribution, $R(x,a)$ is the reward, $\gamma$ is the discount factor, and $(X', A')$ are the next state and action random variables.

Even if we later only use this variable to choose the best action according to it's expected value $\mathbb{E}[Z(x, a)]$, this method allows our network have a richer learning signal, which in turn makes it that the final model is better.


In order to predict any arbitrary distribution, diverse algorithms exist. However, most of them have a common denominator: they approximate its cumulative probability distribution as a "stairwise function"





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

## Results on the LTM Handover procedure


## Results on the Atari 2600 benchmark

The Atari 2600 is well known as a standard benchmark for finite-action RL agents. We want to see if our improvements are only domain-specific, or if they provide a meaningful improvement over the base algorithm in 




## Further 
- IQN, FQF networks

## References: 

[1] Ainna Yue Moreno-Locubiche, et. al, Contextual Bandits and Reconfigurable Intelligent Surfaces for Predictive LTM Handover Decisions, 202?

[2] Distributional Reinforcement Learning