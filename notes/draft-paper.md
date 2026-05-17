# Distributional Reinforcement Learning applied to LTM Handover Decisions

**Abstract:** 

**Index Terms:** Distributional Reinforcement Learning, LTM Handover

## Introduction

### The LTM HO



### Modelling the problem as a MDP

We can model this problem as a Markov Decision Process (Justificació!)


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

All thorughout this article we've been using a uniform distribution over the 0%-100% range for the quantiles.

However, it remains to be seen if a different distribution of quantiles may provide any benefits.


### The Midpoint Rule

Right now, we're using the midpoint rule

### Removing "useless" quantiles in CvAR


### Gaussian Quadrature

Should provide a more accurate with less points. Cou

### Trapezoidal rule

(empirically, it should perform worse) ==> how can I evalueate at the edges? (low=0, high=???)






## Further 
- IQN, FQF networks

## References: 

[1] Ainna Yue Moreno-Locubiche, et. al, Contextual Bandits and Reconfigurable Intelligent Surfaces for Predictive LTM Handover Decisions, 202?

[2] Distributional Reinforcement Learning