# Contextual Bandits and Reconfigurable Intelligent Surfaces for Predictive LTM Handover Decisions

**Ainna Yue Moreno-Locubiche, Josep Vidal, Olga Muñoz-Medina, Margarita Cabrera-Bean** *Dept. of Signal Theory and Communications, Universitat Politècnica de Catalunya - BarcelonaTech, Spain* *{ainna.yue.moreno, josep.vidal, olga.muñoz, marga.cabrera}@upc.edu*

**Abstract**—This article addresses the challenge of optimizing handover (HO) in next-generation wireless networks by integrating Reconfigurable Intelligent Surfaces (RIS), predicting received signal power, and utilizing learning-based decision-making. A conventional reactive HO mechanism, such as lower-layer triggered mobility (LTM), is enhanced through linear prediction to anticipate link degradation. Additionally, the use of RIS helps to mitigate signal blockage and extend coverage. An online trained non-linear Contextual Multi-Armed Bandit (CMAB) agent selects target gNBs based on context features, which reduces unnecessary HO and signaling overhead. Extensive simulations evaluate eight combinations of these techniques under realistic mobility and channel conditions. Results show that CMAB and RSRP prediction consistently reduce the number of HO, ping-pong rate and cell preparations, while RIS improves link reliability.

**Index Terms**—Multi-armed bandits, LTM handover, 6G, LMMSE prediction.

---

## I. INTRODUCTION

Handover (HO) mechanisms in mobile networks are designed to maintain seamless connectivity as users move between cells. The rise of 6G networks, with its higher frequency bands and more stringent service quality standards, introduces new challenges in maintaining seamless connectivity, especially in dense urban areas with non-line-of-sight (NLOS) conditions, and in high-mobility environments such as urban transport, UAVs, and vehicular networks [1]. Traditional HO mechanisms, which rely on reactive signal strength measurements, often fail in these dynamic scenarios, leading to frequent interruptions, high signaling overhead, and degraded service quality [2].

These limitations reduce service reliability and increase latency. To address these limitations, we propose a novel framework that integrates three cutting-edge technologies:

* **Contextual multi-armed bandits (CMABs):** A reinforcement learning-based decision engine that adapts to real-time network conditions, learning best HO decision policies from experience.
* **Prediction of received signal power:** A linear minimum mean square error (LMMSE) predictor that anticipates future received signal power, enabling proactive HO decisions and reducing ping-pong (PP) effects.
* **Reconfigurable intelligent surfaces (RISs):** Passive programmable metasurfaces that reflect and redirect signals to improve coverage and reliability in NLOS conditions.

While the usage of RISs has demonstrated potential in enhancing spectral and energy efficiency, its integration into mobility management remains underexplored. This work bridges that gap by combining the usage of RISs with predictive and learning-based algorithms to improve handover robustness in dense, fast-changing environments. The proposed system aligns with recent 3GPP advancements but goes further by embedding learning into the HO process, transforming it from reactive to anticipatory. The result is a significant improvement in HO performance indicators.

### A. Literature review

Conventional HO procedures, which rely on signal strength thresholds, often fail in dense urban environments and high-mobility scenarios. In ultra-dense networks, overlapping cell coverage leads to frequent handovers, causing PP effects, unnecessary resource reservation, increased signaling overhead and radio link failures. These procedures include [1]: hard HO (break-before-make) used in long-term evolution (LTE) and New Radio (NR), and soft HO (make-before-break) found in WCDMA, with hybrid approaches combining aspects of both.

More advanced mechanisms in 5G and beyond include conditional handover (CHO), where the UE executes transitions based on pre-set conditions; dual active protocol stack (DAPS), which allows simultaneous connections to source and target cells during HO; and low-layer triggered mobility (LTM), which enables ultra-fast switching using MAC and PHY layers [3]. These methods aim to reduce latency and improve reliability but remain non-adaptive rule-based [4], [5].

By leveraging AI-driven models, beamforming optimization, and reconfigurable intelligent surface (RIS)-assisted strategies, researchers aim to address the challenges posed by harsh mobility situations, dense deployments, and dynamic channel conditions. Machine learning offers promising solutions for dynamic HO management and outperforms static approaches. Deep learning models like LSTM can predict user movement [6], but are trained with large historical data sets for fixed terminal speeds. Reinforcement learning algorithms such as deep Q-networks (DQN) and proximal policy optimization (PPO) adapt HO parameters in real time [7]-[9]. In [10], multi-armed bandit (MAB) solutions are explored in ultra-dense mmWave networks, focusing on spatial and temporal learning from empirical mobility trajectories and blockage patterns. While effective in reducing unnecessary HO, these approaches lack real-time context awareness and do not incorporate predictive modeling or physical-layer enhancements.

We advocate for the adoption of a promising new technology, RISs, which are programmable surfaces that can selectively reflect and redirect wireless signals and have the capability to improve coverage and reliability, especially in NLOS scenarios [11]. In 3D networks involving terrestrial, aerial, and satellite layers, the usage of RISs has been shown to maintain seamless connectivity [2]. Cooperative HO strategies that combine RIS and gNB coordination could further enhance performance, although the cited article [12] focuses specifically on reducing signaling overhead caused by RIS reconfiguration during HO management.

### B. Contribution

We aim to address the underexplored intersection of RIS, mobility prediction, and AI-driven HO logic, offering a scalable solution for dense deployments that aligns with current 3GPP specifications. The main contributions are:

1.  **Learning-based decision engine:** We develop a decision engine based on contextual multi-armed bandit (CMAB) to select the optimal handover target based on contextual features (e.g., reference signal received power (RSRP), UE speed, historical performance). CMAB is preferred over supervised learning techniques due to its low computational cost, adaptability, and robustness in sparse-reward scenarios requiring real-time decisions.
2.  **Predictive modeling:** We implement an LMMSE predictor to forecast future RSRP values, leveraging the temporal correlation of shadowing and path loss. Linear prediction is preferred over neural networks, which tend to be data-wise inefficient in dynamic scenarios.
3.  **RIS-assisted HO:** We propose an optimization procedure to configure RIS coefficients in coverage-challenged areas. Additionally, we define a method to integrate RIS into the cell search process using predefined RIS beam patterns that enable synchronization signal block (SSB) detection in harsh propagating conditions.
4.  **System integration and evaluation:** We propose a modular architecture that integrates the three subsystems above and evaluates its performance using a comprehensive set of metrics.

---

## II. LTM HO PROTOCOL

LTM HO was introduced in 3GPP Release 18 to revolutionize HO procedures in 5G-Advanced by enabling URLLC through cell switching in MAC and PHY layers, bypassing traditional RRC-based signaling [13]. The procedure avoids resetting entire stacks such as RRC, PDCP, or RLC. It uses preestablished target cell configurations and relies on faster L1/L2 triggering mechanisms. This approach significantly reduces HO execution time (under 5 ms), HO latency (up to 70%), signaling overhead (since no RRC commands are expected), and packet loss by reducing reinitialization [14].

LTM has been shown to excel in many of the following metrics and events that are typically used to evaluate the performance of any HO system:
* **HO rate:** The number of HOs executions completed per user over a time period.
* **Radio link failures (RLF):** An RLF occurs when N310 consecutive out-of-sync indications are received, signaling that channel quality is deteriorating.
* **HO failures (HOF):** Declared when random access to the target cell is not successful.
* **PP events:** Occur when a UE successfully hands over to a target cell but returns to the previous serving cell within 1000 ms.
* **Reliability:** Measured as the ratio of total outage time to total service time. Outages may result from both failed and successful HOs.
* **Cell preparation events:** It denotes signaling overhead on the radio interface between the UE and the serving cell, as well as the overall network signaling load.
* **Resource reservations:** During HO preparation, the target cell reserves resources for the UE. It is quantified as the time between cell preparation and release for a UE.

---

## III. CMAB-BASED HO DECISIONS

In dynamic 6G environments, traditional HO methods based on signal strength and hysteresis margins fall short due to rapidly changing UE speed and interference. To address this limitation, we introduce a CMAB approach that provides a lightweight and sample-efficient alternative to full reinforcement learning. The CMAB operates under the assumption that actions, that is the selection of the target gNB, do not impact the context vector $x_{t}^{(a)}$ associated to possible action $a$. Optimal actions are selected to maximize long-term rewards related to throughput or HO latency [15].

### A. The contextual model

Let us define the set of actions $\mathcal{A}$, each element $a \in \mathcal{A}$ corresponds to the selection of a specific target gNB. We also define the reward $r \in \mathcal{R}$ as the real value associated to the adopted action. The main goal of applying a CMAB is to make a sequence of HO decisions (actions $a$) that maximize long-term link quality and reliability based on real-time contextual information, $x^{(a)} \in \mathcal{X} \subset \mathbb{R}^{d}$ that contains HO relevant metrics.

At each time step $t$, the HO agent observes a context vector $x^{(a)}$ describing the environment and selects an action $a$ following a strategy that tries to minimize the long term regret, defined as the sum over time of the expected difference between the reward for the best possible action $r_{t}^{*}$ and actual reward observed for action $a$: $l_{t} = \sum_{\tau=1}^{t} \mathbb{E}\{r_{\tau}^{*} - r_{\tau}^{(a)}\}$.

Each context vector $x_{t}^{(a)}$ is associated with a candidate $a$ and is designed to include the following components:
* **Signal quality:** The RSRP to the gNB $a$ at time $t$.
* **Network stability:** Moving p-samples average modulation and coding scheme (MCS) and SNIR value from cell $a$.
* **UE mobility:** Current UE speed.
* **HO dynamics:** Time since last HO and current serving gNB identifier.

In the linear CMAB setting, the expected reward of each $a$ is assumed to be a linear function of the context, enabling efficient algorithms such as LinUCB and Thompson Sampling with provable regret bounds [15]. However, real-world rewards may in general be nonlinear functions of the context. One prominent approach to handle nonlinearity is through kernel methods. Algorithms like KernelUCB and GP-UCB leverage this idea using reproducing kernel Hilbert spaces and Gaussian processes, respectively, to model complex reward functions while maintaining theoretical guarantees on regret [16]. After some extensive simulations, the rule adopted is the KernelUCB using arc-cosine function [17].

As the agent makes a HO decision $a$, it receives a real-valued reward $r_{t}^{(a)}$ that contains the combined effect of the quality of the HO decision.

### B. Performance-based rewards

The reward $r_{t}^{(a)}$ is designed to combine multiple performance indicators observed on the network (see section II):

$$r_t^{(a)} = r_{\mathrm{thr}} \cdot \frac{\alpha_{\mathrm{HO}}^{\mathbb{I}_{\mathrm{HO}_t}} \cdot \alpha_{\mathrm{PP}}^{\mathbb{I}_{\mathrm{PP}_t}} \cdot \alpha_{\mathrm{HOF}}^{\mathbb{I}_{\mathrm{HOF}_t}}}{1 + \exp(2(N_{OOS} - 2))} \quad (1)$$

where each term has the following meaning:
* $N_{OOS}$: number of consecutive out-of-sync situations detected before it actually signals a RLF. The reverse sigmoid function provide a soft transition between a "healthy link" and a "collapsed link."
* $r_{thr}$: Average MCS throughput over the last 100 ms.
* $\mathbb{I}_{HO_t}$, $\mathbb{I}_{HOF_t}$, $\mathbb{I}_{PP_t}$: Indicator functions that equal 1 if HO, HOF, or PP events occurred at time $t$, and 0 otherwise.
* $\alpha$: Weighting coefficients that balance the relative contribution of each reward component, enabling trade-offs between competing objectives. These are selected experimentally to maximize overall system performance: $\alpha_{HOF} = 0.1$, $\alpha_{HO} = 0.8$, $\alpha_{PP} = 0.9$.

---

## IV. LTM BASED ON PREDICTED RSRP LEVELS

To enhance HO reliability in dynamic mobile environments, we propose a predictive strategy that forecasts future RSRP values using LMMSE estimation [18]. By leveraging temporal correlations of the pathloss and shadowing in RSRP, the system can pre-reserve resources in target cells offering more stable connectivity, reducing unnecessary signaling and improving HO efficiency.

The processing pipeline consists of the UE measuring raw RSRP values from SSBs. These raw measurements undergo two layers of filtering—Layer 1 (L1) and Layer 3 (L3)—to mitigate the impact of fast fading and transient fluctuations. The L3-filtered RSRP values, which reflect more stable signal trends, serve as the input for the predictor. The LMMSE prediction horizon $\Delta$ must lie between the decorrelation time of multipath fading (fast variations) and that of shadowing (slow variations).

For the RSRP values received from each gNB, we compute:

$$\hat{s}_{t+\Delta} = w_{0} + w^{T}s_{t} \quad (2)$$

where $s$ is the vector containing current and past L3-filtered RSRP values for a given gNB, and $\{w_{0}, w\}$ are the weights that minimize prediction error for that gNB. The covariance matrix and covariance vector of $s_{t}$ involved are estimated using an exponential window, thus enabling time adaptation. The optimum $\Delta$ is obtained by minimizing the error of the LMMSE solution.

---

## V. RIS-ASSISTED HO PROCEDURE

RIS can enhance wireless communication by intelligently reflecting signals toward UE, especially at the cell edge or shadowed areas, thereby improving coverage and signal quality [11]. This capability enables more stable connections, facilitates better HO decisions, and reduces RLF and PP effects.

### A. Channel model

For RIS-assisted links, the cascaded MISO channel gain for the compound link gNB $\to$ RIS $\to$ UE is expressed as:

$$H_{c} = h_{ru}^{T}\Theta H_{br} \quad (3)$$

where $H_{br} \in \mathbb{C}^{N_{r}\times M}$ contains the channel gains between the $M$ antennas at the gNB and the $N_r$ RIS elements, and $h_{ru} \in \mathbb{C}^{N_{r}\times 1}$ contains the channel gains between the RIS and a single antenna UE. Moreover, $\Theta \in \mathbb{C}^{N_{r}\times N_{r}}$ is the RIS scatter matrix. For the diagonal passive RIS case, the unit-modulus reflection coefficients are stacked in $\Theta = \text{diag}(e^{j\phi}) \in \mathbb{C}^{N_{r}\times N_{r}}$, where $\phi = [\phi_{1}, \phi_{2}, \dots, \phi_{N_{r}}]$ are phase shifts applied to each RIS element independently reflecting the incident signal.

### B. Optimization of RIS scatter matrix

When the UE establishes a connection with the gNB, conventional approaches assume that full channel state information (CSI) can be obtained. In our RIS-assisted scenario case, we want the RIS to reflect omnidirectionally the SSB signal transmitted from a single antenna of a gNB over an area, operating without explicit knowledge of the RIS $\to$ UE channel state. To address this issue, RIS beam patterns are pre-computed to reflect SSBs over a square area of side $N$ meters, enabling UEs on that area to detect signals via indirect paths even when direct links are blocked. This approach supports initial access and synchronization in 5G NR systems, where SSBs are periodically transmitted using single antenna panels.

Two different approaches for the use of RIS beams are considered here. The first consists of sequentially applying different predefined RIS beam patterns per SSB transmission frame. Over multiple time multiplexed frames, the RIS sweeps through $T$ directions sequentially. The second approach consists of simultaneously applying $K$ RIS beam patterns on each SSB time frame. $K$ non-overlapping sets of RIS elements synthesize the precomputed beams and cover different portions of the area, thus allowing faster SSB identification but with lower received power due to the lower beam gain, as each set of RIS elements contain only $N_{r}/K$ reflectors.

With the goal of optimizing the RIS reflection performance over a target coverage area, we propose a worst-case zone-driven optimization approach. The method assumes a set of UE locations uniformly spread over the area. We denote the received signal power at a certain location $b$ as $|h_{eff}^{(b)}|^{2}$, which captures the effective gain of the gNB $\to$ RIS $\to$ UE link. Since we do not have UE-specific CSI, the estimated received power is computed using:

$$P_{r}^{(b)} = |h_{eff}^{(b)}|^{2} = |h_{ru}^{(b)T}\Theta h_{br}|^{2} \quad (4)$$

where vector $h_{br}$ contains the channel gains between the antenna at the gNB transmitting SSB omnidirectionally and the RIS elements. The RIS $\to$ UE channels are determined based on geometric layout, line-of-sight (LOS) propagation, and EM reflection models. The key idea is to maximize the minimum gain over the spatial grid, effectively guaranteeing robust worst-case performance. The optimization goal becomes:

$$\max_{\phi} \min_{b\in\mathcal{B}} P_r^{(b)} \quad (5)$$

where $\mathcal{B}$ is the set of all discrete locations $b$ covering the $N\times N$ grid area. This max-min fairness criterion ensures that the most shadowed zone in the area is enhanced as much as possible, under the principle that providing coverage over the entire area is more critical than optimizing for individual high-gain users.

We will adopt a gradient approach to determine the best RIS coefficients. The maximization of $P_{r}^{(b)}$ reads:

$$\max_{\phi} P_r^{(b_m)} \quad (6)$$
$$\text{s.t.} \quad \phi_{i} \in [0, 2\pi), \forall i = 1, \dots, N_{r}$$

The gradient of $P_{r}$ with respect to RIS phases is obtained from complex matrix differentiation:

$$dP_{r} = \text{Im}[h_{eff}H_{12}\text{diag}(e^{j\phi})]d\phi = (\mathcal{D}_{\phi}P_{r})d\phi \quad (7)$$

where $H_{12} = [\text{vec}(h_{ru,1}^* h_{br,1}^T), \dots, \text{vec}(h_{ru,N_r}^* h_{br,N_r}^T)] \in \mathbb{C}^{M \times N_r}$ and the operator $\text{Im}[\cdot]$ extracts the imaginary part. From (7), the gradient vector $\mathcal{D}_{\phi}P_{r}$ can be read out.

Algorithm 1 iteratively optimizes the RIS coefficients via the Adam gradient method. At each iteration the gradient at the position $b$ with minimum $P_{r}^{(b)}$ is selected. This way, we compute a codebook of pre-defined RIS coefficients that are used.

**Algorithm 1:** Iterative optimization of RIS coefficients on each subarea $\mathcal{B}$
```text
input: $h_{br}$, $H_{ru}$, $\mathcal{B}$, $\epsilon=1e-4$
output: $\varphi$
compute $H_{12}$
Initialize $t \leftarrow 0$, $J_0 \leftarrow 0$, $\phi_0$ random, $\varphi = e^{j\phi_0}$
repeat
    compute $\{P^{(b)}\}_{b \in \mathcal{B}}$   % Power received at locations in B
    $b_m = \arg \min_{b \in \mathcal{B}} P^{(b)}$   % Select location with minimum Pr
    $J_t = P_r^{(b_m)}$   % Received power metric
    $t \leftarrow t + 1$
    $g_t = D_{\phi} J_{t-1}$   % Compute gradient using (7)
    AdamGradient Update $(g_t, \phi_{t-1})$
    $\varphi = e^{j\phi_t}$
until $|\frac{J_t - J_{t-1}}{J_t}| < \epsilon$
```

---

## VI. RESULTS

We compare the proposed RIS-assisted, CMAB-optimized HO strategy with the traditional received signal strength (RSS)-based handover commonly used in LTE/5G systems.

### A. Simulation setup

The simulation parameters are divided into three main categories based on the system modules, each configured to capture realistic 5G urban deployment scenarios.

**a) Channel Model and Scenario:** Three channels are considered: the direct channel gNB $\to$ UE ($h_{bu}$), the channel gNB $\to$ RIS ($h_{br}$), and the channel RIS $\to$ UE ($h_{ru}$). Models for $h_{ru}$ and $h_{br}$ have been described in [19]. The direct channel model follows the 3GPP TR 38.901 specifications [20] for urban macro environments. In regions covered by the RIS, an additional attenuation factor is applied to the gNB $\to$ UE link to account for signal degradation of 20dB due to obstruction. In contrast, the gNB $\to$ RIS and RIS $\to$ UE channels are modeled with large-scale path loss (from the 3GPP TR 38.901) under the assumption that both links operate under LOS. The gNB $\to$ RIS link is static and unobstructed, while the RIS $\to$ UE link is assumed to be dominated by the deterministic geometry of the RIS deployment and the beam configuration.

A total of 7 hexagonal cells are placed, while two RIS panels per cell are deployed on cell edges to enhance RSRP in low coverage zones. Each cell is sectored at $120^{\circ}$ with a frequency reuse of 1/3. Other parameters are listed in Table I.

**TABLE I: Channel and Network Topology Parameters**
| Parameter | Value / Description |
| :--- | :--- |
| Path loss model | LOS 3GPP UMa in TR 38.901 |
| Shadowing standard deviation | 4 dB |
| Small-scale fading | Rayleigh-Jakes model |
| Carrier frequency | 10 GHz |
| Bandwidth | 200 MHz |
| UE speed | 10–18 km/h (running pedestrian) |
| Number of BSs | 7 (hexagonal cell) |
| Inter BS Spacing | 200 m |
| Sectors per BS | 3 (each $120^{\circ}$) |
| BS antenna gain | 3 dBi |
| $\theta_{3\text{dB}}$, $\phi_{3\text{dB}}$ | $65^{\circ}$ |
| Maximum attenuation ($A_m$) | 30 dB |
| Antenna element spacing | $\lambda/2$ |
| Transmit power | 25 dBm |
| Noise power density | -174 dBm/Hz |
| Frequency reuse factor | 3 |

**b) RIS Setup:** Each RIS features a total of $N_r = 12,800$ elements. The strategy considered is the single-beam SSB reflection over a $40\text{m} \times 40\text{m}$ area that will be denoted as $\mathcal{C}$. This area is partitioned into 16 sub-areas, each denoted by $\mathcal{B}_{i}$, $i=1,2,\dots,16$, corresponding to square regions of $10\text{m} \times 10\text{m}$.

To choose the appropriate number of sub-areas, we analyze several spatial granularities by partitioning the region into an increasing number of square sub-areas: from a single $40\text{m} \times 40\text{m}$ zone, to 4 $20\text{m} \times 20\text{m}$ regions, and up to 16 $10\text{m} \times 10\text{m}$ subzones. For each partitioning level, we compare two RIS control strategies described in section V. The most favorable performance is achieved when the area is divided into 16 subareas and all RIS elements are jointly optimized for each individual sub-region.

To optimize the RIS over $\mathcal{C}$, Algorithm 1 is applied independently to each sub-zone $\mathcal{B}_i$, so a codebook of 16 RIS coefficient sets is used to reflect the SSB. In our simulation setup, two RISs are placed on each cell in opposite positions relative to the gNB. $\text{RIS}_1$ is located at coordinates $[-40\text{m}, -40\text{m}]$ while $\text{RIS}_2$ is located opposite to $\text{RIS}_1$, at $[+40\text{m}, +40\text{m}]$ relative to the cell center.

**c) Handover Parameters:** The HO decision procedure includes LTM signaling procedures, prediction-based triggers using LMMSE-predicted RSRP, and penalization of ping-pong and failed handovers. UEs perform periodic SSB measurements, and events such as HO, HOF, and RLF are tracked. Statistical evaluation includes averaging over 1,000 independent simulated UE trajectories of 300 sec each.

**TABLE II: Handover Simulation Parameters**
| Parameter | Value / Description |
| :--- | :--- |
| Prediction Delay | 20 s |
| HO offset threshold | 3 dB |
| Target cell pre-selection | Predicted top-4 neighbors |
| Receiver sensitivity | -95 dBm |
| Time resolution (simulation) | 10 ms |
| Simulation duration | 300 s |
| Number of UEs | 1000 mobile UEs |
| L1/L3 Measurement Report Delay | 10 ms |
| HO Decision Computation | 10 ms |

### B. Comparative performance

To evaluate the effectiveness of each proposed technique in HO performance, we analyze the eight configurations:
1.  LTM
2.  LTM + RIS
3.  LTM + LMMSE
4.  LTM + LMMSE + RIS
5.  LTM + CMAB
6.  LTM + CMAB + RIS
7.  LTM + CMAB + LMMSE
8.  LTM + CMAB + LMMSE + RIS

The ultimate goal is to determine which combination yields the best performance across the multiple KPIs described in Section II. Conclusions from the results in Fig. 4 are compiled below:

![Description of Figure 4](./path-to-your-image/fig4.png)
**Fig. 4:** Fig. 4: Comparison of HO metrics across the eight proposed configurations. The figure demonstrates that integrating the Contextual Multi-Armed Bandit (CMAB) agent significantly reduces the handover rate, ping-pong events, handover failures, and cell preparation overhead, despite a slight increase in resource reservations and delayed exploratory decisions leading to radio link failures. Furthermore, LMMSE forecasting improves reliability and mitigates radio link failures, while Reconfigurable Intelligent Surfaces (RIS) strengthen link quality to further reduce radio link failures, ping-pongs, and handover failures. The eight subplots evaluate the following key performance indicators: (a) Handover rate (HO/min), (b) Handover failures (HOF/min), (c) Ping-pong rate (PP/min), (d) Capacity (bps), (e) Radio link failures (RLF/min), (f) Reliability (%), (g) Cell preparations per minute, and (h) Resource reservations (%).

* **CMAB Decisions:** CMAB reduces the frequency of HOs, PP events, HOFs (at a lesser extent), and significantly cuts down on cell preparation overhead by leveraging contextual information. However, it leads to a slightly higher resource reservations and increased RLFs due to delayed exploratory decisions. When combined with predictive (LMMSE) or physical-layer (RIS) enhancements, reliability remains consistently competitive.
* **RSRP Prediction:** LMMSE forecast of RSRP improves handover reliability and reduces RLFs by anticipating channel degradation. Its benefits are particularly significant in the absence of CMAB. When CMAB is applied, LMMSE contribution becomes marginal, as CMAB already incorporates temporal and contextual trends in the decision-making process.
* **RIS Integration:** RIS strengthens link quality and reduces RLFs by providing alternative reflective paths. This results in improved HO success as it also reduces PP rates and number of HOF. However, it can slightly increase the amount of cell preparations and resource reservations. Still, RIS consistently enhances signal reliability and reduces RLF.

### C. Best configurations based on LTM HO

The configuration combining **LTM + LMMSE + RIS + CMAB** exhibits the best overall performance by balancing high reliability, reduced HOFs and RLFs, and optimized HOs decisions with minimal control-plane overhead. It also achieves the lowest PP rate, benefiting from predictive awareness and RIS-enhanced link support, while accepting slightly higher resource reservations as a trade-off. For reliability-focused scenarios, LTM + LMMSE and LTM + LMMSE + RIS stand out, while CMAB-based approaches—especially those integrating RIS—excel in minimizing HO/min and ping-pongs. Additionally, all CMAB-integrated setups effectively reduce cell preparation overhead by targeting relevant cells.

---

## VII. CONCLUSIONS

This study presents a comprehensive evaluation of advanced HO strategies in 5G systems by integrating multiple technological innovations in a baseline LTM procedure. LMMSE prediction shows enhancement in decision timing, allowing HOs to be initiated before critical link degradation, RIS integration boosts signal propagation and reliability, and CMAB provides better HO decision-making. This combination results in a highly effective strategy that reduces unnecessary HOs, link failures (both HOFs and RLFs), ping-pongs, and cell preparations.

## VIII. ACKNOWLEDGEMENTS

The authors would like to thank Dr. Klaus Pedersen and Dr. Farah Sabouri from Nokia in Aalborg for the fruitful discussions and valuable insights that contributed to this work.

## REFERENCES

[1] M. Giordani, M. Polese, M. Mezzavilla, S. Rangan, and M. Zorzi, "Toward 6G networks: Use cases and technologies," *IEEE Communications Magazine*, vol. 58, no. 3, pp. 55-61, 2020.
[2] A. M. Vegni, Y. Ata, and M. S. Alouini, "Enhancement of handover management through reconfigurable intelligent surfaces in a 3D ground-aerial-space network scenario," *IEEE Transactions on Wireless Communications*, vol. 23, no. 12, pp. 18637-18652, 2024.
[3] Ericsson, "5G Advanced Handover: L1/L2 Triggered Mobility." Ericsson Blog. August 2024. [Online]. Available: https://www.ericsson.com/en/blog/2024/8/5g-advanced-handover-triggered-mobility
[4] A. Haghrah, M. P. Abdollahi, H. Azarhava, and J. M. Niya, "A survey on the handover management in 5G-NR cellular networks: aspects, approaches and challenges," *EURASIP Journal on Wireless Communications and Networking*, vol. 2023, no. 1, June 2023.
[5] 3GPP, "TS 38.300 v18.3.0," Tech. Rep., release 18. October 2024. [Online]. Available: https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3191
[6] M. Alsabah, A. Alqarni, R. Saeed, and M. H. Alsharif, "Handovers in 6G networks: Challenges and solutions," *IEEE Access*, vol. 9, pp. 93564-93585, 2021.
[7] Z. Liu, J. Wang, and X. Lin, "Reinforcement learning for adaptive handover management in 6G mobile networks," *IEEE Transactions on Mobile Computing*, vol. 20, no. 12, pp. 3421-3435, 2021.
[8] J. Voigt, P. J. Gu, and P. Rost, "A deep reinforcement learning-based approach for adaptive handover protocols," 2025. [Online]. Available: https://arxiv.org/abs/2503.21601
[9] Y. Wei, C.-H. Lung, S. Ajila, and R. Paredes Cabrera, "Deep Q-Networks assisted pre-connect handover management for 5G networks," in *97th Vehicular Technology Conference (VTC2023-Spring)*. 2023, pp. 1-6.
[10] L. Sun, J. Hou, and T. Shu, "Spatial and temporal contextual multi-armed bandit handovers in ultra-dense mmWave cellular networks," *IEEE Transactions on Mobile Computing*, vol. 20, pp. 3423-3438, 2021.
[11] E. Basar, M. Di Renzo, J. De Rosny, M. Debbah, M.-S. Alouini, and R. Zhang, "Wireless communications through reconfigurable intelligent surfaces," *IEEE Access*, vol. 7, pp. 116753-116773, 2019.
[12] M. Bensalem and A. Jukan, "Signaling rate and performance of ris reconfiguration and handover management in next generation mobile networks," 2024. [Online]. Available: https://arxiv.org/abs/2407.18183
[13] A. Gündogan, A. Badalıoğlu, P. Spapis, and A. Awada, "On the modelling and performance analysis of lower layer mobility in 5G-advanced," in *2023 IEEE Wireless Communications and Networking Conference (WCNC)*, 2023, pp. 1-6.
[14] B. Khodapanah et al., "On L1/L2-triggered mobility in 3GPP Release 18 and beyond," *IEEE Access*, vol. 12, pp. 171790-171806, 2024.
[15] L. Li, W. Chu, J. Langford, and R. E. Schapire, "A contextual-bandit approach to personalized news article recommendation," in *19th Intern. Conference on World Wide Web*. ACM, 2010, pp. 661-670.
[16] M. Valko, N. Korda, R. Munos, I. N. Flaounas, and N. Cristianini, "Finite-time analysis of kernelised contextual bandits," CORR, vol. abs/1309.6869, 2013. [Online]. Available: http://arxiv.org/abs/1309.6869
[17] Y. Cho and L. K. Saul, "Kernel methods for deep learning," in *Proceedings of the 23rd International Conference on Neural Information Processing Systems*, ser. NIPS'09. Red Hook, NY, USA: Curran Associates Inc., 2009, p. 342-350.
[18] N. Turan and W. Utschick, "Learning the MMSE channel predictor," in *2020 IEEE International Conference on Communications Workshops (ICC Workshops)*, 2020, pp. 1-6.
[19] A. Y. Moreno-Locubiche, J. Vidal, A. Pascual-Iserte, and O. Muñoz, "Reconfigurable intelligent surfaces for receive spatial modulation in rank-deficient channels," in *2023 IEEE Global Communications Conference*, 2023, pp. 5720-5726.
[20] 3GPP, "TR 38.901 v18.0.0," Tech. Rep., June 2024, release 18. [Online]. Available: https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationld=3190
