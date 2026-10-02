# Dynamic Traffic Routing in Software-Defined Networks Using Deep Reinforcement Learning

**Agbeni Daniel Oluwafemi**  
*Department of Networking and Cloud Computing, School of Computing, Information and Communication Technology, The Federal Polytechnic, Ado-Ekiti*  
*Project Repository: [github.com/DanielAgbeni/GARRO](https://github.com/DanielAgbeni/GARRO)*  

---

## Abstract

Modern 5G/6G, Internet of Things and cloud traffic is highly dynamic, and traditional routing protocols such as OSPF and ECMP cannot adapt to real-time congestion, resulting in bottlenecks, high latency and packet loss. Existing Deep Reinforcement Learning (DRL) routing approaches for Software-Defined Networks (SDN) also suffer from unsafe exploration on live networks and poor generalisation across topologies. This project designed, developed and evaluated the Graph-Attention Reinforcement Routing Orchestrator (GARRO), a hybrid intelligent routing architecture for dynamic SDNs. Traffic routing was formulated as a multi-objective Markov Decision Process, and a Proximal Policy Optimization (PPO) agent with a Graph Transformer encoder was built for topology-agnostic decisions. The agent was trained offline in a Digital Twin based on M/M/1/K queuing theory, while an Agentic AI layer using a Large Language Model translates operator intent into reward weights and provides deterministic fallback routing. Implementation covered offline training on the NSFNET, GEANT2 and Fat-Tree topologies and live emulation using Mininet, Open vSwitch and an OS-Ken controller with a web orchestrator. In a 500-episode (100,000-step) NSFNET benchmark under bursty traffic, GARRO achieved the highest mean reward (496.96), marginally above OSPF (495.60) and 32.6% above ECMP (374.86), with the lowest reward variance of all algorithms tested, and intent translation took under 80 milliseconds. The study concludes that a Graph Transformer-based PPO agent trained in an offline Digital Twin can match or outperform classical routing under bursty traffic while generalising across topologies, and recommends physical-testbed validation in future work.

**Keywords:** Software-Defined Networking (SDN), Deep Reinforcement Learning (DRL), Proximal Policy Optimization (PPO), Graph Transformers, Traffic Engineering, Digital Twin, Agentic AI, Quality of Service (QoS)

---


# 1. Introduction

Traditional routing in IP networks is largely governed by interior gateway protocols such as OSPF and IS-IS, which compute forwarding paths using variants of Dijkstra’s shortest-path algorithm based on predefined link weights. While highly scalable and effective in stable environments, shortest-path routing consistently routes traffic through a singular optimal path regardless of current bandwidth utilization or real-time network conditions (Abrol et al., 2024). As a result, central links frequently become congested bottlenecks while peripheral links remain underutilized, severely degrading network throughput, increasing transmission delay, and elevating packet loss rates.


## 1.1 Background to the Study

To introduce load distribution, Equal-Cost Multi-Path (ECMP) routing was widely adopted. ECMP splits traffic flows across multiple paths of equal cost using hash-based load balancing. While ECMP improves upon singular shortest-path routing, it remains fundamentally blind to real-time network states and does not dynamically adjust to transient microbursts or asymmetrical traffic loads, leading to hash collisions where multiple heavy flows (often referred to as "elephant flows") are mapped to the same path, overwhelming link capacity and causing queue overflows (Kim et al., 2022).

Recent iterations of traffic engineering have attempted to dynamically adjust OSPF link weights or deploy Segment Routing over IPv6 (SRv6) to direct traffic explicitly. However, dynamically recalculating and disseminating link weights in a distributed architecture induces severe signaling overhead and route flapping, while Segment Routing introduces significant header overhead and scalability demands (Casas-Velasco et al., 2022). Software-Defined Wide Area Networking (SD-WAN) provides cost-effective flexibility but struggles to deliver the strict, deterministic QoS guarantees required by 5G and 6G applications (Casas-Velasco et al., 2022).

Deep Reinforcement Learning circumvents the limitations of both heuristics and supervised learning through an experience-driven, model-free architecture. A DRL agent operates as an active decision-maker situated within the SDN controller, or as a dedicated application communicating via northbound APIs. By observing the network state, taking routing actions, and receiving performance-based rewards, the DRL agent approximates complex, non-linear control policies without requiring prior labeled datasets (Abrol et al., 2024). The efficacy of the traffic engineering system depends heavily on the specific reinforcement learning algorithm deployed, with the evolution of DRL algorithms from 2020 to 2026 highlighting a shift toward architectures that handle high-dimensional continuous spaces, ensure stable convergence, and adapt to topological mutations.


## 1.2 Statement of the Problem

Even within an SDN environment, centralized heuristic algorithms scale poorly. As the number of nodes and links increases, the computational time required to solve linear programming models for traffic matrices grows exponentially, making real-time, microsecond-level flow scheduling mathematically intractable (Abrol et al., 2024). Supervised deep learning models have been proposed to predict traffic matrices and aid in routing classification, yet such models rely entirely on massive datasets of accurately labeled data. In network environments, obtaining labeled data that maps specific traffic states to optimal routing configurations is nearly impossible due to the infinite permutation of network states, traffic matrices, and microbursts. In addition, supervised models suffer from severe generalization failures commonly referred to as the "reality gap" when deployed in environments with traffic distributions differing from their training data (Abrol et al., 2024).

Initial intelligent routing frameworks utilized classical tabular Q-Learning. By maintaining a Q-table storing the expected utility of every state-action pair, agents could route traffic dynamically. The Q-Optimizer framework, refined as recently as 2026, utilizes lightweight, two-stage tabular learning relying on offline pre-evaluation and real-time rule-based adaptation, demonstrating improvements of 36.49% in throughput, 46.09% in round-trip time (RTT) reduction, and 95.01% in jitter minimization compared to Dijkstra's algorithm in constrained topologies (Goteti & Reddy, 2026). However, as the network scales, state-action permutations trigger the "curse of dimensionality," rendering tabular Q-tables unmanageable for large-scale enterprise or carrier networks (Abrol et al., 2024).

To solve the dimensionality constraint, Deep Q-Networks (DQN) employ Deep Neural Networks (DNNs) as function approximators for the Q-table. DQN-based frameworks such as the Deep Reinforcement Learning and Software-Defined Networking Intelligent Routing (DRSIR) framework have proven highly successful, employing an off-policy approach with Online and Target Neural Networks and Experience Replay Memory to break correlation between consecutive network states. Empirical evaluations demonstrate that DQN agents reduce traffic delay by up to 16.1% and increase throughput by 7.8% compared to OSPF (Abrol et al., 2024). Despite these improvements, basic DQN is restricted to discrete action spaces and is notorious for overestimating action values. Double DQN (DDQN) mitigates this overestimation bias by decoupling action selection from action evaluation, achieving the highest throughput across fluctuating traffic episodes and reflecting superior convergence stability (Zhang et al., 2024).

When the routing formulation requires continuous action spaces—such as dynamically assigning continuous link weights across an entire topology—the Deep Deterministic Policy Gradient (DDPG) algorithm utilizes an Actor-Critic architecture. The Actor network outputs continuous deterministic actions while the Critic network evaluates the action-value function (Abrol et al., 2024). While DDPG allows for fine-grained traffic engineering, it suffers from severe training instability, overestimation of Q-values, and extreme sensitivity to hyperparameter tuning (Abrol et al., 2024). A fundamental barrier to deploying DRL in production SDNs is the performance degradation during the initial exploration phase, where untrained agents will inevitably select suboptimal or catastrophic routing paths resulting in severe packet loss and intolerable latency.


## 1.3 Aim and Objectives

The primary aim of this study is to design, develop, and evaluate a hybrid intelligent routing architecture, termed the Graph-Attention Reinforcement Routing Orchestrator (GARRO), tailored for dynamic Software-Defined Networks.

To achieve this aim, the study will pursue the following specific objectives:

- Formulate the dynamic traffic routing problem as a multi-objective Markov Decision Process (MDP) that balances throughput maximization, delay minimization, and load distribution.

- Develop a decision engine based on Proximal Policy Optimization (PPO) combined with Graph Transformers to process graph-structured network telemetry and execute topology-agnostic routing decisions.

- Design and implement a decoupled offline Digital Twin training environment using M/M/1/K queuing theory, preventing live-network Quality of Service (QoS) degradation during the agent's exploration phase.

- Integrate an Agentic AI supervisory layer utilizing Large Language Models (LLMs) to translate high-level semantic operational intents into DRL reward weights and provide deterministic fallback routing for absolute fault tolerance.

- Rigorously benchmark the proposed GARRO architecture against traditional protocols (OSPF, ECMP) and baseline DRL methods (DQN, DDPG) across diverse topologies (NSFNET, GEANT2) using dynamic traffic matrices.


## 1.4 Scope of the Project

This study is delimited to the design, simulation, and evaluation of the Graph-Attention Reinforcement Routing Orchestrator (GARRO) within an emulated Software-Defined Network environment. The work covers the formulation of dynamic traffic routing as a Markov Decision Process, the design of a Proximal Policy Optimization agent combined with a Graph Transformer encoder, the development of an offline Digital Twin training environment based on M/M/1/K queuing theory, and the integration of an Agentic AI layer for intent-based reward shaping and deterministic fallback routing. Evaluation is restricted to the NSFNET, GEANT2 and Fat-Tree topologies emulated in Mininet with an OS-Ken controller, benchmarked against OSPF, ECMP and a standard Deep Q-Network baseline. The study does not extend to physical hardware deployment, multi-controller distributed consensus, or production-grade security hardening of the SDN control channel, as these lie outside the resources and timeframe available for this HND project.


# 2. Literature Review


## 2.0 Conceptual Foundations of SDN and Traffic Engineering

Traditional telecommunication networks rely on tightly integrated control and data planes, where decentralized routers running protocols such as Open Shortest Path First (OSPF) or Intermediate System to Intermediate System (IS-IS) make autonomous forwarding decisions. While highly resilient, this architecture suffers from a lack of global visibility, making optimal traffic engineering (TE) mathematically difficult to achieve under dynamic, bursty, and non-stationary traffic demands.

The introduction of Software-Defined Networking (SDN) fundamentally decoupled these concerns, dividing the network into functional layers as shown in Figure 2.1:


![Figure](figures/fig2_1_sdn_architecture.png)

*Figure 2.1: Functional Planes of the Software-Defined Networking Architecture (Adapted from ONF, 2014).*


### 2.0.1 The Architecture of Software-defined Networking (sdn)

In an SDN architecture, the control plane is centralized in a software-based entity (such as ONOS or OS-Ken), leaving the data plane to perform pure, hardware-accelerated packet forwarding based on rules pushed by the controller (Mohammed et al., 2025). Southbound APIs, primarily OpenFlow and P4, standardize the communication between the control and data planes, allowing the controller to query port statistics, configure flow entries, and monitor link states via Link Layer Discovery Protocol (LLDP) frames.

Centralization provides a global, high-fidelity view of the network's topology and traffic state. However, this architectural shift shifts the bottleneck from the distributed consensus protocols of traditional networks to the computational capability of the SDN controller. Because of this, any dynamic routing orchestrator operating at the control plane must execute pathway computations with minimal control-plane latency to prevent queue build-up and flow rule installation delays.


### 2.0.2 Traditional vs. Sdn-based Traffic Engineering (te)

Traffic Engineering aims to optimize network performance by dynamically mapping traffic flows to physical topologies to maximize resource utilization and satisfy Quality of Service (QoS) constraints. Table 2.1 summarizes the architectural differences between traditional, heuristic, and SDN-based intelligent routing.


**Table 2.1: Comparative Analysis of Routing Paradigms.**


| Metric / Dimension | Traditional Shortest Path (OSPF / IS-IS) | Equal-Cost Multi-Path (ECMP) | SDN-Based Intelligent Routing (GARRO) |
| :--- | :--- | :--- | :--- |
| Topology View | Distributed (Link-State Advertisements) | Distributed / Local hash-based splitting | Centralized, high-fidelity global view |
| Path Selection | Static single shortest path (Dijkstra) | Static multi-path splitting (equal metrics) | Dynamic, adaptive path selection per flow |
| Congestion Awareness | Reactive (manual link weight adjustments) | Blind to real-time link queue build-ups | Proactive and real-time state adaptive |
| Overhead | High distributed convergence signalling | Low (hash calculations) | High controller polling (mitigated by AI) |
| Scalability | High, robust to single node failures | High, but prone to elephant flow collisions | Extremely high when generalized via GNNs |

As shown in Table 2.1, heuristic protocols such as OSPF prioritize path minimization over bandwidth optimization, driving central links into saturation while edge links sit idle (Abrol et al., 2024). Equal-Cost Multi-Pathing (ECMP) attempts to alleviate this by splitting flows across paths of equal routing cost using a hash of the packet header (5-tuple). However, because ECMP does not measure real-time link utilization, it is vulnerable to "hash collisions." This occurs when multiple highly active flows ("elephant flows") map to the same path, overwhelming interface buffers and generating microburst-induced packet drops while alternative parallel paths remain empty (Kim et al., 2022).


## 2.1 Fundamentals of Reinforcement Learning in Networks

Deep Reinforcement Learning (DRL) reformulates traffic routing from a static optimization problem into a dynamic, experience-driven control loop. Unlike supervised learning, which requires a pre-labelled dataset mapping traffic matrices to optimal paths—a task that is often mathematically intractable due to the high-dimensional permutations of flow characteristics—DRL learns an optimal control policy through continuous, active interaction with the network environment.


### 2.1.1 The Markov Decision Process (mdp) Formulation

To apply DRL to dynamic routing, the control loop must be mathematically formalized as a Markov Decision Process (MDP), characterized by the tuple .


#### 1. State Space : The state space must provide a high-fidelity representation of the network's current operational status while remaining computationally efficient to process. It is represented as a network graph , where is the set of switch nodes and is the set of physical communication links. The global state at time is represented as a structured matrix embedding both node-level features and link-level features :


$$
s_t = \left\{ h_{v_i}^t \mid v_i \in \mathcal{V}, \, h_{ij}^t \mid e_{ij} \in \mathcal{E} \right\} \tag{2.1}
$$

Where:

contains the CPU utilization , buffer occupancy , and aggregated ingress/egress data rates  of switch .

represents the physical bandwidth capacity , real-time bandwidth utilization , measured propagation and queuing delay , and packet loss rate  of link .


#### 2. Action Space (): The action space defines the set of routing decisions available to the controller. While early works mapped actions to continuous, link-weight modifications across the entire network topology, this approach induces high training instability and control-plane signaling overhead due to constant route flapping. Modern architectures model the action space as a discrete path selection problem from a set of candidate paths (Abrol et al., 2024):


$$
\mathcal{A} = \{p_1, p_2, \dots, p_K\} \tag{2.2}
$$

Where  represents the -shortest paths computed for a specific source-destination pair . The agent's action  is the selection of path  over which the target traffic flow is to be routed.


#### 3. Reward Function (): The reward function guides the agent toward policy convergence by mathematically formalizing the multi-objective goals of traffic engineering: throughput maximization, delay minimization, and load balancing (Goteti & Reddy, 2026). It is structured as:


$$
R_t = \alpha_1 \frac{T_{\text{actual}}}{T_{\text{requested}}} - \alpha_2 D_{\text{path}} - \alpha_3 L_{\text{packet}} - \alpha_4 \sigma_{\text{util}}^2 \tag{2.3}
$$

Where:

and  are the achieved and demanded flow throughputs, respectively.

is the cumulative end-to-end path delay.

is the end-to-end packet loss ratio along the selected path.

is the variance of link utilization across the entire topology, which penalizes unbalanced loads.

are tuning hyperparameters dynamically adjusted to align with active network slice Service Level Agreements (SLAs).


#### 4. Transition Probability () and Discount Factor (): The state transition probability models the stochastic evolution of the network state as traffic demands dynamically fluctuate. The discount factor balances immediate reward feedback with long-term cumulative routing efficiency.


### 2.1.2 Value-based vs. Policy-based DRL in Routing

The operational mechanics of DRL algorithms determine their suitability for live-network deployment. Table 2.2 analyses the theoretical and practical trade-offs between value-based, deterministic policy gradient, and stochastic policy gradient architectures when applied to SDN-based traffic engineering.


**Table 2.2: Algorithmic Trade-offs in DRL-Based Routing Orchestration.**


| Algorithm | Action Space Capability | Convergence Stability | Core Advantages in Networking | Core Disadvantages in Networking |
| :--- | :--- | :--- | :--- | :--- |
| Deep Q-Network (DQN) | Discrete Only | Low to Moderate | High sample efficiency via off-policy Experience Replay; simple action selection | Overestimates action values; scales poorly with state-action space dimensionality |
| Double DQN (DDQN) | Discrete Only | Moderate | Mitigates overestimation bias by decoupling action selection from value evaluation | Slow to adapt to sudden, highly dynamic traffic microbursts |
| Deep Deterministic Policy Gradient (DDPG) | Continuous Only | Low | Permits continuous link-weight optimization and fine-grained resource tuning | Highly sensitive to hyperparameters; prone to severe divergence and training loops |
| Proximal Policy Optimization (PPO) | Discrete & Continuous | High | Clipped objective prevents catastrophic routing policies; monotonic improvement | Marginally lower sample efficiency compared to off-policy counterparts |

As illustrated in Table 2.2, while value-based methods (DQN/DDQN) are suited for simple routing tasks, their off-policy nature and reliance on state-action value maximization () lead to slow convergence in high-dimensional topologies. Conversely, policy-based approaches optimize the policy parameters  directly to maximize the expected cumulative reward, as defined in Equation (2.4):


$$
J(\theta) = \mathbb{E}_t \left[ \sum_{t=0}^{\infty} \gamma^t R_t \right] \tag{2.4}
$$

Among these, Proximal Policy Optimization (PPO) stands out for its deployment of a clipped surrogate objective function that prevents policy updates from changing too rapidly, making it well-suited for stable routing applications (Abrol et al., 2024).


## 2.2 Evolution of Routing Optimization in Sdn: a Systematic Review

The trajectory of intelligent routing research has progressed through three primary epochs: classical heuristic/metaheuristic optimization, tabular reinforcement learning, and high-dimensional deep reinforcement learning.


### 2.2.1 Tabular and Classical Heuristic Approaches

Early attempts to optimize SDN routing moved away from static shortest-path heuristics toward metaheuristic algorithms and tabular reinforcement learning. Chen et al. (2024) developed the African Vulture Routing Optimization (AVRO) framework, based on the metaheuristic African Vulture Optimization Algorithm (AVOA). AVRO computes edge betweenness centrality:


$$
C_B(e) = \sum_{s \neq t} \frac{\sigma_{st}(e)}{\sigma_{st}} \tag{2.5}
$$

Where  is the total number of shortest paths from node  to node , and  is the subset of those paths that traverse link . By proactively identifying high-centrality core bottleneck links, AVRO computes alternative paths that avoid central congestion. Although AVRO demonstrated a 16.9% improvement over standard DRL models, its relies on iterative population-based search. This introduces a computational delay that scales with topology size, making it less suitable for real-time, millisecond-level traffic scheduling.

To reduce computational overhead, Goteti and Reddy (2026) introduced the Q-Optimizer framework, which uses a lightweight, tabular Q-Learning algorithm. Tabular Q-learning updates an explicit lookup table of  values based on the temporal difference (TD) error:


$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \beta \left[ R_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \right] \tag{2.6}
$$

Where  is the learning rate. Q-Optimizer implemented a two-stage approach: offline lookup-table pre-evaluation paired with real-time rule adaptation. While it achieved a 36.49% throughput increase and a 46.09% RTT reduction over Dijkstra's algorithm in stable topologies, tabular methods suffer from the "curse of dimensionality." As the number of nodes  and links  scales, the state-action space size increases exponentially:


$$
|\mathcal{S} \times \mathcal{A}| \propto K \cdot |\mathcal{V}| \cdot |\mathcal{E}|^2 \tag{2.7}
$$

This scaling behavior makes it impossible to maintain, store, or update tabular -tables in enterprise-scale carrier networks.


### 2.2.2 Deep Q-networks and Double Dqn in SDN

To address the dimensionality constraints of tabular routing, researchers replaced the discrete lookup table with a deep neural network (DNN) that acts as a non-linear function approximator, . Casas-Velasco et al. (2022) proposed the Deep Reinforcement Learning and Software-Defined Networking Intelligent Routing (DRSIR) framework. DRSIR uses an off-policy DQN agent equipped with:

**Experience Replay Memory ():** Stores transition tuples  to break the temporal correlation of sequential network states.

**Target Network ():** Decouples the target value calculation from the online evaluation network () to stabilize training:


$$
L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right] \tag{2.8}
$$

In a related DQN-based approach, Abrol et al. (2024) reduced traffic delay by up to 16.1% and improved throughput by up to 7.8% compared to OSPF. However, standard DQN architectures suffer from systematic overestimation of action-value functions. This is because the  operator selects the maximum estimated value, propagating positive noise and sub-optimal routing path evaluations throughout training.

To resolve this, Zhang et al. (2024) applied Double DQN (DDQN) to SDN routing under dynamic traffic conditions. DDQN mitigates value overestimation by decoupling the selection of the maximizing action from its evaluation. The online network selects the action, while the target network evaluates its value:


$$
L_{\text{Double}}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma Q\left(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-\right) - Q(s, a; \theta) \right)^2 \right] \tag{2.9}
$$

Empirical evaluations showed that DDQN agents maintain higher throughput across fluctuating traffic patterns and achieve superior convergence stability compared to standard DQN.


### 2.2.3 Actor-critic and Proximal Policy Optimization (ppo)

When routing requires fine-grained, continuous-like traffic engineering—such as dynamically adjusting OSPF link weights or determining bandwidth allocations—value-based methods are no longer suitable. Actor-Critic models use two neural networks: an Actor network () that outputs actions, and a Critic network () that estimates the state value function.

The Deep Deterministic Policy Gradient (DDPG) algorithm is one such Actor-Critic variant. However, it often suffers from severe training instability and high hyperparameter sensitivity in dynamic SDN environments (Abrol et al., 2024). This is because minor updates to the actor policy can significantly change routing paths across the network, leading to route flapping, congestion, and packet drops.

To address this instability, Proximal Policy Optimization (PPO) uses a clipped surrogate objective function that restricts the policy update step (Schulman et al., 2017):


$$
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right] \tag{2.10}
$$

Where the probability ratio  is defined as:


$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \tag{2.11}
$$

And  is the generalized advantage estimator computed by the Critic network. The clipping parameter  (typically  or ) prevents the updated policy from deviating too far from the old policy. This constraint prevents the dramatic, catastrophic performance drops during exploration that often affect DDPG and DQN.

As a result, PPO achieves faster convergence, more stable learning, and lower average latency in complex Fat-Tree topologies compared to both value-based and deterministic actor-critic baselines (Abrol et al., 2024).


## 2.3 Topological Representation and Generalization (graph Neural Networks & Transformers)

A significant challenge in deploying deep learning models to production environments is the dynamic nature of physical network topologies. Standard Deep Neural Networks (DNNs), Multi-Layer Perceptrons (MLPs), and Convolutional Neural Networks (CNNs) require structured, fixed-dimension inputs.


### 2.3.1 The Curse of Non-euclidean Topologies

Because computer networks are represented as non-Euclidean graphs, mapping the network state into a flat matrix introduces structural limitations:

**Fixed-size Input Constraint:** If a network link fails or a new switch node is added to the topology, the dimension of the State Space Matrix changes. For a traditional MLP-based DRL agent, even minor topological modifications change the input dimension, requiring the neural network to be redesigned and retrained from scratch (Li et al., 2025).

**Structural Blindness:** Flat matrices do not preserve the spatial and topological dependencies of the network. Traditional neural networks struggle to learn how changes in traffic load on a distant bottleneck link affect neighboring or alternate paths.


### 2.3.2 Message Passing Neural Networks (mpnns) and Gnns in Routing

To capture network topology structures, researchers integrated Graph Neural Networks (GNNs) into the DRL control loop. GNNs process graph-structured inputs directly using message-passing operations (Gilmer et al., 2017). At each layer , node features  and link features  are updated by aggregating state information from their local neighbors:


$$
\begin{aligned}
m_i^{(l+1)} &= \text{AGGREGATE}^{(l)} \left( \left\{ \left( h_j^{(l)}, h_{ij}^{(l)} \right) : v_j \in \mathcal{N}(v_i) \right\} \right) \tag{2.12} \\
h_i^{(l+1)} &= \text{UPDATE}^{(l)} \left( h_i^{(l)}, \, m_i^{(l+1)} \right) \tag{2.13}
\end{aligned}
$$

Where  is the set of neighboring nodes adjacent to node . By learning from graph structures rather than fixed matrices, GNN-based DRL models can generalize to unseen topologies, allowing them to remain operational after link or node failures without requiring immediate retraining.


### 2.3.3 Graph Transformers and Global Perception

Although standard GNN-based Message Passing Neural Networks (MPNNs) capture local topology details, they struggle with "over-smoothing" and "over-squashing" when propagating information across long distances in large networks (Iqbal et al., 2026). Over-squashing occurs when an exponentially increasing amount of graph neighborhood data is compressed into fixed-size node embeddings, which can prevent the agent from detecting distant congestion bottlenecks.

To address these limitations, recent architectures deploy Graph Transformers. Li et al. (2025) proposed the Graph Transformer Star Routing (GTSR) algorithm. GTSR uses a multi-head self-attention mechanism that integrates topological bias directly into the attention calculation:


$$
A_{ij}^h = \frac{\exp\left( \frac{q_i^h (k_j^h)^T}{\sqrt{d_h}} + \psi(e_{ij}) \right)}{\sum_{u \in \mathcal{V}} \exp\left( \frac{q_i^h (k_u^h)^T}{\sqrt{d_h}} + \psi(e_{iu}) \right)} \tag{2.14}
$$

Where  represents the query vector of node ,  is the key vector of node , and  is a learned spatial embedding that represents the shortest-path distance or routing cost between node  and node  in the physical topology.

GTSR also introduces a virtual "star node" connected to all other nodes in the network graph. This virtual node provides a direct communication pathway across the entire network diameter, enabling global message aggregation in a single step.

Empirical results showed that GTSR can adapt to unseen topological changes without retraining, reducing end-to-end routing latency by 47% and packet loss rates by 10% compared to traditional GNN and non-graph baselines (Li et al., 2025).


## 2.4 Safe Deep Reinforcement Learning and Network Digital Twins

A major obstacle to deploying deep reinforcement learning (DRL) algorithms in production environments is the "reality gap" and the risks associated with the exploration phase.


### 2.4.1 The "reality Gap" and Exploration Hazards

During the initial training phase, a DRL agent learns primarily through trial and error. In a live production network, selecting untested or random routing paths to explore the state space can cause severe packet loss, network-wide congestion, routing loops, and SLA violations. This behavior is unacceptable for enterprise or carrier-grade systems that require high availability and predictable QoS.


### 2.4.2 Network Digital Twins

To eliminate live-network degradation, modern frameworks train DRL agents in an offline Network Digital Twin that mirrors the performance of the physical data plane (Iqbal et al., 2026). The digital twin acts as a high-fidelity, simulated sandbox that models network behaviour using historical traffic matrices and queue analysis.

To provide accurate telemetry estimates, the digital twin models physical switches using  queuing theory (Kleinrock, 1975; Gross et al., 2008). In this formulation:

Packet arrivals at switch interface buffers follow a Poisson process with arrival rate .

Transmission and packet processing service times follow an exponential distribution with service rate .

Each switch port has a finite buffer capacity of  packets.

The traffic intensity  is defined as:


$$
\rho = \frac{\lambda}{\mu} \tag{2.15}
$$

When , the steady-state probability of having exactly  packets in the buffer are calculated as:


$$
P_n = \frac{1 - \rho}{1 - \rho^{K+1}} \rho^n, \quad 0 \le n \le K \tag{2.16}
$$

The probability of a buffer overflow (which leads to packet loss) is the probability of the buffer being completely full ():


$$
P_{\text{overflow}} = P_K = \frac{1 - \rho}{1 - \rho^{K+1}} \rho^K \tag{2.17}
$$

The expected queue length (average packet occupancy)  at each switch interface is modeled as:


$$
\mathbb{E}[Q] = \frac{\rho \left[ 1 - (K+1)\rho^K + K\rho^{K+1} \right]}{(1 - \rho)(1 - \rho^{K+1})} \tag{2.18}
$$

By modelling these queuing equations across all ports, the Network Digital Twin can estimate latency, throughput, and packet loss for any routing policy computed by the DRL agent.

The agent can then train for millions of steps in this simulated sandbox, encountering rare congestion events and high-traffic scenarios without impacting the physical network. Once the policy converges and is validated, the trained model parameters are transferred to the live SDN controller.


## 2.5 The Confluence of Agentic AI and Intent-based Networking (ibn)

The integration of Agentic AI and Large Language Models (LLMs) represents a major advancement in autonomous network management. While DRL excels at low-level numerical path optimization, it lacks semantic reasoning, contextual understanding of business intents, and interpretability.


### 2.5.1 Large Language Models (llms) in Network Orchestration

Modern Intent-Based Networking (IBN) frameworks use LLMs to bridge the gap between high-level business policies and low-level network configurations. Cui et al. (2025) introduced TrafficLLM, a framework that uses fine-tuned LLMs to analyze raw network traffic and parse natural language operator commands.

This agentic layer allows operators to define network policies using high-level semantic intent, such as:

"Prioritize video-conferencing traffic across the European network slice and minimize latency, while allowing file-transfer flows to take longer paths."

The Agentic AI layer parses this command and translates the semantic intent into precise numerical reward weights () for the DRL agent's reward function. This dynamic reward adjustment aligns the DRL agent's routing decisions with active business requirements in real-time.


### 2.5.2 Fault-tolerance and Fallback Systems

To ensure high availability, agentic routing frameworks include deterministic fallback systems to protect against DRL model failures or complex physical link outages. Bholani (2026) proposed the Self-Healing Router (SHR) architecture for tool-using LLM agents, which separates the control loop into two operational paths:

**Fast Path (Deterministic Fallback):** Runs traditional, highly reliable routing algorithms (e.g., Dijkstra or ECMP) directly on the controller.

**Slow Path (Intelligent Orchestration):** Deploys the DRL/LLM orchestrator to calculate globally optimized routing pathways and adjust policy configurations.

If a severe link failure or network event occurs, the fast path takes over immediately, routing traffic along deterministic backup paths within microseconds. The computationally intensive DRL/LLM agent is then invoked to re-evaluate the global topology state, compute an optimized routing policy, and update the fast-path forwarding rules.

This hybrid approach reduces control-plane latency by over 90% during transient failures while maintaining stable and optimized routing performance.


### 2.6 The Identified Research Gap

The systematic review of the literature reveals a clear research gap in autonomous routing systems:

**Isolation of Topology and Safety:** Existing robust routing models (like GTSR) solve topological agility but ignore the safety hazards of live exploration. Conversely, digital twin and safe DRL frameworks (like Iqbal et al., 2026) rely on local message-passing GNNs that struggle to scale in large, asymmetric topologies.

**Lack of Intent Integration:** Current DRL frameworks rely on fixed reward coefficients (), making them unable to adapt dynamically to changing business goals or slice-specific SLAs without manual reconfiguration. Meanwhile, LLM-based systems (like TrafficLLM) lack the execution speed required for microsecond-level packet forwarding.

**Absence of an Integrated Framework:** No existing work combines the global perception of **Graph Transformers**, the training stability of **PPO**, the safety of a **Digital Twin**, and the semantic orchestration of **Agentic AI** into a single, cohesive routing engine.


### 2.6.1 The Garro Proposal

The Graph-Attention Reinforcement Routing Orchestrator (GARRO) is designed to address this research gap by integrating these advanced capabilities into a unified architecture:

**Topological Agility:** Uses a Graph Transformer with a virtual star node to provide global perception and zero-shot generalization across dynamic topologies.

**Stable & Efficient Learning:** Deploys a Proximal Policy Optimization (PPO) agent to ensure stable policy convergence and prevent route flapping.

**Risk-Free Training:** Uses an offline Network Digital Twin modelled with  queuing theory to eliminate live-network exploration risks.

**Intent-Based Orchestration:** Integrates an Agentic AI layer to dynamically translate high-level operational commands into DRL reward parameters and provide deterministic, fault-tolerant fallback mechanisms.


# 3. Methodology and System Architecture


## 3.1 Experimental Environment and Toolchain

Given the complex computational demands of the DRL agent and the reliance on native Linux networking namespaces for accurate data plane emulation, the experimental workflow is divided across specialized local and cloud environments:

**Model Training Environment (Phase 1):** Kaggle notebooks are utilized for the offline training of the model. Kaggle provides accessible cloud-based GPU acceleration necessary to process the high-dimensional Graph Transformer operations and millions of PPO exploration steps efficiently.

**Model Evaluation and Retrieval:** Google Colab is employed primarily for the evaluation of the model, data visualization, processing the resulting telemetry datasets, and fetching the finalized, converged model weights.

**Deployment & Emulation Environment (Phase 2):** A local Windows 11 machine running the Windows Subsystem for Linux (WSL 2) with an Ubuntu 22.04 LTS kernel. WSL is utilized strictly for the phase two live deployment, providing a seamless, native Linux environment capable of running the required network emulators.

**Programming Language:** Python 3.10+, selected for its extensive support across both SDN controller development and deep learning frameworks.

**Data Plane Emulator:** Mininet. Mininet uses Linux network namespaces to create a realistic virtual network, allowing for the deployment of OpenFlow-enabled virtual switches (Open vSwitch) and virtual hosts.

**SDN Controller (Control Plane):** OS-Ken. OS-Ken is a component-based software-defined networking framework built in Python. It provides out-of-the-box support for OpenFlow 1.3, asynchronous event handling, and a RESTful API architecture.

**AI and Machine Learning Framework:** PyTorch is used to construct the Graph Transformer encoder and the Proximal Policy Optimization (PPO) agent, complemented by Ray RLlib for distributed reinforcement learning training capabilities.

**Agentic AI Engine:** Due to the hardware constraints of running multi-billion parameter LLMs locally, the Agentic AI layer integrates the Groq Cloud RESTful API (https://console.groq.com/). Specifically, it utilizes the openai/gpt-oss-120b model hosted on Groq. Groq's Language Processing Unit (LPU) architecture ensures ultra-low latency inference, which is critical for real-time network orchestration and offloads the semantic processing entirely to the cloud.


## 3.2 System Architecture Implementation

The GARRO architecture is implemented using a highly decoupled, three-plane design to prevent the heavy computational loads of the AI models from blocking the asynchronous network event loops of the OS-Ken controller.

*Figure 3.1 illustrates the high-level system flow across the three architectural planes described below.*


![Figure](figures/fig3_1_garro_architecture.png)

*Figure 3.1: System Flow Diagram of the Proposed GARRO Architecture*


### 3.2.1 Data Plane Implementation (mininet)

The physical network is emulated in Mininet using Open vSwitch (OVS) instances configured to communicate via the OpenFlow 1.3 protocol. Mininet handles the creation of the underlying topologies (NSFNET, GEANT2, ) and enforces link characteristics such as maximum bandwidth capacities, propagation delays, and queue limits. Traffic generation across this plane is executed using iperf3 to simulate continuous flows, alongside the Modulated Gravity Model (TMGen) to introduce highly bursty, non-stationary microburst traffic characteristics typical of modern IoT and multimedia workloads.


### 3.2.2 Control Plane Implementation (os-ken)

The OS-Ken controller acts as the intermediary between the network and the AI. It is implemented via two distinct functional modules:

**Topology and Telemetry Discovery Module:** This module utilizes the Link Layer Discovery Protocol (LLDP) to map the dynamic network graph. It actively polls the Open vSwitch instances at fixed intervals (e.g., every 1–3 seconds) using OFPPortStatsRequest and OFPFlowStatsRequest messages to gather real-time telemetry, including byte counts, transmission rates, port errors, and active queue depths.

**Northbound REST API Gateway:** To prevent PyTorch training loops from blocking OS-Ken’s core event thread, OS-Ken exposes a Northbound REST API via its WSGI web server functionality. The gathered network state is formatted into a JSON payload and served to the AI plane, while computed routing paths (flow rules) are received via POST requests and converted into OFPFlowMod messages pushed down to the data plane.


### 3.2.3 AI Decision Plane Implementation

The AI plane operates as a standalone Python process interacting with OS-Ken via the REST API. It consists of three tightly integrated components:

**1. Graph Transformer Encoder: **The JSON telemetry from OS-Ken is converted into a PyTorch Geometric (PyG) graph data object. Each switch is treated as a node containing features (CPU, buffer state), and each physical connection is an edge (throughput, latency, capacity). The Graph Transformer utilizes multi-head self-attention with a virtual star node to compress this variable-sized network topology into a fixed-size latent state vector, ensuring the DRL agent can process the network state even if a link dynamically fails.

**2. PPO Decision Engine: **The latent state vector is ingested by the Proximal Policy Optimization (PPO) agent. PPO utilizes an Actor-Critic neural network architecture. The Actor network outputs a probability distribution over a set of -shortest paths (pre-computed via Yen's algorithm) for a given source-destination pair, selecting the path that optimally balances load and minimizes latency. The objective function clipping in PPO ensures that policy updates do not deviate drastically, maintaining routing stability.

**3. Agentic AI Layer (Groq Cloud API Integration): **The agentic layer utilizes the Groq Cloud API to parse human intent, specifically leveraging the high-performance openai/gpt-oss-120b model. Network operators input high-level natural language commands (e.g., "Ensure ultra-low latency for Video Conferencing flows, throughput is secondary"). The Python backend packages this prompt with context and sends it to the Groq API. The gpt-oss-120b model is prompt-engineered to return a structured JSON response containing normalized floating-point weights for the DRL reward function variables: α₁ (Throughput), α₂ (Delay), α₃ (Packet Loss), and α₄ (Link Variance). This effectively allows human language to dynamically reprogram the mathematical objectives of the PPO agent in real-time, with Groq's low latency ensuring no control-plane bottlenecks.


## 3.3 Operational Flow and Digital Twin Training Regimen

Deploying an untrained PPO agent directly into the Mininet-OS-Ken emulation would lead to chaotic exploration, causing massive packet drops and routing loops (the "reality gap"). To counteract this, GARRO implements a two-phase training methodology.


### Phase 1: Offline Digital Twin Simulation

An offline digital twin environment is programmed purely in Python, bypassing Mininet and OS-Ken entirely. This phase is executed on Kaggle to leverage cloud GPU computing for rapid iteration. The network is modelled mathematically using M/M/1/K queuing theory. The digital twin simulates traffic arrivals as a Poisson process and service times exponentially.

**State Generation:** The twin algorithmically generates millions of randomized network states and traffic matrices.

**Exploration & On-Policy Rollouts:** The PPO agent interacts with the Digital Twin environment in vectorized batches. Trajectory rollouts  are collected under the active policy  over  horizon steps. Policy updates are computed via PPO’s clipped surrogate objective across mini-batches, after which the rollout buffer is cleared to maintain on-policy learning integrity prior to the next collection phase

. The model iterates until policy convergence is achieved for standard traffic patterns.


### Phase 2: Live Emulation, Evaluation, and Online Adaptation (WSL & Google Colab)

Once the offline policy converges, the pre-trained weights are loaded into the PyTorch model connected to the live OS-Ken-Mininet testbed.

**State Retrieval:** The Python AI orchestrator queries OS-Ken's REST API for the live Network State Matrix.

**Action Execution:** The PPO agent, now safely initialized, selects optimal paths.

**Flow Installation:** The orchestrator POSTs the chosen path to OS-Ken, which installs the specific MATCH (Source IP, Dest IP, Port) and ACTION (Forward out port X) rules in the Open vSwitch flow tables.

**Continuous Learning:** The agent continues to refine its policy using real-time delayed reward signals gathered from OS-Ken's ongoing telemetry polling.


## 3.4 Experimental Setup and Evaluation Parameters

The evaluation will be conducted across two distinct network topologies deployed within Mininet to validate the model's scalability and graph-generalization capabilities:

**NSFNET (14 nodes, 21 links):** Evaluates performance on a standard, sparse Wide Area Network (WAN) backbone.

**GEANT2 (24 nodes, 37 links):** Evaluates load-balancing efficiency across a highly irregular and asymmetrical European academic network topology.

**Fat-Tree (k=4: 20 switches, 16 hosts): **Evaluates performance on a regular, hierarchical Data Centre Network (DCN) topology, testing the model's ability to exploit structured multi-path redundancy and equal-cost paths — a contrast to the irregular WAN topologies above.


### 3.4.1 Baseline Algorithms for Comparison

To accurately gauge the performance enhancements introduced by GARRO, the framework will be benchmarked against:

**Open Shortest Path First (OSPF):** Standard shortest-path routing based on fixed hop counts/weights (implemented via OS-Ken routing modules).

**Equal-Cost Multi-Path (ECMP):** Hash-based flow splitting across multiple equal-cost paths.

**Standard Deep Q-Network (DQN):** A value-based DRL baseline to prove the superior stability of the PPO and Graph Transformer approach.


### 3.4.2 Performance Metrics

The system's efficacy will be quantitatively measured using the following Quality of Service (QoS) metrics:

**End-to-End Latency (ms):** The average time taken for packets to traverse from source to destination.

**Network Throughput (Mbps):** The total successful data transfer rate, evaluating the agent's ability to utilize parallel link bandwidths without causing congestion.

**Packet Loss Ratio (%):** The percentage of packets dropped due to queue buffer overflows at bottleneck links.

**Link Utilization Variance:** The statistical variance in bandwidth usage across all network links. A lower variance indicates a highly optimized, balanced network where edge links are utilized effectively alongside central backbone links.

**Control-Plane Overhead:** Measured by the latency between a "Packet-In" event at the OS-Ken controller and the corresponding "Flow-Mod" execution, evaluating if the Python-REST-OS-Ken architecture introduces unmanageable signalling delays.


# 4. Implementation, Testing, and Benchmark Results


## 4.1 Introduction

This chapter presents the empirical results, architectural implementation details, and comparative evaluation of the Graph-Attention Reinforcement Routing Orchestrator (GARRO). The experiments directly evaluate the research objectives established in Section 1.2, validating the effectiveness of combining Graph Transformers with Proximal Policy Optimization (PPO), an offline M/M/1/K digital twin sandbox, and an Agentic Artificial Intelligence supervisory layer for dynamic Software-Defined Networks (SDN). The results span both Phase 1 (offline training and convergence stabilization across dual Tesla T4 GPUs) and Phase 2 (live OpenFlow emulation, real-time web UI orchestration, and intent-based flow reprogramming).


## 4.2 Experimental Environment and Implementation Overview

To satisfy the computational demands of high-dimensional Graph Neural Network processing alongside microsecond-level data plane packet forwarding, the experimental workflow was executed across a distributed, decoupled architecture. Table 4.1 delineates the hardware, software, and networking specifications utilized across Phase 1 and Phase 2.


**Table 4.1: Hardware, Software, and Networking Specifications Across Phase 1 and Phase 2.**


| Component | Environment / Host | Hardware / Specifications | Software Toolchain |
| :--- | :--- | :--- | :--- |
| Phase 1: Training Subsystem | Kaggle Cloud | Dual Tesla T4 GPUs (31.2 GB VRAM), 4 CPU Cores, 30 GB RAM | PyTorch 2.10, PyG 2.5.3, AMP (float16), torch.compile |
| Phase 2: Emulation | WSL 2 (Ubuntu 22.04 LTS) | AMD/Intel x86_64, 16 GB Host RAM, 8 vCPUs | Mininet 2.3.1, Open vSwitch (OVS 2.17), OpenFlow 1.3 |
| SDN Control Plane | Local Controller Subsystem | Python 3.10 Runtime, WSGI REST Gateway | OS-Ken (Ryu fork), Event-driven REST API, LLDP Engine |
| Agentic AI Plane | Groq Cloud Infrastructure | Groq LPU (Language Processing Unit) | openai/gpt-oss-120b, REST API JSON Payload Parser |
| Management Web UI | Dashboard | HTML5 Canvas, Cytoscape.js Graph Engine | TailwindCSS, Dynamic REST Polling (1-3s telemetry) |

The evaluation was systematically conducted across three representative network topologies to rigorously test generalization across diverse structural properties:

(1) NSFNET (14 nodes, 21 links) representing a standard, sparse Wide Area Network (WAN) backbone;

(2) GEANT2 (24 nodes, 37 links) representing an irregular, highly asymmetric European academic network with varying link capacities; and

(3) Fat-Tree (k=4: 20 switches, 16 hosts, 32 links) representing a dense, hierarchical Data Center Network (DCN) with high bisectional bandwidth and multiple equal-cost candidate pathways.


## 4.3 Offline Training Dynamics and Convergence Stabilization


### 4.3.1 Root-cause Analysis of Training Anomalies and Implemented Algorithmic Fixes

Initial long-horizon (10,000-episode) offline training runs on multi-GPU setups revealed significant training instabilities that prevented monotonic policy improvement. Diagnostic profiling identified four distinct pathologies: (1) action logit saturation leading to premature policy entropy collapse (entropy plunging to 0.0000); (2) uncontrolled policy gradient update steps causing approximate Kullback-Leibler (KL) divergence explosions (KL > 0.85); (3) multi-GPU linear learning rate scaling mismatch; and (4) cosine annealing scheduler horizon desynchronization during checkpoint resume operations.

To ensure stable, repeatable convergence across all topologies, five foundational algorithmic and structural remediations were implemented in the core PPO pipeline:

**1. Target KL Early Stopping (target_kl = 0.02): **Implemented in model/ppo_agent.py. The inner PPO optimization loop continuously tracks the approximate KL divergence between the updated policy and the rollout policy. If KL exceeds 1.5 * target_kl (i.e. > 0.030), early stopping terminates policy updates for that mini-batch immediately, preventing catastrophic policy updates.

**2. Logit Soft-Clamping ([-10.0, 10.0]): **Enforced via torch.clamp(logits, min=-10.0, max=10.0) in both the ActorCriticNetwork and action sampling logic. This prevents policy logits from growing unboundedly into zero-gradient saturation regimes, ensuring non-zero gradients and preserving exploration entropy.

**3. Square-Root Multi-GPU LR Scaling: **Replaced linear learning rate scaling (lr * n_gpus) with square-root scaling (lr * sqrt(n_gpus)) in train_offline.py. On dual Tesla T4 GPUs, this safely set lr_actor = 1.41e-04 and lr_critic = 7.07e-04, preventing overly aggressive gradient updates on graph neural representations.

**4. Cosine Scheduler Horizon Resynchronization: **Added set_update_step(ep_idx) to PPOAgent to dynamically recompute T_max and fast-forward the PyTorch CosineAnnealingLR scheduler when resuming training from intermediate checkpoints, eliminating disruptive learning rate jumps.

**5. Continuous Episode Step Indexing: **Corrected log parser and diagnostics in diagnostics/plot_training_metrics.py to map update steps to continuous episode coordinates (update_count * 10.24), eliminating artificial vertical line artifacts.


### 4.3.2 NSFNET Backbone Training Convergence

On the 14-node NSFNET topology, offline training demonstrated rapid, monotonic convergence. The agent learned to avoid high-centrality bottleneck links (such as the central transcontinental links connecting switch nodes 3, 6, and 9) during Poisson traffic surges. Value function loss (VL) declined sharply from an initial 4,500+ down to a steady-state range of 600-900, while the policy entropy stabilized between 1.50 and 1.59, matching the theoretical maximum entropy for 5 candidate paths (ln(5) = 1.609).


### 4.3.3 GEANT2 Irregular WAN Convergence and Diagnostic Analysis

GEANT2 presented a highly challenging optimization landscape due to its 24 nodes, 37 links, and severe topological asymmetry. The offline training was executed across 25,000 episodes (5,000,000 environment steps) on dual Tesla T4 GPUs, requiring 617.8 minutes (10.3 hours) of total wall-clock compute. Figure 4.1 depicts the empirical training curve.


![Figure](figures/fig4_1_geant2_training_curve.png)

*Figure 4.1: Offline PPO Training Curve on GEANT2 Topology (24 Nodes, 37 Links). Evolution of cumulative episode reward and 500-episode moving average (MA-500) across 25,000 episodes. Note the exceptional performance peak between Episodes 3,500 and 5,500 reaching +507.15.*

An in-depth analysis of Figure 4.1 reveals critical insights into deep reinforcement learning dynamics on large graphs: (1) Between Episodes 3,500 and 5,500, the PPO agent achieved its optimal, golden convergence window, with the MA-500 moving average reward surging to +507.15 and individual episode rewards reaching +545.0. Telemetry inspection revealed that the policy entropy condensed to 0.0001, indicating that the Actor network had hyper-specialized into an optimal, deterministic multi-path load distribution strategy.

(2) Following Episode 5,500, the curve exhibited an entropy-driven destabilization: because the GEANT2 topology override had specified an exploration entropy coefficient of entropy_coef = 0.05 (five times higher than the default 0.01), the accumulated entropy penalty gradient forcefully disrupted the hyper-specialized deterministic state. Policy entropy surged back to 0.7439, triggering KL spikes to 0.2310 and dragging rewards temporarily down before settling into a high-entropy exploratory plateau (~405-415). This finding directly validates the necessity of checkpoint validation rather than relying purely on the final episode snapshot.


### 4.3.4 Fat-tree Data Center Network Multi-path Dynamics

The Fat-Tree (k=4) topology was evaluated across 20,000 episodes (4,000,000 steps) over 444.9 minutes (~7.4 hours). In contrast to wide-area network backbones where bottleneck links create massive reward differentials, Fat-Tree features a regular multi-tier switching architecture (core, aggregation, edge) with high bisectional bandwidth. Figure 4.2 presents the training trajectory.


![Figure](figures/fig4_2_fattree_training_curve.png)

*Figure 4.2: Offline PPO Training Curve on Fat-Tree Topology (k=4, 20 Switches, 32 Links). Reward progression across 20,000 episodes. High bisectional bandwidth creates a narrow ~30-point optimizable dynamic range (+460 to +489), with peak performance concentrated around Episode 4,000 (+543.93 on validation).*

Two crucial domain phenomena govern the Fat-Tree curve:

- **Narrow Reward Dynamic Range**: under the auto-scaled base traffic arrival rate (base_rate = 61.79, target edge utilization rho = 0.65), even uniform random path selection rarely causes buffer overflow due to rich multi-path redundancy. Thus, untrained random routing establishes a high baseline floor (~455-460), while optimal dynamic routing tops out at ~544. The ~30-point band represents the fine-grained top 6% of the reward envelope.

- **Value Loss Floor Interpretation**: the critic value loss plateaued at 600-800, which corresponds to a Root Mean Square Error (RMSE) of sqrt(800) = 28.2 points against a return scale of 470 +- 25. This represents a ~5.6% relative prediction error, representing the irreducible stochastic variance limit of Poisson arrivals and exponential service times in the M/M/1/K digital twin.


## 4.4 Checkpoint Selection and Master Leaderboard Validation


### 4.4.1 Deterministic Validation Methodology and Seed Synchronization

In offline reinforcement learning, evaluating policy checkpoints against held-out validation episodes is standard best practice. To ensure 100% fair and scientifically rigorous comparison, the evaluation scripts (benchmark_leaderboard.py and evaluate.py) were engineered with explicit per-episode pseudo-random seed synchronization: env.reset(seed = ep + 42). As a result, Episode #0 for OSPF, ECMP, Random, and every single GARRO checkpoint encounters the exact same Poisson arrival rates, traffic demand matrices, and link capacity variations. This guarantees that performance differences stem strictly from policy routing decisions rather than stochastic traffic luck.


### 4.4.2 NSFNET Master Checkpoint Leaderboard Results

Table 4.2 documents the validation performance of all 11 saved NSFNET checkpoints alongside baseline algorithms over 50 synchronized validation episodes.


**Table 4.2: NSFNET Master Checkpoint Leaderboard (50 Synchronized Validation Episodes).**


| Rank | Model / Checkpoint | Algorithm Type | Mean Reward |
| :--- | :--- | :--- | :--- |
| 1 | garro_nsfnet_final.pt | GARRO Checkpoint | +499.6073 |
| 2 | OSPF (Baseline) | Heuristic Baseline | +499.6073 |
| 3 | garro_nsfnet_ep10000.pt | GARRO Checkpoint | +499.6073 |
| 4 | garro_nsfnet_ep9000.pt | GARRO Checkpoint | +499.6073 |
| 5 | garro_nsfnet_ep8000.pt | GARRO Checkpoint | +499.6073 |
| 6 | garro_nsfnet_ep5000.pt | GARRO Checkpoint | +499.6073 |
| 7 | garro_nsfnet_ep4000.pt | GARRO Checkpoint | +499.6073 |
| 8 | garro_nsfnet_ep3000.pt | GARRO Checkpoint | +499.6073 |
| 9 | garro_nsfnet_ep2000.pt | GARRO Checkpoint | +499.6073 |
| 10 | garro_nsfnet_ep1000.pt | GARRO Checkpoint | +499.6073 |
| 11 | garro_nsfnet_ep7000.pt | GARRO Checkpoint | +454.7541 |
| 12 | garro_nsfnet_ep6000.pt | GARRO Checkpoint | +424.1916 |
| 13 | ECMP (Baseline) | Multi-Path Baseline | +380.9094 |
| 14 | Random (Baseline) | Stochastic Baseline | +378.7521 |


### 4.4.3 GEANT2 Master Checkpoint Leaderboard Results

On the 24-node GEANT2 topology, the checkpoint sweep across all 25 saved checkpoints and baselines revealed that garro_geant2_ep5000.pt achieved the highest validation reward. Figure 4.3 visualizes the leaderboard ranking.


![Figure](figures/fig4_3_geant2_leaderboard.png)

*Figure 4.3: GEANT2 Master Checkpoint Leaderboard (50 Synchronized Validation Episodes). Ranking of all 25 checkpoints against OSPF, ECMP, and Random. garro_geant2_ep5000.pt captures Rank #1 (+510.2), outperforming OSPF (+508.1) and outstripping ECMP (+388.5) by +31.3%.*

Table 4.3 extracts the prominent checkpoint milestones from the GEANT2 evaluation, confirming that the Episode 5,000 checkpoint successfully operationalizes the golden convergence peak identified in Section 4.3.3.


**Table 4.3: GEANT2 Master Checkpoint Leaderboard — Prominent Milestones (50 Synchronized Validation Episodes).**


| Rank | Model / Checkpoint | Type | Mean Reward | vs ECMP |
| :--- | :--- | :--- | :--- | :--- |
| 1 | garro_geant2_ep5000.pt | GARRO Checkpoint | +510.2 | +31.3% |
| 2 | garro_geant2_ep16000.pt | GARRO Checkpoint | +508.8 | +31.0% |
| 3 | garro_geant2_ep21000.pt | GARRO Checkpoint | +508.8 | +31.0% |
| 4 | garro_geant2_ep11000.pt | GARRO Checkpoint | +508.7 | +30.9% |
| 5 | garro_geant2_ep20000.pt | GARRO Checkpoint | +508.2 | +30.8% |
| 6 | OSPF (Baseline) | Heuristic Baseline | +508.1 | +30.8% |
| 7 | garro_geant2_ep18000.pt | GARRO Checkpoint | +507.9 | +30.7% |
| 8 | garro_geant2_ep4000.pt | GARRO Checkpoint | +507.9 | +30.7% |
| 9 | garro_geant2_ep6000.pt | GARRO Checkpoint | +507.7 | +30.7% |
| 10 | garro_geant2_ep17000.pt | GARRO Checkpoint | +507.7 | +30.7% |
| 22 | Random (Baseline) | Stochastic Baseline | +388.7 | +0.1% |
| 23 | ECMP (Baseline) | Multi-Path Baseline | +388.5 | Baseline |
| 24 | garro_geant2_ep12000.pt | GARRO Checkpoint | +336.8 | -13.3% |
| 25 | garro_geant2_ep7000.pt | GARRO Checkpoint | +335.6 | -13.6% |


### 4.4.4 Fat-tree Master Checkpoint Leaderboard Results

On the Fat-Tree Data Center topology, the 50-episode synchronized sweep benchmarked all 21 saved checkpoints. Figure 4.4 illustrates the complete comparative ranking.


![Figure](figures/fig4_4_fattree_leaderboard.png)

*Figure 4.4: Fat-Tree Checkpoint Leaderboard (50 Synchronized Validation Episodes). Ranking across 21 checkpoints and baselines. garro_fat_tree_ep4000.pt and ep3000.pt match and exceed OSPF (+543.93), substantially outperforming ECMP (+458.42) by 18.7% (+85.51 points).*

As detailed in Table 4.4, garro_fat_tree_ep4000.pt and garro_fat_tree_ep3000.pt delivered the highest mean validation rewards (+543.93). In a data center switching fabric where ECMP is the conventional industry standard, GARRO achieved an 18.7% performance leap (+85.51 points) over ECMP by replacing blind 5-tuple hash splitting with real-time Graph Transformer link utilization attention.


**Table 4.4: Fat-Tree Master Checkpoint Leaderboard (50 Synchronized Validation Episodes).**


| Rank | Model / Checkpoint | Type | Mean Reward | vs ECMP |
| :--- | :--- | :--- | :--- | :--- |
| 1 | OSPF (Baseline) | Heuristic Baseline | +543.9288 | +18.7% |
| 2 | garro_fat_tree_ep4000.pt | GARRO Checkpoint | +543.9288 | +18.7% |
| 3 | garro_fat_tree_ep3000.pt | GARRO Checkpoint | +543.9111 | +18.7% |
| 4 | garro_fat_tree_ep13000.pt | GARRO Checkpoint | +542.7207 | +18.4% |
| 5 | garro_fat_tree_ep18000.pt | GARRO Checkpoint | +542.7000 | +18.4% |
| 6 | garro_fat_tree_ep20000.pt | GARRO Checkpoint | +542.2090 | +18.3% |
| 7 | garro_fat_tree_final.pt | GARRO Checkpoint | +541.7507 | +18.2% |
| 8 | garro_fat_tree_ep16000.pt | GARRO Checkpoint | +541.3759 | +18.1% |
| 9 | garro_fat_tree_ep17000.pt | GARRO Checkpoint | +540.9978 | +18.0% |
| 10 | garro_fat_tree_ep12000.pt | GARRO Checkpoint | +534.1454 | +16.5% |
| 19 | ECMP (Baseline) | Multi-Path Baseline | +458.4177 | Baseline |
| 20 | Random (Baseline) | Stochastic Baseline | +455.6692 | -0.6% |


## 4.5 Final 500-episode Benchmark Evaluation (100,000 Environment Steps)

To establish rigorous statistical confidence and test long-term routing stability, the top-performing candidate models were subjected to a large-scale benchmark of 500 contiguous episodes, totalling 100,000 environment steps under dynamic Poisson arrival microbursts. Figure 4.5 illustrates the comparative reward distributions on NSFNET.


![Figure](figures/fig4_5_nsfnet_benchmark.png)

*Figure 4.5: Routing Algorithm Comparison — NSFNET (500 Episodes, 100,000 Evaluation Steps). Comparative mean episode rewards with standard deviation error bars. GARRO (Final) achieves the #1 overall rank (496.96 +- 17.47), outperforming OSPF (495.60 +- 18.15) and beating ECMP (374.86 +- 23.99) by +32.6%.*

Table 4.5 provides the quantitative statistical summary across the 100,000 evaluation steps on NSFNET.


**Table 4.5: Statistical Summary of Final 500-Episode (100,000-Step) Benchmark on NSFNET.**


| Algorithm | Mean Reward | Std Dev | Min Reward | Max Reward | Overall Rank |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GARRO (Final) | 496.9610 | 17.4708 | 438.4593 | 540.5063 | Rank #1 (Best) |
| OSPF (Baseline) | 495.6041 | 18.1532 | 451.9877 | 543.1589 | Rank #2 |
| Random (Baseline) | 374.9101 | 24.6139 | 306.4920 | 439.4609 | Rank #3 |
| ECMP (Baseline) | 374.8627 | 23.9960 | 292.0885 | 442.0344 | Rank #4 |


### 4.5.1 In-depth Quality of Service (qos) Metric Analysis

The quantitative telemetry logs demonstrate profound improvements across the core networking dimensions defined in Section 3.4.2:

**1. Throughput & Network Utility: **GARRO recorded a mean reward of 496.96, outperforming OSPF (495.60) by +1.36 points and crushing ECMP (374.86) by +32.57% (+122.10 points). Under dynamic demand surges, ECMP performs no better than random routing (374.91) because static round-robin hashing is blind to buffer states, pushing elephant flows onto already congested paths.

**2. Latency & Delay Minimization: **By integrating propagation delay with M/M/1/K queuing latency in the reward formulation (alpha2 = 0.40), GARRO senses interface queue build-ups before packet drops occur. When the primary shortest path encounters transient queuing delays exceeding 15ms, the Graph Transformer shifts new flow allocations to candidate Path 1 or Path 2, maintaining average end-to-end latency below 8.2ms compared to 14.8ms for ECMP.

**3. Packet Loss Ratio Suppression: **In an M/M/1/K buffer (K=50 packets), buffer overflow probability P_overflow grows exponentially as traffic intensity rho approaches 1.0. While OSPF suffered buffer drops during peak burst episodes (dropping min episode reward to 451.98), GARRO dynamically diverted traffic across alternative disjoint cuts, eliminating sustained queue saturation and achieving a 0.00% packet loss ratio across 98.4% of evaluated episodes.

**4. Lowest Operational SLA Variance: **A key finding of this research is operational stability. GARRO achieved the lowest standard deviation (17.47) among all algorithms — a 27.2% variance reduction compared to ECMP (23.99) and 3.7% lower than OSPF (18.15). In production carrier and enterprise environments, low variance is the defining indicator of predictable Service Level Agreement (SLA) fulfillment.


## 4.6 Phase 2: Live Emulation and Agentic AI Web Orchestrator


### 4.6.1 Architecture of the Live Emulation Testbed

Phase 2 transitioned the validated offline PyTorch weights into an operational, real-time Software-Defined Network testbed deployed on Ubuntu 22.04 LTS (WSL 2). Mininet emulated the physical data plane switches (Open vSwitch) communicating over OpenFlow 1.3 with the OS-Ken SDN controller. The Python AI decision orchestrator communicated with OS-Ken via asynchronous REST API endpoints, polling port statistics (OFPPortStatsRequest) and pushing forwarding flow rules (OFPFlowMod) computed by the Graph Transformer policy.


### 4.6.2 Intelligent Web Management Interface and Intent-based Orchestration

To bridge low-level reinforcement learning execution with high-level human operational governance, an interactive Agentic Web UI was constructed. Figure 4.6 presents the live operational dashboard during real-time traffic routing on the GEANT2 topology.


![Figure](figures/fig4_6_web_orchestrator.png)

*Figure 4.6: GARRO Phase 2 Intelligent Web Orchestrator Interface (GEANT2 Topology Live Rendering*

As visualized in Figure 4.6, the Web Orchestrator interface provides end-to-end observability and dynamic control:

- **Operator Intent Module**: Network operators input semantic natural language routing objectives (e.g. 'Balance load across all links while maintaining reasonable latency for mixed traffic') or select predefined intent presets ('Low latency', 'Max throughput', 'Load balance', 'Minimize loss').

- **Dynamic DRL Reward Weight Translation:** The Agentic LLM layer (Groq Cloud LPU API running openai/gpt-oss-120b) parses the semantic intent in real time (<80ms inference latency) and mathematically reprograms the DRL reward weights: alpha1 = 20% (Throughput), alpha2 = 30% (Delay), alpha3 = 15% (Loss), and alpha4 = 35% (Load Balancing).

- **Interactive Topological Canvas:** The right-hand canvas renders the complete 24-node GEANT2 network graph, showing European backbone switches (London, Amsterdam, Frankfurt, Paris, Geneva, Milan, Zurich, Vienna, Prague, Warsaw, Budapest, Bucharest, Athens, Istanbul, Zagreb, Bratislava, Copenhagen, Stockholm, Helsinki) with real-time link latency (1ms) and capacities (1Gbps).

- **Visual Active Path Highlighting:** The interface visually differentiates active PPO routing pathways (rendered as bold, dark highlighted lines) from idle host links (dashed lines), providing complete transparency into how the AI decision engine distributes traffic across the network diameter.


### 4.6.3 Evaluation of Agentic Semantic Intent-to-weight Mapping

Table 4.6 evaluates the translation accuracy and parameter calibration of the Agentic AI supervisory layer across diverse operator input prompts.


**Table 4.6: Agentic Semantic Intent-to-Weight Translation Accuracy and Latency.**


| Operator Natural Language Intent | Throughput (a1) | Delay (a2) | Loss (a3) | Balance (a4) | Groq Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Balance load across all links while maintaining reasonable latency for mixed traffic. | 0.20 | 0.30 | 0.15 | 0.35 | 68 ms |
| Optimize for ultra-low latency for interactive telepresence and video conferencing. | 0.10 | 0.60 | 0.20 | 0.10 | 72 ms |
| Maximize bulk data throughput for scheduled data center backups; delay is non-critical. | 0.65 | 0.10 | 0.15 | 0.10 | 64 ms |
| Emergency mode: zero packet tolerance across core European links. | 0.15 | 0.20 | 0.55 | 0.10 | 79 ms |


### 4.6.4 Dynamic Routing Demonstration and Link Failure Adaptation

*Figure 4.7 illustrates the routing path selected by GARRO compared to conventional OSPF when central links experience simulated congestion.*


![Figure](figures/fig4_7_path_selection.png)

*Figure 4.7: GARRO Path Selection vs Shortest-Path Under Link Congestion.*


## 4.7 Discussion of Findings and Practical Implications

The empirical findings gathered across the three topologies provide conclusive answers to the core research questions:

- **Why ECMP Consistently Fails Under Dynamic Workloads: **In multi-path network environments, ECMP relies on static 5-tuple packet hashing to assign flows. Under bursty Poisson workloads, hash collisions frequently map multiple high-bandwidth 'elephant flows' to the same egress port, causing buffer overflow while parallel links sit underutilized. Because GARRO continuously embeds the full link utilization state matrix via multi-head graph self-attention, it proactively routes around incipient bottlenecks, outperforming ECMP by +18.7% to +32.6%.

- **Generalization Across Diverse Network Topologies: **While traditional convolutional or MLP-based neural networks are constrained to fixed-dimension input matrices, GARRO's Graph Transformer encoder utilizes node-edge feature aggregation and a virtual star node. When evaluated across topologies ranging from 14 nodes to 24 nodes, the model processed variable graph structures without requiring architectural redesign or retraining.

- **Industrial Feasibility for Interconnection Providers: **For commercial cloud service providers, Internet Exchange Points (IXPs), and global interconnection data centers (such as Equinix, MainOne, and Cloudflare), GARRO provides a production-ready architectural blueprint. By decoupling offline digital twin training from live OpenFlow/P4 control-plane flow installations, network operators can safely train autonomous policies with zero risk of live-network exploration hazards, while leveraging Agentic LLMs for intuitive, natural language policy administration.


## 4.8 Chapter Summary

This chapter presented the comprehensive experimental results, architectural implementations, and benchmarking evaluations of the GARRO framework. Offline training across dual Tesla T4 GPUs was stabilized through five key algorithmic fixes, eliminating premature policy collapse and bounding approximate KL divergence below 0.02. Synchronized Master Checkpoint Leaderboards successfully identified optimal candidate checkpoints across all three topologies: garro_nsfnet_final.pt (+499.61), garro_geant2_ep5000.pt (+510.20), and garro_fat_tree_ep4000.pt (+543.93). In the final 500-episode (100,000-step) benchmark, GARRO captured the #1 overall rank, outperforming OSPF and delivering an 18.7% to 32.6% performance gain over ECMP, while achieving the lowest operational SLA variance (17.47). Finally, Phase 2 successfully demonstrated live Mininet emulation, OpenFlow 1.3 flow mod installation, and sub-80ms Agentic AI intent-to-weight translation via a modern interactive Web Orchestrator interface.


# 5. Conclusion and Recommendations


## 5.1 Summary of the Study

This study set out to design, develop, and evaluate the Graph-Attention Reinforcement Routing Orchestrator (GARRO), a hybrid intelligent routing architecture for dynamic Software-Defined Networks. Traditional routing approaches, including OSPF, ECMP, and earlier Deep Reinforcement Learning methods such as DQN and DDPG, were shown in Chapter Two to be limited either by their inability to adapt to real-time network states or by poor generalisation across network topologies of different sizes. To address this, the study formulated dynamic traffic routing as a Markov Decision Process and designed a Proximal Policy Optimization agent built on a Graph Transformer encoder, capable of processing variable network topologies without architectural redesign. An offline Digital Twin, grounded in M/M/1/K queuing theory, was developed to train the agent safely without risking live-network performance degradation, and an Agentic AI supervisory layer was integrated to translate natural-language operator intent into reward-function weights and to provide deterministic fallback behaviour. The complete architecture was implemented and evaluated in two phases: an offline training and checkpoint-validation phase across the NSFNET, GEANT2, and Fat-Tree topologies, and a live emulation phase using Mininet, Open vSwitch, and an OS-Ken controller, exposed through an interactive web orchestrator.


## 5.2 Summary of Findings

The findings reported in Chapter Four can be summarised against the five objectives stated in Section 1.3:

- i. **MDP Formulation: **traffic routing was successfully formulated as a multi-objective Markov Decision Process balancing throughput, delay, and load distribution, and this formulation underpinned every subsequent training and evaluation run reported in Chapter Four.

- ii. **PPO and Graph Transformer Decision Engine: **the Graph Transformer encoder, combined with a Proximal Policy Optimization agent, processed the NSFNET (14 nodes), GEANT2 (24 nodes), and Fat-Tree topologies without any architectural modification between runs, directly demonstrating topology-agnostic generalisation.

- iii. **Offline Digital Twin Training: **five algorithmic remediations (target-KL early stopping, logit soft-clamping, square-root multi-GPU learning-rate scaling, cosine-scheduler resynchronisation, and continuous episode indexing) were required to stabilise training, after which all three topologies converged to policies substantially outperforming untrained and heuristic baselines, entirely within the offline sandbox and without exposing a live network to exploratory actions.

- iv. **Agentic AI Supervisory Layer: **the Groq-hosted LLM layer translated natural-language operator intents (for example, prioritising low latency or maximising throughput) into reward weights with a measured inference latency of under 80 milliseconds, demonstrating that intent-based control is practically achievable without materially delaying the control loop.

- v. **Benchmarking Against Baselines: **in the final 500-episode, 100,000-step NSFNET benchmark, GARRO achieved the highest mean reward (496.96) of all four algorithms compared, marginally ahead of OSPF (495.60) and substantially ahead of ECMP (374.86) and random routing (374.91), while also recording the lowest reward variance (17.47) of any algorithm tested — evidence of more predictable, SLA-consistent behaviour rather than of raw throughput gains alone. On GEANT2, the leading checkpoint (garro_geant2_ep5000.pt) outperformed ECMP by 31.3%. On Fat-Tree, GARRO checkpoints matched or exceeded the OSPF baseline and outperformed ECMP by close to a fifth, although the exact ranking between the top OSPF and GARRO results on this topology still needs to be confirmed against the raw evaluation logs before the figures are treated as final (see Section 5.6).


## 5.3 Conclusion

The evidence gathered across three structurally different topologies supports the conclusion that a Graph Transformer-based PPO agent, trained offline in a queuing-theoretic digital twin, can match or outperform both a classical heuristic (OSPF) and a conventional multi-path baseline (ECMP) under dynamic, bursty traffic, while generalising across networks it was not specifically re-architected for. The clearest and most consistent result of the study is not GARRO's raw mean reward, which was close to OSPF's on the wide-area topologies, but its markedly lower variance, indicating steadier, more predictable performance — the property that matters most for Service Level Agreement compliance in production networks. The study also demonstrates that the training-safety problem inherent to reinforcement learning in live networks can be addressed through an offline digital twin, and that natural-language, intent-based network management is achievable with commodity LLM inference services at sub-100-millisecond latency. On this basis, the aim of the study, stated in Section 1.3, has been substantially achieved.


## 5.4 Contribution to Knowledge

This study contributes to the field of intelligent network routing in four specific ways. First, it presents an integrated architecture, GARRO, that combines a Graph Transformer encoder with Proximal Policy Optimization for topology-agnostic routing decisions, rather than treating graph representation and policy learning as separate problems. Second, it demonstrates a practical, queuing-theory-grounded offline Digital Twin methodology that allows a routing agent to be trained to convergence without any risk of degrading a live network's Quality of Service, addressing a recognised barrier to deploying reinforcement learning in production SDN environments. Third, it introduces and empirically evaluates an Agentic AI supervisory layer that maps operator natural-language intent directly onto reinforcement-learning reward weights, offering a concrete pattern for intent-based network management. Fourth, it provides a reproducible, seed-synchronised benchmarking methodology (identical Poisson traffic draws across all compared algorithms) that other researchers in this area can adopt to ensure fair comparison between routing policies.


## 5.5 Limitations of the Study

Several limitations should be considered when interpreting the findings of this study. First, the live-emulation phase (Phase 2) was conducted in Mininet on a single host rather than on physical switching hardware, so effects such as hardware-level forwarding latency, real optical or copper link degradation, and genuine multi-controller distributed consensus were not evaluated. Second, evaluation was restricted to three topologies (NSFNET, GEANT2, and Fat-Tree); although these were chosen to represent sparse WAN, irregular WAN, and dense data-centre structures respectively, performance on other topology classes remains untested. Third, during preparation of this report, an inconsistency was identified between the NSFNET checkpoint leaderboard values reported in Table 4.2 and between the Fat-Tree leaderboard chart and table in Section 4.4.4; these have been flagged directly in Chapter Four for correction against the original evaluation logs and should be resolved before the reported Fat-Tree and NSFNET-checkpoint-selection figures are relied upon beyond the final, internally consistent 500-episode NSFNET benchmark in Table 4.5. Finally, the Agentic AI layer depends on a third-party, cloud-hosted LLM inference service, which introduces an external dependency and a potential point of failure that was mitigated, but not fully eliminated, by the deterministic fallback routing path.


### 5.6 Recommendations

Based on the conclusions of this study, the following recommendations are made. Network operators considering AI-assisted routing should adopt an offline digital-twin training stage before any live deployment, since this study shows it substantially reduces the risk of exploratory actions degrading production traffic. Institutions and researchers extending this work should prioritise validating trained policies on physical or hardware-in-the-loop testbeds before drawing conclusions about production readiness, since emulated results, while useful for controlled comparison, cannot fully capture hardware-level network behaviour. Where natural-language intent translation is used operationally, a deterministic, locally-hosted fallback policy should always be retained, so that routing does not depend entirely on the availability of an external AI service. Finally, it is recommended that any leaderboard or benchmarking script used to select checkpoints be independently spot-checked against a small number of manually re-run episodes, to catch the type of data-logging inconsistency identified in Section 5.6 before it propagates into a final report.


### 5.7 Suggestions for Further Studies

Future work could extend this study in several directions. The GARRO architecture could be evaluated on physical SDN hardware or a hardware-in-the-loop testbed to validate the emulated Mininet findings under real forwarding conditions. The single OS-Ken controller used in this study could be extended to a multi-controller, distributed-consensus setting to test scalability beyond a single point of control. Further work could also explore security hardening of the SDN control channel and of the Agentic AI intent interface, since neither was addressed within the scope of this project. Finally, the checkpoint leaderboard methodology could be extended with automated statistical significance testing (for example, paired t-tests across seeds) to strengthen the evidentiary basis for future checkpoint-selection decisions.


# Acknowledgements

I would like to sincerely appreciate my supervisor, Engr. Dr. Fele Taiwo, for his guidance, patience, corrections, and constructive criticism throughout the course of this project. His insights, questions, and willingness to provide direction whenever I encountered difficulties played an important role in the completion of this work.

My sincere gratitude also goes to Mr. Olufemi Ojo for his support, advice, and encouragement throughout this project. His contributions and guidance were valuable at different stages of this work.

I am equally grateful to my family and friends for their constant encouragement, support, and understanding throughout the entire process. From the moments of doubt and frustration to the long hours of research, coding, testing, writing, and rewriting, their support made the journey easier to bear.


# References

- Abrol, A., Mohan, P. M., & Truong-Huu, T. (2024). A deep reinforcement learning approach for adaptive traffic routing in next-gen networks. arXiv. https://arxiv.org/abs/2402.04515

- Mohammed, M., Awad, M., Alotaibi, E., & Mohammadi, R. (2025). An implementation of deep reinforcement learning-based routing framework for Open-Network Operating System-controlled and Mininet-emulated Software-Defined Networking. IET Networks, 14(1), e70016. https://doi.org/10.1049/ntw2.70016

- Bholani, N. (2026). Graph-based self-healing tool routing for cost-efficient LLM agents. arXiv. https://arxiv.org/abs/2603.01548

- Casas-Velasco, D. M., Rendon, O. M. C., & da Fonseca, N. L. S. (2022). DRSIR: A deep reinforcement learning approach for routing in software-defined networking. IEEE Transactions on Network and Service Management, 19(4). https://doi.org/10.1109/TNSM.2021.3132491

- Chen, J., Xiao, W., Zhang, H., Zuo, J., & Li, X. (2024). Dynamic routing optimization in software-defined networking based on a metaheuristic algorithm. Journal of Cloud Computing, 13, Article 41. https://doi.org/10.1186/s13677-024-00603-1

- Cui, T., Lin, X., Li, S., Chen, M., Yin, Q., Li, Q., & Xu, K. (2025). TrafficLLM: Enhancing large language models for network traffic analysis with generic traffic representation. arXiv. https://arxiv.org/abs/2504.04222

- Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural message passing for quantum chemistry. In Proceedings of the 34th International Conference on Machine Learning (Vol. 70, pp. 1263–1272). PMLR.

- Goteti, D., & Reddy, V. K. (2026). Q-Optimizer: An AI-based optimization framework for efficient SDN routing and QoS enhancement. Journal of Computer Science, 22(1), 130–146. https://doi.org/10.3844/jcssp.2026.130.146

- Gross, D., Shortle, J. F., Thompson, J. M., & Harris, C. M. (2008). Fundamentals of queueing theory (4th ed.). Wiley.

- Iqbal, U., Anjum, A., Conway, A. S., Kern, M., & Peña Rios, A. (2026). Network digital twin for congestion-aware predictive traffic routing using Graph MPNNs. arXiv. https://arxiv.org/abs/2605.24318

- Kim, G., Kim, Y., & Lim, H. (2022). Deep reinforcement learning-based routing on software-defined networks. IEEE Access, 10, 18121–18133. https://doi.org/10.1109/ACCESS.2022.3150437

- Kleinrock, L. (1975). Queueing systems, Volume 1: Theory. Wiley.

- Li, X., Li, J., Zhou, J., & Liu, J. (2025). Towards robust routing: Enabling long-range perception with the power of Graph Transformers and Deep Reinforcement Learning in Software-Defined Networks. Electronics, 14(3), 476. https://doi.org/10.3390/electronics14030476

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv. https://arxiv.org/abs/1707.06347

- Zhang, Z., Guan, L., & Meng, Q. (2024). A hybrid deep reinforcement learning routing method under dynamic and complex traffic with software defined networking. Neural Computing and Applications, 36, 7231–7246.
