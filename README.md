# GARRO: Graph Attention Routing with Reinforcement Learning (Phase 1)

GARRO is a Deep Reinforcement Learning (DRL) routing framework designed for Software-Defined Networks (SDN). It combines a **Graph Attention Transformer (GAT)** to capture complex network topologies and traffic states with **Proximal Policy Optimization (PPO)** to dynamically route traffic requests. 

This repository implements **Phase 1: Offline Digital Twin Training**, allowing the agent to train and benchmark entirely within an analytical M/M/1/K queueing simulation before online deployment.

---

## 🏗️ Project Architecture

The project is structured logically into distinct layers:

```
schproject/
├── config.yaml                    # Centralized hyperparameter and environment config
├── train_offline.py               # CLI entrypoint for training in the Digital Twin
├── evaluate.py                    # CLI entrypoint for baseline benchmarking
├── digital_twin.py                # Standalone M/M/1/K analytical queue validator
│
├── topologies/                    # Network topologies with custom telemetries
│   ├── __init__.py
│   ├── nsfnet.py                  # NSFNET: 14 nodes, 21 edges
│   ├── geant2.py                  # GEANT2: 24 nodes, 37 edges
│   └── fat_tree.py                # Fat-Tree (k=8): 80 nodes, 256 edges
│
├── digital_twin/                  # Gymnasium Simulation Environment
│   ├── __init__.py
│   ├── mm1k_env.py                # Gymnasium wrapper around M/M/1/K simulation
│   └── traffic_generator.py       # Poisson traffic request generator with microbursts
│
├── model/                         # Deep Learning & RL Models
│   ├── __init__.py
│   ├── graph_transformer.py       # PyG Graph Transformer (GraphTransformerEncoder)
│   └── ppo_agent.py               # PPO Actor-Critic agent optimized for graph states
│
└── controller/                    # Ryu/OS-Ken SDN Controller (Stubs for Phase 2)
    └── __init__.py
```

### Core Components

1. **State Encoder (`model/graph_transformer.py`)**:
   - Converts the NetworkX topology graph (carrying node CPU/buffer load and edge capacity/delay/utilization metrics) into a PyTorch Geometric (PyG) `Data` object.
   - Appends a **virtual star node** connected to all nodes to aggregate global graph context.
   - Runs a 3-layer `TransformerConv` network to output a 128-dimensional latent state representation.

2. **PPO Actor-Critic Agent (`model/ppo_agent.py`)**:
   - **Actor**: Predicts routing action probability distributions over candidate $K$-shortest paths.
   - **Critic**: Estimates the state value function to compute Generalized Advantage Estimations (GAE).
   - **Optimized for CPU**: Pre-computes detached graph representations once per epoch, running the Actor-Critic networks efficiently on mini-batches. It does a single encoder gradient update per epoch to prevent redundant backpropagation and avoid slow CPU runs.

3. **Digital Twin Environment (`digital_twin/mm1k_env.py`)**:
   - Implements a standard Gymnasium environment interface.
   - Models links as analytical **M/M/1/K finite-capacity queues** where queue length, packet loss, and latency are calculated deterministically.
   - Dynamically tracks ingress/egress rates, buffer occupancies, and links bandwidth.

4. **Traffic Generator (`digital_twin/traffic_generator.py`)**:
   - Generates traffic matrix requests using Poisson arrivals.
   - Generates **microbursts** (temporary elephant flows at 5× base rate with a 10% probability) to stress-test routing resilience.

---

## ⚙️ Environment Setup

GARRO is configured to run inside a pre-installed Python virtual environment (`garro_env`). 

### Activate the Environment
Before running training or evaluation scripts, always activate the virtual environment:
```bash
source garro_env/bin/activate
```

### Dependencies
Dependencies are listed in [requirements.txt](file:///home/danny/schproject/requirements.txt) and are pre-installed in the virtual environment. Key requirements include:
- `torch` (PyTorch 2.3.1+cpu)
- `torch_geometric` (PyG)
- `torch_scatter` & `torch_sparse` (configured with matching CPU binaries)
- `networkx` (for topology graph representations)
- `gymnasium` (RL environment framework)
- `matplotlib` & `pandas` (for visualization and analysis)

---

## 🚀 How to Execute Training

Train the agent offline in the Digital Twin environment by running `train_offline.py`.

```bash
python train_offline.py --topology <topology_name> --episodes <num_episodes>
```

### Suggested Training Runs

* **NSFNET (14 nodes) — Quick Sanity Check (10-15 mins on CPU)**
  ```bash
  python train_offline.py --topology nsfnet --episodes 1000
  ```
* **GEANT2 (24 nodes) — Mid-size Network (30-45 mins on CPU)**
  ```bash
  python train_offline.py --topology geant2 --episodes 5000
  ```
* **Fat-Tree (80 nodes) — Full Scale Data Center (Overnight on CPU)**
  ```bash
  python train_offline.py --topology fat_tree --episodes 20000
  ```

### Training Outputs
* **Checkpoints**: Periodic model weights are saved to `checkpoints/garro_<topology>_ep<N>.pt`.
* **Final Model**: The converged weights are saved to `checkpoints/garro_<topology>_final.pt`.
* **Training Curve**: A plot of episodic rewards over time is saved to `checkpoints/training_curve_<topology>.png` to track training health.

### Understanding Training Metrics
During updates, the script outputs the following diagnostic metrics:
- **PL (Policy Loss)**: The surrogate objective of PPO. A stable negative or slightly fluctuating value indicates policy improvement.
- **VL (Value Loss)**: Mean-squared error of the critic. Should trend downwards as the critic learns to accurately estimate state values.
- **Ent (Entropy)**: Policy diversity. Starts high (~1.6 for $K=5$ paths) as the agent explores randomly, and should steadily decrease (to ~0.3–0.6) as the agent becomes confident in its routing decisions.
- **KL (KL Divergence)**: Difference between the old and updated policy. PPO clips this; values should stay very low ($<0.02$) to guarantee stable learning.

---

## 📊 How to Run Benchmarking & Evaluation

To evaluate the performance of your trained agent against traditional SDN routing baselines, run `evaluate.py`.

```bash
python evaluate.py --checkpoint <path_to_checkpoint> --topology <topology_name> --episodes <num_episodes>
```

### Example Command
```bash
python evaluate.py \
  --checkpoint checkpoints/garro_nsfnet_final.pt \
  --topology nsfnet \
  --episodes 500
```

### Compared Baselines
- **OSPF (Open Shortest Path First)**: A static shortest-path heuristic choosing the path with the minimum delay (Dijkstra).
- **ECMP (Equal-Cost Multi-Path)**: Distributes traffic round-robin across the candidate paths without network utilization awareness.
- **Random**: Randomly selects path options (defines the lower performance bound).
- **GARRO**: The trained RL agent selecting paths based on GAT-encoded telemetry features.

### Evaluation Outputs
* **CSV Table (`eval_results_<topology>.csv`)**: Tracks mean reward, standard deviation, minimum, and maximum rewards across all evaluation episodes.
* **Bar Chart Plot (`eval_results_<topology>.png`)**: A clean comparison bar chart of the average rewards achieved by each algorithm.

---

## 🛠️ Configuration Details (`config.yaml`)

The file `config.yaml` controls all parameters. Key settings include:

* **Routing Path Options (`network.k_paths: 5`)**: Number of candidate shortest-paths calculated between each source-destination pair.
* **M/M/1/K Queue Settings (`mm1k`)**:
  - `buffer_capacity: 50`: Limit of queue slots per link interface. Lower values increase packet drop rate (stress testing).
  - `base_arrival_rate`: Average packet arrival rate.
  - `base_service_rate`: Queue packet processing speed.
* **Reward Formula Weights (`reward_weights`)**:
  $$\text{Reward} = \alpha_1 \cdot \text{Throughput} - \alpha_2 \cdot \text{Delay} - \alpha_3 \cdot \text{Packet Loss} - \alpha_4 \cdot \text{Link Variance}$$
  - Default weights: $\alpha_1=0.4$ (Throughput), $\alpha_2=0.3$ (Delay), $\alpha_3=0.2$ (Loss), $\alpha_4=0.1$ (Load Balance).
* **PPO Hyperparameters (`ppo`)**: Epoch updates, learning rates, clip coefficients, and batch sizes.

---

## 🖥️ CPU vs. GPU Performance Guide

* **CPU Execution (Default)**: Thanks to the **detached latent pre-computation** implemented in `ppo_agent.py`, the training loop runs at a highly optimized speed (~4.0s per episode on NSFNET).
* **GPU Speedup**: If a CUDA-enabled GPU is detected, PyTorch will automatically offload graph encoding and model updates to CUDA. This cuts NSFNET training times down from hours to minutes.
