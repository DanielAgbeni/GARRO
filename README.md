# GARRO: Graph Attention Routing with Reinforcement Learning

GARRO is a Deep Reinforcement Learning (DRL) routing framework designed for Software-Defined Networks (SDN). It combines a **Graph Attention Transformer (GAT)** to capture complex network topologies and traffic states with **Proximal Policy Optimization (PPO)** to dynamically route traffic requests.

This repository implements both **Phase 1: Offline Digital Twin Training** and **Phase 2: Live Mininet Emulation** using OS-Ken (a modern, Neutron-optimized fork of the Ryu SDN controller).

---

## 🏗️ Project Architecture

The project is structured into three clear planes: the AI Decision Plane, the Control Plane, and the Data/Simulation Plane.

```
                  ┌──────────────────────────────┐
                  │       Agentic AI Layer       │
                  │   - LLM Orchestrator         │ (Refreshes rewards from intent)
                  └──────────────────────────────┘
                                  │ 
                                  ▼ (Weight updates)
                  ┌──────────────────────────────┐
                  │       AI Decision Plane      │
                  │   - PPO + Graph Transformer  │ (model/)
                  │   - Offline Train Loop       │ (train_offline.py)
                  │   - Online Deploy Loop       │ (deploy_online.py)
                  └──────────────────────────────┘
                     ▲                       │
       HTTP GET      │                       │ HTTP POST
       /garro/state  │                       │ /garro/flow
                     │                       ▼
┌──────────────────────────────────────────────────────────────┐
│                    SDN Control Plane                         │
│                    - OS-Ken OpenFlow 1.3 App                 │ (controller/garro_controller.py)
│                    - Flask REST API Integration              │
└──────────────────────────────────────────────────────────────┘
                                  ▲
                                  │ OpenFlow 1.3 (Southbound)
                                  ▼
┌──────────────────────────────────────────────────────────────┐
│             SDN Data Plane (Phase 2 Emulation)               │
│             - Mininet Network Emulation                      │ (topologies/mininet_nsfnet.py)
│             - Open vSwitch (OVS) Kernel Datapath             │
└──────────────────────────────────────────────────────────────┘
```

### Module Descriptions

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

5. **OS-Ken Controller Application (`controller/garro_controller.py`)**:
   - Standard OS-Ken OpenFlow 1.3 controller application.
   - Listens to switch connection and link discovery events (via LLDP) to construct and maintain a live NetworkX `DiGraph` topology.
   - Periodically polls connected switches for port statistics to compute network utilization.
   - Integrates an internal **Flask web server** running on an eventlet green thread to serve a Northbound REST API (`GET /garro/state`, `GET /garro/topology`, `POST /garro/flow`).
   - Receives path updates from the AI plane and pushes OpenFlow 1.3 Flow Mod entries along the network switches.

6. **Mininet Network Topology (`topologies/mininet_nsfnet.py`)**:
   - Instantiates a live emulated NSFNET topology with 14 switches and 21 links.
   - Attaches one host per switch with configured subnet IPs (`10.0.0.1`–`10.0.0.14`) to allow testing.
   - Configures propagation delays matching the real physical topology nodes using Mininet TCLinks.

---

## ⚙️ Environment Setup

GARRO is configured to run inside a Python virtual environment (`garro_env`).

### Activate the Environment
Before running training, evaluation, or deployment scripts, activate the virtual environment:
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
- `Flask` (for controller REST API integration)

---

## 🚀 Phase 1: Offline Digital Twin Training

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
* **Training Curve**: A plot of episodic rewards over time is saved to `checkpoints/training_curve_<topology>.png`.

### Understanding Training Metrics
During updates, the script outputs the following diagnostic metrics:
- **PL (Policy Loss)**: The surrogate objective of PPO. A stable negative or slightly fluctuating value indicates policy improvement.
- **VL (Value Loss)**: Mean-squared error of the critic. Should trend downwards as the critic learns to accurately estimate state values.
- **Ent (Entropy)**: Policy diversity. Starts high (~1.6 for $K=5$ paths) as the agent explores randomly, and should steadily decrease (to ~0.3–0.6) as the agent becomes confident in its routing decisions.
- **KL (KL Divergence)**: Difference between the old and updated policy. PPO clips this; values should stay very low ($<0.02$) to guarantee stable learning.

### How to Run Benchmarking & Evaluation
To evaluate the performance of your trained agent against traditional SDN routing baselines in the Digital Twin, run `evaluate.py`.

```bash
python evaluate.py --checkpoint <path_to_checkpoint> --topology <topology_name> --episodes <num_episodes>
```

* **Example**:
  ```bash
  python evaluate.py \
    --checkpoint checkpoints/garro_nsfnet_final.pt \
    --topology nsfnet \
    --episodes 500
  ```
* **Compared Baselines**:
  - **OSPF (Open Shortest Path First)**: A static shortest-path heuristic choosing the path with the minimum delay (Dijkstra).
  - **ECMP (Equal-Cost Multi-Path)**: Distributes traffic round-robin across the candidate paths without network utilization awareness.
  - **Random**: Randomly selects path options (defines the lower performance bound).
* **Outputs**: Table saved to `eval_results_<topology>.csv` and comparison bar chart saved to `eval_results_<topology>.png`.

---

## 🌐 Phase 2: Live Mininet Emulation

Phase 2 runs the DRL agent in a closed loop, routing real traffic demands in an emulated SDN network.

### Prerequisites (System-Wide)
Running Mininet and Open vSwitch (OVS) requires root permissions (`sudo`) and system-wide packages. They **cannot** be run inside the Python virtual environment.
- On Ubuntu/Debian:
  ```bash
  sudo apt-get update
  sudo apt-get install mininet openvswitch-switch
  ```
- **If using WSL2**, you must ensure the OVS kernel module or service is started before running:
  ```bash
  sudo service openvswitch-switch start
  ```

### Step-by-Step Emulation Execution
You will need **three separate terminal sessions** to execute the loop.

#### 1. Terminal A: Start the OS-Ken Controller
Activate the environment and start the OpenFlow controller app.
```bash
source garro_env/bin/activate
osken-manager controller/garro_controller.py --observe-links
```

#### 2. Terminal B: Launch the Mininet Topology (Requires Sudo)
Launch the emulated data plane using system Python.
```bash
sudo python topologies/mininet_nsfnet.py
```
*(This starts the Mininet prompt `mininet>` once topology initialization is complete).*

#### 3. Terminal C: Launch the DRL Online Agent Loop
Activate the environment and run the deployment loop script pointing to your trained checkpoint file.
```bash
source garro_env/bin/activate
python deploy_online.py --checkpoint checkpoints/garro_nsfnet_final.pt --topology nsfnet
```
*(The agent will start polling the REST API on port `8080` every 2 seconds, selecting paths, and installing them onto the switches).*

---

## 🧪 Verifying Live Traffic Routing in Mininet

Once all three terminal sessions are running, use the Mininet CLI (**Terminal B**) to generate traffic and verify routing behavior:

### 1. Test Network Connectivity
Verify that all hosts can reach one another:
```bash
mininet> pingall
```

### 2. Verify Flow Rules Installation
Ping from host `h1` to host `h14` to trigger PPO-driven routing:
```bash
mininet> h1 ping h14 -c 10
```
While the ping is running, inspect Terminal C and Terminal A. You will see:
- Terminal C log: `[Deploy] Flow installed: [1, 4, 5, 12, 14] | 10.0.0.1→10.0.0.14`
- Terminal A log: `[GARRO] Installed path: [1, 4, 5, 12, 14] for 10.0.0.1 → 10.0.0.14`

### 3. Dump Switch Flows
To inspect the exact OpenFlow rules installed on any switch (e.g. switch `s1`), run this command in a normal bash shell:
```bash
sudo ovs-ofctl -O OpenFlow13 dump-flows s1
```
You will see routing flow rules matching IP source `10.0.0.1` and destination `10.0.0.14` routing packets to the corresponding output ports chosen by the PPO agent.

### 4. Benchmarking Throughput
To run a bandwidth throughput test using iperf:
```bash
mininet> h1 iperf -s &
mininet> h14 iperf -c 10.0.0.1 -t 30
```
This generates traffic between `h1` and `h14` for 30 seconds, forcing the agent to continuously monitor utilization changes and dynamically adjust paths to avoid link congestion.

---

## 🔧 Troubleshooting Guide

| Issue | Cause | Fix |
|---|---|---|
| `AttributeError: module 'os_ken.base.app_manager' has no attribute 'RyuApp'` | legacy Ryu class name conflict | Ensure controller uses `app_manager.OSKenApp` instead of `RyuApp` (fixed in repository). |
| `ModuleNotFoundError: No module named 'os_ken.app.wsgi'` | WSGI module missing/renamed in OS-Ken | Expose APIs using Flask on a background eventlet thread (fixed in repository). |
| `OVS is not running` error in Mininet | Open vSwitch service inactive | Run `sudo service openvswitch-switch start` before launching Mininet. |
| Mininet cannot connect to controller | Controller not running or ports bound | Ensure OS-Ken is running in Terminal A first. Check for bound ports: `ss -tlnp \| grep -E "6633\|8080"`. |
| LLM weight updates fail | `.env` file does not contain keys | The agentic layer defaults to standard fallback weights (`alpha1..alpha4`) if `GEMINI_API_KEY` is not present, avoiding application crash. |
