# GARRO: Graph Attention Routing with Reinforcement Learning

**GitHub Repository:** [github.com/DanielAgbeni/GARRO](https://github.com/DanielAgbeni/GARRO)

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

### 🍏 Apple Silicon (M1/M2/M3/M4 macOS) Support

GARRO fully supports native training on Apple Silicon chips (M-series) using Apple's GPU acceleration via **Metal Performance Shaders (MPS)**.

#### 1. Running Phase 1 (Offline Training) on macOS
To run training locally on your Mac with GPU/MPS acceleration:
1. Ensure you have installed a native macOS python virtual environment.
2. Install PyTorch with MPS support:
   ```bash
   pip install torch
   ```
3. Install PyTorch Geometric (PyG) dependencies compatible with your PyTorch macOS version:
   ```bash
   pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html
   pip install torch-geometric
   ```
4. Run the training script normally. The agent will auto-detect your M-series chip and display:
   `Device      : mps  (Apple Metal Performance Shaders)` in the startup hardware banner:
   ```bash
   python train_offline.py --topology nsfnet --episodes 10000
   ```

#### 2. Running Phase 2 (Live Mininet Emulation) on macOS
> [!IMPORTANT]
> **Mininet is Linux-only and cannot run natively on macOS.**
> To emulate the network topology on an Apple Silicon Mac, you must run it inside a Linux Virtual Machine (VM). We recommend:
> * **UTM (Free/Open Source)** or **Parallels Desktop**: Set up an ARM64 Ubuntu Linux VM.
> * Run both the OS-Ken Controller and Mininet inside that ARM64 Linux VM.

---

## 🚀 Phase 1: Offline Digital Twin Training

Train the agent offline in the Digital Twin environment by running `train_offline.py`.

```bash
python train_offline.py --topology <topology_name> --episodes <num_episodes>
```

### 🌐 Multi-Topology Validation Strategy

To present an academically rigorous and reliable project, it is highly recommended to train and evaluate your agent across **all three topologies**. This proves that the GARRO routing agent can adapt to different types of network layouts (WAN vs. Data Center) and scales (14 to 80 nodes) without overfitting to a single network structure:

1. **Topology Diversity**: Shows the model's ability to handle geographic WAN nodes with high propagation delays (NSFNET, GEANT2) as well as dense, symmetric, low-latency datacenter clusters (Fat-Tree).
2. **Scale & Convergence Proof**: Confirms that the Graph Attention Network (GAT) generalizes well when the network size grows from small (14 switches) to very large (80 switches).
3. **Load Balancing Resilience**: Proves the routing agent can handle highly irregular network links (GEANT2) just as well as standard hierarchical ones.

---

### 📊 Recommended Training Configurations

Depending on your goal (quick testing vs. reliable publication-grade results), use the following recommended parameters:

| Topology | Scale | Checkpoint ID | Sanity Check (Episodes) | Full Convergence (Recommended) | Approx. CPU Time | Key Learning Focus |
|---|---|---|---|---|---|---|
| **NSFNET** | 14 Nodes, 21 Links | `nsfnet` | 1,000 | **5,000 - 10,000** | ~1.5 hours (Full) | Basic routing loops, WAN propagation latency awareness. |
| **GEANT2** | 24 Nodes, 37 Links | `geant2` | 5,000 | **15,000 - 20,000** | ~4 hours (Full) | Load balancing under asymmetric constraints & irregular cross-links. |
| **Fat-Tree** | 80 Nodes, 112 Links | `fat_tree` | 10,000 | **40,000 - 50,000** | ~12 hours (Full) | Hierarchical paths, core-aggregation traffic spreading in data centers. |

---

### 🛠️ Execution Commands

#### 1. NSFNET Training
* **Sanity Check**:
  ```bash
  python train_offline.py --topology nsfnet --episodes 1000
  ```
* **Full Reliable Training**:
  ```bash
  python train_offline.py --topology nsfnet --episodes 10000
  ```

#### 2. GEANT2 Training
* **Sanity Check**:
  ```bash
  python train_offline.py --topology geant2 --episodes 5000
  ```
* **Full Reliable Training**:
  ```bash
  python train_offline.py --topology geant2 --episodes 20000
  ```

#### 3. Fat-Tree Training
* **Sanity Check**:
  ```bash
  python train_offline.py --topology fat_tree --episodes 10000
  ```
* **Full Reliable Training**:
  ```bash
  python train_offline.py --topology fat_tree --episodes 50000
  ```


### ☁️ Running on Google Colab (Recommended for GPU Acceleration)

If you do not have a local GPU, you can train the offline model (Phase 1) on Google Colab to speed up the process using a free NVIDIA T4 GPU.

#### 1. Setup your Colab Notebook
1. Go to [Google Colab](https://colab.research.google.com).
2. Change the Runtime Type to use a GPU:
   * Click **Runtime** > **Change runtime type** > Select **T4 GPU** > Click **Save**.

#### 2. Mount Google Drive (To save checkpoints)
Add this to a notebook cell to mount your Drive so checkpoints aren't lost when your session ends:
```python
from google.colab import drive
drive.mount('/content/drive')
```

#### 3. Clone the Project & Setup Files
Clone your project repository or upload your files to Google Drive, then navigate into the directory:
```bash
# Example if using git:
!git clone <your-repository-url> schproject
%cd schproject
```

#### 4. Install Dependencies
Colab has PyTorch preinstalled, but you must install PyTorch Geometric (PyG) and its dependencies compiled for Colab's specific PyTorch + CUDA version:
```python
import torch
# Automatically detect installed PyTorch and CUDA versions to pull the correct PyG binary wheels
pyg_url = f"https://data.pyg.org/whl/torch-{torch.__version__}.html"
print(f"Installing PyG wheels from: {pyg_url}")

!pip install torch-scatter torch-sparse -f {pyg_url}
!pip install torch-geometric
!pip install gymnasium networkx pyyaml tqdm pandas matplotlib Flask
```

#### 5. Run Training
Run the training script using GPU acceleration. Checkpoints will automatically be saved to the `checkpoints/` folder.
```bash
!python train_offline.py --topology nsfnet --episodes 10000
```

#### 6. Save Checkpoints to Google Drive
Ensure your trained weights are safely copied to your mounted Google Drive:
```bash
!cp -r checkpoints/ /content/drive/MyDrive/garro_checkpoints/
```

> [!WARNING]
> **Phase 2 (Live Mininet Emulation) is NOT supported on Google Colab.** Mininet relies on loading custom Linux kernel modules (Open vSwitch) and low-level network namespace sandboxes, which are not allowed inside the lightweight Docker containers used by Google Colab. Emulation must always be run on your local Linux machine or WSL2 setup.

---

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
