# GARRO: Graph Attention Routing with Reinforcement Learning

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-PyTorch%20Geometric-3C2179.svg)](https://pyg.org/)
[![SDN Controller](https://img.shields.io/badge/SDN-OS--Ken%20(OpenFlow%201.3)-green.svg)](https://github.com/openstack/os-ken)
[![Network Emulation](https://img.shields.io/badge/Emulation-Mininet-orange.svg)](http://mininet.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**GitHub Repository:** [github.com/DanielAgbeni/GARRO](https://github.com/DanielAgbeni/GARRO)

---

## 📖 Overview

**GARRO** is an end-to-end Deep Reinforcement Learning (DRL) routing framework designed for Software-Defined Networks (SDN). Traditional routing protocols such as **OSPF** (Open Shortest Path First) and **ECMP** (Equal-Cost Multi-Path) rely on static metrics (hop count or fixed link delays), leaving networks vulnerable to buffer overflow, packet drops, and severe latency spikes during unexpected microbursts or asymmetric traffic loads.

GARRO replaces static heuristics with an adaptive, topology-aware intelligent agent that combines:
1. **Graph Attention Transformer (GAT)**: Encodes live network states (node CPU/buffer loads and edge utilization/delay) with a virtual star node to aggregate global network context.
2. **Proximal Policy Optimization (PPO)**: Learns optimal traffic distribution policies over $K$-shortest paths, balancing latency, throughput, packet loss, and queue variance.
3. **Agentic LLM Intent Orchestrator**: Translates natural language operator requests (e.g., *"prioritize low latency for video calls"*) into dynamic reward weights ($\alpha_1..\alpha_4$) in real time without retraining the neural network.

The project is structured into two complementary phases:
* **Phase 1: Offline Digital Twin Training** — Fast, reproducible simulation using an analytical $M/M/1/K$ finite-capacity queueing model with Poisson microburst traffic generation. Runs anywhere (Linux, macOS, Windows/WSL2, Google Colab, Kaggle).
* **Phase 2: Live Mininet Emulation** — Closed-loop deployment in an emulated SDN network powered by Open vSwitch (OVS), Mininet, and the OS-Ken OpenFlow 1.3 controller, featuring a real-time Web Monitoring Dashboard.

---

## 📑 Table of Contents

- [🏗️ System Architecture](#️-system-architecture)
  - [Architectural Planes](#architectural-planes)
  - [Repository Module Map](#repository-module-map)
- [💻 Platform Compatibility & Prerequisites](#-platform-compatibility--prerequisites)
- [⚡ Quick Start: 5-Minute Setup (From Scratch)](#-quick-start-5-minute-setup-from-scratch)
  - [Step 1: Clone the Repository](#step-1-clone-the-repository)
  - [Step 2: System Packages (Linux / WSL2)](#step-2-system-packages-linux--wsl2)
  - [Step 3: Create & Activate Python Virtual Environment](#step-3-create--activate-python-virtual-environment)
  - [Step 4: Install PyTorch & PyTorch Geometric (PyG)](#step-4-install-pytorch--pytorch-geometric-pyg)
  - [Step 5: Install Project Dependencies](#step-5-install-project-dependencies)
  - [Step 6: Configure Environment Variables (Optional)](#step-6-configure-environment-variables-optional)
  - [Step 7: Verify Installation (Smoke Test)](#step-7-verify-installation-smoke-test)
- [🚀 Phase 1: Offline Digital Twin Training](#-phase-1-offline-digital-twin-training)
  - [How the Digital Twin Works](#how-the-digital-twin-works)
  - [Supported Topologies](#supported-topologies)
  - [Training Commands](#training-commands)
  - [Training Metrics Explained](#training-metrics-explained)
  - [Running on Google Colab (Free T4 GPU)](#-running-on-google-colab-free-t4-gpu)
  - [Running on Kaggle (Free Dual T4 GPU)](#-running-on-kaggle-free-dual-t4-gpu)
- [📊 Evaluation, Benchmarking & Visualizations](#-evaluation-benchmarking--visualizations)
  - [Benchmarking Against OSPF & ECMP (`evaluate.py`)](#benchmarking-against-ospf--ecmp-evaluatepy)
  - [Visualizing Routing Paths (`visualize_path.py`)](#visualizing-routing-paths-visualize_pathpy)
  - [Interactive Checkpoint Inspector (`inspect_checkpoint.py`)](#interactive-checkpoint-inspector-inspect_checkpointpy)
- [🌐 Phase 2: Live Mininet Emulation & SDN Deployment](#-phase-2-live-mininet-emulation--sdn-deployment)
  - [Why `start_env.sh` is Critical](#why-start_envsh-is-critical)
  - [The 3-Terminal Execution Blueprint](#the-3-terminal-execution-blueprint)
  - [Testing Live Network Traffic](#testing-live-network-traffic)
  - [Inspecting OpenFlow Switch Rules](#inspecting-openflow-switch-rules)
  - [Clean Shutdown & Process Termination](#clean-shutdown--process-termination)
- [🖥️ Web Dashboard & Agentic Intent Orchestration](#️-web-dashboard--agentic-intent-orchestration)
  - [Accessing the Dashboard](#accessing-the-dashboard)
  - [Natural Language Intent Routing](#natural-language-intent-routing)
- [⚙️ Configuration Reference (`config.yaml`)](#️-configuration-reference-configyaml)
- [📁 Repository Structure](#-repository-structure)
- [🔧 Troubleshooting & FAQ](#-troubleshooting--faq)
- [📄 License & Citation](#-license--citation)

---

## 🏗️ System Architecture

### Architectural Planes

```
                     ┌────────────────────────────────────────┐
                     │       Agentic AI Intent Layer          │
                     │  - LLM Intent Orchestrator (Groq/Gemini│ (agentic/llm_orchestrator.py)
                     │  - Translates NLP Intent -> α1..α4     │
                     └────────────────────────────────────────┘
                                         │ Dynamic Reward Weights
                                         ▼
                     ┌────────────────────────────────────────┐
                     │           AI Decision Plane            │
                     │  - Graph Transformer State Encoder     │ (model/graph_transformer.py)
                     │  - PPO Actor-Critic Routing Agent      │ (model/ppo_agent.py)
                     │  - Offline Trainer / Online Deployer   │ (train_offline.py / deploy_online.py)
                     └────────────────────────────────────────┘
                          ▲ Telemetry              │ Flow Decisions
            HTTP GET      │ State                  │ HTTP POST
            /garro/state  │                        ▼ /garro/flow
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                             SDN Control Plane                               │
 │  - OS-Ken OpenFlow 1.3 Controller Application (controller/garro_controller) │
 │  - Real-Time Web Telemetry Dashboard at http://localhost:8080               │
 │  - Flask REST API (Topology discovery, port stats, flow installation)       │
 └─────────────────────────────────────────────────────────────────────────────┘
                          ▲                        │
       LLDP & Port Stats  │                        │ OpenFlow 1.3 FlowMods
                          │                        ▼
 ┌────────────────────────────────────────┐   ┌────────────────────────────────┐
 │     SDN Data Plane (Phase 2 Live)      │   │  Simulation Plane (Phase 1)    │
 │  - Mininet Virtual Topology            │   │  - M/M/1/K Queueing Model      │
 │  - Open vSwitch (OVS) Kernel Datapath  │   │  - Poisson Traffic Generator   │
 │  - Isolated Named Netns Hosts          │   │  - Microburst Stress Testing   │
 │  (topologies/mininet_*.py)             │   │  (digital_twin/)               │
 └────────────────────────────────────────┘   └────────────────────────────────┘
```

### Repository Module Map

| Directory / File | Role | Description |
|---|---|---|
| [`model/`](file:///home/danny/projects/GARRO/model) | **Neural Core** | Graph Attention Network state encoder (`graph_transformer.py`) and PPO Actor-Critic routing agent (`ppo_agent.py`). |
| [`digital_twin/`](file:///home/danny/projects/GARRO/digital_twin) | **Simulation** | Analytical $M/M/1/K$ finite-buffer queueing environment (`mm1k_env.py`) and Poisson traffic generator with microburst injection (`traffic_generator.py`). |
| [`topologies/`](file:///home/danny/projects/GARRO/topologies) | **Topologies** | Network definitions for NSFNET, GEANT2, and Fat-Tree ($k=4$) both as NetworkX graphs and Mininet emulation scripts using `NamespacedHost`. |
| [`controller/`](file:///home/danny/projects/GARRO/controller) | **SDN Controller** | OS-Ken OpenFlow 1.3 controller with integrated Flask REST API, flow-mod pusher, and interactive Web Dashboard (`templates/index.html`). |
| [`agentic/`](file:///home/danny/projects/GARRO/agentic) | **Agentic Layer** | Universal LLM orchestrator translating natural-language operator intent into numerical reward weights ($\alpha_1..\alpha_4$). |
| [`diagnostics/`](file:///home/danny/projects/GARRO/diagnostics) | **Benchmarking** | Comprehensive diagnostic scripts, unit tests, and performance benchmark suites. |
| [`train_offline.py`](file:///home/danny/projects/GARRO/train_offline.py) | **Phase 1 Script** | Main entry point for training the DRL model in the analytical Digital Twin. |
| [`deploy_online.py`](file:///home/danny/projects/GARRO/deploy_online.py) | **Phase 2 Script** | Real-time deployment loop querying OS-Ken telemetry and pushing routing decisions to Mininet. |
| [`start_env.sh`](file:///home/danny/projects/GARRO/start_env.sh) | **Phase 2 Setup** | One-step startup script preparing Linux/WSL2 (OVS services, state flushing, loopback, asyncio eventlet). |
| [`kill_controller.sh`](file:///home/danny/projects/GARRO/kill_controller.sh) | **Process Utility** | Cleanly terminates lingering `osken-manager` controller instances. |
| [`evaluate.py`](file:///home/danny/projects/GARRO/evaluate.py) | **Evaluation** | Benchmarks trained GARRO checkpoints against OSPF, ECMP, and Random heuristics. |
| [`visualize_path.py`](file:///home/danny/projects/GARRO/visualize_path.py) | **Visualization** | Generates visual topology maps highlighting the selected routing paths. |
| [`inspect_checkpoint.py`](file:///home/danny/projects/GARRO/inspect_checkpoint.py) | **Inspection** | Scans and plots neural network weights, layer distributions, and convergence metrics. |

---

## 💻 Platform Compatibility & Prerequisites

| Feature / Phase | Linux (Ubuntu 20.04+) | Windows WSL2 (Ubuntu) | macOS (Apple Silicon) | Google Colab | Kaggle |
|---|---|---|---|---|---|
| **Phase 1 (Offline Training)** | ✅ Native | ✅ Native | ✅ Native (Metal MPS) | ✅ Free T4 GPU | ✅ Free Dual T4 |
| **Phase 1 Evaluation & Plotting** | ✅ Native | ✅ Native | ✅ Native | ✅ Supported | ✅ Supported |
| **Phase 2 (Mininet Emulation)** | ✅ Native | ✅ Native | ⚠️ Needs Linux VM (UTM) | ❌ No Kernel Modules | ❌ No Kernel Modules |
| **Web Dashboard (:8080)** | ✅ Browser | ✅ Windows Browser | ✅ Browser | ⚠️ Needs ngrok | ⚠️ Needs ngrok |

> [!NOTE]
> **macOS Users:** Phase 1 training runs natively on Apple Silicon with full GPU acceleration via **Metal Performance Shaders (MPS)**. However, Mininet (Phase 2) requires the Linux kernel's Open vSwitch modules and network namespaces. To run Phase 2 on a Mac, run Ubuntu inside a virtual machine (such as **UTM** or **Parallels Desktop**).

---

## ⚡ Quick Start: 5-Minute Setup (From Scratch)

Follow these steps on a clean Linux or Windows WSL2 machine to install all prerequisites, set up the virtual environment, and run your first test.

### Step 1: Clone the Repository

```bash
git clone https://github.com/DanielAgbeni/GARRO.git
cd GARRO
```

### Step 2: System Packages (Linux / WSL2)

Install the required system utilities, Mininet, Open vSwitch, and Python build tools:

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git curl openvswitch-switch mininet iproute2 iperf
```

### Step 3: Create & Activate Python Virtual Environment

GARRO is designed to run in an isolated Python virtual environment called `garro_env`:

```bash
python3 -m venv garro_env
source garro_env/bin/activate
pip install --upgrade pip
```

> [!TIP]
> Whenever you open a new terminal to work with GARRO, always re-activate the virtual environment:
> ```bash
> source garro_env/bin/activate
> ```

### Step 4: Install PyTorch & PyTorch Geometric (PyG)

PyTorch Geometric requires specific wheel packages (`torch-scatter` and `torch-sparse`) matched to your PyTorch version and hardware. Choose your platform:

#### Option A: CPU (Standard Local Machine / WSL2)
```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.1+cpu.html
pip install torch-geometric==2.5.3
```

#### Option B: NVIDIA GPU (CUDA 12.1)
```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
pip install torch-geometric==2.5.3
```

#### Option C: Apple Silicon Mac (M1/M2/M3/M4 with Metal MPS)
```bash
pip install torch
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html
pip install torch-geometric
```

### Step 5: Install Project Dependencies

Install the remaining dependencies from [requirements.txt](file:///home/danny/projects/GARRO/requirements.txt):

```bash
pip install -r requirements.txt
```

*(Key libraries installed include: `gymnasium`, `networkx`, `Flask`, `eventlet`, `os-ken`, `matplotlib`, `pandas`, `pyyaml`, `tqdm`, `aiohttp`, and LLM client SDKs).*

### Step 6: Configure Environment Variables (Optional)

If you plan to use the Agentic LLM Intent Orchestrator to dynamically adjust routing weights using natural language, create a `.env` file from the provided template:

```bash
cp .env.example .env
```
Open `.env` in your text editor and provide your free API key from [Google AI Studio](https://aistudio.google.com/) (`GEMINI_API_KEY`) or [Groq Console](https://console.groq.com/) (`GROQ_API_KEY`). Alternatively, you can use local, completely free models with **Ollama** (`provider: ollama` in `config.yaml`) with zero API keys required.

### Step 7: Verify Installation (Smoke Test)

Run a fast 10-episode sanity check in the Digital Twin to verify that PyTorch, PyG, Gymnasium, and NetworkX are functioning properly:

```bash
python train_offline.py --topology nsfnet --episodes 10 --no-compile
```

You should see the hardware detection banner followed by a progress bar completing 10 episodes in a few seconds:
```
======================================================================
  GARRO — Offline Digital Twin Training
======================================================================
  Topology    : nsfnet (14 nodes, 21 links)
  Device      : cpu (Intel/AMD)
  Episodes    : 10
======================================================================
```
🎉 **Congratulations! Your GARRO environment is fully operational.**

---

## 🚀 Phase 1: Offline Digital Twin Training

In Phase 1, the DRL agent is trained entirely inside the analytical Digital Twin environment. No Mininet, Open vSwitch, or root privileges are needed.

### How the Digital Twin Works

* **$M/M/1/K$ Finite Queueing Model**: Switch buffers have a finite capacity ($K=50$ packets). Buffer occupancy, packet drop probability ($P_{\text{loss}}$), and queuing delay are calculated analytically using queueing theory equations.
* **Poisson Traffic Generator with Microbursts**: Traffic requests arrive according to a Poisson process. With a 10% probability, an elephant microburst ($5\times$ the base arrival rate) is injected between random host pairs to stress-test load-balancing capabilities.
* **QoS-Aware Reward Formulation**: The PPO agent optimizes a composite reward balancing throughput, delay, packet loss, and link variance:
  $$R = \alpha_1 \cdot \text{Throughput} - \alpha_2 \cdot \text{Delay} - \alpha_3 \cdot P_{\text{loss}} - \alpha_4 \cdot \text{Variance} - \text{Penalties}$$

### Supported Topologies

| Topology | Type | Scale | Recommended Episodes | Convergence Time | Key Learning Objective |
|---|---|---|---|---|---|
| **`nsfnet`** | Wide Area Network (WAN) | 14 Nodes, 21 Links | 5,000 – 10,000 | ~15 min (GPU) | Learning loop avoidance and propagation latency awareness. |
| **`geant2`** | Pan-European Backbone | 24 Nodes, 37 Links | 15,000 – 20,000 | ~30 min (GPU) | Navigating asymmetric cross-links and avoiding hot-spots. |
| **`fat_tree`** | Data Center Network (DCN) | 20 Switches, 32 Links | 20,000 – 50,000 | ~45 min (GPU) | Hierarchical ECMP traffic spreading and multi-path balancing. |

### Training Commands

Make sure your virtual environment is active (`source garro_env/bin/activate`):

#### 1. NSFNET
```bash
# Quick sanity test (1,000 episodes):
python train_offline.py --topology nsfnet --episodes 1000

# Full convergence (10,000 episodes):
python train_offline.py --topology nsfnet --episodes 10000
```

#### 2. GEANT2
```bash
python train_offline.py --topology geant2 --episodes 20000
```

#### 3. Fat-Tree ($k=4$)
```bash
python train_offline.py --topology fat_tree --episodes 20000
```

#### Resuming Interrupted Training
If training is stopped, resume from any saved checkpoint:
```bash
python train_offline.py --topology nsfnet --episodes 10000 --checkpoint checkpoints/garro_nsfnet_ep5000.pt
```

### Training Metrics Explained

During rollouts and updates, the terminal outputs real-time diagnostic indicators:
* **PL (Policy Loss)**: Surrogate objective loss of PPO. Should fluctuate moderately around stable values.
* **VL (Value Loss)**: Mean-squared error of the critic network estimating state returns. Should steadily decrease over time.
* **Ent (Policy Entropy)**: Measures action diversity. Starts high ($\sim 1.6$ for $K=5$ paths) and gradually decays ($\sim 0.3 - 0.6$) as the agent solidifies optimal paths.
* **KL (KL Divergence)**: Step divergence between policy updates. PPO clips this; values remain small ($< 0.02$) to prevent catastrophic forgetting.

All checkpoints and learning curves are saved automatically:
* `checkpoints/garro_<topology>_ep<N>.pt` — Periodic snapshots.
* `checkpoints/garro_<topology>_final.pt` — Final trained weights.
* `checkpoints/training_curve_<topology>.png` — Episodic reward progress plot.

---

### ☁️ Running on Google Colab (Free T4 GPU)

If you don't have a local GPU, train your Phase 1 models for free on Google Colab:

1. Create a new notebook on [Google Colab](https://colab.research.google.com/) and set the runtime to **T4 GPU** (*Runtime* $\rightarrow$ *Change runtime type* $\rightarrow$ *T4 GPU*).
2. Mount Google Drive so checkpoints persist across restarts:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Clone GARRO and install PyG wheels:
   ```bash
   !git clone https://github.com/DanielAgbeni/GARRO.git /content/GARRO
   %cd /content/GARRO

   import torch
   pyg_url = f"https://data.pyg.org/whl/torch-{torch.__version__}.html"
   !pip install -q torch-scatter torch-sparse -f {pyg_url}
   !pip install -q torch-geometric gymnasium networkx pyyaml tqdm pandas matplotlib Flask aiohttp
   ```
4. Symlink the checkpoint directory to your Google Drive:
   ```bash
   !mkdir -p "/content/drive/MyDrive/garro_checkpoints"
   !rm -rf /content/GARRO/checkpoints
   !ln -s "/content/drive/MyDrive/garro_checkpoints" /content/GARRO/checkpoints
   ```
5. Launch training:
   ```bash
   !python train_offline.py --topology nsfnet --episodes 10000
   ```

---

### 🟠 Running on Kaggle (Free Dual T4 GPU)

Kaggle offers free **Dual NVIDIA T4 GPUs** (16 GB VRAM each) and 12-hour session limits:

1. Create a new Kaggle Notebook, enable **GPU T4 ×2** in the sidebar, and turn **Internet: ON**.
2. Clone and install dependencies:
   ```python
   !git clone https://github.com/DanielAgbeni/GARRO.git /kaggle/working/GARRO
   %cd /kaggle/working/GARRO

   import torch
   TORCH = torch.__version__.split("+")[0]
   CUDA = "cu" + torch.version.cuda.replace(".", "")
   !pip install -q torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
   !pip install -q torch-geometric gymnasium networkx pyyaml tqdm pandas matplotlib Flask aiohttp psutil
   ```
3. Launch multi-GPU training:
   ```python
   !python train_offline.py --topology nsfnet --episodes 10000
   ```
   *(The script automatically detects both GPUs, applies `nn.DataParallel`, scales batch size to 512, and doubles the learning rate).*
4. Download your `.pt` file from the **Output** tab before closing the session.

---

## 📊 Evaluation, Benchmarking & Visualizations

### Benchmarking Against OSPF & ECMP (`evaluate.py`)

Compare your trained GARRO model against traditional routing algorithms across 500 evaluation episodes in the Digital Twin. You can pass a specific checkpoint, or an **entire directory of checkpoints** to benchmark all curriculum stages and crown the optimal model:

```bash
# Benchmark all checkpoints across training stages (ranks top model):
python evaluate.py \
  --checkpoint checkpoints/ \
  --topology fat_tree \
  --episodes 500

# Or benchmark a specific checkpoint:
python evaluate.py \
  --checkpoint checkpoints/garro_fat_tree_final.pt \
  --topology fat_tree \
  --episodes 500
```

**Baselines Evaluated:**
* **OSPF (Dijkstra)**: Routes strictly via the path with minimal static delay.
* **ECMP (Equal-Cost Multi-Path)**: Splits traffic round-robin across candidate shortest paths.
* **Random**: Selects paths uniformly at random (lower-bound baseline).

**Outputs**: Results are exported to `evaluation_outputs/<run_id>/`:
* `eval_results_<model>_<topology>_ep<episodes>.csv` — Comprehensive metric breakdown table (Delay, Packet Loss, Throughput, Utilization Variance).
* `eval_results_<model>_<topology>_ep<episodes>.png` — Comparison bar charts.

### Visualizing Routing Paths (`visualize_path.py`)

Generate network topology diagrams displaying the exact path chosen by GARRO compared to OSPF:

```bash
# Visualize path chosen by GARRO from Seattle (0) to College Park (12) on NSFNET:
python visualize_path.py \
  --topology nsfnet \
  --src 0 \
  --dst 12 \
  --method garro \
  --checkpoint checkpoints/garro_nsfnet_ep10000.pt

# Visualize OSPF shortest path for comparison:
python visualize_path.py --topology nsfnet --src 0 --dst 12 --method ospf
```
The output is saved as `garro_routing_path.png`.

### Interactive Checkpoint Inspector (`inspect_checkpoint.py`)

Inspect the internal architecture, parameter counts, and weight norms of any saved checkpoint without needing to run training:

```bash
python inspect_checkpoint.py
```
This interactive terminal tool scans the `checkpoints/` directory, prints layer-by-layer tables, and produces an informative visual diagnostic plot (`<checkpoint>_inspection.png`).

---

## 🌐 Phase 2: Live Mininet Emulation & SDN Deployment

Phase 2 closes the loop: the trained PPO agent controls a **real, emulated SDN network** running on Linux kernel Open vSwitch datapaths.

```
+-------------------------------------------------------------------------------+
|                                PHASE 2 OVERVIEW                               |
|                                                                               |
|  [Terminal A]                 [Terminal B]                 [Terminal C]       |
|  OS-Ken Controller            Mininet Network              Online DRL Agent   |
|  (OpenFlow 1.3 + Flask)       (OVS Switches & Hosts)       (PPO + GAT)        |
|         |                            |                            |           |
|         |<===== OpenFlow 1.3 =======>|                            |           |
|         |       (Port 6633)          |                            |           |
|         |                            |                            |           |
|         |<==================== Telemetry (GET /garro/state) ======|           |
|         |===================== Flow Rules (POST /garro/flow) ====>|           |
|         |                            |                            |           |
|  Web Dashboard at :8080       Host Traffic (ping/iperf)    Real-Time Routing  |
+-------------------------------------------------------------------------------+
```

### Why `start_env.sh` is Critical

On Linux and particularly WSL2, system services (like Open vSwitch) do not always start automatically, and stale virtual ports or leftover Mininet namespaces from previous runs can block new topologies from connecting.

[start_env.sh](file:///home/danny/projects/GARRO/start_env.sh) prepares the system in one command:
1. **Starts Open vSwitch**: Runs `sudo service openvswitch-switch start` to activate kernel datapath daemons.
2. **Cleans Stale State**: Runs `sudo mn -c` to flush leftover OpenFlow switches, virtual ethernet links, and socket locks.
3. **Activates Loopback**: Ensures `lo` is up for internal controller-to-switch communication.
4. **Configures Eventlet Hub**: Sets `EVENTLET_HUB=asyncio` to eliminate greenlet/asyncio event loop conflicts.
5. **Prints Ready Instructions**: Outputs the exact commands needed for all 3 terminals.

#### Run `start_env.sh` before starting Phase 2:
```bash
bash start_env.sh
```

---

### The 3-Terminal Execution Blueprint

Running live emulation requires **three separate terminal tabs or windows**. Open 3 terminals and follow this sequence:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🖥️  Terminal A — Start the OS-Ken Controller                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ source garro_env/bin/activate                                               │
│ export EVENTLET_HUB=asyncio                                                 │
│ python run_controller.py controller/garro_controller.py --observe-links      │
│                                                                             │
│ (Alternative standard command:                                              │
│  python3 /usr/bin/osken-manager controller/garro_controller.py              │
│          --observe-links)                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │ (Wait ~3 seconds for controller to bind)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🖥️  Terminal B — Launch the Mininet Topology (Must use sudo)                │
├─────────────────────────────────────────────────────────────────────────────┤
│ # Choose the topology matching your trained model:                          │
│ sudo python3 topologies/mininet_nsfnet.py                                   │
│ # (or: sudo python3 topologies/mininet_geant2.py)                           │
│ # (or: sudo python3 topologies/mininet_fat_tree.py)                         │
│                                                                             │
│ (Wait ~5 seconds until the "mininet>" interactive prompt appears)           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │ (Wait ~5 seconds for LLDP discovery)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🖥️  Terminal C — Launch the Online DRL Agent Loop                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ source garro_env/bin/activate                                               │
│ # NOTE: Run as regular user (DO NOT USE SUDO), so the venv is used:         │
│ python deploy_online.py \                                                   │
│   --checkpoint checkpoints/garro_nsfnet_ep10000.pt \                        │
│   --topology nsfnet                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Testing Live Network Traffic

Once all 3 terminals are active, generate traffic from the **Mininet CLI in Terminal B**:

#### 1. Test Overall Connectivity (`pingall`)
```bash
mininet> pingall
```
*(Confirms that switches and hosts are linked and initial ARP requests resolve).*

#### 2. Trigger PPO Intelligent Routing (`ping`)
Send 10 ping packets between host `h1` and host `h14`:
```bash
mininet> h1 ping h14 -c 10
```
While the ping is running:
* **Terminal C** will print the path selected by the GAT-PPO agent:
  ```
  [Deploy] Flow installed: [1, 4, 5, 12, 14] | 10.0.0.1 -> 10.0.0.14
  ```
* **Terminal A** will confirm the OpenFlow 1.3 FlowMod rules pushed to the switches:
  ```
  [GARRO] Installed path: [1, 4, 5, 12, 14] for 10.0.0.1 -> 10.0.0.14
  ```

#### 3. High-Throughput Stress Test (`iperf`)
Generate a 30-second high-bandwidth TCP stream to force dynamic rerouting:
```bash
mininet> h1 iperf -s &
mininet> h14 iperf -c 10.0.0.1 -t 30
```
As link utilization increases along the primary path, watch the PPO agent in Terminal C dynamically choose alternative paths to avoid congestion.

---

### Inspecting OpenFlow Switch Rules

To inspect the OpenFlow rules installed on any switch (e.g. `s1`), open another terminal and run:
```bash
sudo ovs-ofctl -O OpenFlow13 dump-flows s1
```
You will observe specific flow entries matching IP source `10.0.0.1` and destination `10.0.0.14` routing packets to output ports designated by GARRO.

---

### Clean Shutdown & Process Termination

When you are finished testing Phase 2:
1. In **Terminal B (Mininet)**: Type `exit` or press `Ctrl+D`.
2. In **Terminal C (Agent)**: Press `Ctrl+C`.
3. In **Terminal A (Controller)**: Press `Ctrl+C`.
4. If the controller process lingers in the background or locks port 6633/8080, run the cleanup script:
   ```bash
   bash kill_controller.sh
   ```
5. Clean leftover Mininet virtual interfaces:
   ```bash
   sudo mn -c
   ```

---

## 🖥️ Web Dashboard & Agentic Intent Orchestration

### Accessing the Dashboard

While the OS-Ken controller is running in Terminal A, open your web browser and navigate to:
```
http://localhost:8080/
```
*(If running on WSL2, open the URL directly in your Windows host browser — WSL2 forwards `localhost` automatically).*

#### Dashboard Capabilities:
* **Interactive Network Canvas**: Visualizes switch topology nodes, links, and real-time packet counters.
* **Telemetry & Port Utilization**: Real-time charts showing link utilization, queue buffer occupancy, and transmission rates.
* **Host Explorer**: Lists connected hosts, IP addresses, MAC addresses, and attached switch ports.
* **Built-in Speedtest**: Run an iperf throughput test directly from the browser interface (`/garro/speedtest`).

---

### Natural Language Intent Routing

GARRO includes an **Agentic AI Intent Layer** that allows network operators to steer routing policies using natural language without touching neural network weights or retraining the model.

```
                     Operator Intent (Natural Language)
             "Prioritize video conferencing — latency is critical"
                                     │
                                     ▼
                      LLM Intent Orchestrator (Groq/Gemini)
                                     │
                                     ▼
                  Dynamic Reward Weights Computed:
                  α1 (Throughput) = 0.15
                  α2 (Delay)      = 0.55  <-- Latency prioritized
                  α3 (Loss)       = 0.20
                  α4 (Balance)    = 0.10
                                     │
                                     ▼
                      PPO Agent Evaluates K-Paths:
                   Picks lowest-latency path dynamically!
```

#### Supported Providers (`config.yaml`):
| Provider | Example Model | Setup | Cost |
|---|---|---|---|
| **`groq`** (Default) | `llama3-70b-8192` | Add `GROQ_API_KEY` to `.env` | Free, ultra-fast LPU inference |
| **`gemini`** | `gemini-1.5-flash` | Add `GEMINI_API_KEY` to `.env` | Free tier |
| **`ollama`** | `llama3` | Run `ollama pull llama3` locally | 100% Free & offline (no key required) |
| **`openai`** | `gpt-4o-mini` | Add `OPENAI_API_KEY` to `.env` | Paid |

#### Submitting Intents:
* **Via Web UI**: Enter your intent into the Intent input box on `http://localhost:8080` and click **Apply Intent**.
* **Via REST API**:
  ```bash
  curl -X POST http://localhost:8080/garro/intent \
    -H "Content-Type: application/json" \
    -d '{"intent": "Maximize throughput for large database backups across the network"}'
  ```

---

## ⚙️ Configuration Reference (`config.yaml`)

All system parameters are centralized in [config.yaml](file:///home/danny/projects/GARRO/config.yaml):

```yaml
network:
  k_paths: 5                      # Number of candidate K-shortest paths evaluated per (src, dst)
  topology: "nsfnet"              # Active topology: nsfnet | geant2 | fat_tree
  polling_interval: 2.0           # Seconds between live telemetry polls in Phase 2

mm1k:
  buffer_capacity: 50             # Buffer capacity (K) packets per switch port
  base_arrival_rate: 100.0        # Base arrival rate (lambda) in packets/sec
  base_service_rate: 150.0        # Base processing rate (mu) in packets/sec

graph_transformer:
  hidden_dim: 256                 # Latent embedding dimension
  num_heads: 8                    # Multi-head attention heads
  num_layers: 4                   # Number of TransformerConv layers
  dropout: 0.15                   # Regularization dropout rate

ppo:
  lr_actor: 1.0e-4                # Actor & GAT encoder learning rate
  lr_critic: 5.0e-4               # Critic network learning rate
  gamma: 0.99                     # Discount factor
  gae_lambda: 0.98                # Generalized Advantage Estimation (GAE) lambda
  clip_epsilon: 0.1               # PPO surrogate clipping threshold
  batch_size: 256                 # Mini-batch size
  entropy_coef: 0.03              # Entropy bonus coefficient (encourages exploration)

training:
  offline_episodes: 50000         # Total training episodes
  checkpoint_interval: 1000       # Save checkpoint every N episodes
  checkpoint_path: "checkpoints/" # Checkpoint destination directory
  compile_model: true             # torch.compile acceleration (15-40% speedup)

reward_weights:
  alpha1: 0.35                    # Throughput weight
  alpha2: 0.40                    # Latency / Delay weight
  alpha3: 0.15                    # Packet loss weight
  alpha4: 0.10                    # Link utilization variance weight
  hop_weight: 0.02                # Penalty cost per hop
  congestion_weight: 1.0          # Penalty multiplier when link utilization exceeds 70%

agentic:
  provider: "groq"                # gemini | groq | ollama | openai | mistral | cohere
  model: "llama3-70b-8192"        # Model identifier
```

---

## 📁 Repository Structure

```
GARRO/
├── .env.example                # Template for LLM API keys (Gemini, Groq, OpenAI, Ollama)
├── config.yaml                 # Master configuration (PPO, GAT, reward weights, LLM)
├── requirements.txt            # Python dependencies
├── start_env.sh                # Linux/WSL2 Phase 2 environment startup script
├── kill_controller.sh          # Utility script to terminate lingering controller processes
├── run_controller.py           # Controller launcher resolving system & venv paths
│
├── train_offline.py            # Phase 1: Digital Twin training entry point
├── deploy_online.py            # Phase 2: Live Mininet deployment loop
├── evaluate.py                 # Benchmarks GARRO against OSPF, ECMP, and Random
├── visualize_path.py           # Generates routing path topology diagrams
├── inspect_checkpoint.py       # Interactive neural network checkpoint analyzer
│
├── model/                      # Deep Reinforcement Learning Architecture
│   ├── graph_transformer.py    # GAT state encoder with virtual star node
│   └── ppo_agent.py            # PPO Actor-Critic agent with hardware auto-scaling
│
├── digital_twin/               # Analytical Simulation Engine
│   ├── mm1k_env.py             # Gymnasium M/M/1/K queueing simulation environment
│   └── traffic_generator.py    # Poisson traffic matrix generator with microbursts
│
├── controller/                 # SDN Control Plane
│   ├── garro_controller.py     # OS-Ken OpenFlow 1.3 application & REST API
│   └── templates/
│       └── index.html          # Real-time Web Monitoring Dashboard
│
├── topologies/                 # Network Topology Definitions
│   ├── nsfnet.py               # NSFNET NetworkX definition (14 nodes, 21 links)
│   ├── geant2.py               # GEANT2 NetworkX definition (24 nodes, 37 links)
│   ├── fat_tree.py             # Fat-Tree k=4 DCN definition (20 switches)
│   ├── mininet_nsfnet.py       # NSFNET Mininet script with TCLinks & NamespacedHost
│   ├── mininet_geant2.py       # GEANT2 Mininet script
│   ├── mininet_fat_tree.py     # Fat-Tree Mininet script
│   └── namespaced_host.py      # Bind-mounts network namespaces for host isolation
│
├── agentic/                    # Intent-Based Networking Layer
│   └── llm_orchestrator.py     # Universal multi-provider LLM intent parser
│
├── diagnostics/                # Diagnostics & Unit Tests
│   ├── benchmark_leaderboard.py# Multi-topology evaluation runner
│   ├── plot_training_metrics.py# Training curve visualizer
│   └── test_*.py               # Automated unit tests
│
└── checkpoints/                # Saved model weights (.pt) and training plots
```

---

## 🔧 Troubleshooting & FAQ

### 1. `start_env.sh: Permission denied` or OVS daemon fails
* **Cause**: `start_env.sh` needs executable permissions or sudo privileges to start system services.
* **Fix**: Run with bash directly:
  ```bash
  bash start_env.sh
  ```
  If prompted, enter your Linux/WSL2 password so `sudo service openvswitch-switch start` can execute.

### 2. Controller fails with `Address already in use` (Port 8080 or 6633)
* **Cause**: An earlier instance of `osken-manager` or another web server is still running in the background.
* **Fix**: Terminate lingering processes:
  ```bash
  bash kill_controller.sh
  ```
  Verify the ports are free:
  ```bash
  ss -tlnp | grep -E "6633|8080"
  ```

### 3. Mininet fails with `Cannot find switch` or `OVS is not running`
* **Cause**: The Open vSwitch daemon is inactive.
* **Fix**:
  ```bash
  sudo service openvswitch-switch start
  sudo mn -c
  ```

### 4. `deploy_online.py` fails with `ModuleNotFoundError` when run with `sudo`
* **Cause**: Running `sudo python deploy_online.py` switches to the root user's Python instead of the virtual environment.
* **Fix**: Run `deploy_online.py` **as a regular user without sudo** inside the virtual environment:
  ```bash
  source garro_env/bin/activate
  python deploy_online.py --checkpoint checkpoints/garro_nsfnet_ep10000.pt --topology nsfnet
  ```

### 5. Mininet outputs `Cannot find named namespace` during speedtest
* **Cause**: Mininet hosts were not launched using `NamespacedHost`.
* **Fix**: Always use the provided Mininet scripts (`topologies/mininet_nsfnet.py`), which automatically bind-mount host network namespaces into `/run/netns`.

### 6. Eventlet / Asyncio Loop Conflict (`RuntimeError: Eventlet hub conflict`)
* **Cause**: OS-Ken uses `eventlet` green threads while `deploy_online.py` and modern libraries use Python `asyncio`.
* **Fix**: Ensure `export EVENTLET_HUB=asyncio` is executed before starting the controller (already handled automatically in `start_env.sh` and `controller/garro_controller.py`).

### 7. Missing `aiohttp` in virtual environment
* **Cause**: `aiohttp` is required by the online deployment loop for non-blocking HTTP requests.
* **Fix**:
  ```bash
  source garro_env/bin/activate
  pip install aiohttp==3.14.1
  ```

### 8. Mininet on macOS
* **Cause**: Mininet requires low-level Linux kernel network namespaces and Open vSwitch modules which cannot run directly on macOS.
* **Fix**: Install an Ubuntu 22.04 / 24.04 ARM64 VM using **UTM** (free) or **Parallels Desktop**, and run Phase 2 inside the VM.

---

## 📄 License & Citation

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

### Citation
If you use GARRO in your research or educational project, please cite:

```bibtex
@misc{garro2026,
  author = {Daniel Agbeni},
  title = {GARRO: Graph Attention Routing with Reinforcement Learning for Software-Defined Networks},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DanielAgbeni/GARRO}}
}
```
