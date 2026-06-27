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

#### 🚀 GPU & System Resource Utilization Highlights
The codebase is designed to automatically optimize hardware resources in a Google Colab notebook environment:
*   **Automatic GPU Detection (CUDA)**: The agent (`model/ppo_agent.py`) auto-detects Colab's allocated NVIDIA GPU. Neural network forward/backward passes are fully accelerated on the GPU device.
*   **Automatic Mixed Precision (AMP)**: Under CUDA, PyTorch's `torch.autocast` is enabled dynamically. The agent queries hardware support: it utilizes `float16` on standard T4 GPU runtimes and automatically upgrades to `bfloat16` (Tensor Cores) if running on premium L4/A100 runtimes. This halves VRAM requirements and maximizes matrix multiplication throughput.
*   **Non-Blocking H→D Transfers**: A custom `FastGraphConverter` pins CPU memory buffers and streams converted Graph state tensors asynchronously (`non_blocking=True`) to the GPU, overlapping NetworkX-to-PyG conversion with model execution.
*   **JIT Compilation (`torch.compile`)**: The Graph Attention Transformer and Actor-Critic models are JIT-compiled using PyTorch 2.x's `mode="reduce-overhead"` to optimize the GPU execution graph and fuse kernel operations, boosting overall iteration speeds.
*   **CPU-Thread Pinning**: While neural network weights reside on the GPU, the analytical finite queue simulation (`digital_twin/mm1k_env.py`) and Poisson-gravity traffic generator (`digital_twin/traffic_generator.py`) run on the CPU. The training initialization pins PyTorch interop/intra-op threads to use all virtual cores allocated by Colab.

---

#### 1. Setup your Colab Notebook
1. Go to [Google Colab](https://colab.research.google.com).
2. Change the Runtime Type to use a GPU:
   * Click **Runtime** > **Change runtime type** > Select **T4 GPU** (or L4/A100 if available) > Click **Save**.
3. Verify the GPU assignment by running this in a cell:
   ```python
   !nvidia-smi
   ```

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
!git clone https://github.com/DanielAgbeni/GARRO schproject
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

Verify that PyTorch successfully binds to the GPU device by executing:
```python
import torch
print("GPU Available:", torch.cuda.is_available())
print("Active Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

#### 5. Redirect Checkpoints to Google Drive (To prevent data loss)
To avoid manual copying and guarantee you don't lose checkpoints if Colab crashes or disconnects, symlink the project's checkpoint folder directly to your Google Drive:
```bash
# Create checkpoints folder in Drive
!mkdir -p "/content/drive/MyDrive/garro_checkpoints"

# Delete default local directory (if any) and symlink to Google Drive
!rm -rf /content/schproject/checkpoints
!ln -s "/content/drive/MyDrive/garro_checkpoints" /content/schproject/checkpoints
```

#### 6. Run or Resume Training
Run the training script using GPU acceleration. Checkpoints will automatically write directly to your Google Drive via the symlink.

* **Start Training from scratch**:
  ```bash
  !python train_offline.py --topology nsfnet --episodes 10000
  ```

* **Resume Training after a disconnect**:
  If Colab times out or you stop the cell, mount your Drive, recreate the symlink (Steps 2 & 5), look in your Drive folder for the latest saved epoch (e.g. `garro_nsfnet_ep3000.pt`), and run:
  ```bash
  !python train_offline.py --topology nsfnet --episodes 10000 --resume checkpoints/garro_nsfnet_ep3000.pt
  ```

> [!NOTE]
> **What happens during Resume?** 
> The agent automatically parses the starting episode index from the filename (e.g., `ep3000` -> resumes from episode 3000). It loads the model weights along with the optimizer and gradient scaler states, allowing training to continue seamlessly with correct learning momentum.

> [!WARNING]
> **Phase 2 (Live Mininet Emulation) is NOT supported on Google Colab.** Mininet relies on loading custom Linux kernel modules (Open vSwitch) and low-level network namespace sandboxes, which are not allowed inside the lightweight Docker containers used by Google Colab. Emulation must always be run on your local Linux machine or WSL2 setup.

---

### 🟠 Running on Kaggle (Free T4 GPU — No Account Upgrade Required)

Kaggle provides a **free NVIDIA Tesla T4 GPU (16 GB VRAM)** with 29 GB RAM and up to **12 hours per session** — a strong alternative to Google Colab for longer runs (GEANT2 / Fat-Tree). No credit card is required.

> [!IMPORTANT]
> Kaggle sessions auto-shutdown after **12 hours**. For Fat-Tree (50 000 ep) you must save
> a checkpoint and resume in a new session using `--checkpoint`. Always download your `.pt`
> file before the session ends — use the **Output** tab in the Kaggle sidebar.

#### GPU & System Resource Utilization Highlights
The codebase auto-detects Kaggle's T4 and applies CUDA-optimised overrides:
- **CUDA Auto-scaling**: `batch_size → 256`, `update_interval → 2048`, `update_epochs → 15` (no config edit required).
- **AMP `float16`**: The T4 has no native `bfloat16` hardware support. Set `amp_dtype: float16` in `config.yaml` for optimal throughput.
- **Non-blocking H→D Transfers**: Graph state tensors are streamed asynchronously to the GPU, overlapping CPU simulation with GPU inference.
- **`torch.compile` off by default**: Disabled on Kaggle T4 to avoid stall-on-compile warnings; you still get full CUDA speedups via the scaled batch/rollout sizes.

#### Approximate T4 Speed Benchmarks

| Topology | Nodes | ep/s (T4) | 10 000 ep ETA |
|---|---|---|---|
| **NSFNET** | 14 | 7–10 | ~20–25 min |
| **GEANT2** | 24 | 3–5 | ~35–55 min |
| **Fat-Tree k=8** | 80 | 0.5–1 | ~3–6 h |

---

#### Step 1 — Create a Kaggle Notebook with GPU

1. Go to [kaggle.com](https://www.kaggle.com) and sign in (free account).
2. Click **+ New Notebook**.
3. In the right sidebar:
   - **Accelerator** → **GPU T4 ×2** ← select the dual-GPU option for maximum throughput
   - **Internet** → **On**
   - **Persistence** → **Files only** (keeps `/kaggle/working/` between sessions)
4. Click **Save** to start the session.

#### Step 2 — Clone the Repository

```python
# Cell 1
!git clone https://github.com/DanielAgbeni/GARRO.git /kaggle/working/schproject
%cd /kaggle/working/schproject
!ls -la
```

> If the repo is private, use a Personal Access Token:
> ```python
> !git clone https://YOUR_TOKEN@github.com/DanielAgbeni/GARRO.git /kaggle/working/schproject
> ```

#### Step 3 — Install Dependencies

Kaggle already ships PyTorch + CUDA. Only install the missing packages:

```python
# Cell 2 — Install project packages and matching PyG CUDA wheels
!pip install -q \
    torch-geometric==2.5.3 \
    gymnasium==1.2.2 \
    networkx==3.6.1 \
    numpy==2.4.4 \
    matplotlib==3.11.0 \
    pyyaml==6.0.3 \
    tqdm==4.68.3 \
    psutil==7.2.2

import torch
TORCH = torch.__version__.split("+")[0]          # e.g. "2.3.1"
CUDA  = "cu" + torch.version.cuda.replace(".", "")  # e.g. "cu121"
print(f"PyTorch: {TORCH} | CUDA tag: {CUDA}")

!pip install -q \
    torch-scatter -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html \
    torch-sparse  -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
```

#### Step 4 — Verify the GPU

```python
# Cell 3 — Sanity check (dual GPU)
import torch, os, sys

print("CUDA available :", torch.cuda.is_available())
print("GPU count      :", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}       : {props.name}  ({props.total_memory/1e9:.1f} GB VRAM)")

sys.path.insert(0, "/kaggle/working/schproject")
print("Python path    : OK")
```

Expected output (T4 ×2):
```
CUDA available : True
GPU count      : 2
  GPU 0        : Tesla T4  (15.8 GB VRAM)
  GPU 1        : Tesla T4  (15.8 GB VRAM)
```

#### Step 5 — Patch `config.yaml` for T4 ×2

```python
# Cell 4 — Configure for T4 ×2 (compile ON, penalties set per topology)
import yaml, torch

CONFIG_PATH = "/kaggle/working/schproject/config.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

# torch.compile mode="default" + dynamic=True is stable on T4 + PyTorch 2.x
# and gives 10–20% encoder speedup. "reduce-overhead" caused stalls; this does not.
cfg["training"]["compile_model"] = True

# amp_dtype is auto-detected as float16 for T4 — no manual override needed.
# (ppo_agent._autocast_dtype() detects bfloat16 support and falls back to float16)

# Set topology penalty weights (change before each topology run):
cfg["reward_weights"]["hop_weight"]        = 0.02   # NSFNET / GEANT2
cfg["reward_weights"]["congestion_weight"] = 1.0    # NSFNET / GEANT2
# Fat-Tree: set 0.05 and 2.0 instead

with open(CONFIG_PATH, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"config.yaml patched — compile_model=True, amp_dtype=auto (float16 on T4)")
print(f"GPUs visible to PyTorch: {torch.cuda.device_count()}")
```

#### Step 6 — Run Training

> [!IMPORTANT]
> With **GPU T4 ×2** the training script automatically:
> - Detects both GPUs and wraps encoder + AC-net with `nn.DataParallel`
> - Doubles `batch_size` → **512** and `update_interval` → **4096**
> - Applies the linear LR scaling rule (`lr_actor` → 2×, `lr_critic` → 2×)
> - Prints `[MultiGPU] 2× T4 detected` in the banner
>
> No extra flags needed — `--no-compile` is no longer used.

#### Approximate Speed on T4 ×2

| Topology | ep/s (single T4) | ep/s (T4 ×2) | 10 000 ep ETA (×2) |
|---|---|---|---|
| **NSFNET** | 7–10 | **10–14** | ~12–17 min |
| **GEANT2** | 3–5 | **5–8** | ~22–35 min |
| **Fat-Tree k=8** | 0.5–1 | **0.8–1.5** | ~2–4 h |

**NSFNET** (~12–17 min for 10 000 episodes on T4 ×2):
```python
!cd /kaggle/working/schproject && \
    python train_offline.py --topology nsfnet --episodes 10000
```

**GEANT2** (~30 min for 20 000 episodes):
```python
!cd /kaggle/working/schproject && \
    python train_offline.py --topology geant2 --episodes 20000
```

**Fat-Tree** (first session, 0 → 10 000 episodes):
```python
!cd /kaggle/working/schproject && \
    python train_offline.py --topology fat_tree --episodes 10000
```

> [!TIP]
> **Topology-aware penalty weights** — update `config.yaml → reward_weights`
> before training each topology to avoid reward scaling bias:
> ```python
> # Wide-area (NSFNET / GEANT2)
> cfg["reward_weights"]["hop_weight"]        = 0.02
> cfg["reward_weights"]["congestion_weight"] = 1.0
>
> # Data-centre (Fat-Tree)
> cfg["reward_weights"]["hop_weight"]        = 0.05
> cfg["reward_weights"]["congestion_weight"] = 2.0
> ```

#### Step 7 — Download Checkpoints Before Session Ends

```python
# Cell 6 — List all outputs
import os, glob

for f in sorted(glob.glob("/kaggle/working/schproject/checkpoints/*")):
    size = os.path.getsize(f) / 1e6
    print(f"{os.path.basename(f):50s}  {size:.1f} MB")
```

Then click **Output** in the Kaggle sidebar → download `garro_<topology>_final.pt`
and `training_curve_<topology>.png`.

> [!CAUTION]
> The session disk is wiped on shutdown if Persistence is **off**. Always download
> your checkpoint, or copy it to a Kaggle Dataset for permanent cloud storage.

#### Step 8 — Resume Training in a New Session (Multi-Session Fat-Tree)

Upload your saved `.pt` file as a Kaggle Dataset or re-upload it to the notebook input, then:

```python
# Cell 7 — Resume Fat-Tree from episode 10 000
CHECKPOINT = "/kaggle/working/schproject/checkpoints/garro_fat_tree_ep10000.pt"

!cd /kaggle/working/schproject && \
    python train_offline.py \
        --topology fat_tree \
        --episodes 20000 \
        --checkpoint {CHECKPOINT} \
        --no-compile
```

The script parses the episode index from the filename and resumes the progress bar automatically.

#### Step 9 — Evaluate the Trained Model

```python
# Cell 8 — Evaluate GARRO vs OSPF / ECMP / Random
CHECKPOINT = "/kaggle/working/schproject/checkpoints/garro_nsfnet_final.pt"

!cd /kaggle/working/schproject && \
    python evaluate.py \
        --checkpoint {CHECKPOINT} \
        --topology nsfnet \
        --episodes 500 \
        --output-dir evaluation_outputs
```

Results are saved under `evaluation_outputs/<run_id>/` as a `.csv` metrics table
and a `.png` comparison bar chart.

> [!WARNING]
> **Phase 2 (Live Mininet Emulation) is NOT supported on Kaggle.** Mininet requires
> loading custom Linux kernel modules (Open vSwitch) and network namespaces that are
> blocked in Kaggle's container environment. Phase 2 must be run on a local Linux
> machine or WSL2 setup.

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
* **Outputs**: Each evaluation creates its own folder under `evaluation_outputs/`, named with the evaluated checkpoint/model name, topology, episode count, and timestamp. The results are saved as `eval_results_<model>_<topology>_ep<episodes>.csv` and `eval_results_<model>_<topology>_ep<episodes>.png` inside that folder.

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

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](file:///home/danny/schproject/LICENSE) file for details.

