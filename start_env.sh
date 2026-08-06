#!/usr/bin/env bash
# ============================================================
#  GARRO / Mininet startup script for WSL2
#  Usage: bash start_env.sh
# ============================================================
set -e

echo "======================================================"
echo "  GARRO — WSL2 Environment Setup"
echo "======================================================"

# 1. Start OVS daemons (not auto-started in WSL)
echo "[1/4] Starting Open vSwitch daemons..."
sudo service openvswitch-switch start 2>/dev/null || true
sleep 1

# 2. Clean stale Mininet/OVS state
echo "[2/4] Cleaning stale Mininet state..."
sudo mn -c 2>/dev/null || true

# 3. Ensure loopback is up
echo "[3/4] Ensuring loopback is up..."
sudo ip link set lo up 2>/dev/null || true

# 4. Print instructions
echo "[4/4] Ready."
echo ""
echo "======================================================"
echo "  Terminal A — start controller:"
echo "    python3 /usr/bin/osken-manager controller/garro_controller.py --observe-links"
echo "    (or: osken-manager controller/garro_controller.py --observe-links)"
echo ""
echo "  Terminal B — (wait 3s) run topology with named netns:"
echo "    NSFNET:    sudo python3 topologies/mininet_nsfnet.py"
echo "    GEANT2:    sudo python3 topologies/mininet_geant2.py"
echo "    Fat-Tree:  sudo python3 topologies/mininet_fat_tree.py"
echo ""
echo "  Terminal C — (wait 5s after Mininet starts) run agent:"
echo "    NSFNET:    python deploy_online.py --checkpoint checkpoints/garro_nsfnet_ep6000.pt --topology nsfnet"
echo "    GEANT2:    python deploy_online.py --checkpoint checkpoints/garro_geant2_ep20000.pt --topology geant2"
echo "    Fat-Tree:  python deploy_online.py --checkpoint checkpoints/garro_fat_tree_final.pt --topology fat_tree"
echo "    NOTE: run as regular user (NOT sudo) so the venv Python is used."
echo ""
echo "  Mininet CLI — wait 5s, then:"
echo "    mininet> pingall"
echo ""
echo "  If aiohttp is missing, install it inside the venv:"
echo "    pip install aiohttp==3.14.1"
echo "======================================================"

