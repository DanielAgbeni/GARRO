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
echo "    osken-manager controller/garro_controller.py --observe-links"
echo ""
echo "  Terminal B — (wait 3s) run topology:"
echo "    sudo python topologies/mininet_nsfnet.py"
echo ""
echo "  Mininet CLI — wait 5s, then:"
echo "    mininet> pingall"
echo "======================================================"
