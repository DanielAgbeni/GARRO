"""
Mininet topology for Fat-Tree k=4 (20 switches, 16 hosts).
Run with: sudo python topologies/mininet_fat_tree.py

Requires Mininet installed system-wide (not in venv).
Start the OS-Ken controller first:
    osken-manager controller/garro_controller.py --observe-links
    OR
    python3 /usr/bin/osken-manager controller/garro_controller.py --observe-links
"""
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel
try:
    from topologies.namespaced_host import NamespacedHost
except ModuleNotFoundError:
    from namespaced_host import NamespacedHost


def build_fat_tree(k: int = 4):
    setLogLevel("info")

    net = Mininet(
        controller=RemoteController,
        switch=OVSKernelSwitch,
        link=TCLink,
        host=NamespacedHost,
        autoSetMacs=True,
    )

    # Remote controller (OS-Ken)
    c0 = net.addController("c0", ip="127.0.0.1", port=6633)

    num_core = (k // 2) ** 2               # 4
    num_agg  = k * (k // 2)                 # 8
    num_edge = k * (k // 2)                 # 8
    total_switches = num_core + num_agg + num_edge  # 20

    core_start = 0
    agg_start  = num_core
    edge_start = num_core + num_agg

    # Add 20 switches (s1 to s20)
    switches = []
    for i in range(1, total_switches + 1):
        sw = net.addSwitch(f"s{i}", protocols="OpenFlow13")
        switches.append(sw)

    # Add 16 hosts attached to edge switches (2 hosts per edge switch)
    hosts = []
    host_idx = 1
    for edge_i in range(edge_start, total_switches):
        edge_sw = switches[edge_i]
        for _ in range(k // 2):
            h = net.addHost(f"h{host_idx}", ip=f"10.0.0.{host_idx}/24")
            net.addLink(h, edge_sw, bw=100, delay="1ms")
            hosts.append(h)
            host_idx += 1

    # Wire up Fat-Tree pods
    for pod in range(k):
        for agg_idx in range(k // 2):
            agg_id = agg_start + pod * (k // 2) + agg_idx

            # Aggregation → Edge (within same pod)
            for edge_idx in range(k // 2):
                edge_id = edge_start + pod * (k // 2) + edge_idx
                net.addLink(
                    switches[agg_id], switches[edge_id],
                    bw=1000, delay="1ms", max_queue_size=50
                )

            # Aggregation → Core
            for core_idx in range(k // 2):
                core_id = core_start + agg_idx * (k // 2) + core_idx
                net.addLink(
                    switches[core_id], switches[agg_id],
                    bw=1000, delay="1ms", max_queue_size=50
                )

    net.start()

    # Ensure OpenFlow 1.3 on all switches
    for sw in switches:
        sw.cmd(f"ovs-vsctl set bridge {sw.name} protocols=OpenFlow13")

    print("\n[Mininet] Fat-Tree (k=4) topology running.")
    print(f"[Mininet] Hosts: h1-h{len(hosts)}, Switches: s1-s{total_switches}")
    print("[Mininet] Controller: 127.0.0.1:6633\n")

    CLI(net)
    net.stop()


if __name__ == "__main__":
    build_fat_tree()
