"""
Mininet topology for NSFNET (14 nodes, 21 links).
Run with: sudo python topologies/mininet_nsfnet.py

Requires Mininet installed system-wide (not in venv).
Start the OS-Ken controller first.
"""
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel


def build_nsfnet():
    setLogLevel("info")

    net = Mininet(
        controller=RemoteController,
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=True,
    )

    # Remote controller (OS-Ken)
    c0 = net.addController("c0", ip="127.0.0.1", port=6633)

    # Add 14 switches (one per NSFNET node)
    switches = []
    for i in range(1, 15):
        sw = net.addSwitch(f"s{i}", protocols="OpenFlow13")
        switches.append(sw)

    # Add hosts (one per switch for testing)
    hosts = []
    for i, sw in enumerate(switches, start=1):
        h = net.addHost(f"h{i}", ip=f"10.0.0.{i}/24")
        net.addLink(h, sw, bw=100, delay="1ms")
        hosts.append(h)

    # NSFNET edges: (src_idx, dst_idx, bw_Mbps, delay_ms)
    edges = [
        (0,1,1000,11),(0,2,1000,9),(0,4,1000,29),(1,2,1000,6),
        (1,3,1000,12),(2,5,1000,22),(3,4,1000,7),(4,5,1000,12),
        (4,9,1000,26),(5,6,1000,9),(6,7,1000,9),(6,8,1000,13),
        (7,8,1000,4),(7,12,1000,7),(8,9,1000,5),(9,10,1000,4),
        (10,11,1000,3),(10,12,1000,4),(11,12,1000,5),(11,13,1000,18),
        (5,11,1000,8),
    ]
    for src_i, dst_i, bw, delay in edges:
        net.addLink(
            switches[src_i], switches[dst_i],
            bw=bw, delay=f"{delay}ms", max_queue_size=50
        )

    net.start()
    # Set OpenFlow version on all switches
    for sw in switches:
        sw.cmd("ovs-vsctl set bridge", sw, "protocols=OpenFlow13")

    print("\n[Mininet] NSFNET topology running.")
    print("[Mininet] Hosts: h1-h14, Switches: s1-s14")
    print("[Mininet] Controller: 127.0.0.1:6633\n")

    CLI(net)
    net.stop()


if __name__ == "__main__":
    build_nsfnet()
