"""
GEANT2 Mininet Emulation Script

Requires Mininet installed system-wide (not in venv).
Start the OS-Ken controller first:
    osken-manager controller/garro_controller.py --observe-links
"""
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel


def build_geant2():
    setLogLevel("info")

    net = Mininet(
        controller=RemoteController,
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=True,
    )

    # Remote controller (OS-Ken)
    c0 = net.addController("c0", ip="127.0.0.1", port=6633)

    # Add 24 switches (one per GEANT2 node)
    switches = []
    for i in range(1, 25):
        sw = net.addSwitch(f"s{i}", protocols="OpenFlow13")
        switches.append(sw)

    # Add hosts — all on the same /24 subnet so pure L2 switching works.
    hosts = []
    for i, sw in enumerate(switches, start=1):
        h = net.addHost(f"h{i}", ip=f"10.0.0.{i}/24")
        net.addLink(h, sw, bw=100, delay="1ms")
        hosts.append(h)

    # GEANT2 edges (src_idx, dst_idx, bw_Mbps, delay_ms)
    edges = [
        (0,  1,  1000,  5),
        (0,  2,  1000, 12),
        (1,  3,  1000,  8),
        (2,  4,  1000, 15),
        (3,  5,  1000,  6),
        (4,  5,  1000, 20),
        (5,  6,  1000,  7),
        (6,  7,  1000,  9),
        (7,  8,  1000, 11),
        (8,  9,  1000, 14),
        (9,  10, 1000, 18),
        (10, 11, 1000,  6),
        (11, 12, 1000,  8),
        (12, 13, 1000, 10),
        (13, 14, 1000,  7),
        (14, 15, 1000, 12),
        (15, 16, 1000,  9),
        (16, 17, 1000,  6),
        (17, 18, 1000,  8),
        (18, 19, 1000, 11),
        (19, 20, 1000, 14),
        (20, 21, 1000,  9),
        (21, 22, 1000,  7),
        (22, 23, 1000,  6),
        # Cross-links
        (0,  6,  1000, 25),
        (1,  8,  1000, 22),
        (3,  12, 1000, 30),
        (5,  15, 1000, 27),
        (7,  17, 1000, 19),
        (9,  19, 1000, 28),
        (11, 21, 1000, 24),
        (13, 22, 1000, 31),
        (2,  10, 1000, 35),
        (4,  14, 1000, 32),
        (6,  18, 1000, 29),
        (8,  20, 1000, 26),
        (10, 23, 1000, 21),
    ]

    for src, dst, bw, delay in edges:
        # Mininet uses actual switch objects
        s_src = switches[src]
        s_dst = switches[dst]
        net.addLink(s_src, s_dst, bw=bw, delay=f"{delay}ms", max_queue_size=50)

    net.start()

    # Ensure OpenFlow 1.3 on all switches
    for sw in switches:
        sw.cmd(f"ovs-vsctl set bridge {sw.name} protocols=OpenFlow13")

    print("\n*** GEANT2 Topology running. Ping between hosts to test routing.")
    CLI(net)
    net.stop()


if __name__ == "__main__":
    build_geant2()
