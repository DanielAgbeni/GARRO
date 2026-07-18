"""
Mininet topology for NSFNET (14 nodes, 21 links).
Run with: sudo python topologies/mininet_nsfnet.py

Requires Mininet installed system-wide (not in venv).
Start the OS-Ken controller first:
    osken-manager controller/garro_controller.py --observe-links
    OR
    python3 /usr/bin/osken-manager controller/garro_controller.py --observe-links
"""
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController, Host
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel
import subprocess


class NamespacedHost(Host):
    """Host subclass that registers itself as a named network namespace.

    Mininet creates each host inside an isolated network namespace but
    does NOT bind-mount it under /run/netns by default.  This means
    ``ip netns exec <hostname>`` fails, forcing the speedtest code to
    fall back to scanning /proc (which requires root).

    We override startShell() — called after the host process is spawned
    and self.pid is available — to bind-mount the namespace into
    /run/netns/<hostname> so ``ip netns exec <hostname>`` works without
    elevated privileges in the controller process.

    setup() is intentionally left as a classmethod passthrough so
    Mininet's checkSetup() call (cls.setup()) does not break.
    """

    @classmethod
    def setup(cls):
        # Mininet calls cls.setup() as a class method during sanity checks.
        # Delegate to the parent class method cleanly.
        Host.setup()

    def startShell(self, *args, **kwargs):
        super().startShell(*args, **kwargs)
        # self.pid is now valid — bind-mount the netns
        subprocess.run(["mkdir", "-p", "/run/netns"], check=False)
        netns_path = f"/run/netns/{self.name}"
        proc_ns = f"/proc/{self.pid}/ns/net"
        subprocess.run(["touch", netns_path], check=False)
        subprocess.run(
            ["mount", "--bind", proc_ns, netns_path], check=False
        )

    def terminate(self):
        # Clean up the bind-mount when the host exits
        netns_path = f"/run/netns/{self.name}"
        subprocess.run(["umount", netns_path], check=False)
        subprocess.run(["rm", "-f", netns_path], check=False)
        super().terminate()


def build_nsfnet():
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

    # Add 14 switches (one per NSFNET node)
    switches = []
    for i in range(1, 15):
        sw = net.addSwitch(f"s{i}", protocols="OpenFlow13")
        switches.append(sw)

    # Add hosts — all on the same /24 subnet so pure L2 switching works.
    # No gateway/router needed: hosts ARP for each other directly.
    hosts = []
    for i, sw in enumerate(switches, start=1):
        h = net.addHost(f"h{i}", ip=f"10.0.0.{i}/24")
        net.addLink(h, sw, bw=100, delay="1ms")
        hosts.append(h)

    # NSFNET edges aligned with nsfnet.py node ordering:
    #   0=Seattle, 1=Palo Alto, 2=San Diego, 3=Salt Lake City, 4=Boulder,
    #   5=Lincoln, 6=Houston, 7=Champaign, 8=Atlanta, 9=Ann Arbor,
    #   10=Pittsburgh, 11=Princeton, 12=College Park, 13=Ithaca
    # (src_idx, dst_idx, bw_Mbps, delay_ms)
    edges = [
        (0,  1,  1000, 11),
        (0,  3,  1000,  9),
        (0,  5,  1000, 29),
        (1,  2,  1000,  6),
        (1,  3,  1000, 12),
        (2,  6,  1000, 22),
        (3,  4,  1000,  7),
        (4,  5,  1000, 12),
        (4,  10, 1000, 26),
        (5,  7,  1000,  9),
        (6,  8,  1000,  9),
        (7,  8,  1000, 13),
        (7,  9,  1000,  4),
        (8,  12, 1000,  8),
        (9,  10, 1000,  5),
        (9,  13, 1000,  7),
        (10, 11, 1000,  4),
        (11, 12, 1000,  3),
        (11, 13, 1000,  4),
        (12, 13, 1000,  5),
        (6,  12, 1000, 18),
    ]
    for src_i, dst_i, bw, delay in edges:
        net.addLink(
            switches[src_i], switches[dst_i],
            bw=bw, delay=f"{delay}ms", max_queue_size=50
        )

    net.start()

    # Ensure OpenFlow 1.3 on all switches
    for sw in switches:
        sw.cmd(f"ovs-vsctl set bridge {sw.name} protocols=OpenFlow13")

    print("\n[Mininet] NSFNET topology running.")
    print("[Mininet] Hosts: h1-h14, Switches: s1-s14")
    print("[Mininet] Controller: 127.0.0.1:6633\n")

    CLI(net)
    net.stop()


if __name__ == "__main__":
    build_nsfnet()
