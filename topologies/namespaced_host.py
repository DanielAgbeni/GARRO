"""
Shared NamespacedHost for all Mininet topology scripts.

Mininet creates each host inside an isolated network namespace but does
NOT bind-mount it under /run/netns by default, so ``ip netns exec <host>``
fails.  This subclass fixes that by bind-mounting the namespace on
startShell() and cleaning up on terminate().

Usage in any Mininet topology script::

    from topologies.namespaced_host import NamespacedHost

    net = Mininet(..., host=NamespacedHost, ...)
"""
import subprocess
from mininet.node import Host


class NamespacedHost(Host):
    """Host subclass that registers itself as a named network namespace.

    Bind-mounts /proc/<pid>/ns/net → /run/netns/<hostname> so that
    ``ip netns exec <hostname>`` works for the controller's speedtest
    endpoint and any other tooling that needs namespace access.
    """

    @classmethod
    def setup(cls):
        # Mininet calls cls.setup() as a class method during sanity checks.
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
