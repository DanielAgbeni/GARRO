"""
OS-Ken OpenFlow 1.3 controller application for GARRO.
Handles:
  - Topology discovery via LLDP
  - Real-time telemetry collection (port stats, flow stats)
  - Flask-based REST API exposing network state JSON
  - Flow rule installation from PPO agent routing decisions

Run with:
    osken-manager controller/garro_controller.py \
        --observe-links
    OR:
    python3 /usr/bin/osken-manager controller/garro_controller.py \
        --observe-links
"""
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import (
    CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
)
from os_ken.ofproto import ofproto_v1_3
from os_ken.lib.packet import packet, ethernet, ipv4, ether_types
from os_ken.topology import event as topo_event
from os_ken.topology.api import get_switch, get_link
from os_ken.lib import hub

import json
import time
import os
import networkx as nx
from collections import defaultdict

# Apply the recommended eventlet migration fix: switch to the asyncio hub
import os
os.environ["EVENTLET_HUB"] = "asyncio"

import eventlet
import eventlet.hubs
eventlet.hubs.use_hub("eventlet.hubs.asyncio")

import eventlet.wsgi
from flask import Flask, jsonify, request, render_template

# Find template folder next to this controller file
HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask("garro_controller_api", template_folder=os.path.join(HERE, "templates"))
controller_instance = None


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/garro/intent", methods=["GET", "POST"])
def manage_intent():
    if controller_instance is None:
        return jsonify({"error": "Controller not initialized"}), 503
    if request.method == "POST":
        body = request.get_json(force=True, silent=True) or {}
        intent = body.get("intent", "").strip()
        if intent:
            controller_instance.current_intent = intent
            controller_instance.intent_status = "pending"
            controller_instance.intent_error = ""
            return jsonify({"status": "ok", "intent": intent})
        return jsonify({"error": "Missing intent parameter"}), 400
    return jsonify({"intent": controller_instance.current_intent})


@app.route("/garro/intent_status", methods=["GET", "POST"])
def manage_intent_status():
    if controller_instance is None:
        return jsonify({"error": "Controller not initialized"}), 503
    if request.method == "POST":
        body = request.get_json(force=True, silent=True) or {}
        status = body.get("status", "").strip()
        if status in ("pending", "processing", "success", "error"):
            controller_instance.intent_status = status
            controller_instance.intent_error = body.get("message", "")
            return jsonify({"status": "ok", "intent_status": status})
        return jsonify({"error": "Invalid status value"}), 400
    return jsonify({
        "intent_status": controller_instance.intent_status,
        "intent_error": controller_instance.intent_error,
    })


@app.route("/garro/weights", methods=["GET", "POST"])
def manage_weights():
    if controller_instance is None:
        return jsonify({"error": "Controller not initialized"}), 503
    if request.method == "POST":
        body = request.get_json(force=True, silent=True) or {}
        weights = body.get("weights")
        if isinstance(weights, dict):
            controller_instance.current_weights = weights
            return jsonify({"status": "ok", "weights": weights})
        return jsonify({"error": "Invalid weights format"}), 400
    return jsonify({"weights": controller_instance.current_weights})


@app.route("/garro/state", methods=["GET"])
def get_state():
    if controller_instance is None:
        return jsonify({"error": "Controller not initialized"}), 503
    state = controller_instance.get_network_state()
    return jsonify(state)


@app.route("/garro/flow", methods=["POST"])
def install_flow():
    if controller_instance is None:
        return jsonify({"error": "Controller not initialized"}), 503
    try:
        body = request.get_json(force=True)
        path = body.get("path", [])
        src_ip = body["src_ip"]
        dst_ip = body["dst_ip"]
        is_fallback = False
        if not path or len(path) < 2:
            # Self-Healing Router (SHR) Fast-Path Fallback
            src_dpid = int(src_ip.split(".")[-1])
            dst_dpid = int(dst_ip.split(".")[-1])
            path = controller_instance.compute_dijkstra_fallback_path(src_dpid, dst_dpid)
            is_fallback = True

        controller_instance.install_path_flow(path, src_ip, dst_ip)
        return jsonify({"status": "ok", "path_used": path, "fallback": is_fallback})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/garro/topology", methods=["GET"])
def get_topology():
    if controller_instance is None:
        return jsonify({"error": "Controller not initialized"}), 503
    nodes = list(controller_instance.topology.nodes())
    edges = [
        {"src": u, "dst": v}
        for u, v in controller_instance.topology.edges()
    ]
    return jsonify({"nodes": nodes, "edges": edges})


@app.route("/garro/hosts", methods=["GET"])
def get_hosts():
    """Return a list of known Mininet hosts derived from the topology.
    Hosts are h1–hN where N = number of switches discovered.
    """
    if controller_instance is None:
        return jsonify({"error": "Controller not initialized"}), 503
    n = len(controller_instance.topology.nodes())
    hosts = [
        {"name": f"h{i}", "ip": f"10.0.0.{i}"}
        for i in range(1, n + 1)
    ]
    return jsonify({"hosts": hosts})


@app.route("/garro/speedtest", methods=["POST"])
def run_speedtest():
    """Network performance probe between two Mininet hosts.

    Body JSON::
        {
          "src_host":  "h1",
          "dst_host":  "h14",
          "dst_ip":    "10.0.0.14",   # optional
          "test_type": "full",         # ping | tcp | udp | traceroute | full
          "duration":  5               # iperf3 seconds (default 5)
        }

    Namespace detection order:
      1. ``ip netns exec <hostname>`` (works if Mininet created named netns)
      2. ``nsenter -t <pid> -n`` (searches /proc/*/net/dev for the host iface)
      3. Fall back with an informative error message.
    """
    import subprocess
    import re
    import json as _json
    import time as _time

    if controller_instance is None:
        return jsonify({"error": "Controller not initialized"}), 503

    body      = request.get_json(force=True, silent=True) or {}
    src_host  = body.get("src_host", "h1")
    dst_host  = body.get("dst_host", "h2")
    duration  = int(body.get("duration", 5))
    test_type = body.get("test_type", "full")   # ping|tcp|udp|traceroute|full

    # Derive destination IP
    dst_ip = body.get("dst_ip") or None
    if dst_ip is None:
        m = re.match(r"h(\d+)", dst_host)
        if m:
            dst_ip = f"10.0.0.{m.group(1)}"
        else:
            return jsonify({"error": f"Cannot infer IP for {dst_host}"}), 400

    src_ip = None
    m = re.match(r"h(\d+)", src_host)
    if m:
        src_ip = f"10.0.0.{m.group(1)}"

    result = {
        "src_host":  src_host,
        "dst_host":  dst_host,
        "dst_ip":    dst_ip,
        "test_type": test_type,
        "ping":      None,
        "iperf":     None,
        "udp":       None,
        "traceroute": None,
        "errors":    []
    }

    # ─────────────────────────────────────────────────────────────────
    # Namespace detection
    # Returns the command prefix list to run a command inside the host
    # network namespace, or None if we can't find it.
    # ─────────────────────────────────────────────────────────────────
    def _get_ns_prefix(hostname):
        # Use named network namespace created by NamespacedHost in mininet_nsfnet.py.
        # setns() requires CAP_SYS_ADMIN; try without sudo first (controller may be
        # root), then fall back to sudo (requires NOPASSWD sudoers entry for
        # 'ip netns exec').
        for prefix in (
            ["ip", "netns", "exec", hostname],
            ["sudo", "-n", "ip", "netns", "exec", hostname],
        ):
            try:
                test = subprocess.run(
                    prefix + ["true"],
                    capture_output=True, timeout=2
                )
                if test.returncode == 0:
                    return prefix
            except Exception:
                pass

        return None  # Namespace not found — topology may not use NamespacedHost

    src_prefix = _get_ns_prefix(src_host)
    dst_prefix = _get_ns_prefix(dst_host)

    if src_prefix is None:
        result["errors"].append(
            f"Cannot find network namespace for {src_host}. "
            "Ensure Mininet is running and named netns or nsenter is available."
        )

    do_ping  = test_type in ("ping",  "full")
    do_tcp   = test_type in ("tcp",   "full")
    do_udp   = test_type in ("udp",   "full")
    do_trace = test_type in ("traceroute", "full")

    # ── 1. Ping ──────────────────────────────────────────────────────
    if do_ping and src_prefix:
        try:
            ping_cmd = src_prefix + [
                "ping", "-c", "10", "-i", "0.2", "-W", "2", dst_ip
            ]
            ping_out = subprocess.run(
                ping_cmd, capture_output=True, text=True, timeout=30
            )
            raw = ping_out.stdout + ping_out.stderr

            rtt_m  = re.search(r"rtt min/avg/max/mdev = ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)", raw)
            loss_m = re.search(r"(\d+)% packet loss", raw)
            pkt_m  = re.search(r"(\d+) packets transmitted, (\d+) received", raw)

            if rtt_m:
                result["ping"] = {
                    "rtt_min_ms":      float(rtt_m.group(1)),
                    "rtt_avg_ms":      float(rtt_m.group(2)),
                    "rtt_max_ms":      float(rtt_m.group(3)),
                    "rtt_mdev_ms":     float(rtt_m.group(4)),
                    "packet_loss_pct": int(loss_m.group(1)) if loss_m else None,
                    "tx": int(pkt_m.group(1)) if pkt_m else None,
                    "rx": int(pkt_m.group(2)) if pkt_m else None,
                    "raw": raw.strip()
                }
            else:
                result["errors"].append(f"ping parse failed: {raw.strip()[:300]}")
        except subprocess.TimeoutExpired:
            result["errors"].append("ping timed out")
        except Exception as exc:
            result["errors"].append(f"ping error: {exc}")

    # ── 2. TCP iperf3 ────────────────────────────────────────────────
    if do_tcp and src_prefix and dst_prefix:
        try:
            server_cmd = dst_prefix + ["iperf3", "-s", "-1"]
            server_proc = subprocess.Popen(
                server_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            _time.sleep(0.6)

            client_cmd = src_prefix + [
                "iperf3", "-c", dst_ip, "-t", str(duration), "-J"
            ]
            client_out = subprocess.run(
                client_cmd, capture_output=True, text=True, timeout=duration + 20
            )
            try:
                data = _json.loads(client_out.stdout)
                end  = data.get("end", {})
                sent = end.get("sum_sent", {})
                recv = end.get("sum_received", {})
                result["iperf"] = {
                    "mbps_sent":       round(sent.get("bits_per_second", 0) / 1e6, 2),
                    "mbps_received":   round(recv.get("bits_per_second", 0) / 1e6, 2),
                    "bytes_sent":      sent.get("bytes"),
                    "bytes_received":  recv.get("bytes"),
                    "retransmits":     sent.get("retransmits"),
                    "duration_s":      duration,
                }
            except (_json.JSONDecodeError, KeyError, TypeError):
                result["errors"].append(f"iperf3 TCP parse failed: {client_out.stdout[:200]}")
            try:
                server_proc.terminate()
            except Exception:
                pass
        except subprocess.TimeoutExpired:
            result["errors"].append("iperf3 TCP timed out")
        except FileNotFoundError:
            result["errors"].append("iperf3 not found — sudo apt install iperf3")
        except Exception as exc:
            result["errors"].append(f"iperf3 TCP error: {exc}")

    # ── 3. UDP iperf3 (jitter & loss) ────────────────────────────────
    if do_udp and src_prefix and dst_prefix:
        try:
            srv_cmd = dst_prefix + ["iperf3", "-s", "-1"]
            srv_proc = subprocess.Popen(
                srv_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            _time.sleep(0.6)

            udp_cmd = src_prefix + [
                "iperf3", "-c", dst_ip, "-u",
                "-t", str(duration),
                "-b", "100M",   # 100 Mbps UDP load
                "-J"
            ]
            udp_out = subprocess.run(
                udp_cmd, capture_output=True, text=True, timeout=duration + 20
            )
            try:
                data = _json.loads(udp_out.stdout)
                end  = data.get("end", {})
                # UDP receiver summary is under sum (not sum_received)
                summ = end.get("sum", {})
                result["udp"] = {
                    "jitter_ms":    summ.get("jitter_ms"),
                    "lost_percent": summ.get("lost_percent"),
                    "packets":      summ.get("packets"),
                    "mbps": round(summ.get("bits_per_second", 0) / 1e6, 2),
                }
            except (_json.JSONDecodeError, KeyError, TypeError):
                result["errors"].append(f"iperf3 UDP parse failed: {udp_out.stdout[:200]}")
            try:
                srv_proc.terminate()
            except Exception:
                pass
        except subprocess.TimeoutExpired:
            result["errors"].append("iperf3 UDP timed out")
        except FileNotFoundError:
            result["errors"].append("iperf3 not found — sudo apt install iperf3")
        except Exception as exc:
            result["errors"].append(f"iperf3 UDP error: {exc}")

    # ── 4. Traceroute ────────────────────────────────────────────────
    if do_trace and src_prefix:
        try:
            tr_cmd = src_prefix + [
                "traceroute", "-n", "-m", "20", "-w", "2", "-q", "1", dst_ip
            ]
            tr_out = subprocess.run(
                tr_cmd, capture_output=True, text=True, timeout=60
            )
            hops = []
            for line in tr_out.stdout.splitlines():
                # Lines look like: " 1  10.0.0.1  0.543 ms"  or  " 2  * * *"
                hop_m = re.match(r"\s*(\d+)\s+([\d.*]+)\s+([\d.]+)\s+ms", line)
                star_m = re.match(r"\s*(\d+)\s+\*", line)
                if hop_m:
                    hops.append({
                        "hop": int(hop_m.group(1)),
                        "ip":  hop_m.group(2),
                        "host": hop_m.group(2),
                        "rtt": float(hop_m.group(3)),
                    })
                elif star_m:
                    hops.append({
                        "hop": int(star_m.group(1)),
                        "ip":  None,
                        "host": None,
                        "rtt": None,
                    })
            result["traceroute"] = {
                "hops": hops,
                "raw":  tr_out.stdout.strip()
            }
        except subprocess.TimeoutExpired:
            result["errors"].append("traceroute timed out")
        except FileNotFoundError:
            result["errors"].append("traceroute not found — sudo apt install traceroute")
        except Exception as exc:
            result["errors"].append(f"traceroute error: {exc}")

    return jsonify(result)

# ── City / Node label maps (used by get_network_state for UI display) ──────
NSFNET_LABELS = {
    1: "Seattle", 2: "Palo Alto", 3: "San Diego", 4: "Salt Lake City",
    5: "Boulder", 6: "Lincoln", 7: "Houston", 8: "Champaign",
    9: "Atlanta", 10: "Ann Arbor", 11: "Pittsburgh", 12: "Princeton",
    13: "College Park", 14: "Ithaca",
}
GEANT2_LABELS = {
    1: "London", 2: "Amsterdam", 3: "Frankfurt", 4: "Paris",
    5: "Brussels", 6: "Geneva", 7: "Milan", 8: "Zurich",
    9: "Vienna", 10: "Prague", 11: "Warsaw", 12: "Budapest",
    13: "Bucharest", 14: "Athens", 15: "Istanbul", 16: "Zagreb",
    17: "Ljubljana", 18: "Bratislava", 19: "Copenhagen", 20: "Stockholm",
    21: "Helsinki", 22: "Tallinn", 23: "Riga", 24: "Vilnius",
}

class GARROController(app_manager.OSKenApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global controller_instance
        controller_instance = self

        # Network state
        self.topology: nx.DiGraph = nx.DiGraph()
        self.datapaths: dict = {}          # dpid → datapath object
        self.port_stats: dict = defaultdict(dict)   # dpid → port stats
        self.flow_stats: dict = defaultdict(dict)
        self.mac_to_port: dict = defaultdict(dict)  # dpid → mac → port
        self.pending_flows: list = []       # Flow rules waiting to be installed
        self.active_paths: dict = {}        # "src_ip->dst_ip" -> dpid list
        self.current_intent: str = "Balance load across all links while maintaining reasonable latency for mixed traffic."
        self.current_weights: dict = {"alpha1": 0.4, "alpha2": 0.3, "alpha3": 0.2, "alpha4": 0.1}
        self.intent_status: str = "success"   # pending | processing | success | error
        self.intent_error: str = ""

        # Tracks (dpid, src_mac) pairs that have already been flooded this
        # cycle. Cleared every 30s (matching flow idle_timeout) so hosts can
        # re-ARP after flow entries expire. Using a set avoids broadcast storms
        # in the looped mesh topology without permanently silencing hosts.
        self.flooded_srcs: dict = {}

        # Proactive forwarding state
        self._proactive_installed = False   # True once proactive flows are in
        self._topo_stable_time = None       # Timestamp of last topology change

        # Telemetry polling thread (every 2 seconds)
        self.monitor_thread = hub.spawn(self._monitor_loop)

        # Flask REST API thread (running on port 8080)
        self.flask_thread = hub.spawn(self._run_flask_server)

        # Periodic flood-set clearer (every 30 seconds)
        self.flood_clear_thread = hub.spawn(self._clear_flooded_srcs)

        # Topology stabilization watcher — installs proactive flows once
        # no new switches/links appear for a few seconds
        self._topo_watcher_thread = hub.spawn(self._topo_stabilisation_watcher)

    def _run_flask_server(self):
        """Runs the Flask REST API on eventlet's WSGI server."""
        self.logger.info("[GARRO] Starting REST API on http://127.0.0.1:8080")
        try:
            # Silence default WSGI logging to avoid flooding the controller output
            wsgi_logger = open("/dev/null", "w")
            eventlet.wsgi.server(
                eventlet.listen(("127.0.0.1", 8080)),
                app,
                log=wsgi_logger
            )
        except Exception as e:
            self.logger.error(f"[GARRO] REST API server failed to start: {e}")

    # ── OpenFlow Event Handlers ────────────────────────────────────────────

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Install table-miss flow entry on every new switch."""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        self.datapaths[datapath.id] = datapath

        # Table-miss: send unmatched packets to controller
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(
            ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER
        )]
        self._add_flow(datapath, 0, match, actions)
        self.logger.info(f"[GARRO] Switch connected: dpid={datapath.id:016x}")

    @set_ev_cls(topo_event.EventSwitchEnter)
    def switch_enter(self, ev):
        self._update_topology()
        self._mark_topo_changed()

    @set_ev_cls(topo_event.EventLinkAdd)
    def link_add(self, ev):
        self._update_topology()
        self._mark_topo_changed()

    @set_ev_cls(topo_event.EventLinkDelete)
    def link_delete(self, ev):
        self._update_topology()
        self._mark_topo_changed()

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Basic L2 learning switch for non-DRL-managed flows.

        Handles:
        - LLDP:      ignored (topology module handles it)
        - Broadcast: flooded with per-switch loop detection to prevent storms
                     in the mesh topology (ARP requests, etc.)
        - Unicast:   forwarded via MAC table; flooded only if dst unknown
        """
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        in_port = msg.match["in_port"]
        dpid = dp.id

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return   # Let topology module handle LLDP

        dst = eth.dst
        src = eth.src

        # Unicast or Broadcast: look up destination in MAC table.
        # Broadcast MACs won't be found, so they default to OFPP_FLOOD.
        out_port = self.mac_to_port[dpid].get(dst, ofp.OFPP_FLOOD)

        if out_port == ofp.OFPP_FLOOD:
            flood_key = (dpid, hash(msg.data))
            if flood_key in self.flooded_srcs:
                # Already flooded this exact packet on this switch.
                # It's re-circulating through a mesh loop — drop it.
                return
            import time
            self.flooded_srcs[flood_key] = time.time()

        # Learn source MAC → in_port (after loop detection to avoid poisoning MAC table!)
        self.mac_to_port[dpid][src] = in_port

        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofp.OFPP_FLOOD:
            # Install forward flow
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst)
            self._add_flow(dp, 1, match, actions, idle_timeout=30)

            # Install reverse flow so replies don't re-hit the controller
            known_src_port = self.mac_to_port[dpid].get(src)
            if known_src_port is not None:
                rev_match = parser.OFPMatch(in_port=out_port, eth_dst=src)
                rev_actions = [parser.OFPActionOutput(known_src_port)]
                self._add_flow(dp, 1, rev_match, rev_actions, idle_timeout=30)

        data = msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None
        out = parser.OFPPacketOut(
            datapath=dp,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data,
        )
        dp.send_msg(out)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply(self, ev):
        dpid = ev.msg.datapath.id
        for stat in ev.msg.body:
            self.port_stats[dpid][stat.port_no] = {
                "rx_bytes": stat.rx_bytes,
                "tx_bytes": stat.tx_bytes,
                "rx_packets": stat.rx_packets,
                "tx_packets": stat.tx_packets,
                "rx_dropped": stat.rx_dropped,
                "tx_dropped": stat.tx_dropped,
                "rx_errors": stat.rx_errors,
            }

    # ── Telemetry Polling ──────────────────────────────────────────────────

    def _clear_flooded_srcs(self):
        """Clear the broadcast flood-tracking dict safely.

        Entries are kept for at least 1 second to ensure that propagating 
        broadcasts are fully suppressed across all mesh loops before being forgotten.
        """
        import time
        while True:
            hub.sleep(1)
            now = time.time()
            stale = [k for k, v in self.flooded_srcs.items() if now - v > 1.0]
            for k in stale:
                del self.flooded_srcs[k]

    def _monitor_loop(self):
        """Continuously poll switch statistics every 2 seconds."""
        while True:
            for dpid, dp in list(self.datapaths.items()):
                self._request_port_stats(dp)
            hub.sleep(2)

    def _request_port_stats(self, datapath):
        parser = datapath.ofproto_parser
        req = parser.OFPPortStatsRequest(
            datapath, 0, datapath.ofproto.OFPP_ANY
        )
        datapath.send_msg(req)

    # ── Topology Management ────────────────────────────────────────────────

    def _update_topology(self):
        """Rebuild NetworkX graph from OS-Ken topology API."""
        switches = get_switch(self, None)
        links = get_link(self, None)

        self.topology.clear()
        for sw in switches:
            self.topology.add_node(sw.dp.id)

        for link in links:
            self.topology.add_edge(
                link.src.dpid, link.dst.dpid,
                src_port=link.src.port_no,
                dst_port=link.dst.port_no,
                bandwidth=1000,
                delay=1.0,
                utilization=0.0,
                packet_loss=0.0,
            )

    def _mark_topo_changed(self):
        """Record that the topology just changed; reset proactive state."""
        self._topo_stable_time = time.time()
        self._proactive_installed = False

    def _topo_stabilisation_watcher(self):
        """Wait until topology stops changing, then install proactive flows.

        After each topology event we record the timestamp.  This thread
        checks every 2 s whether 5 seconds have passed without changes.
        Once stable, it computes shortest paths and pushes forwarding
        rules for every host pair — eliminating the need for flood-based
        MAC learning which fails in dense mesh topologies like GEANT2.
        """
        STABLE_WAIT = 5          # seconds of quiet before we act
        while True:
            hub.sleep(2)
            if self._proactive_installed:
                continue
            if self._topo_stable_time is None:
                continue
            if time.time() - self._topo_stable_time < STABLE_WAIT:
                continue
            # Topology has been stable long enough — install proactive flows
            n_nodes = self.topology.number_of_nodes()
            n_edges = self.topology.number_of_edges()
            if n_nodes == 0:
                continue
            self.logger.info(
                f"[GARRO] Topology stable ({n_nodes} nodes, {n_edges} edges). "
                f"Installing proactive shortest-path flows..."
            )
            try:
                self._install_proactive_flows()
                self._proactive_installed = True
                self.logger.info(
                    "[GARRO] Proactive flows installed successfully."
                )
            except Exception as e:
                self.logger.error(
                    f"[GARRO] Proactive flow installation failed: {e}"
                )

    def _install_proactive_flows(self):
        """Compute shortest paths and install L2+L3 forwarding for all pairs.

        For each destination host (connected to switch N on port 1):
          - At switch N itself: deliver to port 1 (local host).
          - At every other switch: forward towards next hop on the
            shortest path to switch N.

        Both IP-match (priority 10) and MAC-match (priority 10) rules
        are installed so that ARP replies and IP data traffic are both
        forwarded correctly without flooding.
        """
        # Build an undirected version for shortest-path computation
        G = self.topology.to_undirected()
        nodes = sorted(G.nodes())

        # Pre-compute all-pairs shortest paths (node lists)
        all_paths = dict(nx.all_pairs_shortest_path(G))

        for dst_dpid in nodes:
            # Host i is connected to switch i on port 1
            # Host IP = 10.0.0.<dpid>, MAC = 00:00:00:00:00:<dpid hex>
            dst_ip = f"10.0.0.{dst_dpid}"
            dst_mac = f"00:00:00:00:00:{dst_dpid:02x}"

            for src_dpid in nodes:
                if src_dpid == dst_dpid:
                    # Local delivery: traffic for this switch's own host
                    dp = self.datapaths.get(src_dpid)
                    if dp is None:
                        continue
                    parser = dp.ofproto_parser
                    # IP-based rule
                    match_ip = parser.OFPMatch(
                        eth_type=0x0800, ipv4_dst=dst_ip
                    )
                    actions = [parser.OFPActionOutput(1)]  # host port
                    self._add_flow(dp, 10, match_ip, actions)
                    # MAC-based rule (for ARP replies)
                    match_mac = parser.OFPMatch(eth_dst=dst_mac)
                    self._add_flow(dp, 10, match_mac, actions)
                    continue

                path = all_paths.get(src_dpid, {}).get(dst_dpid)
                if path is None:
                    self.logger.warning(
                        f"[GARRO] No path from {src_dpid} to {dst_dpid}"
                    )
                    continue

                # Install a forwarding rule at each hop along the path
                for idx in range(len(path) - 1):
                    current = path[idx]
                    nxt = path[idx + 1]
                    dp = self.datapaths.get(current)
                    if dp is None:
                        continue
                    parser = dp.ofproto_parser

                    # Find the output port from 'current' towards 'nxt'
                    edge = self.topology.edges.get((current, nxt))
                    if edge is None:
                        # Try reverse direction
                        edge_rev = self.topology.edges.get((nxt, current))
                        if edge_rev is None:
                            continue
                        out_port = edge_rev["dst_port"]
                    else:
                        out_port = edge["src_port"]

                    actions = [parser.OFPActionOutput(out_port)]

                    # IP-based forwarding (data traffic)
                    match_ip = parser.OFPMatch(
                        eth_type=0x0800, ipv4_dst=dst_ip
                    )
                    self._add_flow(dp, 10, match_ip, actions)

                    # MAC-based forwarding (ARP replies)
                    match_mac = parser.OFPMatch(eth_dst=dst_mac)
                    self._add_flow(dp, 10, match_mac, actions)

        self.logger.info(
            f"[GARRO] Proactive flows: {len(nodes)} hosts, "
            f"{len(nodes) * (len(nodes) - 1)} paths installed."
        )

    # ── Flow Installation ──────────────────────────────────────────────────

    def _add_flow(self, datapath, priority, match, actions,
                  idle_timeout=0, hard_timeout=0):
        ofp = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=idle_timeout,
            hard_timeout=hard_timeout,
        )
        datapath.send_msg(mod)

    def compute_dijkstra_fallback_path(self, src_dpid: int, dst_dpid: int) -> list:
        """
        Self-Healing Router (SHR) Fast-Path Fallback Engine.
        Computes a deterministic Dijkstra shortest path based on current link delays.
        Used as immediate failover if AI Decision Plane fails or times out.
        """
        try:
            G = self.topology.to_undirected()
            if src_dpid in G and dst_dpid in G:
                path = nx.dijkstra_path(G, src_dpid, dst_dpid, weight="delay")
                self.logger.info(f"[GARRO/SHR] Fast-Path Dijkstra Fallback computed path: {path}")
                return path
        except Exception as e:
            self.logger.error(f"[GARRO/SHR] Fast-Path Dijkstra Fallback error: {e}")
        return []

    def install_path_flow(self, path: list, src_ip: str, dst_ip: str,
                          priority: int = 100):
        """
        Install flow rules along a computed path.
        path: list of dpid values [dpid1, dpid2, ..., dpidN]
        """
        if len(path) < 2:
            self.logger.warning("[GARRO] Path too short to install flows")
            return

        for i in range(len(path) - 1):
            current_dpid = path[i]
            next_dpid = path[i + 1]
            dp = self.datapaths.get(current_dpid)
            if dp is None:
                continue

            edge_data = self.topology.edges.get((current_dpid, next_dpid))
            if edge_data is None:
                continue

            out_port = edge_data["src_port"]
            parser = dp.ofproto_parser

            match = parser.OFPMatch(
                eth_type=0x0800,
                ipv4_src=src_ip,
                ipv4_dst=dst_ip,
            )
            actions = [parser.OFPActionOutput(out_port)]
            self._add_flow(dp, priority, match, actions,
                           idle_timeout=60, hard_timeout=120)

        # Track this path as active
        self.active_paths[f"{src_ip}->{dst_ip}"] = path

        self.logger.info(
            f"[GARRO] Installed path: {path} for {src_ip} → {dst_ip}"
        )

    # ── REST API Data Builder ──────────────────────────────────────────────

    def get_network_state(self) -> dict:
        """Build network state JSON for the AI plane."""
        num_nodes = self.topology.number_of_nodes()
        # Pick label map by node count
        if num_nodes <= 14:
            label_map = NSFNET_LABELS
            topo_name = "nsfnet"
        else:
            label_map = GEANT2_LABELS
            topo_name = "geant2"

        nodes = []
        for n in self.topology.nodes():
            stats = self.port_stats.get(n, {})
            total_rx = sum(s.get("rx_bytes", 0) for s in stats.values())
            total_tx = sum(s.get("tx_bytes", 0) for s in stats.values())
            nodes.append({
                "dpid": n,
                "label": label_map.get(n, f"Node {n}"),
                "cpu": 0.5,          # Placeholder — extend with SNMP
                "buffer_occ": 0.3,
                "ingress_rate": total_rx,
                "egress_rate": total_tx,
            })

        edges = []
        for u, v, data in self.topology.edges(data=True):
            edges.append({
                "src": u, "dst": v,
                "bandwidth": data.get("bandwidth", 1000),
                "utilization": data.get("utilization", 0.0),
                "delay": data.get("delay", 1.0),
                "packet_loss": data.get("packet_loss", 0.0),
                "src_port": data.get("src_port", 0),
                "dst_port": data.get("dst_port", 0),
            })

        return {
            "timestamp": time.time(),
            "topology": topo_name,
            "nodes": nodes,
            "edges": edges,
            "active_paths": self.active_paths,
            "current_intent": self.current_intent,
            "current_weights": self.current_weights,
            "intent_status": self.intent_status,
            "intent_error": self.intent_error,
        }
