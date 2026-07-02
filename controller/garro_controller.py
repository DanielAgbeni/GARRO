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
        path = body["path"]        # List of dpid values
        src_ip = body["src_ip"]
        dst_ip = body["dst_ip"]
        controller_instance.install_path_flow(path, src_ip, dst_ip)
        return jsonify({"status": "ok"})
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

        # Telemetry polling thread (every 2 seconds)
        self.monitor_thread = hub.spawn(self._monitor_loop)

        # Flask REST API thread (running on port 8080)
        self.flask_thread = hub.spawn(self._run_flask_server)

        # Periodic flood-set clearer (every 30 seconds)
        self.flood_clear_thread = hub.spawn(self._clear_flooded_srcs)

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

    @set_ev_cls(topo_event.EventLinkAdd)
    def link_add(self, ev):
        self._update_topology()

    @set_ev_cls(topo_event.EventLinkDelete)
    def link_delete(self, ev):
        self._update_topology()

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
        nodes = []
        for n in self.topology.nodes():
            stats = self.port_stats.get(n, {})
            total_rx = sum(s.get("rx_bytes", 0) for s in stats.values())
            total_tx = sum(s.get("tx_bytes", 0) for s in stats.values())
            nodes.append({
                "dpid": n,
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
            "nodes": nodes,
            "edges": edges,
            "active_paths": self.active_paths,
            "current_intent": self.current_intent,
            "current_weights": self.current_weights,
            "intent_status": self.intent_status,
            "intent_error": self.intent_error,
        }
