"""
GARRO Arena server — live, side-by-side comparison of GARRO vs ECMP / Hedera /
OSPF on a Fat-Tree digital twin.  Runs standalone (no Mininet, no sudo).

    python arena_server.py --checkpoint checkpoints/garro_fat_tree_ep20000.pt
    # → open http://127.0.0.1:8090/

Endpoints
---------
GET  /                  Arena dashboard (arena/static/arena.html)
GET  /arena/topology    Static topology + ToD model parameters
GET  /arena/status      Model / runtime status (also used by the Live dashboard)
GET  /arena/stream      Server-Sent Events: one JSON frame per simulation tick
POST /arena/control     {"action": ..., ...} — see _handle_control()
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import yaml
from aiohttp import web

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from arena.simulator import ArenaSimulator  # noqa: E402
from model.ppo_agent import PPOAgent, _best_device  # noqa: E402

STATIC = ROOT / "arena" / "static"

# Speed presets: (ticks per second, simulated minutes per tick)
SPEEDS = {
    "1x":  (1.0, 1),
    "10x": (2.0, 5),
    "60x": (4.0, 15),
}


class ArenaApp:
    def __init__(self, sim: ArenaSimulator, status: dict):
        self.sim = sim
        self.status = status
        self.running = True
        self.speed = "10x"
        self.clients: set[asyncio.Queue] = set()
        sim.set_minutes_per_tick(SPEEDS[self.speed][1])

    # ── broadcasting ─────────────────────────────────────────────────────

    def _publish(self, frame: dict) -> None:
        data = json.dumps(frame, separators=(",", ":"))
        for q in list(self.clients):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(data)

    def _runtime(self) -> dict:
        return {"type": "runtime", "running": self.running, "speed": self.speed,
                "status": self.status}

    async def sim_loop(self) -> None:
        while True:
            if not self.running:
                await asyncio.sleep(0.1)
                continue
            tps, _ = SPEEDS[self.speed]
            t0 = time.perf_counter()
            try:
                frame = await asyncio.to_thread(self.sim.tick)
                frame["running"] = self.running
                frame["speed"] = self.speed
                self._publish(frame)
            except Exception as exc:  # keep the arena alive no matter what
                print(f"[Arena] tick failed: {exc!r}")
            await asyncio.sleep(max(0.02, 1.0 / tps - (time.perf_counter() - t0)))

    # ── HTTP handlers ────────────────────────────────────────────────────

    async def index(self, _req: web.Request) -> web.StreamResponse:
        return web.FileResponse(STATIC / "arena.html")

    async def topology(self, _req: web.Request) -> web.Response:
        return web.json_response(self.sim.topology())

    async def get_status(self, _req: web.Request) -> web.Response:
        return web.json_response({**self.status, "running": self.running, "speed": self.speed})

    async def stream(self, req: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        })
        await resp.prepare(req)
        q: asyncio.Queue = asyncio.Queue(maxsize=4)
        self.clients.add(q)
        try:
            await resp.write(f"data: {json.dumps(self._runtime())}\n\n".encode())
            if self.sim.last_frame:
                await resp.write(f"data: {json.dumps(self.sim.last_frame)}\n\n".encode())
            while True:
                try:
                    data = await asyncio.wait_for(q.get(), timeout=15)
                    await resp.write(f"data: {data}\n\n".encode())
                except asyncio.TimeoutError:
                    await resp.write(b": keep-alive\n\n")
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self.clients.discard(q)
        return resp

    async def control(self, req: web.Request) -> web.Response:
        try:
            body = await req.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        try:
            result = await asyncio.to_thread(self._handle_control, body)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)
        self._publish(self._runtime())
        return web.json_response({"ok": True, **(result or {})})

    def _handle_control(self, b: dict) -> dict | None:
        a = b.get("action")
        s = self.sim
        if a == "play":
            self.running = True
        elif a == "pause":
            self.running = False
        elif a == "speed":
            if b.get("value") not in SPEEDS:
                raise ValueError("speed must be one of " + ", ".join(SPEEDS))
            self.speed = b["value"]
            s.set_minutes_per_tick(SPEEDS[self.speed][1])
        elif a == "reset":
            s.reset(seed=b.get("seed"), start_hour=b.get("hour"))
            s.set_minutes_per_tick(SPEEDS[self.speed][1])
        elif a == "load":
            s.set_load(float(b["value"]))
        elif a == "elephant":
            s.set_elephant_prob(float(b["value"]))
        elif a == "arrivals":
            s.set_arrivals(int(b["value"]))
        elif a == "pattern":
            s.set_pattern(str(b["value"]))
        elif a == "hour":
            s.set_hour(float(b["value"]))
        elif a == "event":
            s.trigger_event(str(b["value"]), int(b.get("ticks", 25)))
        elif a == "toggle_link":
            return {"failed": s.toggle_link(int(b["u"]), int(b["v"]))}
        elif a == "fail_random":
            e = s.fail_random_link(b.get("tier", "core"))
            return {"link": list(e) if e else None}
        elif a == "restore_all":
            s.restore_all()
        else:
            raise ValueError(f"unknown action {a!r}")
        return None


def _cors(app_handler):
    @web.middleware
    async def mw(request, handler):
        if request.method == "OPTIONS":
            resp = web.Response()
        else:
            resp = await handler(request)
        resp.headers.setdefault("Access-Control-Allow-Origin", "*")
        resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type")
        resp.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        return resp
    return mw


def build(args) -> tuple[web.Application, ArenaApp]:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["network"]["topology"] = "fat_tree"

    device = _best_device() if args.device == "auto" else args.device
    device = str(device)
    # BF16 autocast is emulated (≈10× slower) on CPUs without AVX-512 BF16;
    # FP32 gives identical decisions for inference.
    if device == "cpu":
        config.setdefault("training", {})["amp_dtype"] = "float32"

    agent = PPOAgent(config, k_paths=config["network"]["k_paths"], num_nodes=20,
                     device=device, compile_model=False)
    loaded, err = False, None
    try:
        agent.load(args.checkpoint)
        loaded = True
    except Exception as exc:
        err = str(exc).splitlines()[0][:300]
        if not args.allow_random:
            print(f"\n[Arena] ✗ Checkpoint failed to load: {err}")
            print("[Arena]   Refusing to run a comparison with random weights.")
            print("[Arena]   Check the path / pull the latest code, or pass --allow-random.\n")
            sys.exit(2)
        print(f"[Arena] ⚠ checkpoint load failed ({err}) — RANDOM weights (--allow-random)")
    agent.encoder.eval()
    agent.ac_net.eval()

    n_params = sum(p.numel() for p in agent.encoder.parameters()) + \
        sum(p.numel() for p in agent.ac_net.parameters())
    status = {
        "checkpoint": os.path.basename(args.checkpoint),
        "loaded": loaded,
        "error": err,
        "params": n_params,
        "device": device,
        "topology": "fat_tree (k=4)",
    }

    print("[Arena] Building digital twin + initial routing for all policies …")
    sim = ArenaSimulator(config, agent=agent, seed=args.seed)
    arena = ArenaApp(sim, status)

    app = web.Application(middlewares=[_cors(None)])
    app.router.add_get("/", arena.index)
    app.router.add_get("/arena", arena.index)
    app.router.add_get("/arena/topology", arena.topology)
    app.router.add_get("/arena/status", arena.get_status)
    app.router.add_get("/arena/stream", arena.stream)
    app.router.add_post("/arena/control", arena.control)
    app.router.add_route("OPTIONS", "/arena/control", lambda r: web.Response())
    app.router.add_static("/static/", STATIC)

    async def _start(_app):
        _app["sim_task"] = asyncio.create_task(arena.sim_loop())

    async def _stop(_app):
        _app["sim_task"].cancel()

    app.on_startup.append(_start)
    app.on_cleanup.append(_stop)
    return app, arena


def main():
    ap = argparse.ArgumentParser(description="GARRO Arena — live policy comparison")
    ap.add_argument("--checkpoint", default="checkpoints/garro_fat_tree_ep20000.pt")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--allow-random", action="store_true",
                    help="Run even if the checkpoint fails to load (NOT for demos)")
    args = ap.parse_args()

    app, arena = build(args)
    st = arena.status
    print("\n" + "=" * 60)
    print("  GARRO Arena")
    print(f"  Model     : {st['checkpoint']}  ({'✓ loaded' if st['loaded'] else '⚠ RANDOM'})")
    print(f"  Params    : {st['params']:,}   Device: {st['device']}")
    print(f"  Dashboard : http://{args.host}:{args.port}/")
    print("=" * 60 + "\n")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
