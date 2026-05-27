#!/usr/bin/env python3
"""Quick smoke-test bot: connects to a room as an SDK client and drives
randomly so a watching browser tab sees a ghost car moving.

Defaults to the local dev server. Pass --host to point at prod.

Usage:
    python test_bot.py
    python test_bot.py --name Alice --host ml.ferit.tech --secure
    python test_bot.py --host localhost:3001 --name Bob
"""
import argparse
import json
import random
import threading
import time
import websocket


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost:3001",
                    help="host:port (default localhost:3001 — Vite dev proxies /ws to here)")
    ap.add_argument("--secure", action="store_true",
                    help="use wss:// instead of ws://")
    ap.add_argument("--room", default="main")
    ap.add_argument("--name", default=f"RandomBot-{random.randint(100, 999)}")
    ap.add_argument("--api-key", default=None,
                    help="required only when server runs in non-dev mode")
    ap.add_argument("--rate-hz", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed for reproducibility")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    scheme = "wss" if args.secure else "ws"
    url = f"{scheme}://{args.host}/ws/room/bot?room={args.room}&name={args.name}"
    if args.api_key:
        url += f"&api_key={args.api_key}"

    print(f"Connecting to {url}")

    running = True
    bot_key_holder = {"key": None}

    def on_open(ws):
        print(f"[{args.name}] connected — sending ready")
        ws.send(json.dumps({"type": "ready", "ready": True}))

    def on_message(ws, raw):
        try:
            msg = json.loads(raw)
        except Exception:
            return
        t = msg.get("type")
        if t == "bot_assigned":
            bot_key_holder["key"] = msg.get("bot_key")
            print(f"[{args.name}] bot_key={bot_key_holder['key']}")
        elif t == "round_start":
            print(f"[{args.name}] round_start idx={msg.get('round_index')} "
                  f"seed={msg.get('seed')} obstacles={msg.get('obstacles')}")
        elif t == "round_end":
            print(f"[{args.name}] round_end idx={msg.get('round_index')}")
        elif t == "tournament_end":
            print(f"[{args.name}] tournament_end")
            standings = msg.get("standings", [])
            for r in standings:
                print(f"  #{r.get('rank')} {r.get('name')} cps={r.get('total_checkpoints')}")
        elif t == "error":
            print(f"[{args.name}] error: {msg.get('code')} {msg.get('message')}")

    def on_error(ws, err):
        print(f"[{args.name}] error: {err}")

    def on_close(ws, code, reason):
        nonlocal running
        running = False
        print(f"[{args.name}] disconnected ({code} {reason})")

    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
    ws_thread.start()

    # Wait briefly for connection
    time.sleep(0.5)

    period = 1.0 / args.rate_hz
    # Smooth-ish random walk so the car isn't strobing direction every tick.
    throttle = 0.5
    steering = 0.0
    try:
        while running:
            throttle += random.uniform(-0.15, 0.15)
            throttle = max(-1.0, min(1.0, throttle))
            steering += random.uniform(-0.3, 0.3)
            steering = max(-1.0, min(1.0, steering))
            # Bias toward forward motion so spectators see progress.
            cmd_throttle = max(0.3, throttle)
            try:
                ws.send(json.dumps({
                    "type": "control",
                    "throttle": cmd_throttle,
                    "steering": steering,
                }))
            except Exception:
                break
            time.sleep(period)
    except KeyboardInterrupt:
        print(f"\n[{args.name}] interrupted")
    finally:
        try:
            ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
