"""Tournament entry point — hybrid arrow-following + ML obstacle avoidance.

Usage:
    # 1. Train weights first (one time, ~5 min):
    #    python train_v4.py

    # 2. COMPETITION MODE (tournament day — professor runs the room):
    python 99_compete.py --competition --room ROOM_NAME --name YOUR_NAME

    # 3. Practice / benchmark mode (time_trial):
    python 99_compete.py --weights nav_v4.npz --seed 42 --duration 300

    # 4. Benchmark (5 runs, seeds 42 / 7 / 99):
    python 03_benchmark.py --tag v4-hybrid --weights nav_v4.npz --module drive2win.agent --seeds 42 7 99

How the hybrid works:
    ┌─────────────────────────────────────────────────────┐
    │  ARROW LAYER (rule-based P-controller)              │
    │    heading_error → proportional steering            │
    │    checkpoint_distance → throttle trim              │
    │  + ML LAYER (obstacle avoidance)                    │
    │    blends in as front rays get short (< 17 m)       │
    │  + STUCK RECOVERY                                   │
    │    reverse → avoid (turn around obstacle) → reorient│
    │  + EMA smoothing (α=0.7)                            │
    └─────────────────────────────────────────────────────┘

STEER_SIGN: run train_v4.py and read the printed correlation.
    correlation < 0 → STEER_SIGN = -1  (default)
    correlation > 0 → STEER_SIGN = +1
If the bot drives AWAY from checkpoints, flip STEER_SIGN here.
"""
from __future__ import annotations
import argparse
import json
import threading
import time
import numpy as np
import websocket

from game_client import GameClient
from drive2win import nn as nn_mod
from drive2win.normalize import sensors_to_input, clip_action
from drive2win.eval import run_policy

# ── Tunable constants ───────────────────────────────────────────────────────

STEER_SIGN      = -1.0
ARROW_THROTTLE  = 1.0    # max throttle (revert to 0.975 if too aggressive)
ARROW_STEER_KP  = 0.9
OBSTACLE_THRESH = 0.35
ARROW_BLEND_MIN = 0.84   # arrow keeps 84% of the steer vote, ML max 16%
GRACE_FRAMES    = 100
STUCK_THRESH    = 30   # fire recovery sooner (revert to 45 if it triggers too eagerly)
STUCK_SPEED     = 0.3
RECOVERY_FRAMES  = 25    # total backwards frames
RECOVERY_TURN_AT = 12    # frames-remaining at which we start steering during reverse
AVOID_FRAMES     = 8     # forward-burst frames after reverse (≈5 burst + 3 buffer)
AVOID_STEER      = 0.75  # magnitude of the avoid turn (game steer units)
AVOID_THROTTLE   = 1.0   # FULL speed during the forward burst
EMA_ALPHA       = 0.7


# ── Tournament obs adapter ───────────────────────────────────────────────────

def _obs_to_state(msg: dict) -> dict:
    """Convert the tournament room-bot obs dict to the sensors format our policy expects.

    Tournament format (TOURNAMENT.md):
        obs["speed"], obs["navigation"]["heading_error"],
        obs["navigation"]["distance"], obs["rays"], obs["ground_friction"]

    Our policy expects:
        state["sensors"]["speed"], state["sensors"]["heading_error"], ...
    """
    nav = msg.get("navigation") or {}
    sensors = {
        "speed":               float(msg.get("speed", 0.0)),
        "heading_error":       float(nav.get("heading_error", 0.0)),
        "checkpoint_distance": float(nav.get("distance", 0.0)),
        "rays":                list(msg.get("rays", [50.0] * 8)),
        "ground_friction":     float(msg.get("ground_friction", 1.0)),
        "navigation": {
            "checkpoints_completed": int(msg.get("checkpoints_passed", 0) or 0),
        },
    }
    return {
        "sensors":  sensors,
        "position": msg.get("position") or {},
        "heading":  float(msg.get("heading", 0.0)),
    }


# ── Policy factory ───────────────────────────────────────────────────────────

def make_policy(weights_path: str):
    """Build the hybrid policy.

    Returns a callable `(state_dict) -> (throttle, steering)` driven by the
    arrow P-controller, blended with ML steering for close obstacles, with a
    staged reverse → forward-burst state machine for when the car gets stuck.
    """
    w   = nn_mod.load(weights_path)
    ctx = {"frame": 0, "stuck": 0, "recovery": 0,
           "avoid": 0, "avoid_dir": 0.0,
           "ema_t": 0.0, "ema_s": 0.0}

    def policy(game_state: dict) -> tuple[float, float]:
        ctx["frame"] += 1
        frame = ctx["frame"]

        sensors = game_state.get("sensors") or {}
        x = sensors_to_input(sensors)

        heading_error = float(sensors.get("heading_error", 0.0))
        speed         = float(sensors.get("speed", 0.0))

        ml_out   = nn_mod.forward(x, w)
        ml_steer = float(ml_out[1])

        arr_steer = float(np.clip(STEER_SIGN * heading_error * ARROW_STEER_KP, -1.0, 1.0))
        # Steer-trim coefficient lowered from 0.4 → 0.25 to carry more speed
        # through corners (revert to 0.4 if the car starts losing the line).
        thr       = ARROW_THROTTLE * (1.0 - 0.25 * abs(arr_steer))

        front_clearance = min(float(x[3]), float(x[4]), float(x[10]))
        # Cap ML influence at (1 - ARROW_BLEND_MIN) so arrow stays dominant.
        obstacle_factor = float(np.clip(1.0 - front_clearance / OBSTACLE_THRESH,
                                        0.0, 1.0 - ARROW_BLEND_MIN))
        steer = (1.0 - obstacle_factor) * arr_steer + obstacle_factor * ml_steer

        if ctx["recovery"] > 0:
            thr = -1.0
            # At the midpoint of the reverse, sample rays and pick the side
            # with more clearance — then steer that way while still reversing.
            if ctx["recovery"] == RECOVERY_TURN_AT:
                left_clr  = float(x[4]) + float(x[5])   # +45, +90
                right_clr = float(x[10]) + float(x[9])  # -45, -90
                # negative game steer = LEFT (STEER_SIGN = -1 convention)
                ctx["avoid_dir"] = -1.0 if left_clr >= right_clr else 1.0
            if ctx["recovery"] <= RECOVERY_TURN_AT:
                steer = ctx["avoid_dir"] * AVOID_STEER
            else:
                steer = 0.0
            ctx["recovery"] -= 1
            if ctx["recovery"] == 0:
                # Re-sample rays once we've stopped reversing and pick the
                # clearer forward direction, then throttle full speed that way.
                left_clr  = float(x[4]) + float(x[5])
                right_clr = float(x[10]) + float(x[9])
                ctx["avoid_dir"] = -1.0 if left_clr >= right_clr else 1.0
                ctx["avoid"]     = AVOID_FRAMES
                ctx["ema_t"]     = 0.0
                ctx["ema_s"]     = 0.0

        elif ctx["avoid"] > 0:
            thr   = AVOID_THROTTLE                     # full speed forward
            steer = ctx["avoid_dir"] * AVOID_STEER
            ctx["avoid"] -= 1
            # When avoid expires, arrow (ARROW_BLEND_MIN = 0.84) takes back over.

        elif frame > GRACE_FRAMES:
            if speed < STUCK_SPEED:
                ctx["stuck"] += 1
            else:
                ctx["stuck"] = 0
            if ctx["stuck"] >= STUCK_THRESH:
                ctx["recovery"] = RECOVERY_FRAMES
                ctx["stuck"]    = 0

        ctx["ema_t"] = EMA_ALPHA * thr   + (1 - EMA_ALPHA) * ctx["ema_t"]
        ctx["ema_s"] = EMA_ALPHA * steer + (1 - EMA_ALPHA) * ctx["ema_s"]

        return clip_action(np.array([ctx["ema_t"], ctx["ema_s"]]))

    return policy


# ── Practice / benchmark runner (time_trial) ────────────────────────────────

def practice_main(args):
    print("=" * 60)
    print(f"  Drive2Win — PRACTICE mode")
    print(f"  weights  : {args.weights}")
    print(f"  seed     : {args.seed}")
    print(f"  duration : {args.duration:.0f} s")
    print("=" * 60)

    client = GameClient(args.server, args.api_key)
    session = client.create_session(
        mode="time_trial",
        player_name=args.name,
        config={"seed": args.seed, "wind_enabled": False},
    )
    print(f"\nOpen this URL in a browser tab before pressing Enter:")
    print(f"  {session.get('browser_url', '')}")
    input("\nPress Enter once the browser tab is open and the car is visible... ")

    client.connect_ws()
    time.sleep(0.6)

    policy = make_policy(args.weights)

    result = run_policy(client, policy, duration=args.duration, hz=20.0)

    print("\n" + "=" * 60)
    print(f"  checkpoints : {result['checkpoints_passed']}")
    print(f"  crashes     : {result['crashes']}")
    print(f"  stuck streak: {result['min_speed_streak']} frames")
    print(f"  steps       : {result['steps']}")
    print("=" * 60)

    client.disconnect_ws()
    try:
        client.delete_session()
    except Exception:
        pass


# ── Competition runner (tournament day) ──────────────────────────────────────

def competition_main(args):
    """Connect to the professor's room via the room-bot WebSocket and race.

    Based on test_bot.py + TOURNAMENT.md protocol:
      - URL  : wss://ml.ferit.tech/ws/room/bot?room=<room>&name=<name>
      - Ready: send {"type":"ready","ready":true} after connect
      - State: server broadcasts obs dicts each tick (type="state" or similar)
      - Ctrl : send {"type":"control","throttle":x,"steering":y}
      - Events: round_start / round_end / tournament_end
    """
    scheme = "wss" if args.server.startswith("https") else "ws"
    host   = args.server.split("//")[-1]
    url    = f"{scheme}://{host}/ws/room/bot?room={args.room}&name={args.name}"

    print("=" * 60)
    print(f"  Drive2Win — TOURNAMENT mode")
    print(f"  room    : {args.room}")
    print(f"  name    : {args.name}")
    print(f"  weights : {args.weights}")
    print(f"  url     : {url}")
    print("=" * 60)

    # Shared state between WS callback thread and main control loop
    latest_obs  = {"data": None}
    obs_lock    = threading.Lock()
    round_start = threading.Event()
    round_end   = threading.Event()
    tourney_end = threading.Event()
    round_meta  = {"index": 0, "seed": None, "obstacles": False}

    def on_open(ws):
        ws.send(json.dumps({"type": "ready", "ready": True}))
        print(f"Connected — sent ready")

    def on_message(ws, raw):
        try:
            msg = json.loads(raw)
        except Exception:
            return
        t = msg.get("type", "")

        if t == "bot_assigned":
            print(f"Bot key: {msg.get('bot_key')}")

        elif t == "round_start":
            round_meta["index"]     = msg.get("round_index", 0)
            round_meta["seed"]      = msg.get("seed")
            round_meta["obstacles"] = msg.get("obstacles", False)
            print(f"\n>>> ROUND {round_meta['index'] + 1}/5  "
                  f"seed={round_meta['seed']}  "
                  f"obstacles={round_meta['obstacles']}")
            round_end.clear()
            round_start.set()

        elif t == "round_end":
            print(f"<<< ROUND {round_meta['index'] + 1}/5 ended")
            round_start.clear()
            round_end.set()

        elif t == "tournament_end":
            print("\n=== TOURNAMENT ENDED ===")
            for s in msg.get("standings", []):
                print(f"  #{s.get('rank')}  {s.get('name')}  "
                      f"{s.get('total_checkpoints')} checkpoints")
            round_end.set()
            tourney_end.set()

        elif t == "error":
            print(f"Server error: {msg.get('code')} — {msg.get('message')}")

        else:
            # Any other message that looks like a state/obs broadcast
            if "navigation" in msg or "speed" in msg:
                with obs_lock:
                    latest_obs["data"] = msg

    def on_error(ws, err):
        print(f"WebSocket error: {err}")

    def on_close(ws, code, reason):
        print(f"Disconnected ({code} {reason})")
        round_end.set()
        tourney_end.set()

    ws = websocket.WebSocketApp(url,
        on_open=on_open, on_message=on_message,
        on_error=on_error, on_close=on_close)

    ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
    ws_thread.start()
    time.sleep(0.5)

    print("Waiting for professor to start the tournament...")

    round_scores = []
    ROUND_TIMEOUT = 370.0  # 6-min safety cap

    for rnd in range(5):
        if tourney_end.is_set():
            break

        # Wait up to 30 min for the round to start
        if not round_start.wait(timeout=1800):
            print("Timed out waiting for round start.")
            break
        round_start.clear()

        # Fresh policy each round — resets frame/stuck/EMA counters
        policy  = make_policy(args.weights)
        interval = 1.0 / 20.0
        start    = time.time()
        steps    = 0
        max_cp   = 0
        cp_start = 0
        first    = True

        while not round_end.is_set() and time.time() - start < ROUND_TIMEOUT:
            with obs_lock:
                obs = latest_obs["data"]

            if obs:
                # Record baseline checkpoint count at start of round
                current_cp = int(obs.get("checkpoints_passed", 0) or 0)
                if first:
                    cp_start = current_cp
                    first = False
                max_cp = max(max_cp, current_cp - cp_start)

                # Only drive when race is actually running
                phase = obs.get("race_phase", "racing")
                if phase == "racing":
                    state = _obs_to_state(obs)
                    thr, steer = policy(state)
                    try:
                        ws.send(json.dumps({
                            "type":     "control",
                            "throttle": float(np.clip(thr,   -1, 1)),
                            "steering": float(np.clip(steer, -1, 1)),
                        }))
                        steps += 1
                    except Exception:
                        break

            time.sleep(interval)

        elapsed = time.time() - start
        round_scores.append(max_cp)
        print(f"  Round {rnd + 1}: {max_cp} checkpoints  "
              f"({steps} steps, {elapsed:.1f} s)")
        round_end.clear()

    print("\n" + "=" * 60)
    print(f"  Scores by round  : {round_scores}")
    print(f"  Total checkpoints: {sum(round_scores)}")
    print("=" * 60)

    ws.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Drive2Win tournament entry")
    ap.add_argument("--weights",     default="nav_v4.npz")
    ap.add_argument("--name",        default="albertkola",
                    help="Display name on the leaderboard")
    ap.add_argument("--server",      default="https://ml.ferit.tech")

    # Practice mode
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--duration",    type=float, default=300.0)
    ap.add_argument("--api-key",     default="None")

    # Tournament mode
    ap.add_argument("--competition", action="store_true",
                    help="Join the professor's room (tournament day)")
    ap.add_argument("--room",        default="",
                    help="Room name announced by professor (required for --competition)")

    args = ap.parse_args()

    if args.competition:
        if not args.room:
            print("ERROR: --room is required for tournament mode.")
            print("  python 99_compete.py --competition --room final2026 --name albertkola")
            return
        competition_main(args)
    else:
        practice_main(args)


if __name__ == "__main__":
    main()
