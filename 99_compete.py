"""Tournament entry point — hybrid arrow-following + ML obstacle avoidance.

Usage:
    # 1. Train weights first (one time, ~5 min):
    #    python train_v4.py

    # 2. COMPETITION MODE (tournament day — professor runs the room):
    python 99_compete.py --competition --student-id YOUR_ID --name YOUR_NAME

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
    │    forces reverse + hard steer after 60 stuck frames│
    │  + EMA smoothing (α=0.7)                            │
    └─────────────────────────────────────────────────────┘

STEER_SIGN: run train_v4.py and read the printed correlation.
    correlation < 0 → STEER_SIGN = -1  (default)
    correlation > 0 → STEER_SIGN = +1
If the bot drives AWAY from checkpoints, flip STEER_SIGN here.
"""
from __future__ import annotations
import argparse
import threading
import time
import numpy as np

from game_client import GameClient
from drive2win import nn as nn_mod
from drive2win.normalize import sensors_to_input, clip_action
from drive2win.eval import run_policy

# ── Tunable constants — adjust here before tournament ───────────────────────

# Arrow following
STEER_SIGN      = -1.0   # flip to +1.0 if car steers AWAY from checkpoints
ARROW_THROTTLE  = 0.85   # base speed when path is clear — raised for lap time
ARROW_STEER_KP  = 0.9    # heading_error → steering gain [0.5–1.5]
STRAIGHT_BOOST  = 0.15   # extra throttle added when heading is aligned (< 15°)
CORNER_FACTOR   = 0.25   # throttle reduction per unit of |steering| (was 0.4)

# ML obstacle blend
# ML activates only when an obstacle is within OBSTACLE_THRESH × 50 m.
# Smaller value = car brakes for obstacles later = faster on clear terrain.
OBSTACLE_THRESH = 0.25  # 0.25 × 50 m = 12.5 m look-ahead (was 0.35 / 17.5 m)

# Stuck recovery
GRACE_FRAMES      = 100   # skip stuck detection while car gets up to speed at spawn
STUCK_THRESH      = 60    # frames at low speed before forcing reverse
RECOVERY_FRAMES   = 60    # frames of forced reverse
STUCK_SPEED       = 0.3   # m/s threshold for "stuck"
REORIENT_FRAMES   = 25    # frames of high-gain steering after recovery ends
REORIENT_KP       = 1.5   # stronger steering gain to snap back toward checkpoint
WIGGLE_THRESH     = 150   # frames stuck before trying rapid wiggle (physics trap escape)
WIGGLE_PERIOD     = 3     # flip throttle every N frames during wiggle

# EMA smoothing (from CHECKPOINT.md §5.4 — confirmed safe, helps steering jitter)
EMA_ALPHA         = 0.7


# ── Policy factory ───────────────────────────────────────────────────────────

def make_policy(weights_path: str):
    """Build the hybrid policy.  Called by 03_benchmark.py when used as --module.

    Returns:
        policy_fn: (state_dict) -> (throttle, steering)
    """
    w = nn_mod.load(weights_path)

    ctx = {
        "frame": 0,
        "stuck": 0,
        "recovery": 0,   # countdown: > 0 means we are in forced-reverse phase
        "reorient": 0,   # countdown: > 0 means snap-steer phase after recovery
        "ema_t": 0.0,
        "ema_s": 0.0,
    }

    def policy(game_state: dict) -> tuple[float, float]:
        ctx["frame"] += 1
        frame = ctx["frame"]

        sensors = game_state.get("sensors") or {}
        x = sensors_to_input(sensors)

        heading_error = float(sensors.get("heading_error", 0.0))
        speed         = float(sensors.get("speed", 0.0))

        # ── ML forward pass ──────────────────────────────────────────────
        ml_out   = nn_mod.forward(x, w)
        ml_thr   = float(ml_out[0])
        ml_steer = float(ml_out[1])

        # ── Arrow-following P-controller ─────────────────────────────────
        raw_steer = STEER_SIGN * heading_error * ARROW_STEER_KP
        arr_steer = float(np.clip(raw_steer, -1.0, 1.0))
        # Slower in corners, speed boost when arrow is nearly straight ahead
        straight  = float(max(0.0, 1.0 - abs(heading_error) / 0.26))  # 1.0 at 0°, 0 at 15°
        arr_thr   = float(min(1.0, ARROW_THROTTLE * (1.0 - CORNER_FACTOR * abs(arr_steer))
                              + STRAIGHT_BOOST * straight))

        # ── Obstacle blend factor ─────────────────────────────────────────
        # x[3]=front, x[4]=front-left(+45°), x[10]=front-right(-45°)
        front_clearance = min(float(x[3]), float(x[4]), float(x[10]))
        obstacle_factor = float(np.clip(
            1.0 - front_clearance / OBSTACLE_THRESH, 0.0, 1.0
        ))

        # ── Blend arrow + ML ─────────────────────────────────────────────
        thr   = (1.0 - obstacle_factor) * arr_thr   + obstacle_factor * ml_thr
        steer = (1.0 - obstacle_factor) * arr_steer + obstacle_factor * ml_steer

        # ── Recovery state machine ────────────────────────────────────────
        if ctx["recovery"] > 0:
            # Phase 1: forced reverse.
            # STEER_SIGN is applied so the nose swings toward the target
            # consistently with the forward driving convention.
            thr   = -1.0
            steer = STEER_SIGN * (1.0 if heading_error >= 0 else -1.0)
            ctx["recovery"] -= 1
            if ctx["recovery"] == 0:
                # Recovery just ended — flush the EMA so the extreme reverse
                # values don't bleed into the next phase.
                ctx["ema_t"] = 0.0
                ctx["ema_s"] = 0.0
                ctx["reorient"] = REORIENT_FRAMES

        elif ctx["reorient"] > 0:
            # Phase 2: snap-steer toward checkpoint with higher gain.
            # The car has just reversed and may be pointing in any direction.
            snap_steer = STEER_SIGN * heading_error * REORIENT_KP
            steer = float(np.clip(snap_steer, -1.0, 1.0))
            thr   = ARROW_THROTTLE * 0.6   # go slower while reorienting
            ctx["reorient"] -= 1

        elif frame > GRACE_FRAMES:
            if speed < STUCK_SPEED:
                ctx["stuck"] += 1
            else:
                ctx["stuck"] = 0

            if ctx["stuck"] >= WIGGLE_THRESH:
                # Physics trap: car has been immobile for 7+ seconds even after
                # normal recovery attempts.  Rapidly alternate throttle direction
                # to create micro-oscillations that can break physics locks.
                wiggle_sign = 1.0 if (ctx["stuck"] // WIGGLE_PERIOD) % 2 == 0 else -1.0
                thr   = wiggle_sign
                steer = STEER_SIGN * (1.0 if heading_error >= 0 else -1.0)
            elif ctx["stuck"] >= STUCK_THRESH:
                ctx["recovery"] = RECOVERY_FRAMES
                ctx["stuck"]    = 0

        # ── EMA smoothing ────────────────────────────────────────────────
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

    policy = make_policy(args.weights)
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
    """Join the professor's competition room and run 5 rounds automatically.

    The professor controls when each round starts/ends.  This script listens
    for round_start / round_end events and runs the policy in between.
    Policy state (frame counter, stuck streak, EMA) resets each round.
    """
    print("=" * 60)
    print(f"  Drive2Win — COMPETITION mode")
    print(f"  student  : {args.student_id}")
    print(f"  agent    : {args.name}")
    print(f"  weights  : {args.weights}")
    print("=" * 60)

    client = GameClient(args.server, args.api_key)

    room = client.join_competition(
        student_id=args.student_id,
        agent_name=args.name,
    )
    print(f"Joined room: {room.get('room_id', '?')}  "
          f"({room.get('players', '?')} players)")

    # Threading events to coordinate between WS callback thread and main loop
    round_started = threading.Event()
    round_ended   = threading.Event()
    done          = threading.Event()

    round_scores = []

    def on_round_start(data):
        round_no = data.get("round", "?")
        print(f"\n>>> ROUND {round_no} STARTED — running policy")
        round_ended.clear()
        round_started.set()

    def on_round_end(data):
        round_no = data.get("round", "?")
        cp = data.get("checkpoints", "?")
        print(f"<<< ROUND {round_no} ENDED  (server reports: {cp} checkpoints)")
        round_started.clear()
        round_ended.set()

    def on_competition_end(data):
        print("\n=== COMPETITION ENDED ===")
        round_ended.set()
        done.set()

    client.on_event("round_start",      on_round_start)
    client.on_event("round_end",        on_round_end)
    client.on_event("competition_end",  on_competition_end)

    client.connect_competition_ws(args.student_id, args.name)
    print("Connected. Waiting for professor to start the first round...")

    MAX_ROUNDS    = 5
    ROUND_TIMEOUT = 360.0   # safety cap per round (6 min)

    for rnd in range(1, MAX_ROUNDS + 1):
        if done.is_set():
            break

        # Wait for the professor to start this round (up to 30 min)
        if not round_started.wait(timeout=1800):
            print("Timed out waiting for round start. Exiting.")
            break

        # Fresh policy each round so state doesn't carry over
        policy = make_policy(args.weights)

        interval = 1.0 / 20.0
        start    = time.time()
        steps    = 0
        max_cp   = 0

        while not round_ended.wait(timeout=0) and time.time() - start < ROUND_TIMEOUT:
            state = client.get_latest_state()
            if state and state.get("sensors"):
                nav = (state.get("sensors") or {}).get("navigation") or {}
                cp  = nav.get("checkpoints_completed", 0) or 0
                max_cp = max(max_cp, cp)
                thr, steer = policy(state)
                client.send_control_ws(thr, steer)
                steps += 1
            time.sleep(interval)

        elapsed = time.time() - start
        round_scores.append(max_cp)
        print(f"  Round {rnd}: {max_cp} checkpoints  "
              f"({steps} steps, {elapsed:.1f} s)")

        # Wait briefly between rounds
        round_ended.wait(timeout=5)

    print("\n" + "=" * 60)
    print(f"  Final scores by round: {round_scores}")
    print(f"  Total checkpoints    : {sum(round_scores)}")
    print("=" * 60)

    client.disconnect_ws()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Drive2Win tournament entry")
    ap.add_argument("--weights",      default="nav_v4.npz")
    ap.add_argument("--name",         default="albertkola",
                    help="Display name shown on leaderboard")
    ap.add_argument("--server",       default="https://ml.ferit.tech")
    ap.add_argument("--api-key",      default="None")

    # Practice mode flags
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--duration",     type=float, default=300.0)

    # Competition mode flag
    ap.add_argument("--competition",  action="store_true",
                    help="Join professor's competition room (tournament day)")
    ap.add_argument("--student-id",   default="",
                    help="Your student ID (required for --competition)")

    args = ap.parse_args()

    if args.competition:
        if not args.student_id:
            print("ERROR: --student-id is required in competition mode.")
            print("  python 99_compete.py --competition --student-id YOUR_ID")
            return
        competition_main(args)
    else:
        practice_main(args)


if __name__ == "__main__":
    main()
