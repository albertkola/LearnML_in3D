"""Tournament day competition entry point.

Single model (recommended for speed):
    python 99_compete.py --weights nav_v3-torch.pt --module drive2win.torch_mlp

Ensemble of two models:
    python 99_compete.py --weights nav_v3-torch.pt nav_v2-simple.npz

Tune the stuck-recovery if the bot keeps getting wedged:
    python 99_compete.py --weights nav_v3-torch.pt --stuck 30 --alpha 0.8

Run a practice round first to verify it works end-to-end:
    python 99_compete.py --weights nav_v3-torch.pt --practice
"""
from __future__ import annotations
import argparse
import importlib
import time

from game_client import GameClient
from drive2win.eval import run_policy
from drive2win.smoothed_policy import make_smoothed_policy

SERVER_URL      = "https://ml.ferit.tech"
API_KEY         = "None"
ROUND_DURATION  = 300.0   # 5 minutes per tournament round


def _load_policy(weights_list: list[str], module: str | None):
    """Load policy from one or more weight files."""
    if len(weights_list) == 1 and module:
        mod = importlib.import_module(module)
        return mod.make_policy(weights_list[0])

    if len(weights_list) > 1:
        from drive2win.ensemble import make_ensemble_policy
        print(f"Ensemble mode: {weights_list}")
        return make_ensemble_policy(weights_list)

    # Single file, no explicit module — auto-detect by extension
    path = weights_list[0]
    if path.endswith(".pt"):
        from drive2win.torch_mlp import make_policy
    else:
        from drive2win.ensemble import make_policy
    return make_policy(path)


def _run_round(client, policy, duration: float, seed: int, player: str,
               alpha: float, stuck: int):
    """Create a session, wrap the policy, run one round, return result."""
    session = client.create_session(
        mode="time_trial",
        player_name=player,
        config={"seed": seed, "wind_enabled": False},
    )
    print(f"\n  Browser URL: {session.get('browser_url')}")
    input("  Press Enter once the browser tab is open and you can see the car... ")

    client.connect_ws()
    time.sleep(1.0)

    wrapped = make_smoothed_policy(policy, alpha=alpha, stuck_threshold=stuck)
    print(f"  Running {duration:.0f}s — seed {seed}...")
    result = run_policy(client, wrapped, duration=duration)

    print(f"  Checkpoints : {result['checkpoints_passed']}")
    print(f"  Crashes     : {result['crashes']}")
    print(f"  Stuck streak: {result['min_speed_streak']} frames")

    try:
        client.disconnect_ws()
        client.delete_session()
    except Exception:
        pass

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True,
                    help="Weight file(s). Multiple = ensemble.")
    ap.add_argument("--module", default=None,
                    help="Module with make_policy (single-model only).")
    ap.add_argument("--alpha", type=float, default=0.7,
                    help="EMA smoothing (0.6–0.8).")
    ap.add_argument("--stuck", type=int, default=40,
                    help="Frames before stuck-recovery fires.")
    ap.add_argument("--player", default="albertkola",
                    help="Player name shown in the tournament.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Terrain seed for a single practice round.")
    ap.add_argument("--practice", action="store_true",
                    help="Run one practice round instead of tournament mode.")
    ap.add_argument("--duration", type=float, default=ROUND_DURATION,
                    help="Seconds per round (default 300).")
    args = ap.parse_args()

    print("=== Drive2Win Competition Agent ===")
    print(f"Weights : {args.weights}")
    print(f"EMA α   : {args.alpha}  |  stuck threshold: {args.stuck} frames")

    policy = _load_policy(args.weights, args.module)
    client = GameClient(SERVER_URL, API_KEY)

    if args.practice:
        print("\n[PRACTICE MODE — single round]")
        result = _run_round(client, policy, args.duration, args.seed,
                            args.player, args.alpha, args.stuck)
        print(f"\nPractice done: {result['checkpoints_passed']} checkpoints.")
        return

    # ── Tournament mode: prompt for each round's seed ───────────────────
    print("\n[TOURNAMENT MODE]")
    print("After each round ends, enter the next seed when the tournament")
    print("director announces it (or press Ctrl-C to stop early).\n")

    total_checkpoints = 0
    round_num = 0
    try:
        while True:
            round_num += 1
            seed_str = input(f"Round {round_num} seed (or 'q' to quit): ").strip()
            if seed_str.lower() == "q":
                break
            try:
                seed = int(seed_str)
            except ValueError:
                print("  Invalid seed, skipping.")
                continue

            result = _run_round(client, policy, args.duration, seed,
                                args.player, args.alpha, args.stuck)
            total_checkpoints += result["checkpoints_passed"]
            print(f"  Running total: {total_checkpoints} checkpoints over "
                  f"{round_num} round(s).")
    except KeyboardInterrupt:
        pass

    print(f"\n=== FINAL: {total_checkpoints} checkpoints over {round_num} rounds ===")


if __name__ == "__main__":
    main()
