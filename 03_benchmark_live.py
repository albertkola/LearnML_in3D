"""Live benchmark — runs all laps on ONE session (browser stays open).

The game server only simulates when a browser tab is connected.
This script prints the URL ONCE, waits for you to open it, then runs
all 5 laps automatically by reconfiguring the seed between laps.

Usage:
    python 03_benchmark_live.py --tag v3-torch --weights nav_v3-torch.pt \
        --module drive2win.torch_mlp --seeds 42 7 99

For each seed you will be prompted once to confirm the browser is open.
All 5 runs for that seed then happen automatically — no more interaction needed.
Output format is identical to 03_benchmark.py so 04_compare.py works unchanged.
"""
from __future__ import annotations
import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np

from game_client import GameClient
from drive2win import nn as nn_mod, viz
from drive2win.eval import run_policy, score_runs
from drive2win.normalize import sensors_to_input, clip_action

SERVER_URL = "https://ml.ferit.tech"
API_KEY    = "None"
TARGET_CPS = 8   # one full lap


def _load_policy(weights: str, module: str | None):
    if module:
        mod = importlib.import_module(module)
        return mod.make_policy(weights)
    w = nn_mod.load(weights)
    def policy(state):
        x = sensors_to_input(state["sensors"])
        return clip_action(nn_mod.forward(x, w))
    return policy


def _wait_for_state(client: GameClient, timeout: float = 30.0) -> bool:
    """Block until the WS delivers at least one state, or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if client.get_latest_state() is not None:
            return True
        time.sleep(0.2)
    return False


def benchmark_seed(client: GameClient, policy_fn, seed: int,
                   runs: int, duration: float, tag: str) -> dict:
    """Run `runs` laps on the same session, reconfiguring seed between laps."""

    # First lap: reconfigure to this seed and wait for simulation to start
    client.configure(terrain_seed=seed, wind_enabled=False)
    time.sleep(1.5)  # let the server re-roll terrain

    runs_out = []
    for i in range(runs):
        if i > 0:
            # Reset by reconfiguring the same seed (respawns the car)
            client.configure(terrain_seed=seed)
            time.sleep(1.5)

        # Drain any stale state
        time.sleep(0.3)

        print(f"  run {i+1}/{runs}  seed={seed}  ", end="", flush=True)
        result = run_policy(client, policy_fn, duration=duration, hz=20.0)
        print(f"checkpoints={result['checkpoints_passed']}/{TARGET_CPS}  "
              f"crashes={result['crashes']}  steps={result['steps']}")
        runs_out.append(result)

    return runs_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag",      required=True)
    ap.add_argument("--weights",  required=True)
    ap.add_argument("--module",   default=None)
    ap.add_argument("--seeds",    type=int, nargs="+", default=[42])
    ap.add_argument("--runs",     type=int,   default=5)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--data",     default=None,
                    help="Optional .npz for path-overlay PNG")
    args = ap.parse_args()

    out_dir = Path("benchmarks"); out_dir.mkdir(exist_ok=True)
    policy  = _load_policy(args.weights, args.module)

    client  = GameClient(SERVER_URL, API_KEY)

    # Create ONE session and keep it alive for all seeds
    first_seed = args.seeds[0]
    session = client.create_session(
        mode="time_trial",
        player_name=f"benchmark_{args.tag}",
        config={"seed": first_seed, "wind_enabled": False},
    )
    print(f"\nOpen this URL in your browser (ONE tab, keep it open):")
    print(f"  {session.get('browser_url')}\n")
    input("Press Enter once the game is visible in the browser... ")

    client.connect_ws()
    print("Waiting for first state broadcast...", end="", flush=True)
    if not _wait_for_state(client, timeout=15.0):
        print("\nERROR: no state received after 15s — is the browser tab open?")
        client.disconnect_ws(); client.delete_session()
        return
    print(" OK\n")

    all_results = []
    try:
        for seed in args.seeds:
            print(f"=== seed {seed} ===")
            runs_out = benchmark_seed(client, policy, seed,
                                      runs=args.runs, duration=args.duration,
                                      tag=args.tag)
            summary = score_runs(runs_out, TARGET_CPS)
            s = summary
            print(f"  SEED {seed}: complete={int(s['completion_rate']*s['n_runs'])}"
                  f"/{s['n_runs']}  crashes={s['mean_crashes']:.1f}"
                  f"  max_cp={s['max_checkpoints']}\n")
            all_results.append({"seed": seed, "summary": summary, "runs": runs_out})
    finally:
        try: client.disconnect_ws()
        except Exception: pass
        try: client.delete_session()
        except Exception: pass

    # ── Print headline ──────────────────────────────────────────────────
    print("=" * 56)
    print(f"  iteration: {args.tag}    weights: {args.weights}")
    for r in all_results:
        s = r["summary"]
        ml = "inf" if s["median_lap_time"] == float("inf") else f"{s['median_lap_time']:.1f}s"
        print(f"  seed {r['seed']:>4}  "
              f"complete={int(s['completion_rate']*s['n_runs'])}/{s['n_runs']}  "
              f"median_lap={ml}  crashes={s['mean_crashes']:.1f}  "
              f"max_cp={s['max_checkpoints']}")
    print("=" * 56)

    # ── Write JSON (same schema as 03_benchmark.py) ──────────────────────
    log = {
        "tag": args.tag, "weights": args.weights, "module": args.module,
        "runs_per_seed": args.runs, "duration_s": args.duration,
        "seeds": all_results,
    }
    log_path = out_dir / f"{args.tag}.json"
    log_path.write_text(json.dumps(log, indent=2, default=float))
    print(f"\nwrote {log_path}")

    # ── Visuals ──────────────────────────────────────────────────────────
    flat_runs = [run for r in all_results for run in r["runs"]]
    viz.plot_multi_run_paths(flat_runs,
                             out=str(out_dir / f"{args.tag}_paths.png"),
                             title=f"All paths — {args.tag}")
    viz.plot_checkpoint_progress(flat_runs,
                                 out=str(out_dir / f"{args.tag}_progress.png"))
    if args.data:
        d = np.load(args.data, allow_pickle=False)
        train_xz = d.get("positions") if "positions" in d.files else None
        first_track = flat_runs[0].get("track") or []
        viz.plot_path_overlay(train_xz, first_track,
                              out=str(out_dir / f"{args.tag}_overlay.png"),
                              title=f"{args.tag} — drive (gray) vs NN (blue)")


if __name__ == "__main__":
    main()
