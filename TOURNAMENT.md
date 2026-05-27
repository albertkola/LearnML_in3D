# Tournament Guide

You've trained a controller in the curriculum (`LearnML_in3D/01_collect.py` … `04_compare.py`). Now you'll plug it into a `RoomBot`, race it against your classmates' bots, and (hopefully) win.

This guide walks you through every step: connecting, testing solo, plugging in a trained model, and the actual presentation-day flow.

---

## 1. Prerequisites

- **Python 3.10+**
- Install the SDK deps:
  ```bash
  pip install requests websocket-client numpy
  ```

The tournament server lives at **`https://ml.ferit.tech`**. All examples below assume that host. **No API key needed** for the tournament — anyone with the room name can join.

---

## 2. 30-second smoke test

Confirm your machine can reach the server and connect to a room. The repo ships a `test_bot.py` that drives a random walk — perfect for sanity-checking wiring.

```bash
python test_bot.py --host ml.ferit.tech --secure --room <your-name>
```

You should see lines like:

```
Connecting to wss://ml.ferit.tech/ws/room/bot?room=...
[RandomBot-123] connected — sending ready
[RandomBot-123] bot_key=bot:RandomBot-123
```

If the script hangs on `Connecting`, your firewall is probably blocking the WebSocket — try a different network.

---

## 3. Hello bot in 10 lines

Open a browser tab at `https://ml.ferit.tech/?room=demo` first (you'll be auto-promoted to admin — the race won't start until your bot is connected and ready). Then save this as `hello_bot.py`:

```python
from LearnML_in3D.game_client import RoomBot

def controller(obs):
    # Steer toward the next checkpoint; floor it.
    return 0.7, obs["navigation"]["heading_error"] * 0.5

bot = RoomBot("https://ml.ferit.tech", room="demo", name="Alice")
standings = bot.run(controller, hz=20.0)
print(standings)
```

Run it:

```bash
python hello_bot.py
```

Watch your bot drive in the browser tab. After 5 rounds (~25 minutes), the standings will print in the terminal.

---

## 4. Solo testing flow

You don't need anyone else to develop your bot. The race starts as soon as every connected bot is ready, and a one-bot room is a valid race.

1. **Open the browser tab** at `https://ml.ferit.tech/?room=<your-name>`. You get auto-promoted to admin.
2. **In a terminal**, run your bot:
   ```bash
   python my_bot.py --room <your-name> --name <your-name>
   ```
3. **Watch** your bot drive in the browser. The scoreboard, bot overlay, and free-orbit camera all work as documented.
4. **Tear down**: close the browser tab + Ctrl+C the script. The room disappears when both leave.

Tip: pin the terrain by passing the same room name every time. The server uses a per-room base seed, so your test conditions stay stable across runs.

---

## 5. Plug in your trained model

If you've been through the curriculum, you already have a behavioral-cloning model that takes a 12-feature state vector and returns `(throttle, steering)`. Plugging it into a tournament bot is just a function:

```python
import numpy as np
from LearnML_in3D.game_client import RoomBot
# from your training script:
# model = load_trained_model("my_bc_model.pt")

def controller(obs):
    nav = obs["navigation"]
    features = np.concatenate([
        [obs["speed"], nav["heading_error"], nav["distance"]],
        obs["rays"],            # 8 floats
        [obs["ground_friction"]],
    ])  # → shape (12,) — same layout as RecordingSample.state
    throttle, steering = model.predict_one(features)
    return throttle, steering

bot = RoomBot("https://ml.ferit.tech", room="demo", name="Alice")
bot.run(controller, hz=20.0)
```

The feature order **matches** `GameClient.get_recording_as_arrays()` — same 12 floats, same units, same normalization. If your model trained on that, it works here unchanged.

For CNN models, `obs["grid32"]` is the same `(4, 32, 32)` channels-first array `get_grid_local()` returns.

---

## 6. Presentation day

Step-by-step for the actual tournament:

1. **Instructor announces** the tournament room name and start time. Example: `room=final2026`, 14:00.
2. **(Optional) Open a spectator tab**: `https://ml.ferit.tech/?room=final2026`. You'll see the whole field in one view.
3. **Start your bot** a few minutes before the announced time:
   ```bash
   python my_bot.py --room final2026 --name <your-name>
   ```
   Your bot auto-readies on connect.
4. **Race starts** when **every** connected bot is ready (or instructor force-starts). 5 rounds, ~5 minutes each. Rounds 4 and 5 add obstacles.
5. **Final standings** print in your terminal and on the projector. Your `bot.run(...)` returns the standings list — keep that around if you want to log it.

---

## 7. Observation reference

The `obs` dict passed to your controller every tick:

| Key | Type | Units | Notes |
|---|---|---|---|
| `position` | `dict {x, y, z}` | world units (m) | bot's world position; `y` is height |
| `heading` | `float` | radians, `[-π, π]` | yaw; 0 = looking down −Z |
| `speed` | `float` | m/s | finite-difference estimate, EMA-smoothed |
| `rays` | `list[8] float` | m, `[0, 50]` | obstacle distances, 0°,45°,…,315° relative to heading |
| `ground_friction` | `float` | unitless, ~`[0.4, 1.2]` | terrain-id → friction lookup (ice=0.4, rock=1.2) |
| `grid32` | `np.ndarray (4, 32, 32)` | normalized `[0, 1]` | terrain / elevation / obstacles / nav-gradient |
| `navigation` | `dict` | — | `distance` (m), `heading_error` (rad, `[-π, π]`), `checkpoint_index` (int) |
| `checkpoints_passed` | `int` | count | cumulative checkpoints across all laps |
| `round_index` | `int` | `[0, 4]` | current round |
| `race_phase` | `str` | — | `lobby` / `countdown` / `racing` / `round_end` / `finished` |
| `other_bots` | `list[dict]` | — | other bots' `RoomBotState` (position, rotation, checkpoints) |

`controller(obs)` returns `(throttle, steering)`, both clipped to `[-1, 1]`. Throttle ≥ 0 drives forward, < 0 reverses; positive steering turns right.

---

## 8. Troubleshooting

| Symptom | What's wrong | Fix |
|---|---|---|
| `room_not_ready` (404 from `/api/room/.../world_map`) | No admin browser tab has opened the room yet | Open `https://ml.ferit.tech/?room=<room>` in a tab |
| Bot appears in browser but doesn't drive | `race_phase` may still be `lobby` or `countdown` | Wait for `round_start` to print in terminal |
| Controller too slow (warning: `avg dt … exceeds budget`) | Your `controller(obs)` runs > 100 ms | Move heavy work out of the loop; cache features; lighter model |
| Bot drives erratically | `controller` raised an exception | The SDK falls back to `(0, 0)` and logs the error — check stderr |
| Standings list is empty when `run()` returns | Tournament ended before you connected, or network drop | Reconnect and wait for the next tournament |

For anything else, ping your instructor with the full terminal output.

Good luck. Don't crash. 🏁
