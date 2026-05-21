# Architectures — using PyTorch, CNNs, and hybrid models

The baseline (`drive2win/nn.py` + `02_train.py`) is a 12→64→32→2 NumPy MLP. Once that works, the README invites you to try richer architectures — deeper MLPs, CNNs over the 32×32 terrain grid, hybrid models, ensembles.

This document is the practical how-to for those iterations. It tells you exactly what works today, what needs a server-side change, and the code patterns you copy to plug into `03_benchmark.py` without ever editing `drive2win/benchmark.py`.

> **Rule of thumb:** the harder you change the architecture, the more important it is that you re-run `03_benchmark.py` on **three seeds** (not just `42`). A bigger model is the easiest way to memorize one map.

---

## 1. What's wired today

| Feature | Status | Where |
|---|---|---|
| `--module` plug-in (any custom policy) | ✅ works | `drive2win/benchmark.py:51` |
| Recording the 12-feature vector | ✅ works | `client.start_recording(sample_rate=20)` |
| Recording the 32×32×4 terrain grid | ✅ works | `client.start_recording(..., include_grid=True)` then `client.get_recording_with_grid()` |
| Per-frame grid via REST at inference | ✅ works (HTTP, adds latency) | `client.get_grid_observation()` |
| Per-frame grid in the WS state broadcast | ❌ **not wired** | `src/main.ts:127-144` — broadcast omits `grid32` |
| Saving NumPy weights (`.npz`) | ✅ works | `drive2win/nn.py:save/load` |
| Loading PyTorch weights (`.pt`) | ✅ works | you write the loader inside your `make_policy` |

The one consequential gap is the last one: at inference time, `run_policy` calls `policy_fn(state)` with the WS state dict, and that dict doesn't contain `grid32`. A pure-CNN policy needs the grid every frame. The three workarounds are spelled out in section 5.

---

## 2. The universal hook — `make_policy(weights_path)`

Every iteration that uses something other than the default NumPy MLP plugs in the same way: write a Python module that exposes one function.

```python
# drive2win/<your_module>.py

def make_policy(weights_path: str):
    """Load weights, return a callable that maps a WS state dict to (throttle, steering)."""
    # 1. load your weights (any format you want — .pt, .npz, .pkl)
    # 2. construct your model
    # 3. return a function: (state_dict) -> (float, float) in [-1, 1]
    ...
```

Benchmark it with:

```
python 03_benchmark.py --tag <tag> --weights <path> --module drive2win.<your_module>
```

The benchmark script never needs to know what's inside your module. It just calls `make_policy(weights_path)` and uses the returned callable. **You do not edit `drive2win/benchmark.py`.**

`state` is the dict from `client.get_latest_state()`. The fields you can rely on (from `src/main.ts:127-144`):

```python
state = {
    "position": {"x": ..., "y": ..., "z": ...},
    "sensors": {
        "speed": float,
        "heading_error": float,
        "checkpoint_distance": float,
        "rays": [r0, r1, ..., r7],
        "ground_friction": float,
        "navigation": {
            "checkpoints_completed": int,
            "respawn_count": int,
        },
    },
}
```

`state["sensors"]` is exactly the input to `drive2win.normalize.sensors_to_input(...)`, which gives you the same normalized 12-vector you trained on. Always use that helper — never rebuild normalization by hand.

---

## 3. Path 1 — A deeper / different MLP in PyTorch

This is the cheapest jump in expressiveness, and it works fully today. You stay on the same 12-feature input, change the model, and plug back in via `make_policy`.

### Train it

```python
# 02_train_torch.py
import argparse, numpy as np, torch, torch.nn as nn
from drive2win.normalize import normalize_states

class DeeperMLP(nn.Module):
    def __init__(self, n_in=12, h=(128, 64, 32), n_out=2):
        super().__init__()
        sizes = [n_in, *h]
        layers = []
        for a, b in zip(sizes, sizes[1:]):
            layers += [nn.Linear(a, b), nn.LeakyReLU(0.1)]
        layers += [nn.Linear(sizes[-1], n_out), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    X = torch.tensor(normalize_states(d["states"]), dtype=torch.float32)
    Y = torch.tensor(d["actions"], dtype=torch.float32)
    perm = torch.randperm(len(X))
    n_val = max(1, len(X) // 10)
    Xtr, Ytr = X[perm[n_val:]], Y[perm[n_val:]]
    Xva, Yva = X[perm[:n_val]], Y[perm[:n_val]]

    model = DeeperMLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for epoch in range(300):
        model.train()
        idx = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            b = idx[i:i+64]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), Ytr[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xva), Yva).item()
        if epoch % 25 == 0:
            print(f"epoch {epoch:3d}  val={v:.4f}")

    torch.save(model.state_dict(), f"nav_{args.tag}.pt")
    print(f"saved nav_{args.tag}.pt")

if __name__ == "__main__":
    main()
```

### Plug into the benchmark

```python
# drive2win/torch_mlp.py
import torch
from .normalize import sensors_to_input, clip_action
# Import the same DeeperMLP class. In real code put it in this file or import.
from your_train_module import DeeperMLP  # or copy the class here

def make_policy(weights_path: str):
    model = DeeperMLP()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    def policy(state):
        x = sensors_to_input(state["sensors"])              # (12,) float32
        with torch.no_grad():
            y = model(torch.from_numpy(x).unsqueeze(0))[0].numpy()
        return clip_action(y)

    return policy
```

### Run it

```
python 02_train_torch.py --data data_v3.npz --tag v3-deepnet
python 03_benchmark.py   --tag v3-deepnet --weights nav_v3-deepnet.pt --module drive2win.torch_mlp
```

That's it. No platform change, no recording change. Everything in section 2 is hot-pluggable.

---

## 4. Path 2 — Temporal MLP (stack a short history of the 12-vector)

Pure-CNN inference has a gap (section 5). If you want richer-than-MLP behavior **today, with no platform change**, stack the last K frames of the 12-vector into a 12·K input. This often beats a one-shot MLP on stuck-recovery — the model sees that it's been still for half a second.

### Training data — built from your existing recording

```python
def make_windowed(states_raw, actions, K=4):
    # repeat-pad at the start so the first samples have a valid history
    pad = np.repeat(states_raw[:1], K - 1, axis=0)
    s = np.concatenate([pad, states_raw], axis=0)
    windows = np.stack([s[i:i+K].reshape(-1) for i in range(len(states_raw))])
    return windows.astype(np.float32), actions
```

Train any architecture on `(N, 12*K) -> (N, 2)`. At inference, the policy callable holds a rolling buffer of the last K-1 sensor vectors and appends the current one:

```python
# drive2win/temporal.py
from collections import deque
import torch
from .normalize import sensors_to_input, clip_action

K = 4

def make_policy(weights_path):
    model = MyTemporalNet()                          # 48 -> ... -> 2
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    buf = deque(maxlen=K)

    def policy(state):
        x = sensors_to_input(state["sensors"])
        while len(buf) < K - 1:
            buf.append(x.copy())                     # warm start
        buf.append(x)
        window = np.concatenate(list(buf)).astype(np.float32)
        with torch.no_grad():
            y = model(torch.from_numpy(window).unsqueeze(0))[0].numpy()
        return clip_action(y)

    return policy
```

A 1D-Conv over the K dimension is also fine here. Either way, no server change is required.

---

## 5. Path 3 — CNN over the 32×32 terrain grid

The grid (`grid32`) is the only input that gives the model spatial context — checkpoints in front of obstacles, narrow passes, friction patches. It's also the path with the only real platform friction.

### What the grid is

A `32 × 32 × 4` tensor (`get_recording_with_grid()` returns it as `(N, 32, 32, 4)` — heading-aligned):

- channel 0 — terrain type
- channel 1 — elevation
- channel 2 — obstacles
- channel 3 — navigation gradient (direction-to-next-checkpoint hint)

For PyTorch, transpose to `(N, 4, 32, 32)` (channels-first). The per-frame helper `client.get_grid_observation()` already returns `(4, 32, 32)`.

### Collecting training data (works today)

Edit `01_collect.py` (or write `01_collect_grid.py`) to start recording with grids:

```python
client.start_recording(sample_rate=20, include_grid=True)
# ... your existing 5-phase driving ...
states_raw, actions, grids = client.get_recording_with_grid()
# states_raw: (N, 12)  actions: (N, 2)  grids: (N, 32, 32, 4)
np.savez(f"data_{args.tag}.npz", states=states_raw, actions=actions, grids=grids)
```

Bandwidth is ~5 MB/min at 20 Hz. Fine for a six-minute capture.

### Training a CNN

```python
# 02_train_cnn.py
import argparse, numpy as np, torch, torch.nn as nn
from drive2win.normalize import normalize_states

class TerrainCNN(nn.Module):
    def __init__(self, n_out=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                              # 16x16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                              # 8x8
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                      # 1x1
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, n_out), nn.Tanh(),
        )

    def forward(self, g):                                 # g: (B, 4, 32, 32)
        return self.head(self.conv(g))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    G = torch.tensor(d["grids"], dtype=torch.float32).permute(0, 3, 1, 2)  # NCHW
    Y = torch.tensor(d["actions"], dtype=torch.float32)
    # ... standard 90/10 split + Adam loop, identical to path 1 ...
    torch.save(model.state_dict(), f"nav_{args.tag}.pt")

if __name__ == "__main__":
    main()
```

### Inference — the gap, and three workarounds

The WS broadcast (`src/main.ts:127-144`) does **not** include `grid32`. `policy_fn(state)` sees only `state["sensors"]` (the 12-vector). You have three options:

#### Option A (cleanest, needs the instructor) — add `grid32` to the WS state

Single-line change in `src/main.ts`. The broadcast becomes:

```ts
sensors: {
  speed: agent.getSpeed(),
  // ...existing fields...
  grid32: sensorSystem.getGrid32x32().data,   // <— add this
}
```

Then in your policy:

```python
def policy(state):
    g = np.array(state["sensors"]["grid32"], dtype=np.float32)  # (32, 32, 4)
    g = torch.from_numpy(g).permute(2, 0, 1).unsqueeze(0)        # (1, 4, 32, 32)
    with torch.no_grad():
        y = model(g)[0].numpy()
    return clip_action(y)
```

If you want to try a CNN, **ask the instructor** to make this change before you start iterating. It's listed as an open item under "Recording — needed for the path overlay and for any CNN iteration" in `INSTRUCTOR_TODO.md` (the recording-time `include_grid` is done; the live-broadcast piece is the missing companion). Bandwidth at 20 Hz over WS is ~80 KB/s per client — fine.

#### Option B (works today, slower) — pull the grid via REST inside your policy

Your `make_policy` opens its own GameClient pointed at the same server, and every step calls `/sensors/grid32`. The benchmark already creates a client with `api_key` set — you can pass the same key and read the active session id from the broadcast.

The catch: there is no clean way for `make_policy(weights_path)` to learn the session id, because the session is created **after** the policy is built. Workaround:

```python
# drive2win/cnn_rest.py
import os, numpy as np, torch
from game_client import GameClient
from .normalize import clip_action

def make_policy(weights_path):
    model = TerrainCNN()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    side_client = GameClient(
        os.environ.get("DRIVE2WIN_SERVER", "https://ml.ferit.tech"),
        os.environ.get("DRIVE2WIN_API_KEY", "None"),
    )

    def policy(state):
        # state has no session id; we need one. Stash it from the env or
        # discover it from the active client by listing your sessions.
        # Practical hack: write the active session id to a temp file from
        # a wrapper script; read it here.
        if not side_client.session_id:
            side_client.session_id = os.environ["DRIVE2WIN_ACTIVE_SESSION"]
        g = side_client.get_grid_observation()                  # (4, 32, 32)
        with torch.no_grad():
            y = model(torch.from_numpy(g).unsqueeze(0))[0].numpy()
        return clip_action(y)

    return policy
```

This adds a synchronous HTTP round-trip per step, so your effective control rate drops below 20 Hz. It works for an exploratory iteration but is not how you'd run a tournament. Use option A.

#### Option C (always available) — skip the grid; do path 2 instead

Path 2 (temporal MLP) gives you more model capacity than the baseline MLP without needing the grid at inference. If the instructor change for option A is not in place, this is what you ship.

---

## 6. Path 4 — Hybrid CNN + MLP

The single architecture that most reliably beats the baseline on **obstacle rounds** combines:

- the 12-vector (close-range raycasts, heading, speed) → small dense branch
- the 32×32×4 grid (medium-range terrain layout) → CNN branch
- concatenated features → final FC → tanh

```python
class Hybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(  # → 32-d
            nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.mlp = nn.Sequential(  # → 32-d
            nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2), nn.Tanh(),
        )

    def forward(self, x12, grid):
        return self.head(torch.cat([self.mlp(x12), self.cnn(grid)], dim=1))
```

The inference-time gap from path 3 applies identically: the grid is not in the WS state. You need option A (instructor change) or option B (REST polling) for this to work in `03_benchmark.py`.

If option A is in place, your policy is just:

```python
def policy(state):
    x12 = torch.from_numpy(sensors_to_input(state["sensors"])).unsqueeze(0)
    g = np.array(state["sensors"]["grid32"], dtype=np.float32)
    g = torch.from_numpy(g).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        y = model(x12, g)[0].numpy()
    return clip_action(y)
```

---

## 7. Save / load conventions

| File suffix | Used for | Loaded by |
|---|---|---|
| `nav_<tag>.npz` | NumPy MLP weights (baseline) | `drive2win.nn.load` |
| `nav_<tag>.pt` | PyTorch `state_dict` | `torch.load` inside your `make_policy` |
| `data_<tag>.npz` | training data, `states` + `actions` (+ optional `grids`) | your training script |

`03_benchmark.py` accepts both `.npz` and `.pt` — the file is just a string that gets passed to your `make_policy`. Name them after your iteration tag so the git history reads cleanly.

`04_compare.py` reads `benchmarks/*.json` and doesn't care what architecture produced each entry — the JSON shape is identical.

---

## 8. Pitfalls

- **Normalization drift.** If you change `drive2win/normalize.py` (e.g. switch to z-score on rays) you must retrain. The PyTorch path is *especially* easy to mess this up — your training loop and your `policy` callable must call exactly the same `normalize_states` / `sensors_to_input`.
- **Channels-last vs channels-first.** `client.get_recording_with_grid()` returns grids as `(N, 32, 32, 4)`. `client.get_grid_observation()` returns `(4, 32, 32)`. PyTorch convolutions want `(B, C, H, W)`. Always `.permute(...)` in the same place in train and infer.
- **Forgetting `model.eval()` + `torch.no_grad()` in `policy`.** Dropout/BatchNorm in train mode will misbehave at inference. `torch.no_grad()` also halves the latency.
- **CPU is fine.** The model is small and inference is 20 Hz. Don't `.cuda()` in your `policy` unless you've measured it helps — moving tensors to GPU and back can be slower than CPU at this size.
- **Test on at least 3 seeds** (`--seed 42`, `--seed 7`, `--seed 99`). A CNN with no data augmentation will overfit map shape much harder than the baseline MLP. If `seed=42` is great and `seed=7` is dead, you have a memorized-map problem, not a model problem.
- **One change at a time.** "I added a CNN and also doubled my dataset and also lowered the LR" is unscored — you can't tell which of the three moved the needle. Land each change on its own iteration with its own `benchmarks/<tag>.json`.

---

## 9. Quick reference

| Iteration idea | Server change needed? | Training data | Policy module |
|---|---|---|---|
| Deeper / wider NumPy MLP | no | `data_<tag>.npz` (states + actions) | none — edit `drive2win/nn.py` and use the default path |
| PyTorch MLP / LeakyReLU / dropout | no | `data_<tag>.npz` | `drive2win/torch_mlp.py` |
| Temporal MLP (K-frame stack) | no | `data_<tag>.npz` | `drive2win/temporal.py` |
| CNN over `grid32` | **yes** (add `grid32` to WS broadcast) — or workaround | `data_<tag>.npz` with grids | `drive2win/cnn.py` |
| Hybrid CNN + MLP | **yes** (same as above) | `data_<tag>.npz` with grids | `drive2win/hybrid.py` |
| Ensemble of two seeds | no | reuse existing data | a `make_policy` that loads two checkpoints and averages |

If you're not sure where to start, pick the cheapest row that addresses your current worst metric: low completion → temporal MLP. High crashes on obstacle rounds → hybrid (after the server change). Variance between seeds → ensemble.
