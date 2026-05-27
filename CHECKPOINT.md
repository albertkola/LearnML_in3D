# Drive2Win — Project Checkpoint & Agent Handoff
**Last updated:** 2026-05-27 | **Read this entire file before touching any code.**

---

## 1. PROJECT OVERVIEW

**Task:** Train a policy network that outputs `[throttle, steering]` (both in `[-1, 1]`, tanh output) given a 12-feature sensor input vector at 20 Hz to autonomously drive a car around a checkpoint course.

**Server:** `https://ml.ferit.tech` — GameClient SDK connects via REST + WebSocket.

**Input vector (12 features, indexes 0–11):**
```
0: speed            1: heading_error       2: checkpoint_distance
3: ray_0 (front)    4: ray_1 (front-left)  5: ray_2 (front-right)
6: ray_3 (left)     7: ray_4 (right)       8: ray_5 (back-left)
9: ray_6 (back-right) 10: ray_7 (back)     11: ground_friction
```

**Architecture:** 4-layer MLP — `12 → 128 → 64 → 32 → 2`, ReLU hidden, tanh output.

**Training method:** Behavioral Cloning — supervised imitation of human driving via MSE loss.

**Grade:** 50% process (git commits + `benchmarks/` folder) + 50% tournament performance.

**Tournament:** 5 rounds × 5 minutes, terrain seed changes per round (3 clean + 2 obstacle rounds), scored on total checkpoints collected across all rounds. TARGET_CHECKPOINTS = 8 (one full lap).

---

## 2. CRITICAL GAME SERVER BEHAVIOUR

> **The game simulation ONLY runs when a browser tab is open on the session URL.**

The WebSocket will connect successfully without a browser, but `get_latest_state()` returns `None` forever — steps=0, checkpoints=0. This means:

- `03_benchmark.py` (the canonical evaluator) does **not** work headlessly. It requires a browser tab open and navigated to each session's URL.
- When benchmarking, the user must keep a browser tab open. The session URL is printed at the start of each run.
- `01_collect.py` already handles this correctly with its "Press Enter once browser is open" prompt.

**Implication for any benchmark script you write:** Either prompt the user to open the browser URL before starting the policy loop, or wait for the first non-None state with a timeout.

---

## 3. CURRENT DATA FILES

All 4 files are in the repo root. **Do not delete or combine these without preserving them individually.**

| File | Samples | Seed | Contents |
|---|---|---|---|
| `data_v1.npz` | 6,609 | 42 | First collection. 5 phases: smooth laps, tight turns, obstacle clusters, bad terrain, recovery. Mostly clean driving — minimal recovery frames. |
| `data_v2_recovery42.npz` | 11,967 | 42 | Second collection same seed. Heavy emphasis on Phase 5 (deliberate wall crashes + reversal). ~8% of frames have throttle < −0.3 (reversing). |
| `data_v_seed7.npz` | 13,193 | 7 | Full 5-phase session on seed 7. Different terrain layout — critical for tournament generalization. |
| `data_v_seed99.npz` | 13,042 | 99 | Full 5-phase session on seed 99. Different terrain layout. |

**Arrays in each file:** `states` (N, 12) float32 raw sensor readings, `actions` (N, 2) float32 [throttle, steering], `positions` (M, 2) float32 [x, z] polled at 5 Hz.

**Normalization:** Always use `drive2win.normalize.normalize_states()` before training and `drive2win.normalize.sensors_to_input()` at inference. Never rebuild normalization by hand — it is the single source of truth.

---

## 4. ITERATION HISTORY & CHANGELOG

### v1-baseline (May 25, 2026)
- **Model:** NumPy MLP 12→128→64→32→2 trained on `data_v1.npz` (6,609 samples, seed 42 only)
- **Backward pass:** `my_backward()` in `02_train.py` verified correct (max relative error ~8.5e-6 vs numerical gradient)
- **Result:** val_loss=0.137, train_loss=0.088
- **Benchmark:** seed 42 → 0/5 completion, max_cp=3, crashes=0.2
- **Failure mode:** Gets stuck at the same corner (checkpoint 3). min_speed_streak up to 528 frames. Zero recovery training examples → model learned to decelerate near walls but never learned to reverse out.

### v2-simple (May 26, 2026)
- **Model:** Same NumPy MLP 12→128→64→32→2 trained on `data_v1.npz` with mirror-steering augmentation (→ 13,218 samples)
- **Augmentation:** Flip heading_error sign + swap symmetric ray pairs (4↔10, 5↔9, 6↔8) + negate steering action
- **Result:** val_loss=0.164, train_loss=0.098
- **Benchmark:** seed 42 → 0/5 completion, max_cp=2, crashes=0.4
- **Note:** Slightly regressed from v1. Mirror augmentation solved trajectory asymmetry but not the stuck problem. Still no recovery data.

### Session 2 — Full pipeline attempt (May 27, 2026)
*This session attempted a large-scale PyTorch approach. Key findings documented as lessons — see Section 5.*

**What was built and then removed:**
- PyTorch training script (`02_torch.py`) with BatchNorm + AdamW + cosine LR + weighted loss
- `drive2win/augment.py`, `combine_data.py`, `torch_mlp.py`, `smoothed_policy.py`, `ensemble.py`
- `03_benchmark_live.py` (single-session benchmark with browser prompt)
- `99_compete.py` (tournament entry point)

**What was trained:**
- Combined dataset: all 4 data files + mirror augmentation → 89,622 samples
- v3-torch (128-64-32, 500 epochs, 3x recovery weight): val=0.2714, **max_cp=3** on seeds 42/7/99
- v3-wide (256-128-64, 500 epochs, 3x recovery weight): val=0.2456, **max_cp=0–1** (BROKEN)
- v3-wide-nw (256-128-64, 300 epochs, no recovery weight): val=0.2375, not benchmarked before cleanup

**All trained weights deleted at end of session. Starting fresh.**

---

## 5. LESSONS LEARNED — READ BEFORE WRITING ANY CODE

### 5.1 Recovery weight backfired
Upweighting recovery frames (throttle < −0.3) by 3x caused the 256-128-64 model to learn near-zero throttle as a "safe" default everywhere. Recovery frames appear in sensor contexts nearly identical to normal driving (same track, just lower speed + wall nearby). A large model with too-strong weighting memorized "hesitate" as a loss-minimizing strategy.

**Rule:** If you use sample weighting for recovery, keep the multiplier ≤ 1.5x. Or don't weight — use the grace-period + stuck-recovery heuristic at inference time instead.

### 5.2 Lower val_loss ≠ better benchmark
- v3-wide had best val_loss (0.2456) but worst benchmark (max_cp=0, car frozen)
- v2-simple had val_loss=0.164 but achieved max_cp=6 on seed 7
- The validation set composition matters. When val includes recovery/multi-seed data, losses are naturally higher but don't indicate worse real driving.

### 5.3 Stuck-recovery heuristic needs a spawn grace period
Adding a stuck-recovery heuristic (if speed < 0.3 for N frames → force reverse + hard steer) caused the car to immediately reverse at spawn. The car spawns at speed=0, which looks identical to "stuck against a wall." Without a grace period of ~100 frames (5 seconds at 20 Hz), the car drives backward from the start for the entire run.

**Fix confirmed working:** Skip stuck detection for the first 100 frames of each run.

### 5.4 EMA smoothing is safe and helps
Exponential Moving Average on outputs (α=0.7) reduces steering jitter without harming lap times. Safe to always include at inference time.

### 5.5 Large combined dataset doesn't always beat focused data
v2-simple (trained only on seed 42) achieved max_cp=6 on seed 7, beating v3-torch (trained on 4 seeds combined). More data is not always better if the training signal gets diluted. Consider training separate models per seed or using curriculum learning.

### 5.6 The NumPy MLP is not the bottleneck
The core architecture (128-64-32, He init, Adam, MSE) is fine. The problem is data quality and inference-time robustness, not model capacity.

---

## 6. CURRENT FILE STRUCTURE

### Root
```
01_collect.py          — 5-phase manual data collection (WASD driving)
02_train.py            — NumPy MLP training with my_backward() + gradient check
03_benchmark.py        — Canonical evaluator (DO NOT EDIT)
04_compare.py          — Cross-iteration comparison, generates benchmarks/_history.png
game_client.py         — SDK (DO NOT EDIT)
ARCHITECTURES.md       — Guide for PyTorch/CNN/hybrid iterations
README.md
INSTRUCTOR_TODO.md
project_checkpoint.md  — Old checkpoint (superseded by this file)
CHECKPOINT.md          — This file
data_v1.npz            — 6,609 samples, seed 42
data_v2_recovery42.npz — 11,967 samples, seed 42 + recovery
data_v_seed7.npz       — 13,193 samples, seed 7
data_v_seed99.npz      — 13,042 samples, seed 99
```

### drive2win/ (core module — original files only)
```
__init__.py
nn.py          — NumPy MLP: forward, backward, Adam, He-init, save/load
                 Current arch: H1=128, H2=64, H3=32
normalize.py   — normalize_states(), sensors_to_input(), clip_action()
viz.py         — Plotting utilities
benchmark.py   — run_benchmark() — DO NOT EDIT
eval.py        — run_policy(), score_runs()
```

### benchmarks/
```
README.md          — Explains the folder format
v1.json            — Benchmark for v1 model (max_cp=3, completion=0/5)
v1_paths.png
v1_progress.png
v1_overlay.png
```

---

## 7. WHAT THE NEXT AGENT SHOULD DO

### Immediate priority: fix the stuck-at-corner problem

The car consistently reaches checkpoint 3 then freezes. The two proven approaches:

**Option A — Data approach (most reliable):**
The user has collected `data_v2_recovery42.npz` which contains deliberate wall-crash recovery frames. Train a new model that includes this data. The key is to include recovery data WITHOUT over-weighting it (keep recovery_weight ≤ 1.5x if weighting at all).

Suggested first run:
```
# Combine v1 + recovery42 only (stay on seed 42, focused)
# Augment with mirror-steering
# Train NumPy MLP first (simple baseline), then PyTorch if needed
```

**Option B — Inference-time approach (complements data approach):**
Add a stuck-recovery heuristic wrapper to the policy:
- If speed < 0.3 for > 60 consecutive frames (3 seconds at 20Hz)
- AND frame_count > 100 (past spawn grace period)
- THEN output throttle=−0.8, steering=±1.0 for 40 frames (steer direction from heading_error sign)

This was implemented, tested, and confirmed working in the previous session. It should always be included at tournament time.

### Process grade requirements
For each training iteration, commit:
1. The trained weights (`nav_<tag>.npz` or `.pt`)
2. `benchmarks/<tag>.json` + PNGs (from `03_benchmark.py`)
3. A meaningful commit message: `<tag>: <hypothesis> → <observed result>`

Run `python 04_compare.py v1 v2-simple <new-tag>` after each benchmark to update `benchmarks/_history.png`.

### Tournament strategy
- Benchmark on seeds 42, 7, 99 minimum before tournament day
- Test with obstacles enabled
- Use EMA smoothing (α=0.7) + stuck-recovery heuristic at tournament time
- Have a `99_compete.py`-style entry script ready before tournament day

---

## 8. KEY COMMANDS REFERENCE

```bash
# Collect new data (5 phases, ~5.5 min per session)
python 01_collect.py --tag <tag> --seed <seed>

# Train NumPy baseline
python 02_train.py --data <file.npz> --tag <tag> --epochs 300

# Benchmark (browser must be open — opens session URL, user confirms)
python 03_benchmark.py --tag <tag> --weights nav_<tag>.npz --seeds 42 7 99

# Compare iterations
python 04_compare.py v1 v2-simple <tag> ...
```

**Benchmark note:** `03_benchmark.py` creates a new session per run and prints the URL. The user must navigate to that URL in the browser before each run, or the simulation won't start (steps=0). A wrapper that prompts "Press Enter once browser is open" before each run fixes this.

---

## 9. ARCHITECTURE REFERENCE

### Current NumPy MLP (drive2win/nn.py)
```
H1=128, H2=64, H3=32 (set at top of file)
Input: 12 normalized features
Layer 1: Linear(12→128) + ReLU  — He init: σ = sqrt(2/12)
Layer 2: Linear(128→64) + ReLU  — He init: σ = sqrt(2/128)
Layer 3: Linear(64→32)  + ReLU  — He init: σ = sqrt(2/64)
Layer 4: Linear(32→2)   + tanh  — Xavier-ish init: σ = sqrt(1/32)
Loss: MSE over [throttle, steering]
Optimizer: Adam (β1=0.9, β2=0.999, ε=1e-8)
```

### PyTorch upgrade path (from ARCHITECTURES.md)
- Deeper/wider MLP: BatchNorm + LeakyReLU + Dropout + AdamW + cosine LR
- CNN over 32×32 terrain grid: use `client.cache_world_map()` + `client.get_grid_local()` for zero-latency local grid (no HTTP round-trip)
- Hybrid CNN + MLP: best for obstacle rounds
- Temporal MLP (stack last K frames): helps detect stuck state from history

---

## 10. NORMALISATION CONSTANTS (do not change without retraining)

```python
SPD_MAX  = 20.0   # speed clipped to [0, 1]
DIST_MAX = 100.0  # checkpoint_distance clipped to [0, 1]
RAY_MAX  = 50.0   # ray values clipped to [0, 1]
# heading_error divided by pi → [-1, 1]
# ground_friction already [0, 1] — pass through unchanged
```
