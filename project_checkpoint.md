# Drive2Win Project Checkpoint

This checkpoint file contains all the critical information regarding the current state of the **Drive2Win** Behavioral Cloning project. If you are a new AI agent resuming this task after token limits or context resets, **read this file in full first**.

---

## 1. PROJECT OVERVIEW
* **Task:** Train a policy network that outputs `[throttle, steering]` (both in `[-1, 1]`, tanh-activated) given a 12-feature sensor input vector every timestep.
* **Server:** `https://ml.ferit.tech`
* **Network Architecture:** 4-layer NumPy MLP — `12 → 128 → 64 → 32 → 2`, ReLU hidden activations, tanh output activation.
* **Training Method:** Behavioral Cloning (imitation learning of manual driving with obstacles).
* **Grade Structure:** 50% process (git commit log + benchmarks/) + 50% tournament performance.

---

## 2. CURRENT PROJECT STATE & ITERATION HISTORY (As of May 26, 2026)

### 📈 Iteration logs:
1. **`v1-baseline` (Baseline NumPy MLP)**:
   * **Setup:** Trained 12-128-64-32-2 NumPy MLP on raw manual driving data `data_v1.npz` (seed 42, driven with obstacles).
   * **Gradients:** Verified 100% correct with max relative error $\approx 8.5 \times 10^{-6}$ against numerical checks.
   * **Result:** `train_loss = 0.0875`, `val_loss = 0.1369`.
   * **Benchmark:** Seed 42 got `0/5` laps completed, `max_checkpoints = 3/8`, `mean_crashes = 0.2`. Got stuck at speed 0 due to zero recovery states in raw cleanliness of manual driving data.
   * **Commit:** `v1-baseline: 12-128-64-32-2 NumPy MLP will navigate course → val=0.137, completion=0/5, max_cp=3 (stuck/recovery issues)`

2. **`v2-simple` (NumPy MLP + Job A Data Augmentation)**:
   * **Strategy:** Keeping it elegant and simple!
   * **Job A Implementation:** Built `drive2win/augment.py` (mirror-steering) and `drive2win/combine_data.py` (compiling/merging). Generated `data_v1_aug.npz` with **13,218 perfectly symmetric samples** (doubled for free!).
   * **Job C PyTorch Option:** Built `02_torch.py` (Batch Norm, Dropout, AdamW, Cosine LR scheduler) and `drive2win/torch_mlp.py`. Tested it and found that for a tiny 12-feature control vector, over-parameterization/Dropout/Batch Norm introduced too much training-time noise, leading to steering oscillation. Returned to a simpler, more elegant NumPy MLP.
   * **Result:** Trained clean NumPy MLP on augmented mirrored data. `train_loss = 0.0982`, `val_loss = 0.1638`.
   * **Benchmark:** Seed 42 got `0/5` laps completed, `max_checkpoints = 2/8`, `mean_crashes = 0.2`. Perfectly symmetric sensor distributions, but gets stuck at the exact same spot due to steering jitter/oscillation bleeding off momentum.
   * **Commit:** `v2-simple: 12-128-64-32-2 NumPy MLP trained on mirrored data resolves trajectory asymmetry → val=0.164, completion=0/5, max_cp=2 (stuck at same corner)`

---

## 3. FILE SYSTEM STRUCTURE
* [01_collect.py](file:///c:/Users/alber/OneDrive/Desktop/LearnML_in3D/01_collect.py): Manual data collection tool.
* [02_train.py](file:///c:/Users/alber/OneDrive/Desktop/LearnML_in3D/02_train.py): Baseline NumPy training script with `my_backward()`.
* [02_torch.py](file:///c:/Users/alber/OneDrive/Desktop/LearnML_in3D/02_torch.py): PyTorch training pipeline.
* [03_benchmark.py](file:///c:/Users/alber/OneDrive/Desktop/LearnML_in3D/03_benchmark.py): Canonical evaluator.
* [04_compare.py](file:///c:/Users/alber/OneDrive/Desktop/LearnML_in3D/04_compare.py): Cross-iteration comparison utility.
* [drive2win/nn.py](file:///c:/Users/alber/OneDrive/Desktop/LearnML_in3D/drive2win/nn.py): Core NumPy net architecture.
* [drive2win/normalize.py](file:///c:/Users/alber/OneDrive/Desktop/LearnML_in3D/drive2win/normalize.py): Input/output scaling.
* [drive2win/augment.py](file:///c:/Users/alber/OneDrive/Desktop/LearnML_in3D/drive2win/augment.py): Steering-mirroring augmentation.
* [drive2win/combine_data.py](file:///c:/Users/alber/OneDrive/Desktop/LearnML_in3D/drive2win/combine_data.py): Merge and compile datasets without encoding conflicts on Windows.
* [drive2win/torch_mlp.py](file:///c:/Users/alber/OneDrive/Desktop/LearnML_in3D/drive2win/torch_mlp.py): PyTorch model adapter for standard benchmarks.

---

## 4. NEXT OPTIMAL ACTION PLAN (EMA smoothing & Recovery)
When resuming the project:
1. **Implement Job D: Action Smoothing / EMA (`smoothed_policy.py`)**:
   * Build an inference-time policy adapter that applies Exponential Moving Average:
     $$\text{Action}_t = \alpha \cdot \text{Action}_{\text{predicted}} + (1 - \alpha) \cdot \text{Action}_{t-1}$$
   * Tune $\alpha \in [0.6, 0.8]$ to eliminate steering jitter/oscillation, keeping velocity high and allowing the bot to glide past the corner obstacles.
2. **Collect targeted wall-recovery and obstacle correction data**:
   * If EMA smoothing is not enough, have the user collect a short `.npz` dataset focusing *strictly* on driving directly into walls/obstacles, reversing, correcting steering, and driving away.
   * Merge it with `data_v1_aug.npz` using `drive2win.combine_data`.
