# Post-Collection Pipeline — Run These After All 3 Driving Sessions

Run each block in order. Each block is one terminal command.
Bold = **mandatory**. Normal = recommended for process grade.

---

## STEP 1 — Combine all data into one augmented file

```
python -m drive2win.combine_data --inputs data_v1.npz data_v2_recovery42.npz data_v_seed7.npz data_v_seed99.npz --output data_full.npz --augment
```

Expected output: `Augmented dataset size: ~XXXXX samples (Doubled OK)`

---

## STEP 2 — Train PyTorch model (500 epochs, ~10–15 min on CPU)

```
python 02_torch.py --data data_full.npz --tag v3-torch --epochs 500 --lr 1e-3 --batch 64 --weight_decay 1e-4
```

Produces: `nav_v3-torch.pt` + `fig_loss_v3-torch.png`
Watch the val_loss — aim for < 0.08. If train_loss << val_loss, overfitting.

---

## IMPORTANT — The game server requires a browser tab to run the simulation

Use `03_benchmark_live.py` (not `03_benchmark.py`) for all manual benchmarks.
It opens ONE session, asks you to open it in the browser, then runs all laps
automatically. You only need to open the browser ONCE per benchmark tag.

---

## STEP 3 — Benchmark v2-simple baseline (browser required)

```
python 03_benchmark_live.py --tag v2-simple --weights nav_v2-simple.npz --seeds 42 7 99
```

Open the printed URL in your browser when asked. All 15 runs happen automatically.

---

## STEP 4 — Benchmark the PyTorch model on 3 seeds (browser required)

```
python 03_benchmark_live.py --tag v3-torch --weights nav_v3-torch.pt --module drive2win.torch_mlp --seeds 42 7 99
```

---

## STEP 5 — (Optional) Ensemble v2-simple + v3-torch

```
python 03_benchmark_live.py --tag v4-ensemble --weights nav_v3-torch.pt --module drive2win.my_ensemble --seeds 42 7 99
```

---

## STEP 6 — Plot the improvement curve

```
python 04_compare.py v1 v2-simple v3-torch
```

Produces: `_history.png` — this goes in your final submission.

---

## STEP 7 — Commit everything (one commit per iteration for process grade)

```
git add nav_v2-simple.npz benchmarks/v2-simple.json benchmarks/v2-simple_paths.png benchmarks/v2-simple_progress.png
git commit -m "v2-simple: 12-128-64-32-2 numpy on augmented seed-42 data -> val=0.164, completion=?/5, max_cp=?"

git add data_v2_recovery42.npz data_v_seed7.npz data_v_seed99.npz data_full.npz
git commit -m "data: add recovery42 + seed7 + seed99 + mirror-augmented full dataset"

git add nav_v3-torch.pt fig_loss_v3-torch.png benchmarks/v3-torch*.json benchmarks/v3-torch*.png
git commit -m "v3-torch: PyTorch 128-64-32 BatchNorm+AdamW+cosine on data_full -> val=X.XXX, completion=?/5"

git add drive2win/smoothed_policy.py drive2win/ensemble.py 99_compete.py
git commit -m "add smoothed_policy (EMA+stuck-recovery), ensemble, and tournament entry (99_compete.py)"
```

---

## STEP 8 — Practice run before tournament

```
python 99_compete.py --weights nav_v3-torch.pt --module drive2win.torch_mlp --practice --seed 42
```

If it gets stuck even with recovery, lower `--stuck 25` to trigger recovery sooner.

---

## TOURNAMENT DAY COMMAND

```
python 99_compete.py --weights nav_v3-torch.pt --module drive2win.torch_mlp --player albertkola
```

The script will prompt you for each round's seed.
To use the ensemble instead: `--weights nav_v3-torch.pt nav_v2-simple.npz`
