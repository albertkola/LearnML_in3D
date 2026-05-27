"""Train v5 — all 4 seeds + mirror augmentation for tournament generalization.

Run:  python train_v5.py
Output: nav_v5.npz

Uses all collected data so the ML obstacle layer has seen obstacle patterns
from seeds 42, 7, and 99.  The arrow-following layer already works on any
seed; this improves the ML blend that activates near obstacles.
"""
from __future__ import annotations
import numpy as np
from drive2win import nn as nn_mod
from drive2win.normalize import normalize_states


def mirror_augment(states_raw: np.ndarray, actions: np.ndarray):
    s = states_raw.copy()
    a = actions.copy()
    s[:, 1] = -s[:, 1]
    s[:, 4],  s[:, 10] = states_raw[:, 10].copy(), states_raw[:, 4].copy()
    s[:, 5],  s[:, 9]  = states_raw[:, 9].copy(),  states_raw[:, 5].copy()
    s[:, 6],  s[:, 8]  = states_raw[:, 8].copy(),  states_raw[:, 6].copy()
    a[:, 1] = -a[:, 1]
    return s, a


def train(X, Y, epochs=300, lr=1e-3, batch_size=64, val_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_val = max(1, int(len(X) * val_frac))
    v_idx, t_idx = perm[:n_val], perm[n_val:]
    Xtr, Ytr, Xva, Yva = X[t_idx], Y[t_idx], X[v_idx], Y[v_idx]

    w   = nn_mod.init_weights(seed=seed)
    opt = nn_mod.init_adam(w)
    best_val = float("inf")
    best_w   = {k: v.copy() for k, v in w.items()}

    for epoch in range(epochs):
        idx = rng.permutation(len(Xtr))
        Xs, Ys = Xtr[idx], Ytr[idx]
        ep_loss, nb = 0.0, 0
        for i in range(0, len(Xs), batch_size):
            xb, yb = Xs[i:i+batch_size], Ys[i:i+batch_size]
            cache = nn_mod.forward_all(xb, w)
            ep_loss += nn_mod.mse_loss(cache["y"], yb)
            nb += 1
            grads = nn_mod.backward(xb, yb, w, cache)
            nn_mod.adam_step(w, grads, opt, lr=lr)
        val = nn_mod.mse_loss(nn_mod.forward(Xva, w), Yva)
        if val < best_val:
            best_val = val
            best_w   = {k: v.copy() for k, v in w.items()}
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d}  train={ep_loss/max(nb,1):.4f}  "
                  f"val={val:.4f}  best={best_val:.4f}")

    return best_w, best_val


def main():
    files = [
        "data_v1.npz",
        "data_v2_recovery42.npz",
        "data_v_seed7.npz",
        "data_v_seed99.npz",
    ]
    all_states, all_actions = [], []
    for f in files:
        d = np.load(f, allow_pickle=False)
        print(f"  {f}: {d['states'].shape[0]} samples")
        all_states.append(d["states"])
        all_actions.append(d["actions"])

    states_raw = np.concatenate(all_states,  axis=0)
    actions    = np.concatenate(all_actions, axis=0).astype(np.float32)
    print(f"Combined : {len(states_raw)} samples")

    s_mir, a_mir = mirror_augment(states_raw, actions)
    states_raw = np.concatenate([states_raw, s_mir], axis=0)
    actions    = np.concatenate([actions,    a_mir], axis=0)
    print(f"Augmented: {len(states_raw)} samples\n")

    X = normalize_states(states_raw)
    Y = actions

    print("Training...")
    w, best_val = train(X, Y, epochs=300)

    nn_mod.save(w, "nav_v5.npz")
    print(f"\nSaved nav_v5.npz  (best val={best_val:.4f})")
    print("Run: python 99_compete.py --weights nav_v5.npz --seed 42 --duration 60")


if __name__ == "__main__":
    main()
