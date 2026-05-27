"""Quick training script for v4 — combines v1 + recovery42, mirror augmentation.

Run:  python train_v4.py
Output:  nav_v4.npz  (~5 min on CPU)

This produces weights for the hybrid policy in 99_compete.py.
"""
from __future__ import annotations
import numpy as np
from drive2win import nn as nn_mod
from drive2win.normalize import normalize_states


def mirror_augment(states_raw: np.ndarray, actions: np.ndarray):
    """Flip left/right to double the dataset size.

    Negates heading_error, swaps symmetric ray pairs, negates steering.
    Ray index reference (from normalize.py):
        3=front, 4=+45, 5=+90, 6=+135, 7=back, 8=-135, 9=-90, 10=-45
    """
    s = states_raw.copy()
    a = actions.copy()
    s[:, 1] = -s[:, 1]                             # heading_error: negate
    s[:, 4], s[:, 10] = states_raw[:, 10].copy(), states_raw[:, 4].copy()  # +45 <-> -45
    s[:, 5], s[:, 9]  = states_raw[:, 9].copy(),  states_raw[:, 5].copy()  # +90 <-> -90
    s[:, 6], s[:, 8]  = states_raw[:, 8].copy(),  states_raw[:, 6].copy()  # +135 <-> -135
    a[:, 1] = -a[:, 1]                             # steering: negate
    return s, a


def train(X: np.ndarray, Y: np.ndarray,
          epochs: int = 300, lr: float = 1e-3,
          batch_size: int = 64, val_frac: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_val = max(1, int(len(X) * val_frac))
    v_idx, t_idx = perm[:n_val], perm[n_val:]
    Xtr, Ytr, Xva, Yva = X[t_idx], Y[t_idx], X[v_idx], Y[v_idx]

    w   = nn_mod.init_weights(seed=seed)
    opt = nn_mod.init_adam(w)
    best_val = float("inf")
    best_w = {k: v.copy() for k, v in w.items()}

    for epoch in range(epochs):
        idx = rng.permutation(len(Xtr))
        Xs, Ys = Xtr[idx], Ytr[idx]
        ep_loss, nb = 0.0, 0
        for i in range(0, len(Xs), batch_size):
            xb, yb = Xs[i:i + batch_size], Ys[i:i + batch_size]
            cache = nn_mod.forward_all(xb, w)
            ep_loss += nn_mod.mse_loss(cache["y"], yb)
            nb += 1
            grads = nn_mod.backward(xb, yb, w, cache)
            nn_mod.adam_step(w, grads, opt, lr=lr)
        val = nn_mod.mse_loss(nn_mod.forward(Xva, w), Yva)
        if val < best_val:
            best_val = val
            best_w = {k: v.copy() for k, v in w.items()}
        if epoch % 50 == 0 or epoch == epochs - 1:
            train_l = ep_loss / max(nb, 1)
            print(f"  epoch {epoch:3d}  train={train_l:.4f}  val={val:.4f}  best={best_val:.4f}")

    return best_w, best_val


def main():
    files = ["data_v1.npz", "data_v2_recovery42.npz"]
    all_states, all_actions = [], []
    for f in files:
        d = np.load(f, allow_pickle=False)
        print(f"Loaded {f}: {d['states'].shape[0]} samples")
        all_states.append(d["states"])
        all_actions.append(d["actions"])

    states_raw = np.concatenate(all_states, axis=0)
    actions    = np.concatenate(all_actions, axis=0).astype(np.float32)
    print(f"Combined : {len(states_raw)} samples")

    s_mir, a_mir = mirror_augment(states_raw, actions)
    states_raw = np.concatenate([states_raw, s_mir], axis=0)
    actions    = np.concatenate([actions, a_mir], axis=0)
    print(f"Augmented: {len(states_raw)} samples")

    # Print heading_error / steering correlation so 99_compete.py knows sign.
    corr = float(np.corrcoef(states_raw[:, 1], actions[:, 1])[0, 1])
    print(f"\nheading_error vs steering correlation: {corr:+.3f}")
    if corr < 0:
        print("  → STEER_SIGN = -1  (positive heading_error → steer right, negative heading_error → steer left)")
    else:
        print("  → STEER_SIGN = +1  (positive heading_error → steer left, positive heading_error → steer right)")
    print("  → Update STEER_SIGN in 99_compete.py if needed.\n")

    X = normalize_states(states_raw)
    Y = actions

    print("Training...")
    w, best_val = train(X, Y, epochs=300)

    nn_mod.save(w, "nav_v4.npz")
    print(f"\nSaved nav_v4.npz  (best val={best_val:.4f})")


if __name__ == "__main__":
    main()
