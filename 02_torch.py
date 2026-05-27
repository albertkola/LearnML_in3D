"""Step 2 (PyTorch) — Train a deep neural network policy in PyTorch.

Standard run (128-64-32, weighted loss):
    python 02_torch.py --data data_full.npz --tag v3-torch

Wider net experiment (256-128-64):
    python 02_torch.py --data data_full.npz --tag v3-wide --hidden 256 128 64

Weighted loss upweights recovery/reversing frames (throttle < -0.3) by 3x,
so the model pays extra attention to the hardest-to-learn behaviour.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from drive2win.normalize import normalize_states, N_FEATURES, N_ACTIONS
from drive2win import viz


class DeeperMLP(nn.Module):
    def __init__(self, n_in=N_FEATURES, h=(128, 64, 32), n_out=N_ACTIONS, dropout_p=0.1):
        super().__init__()
        layers = []
        sizes = [n_in, *h]
        for a, b in zip(sizes, sizes[1:]):
            layers += [
                nn.Linear(a, b),
                nn.BatchNorm1d(b),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_p),
            ]
        layers += [nn.Linear(sizes[-1], n_out), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def weighted_mse(pred: torch.Tensor, target: torch.Tensor,
                 weights: torch.Tensor) -> torch.Tensor:
    """Per-sample weighted MSE loss.

    weights: (N,) tensor — higher weight = more gradient signal from that sample.
    """
    sq = (pred - target) ** 2          # (N, 2)
    per_sample = sq.mean(dim=1)        # (N,)
    return (per_sample * weights).mean()


def make_sample_weights(actions_np: np.ndarray,
                        recovery_weight: float = 3.0) -> torch.Tensor:
    """Upweight recovery frames (negative throttle) so the model learns
    reversal behaviour from the deliberately collected crash-and-back-out data.

    Args:
        actions_np: (N, 2) float32 array — [throttle, steering].
        recovery_weight: multiplier applied to frames where throttle < -0.3.

    Returns:
        (N,) float32 tensor of per-sample weights, mean ≈ 1.0.
    """
    throttle = actions_np[:, 0]
    w = np.ones(len(throttle), dtype=np.float32)
    w[throttle < -0.3] = recovery_weight
    # Normalise so the total loss magnitude is comparable across experiments
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def train(X, Y, sample_weights, epochs=500, batch_size=64, lr=1e-3,
          weight_decay=1e-4, val_frac=0.1, seed=0, hidden=(128, 64, 32)):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    n_val = max(1, int(n_samples * val_frac))
    val_idx, tr_idx = indices[:n_val], indices[n_val:]

    X_tr, Y_tr, W_tr = X[tr_idx], Y[tr_idx], sample_weights[tr_idx]
    X_va, Y_va       = X[val_idx], Y[val_idx]

    tr_dataset = TensorDataset(X_tr, Y_tr, W_tr)
    tr_loader  = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)

    model     = DeeperMLP(h=hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=epochs)
    val_loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val  = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        ep_loss, n_batches = 0.0, 0
        for xb, yb, wb in tr_loader:
            optimizer.zero_grad()
            loss = weighted_mse(model(xb), yb, wb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n_batches += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = val_loss_fn(model(X_va), Y_va).item()

        tr_loss = ep_loss / max(1, n_batches)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d}  train={tr_loss:.4f}  val={val_loss:.4f}"
                  f"  best={best_val:.4f}")

    return best_state, train_losses, val_losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",         default="data_full.npz")
    ap.add_argument("--tag",          default="v3-torch")
    ap.add_argument("--epochs",       type=int,   default=500)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--batch",        type=int,   default=64)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden",       type=int,   nargs="+",
                    default=[128, 64, 32],
                    help="Hidden layer sizes, e.g. --hidden 256 128 64")
    ap.add_argument("--recovery_weight", type=float, default=3.0,
                    help="Loss multiplier for reversing frames (throttle < -0.3)")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    states_raw, actions = d["states"], d["actions"]
    print(f"Loaded: states {states_raw.shape}, actions {actions.shape}")

    recovery_frames = int((actions[:, 0] < -0.3).sum())
    print(f"Recovery frames (throttle < -0.3): {recovery_frames} "
          f"({100*recovery_frames/len(actions):.1f}%)")

    X = torch.tensor(normalize_states(states_raw), dtype=torch.float32)
    Y = torch.tensor(actions, dtype=torch.float32)
    W = make_sample_weights(actions, recovery_weight=args.recovery_weight)

    hidden = tuple(args.hidden)
    print(f"Architecture: {N_FEATURES} -> {' -> '.join(map(str, hidden))} -> {N_ACTIONS}")
    print(f"Recovery weight: {args.recovery_weight}x")

    best_weights, tr_losses, va_losses = train(
        X, Y, W,
        epochs=args.epochs, batch_size=args.batch, lr=args.lr,
        weight_decay=args.weight_decay, hidden=hidden,
    )

    viz.plot_loss_curves(tr_losses, va_losses, out=f"fig_loss_{args.tag}.png")

    out_path = f"nav_{args.tag}.pt"
    torch.save(best_weights, out_path)
    print(f"\nSaved {out_path}  (best val_loss={min(va_losses):.4f})")


if __name__ == "__main__":
    main()
