"""Step 2 (PyTorch) — Train a deep neural network policy in PyTorch.

Run:  python 02_torch.py --data data_v1_aug.npz --tag v2-torch
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
                nn.Dropout(dropout_p)
            ]
        layers += [nn.Linear(sizes[-1], n_out), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train(X, Y, epochs=300, batch_size=64, lr=1e-3, weight_decay=1e-4, val_frac=0.1, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Train-val split
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    n_val = max(1, int(n_samples * val_frac))
    val_idx, tr_idx = indices[:n_val], indices[n_val:]
    
    X_tr, Y_tr = X[tr_idx], Y[tr_idx]
    X_va, Y_va = X[val_idx], Y[val_idx]
    
    # DataLoaders
    tr_dataset = TensorDataset(X_tr, Y_tr)
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
    
    # Model & Optim
    model = DeeperMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        n_batches = 0
        for xb, yb in tr_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            
            ep_loss += loss.item()
            n_batches += 1
            
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va), Y_va).item()
            
        train_loss = ep_loss / max(1, n_batches)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 25 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  best={best_val_loss:.4f}")
            
    return best_state, train_losses, val_losses

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_v1_aug.npz",
                    help="Consolidated or augmented dataset.")
    ap.add_argument("--tag", default="v2-torch",
                    help="Output suffix (nav_<tag>.pt, fig_*_<tag>.png)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    args = ap.parse_args()

    # Load dataset
    d = np.load(args.data, allow_pickle=False)
    states_raw, actions = d["states"], d["actions"]
    print(f"Loaded dataset: states {states_raw.shape}, actions {actions.shape}")

    X = torch.tensor(normalize_states(states_raw), dtype=torch.float32)
    Y = torch.tensor(actions, dtype=torch.float32)

    # Train
    best_weights, tr_losses, va_losses = train(
        X, Y, epochs=args.epochs, lr=args.lr, batch_size=args.batch, weight_decay=args.weight_decay
    )

    # Save diagnostics and model
    viz.plot_loss_curves(tr_losses, va_losses, out=f"fig_loss_{args.tag}.png")
    
    out_path = f"nav_{args.tag}.pt"
    torch.save(best_weights, out_path)
    print(f"Saved {out_path} ✓")

if __name__ == "__main__":
    main()
