"""Two-model ensemble: NumPy v2-simple + PyTorch v3-torch (or v3-wide).

Usage:
    python 03_benchmark.py --tag v4-ensemble --weights nav_v3-torch.pt \
        --module drive2win.my_ensemble --seed 42

The weights_path argument is ignored — both models are hard-coded here so the
benchmark protocol (single --weights argument) stays intact.
Edit MODELS below if you want to swap in v3-wide instead.
"""
from drive2win.ensemble import make_ensemble_policy

MODELS = [
    "nav_v3-torch.pt",     # PyTorch 128-64-32, full data (consistent 3-seed coverage)
    "nav_v3-wide-nw.pt",   # PyTorch 256-128-64, best val_loss=0.2375 (standard MSE)
]


def make_policy(weights_path: str):
    """weights_path is unused — ensemble loads MODELS above."""
    return make_ensemble_policy(MODELS)
