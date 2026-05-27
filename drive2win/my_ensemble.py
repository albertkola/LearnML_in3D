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
    "nav_v2-simple.npz",   # NumPy 128-64-32, seed-42 data
    "nav_v3-torch.pt",     # PyTorch 128-64-32, full data + weighted loss
]


def make_policy(weights_path: str):
    """weights_path is unused — ensemble loads MODELS above."""
    return make_ensemble_policy(MODELS)
