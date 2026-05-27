"""Multi-model ensemble policy for Drive2Win.

Loads two or more saved models (NumPy .npz or PyTorch .pt) and averages
their (throttle, steering) predictions. Averaging reduces variance between
runs and often beats any single model on unseen seeds.

Usage in benchmark:
    python 03_benchmark.py --tag v4-ensemble \
        --weights nav_v3-torch.pt \
        --module drive2win.ensemble_torch  # see make_torch_ensemble below

Or call make_ensemble_policy() directly in 99_compete.py.
"""
from __future__ import annotations
import numpy as np

from .normalize import sensors_to_input, clip_action
from . import nn as nn_mod


# ── NumPy model loader ───────────────────────────────────────────────────────

def _make_numpy_policy(weights_path: str):
    """Load a NumPy MLP (.npz) and return a raw policy function."""
    w = nn_mod.load(weights_path)

    def policy(state):
        x = sensors_to_input(state["sensors"])
        y = nn_mod.forward(x, w)
        return (float(y[0]), float(y[1]))

    return policy


# ── PyTorch model loader ─────────────────────────────────────────────────────

def _make_torch_policy(weights_path: str):
    """Load a PyTorch MLP (.pt) and return a raw policy function."""
    try:
        import torch
        from .torch_mlp import DeeperMLP
    except ImportError as e:
        raise ImportError("PyTorch required for .pt models. Install it first.") from e

    model = DeeperMLP()
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=False))
    model.eval()

    def policy(state):
        x = sensors_to_input(state["sensors"])
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y = model(x_t)[0].numpy()
        return (float(y[0]), float(y[1]))

    return policy


# ── Public API ────────────────────────────────────────────────────────────────

def make_ensemble_policy(weights_paths: list):
    """Create an averaging ensemble from a list of weight paths.

    Supports mixed .npz (NumPy) and .pt (PyTorch) paths.

    Args:
        weights_paths: List of paths, e.g.
            ["nav_v2-simple.npz", "nav_v3-torch.pt"]

    Returns:
        Policy callable (state_dict) -> (throttle, steering).
    """
    policies = []
    for path in weights_paths:
        if path.endswith(".pt"):
            policies.append(_make_torch_policy(path))
        else:
            policies.append(_make_numpy_policy(path))

    if not weights_paths:
        raise ValueError("weights_paths must be non-empty")
    assert len(policies) >= 1, "Need at least one model path."

    def policy(state):
        preds = [p(state) for p in policies]
        outputs = np.array([[r[0], r[1]] for r in preds], dtype=np.float32)
        mean = outputs.mean(axis=0)
        return clip_action(mean)

    return policy


def make_policy(weights_path: str):
    """Single-model shim so this module works as a --module argument.

    Example:
        python 03_benchmark.py --tag v3-torch \
            --weights nav_v3-torch.pt \
            --module drive2win.ensemble
    """
    return make_ensemble_policy([weights_path])
