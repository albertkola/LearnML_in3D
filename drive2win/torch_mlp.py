"""PyTorch MLP policy adapter for Drive2Win benchmark.

Exposes `make_policy(weights_path) -> policy_fn`.

Auto-detects the hidden layer sizes from the saved state dict, so the same
module works for any architecture trained with 02_torch.py (128-64-32,
256-128-64, etc.) without needing to pass flags to 03_benchmark.py.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np

from .normalize import sensors_to_input, clip_action, N_FEATURES, N_ACTIONS


class DeeperMLP(nn.Module):
    def __init__(self, n_in=N_FEATURES, h=(128, 64, 32), n_out=N_ACTIONS,
                 dropout_p=0.0):
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


def _infer_hidden(state_dict: dict) -> tuple:
    """Read hidden-layer sizes from the weight keys so we don't hard-code them."""
    hidden = []
    i = 0
    while True:
        key = f"net.{i}.weight"
        if key not in state_dict:
            break
        out_features = state_dict[key].shape[0]
        # The last Linear maps to N_ACTIONS (2) — that's the output layer, skip it
        if out_features != N_ACTIONS:
            hidden.append(out_features)
        i += 4  # Linear / BatchNorm1d / LeakyReLU / Dropout = 4 modules each
    return tuple(hidden)


def make_policy(weights_path: str):
    """Load weights and return a policy callable for use in benchmarks."""
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    hidden = _infer_hidden(state_dict)
    if not hidden:
        hidden = (128, 64, 32)  # safe fallback

    model = DeeperMLP(h=hidden)
    model.load_state_dict(state_dict)
    model.eval()

    def policy(state):
        x = sensors_to_input(state["sensors"])
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y = model(x_t)[0].numpy()
        return clip_action(y)

    return policy
