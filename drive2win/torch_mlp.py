"""PyTorch MLP Policy adapter for Drive2Win benchmark.

Exposes `make_policy(weights_path) -> policy_fn`.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np

from .normalize import sensors_to_input, clip_action, N_FEATURES, N_ACTIONS

class DeeperMLP(nn.Module):
    def __init__(self, n_in=N_FEATURES, h=(128, 64, 32), n_out=N_ACTIONS):
        super().__init__()
        layers = []
        sizes = [n_in, *h]
        for a, b in zip(sizes, sizes[1:]):
            layers += [
                nn.Linear(a, b),
                nn.BatchNorm1d(b),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.0)  # Dropout is deactivated at inference, set to 0 here to keep architecture matching
            ]
        layers += [nn.Linear(sizes[-1], n_out), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def make_policy(weights_path: str):
    model = DeeperMLP()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    def policy(state):
        x = sensors_to_input(state["sensors"])              # (12,) float32
        # Convert to batch size 1 tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y = model(x_tensor)[0].numpy()
        return clip_action(y)

    return policy
