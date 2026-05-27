"""Adapter so 03_benchmark.py can use the hybrid policy via --module drive2win.agent.

Usage:
    python 03_benchmark.py --tag v4-hybrid --weights nav_v4.npz \\
        --module drive2win.agent --seeds 42 7 99
"""
from __future__ import annotations
import importlib.util
import os

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_SPEC = importlib.util.spec_from_file_location(
    "compete", os.path.join(_ROOT, "99_compete.py")
)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def make_policy(weights_path: str):
    """Delegate to the hybrid policy in 99_compete.py."""
    return _MOD.make_policy(weights_path)
