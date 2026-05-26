"""Combine and Augment datasets.

Loads multiple data files, optionally applies mirroring augmentation, and
saves the consolidated dataset.
"""
from __future__ import annotations
import argparse
import numpy as np

from .augment import augment_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Paths to input .npz files.")
    ap.add_argument("--output", required=True,
                    help="Path to output augmented .npz file.")
    ap.add_argument("--augment", action="store_true",
                    help="Apply mirror-steering data augmentation.")
    args = ap.parse_args()

    all_states, all_actions, all_positions = [], [], []
    has_positions = True

    for path in args.inputs:
        d = np.load(path, allow_pickle=False)
        all_states.append(d["states"])
        all_actions.append(d["actions"])
        if "positions" in d.files:
            all_positions.append(d["positions"])
        else:
            has_positions = False

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    positions = np.concatenate(all_positions, axis=0) if has_positions else None

    print(f"Loaded {len(states)} samples from {len(args.inputs)} files.")

    if args.augment:
        if positions is not None:
            states, actions, positions = augment_dataset(states, actions, positions)
        else:
            states, actions = augment_dataset(states, actions)
        print(f"Augmented dataset size: {len(states)} samples (Doubled OK)")

    save_dict = {"states": states, "actions": actions}
    if positions is not None:
        save_dict["positions"] = positions

    np.savez(args.output, **save_dict)
    print(f"Saved consolidated dataset to {args.output}")

if __name__ == "__main__":
    main()
