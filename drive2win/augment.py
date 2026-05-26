"""Data Augmentation for Drive2Win.

Provides steering-mirroring functionality:
  1. Negates heading_error.
  2. Negates steering action.
  3. Swaps symmetric ray sensors:
     - ray_1_+45 (index 4) <-> ray_7_-45 (index 10)
     - ray_2_+90 (index 5) <-> ray_6_-90 (index 9)
     - ray_3_+135 (index 6) <-> ray_5_-135 (index 8)
     - ray_0_front (index 3) and ray_4_back (index 7) remain unchanged.
"""
from __future__ import annotations
import numpy as np

def mirror_states(states: np.ndarray) -> np.ndarray:
    """Mirror the states horizontally.
    
    Args:
        states: (N, 12) array of raw or normalized states.
    Returns:
        (N, 12) array of mirrored states.
    """
    mirrored = states.copy()
    
    # 1. Negate heading_error (index 1)
    mirrored[:, 1] = -states[:, 1]
    
    # 2. Swap ray pairs:
    # ray_1_+45 (idx 4) <-> ray_7_-45 (idx 10)
    mirrored[:, 4] = states[:, 10]
    mirrored[:, 10] = states[:, 4]
    
    # ray_2_+90 (idx 5) <-> ray_6_-90 (idx 9)
    mirrored[:, 5] = states[:, 9]
    mirrored[:, 9] = states[:, 5]
    
    # ray_3_+135 (idx 6) <-> ray_5_-135 (idx 8)
    mirrored[:, 6] = states[:, 8]
    mirrored[:, 8] = states[:, 6]
    
    return mirrored

def mirror_actions(actions: np.ndarray) -> np.ndarray:
    """Mirror steering actions horizontally.
    
    Args:
        actions: (N, 2) array containing [throttle, steering].
    Returns:
        (N, 2) array of mirrored actions.
    """
    mirrored = actions.copy()
    # Negate steering action (index 1)
    mirrored[:, 1] = -actions[:, 1]
    return mirrored

def augment_dataset(states: np.ndarray, actions: np.ndarray, positions: np.ndarray = None) -> tuple:
    """Double the dataset by appending the mirrored version.
    
    Args:
        states: (N, 12) array
        actions: (N, 2) array
        positions: (N, 2) array of [x, z] paths (optional)
    """
    mir_states = mirror_states(states)
    mir_actions = mirror_actions(actions)
    
    aug_states = np.concatenate([states, mir_states], axis=0)
    aug_actions = np.concatenate([actions, mir_actions], axis=0)
    
    if positions is not None:
        # Negate X coordinates in world space to keep positions mirrored
        mir_positions = positions.copy()
        mir_positions[:, 0] = -positions[:, 0]
        aug_positions = np.concatenate([positions, mir_positions], axis=0)
        return aug_states, aug_actions, aug_positions
        
    return aug_states, aug_actions
