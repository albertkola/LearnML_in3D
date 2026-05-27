"""EMA action smoothing + stuck-recovery heuristic wrapper.

Wraps any policy callable with two layers of protection:

1. Exponential Moving Average on (throttle, steering) — smooths out jittery
   outputs so the car doesn't oscillate on straights.

2. Stuck-recovery override — if the bot has been near-stationary for
   STUCK_THRESHOLD frames at 20 Hz (~2 s), it forces a reversal + hard turn
   until speed recovers. The steer direction is biased toward the track using
   heading_error so it doesn't back deeper into the wall.
"""
from __future__ import annotations
import numpy as np
from .normalize import clip_action

# ── Defaults (all overridable via make_smoothed_policy kwargs) ──────────────
EMA_ALPHA       = 0.7    # higher = faster response, lower = smoother output
STUCK_THRESHOLD = 40     # frames of speed < STUCK_SPEED before override fires
STUCK_SPEED     = 0.3    # m/s — what counts as "not moving"
RECOVERY_FRAMES = 30     # how many frames to hold the recovery action
RECOVERY_THROTTLE = -0.7 # reverse thrust
RECOVERY_STEER    = 0.9  # hard turn magnitude


def make_smoothed_policy(
    policy_fn,
    alpha: float = EMA_ALPHA,
    stuck_threshold: int = STUCK_THRESHOLD,
    recovery_frames: int = RECOVERY_FRAMES,
):
    """Wrap a raw policy with EMA smoothing and automatic stuck-recovery.

    Args:
        policy_fn: Any callable (state_dict) -> (throttle, steering).
        alpha: EMA coefficient in (0, 1]. Higher = less smoothing.
        stuck_threshold: Consecutive slow frames before recovery kicks in.
        recovery_frames: How many frames to hold the recovery action.

    Returns:
        A new policy callable with the same (state_dict) -> (float, float)
        signature, compatible with drive2win.eval.run_policy and
        drive2win.benchmark.
    """
    ema = np.array([0.0, 0.0], dtype=np.float32)
    stuck_count   = [0]
    recovery_left = [0]
    recovery_dir  = [1.0]

    def policy(state):
        sensors = state.get("sensors", {})
        speed   = float(sensors.get("speed", 0.0))

        # ── Stuck-recovery path ─────────────────────────────────────────
        if recovery_left[0] > 0:
            recovery_left[0] -= 1
            throttle = RECOVERY_THROTTLE
            steering = recovery_dir[0] * RECOVERY_STEER
            if speed > 1.5:
                # Recovered — cut the override short and clear stale EMA
                recovery_left[0] = 0
                stuck_count[0]   = 0
                ema[:] = 0.0
            return clip_action(np.array([throttle, steering], dtype=np.float32))

        # ── Stuck detection ─────────────────────────────────────────────
        if speed < STUCK_SPEED:
            stuck_count[0] += 1
        else:
            stuck_count[0] = 0

        if stuck_count[0] >= stuck_threshold:
            # Choose steer direction: turn away from the wall using heading_error.
            # Positive heading_error means we're pointing left of target → steer right.
            he = float(sensors.get("heading_error", 0.0))
            recovery_dir[0]  = -1.0 if he >= 0 else 1.0
            recovery_left[0] = recovery_frames
            stuck_count[0]   = 0
            # Reset EMA so we don't inherit the stuck-state smoothed output
            ema[:] = 0.0
            return clip_action(
                np.array([RECOVERY_THROTTLE, recovery_dir[0] * RECOVERY_STEER],
                         dtype=np.float32)
            )

        # ── Normal path: run the underlying policy + EMA ────────────────
        raw = policy_fn(state)
        raw_arr = np.array([float(raw[0]), float(raw[1])], dtype=np.float32)
        ema[:] = alpha * raw_arr + (1.0 - alpha) * ema
        return clip_action(ema.copy())

    return policy
