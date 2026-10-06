"""Sequential change detection with simulation-calibrated thresholds.

Testing each batch independently at alpha = 0.01 produces a false alarm every
~100 batches per statistic, and small persistent shifts are never detected.
CUSUM accumulates evidence across batches (Page, 1954):

    S_t = max(0, S_{t-1} + z_t - k),  alarm when S_t > h

``z_t`` is the batch statistic standardised against its *empirical null*
(statistics recomputed on fault-free batches resampled from the reference
period), and ``h`` is chosen by simulation so that the probability of any
false alarm within a monitoring horizon is at most ``target_fa``. That makes
the alarm threshold a stated, testable property instead of a magic number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class CusumState:
    s: float = 0.0
    n: int = 0
    alarms: int = 0
    last_alarm: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CusumConfig:
    k: float
    h: float
    target_fa: float
    horizon: int


def cusum_update(state: CusumState, z: float, cfg: CusumConfig) -> tuple[CusumState, bool]:
    if not np.isfinite(z):
        return CusumState(state.s, state.n + 1, state.alarms, state.last_alarm), False
    s = max(0.0, state.s + z - cfg.k)
    alarm = s > cfg.h
    if alarm:
        return CusumState(0.0, state.n + 1, state.alarms + 1, state.n), True
    return CusumState(s, state.n + 1, state.alarms, state.last_alarm), False


def calibrate_cusum(
    null_z: np.ndarray,
    k: float = 0.5,
    horizon: int = 100,
    target_fa: float = 0.02,
    n_sim: int = 4000,
    seed: int = 0,
) -> CusumConfig:
    """Choose h so P(max_t S_t > h within ``horizon`` batches | no change) = target_fa.

    Null sequences are bootstrapped from ``null_z``, so heavy tails or skew in
    the real statistic are respected (no Gaussian assumption).
    """
    null_z = np.asarray(null_z, dtype=float)
    null_z = null_z[np.isfinite(null_z)]
    if len(null_z) < 10:
        raise ValueError("need at least 10 null statistics to calibrate CUSUM")
    rng = np.random.default_rng(seed)
    draws = rng.choice(null_z, size=(n_sim, horizon), replace=True)
    s = np.zeros(n_sim)
    peak = np.zeros(n_sim)
    for t in range(horizon):
        s = np.maximum(0.0, s + draws[:, t] - k)
        peak = np.maximum(peak, s)
    h = float(np.quantile(peak, 1 - target_fa))
    return CusumConfig(k=k, h=max(h, 1.0), target_fa=target_fa, horizon=horizon)


def standardize(value: float, mean: float, std: float) -> float:
    return float((value - mean) / max(std, 1e-9))
