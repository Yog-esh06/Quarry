"""Exact Shapley attributions over features (d <= 12) by coalition enumeration.

Two uses, one engine:

* **Local explanation** of a single prediction: v(S) = E_b[f(x_S, b_{-S})]
  over a background sample b (interventional Shapley values).
* **Prediction-shift explanation** for an incident: why did the mean predicted
  default rate move between reference and current traffic?
  v(S) = mean_i f(cur_i[S], ref_i[-S]) over random (cur, ref) row pairs, so
  phi_j is feature j's share of mean(f(cur)) - mean(f(ref)).

Enumerating all 2^d coalitions is exact (no sampling noise), and with d = 11
it is ~2k coalitions: a few hundred thousand vectorised predictions. Note the
interventional value function breaks feature dependence; the causal layer is
the tool for "which mechanism", this one answers "which inputs moved the
score".
"""

from __future__ import annotations

from math import factorial
from typing import Callable

import numpy as np
import pandas as pd

MAX_EXACT = 12


def _coalition_masks(d: int) -> np.ndarray:
    ints = np.arange(2**d)
    return ((ints[:, None] >> np.arange(d)[None, :]) & 1).astype(bool)


def shapley_from_values(values: np.ndarray, d: int) -> np.ndarray:
    """values[s] = v(S) for coalition bitmask s; returns phi (d,)."""
    ints = np.arange(2**d)
    sizes = np.array([bin(i).count("1") for i in ints])
    weights = np.array([factorial(s) * factorial(d - s - 1) / factorial(d) if s < d else 0.0 for s in range(d + 1)])
    phi = np.zeros(d)
    for j in range(d):
        without = ints[(ints >> j) & 1 == 0]
        phi[j] = np.sum(weights[sizes[without]] * (values[without | (1 << j)] - values[without]))
    return phi


def _hybrid_frame(fg: pd.DataFrame, bg: pd.DataFrame, features: list[str], masks: np.ndarray) -> pd.DataFrame:
    """Rows for every (coalition, pair): coalition features from fg, the rest from bg."""
    n = len(fg)
    k = len(masks)
    cols = {}
    for j, name in enumerate(features):
        f, b = fg[name].to_numpy(), bg[name].to_numpy()
        take = np.repeat(masks[:, j], n)
        cols[name] = np.where(take, np.tile(f, k), np.tile(b, k))
    return pd.DataFrame(cols)


def shift_attribution(
    predict: Callable[[pd.DataFrame], np.ndarray],
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    features: list[str],
    n_pairs: int = 256,
    seed: int = 0,
) -> dict:
    if len(features) > MAX_EXACT:
        raise ValueError(f"exact Shapley supports at most {MAX_EXACT} features")
    rng = np.random.default_rng(seed)
    n = min(n_pairs, len(ref), len(cur))
    fg = cur.iloc[rng.choice(len(cur), n, replace=False)][features].reset_index(drop=True)
    bg = ref.iloc[rng.choice(len(ref), n, replace=False)][features].reset_index(drop=True)
    d = len(features)
    masks = _coalition_masks(d)
    preds = predict(_hybrid_frame(fg, bg, features, masks))
    values = preds.reshape(len(masks), n).mean(axis=1)
    phi = shapley_from_values(values, d)
    return {
        "base": float(values[0]),  # mean prediction on reference rows
        "current": float(values[-1]),  # mean prediction on current rows
        "contributions": {f: float(p) for f, p in zip(features, phi)},
        "n_pairs": n,
    }


def local_attribution(
    predict: Callable[[pd.DataFrame], np.ndarray],
    row: pd.DataFrame,
    background: pd.DataFrame,
    features: list[str],
    n_background: int = 64,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    n = min(n_background, len(background))
    bg = background.iloc[rng.choice(len(background), n, replace=False)][features].reset_index(drop=True)
    fg = pd.concat([row[features]] * n, ignore_index=True)
    d = len(features)
    masks = _coalition_masks(d)
    preds = predict(_hybrid_frame(fg, bg, features, masks))
    values = preds.reshape(len(masks), n).mean(axis=1)
    phi = shapley_from_values(values, d)
    return {
        "base": float(values[0]),
        "prediction": float(values[-1]),
        "contributions": {f: float(p) for f, p in zip(features, phi)},
    }
