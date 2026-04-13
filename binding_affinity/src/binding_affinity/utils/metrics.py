from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """A small scipy-free replacement for rankdata(method='average')."""
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values.copy()

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)

    i = 0
    while i < values.shape[0]:
        j = i + 1
        while j < values.shape[0] and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (float(i + 1) + float(j)) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def pearsonr(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())) + eps
    return float((x * y).sum() / denom)


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    rx = _rankdata_average(np.asarray(x, dtype=np.float64))
    ry = _rankdata_average(np.asarray(y, dtype=np.float64))
    return pearsonr(rx, ry)


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.sqrt(np.mean((x - y) ** 2)))


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    return {
        "pearson": pearsonr(pred, true),
        "spearman": spearmanr(pred, true),
        "rmse": rmse(pred, true),
    }


def fit_affine_calibration(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> Tuple[float, float]:
    """Fit affine calibration y ~= a * pred + b in least squares sense."""
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    y = np.asarray(true, dtype=np.float64).reshape(-1)
    if p.size == 0 or y.size == 0:
        return 1.0, 0.0
    if p.shape != y.shape:
        raise ValueError(f"pred/true must have same shape, got {p.shape} vs {y.shape}")

    p_mean = float(p.mean())
    y_mean = float(y.mean())
    p0 = p - p_mean
    y0 = y - y_mean
    var = float(np.mean(p0 * p0))
    if var <= float(eps):
        return 0.0, y_mean
    cov = float(np.mean(p0 * y0))
    a = cov / (var + float(eps))
    b = y_mean - a * p_mean
    return float(a), float(b)


def apply_affine(pred: np.ndarray, a: float, b: float) -> np.ndarray:
    p = np.asarray(pred, dtype=np.float64)
    return p * float(a) + float(b)
