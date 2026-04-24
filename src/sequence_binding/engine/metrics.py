from __future__ import annotations

import math

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64, copy=False)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_ranks = np.empty(order.shape[0], dtype=np.float64)
    start = 0
    while start < order.shape[0]:
        end = start + 1
        while end < order.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        sorted_ranks[start:end] = avg_rank
        start = end
    ranks = np.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    pos_rank_sum = ranks[pos].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def binary_aupr(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64, copy=False)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def binary_classification_metrics(
    *,
    labels: np.ndarray,
    logits: np.ndarray,
    loss: float,
) -> dict[str, float]:
    probs = sigmoid(logits)
    preds = (probs >= 0.5).astype(np.int64)
    labels_i = labels.astype(np.int64, copy=False)
    accuracy = float((preds == labels_i).mean())
    tp = int(np.sum((preds == 1) & (labels_i == 1)))
    fp = int(np.sum((preds == 1) & (labels_i == 0)))
    fn = int(np.sum((preds == 0) & (labels_i == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    positive_rate = float(labels_i.mean())
    return {
        "loss": float(loss),
        "auc": binary_auc(labels_i, probs),
        "aupr": binary_aupr(labels_i, probs),
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_rate": positive_rate,
    }


def finite_or_none(value: float) -> float | None:
    if value is None:
        return None
    return None if math.isnan(value) else float(value)


def goodput(
    *,
    examples_per_sec: float,
    val_auc: float,
    baseline_val_auc: float,
    scaling_efficiency: float,
) -> float:
    """Goodput = throughput x statistical-quality ratio x hardware-scaling ratio.

    Product of raw throughput (examples/sec), a convergence-quality factor
    (<=1 when the parallel run reaches a worse AUC than the single-GPU
    reference), and a hardware-scaling factor (<=1 when parallel overhead
    dominates). Clamped at 0 to avoid spurious negative goodput.
    """
    quality_ratio = max(float(val_auc), 0.0) / max(float(baseline_val_auc), 1e-9)
    return float(examples_per_sec) * quality_ratio * max(float(scaling_efficiency), 0.0)
