from __future__ import annotations

import numpy as np


def _to_binary(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def intersection_over_union(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_mask = _to_binary(y_true)
    pred_mask = _to_binary(y_pred)
    intersection = np.logical_and(true_mask, pred_mask).sum()
    union = np.logical_or(true_mask, pred_mask).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def dice_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_mask = _to_binary(y_true)
    pred_mask = _to_binary(y_pred)
    intersection = np.logical_and(true_mask, pred_mask).sum()
    total = true_mask.sum() + pred_mask.sum()
    if total == 0:
        return 1.0
    return float((2.0 * intersection) / total)


def lpips_score(*_args: object, **_kwargs: object) -> float:
    raise NotImplementedError("LPIPS là optional dependency và chưa được triển khai trong phase này.")


def fid_score(*_args: object, **_kwargs: object) -> float:
    raise NotImplementedError("FID là optional dependency và chưa được triển khai trong phase này.")
