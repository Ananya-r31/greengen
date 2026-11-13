"""Regression model evaluation metrics: MAE, RMSE, R2, and conditional MAPE."""
from typing import Any, Dict, List
import numpy as np


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot > 1e-12:
        r2 = float(1.0 - (ss_res / ss_tot))
    else:
        r2 = 0.0

    # MAPE is only mathematically appropriate when target values are strictly non-zero (not near zero)
    near_zero_mask = np.abs(y_true) < 1e-3
    if np.any(near_zero_mask):
        mape = None
        mape_note = "Omitted: target contains zero or near-zero (<0.001) values where MAPE is undefined/unstable."
    else:
        mape = round(float(np.mean(np.abs(residuals / y_true)) * 100.0), 3)
        mape_note = "Valid: all held-out target values are strictly non-zero."

    # Build actual vs predicted and residual series for interactive charts (capped at 150 points for payload efficiency)
    limit = min(len(y_true), 150)
    comparison_points: List[Dict[str, float]] = []
    for idx in range(limit):
        comparison_points.append({
            "index": int(idx + 1),
            "actual": round(float(y_true[idx]), 4),
            "predicted": round(float(y_pred[idx]), 4),
            "residual": round(float(residuals[idx]), 4),
        })

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "mape": mape,
        "mape_note": mape_note,
        "n_test_samples": int(len(y_true)),
        "comparison_points": comparison_points,
    }
