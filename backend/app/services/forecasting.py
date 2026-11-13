"""Time-series emissions forecasting with strict temporal validation."""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from app.services.model_training import MODEL_KEYS, predict_single


def generate_emissions_forecast(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
    bundle: Dict[str, Any],
    models_summary: Dict[str, Any],
    horizon_periods: int = 5,
    time_column: Optional[str] = None,
    model_key: Optional[str] = None,
    feature_growth_rates: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    c2c = mapping.get("concept_to_columns", {})
    temporal_cols = c2c.get("temporal", [])
    chosen_time_col = time_column if (time_column and time_column in df.columns) else (temporal_cols[0] if temporal_cols else None)

    if not chosen_time_col or chosen_time_col not in df.columns:
        raise ValueError(
            "Forecasting is not supported for this dataset because no valid temporal column (Year, Month, or Date) was found."
        )

    work_df = df.dropna(subset=[chosen_time_col, bundle["target_column"]]).copy()
    if len(work_df) < 6:
        raise ValueError(
            f"Insufficient historical data ({len(work_df)} records) for reliable forecasting. At least 6 historical observations are required."
        )

    chosen_model_key = model_key if (model_key and model_key in bundle["models"]) else bundle["best_model_key"]
    target_col = bundle["target_column"]
    feature_cols = bundle["feature_columns"]

    # Aggregate historical series by time period
    is_num_time = pd.api.types.is_numeric_dtype(work_df[chosen_time_col])
    if is_num_time:
        work_df = work_df.sort_values(by=chosen_time_col)
        grouped = work_df.groupby(chosen_time_col, as_index=False).mean(numeric_only=True)
        periods = grouped[chosen_time_col].to_numpy(dtype=float)
        step = float(np.median(np.diff(periods))) if len(periods) > 1 else 1.0
        if step <= 0:
            step = 1.0
    else:
        work_df["_parsed_dt"] = pd.to_datetime(work_df[chosen_time_col], errors="coerce")
        work_df = work_df.dropna(subset=["_parsed_dt"]).sort_values(by="_parsed_dt")
        if len(work_df) < 4:
            raise ValueError(f"Temporal column '{chosen_time_col}' could not be parsed into a valid chronological sequence.")
        grouped = work_df.groupby("_parsed_dt", as_index=False).mean(numeric_only=True)
        periods = grouped["_parsed_dt"].tolist()
        step_days = int(np.median([(periods[i + 1] - periods[i]).days for i in range(len(periods) - 1)])) if len(periods) > 1 else 90

    historical_series: List[Dict[str, Any]] = []
    for _, row in grouped.iterrows():
        p_label = str(int(row[chosen_time_col])) if is_num_time else row["_parsed_dt"].strftime("%Y-%m-%d")
        historical_series.append({
            "period": p_label,
            "actual_emissions": round(float(row[target_col]), 4),
            "type": "historical",
        })

    # Estimate empirical per-period drift for numeric features using recent historical trend
    last_row = work_df.iloc[-1].to_dict()
    feature_drifts: Dict[str, float] = {}
    for fcol in feature_cols:
        if fcol == chosen_time_col:
            continue
        if feature_growth_rates and fcol in feature_growth_rates:
            feature_drifts[fcol] = float(feature_growth_rates[fcol]) / 100.0
        elif fcol in grouped.columns and len(grouped) >= 3:
            vals = grouped[fcol].to_numpy(dtype=float)
            if abs(vals[0]) > 1e-6:
                total_rel = (vals[-1] - vals[0]) / abs(vals[0])
                per_step = np.clip(total_rel / max(1, len(vals) - 1), -0.08, 0.08)
                feature_drifts[fcol] = float(per_step)
            else:
                feature_drifts[fcol] = 0.0
        else:
            feature_drifts[fcol] = 0.0

    forecast_series: List[Dict[str, Any]] = []
    current_inputs = {col: last_row.get(col) for col in feature_cols}

    for h in range(1, horizon_periods + 1):
        if is_num_time:
            next_period_val = float(periods[-1]) + h * step
            next_label = str(int(next_period_val)) if step.is_integer() else f"{next_period_val:.1f}"
            if chosen_time_col in current_inputs:
                current_inputs[chosen_time_col] = next_period_val
        else:
            next_dt = periods[-1] + pd.Timedelta(days=h * max(1, step_days))
            next_label = next_dt.strftime("%Y-%m-%d")

        # Advance numeric operational features by their historical or user-specified drift
        step_inputs = dict(current_inputs)
        for fcol, drift in feature_drifts.items():
            if fcol in step_inputs and isinstance(step_inputs[fcol], (int, float, np.number)):
                step_inputs[fcol] = float(step_inputs[fcol]) * ((1.0 + drift) ** h)

        pred_out = predict_single(bundle, models_summary, step_inputs, model_key=chosen_model_key)
        # Widening uncertainty cone over forecast horizon
        horizon_multiplier = float(np.sqrt(h))
        half_width = 1.96 * (pred_out["uncertainty_std"] or 0.0) * horizon_multiplier

        forecast_series.append({
            "horizon_step": h,
            "period": next_label,
            "predicted_emissions": pred_out["predicted_value"],
            "lower_bound_95": round(pred_out["predicted_value"] - half_width, 4),
            "upper_bound_95": round(pred_out["predicted_value"] + half_width, 4),
            "assumed_inputs": {k: (round(float(v), 3) if isinstance(v, (int, float, np.number)) else v) for k, v in step_inputs.items()},
            "type": "forecast",
        })

    return {
        "supported": True,
        "time_column": chosen_time_col,
        "target_column": target_col,
        "unit": bundle.get("target_unit", "tCO2e"),
        "model_key": chosen_model_key,
        "model_name": MODEL_KEYS.get(chosen_model_key, chosen_model_key),
        "horizon_periods": horizon_periods,
        "historical_series": historical_series,
        "forecast_series": forecast_series,
        "methodology_note": (
            f"Forecast generated across {horizon_periods} future periods along '{chosen_time_col}' using "
            f"{MODEL_KEYS.get(chosen_model_key, chosen_model_key)} with √h uncertainty scaling."
        ),
    }
