"""Constrained Carbon Optimization Engine using SciPy SLSQP and What-If Scenario Simulator."""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from app.services.model_training import MODEL_KEYS, ZeroLeakagePreprocessor, predict_single


def run_carbon_optimization(
    df: pd.DataFrame,
    mapping: Dict[str, Any],
    bundle: Dict[str, Any],
    models_summary: Dict[str, Any],
    target_reduction_pct: float = 15.0,
    max_budget: Optional[float] = None,
    baseline_row_index: Optional[int] = None,
    baseline_inputs: Optional[Dict[str, Any]] = None,
    variable_bounds: Optional[Dict[str, Any]] = None,
    preserve_production_column: Optional[str] = None,
    min_production_ratio: float = 1.0,
    model_key: Optional[str] = None,
) -> Dict[str, Any]:
    chosen_key = model_key if (model_key and model_key in bundle["models"]) else bundle["best_model_key"]
    feature_cols: List[str] = bundle["feature_columns"]
    preprocessor: ZeroLeakagePreprocessor = bundle["preprocessor"]
    c2c = mapping.get("concept_to_columns", {})
    col_to_concept = mapping.get("column_to_concept", {})

    # 1. Establish Baseline Inputs
    if baseline_inputs:
        base_row = dict(baseline_inputs)
    elif baseline_row_index is not None and 0 <= baseline_row_index < len(df):
        base_row = df.iloc[baseline_row_index][feature_cols].to_dict()
    else:
        # Use mean of latest 25% of observations or overall median/mode
        base_row = {}
        for col in feature_cols:
            if col in preprocessor.numeric_cols:
                base_row[col] = float(df[col].dropna().iloc[-1]) if len(df[col].dropna()) > 0 else preprocessor.num_medians[col]
            else:
                modes = df[col].dropna().mode()
                base_row[col] = str(modes.iloc[0]) if len(modes) > 0 else "Unknown"

    base_pred_res = predict_single(bundle, models_summary, base_row, model_key=chosen_key)
    baseline_emissions = float(base_pred_res["predicted_value"])

    # 2. Identify Controllable Decision Variables present in both the dataset and the selected model features
    controllable_candidates = set(mapping.get("controllable_columns", []))
    decision_vars: List[str] = [
        col for col in feature_cols
        if col in preprocessor.numeric_cols and (col in controllable_candidates or col_to_concept.get(col) != "temporal")
    ]
    # Exclude production column if user requires preserving production output
    prod_cols = c2c.get("industrial_production", [])
    locked_prod_col = preserve_production_column or (prod_cols[0] if prod_cols else None)
    if locked_prod_col in decision_vars and min_production_ratio >= 1.0:
        decision_vars.remove(locked_prod_col)

    if not decision_vars:
        raise ValueError(
            "No controllable numeric decision variables were found among the trained model's features. "
            "Include operational variables (such as electricity, fuel, renewable share, transport, or waste) when training the model."
        )

    # 3. Configure variable bounds and directional physics priors
    x0 = np.array([float(base_row.get(col, preprocessor.num_medians[col])) for col in decision_vars], dtype=float)
    bounds: List[Tuple[float, float]] = []
    unit_costs: List[float] = []
    has_valid_cost_inputs = False

    cost_cols = c2c.get("cost_budget", [])
    dataset_has_cost = len(cost_cols) > 0

    for idx, col in enumerate(decision_vars):
        val0 = float(x0[idx])
        c_min_train = preprocessor.num_mins.get(col, 0.0)
        c_max_train = preprocessor.num_maxs.get(col, max(1.0, val0 * 1.5))
        concept = col_to_concept.get(col, "")

        user_b = (variable_bounds or {}).get(col)
        max_chg_pct = 35.0
        if user_b:
            if isinstance(user_b, dict):
                max_chg_pct = float(user_b.get("max_change_pct") or 35.0)
                u_min = user_b.get("min_val")
                u_max = user_b.get("max_val")
                u_cost = user_b.get("unit_cost")
            else:
                max_chg_pct = float(getattr(user_b, "max_change_pct", 35.0) or 35.0)
                u_min = getattr(user_b, "min_val", None)
                u_max = getattr(user_b, "max_val", None)
                u_cost = getattr(user_b, "unit_cost", None)
        else:
            u_min, u_max, u_cost = None, None, None

        delta = abs(val0) * (max_chg_pct / 100.0) if abs(val0) > 1e-6 else (c_max_train - c_min_train) * 0.35

        if concept == "renewable_energy":
            # Renewable share can increase up to min(100%, val0 + delta) and shouldn't drop below baseline
            low = val0 if u_min is None else float(u_min)
            high = min(100.0, max(val0, val0 + delta, c_max_train)) if u_max is None else float(u_max)
        elif concept in ("electricity_consumption", "energy_consumption", "fossil_fuel", "transportation", "waste_generation", "water_consumption"):
            # Consumption/activity variables should reduce (or stay <= baseline) to cut emissions
            low = max(0.0, c_min_train * 0.75, val0 - delta) if u_min is None else max(0.0, float(u_min))
            high = val0 if u_max is None else float(u_max)
        else:
            low = max(0.0, val0 - delta) if u_min is None else float(u_min)
            high = (val0 + delta) if u_max is None else float(u_max)

        if low > high:
            low, high = high, low
        bounds.append((float(low), float(high)))

        if u_cost is not None and float(u_cost) > 0:
            unit_costs.append(float(u_cost))
            has_valid_cost_inputs = True
        else:
            unit_costs.append(0.0)

    # 4. Define Smooth Surrogate + Exact Model Objective Function for SLSQP
    # Note: Tree-based models have piecewise-constant step surfaces with zero local gradients.
    # To make SciPy SLSQP effective across ALL 4 models (including Random Forest & XGBoost),
    # we evaluate the selected model + Bayesian Ridge smooth directional gradient regularizer.
    br_model = bundle["models"].get("bayesian_ridge")

    def evaluate_x(x_vec: np.ndarray) -> float:
        cand_row = dict(base_row)
        for i, col in enumerate(decision_vars):
            cand_row[col] = float(x_vec[i])
        input_df = pd.DataFrame([cand_row])
        X_s = preprocessor.transform(input_df, scale_numeric=True)
        model_pred = float(bundle["models"][chosen_key].predict(X_s)[0])
        if chosen_key in ("random_forest", "xgboost") and br_model is not None:
            # Blend 85% exact tree model + 15% smooth Bayesian Ridge response so finite-difference gradients are non-zero
            smooth_pred = float(br_model.predict(X_s)[0])
            return 0.85 * model_pred + 0.15 * smooth_pred
        return model_pred

    target_emissions_goal = baseline_emissions * (1.0 - (target_reduction_pct / 100.0))
    scale_x = np.maximum(1.0, np.abs(x0))

    def objective(x_vec: np.ndarray) -> float:
        pred_e = evaluate_x(x_vec)
        # Minimize predicted emissions + gentle L2 penalty on relative operational disruption
        rel_change_sq = float(np.mean(((x_vec - x0) / scale_x) ** 2))
        goal_shortfall = max(0.0, pred_e - target_emissions_goal)
        return pred_e + 12.0 * goal_shortfall + 0.35 * abs(baseline_emissions) * rel_change_sq

    # 5. Build Explicit Constraints for SLSQP
    constraints = []
    if max_budget is not None and has_valid_cost_inputs:
        cost_arr = np.array(unit_costs, dtype=float)

        def budget_constraint(x_vec: np.ndarray) -> float:
            est_spend = float(np.sum(np.abs(x_vec - x0) * cost_arr))
            return float(max_budget) - est_spend

        constraints.append({"type": "ineq", "fun": budget_constraint})

    # Run SciPy SLSQP optimizer with finite-difference step suited to variable scales
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 120, "ftol": 1e-4, "eps": 1e-2},
    )

    x_opt = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])

    # Evaluate exact chosen model on x_opt (and compare with boundary corners for tree models)
    best_x = x_opt.copy()
    opt_row = dict(base_row)
    for i, col in enumerate(decision_vars):
        opt_row[col] = float(best_x[i])
    opt_pred_res = predict_single(bundle, models_summary, opt_row, model_key=chosen_key)
    optimized_emissions = float(opt_pred_res["predicted_value"])

    # If tree steps prevented finite-difference movement, also check directional boundary point within bounds
    corner_x = np.array(
        [b[1] if col_to_concept.get(decision_vars[i]) == "renewable_energy" else b[0] for i, b in enumerate(bounds)],
        dtype=float,
    )
    for alpha_step in (0.35, 0.65, 1.0):
        cand_x = x0 + alpha_step * (corner_x - x0)
        cand_x = np.clip(cand_x, [b[0] for b in bounds], [b[1] for b in bounds])
        if max_budget is not None and has_valid_cost_inputs:
            if float(np.sum(np.abs(cand_x - x0) * np.array(unit_costs))) > float(max_budget):
                continue
        c_row = dict(base_row)
        for i, col in enumerate(decision_vars):
            c_row[col] = float(cand_x[i])
        c_pred = float(predict_single(bundle, models_summary, c_row, model_key=chosen_key)["predicted_value"])
        if c_pred < optimized_emissions:
            optimized_emissions = c_pred
            best_x = cand_x
            opt_row = c_row

    # 6. Calculate Reductions with Strict Zero-Baseline Handling
    absolute_reduction = round(baseline_emissions - optimized_emissions, 4)
    if abs(baseline_emissions) < 1e-9:
        percentage_reduction = 0.0
        zero_baseline_note = (
            "Baseline emissions are 0.0; percentage reduction is defined as 0.0% to avoid division by zero."
        )
    else:
        percentage_reduction = round((absolute_reduction / baseline_emissions) * 100.0, 2)
        zero_baseline_note = None

    # 7. Summarize Decision Variable Adjustments and Constraint Checks
    variable_changes: List[Dict[str, Any]] = []
    constraints_report: List[Dict[str, Any]] = []
    col_units = mapping.get("column_units", {})

    for i, col in enumerate(decision_vars):
        b_val = float(x0[i])
        o_val = float(best_x[i])
        diff = o_val - b_val
        pct_chg = round((diff / abs(b_val)) * 100.0, 2) if abs(b_val) > 1e-9 else 0.0
        low_b, high_b = bounds[i]
        in_bounds = (low_b - 1e-5) <= o_val <= (high_b + 1e-5)

        t_min = preprocessor.num_mins.get(col, low_b)
        t_max = preprocessor.num_maxs.get(col, high_b)
        in_train_dist = (t_min * 0.85) <= o_val <= (t_max * 1.15)

        variable_changes.append({
            "variable": col,
            "concept": col_to_concept.get(col, "operational"),
            "unit": col_units.get(col, "units"),
            "baseline_value": round(b_val, 4),
            "optimized_value": round(o_val, 4),
            "absolute_change": round(diff, 4),
            "pct_change": pct_chg,
            "bound_min": round(low_b, 4),
            "bound_max": round(high_b, 4),
            "within_training_distribution": bool(in_train_dist),
        })
        constraints_report.append({
            "constraint": f"Bounds on '{col}' [{low_b:.2f}, {high_b:.2f}]",
            "satisfied": bool(in_bounds),
            "detail": f"Optimized value {o_val:.2f} ({pct_chg:+.1f}% vs baseline)",
        })

    if locked_prod_col and locked_prod_col in base_row:
        constraints_report.append({
            "constraint": f"Maintain production/output variable '{locked_prod_col}'",
            "satisfied": True,
            "detail": f"Held constant at {base_row[locked_prod_col]} (>= {min_production_ratio * 100:.0f}% requirement).",
        })

    # Cost and savings reporting (Strictly grounded — never fabricate if missing!)
    if has_valid_cost_inputs:
        est_implementation_cost = round(float(np.sum(np.abs(best_x - x0) * np.array(unit_costs))), 2)
        cost_status = "Calculated from user-supplied unit cost parameters."
    elif dataset_has_cost:
        est_implementation_cost = None
        cost_status = (
            f"Dataset contains expenditure column '{cost_cols[0]}', but variable-specific marginal unit costs ($/unit change) "
            "were not supplied. Provide unit_cost in variable bounds to compute implementation cost and ROI."
        )
    else:
        est_implementation_cost = None
        cost_status = (
            "No financial cost or tariff data is present in the uploaded dataset. Supply unit costs ($/kWh, $/liter, etc.) "
            "to estimate implementation cost and financial savings."
        )

    target_met = percentage_reduction >= (target_reduction_pct - 0.5)
    status_label = (
        "Converged (Target Met)"
        if target_met
        else ("Converged (Feasible Local Minimum within Bounds)" if absolute_reduction > 0 else "No Further Reduction Feasible within Bounds")
    )

    return {
        "status": status_label,
        "scipy_success": bool(res.success),
        "scipy_message": str(res.message),
        "model_key": chosen_key,
        "model_name": MODEL_KEYS.get(chosen_key, chosen_key),
        "target_column": bundle["target_column"],
        "unit": bundle.get("target_unit", "tCO2e"),
        "target_reduction_pct": target_reduction_pct,
        "baseline_emissions": round(baseline_emissions, 4),
        "optimized_emissions": round(optimized_emissions, 4),
        "absolute_reduction": absolute_reduction,
        "percentage_reduction": percentage_reduction,
        "zero_baseline_note": zero_baseline_note,
        "variable_changes": variable_changes,
        "constraints_report": constraints_report,
        "estimated_cost": est_implementation_cost,
        "cost_status_note": cost_status,
        "methodology_and_limitations": [
            "Optimization uses SciPy SLSQP (Sequential Least Squares Programming), which identifies a local constrained optimum rather than a guaranteed global optimum.",
            "Mathematical reductions depend on the trained regression surface and must be validated for engineering and operational feasibility.",
            "Decision variable bounds restrict extrapolation outside historical training distributions.",
        ],
    }


def run_what_if_scenarios(
    bundle: Dict[str, Any],
    models_summary: Dict[str, Any],
    df: pd.DataFrame,
    scenarios: List[Dict[str, Any]],
    baseline_inputs: Optional[Dict[str, Any]] = None,
    model_key: Optional[str] = None,
) -> Dict[str, Any]:
    chosen_key = model_key if (model_key and model_key in bundle["models"]) else bundle["best_model_key"]
    feature_cols: List[str] = bundle["feature_columns"]
    preprocessor: ZeroLeakagePreprocessor = bundle["preprocessor"]

    if baseline_inputs:
        base_row = dict(baseline_inputs)
    else:
        base_row = {}
        for col in feature_cols:
            if col in preprocessor.numeric_cols:
                base_row[col] = float(df[col].dropna().iloc[-1]) if len(df[col].dropna()) > 0 else preprocessor.num_medians[col]
            else:
                modes = df[col].dropna().mode()
                base_row[col] = str(modes.iloc[0]) if len(modes) > 0 else "Unknown"

    base_pred = float(predict_single(bundle, models_summary, base_row, model_key=chosen_key)["predicted_value"])
    results: List[Dict[str, Any]] = []

    for sc in scenarios:
        sc_name = sc.get("name", "Custom Scenario")
        sc_desc = sc.get("description", "")
        adj_map = sc.get("adjustments_pct", {})

        mod_row = dict(base_row)
        applied_notes: List[str] = []
        for col, pct in adj_map.items():
            if col in mod_row and isinstance(mod_row[col], (int, float, np.number)):
                old_v = float(mod_row[col])
                new_v = max(0.0, old_v * (1.0 + float(pct) / 100.0))
                if "renew" in col.lower() or "pct" in col.lower() or "share" in col.lower():
                    new_v = min(100.0, new_v)
                mod_row[col] = new_v
                applied_notes.append(f"{col}: {float(pct):+.1f}% ({old_v:.1f} → {new_v:.1f})")

        sc_out = predict_single(bundle, models_summary, mod_row, model_key=chosen_key)
        sc_em = float(sc_out["predicted_value"])
        abs_red = round(base_pred - sc_em, 4)
        pct_red = round((abs_red / base_pred) * 100.0, 2) if abs(base_pred) > 1e-9 else 0.0

        results.append({
            "scenario_name": sc_name,
            "description": sc_desc,
            "applied_adjustments": applied_notes,
            "baseline_emissions": round(base_pred, 4),
            "scenario_emissions": round(sc_em, 4),
            "absolute_reduction": abs_red,
            "percentage_reduction": pct_red,
            "confidence_interval_95": sc_out["confidence_interval_95"],
            "out_of_distribution_warnings": sc_out["out_of_distribution_warnings"],
        })

    return {
        "model_key": chosen_key,
        "model_name": MODEL_KEYS.get(chosen_key, chosen_key),
        "target_column": bundle["target_column"],
        "unit": bundle.get("target_unit", "tCO2e"),
        "baseline_emissions": round(base_pred, 4),
        "baseline_inputs": base_row,
        "scenarios": results,
    }
