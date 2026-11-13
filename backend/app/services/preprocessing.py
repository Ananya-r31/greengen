"""Documented dataset preprocessing: type overrides, missing values, duplicates, outlier clipping, and emission factor calculation."""
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from app.services.data_loader import UPLOAD_DIR
from app.services.feature_mapping import detect_column_concepts
from app.services.data_profiler import profile_dataset


def preprocess_dataset(
    dataset_id: str,
    df: pd.DataFrame,
    missing_numeric_strategy: str = "median",
    missing_categorical_strategy: str = "mode",
    drop_duplicates: bool = True,
    outlier_method: str = "none",
    type_overrides: Optional[Dict[str, str]] = None,
    concept_overrides: Optional[Dict[str, str]] = None,
    emission_factors: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, str, Dict[str, Any], Dict[str, Any], List[str]]:
    clean_df = df.copy()
    logs: List[str] = []

    # 1. Apply explicit data-type overrides
    if type_overrides:
        for col, target_type in type_overrides.items():
            if col in clean_df.columns:
                if target_type == "numeric":
                    clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
                    logs.append(f"Converted column '{col}' to numeric type.")
                elif target_type == "categorical":
                    clean_df[col] = clean_df[col].astype(str)
                    logs.append(f"Converted column '{col}' to categorical string type.")
                elif target_type == "datetime":
                    clean_df[col] = pd.to_datetime(clean_df[col], errors="coerce")
                    logs.append(f"Converted column '{col}' to datetime type.")

    # 2. Remove duplicate records if requested
    if drop_duplicates:
        before_rows = len(clean_df)
        clean_df = clean_df.drop_duplicates().reset_index(drop=True)
        removed = before_rows - len(clean_df)
        if removed > 0:
            logs.append(f"Removed {removed} exact duplicate rows.")
        else:
            logs.append("Checked for duplicate rows (0 duplicates found).")

    # 3. Handle missing values
    if missing_numeric_strategy == "drop" or missing_categorical_strategy == "drop":
        before_na = len(clean_df)
        clean_df = clean_df.dropna().reset_index(drop=True)
        logs.append(f"Dropped {before_na - len(clean_df)} rows containing missing values.")
    else:
        for col in clean_df.columns:
            na_count = int(clean_df[col].isna().sum())
            if na_count == 0:
                continue
            if pd.api.types.is_numeric_dtype(clean_df[col]):
                if missing_numeric_strategy == "mean":
                    fill_val = float(clean_df[col].mean()) if not clean_df[col].dropna().empty else 0.0
                elif missing_numeric_strategy == "zero":
                    fill_val = 0.0
                else:
                    fill_val = float(clean_df[col].median()) if not clean_df[col].dropna().empty else 0.0
                clean_df[col] = clean_df[col].fillna(fill_val)
                logs.append(
                    f"Imputed {na_count} missing values in numeric column '{col}' using {missing_numeric_strategy} ({fill_val:.4g})."
                )
            else:
                if missing_categorical_strategy == "unknown":
                    fill_str = "Unknown"
                else:
                    modes = clean_df[col].dropna().mode()
                    fill_str = str(modes.iloc[0]) if len(modes) > 0 else "Unknown"
                clean_df[col] = clean_df[col].fillna(fill_str)
                logs.append(
                    f"Imputed {na_count} missing values in categorical column '{col}' using {missing_categorical_strategy} ('{fill_str}')."
                )

    # 4. Handle outliers on numerical columns
    if outlier_method in ("iqr_clip", "zscore_clip"):
        for col in clean_df.columns:
            if not pd.api.types.is_numeric_dtype(clean_df[col]):
                continue
            s = clean_df[col].dropna()
            if len(s) < 5:
                continue
            if outlier_method == "iqr_clip":
                q1 = float(s.quantile(0.25))
                q3 = float(s.quantile(0.75))
                iqr = q3 - q1
                if iqr > 0:
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    clipped_count = int(((clean_df[col] < lower) | (clean_df[col] > upper)).sum())
                    if clipped_count > 0:
                        clean_df[col] = clean_df[col].clip(lower=lower, upper=upper)
                        logs.append(f"Clipped {clipped_count} IQR outliers in '{col}' to [{lower:.2f}, {upper:.2f}].")
            elif outlier_method == "zscore_clip":
                mean, std = float(s.mean()), float(s.std())
                if std > 0:
                    lower, upper = mean - 3.0 * std, mean + 3.0 * std
                    clipped_count = int(((clean_df[col] < lower) | (clean_df[col] > upper)).sum())
                    if clipped_count > 0:
                        clean_df[col] = clean_df[col].clip(lower=lower, upper=upper)
                        logs.append(f"Clipped {clipped_count} Z-score (>3 SD) outliers in '{col}'.")

    # 5. Optional activity-based CO2 calculation when user provides explicit emission factors
    mapping = detect_column_concepts(clean_df)
    if emission_factors:
        c2c = mapping["concept_to_columns"]
        computed_kg = pd.Series(0.0, index=clean_df.index)
        used_factors = []

        elec_cols = c2c.get("electricity_consumption", [])
        if elec_cols and "electricity_kg_per_kwh" in emission_factors:
            ef = float(emission_factors["electricity_kg_per_kwh"])
            computed_kg += clean_df[elec_cols[0]].astype(float) * ef
            used_factors.append(f"{elec_cols[0]} × {ef} kgCO2e/kWh")

        fuel_cols = c2c.get("fossil_fuel", [])
        if fuel_cols and "fuel_kg_per_liter" in emission_factors:
            ef = float(emission_factors["fuel_kg_per_liter"])
            computed_kg += clean_df[fuel_cols[0]].astype(float) * ef
            used_factors.append(f"{fuel_cols[0]} × {ef} kgCO2e/unit")

        trans_cols = c2c.get("transportation", [])
        if trans_cols and "transport_kg_per_km" in emission_factors:
            ef = float(emission_factors["transport_kg_per_km"])
            computed_kg += clean_df[trans_cols[0]].astype(float) * ef
            used_factors.append(f"{trans_cols[0]} × {ef} kgCO2e/km")

        waste_cols = c2c.get("waste_generation", [])
        if waste_cols and "waste_kg_per_tonne" in emission_factors:
            ef = float(emission_factors["waste_kg_per_tonne"])
            computed_kg += clean_df[waste_cols[0]].astype(float) * ef
            used_factors.append(f"{waste_cols[0]} × {ef} kgCO2e/tonne")

        if used_factors:
            clean_df["calculated_co2_tonnes"] = (computed_kg / 1000.0).round(4)
            logs.append(
                f"Calculated 'calculated_co2_tonnes' (tCO2e) using user-supplied emission factors: {', '.join(used_factors)}."
            )
            mapping = detect_column_concepts(clean_df)

    # 6. Apply user concept overrides
    if concept_overrides:
        for concept_key, col_name in concept_overrides.items():
            if col_name in clean_df.columns and concept_key in mapping["concept_to_columns"]:
                if col_name not in mapping["concept_to_columns"][concept_key]:
                    mapping["concept_to_columns"][concept_key].insert(0, col_name)
                mapping["column_to_concept"][col_name] = concept_key
                if concept_key == "co2_emissions":
                    mapping["primary_target"] = col_name
                logs.append(f"Mapped column '{col_name}' to environmental concept '{concept_key}'.")

    cleaned_path = os.path.join(UPLOAD_DIR, f"{dataset_id}_cleaned.csv")
    clean_df.to_csv(cleaned_path, index=False)
    profile = profile_dataset(clean_df, mapping)

    return clean_df, cleaned_path, profile, mapping, logs
